"""Read Tecplot binary data through an isolated ParaView subprocess."""

# standard library packages
from collections import deque
from glob import glob
import json
import logging
from os import environ, sep
from os.path import join
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
from threading import Lock, Thread
from typing import Any, Dict, List, Literal, Union

# third party packages
import numpy as np
import torch as pt

# flowtorch packages
from flowtorch import DEFAULT_DTYPE
from .dataloader import Dataloader, _preallocate_time_series
from .utils import check_and_standardize_path, check_list_or_str

logger = logging.getLogger(__name__)

_HEADER_SIZE = 8
_MAX_HEADER_SIZE = 1024 * 1024


class ParaViewProcessError(RuntimeError):
    """Error reported by, or communicating with, the pvpython worker."""


class _PvpythonClient:
    """Manage a persistent pvpython process and its binary response channel."""

    def __init__(self, executable: str, worker_path: Path | None = None):
        if sys.platform == "win32":
            raise NotImplementedError(
                "TecplotDataloader is not supported on Windows. Use Linux or macOS."
            )
        self._lock = Lock()
        self._stderr: deque[str] = deque(maxlen=50)
        self._closed = False
        worker = worker_path or Path(__file__).with_name("_tecplot_pvpython.py")
        parent_socket, child_socket = socket.socketpair()
        parent_socket.settimeout(30.0)
        try:
            self._process = subprocess.Popen(
                [executable, str(worker), str(child_socket.fileno())],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                pass_fds=(child_socket.fileno(),),
                bufsize=0,
            )
        except Exception:
            parent_socket.close()
            child_socket.close()
            raise
        child_socket.close()
        self._socket = parent_socket
        self._stderr_thread = Thread(target=self._collect_stderr, daemon=True)
        self._stderr_thread.start()
        try:
            ready = self._receive()
        except Exception:
            self.close(force=True)
            raise
        self._socket.settimeout(None)
        if ready != {"protocol": 1}:
            self.close(force=True)
            raise ParaViewProcessError(
                f"Unexpected response from pvpython worker: {ready!r}"
            )

    def _collect_stderr(self):
        """Continuously consume stderr so a verbose worker cannot deadlock."""
        if self._process.stderr is None:
            return
        for line in iter(self._process.stderr.readline, b""):
            self._stderr.append(line.decode("utf-8", errors="replace").rstrip())

    def _stderr_message(self) -> str:
        return "\n".join(self._stderr)

    def _read_exact(self, size: int) -> bytes:
        chunks = bytearray(size)
        view = memoryview(chunks)
        offset = 0
        while offset < size:
            received = self._socket.recv_into(view[offset:])
            if received == 0:
                details = self._stderr_message()
                suffix = f"\npvpython stderr:\n{details}" if details else ""
                raise ParaViewProcessError(
                    f"pvpython worker closed the binary channel unexpectedly{suffix}"
                )
            offset += received
        return bytes(chunks)

    def _receive(self) -> Any:
        header_size = struct.unpack("!Q", self._read_exact(_HEADER_SIZE))[0]
        if header_size > _MAX_HEADER_SIZE:
            raise ParaViewProcessError(
                f"pvpython response header is too large: {header_size} bytes"
            )
        try:
            header = json.loads(self._read_exact(header_size))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ParaViewProcessError(
                "pvpython returned an invalid response header"
            ) from error
        if header.get("status") == "error":
            error_type = header.get("error_type", "Error")
            message = header.get("message", "unknown pvpython error")
            raise ParaViewProcessError(f"{error_type}: {message}")
        if header.get("status") != "ok":
            raise ParaViewProcessError(f"Unknown pvpython response: {header!r}")
        kind = header.get("kind")
        if kind == "json":
            return header.get("value")
        if kind != "array":
            raise ParaViewProcessError(f"Unknown pvpython payload kind: {kind!r}")

        try:
            dtype = np.dtype(header["dtype"])
            shape = tuple(int(value) for value in header["shape"])
            nbytes = int(header["nbytes"])
        except (KeyError, TypeError, ValueError) as error:
            raise ParaViewProcessError(
                f"Invalid array metadata from pvpython: {header!r}"
            ) from error
        if any(value < 0 for value in shape):
            raise ParaViewProcessError(f"Invalid array shape from pvpython: {shape!r}")
        expected_nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if nbytes != expected_nbytes:
            raise ParaViewProcessError(
                f"Invalid array size from pvpython: expected {expected_nbytes}, "
                f"received {nbytes}"
            )
        array = np.empty(shape, dtype=dtype)
        view = memoryview(array).cast("B")
        offset = 0
        while offset < nbytes:
            received = self._socket.recv_into(view[offset:])
            if received == 0:
                raise ParaViewProcessError(
                    "pvpython worker closed the channel while sending an array"
                )
            offset += received
        return array

    def request(self, operation: str, **parameters) -> Any:
        """Send one request and return its decoded JSON value or NumPy array."""
        with self._lock:
            if self._closed or self._process.poll() is not None:
                details = self._stderr_message()
                suffix = f"\npvpython stderr:\n{details}" if details else ""
                raise ParaViewProcessError(f"pvpython worker is not running{suffix}")
            if self._process.stdin is None:
                raise ParaViewProcessError("pvpython worker has no command channel")
            request = json.dumps(
                {"operation": operation, **parameters}, separators=(",", ":")
            )
            try:
                self._process.stdin.write(request.encode("utf-8") + b"\n")
                self._process.stdin.flush()
            except BrokenPipeError as error:
                details = self._stderr_message()
                suffix = f"\npvpython stderr:\n{details}" if details else ""
                raise ParaViewProcessError(
                    f"pvpython command channel closed unexpectedly{suffix}"
                ) from error
            return self._receive()

    def close(self, force: bool = False):
        """Stop the worker and release its communication channels."""
        if self._closed:
            return
        self._closed = True
        if (
            not force
            and self._process.poll() is None
            and self._process.stdin is not None
        ):
            try:
                request = json.dumps({"operation": "close"}).encode("utf-8") + b"\n"
                self._process.stdin.write(request)
                self._process.stdin.flush()
                self._receive()
            except (BrokenPipeError, OSError, ParaViewProcessError):
                force = True
        if self._process.stdin is not None:
            self._process.stdin.close()
        self._socket.close()
        if force and self._process.poll() is None:
            self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        if self._process.stderr is not None:
            self._process.stderr.close()

    def __del__(self):
        try:
            self.close(force=True)
        except Exception:
            pass


def _resolve_pvpython(pvpython: Union[str, Path, None]) -> str:
    """Resolve an explicit, configured, or PATH-provided pvpython executable."""
    requested = (
        str(pvpython)
        if pvpython is not None
        else environ.get("FLOWTORCH_PVPYTHON", "pvpython")
    )
    executable = shutil.which(requested)
    if executable is None:
        raise FileNotFoundError(
            "Could not find ParaView's pvpython executable. Pass pvpython=..., "
            "set FLOWTORCH_PVPYTHON, or add pvpython to PATH."
        )
    return executable


class TecplotDataloader(Dataloader):
    """Dataloader for Tecplot binary files using an isolated pvpython worker.

    This loader is supported on Linux and macOS only.

    ParaView and its bundled Python packages remain in a separate process.
    Arrays are transferred through a binary socket protocol without temporary
    files. Nested composite datasets are supported; unique leaf names are
    exposed as zones, while duplicate leaf names are disambiguated by their
    full composite paths.

    Examples

    >>> from flowtorch import DATASETS
    >>> from flowtorch.data import TecplotDataloader
    >>> path = DATASETS["plt_naca2409_surface"]
    >>> loader = TecplotDataloader.from_tau(
    ...     path,
    ...     "alfa16.surface.pval.unsteady_",
    ...     pvpython="/opt/ParaView/bin/pvpython",
    ... )
    >>> loader.zone_names
    ["ls", "te", "us"]
    >>> loader.zone = "us"
    >>> density = loader.load_snapshot("density", loader.write_times)
    >>> density.shape
    torch.Size([300, 3])
    >>> loader.close()
    """

    def __init__(
        self,
        path: str,
        file_names: Dict[str, str],
        dtype: pt.dtype = DEFAULT_DTYPE,
        pvpython: Union[str, Path, None] = None,
    ):
        """Create a loader backed by a persistent pvpython subprocess.

        :param path: path to snapshot location
        :param file_names: mapping of write times to snapshot names
        :param dtype: tensor data type used for weights
        :param pvpython: path or command name of ParaView's pvpython executable
        """
        self._path = path
        self._file_names = file_names
        self._dtype = dtype
        self._client = _PvpythonClient(_resolve_pvpython(pvpython))
        self._zone_names: List[str] | None = None
        self._zone_paths: Dict[str, List[int]] = {}
        self._field_names: Dict[tuple[str, str], Dict[str, List[str]]] = {}
        self._zone = self.zone_names[0]

    @classmethod
    def from_tau(
        cls,
        path: str,
        base_name: str = "",
        suffix: str = ".plt",
        dtype: pt.dtype = DEFAULT_DTYPE,
        pvpython: Union[str, Path, None] = None,
    ):
        """Construct a Tecplot dataloader from TAU snapshots.

        :param path: path to snapshot location
        :param base_name: common basename of all snapshots
        :param suffix: snapshot file suffix
        :param dtype: tensor data type used for weights
        :param pvpython: path or command name of ParaView's pvpython executable
        :raises FileNotFoundError: if snapshots or pvpython cannot be found
        :return: Tecplot dataloader object
        :rtype: TecplotDataloader
        """
        path = check_and_standardize_path(path)
        file_paths = glob(join(path, f"{base_name}i=*t=*{suffix}"))
        discovered_names = [file_path.split(sep)[-1] for file_path in file_paths]
        write_times = [name.split("t=")[-1] for name in discovered_names]
        if suffix:
            write_times = [time[: -len(suffix)] for time in write_times]
        sorted_names = sorted(
            zip(write_times, discovered_names), key=lambda item: float(item[0])
        )
        file_names = {time: name for time, name in sorted_names}
        if not file_names:
            raise FileNotFoundError(f"Could not find solution files in {path}")
        return cls(path, file_names, dtype=dtype, pvpython=pvpython)

    def _assemble_file_path(self, time: str) -> str:
        """Assemble the path to a single snapshot."""
        return join(self._path, self._file_names[time])

    @staticmethod
    def _parse_block_name(meta_data: str) -> str:
        """Extract a block name from ParaView metadata."""
        for line in meta_data.split("\n"):
            if "NAME" in line:
                return line.split(":")[-1].strip()
        raise ValueError("Could not find a block name in the Tecplot metadata")

    def _request_for_time(self, operation: str, time: str, **parameters):
        return self._client.request(
            operation,
            file_path=self._assemble_file_path(time),
            registration_name=self._file_names[time],
            **parameters,
        )

    @staticmethod
    def _validate_association(association: str) -> Literal["point", "cell"]:
        """Validate and normalize a VTK array association."""
        if association not in ("point", "cell"):
            raise ValueError(
                f"Unknown field association {association!r}; expected 'point' or 'cell'"
            )
        return association

    def _load_single_snapshot(
        self, field_name: str, time: str, association: str = "point"
    ) -> pt.Tensor:
        """Load a single field from one snapshot."""
        association = self._validate_association(association)
        field_names = self.field_names_for(association)[self.write_times[0]]
        if field_name not in field_names:
            raise ValueError(
                f"Unknown field {field_name!r} in {association} data for zone "
                f"{self.zone!r}. Available fields are {field_names}"
            )
        array = self._request_for_time(
            "field",
            time,
            field_name=field_name,
            association=association,
            block_path=self._zone_paths[self.zone],
        )
        return pt.from_numpy(array)

    def _load_multiple_snapshots(
        self, field_name: str, times: List[str], association: str = "point"
    ) -> pt.Tensor:
        """Load one field from multiple snapshots."""
        return _preallocate_time_series(
            lambda time: self._load_single_snapshot(field_name, time, association),
            times,
        )

    def load_snapshot(
        self,
        field_name: Union[List[str], str],
        time: Union[List[str], str],
        association: str = "point",
    ) -> Union[List[pt.Tensor], pt.Tensor]:
        """Load snapshots for one or more fields and write times.

        :param field_name: one field name or a list of field names
        :param time: one write time or a list of write times
        :param association: VTK array association, either ``"point"`` (the
            backward-compatible default) or ``"cell"``
        """
        check_list_or_str(field_name, "field_name")
        check_list_or_str(time, "time")
        association = self._validate_association(association)
        if isinstance(field_name, list):
            if isinstance(time, list):
                return [
                    self._load_multiple_snapshots(name, time, association)
                    for name in field_name
                ]
            return [
                self._load_single_snapshot(name, time, association)
                for name in field_name
            ]
        if isinstance(time, list):
            return self._load_multiple_snapshots(field_name, time, association)
        return self._load_single_snapshot(field_name, time, association)

    @property
    def zone_names(self) -> List[str]:
        """Names of available leaf blocks/zones.

        Unique leaf names are kept short. If a leaf name occurs more than
        once, its full composite path is returned instead.
        """
        if self._zone_names is None:
            zones = self._request_for_time("zone_names", self.write_times[0])
            if not zones:
                raise ParaViewProcessError(
                    "The Tecplot reader did not expose any leaf blocks/zones"
                )
            self._zone_names = [zone["name"] for zone in zones]
            self._zone_paths = {zone["name"]: zone["path"] for zone in zones}
        return self._zone_names

    @property
    def zone(self) -> str:
        """Currently selected block/zone."""
        return self._zone

    @zone.setter
    def zone(self, value: str):
        """Select the active block/zone."""
        if value in self.zone_names:
            self._zone = value
        else:
            logger.warning(
                "%s not found. Available zones are %s", value, self.zone_names
            )

    @property
    def write_times(self) -> List[str]:
        """Available snapshot write times."""
        return list(self._file_names)

    @property
    def field_names(self) -> Dict[str, List[str]]:
        """Point-data field names in the selected zone's first snapshot."""
        return self.field_names_for("point")

    def field_names_for(self, association: str = "point") -> Dict[str, List[str]]:
        """Return fields for one association in the currently selected zone.

        :param association: VTK array association, ``"point"`` or ``"cell"``
        """
        association = self._validate_association(association)
        cache_key = (self.zone, association)
        if cache_key not in self._field_names:
            time = self.write_times[0]
            self._field_names[cache_key] = {
                time: self._request_for_time(
                    "field_names",
                    time,
                    association=association,
                    block_path=self._zone_paths[self.zone],
                )
            }
        return self._field_names[cache_key]

    @property
    def vertices(self) -> pt.Tensor:
        """Points at which field values are defined."""
        array = self._request_for_time(
            "vertices",
            self.write_times[0],
            block_path=self._zone_paths[self.zone],
        )
        return pt.from_numpy(array)

    @property
    def weights(self) -> pt.Tensor:
        """Unit weights for POD/DMD analysis."""
        return pt.ones(self.vertices.shape[0], dtype=self._dtype)

    def close(self):
        """Stop the pvpython worker."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
