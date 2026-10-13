"""Lightweight computation provenance without reading or copying array contents."""

from __future__ import annotations

from contextvars import ContextVar
from copy import deepcopy
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from functools import wraps
from importlib import metadata
from inspect import signature
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Callable
from uuid import uuid4

import torch

SCHEMA_VERSION = 1
_session: ContextVar[Any] = ContextVar("flowtorch_provenance_session", default=None)


def describe(value: Any) -> Any:
    """Describe values without materializing tensors, lazy sources, or properties."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite": str(value)}
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "content_recorded": False,
        }
    if isinstance(value, (torch.dtype, torch.device, Path)):
        return str(value)
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if isinstance(value, ProvenanceMixin):
        return {"type": qualified_name(type(value)), "provenance": value.provenance}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: describe(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): describe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [describe(item) for item in value]
    if callable(value):
        return {"callable": qualified_name(value), "configuration_recorded": False}
    return {"type": qualified_name(type(value)), "configuration_recorded": False}


def qualified_name(value: Any) -> str:
    """Return an import-oriented identifier, without calling user representations."""
    return f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', type(value).__qualname__)}"


def capture_environment() -> dict:
    """Capture numerical settings without initializing CUDA or optional readers."""
    import flowtorch

    versions = {"python": platform.python_version(), "flowtorch": flowtorch.__version__}
    for name in ("torch", "numpy", "scipy", "matplotlib", "vtk", "h5py", "netCDF4"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    versions["torch"] = str(torch.__version__)
    code = {
        "import_path": str(Path(flowtorch.__file__).resolve()),
        "commit": None,
        "dirty": None,
    }
    checkout = Path(flowtorch.__file__).resolve().parent.parent
    if (checkout / ".git").exists():
        try:
            code["commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=checkout,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
            ).strip()
            code["dirty"] = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    cwd=checkout,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=2,
                ).strip()
            )
        except (OSError, subprocess.SubprocessError):
            pass
    runtime = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "default_dtype": str(torch.get_default_dtype()),
        "grad_enabled": torch.is_grad_enabled(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_build": torch.version.cuda,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "environment_variables": {
            name: os.environ.get(name)
            for name in (
                "CUBLAS_WORKSPACE_CONFIG",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "processor": platform.processor(),
    }
    for name, backend in (
        ("cuda_matmul", torch.backends.cuda.matmul),
        ("cudnn", torch.backends.cudnn),
    ):
        value = getattr(backend, "fp32_precision", None)
        runtime[name + "_precision"] = (
            value
            if value is not None
            else {"allow_tf32": getattr(backend, "allow_tf32", None)}
        )
    precision = getattr(torch, "get_float32_matmul_precision", None)
    if precision is not None:
        runtime["float32_matmul_precision"] = precision()
    if torch.cuda.is_initialized():
        runtime["cuda_devices"] = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        runtime["cudnn_version"] = torch.backends.cudnn.version()
    return {"versions": versions, "code": code, "runtime": runtime}


def _write(path: str | Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


class ProvenanceMixin:
    """Expose captured settings and chronological operation history.

    Subclasses explicitly decorate computational entry points with :func:`tracked`.
    Metadata contains descriptions rather than input contents or replayable code.
    """

    def _effective_settings(self) -> dict:
        return {}

    @property
    def provenance(self) -> dict:
        return deepcopy(
            {
                "schema_version": SCHEMA_VERSION,
                "type": qualified_name(type(self)),
                "records": vars(self).get("_provenance_records", []),
                "current": describe(self._effective_settings()),
            }
        )

    @property
    def settings(self) -> dict:
        records = vars(self).get("_provenance_records", [])
        return deepcopy(records[0]["requested"] if records else {})

    @property
    def versions(self) -> dict:
        records = vars(self).get("_provenance_records", [])
        return deepcopy(records[0]["environment"]["versions"] if records else {})

    def dump_settings(self, path: str | Path) -> None:
        """Write captured initial arguments as JSON."""
        _write(path, self.settings)

    def dump_versions(self, path: str | Path) -> None:
        """Write versions captured at the initial operation as JSON."""
        _write(path, self.versions)

    def dump_provenance(self, path: str | Path) -> None:
        """Write history and current settings; distributed callers choose a writer."""
        _write(path, self.provenance)


def tracked(function: Callable) -> Callable:
    """Record a selected constructor, factory, method, or session-scoped function.

    Nested constructor wrappers on the same instance are suppressed. Other calls
    retain their own records. Failed operations are marked explicitly.
    """
    call_signature = signature(function)

    @wraps(function)
    def wrapper(*args, **kwargs):
        owner = args[0] if args and isinstance(args[0], ProvenanceMixin) else None
        session = _session.get()
        factory = bool(
            args and isinstance(args[0], type) and issubclass(args[0], ProvenanceMixin)
        )
        if owner is None and session is None and not factory:
            return function(*args, **kwargs)
        if (
            owner is not None
            and function.__name__ == "__init__"
            and vars(owner).get("_provenance_initializing", False)
        ):
            return function(*args, **kwargs)
        bound = call_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        record = {
            "id": str(uuid4()),
            "operation": qualified_name(function),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "requested": {
                name: describe(value)
                for name, value in bound.arguments.items()
                if name not in ("self", "cls")
            },
            "environment": capture_environment(),
            "status": "running",
        }
        initializing = owner is not None and function.__name__ == "__init__"
        if initializing:
            owner._provenance_initializing = True
        target = owner
        try:
            result = function(*args, **kwargs)
            if target is None and isinstance(result, ProvenanceMixin):
                target = result
            record["effective"] = (
                describe(target._effective_settings()) if target is not None else {}
            )
            record["status"] = "complete"
            return result
        except Exception as error:
            record["status"] = "failed"
            record["error_type"] = qualified_name(type(error))
            raise
        finally:
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            if initializing:
                owner._provenance_initializing = False
            if target is not None:
                vars(target).setdefault("_provenance_records", []).append(
                    deepcopy(record)
                )
            if session is not None:
                session.records.append(deepcopy(record))

    return wrapper


class ProvenanceSession:
    """Collect selected function calls without changing their return types.

    Sessions are context-local. Nested sessions receive only their own calls.
    """

    def __init__(self):
        self.records: list[dict] = []

    def __enter__(self):
        self._token = _session.set(self)
        return self

    def __exit__(self, *exc):
        _session.reset(self._token)

    def dump_provenance(self, path: str | Path) -> None:
        """Write session operations as JSON."""
        _write(path, {"schema_version": SCHEMA_VERSION, "records": self.records})


def dump_figure_provenance(
    figure: Any, path: str | Path, session: ProvenanceSession | None = None
) -> None:
    """Write supported final Matplotlib settings with optional plotting history.

    This describes appearance; it does not serialize artists or recreate a figure.
    """
    axes = []
    for axis in figure.axes:
        axes.append(
            {
                "title": axis.get_title(),
                "xlabel": axis.get_xlabel(),
                "ylabel": axis.get_ylabel(),
                "xlim": list(axis.get_xlim()),
                "ylim": list(axis.get_ylim()),
                "xscale": axis.get_xscale(),
                "yscale": axis.get_yscale(),
                "images": [
                    {"cmap": image.get_cmap().name, "clim": list(image.get_clim())}
                    for image in axis.images
                ],
            }
        )
    _write(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "records": [] if session is None else session.records,
            "figure": {
                "size_inches": figure.get_size_inches().tolist(),
                "dpi": figure.dpi,
                "axes": describe(axes),
            },
            "export_environment": capture_environment(),
        },
    )
