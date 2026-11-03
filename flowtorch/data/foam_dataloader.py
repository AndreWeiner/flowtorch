"""Classes to work with OpenFOAM cases, meshes, and fields.

The :class:`FOAMDataloader` class allows to load fields from
an OpenFOAM simulation folder. Currently, only the ESI-OpenCFD
branch of OpenFOAM is supported (v1912, v2006). The :class:`FOAMCase`
class assembles information about the folder and file structure
of a simulation. The :class:`FOAMMesh` allows loading and parsing
the finite volume mesh.
"""

# standard library packages
import struct
import logging
import re

from glob import glob
from os import listdir
from os.path import isdir, isfile, join
from typing import List, Dict, Sequence, Tuple, Union

# third party packages
import torch as pt

# flowtorch packages
from flowtorch import DEFAULT_DTYPE
from .dataloader import Dataloader, _preallocate_time_series
from ._foam_stream import (
    foam_open,
    resolve,
    field_info,
    read_values,
    header,
    token,
    binary_dtype,
    read_list,
)
from .utils import check_and_standardize_path, check_list_or_str

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# volSymmTensorField as 6 entries since it is symmetric:
# the order is: XX, XY, XZ, YY, YZ, ZZ
# volTensorField as 9 entries since:
# the order is: XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ
FIELD_TYPE_DIMENSION = {
    b"volScalarField": 1,
    b"volSphericalTensorField": 1,
    b"volVectorField": 3,
    b"volSymmTensorField": 6,
    b"volTensorField": 9,
}
CONSTANT_PATH = "constant"
POLYMESH_PATH = join("constant", "polyMesh")
MESH_FILES = ("points", "owner", "neighbour", "faces", "boundary")


MAX_LINE_HEADER = 30
MAX_LINE_INTERNAL_FIELD = 40

SIZE_OF_CHAR = struct.calcsize("c")
SIZE_OF_INT = struct.calcsize("i")
SIZE_OF_DOUBLE = struct.calcsize("d")


class FOAMDataloader(Dataloader):
    """Load internal fields and mesh properties of OpenFOAM cases.

    The project ofpp_ by Xu Xianghua has been a great
    help to implement some methods.

    .. _ofpp: https://github.com/xu-xianghua/ofpp

    Examples

    >>> from flowtorch import DATASETS
    >>> from flowtorch.data import FOAMDataloader
    >>> path = DATASETS["of_cavity_ascii_parallel"]
    >>> loader = FOAMDataloader(path)
    >>> loader.write_times
    ['0', '0.1', '0.2', '0.3', '0.4', '0.5']
    >>> loader.field_names
    {'0': ['p', 'U'], '0.1': ['p', 'phi', 'U'], '0.2': ['p', 'phi', 'U'], '0.3': [
        'p', 'phi', 'U'], '0.4': ['p', 'phi', 'U'], '0.5': ['p', 'phi', 'U']}
    >>> vertices = loader.vertices
    >>> vertices[:3]
    tensor([[0.0025, 0.0025, 0.0050],
            [0.0075, 0.0025, 0.0050],
            [0.0125, 0.0025, 0.0050]])
    >>> loader.weights[:3]  # cell volumes
    tensor([2.5000e-07, 2.5000e-07, 2.5000e-07])
    >>> p = loader.load_snapshot("p", "0.5")
    >>> p.shape
    torch.Size([400])
    >>> p[:3]
    tensor([ 4.2993e-06, -5.8226e-03, -1.2960e-02])

    """

    def __init__(
        self,
        path: str,
        dtype: pt.dtype = DEFAULT_DTYPE,
        distributed: Union[bool, None] = None,
        verbose: bool = False,
    ):
        """Create a FOAMDataloader instance from a path.

        :param path: path to an OpenFOAM simulation folder.
        :type path: str
        :param dtype: tensor type; default is single precision, `torch.float32`
        :type dtype: str
        :param distributed: case is considered distributed if True; if None,
            the case type (parallel/serial) is determined automatically;
            defaults to None
        :type distributed: bool
        :param verbose: display status information about the current field and time step when loading snapshots;
                        recommended for large fields and/or large number of time steps to load
        :type verbose: bool
        """
        self._case = FOAMCase(path, distributed)
        self._mesh = FOAMMesh(self._case, dtype)
        self._dtype = dtype
        self._verbose = verbose

    def _parse_data(self, data: List[bytes]) -> pt.Tensor:
        """Compatibility parser for an already materialized nonuniform field."""
        from io import BytesIO

        stream = BytesIO(b"".join(data))
        metadata, dim, count, uniform = field_info(stream)
        if uniform is not None:
            raise ValueError("Uniform fields require a cell count; use load_snapshot")
        values = read_values(stream, metadata, dim, count, 0, count)
        shape = (count,) if dim == 1 else (count, dim)
        return pt.as_tensor(values, dtype=self._dtype).reshape(shape)

    @staticmethod
    def _field_type(data: Sequence[bytes]) -> bytes | None:
        """Read the declared field class from header text."""
        match = re.search(rb"\bclass\s+(\w+)\s*;", b"".join(data))
        return match[1] if match else None

    def _load_single_snapshot(self, field_name: str, time: str) -> pt.Tensor:
        """Load a single snapshot of a single field from disk.

        :param field_name: name of the field
        :type field_name: str
        :param time: snapshot write time
        :type time: str
        :return: tensor holding the field
        :rtype: pt.Tensor
        """
        # print some info because for large fields and large number of time steps it takes quite a while to load
        if self._verbose:
            logger.info(f"Loading time step {time}s for field {field_name}.")
        return self._load_slice(field_name, time, slice(None))

    def _field_layout(self, field_name: str, time: str) -> List[Tuple[str, int, int]]:
        layout = []
        for proc in range(self._case.processors):
            path = self._case.build_file_path(field_name, time, proc)
            with foam_open(path) as stream:
                metadata, dim, count, uniform = field_info(stream)
            if count is None:
                mesh_path = self._case.build_file_path("", POLYMESH_PATH, proc)
                count = self._mesh._get_n_cells(mesh_path)
            if layout and dim != layout[0][1]:
                raise ValueError("Field component count differs across processors")
            layout.append((path, dim, count))
        return layout

    def snapshot_shape(self, field_name: str, time: str) -> Tuple[int, ...]:
        """Read field metadata without materializing nonuniform field values."""
        layout = self._field_layout(field_name, time)
        count = sum(item[2] for item in layout)
        dim = layout[0][1]
        return (count,) if dim == 1 else (count, dim)

    def _load_slice(
        self, field_name: str, time: str, spatial_slice: slice
    ) -> pt.Tensor:
        layout = self._field_layout(field_name, time)
        count = sum(item[2] for item in layout)
        start, stop, step = spatial_slice.indices(count)
        if step != 1:
            raise ValueError("spatial_slice must have unit stride")
        dim = layout[0][1]
        shape = (max(0, stop - start),) if dim == 1 else (max(0, stop - start), dim)
        result = pt.empty(shape, dtype=self._dtype)
        offset = 0
        for path, _, local_count in layout:
            first, last = max(start, offset), min(stop, offset + local_count)
            if first < last:
                with foam_open(path) as stream:
                    metadata, _, parsed_count, uniform = field_info(stream)
                    if uniform is not None:
                        values = pt.tensor(uniform, dtype=self._dtype)
                        result[first - start : last - start] = (
                            values[0] if dim == 1 else values
                        )
                    else:
                        if parsed_count != local_count:
                            raise ValueError(f"Field size changed while reading {path}")
                        values = read_values(
                            stream,
                            metadata,
                            dim,
                            local_count,
                            first - offset,
                            last - offset,
                        )
                        target = result[first - start : last - start]
                        target.copy_(
                            pt.as_tensor(values, dtype=self._dtype).reshape(
                                target.shape
                            )
                        )
            offset += local_count
        return result

    def load_snapshot_slice(self, field_name, time, spatial_slice):
        """Load a contiguous cell range in processor concatenation order.

        Plain binary fields seek directly to the range. ASCII fields scan the
        preceding text; gzip streams decompress the preceding bytes. Neither
        path materializes a full nonuniform snapshot. Geometry remains eager.
        Non-unit strides are rejected explicitly.
        """
        check_list_or_str(field_name, "field_name")
        check_list_or_str(time, "time")
        if not isinstance(spatial_slice, slice):
            raise TypeError("spatial_slice must be a slice")
        if isinstance(field_name, list):
            return [
                self.load_snapshot_slice(name, time, spatial_slice)
                for name in field_name
            ]
        if isinstance(time, list):
            return _preallocate_time_series(
                lambda value: self._load_slice(field_name, value, spatial_slice), time
            )
        return self._load_slice(field_name, time, spatial_slice)

    def _load_multiple_snapshots(self, field_name: str, times: List[str]) -> pt.Tensor:
        """Load multiple snapshots of a single field.

        :param field_name: name of the field
        :type field_name: str
        :param times: list of write times to load
        :type times: List[str]
        :return: tensor holding multiple snapshots; the time dimension is always
            the last dimension
        :rtype: pt.Tensor
        """
        return _preallocate_time_series(
            lambda time: self._load_single_snapshot(field_name, time), times
        )

    def load_snapshot(
        self, field_name: Union[List[str], str], time: Union[List[str], str]
    ) -> Union[List[pt.Tensor], pt.Tensor]:
        check_list_or_str(field_name, "field_name")
        check_list_or_str(time, "time")
        # load multiple fields
        if isinstance(field_name, list):
            if isinstance(time, list):
                return [
                    self._load_multiple_snapshots(name, time) for name in field_name
                ]
            else:
                return [self._load_single_snapshot(name, time) for name in field_name]
        # load a single field
        else:
            if isinstance(time, list):
                return self._load_multiple_snapshots(field_name, time)
            else:
                return self._load_single_snapshot(field_name, time)

    @property
    def write_times(self) -> List[str]:
        """
        Access to available snapshot/write times via :func:`FOAMCase._eval_write_times`.

        :getter: returns the available write times
        :type: List[str]
        """
        return self._case.time_folders

    @property
    def field_names(self) -> Dict[str, List[str]]:
        """
        Access to the available field names for all available write times via
        :func:`FOAMCase._eval_field_names`.

        :getter: returns names of available fields
        :type: Dict[str, List[str]]
        """
        return self._case.field_names

    @property
    def vertices(self) -> pt.Tensor:
        """
        In OpenFOAM, fields for post-processing are defined at the control volume's
        center (*vol<Type>Fields*). Therefore, the `vertices` property enables access
        to cell center locations via :class:`FOAMMesh`.

        :getter: returns control volume centers
        :type: pt.Tensor
        """
        return self._mesh.get_cell_centers()

    @property
    def weights(self) -> pt.Tensor:
        """
        For results obtained using a finite volume method with co-located
        arrangement (OpenFOAM), a sensible weight for a cell-centered value
        is the cell volume. The cell volumes are available via the
        :class:`FOAMMesh` class.

        :getter: returns cell volumes
        :type: pt.Tensor
        """
        return self._mesh.get_cell_volumes()


class FOAMCase(object):
    """Class to access and parse OpenFOAM cases.

    Most of the attributes and methods are private because they are
    typically accessed via a :class:`FOAMDataloader` instance.

    .. automethod:: _eval_distributed
    .. automethod:: _eval_processors
    .. automethod:: _eval_write_times
    .. automethod:: _eval_field_names
    """

    def __init__(self, path: str, distributed: Union[bool, None] = None):
        """Create a `FOAMCase` instance based on a path.

        :param path: path to OpenFOAM simulation case
        :type path: str
        :param distributed: case is considered distributed if True; if None,
            the presence of processor folders is checked to evaluate the
            parameter; defaults to False
        :type distributed: bool
        """
        self._path = check_and_standardize_path(path)

        self._distributed = (
            distributed if distributed is not None else self._eval_distributed()
        )
        self._processors = self._eval_processors()
        self._time_folders = self._eval_write_times()
        self._field_names = self._eval_field_names()
        if not self._check_mesh_files():
            raise FileNotFoundError(f"Could not find valid mesh in case {self._path}")

    @staticmethod
    def _is_binary(header: List[Union[bytes, str]]) -> bool:
        """Determine if the writing format is binary.

        :param header: header of the OpenFOAM file
        :type header: List[str]
        :return: True if the format is binary
        :rtype: bool
        """
        for line in header:
            if b"format" in line:
                if b"binary" in line:
                    return True
                else:
                    return False
        return False

    def _check_mesh_files(self) -> bool:
        """Check if all mesh files are available.

        :return: True if all required files are available.
        :rtype: bool
        """
        if self._distributed:
            files_found = []
            for proc in range(self._processors):
                files_found += [
                    isfile(
                        resolve(
                            join(
                                self._path,
                                f"processor{proc:d}",
                                POLYMESH_PATH,
                                mesh_file,
                            )
                        )
                    )
                    for mesh_file in MESH_FILES
                ]
        else:
            files_found = [
                isfile(resolve(join(self._path, POLYMESH_PATH, mesh_file)))
                for mesh_file in MESH_FILES
            ]
        return all(files_found)

    def _eval_distributed(self) -> bool:
        """Check if the simulation case is distributed (parallel).

        .. warning::
            Collated output is currently not supported/not detected.

        :return: `True` if distributed
        :rtype: bool
        """
        proc_dirs = glob(join(self._path, "processor*"))
        return any(
            isdir(path) and re.fullmatch(r"processor\d+", path.rsplit("/", 1)[-1])
            for path in proc_dirs
        )

    def _eval_processors(self) -> int:
        """Get number of processor folders.

        :return: number of processor folders or 1 for serial runs
        :rtype: int
        """
        if self._distributed:
            ids = sorted(
                int(name[len("processor") :])
                for name in listdir(self._path)
                if re.fullmatch(r"processor\d+", name) and isdir(join(self._path, name))
            )
            if not ids or ids != list(range(len(ids))):
                raise ValueError(
                    "Expected contiguous processor folders starting at processor0"
                )
            return len(ids)
        else:
            return 1

    def _eval_write_times(self) -> List[str]:
        """Assemble a list of all write times.

        :return: a list of all time folders
        :rtype: List[str]

        .. warning::
            For distributed simulations, it is assumed that all processor
            folders contain the same time folders.
        """
        time_path = join(self._path, "processor0") if self._distributed else self._path
        dirs = [
            folder for folder in listdir(time_path) if isdir(join(time_path, folder))
        ]
        time_dirs = []
        for entry in dirs:
            try:
                # check if the time step can be converted to a real number
                _ = float(entry)
                time_dirs.append(entry)
            except ValueError:
                continue
        if len(time_dirs) < 2:
            logger.warning(
                "Found only one or less time folders in {:s}".format(self._path)
            )
        return sorted(time_dirs, key=float)

    def _eval_field_names(self) -> Dict[str, List[str]]:
        """Get a dictionary of all fields and files in all time folders.

        .. warning::
            For distributed cases, only *processor0* is evaluated. The fields
            for all other processors are assumed to be the same.

        :return: dictionary with write times as keys and a list of field names
            for each time as values
        :rtype: Dict[str, List[str]]
        """
        all_time_folders = [
            self.build_file_path("", time, 0) for time in self._time_folders
        ]
        return {
            self._time_folders[i]: [
                field
                for field in sorted(
                    {
                        name.removesuffix(".gz")
                        for name in listdir(folder)
                        if isfile(join(folder, name))
                    }
                )
            ]
            for i, folder in enumerate(all_time_folders)
        }

    def build_file_path(self, field_name: str, time: str, processor: int = 0) -> str:
        """Create the path to file inside the time folder of a simulation.

        :param field_name: name of the field or file, e.g., \"U\" or \"p\"
        :type field_name: str
        :param time: name of the time folder, e.g., \"0.01\"
        :type time: str
        :param processor: processor folder to load the data from; ignored
            in serial simulation cases; defaults to `0`
        :type processor: int, optional
        :return: path to file inside a time folder
        :rtype: str

        Examples

        >>> from flowtorch.data import FOAMCase
        >>> case = FOAMCase("./cavity_binary_parallel/")
        >>> case._distributed
        True
        >>> case._processors
        4
        >>> case._time_folders
        ['0', '0.1', '0.2', '0.3', '0.4', '0.5']
        >>> case._field_names
        {'0': ['U', 'p'], '0.1': ['U', 'p', 'phi'], '0.2': ['U', 'p', 'phi'], '0.3': [
            'U', 'p', 'phi'], '0.4': ['U', 'p', 'phi'], '0.5': ['U', 'p', 'phi']}
        >>> case.build_file_path("U", "0.1", 1)
        './cavity_binary_parallel/processor1/0.1/U'
        """
        if self._distributed:
            file_path = join(self._path, f"processor{processor:d}", time, field_name)
        else:
            file_path = join(self._path, time, field_name)
        return resolve(file_path) if field_name else file_path

    @property
    def distributed(self) -> bool:
        return self._distributed

    @property
    def processors(self) -> int:
        return self._processors

    @property
    def is_binary(self):
        return self._is_binary

    @property
    def path(self):
        return self._path

    @property
    def field_names(self):
        return self._field_names

    @property
    def time_folders(self):
        return self._time_folders


class FOAMMesh(object):
    """Class to load and process OpenFOAM meshes.

    OpenFOAM stores the finite volume mesh as a collection of several
    files located in *constant/polyMesh* or in *processorXX/constant/polyMesh*
    for serial and distributed cases, respectively. Even though OpenFOAM
    is a cell-centered finite volume method, the cell-centers and volumes are
    not explicitly stored. Instead, a so-called face-addressing storage is used.
    All internal faces have an owner cell and a neighbor cell. Boundary faces only
    have an owner cell. The mesh attributes are defined in several files:

    - **points**: list of vertices forming cell faces; the list index of a point is used as label
    - **faces**: list of all cell faces defined as point labels
    - **owner**: list of cell labels that are face owners
    - **neighbour**: list of cell labels that are face neighbors; BE spelling
    - **boundary**: definition of faces belonging to a patch

    Examples

    >>> from flowtorch.data import FOAMMesh
    >>> mesh = FOAMMesh.from_path("./")
    >>> centers = mesh.get_cell_centers()
    >>> centers.size()
    torch.Size([400, 3])
    >>> centers[:2]
    tensor([[0.0025, 0.0025, 0.0050],
            [0.0075, 0.0025, 0.0050]])
    >>> volumes = mesh.get_cell_volumes()
    >>> volumes.size()
    torch.Size([400])
    >>> volumes[:2]
    tensor([2.5000e-07, 2.5000e-07])

    .. warning::
            Dynamically changing meshes are currently not supported.

    .. warning::
            Distributed meshes may be parsed and concatenated, but
            the cell centers and volumes won't have the same ordering
            as when computed from a reconstructed mesh.

    .. automethod:: _compute_face_centers_and_areas
    .. automethod:: _compute_cell_centers_and_volumes
    """

    def __init__(self, case: FOAMCase, dtype: pt.dtype = DEFAULT_DTYPE):
        """
        Create FOAMMesh object based on :class:`FOAMCase`.
        """
        if not isinstance(case, FOAMCase):
            raise TypeError(
                "Error: case must be of type FOAMCase, not {:s}".format(
                    type(case).__name__
                )
            )
        self._case = case
        self._dtype = dtype
        self._itype = pt.int64
        self._cell_centers = None
        self._cell_volumes = None

    @classmethod
    def from_path(cls, path: str, dtype: pt.dtype = DEFAULT_DTYPE):
        """Create a new instance based on the simulation's path.

        :param path: path to the OpenFOAM simulation
        :type path: str
        :param dtype: tensor type, defaults to DEFAULT_DTYPE
        :type dtype: str, optional
        :return: FOAMMesh instance
        :rtype: FOAMMesh
        """
        return cls(FOAMCase(path), dtype)

    @staticmethod
    def _get_list_length(data: Sequence[bytes]) -> Tuple[int, int]:
        """Find list length of points, faces, and cells.

        :param data: content of points, faces, or owner/neighbor file
        :type data: List[str]
        :return: line with first list entry and number of elements in a list;
            returns zeros if the number of elements could not be determined
        :rtype: Tuple[int, int]
        """
        for i, line in enumerate(data):
            try:
                n_entries = int(line)
            except ValueError:
                pass
            else:
                return i, n_entries
        return 0, 0

    @staticmethod
    def _get_n_cells(mesh_path: str) -> int:
        """Extract number of cells from *owner* file.

        :param mesh_path: mesh location
        :type mesh_path: str
        :return: number of cells; defaults to 0
        :rtype: int
        """
        import numpy as np

        with foam_open(join(mesh_path, "owner")) as file:
            metadata = header(file)
            match = re.search(rb"nCells\s*:\s*(\d+)", metadata.get(b"note", b""))
            if match:
                return int(match[1])
            length = int(token(file))
            if length < 0 or token(file) != b"(":
                raise ValueError("Invalid owner list")
            maximum = -1
            if metadata.get(b"format") == b"binary":
                dtype = binary_dtype(metadata, "label")
                for start in range(0, length, 65536):
                    size = min(65536, length - start) * dtype.itemsize
                    raw = file.read(size)
                    if len(raw) != size:
                        raise ValueError("Truncated owner list")
                    values = np.frombuffer(raw, dtype=dtype)
                    if np.any(values < 0):
                        raise ValueError("Negative owner label")
                    maximum = max(maximum, int(values.max()))
            else:
                for _ in range(length):
                    value = int(token(file))
                    if value < 0:
                        raise ValueError("Negative owner label")
                    maximum = max(maximum, value)
            if token(file) != b")":
                raise ValueError("Missing owner list closing delimiter")
            return maximum + 1

    def _primitive_list(
        self, path: str, dim: int = 1, kind: str = "scalar"
    ) -> pt.Tensor:
        with foam_open(path) as stream:
            metadata = header(stream)
            if metadata.get(b"class", b"").startswith(b"vol"):
                stream.seek(0)
                metadata, field_dim, count, uniform = field_info(stream)
                if field_dim != dim or uniform is not None:
                    raise ValueError(
                        "Precomputed geometry requires a matching nonuniform field"
                    )
                values = read_values(stream, metadata, dim, count, 0, count)
                if dim > 1:
                    values = values.reshape(count, dim)
            else:
                values = read_list(stream, metadata, dim, kind)
        return pt.as_tensor(
            values, dtype=self._dtype if kind == "scalar" else self._itype
        )

    def _parse_points(self, mesh_path: str) -> pt.Tensor:
        """Read mesh points, including compressed and architecture-tagged files."""
        return self._primitive_list(join(mesh_path, "points"), 3)

    def _parse_faces(self, mesh_path: str) -> Tuple[pt.Tensor, pt.Tensor]:
        """Read ASCII faces or binary compact face addressing."""
        with foam_open(join(mesh_path, "faces")) as stream:
            metadata = header(stream)
            if metadata.get(b"format") == b"binary":
                if metadata.get(b"class") != b"faceCompactList":
                    raise ValueError("Only faceCompactList binary faces are supported")
                offsets = pt.as_tensor(
                    read_list(stream, metadata, kind="label"), dtype=self._itype
                )
                labels = pt.as_tensor(
                    read_list(stream, metadata, kind="label"), dtype=self._itype
                )
                if (
                    offsets.numel() < 1
                    or offsets[0] != 0
                    or offsets[-1] != labels.numel()
                    or pt.any(offsets[1:] < offsets[:-1])
                ):
                    raise ValueError("Invalid compact face offsets")
                counts = offsets[1:] - offsets[:-1]
                width = int(counts.max()) if counts.numel() else 0
                columns = pt.arange(width).unsqueeze(0)
                mask = columns < counts.unsqueeze(1)
                faces = pt.zeros((counts.numel(), width), dtype=self._itype)
                addressing = offsets[:-1].unsqueeze(1) + columns
                faces[mask] = labels[addressing[mask]]
            else:
                length = int(token(stream))
                if length < 0 or token(stream) != b"(":
                    raise ValueError("Invalid face list")
                rows = [
                    read_list(stream, metadata, kind="label") for _ in range(length)
                ]
                if token(stream) != b")":
                    raise ValueError("Missing face list closing delimiter")
                counts = pt.tensor([len(row) for row in rows], dtype=self._itype)
                width = int(counts.max()) if length else 0
                faces = pt.zeros((length, width), dtype=self._itype)
                if length:
                    import numpy as np

                    faces[pt.arange(width).unsqueeze(0) < counts.unsqueeze(1)] = (
                        pt.as_tensor(np.concatenate(rows), dtype=self._itype)
                    )
            if pt.any(counts < 3) or pt.any(faces < 0):
                raise ValueError(
                    "Faces must have at least three nonnegative point labels"
                )
            return counts.unsqueeze(1), faces

    def _parse_owners_and_neighbors(
        self, mesh_path: str
    ) -> Tuple[pt.Tensor, pt.Tensor]:
        """Read owner and neighbour labels with their declared integer width."""
        return (
            self._primitive_list(join(mesh_path, "owner"), kind="label"),
            self._primitive_list(join(mesh_path, "neighbour"), kind="label"),
        )

    @staticmethod
    def _centers_and_volumes_computed(path: str) -> bool:
        """Check if files *C* and *V* exist in the specified location.

        :param path: mesh location
        :type path: str
        :return: True if volumes and cell centers are available
        :rtype: bool
        """
        return isfile(resolve(join(path, "C"))) and isfile(resolve(join(path, "V")))

    def _parse_cell_centers(self, path: str) -> pt.Tensor:
        """Read precomputed cell centres from the constant directory."""
        return self._primitive_list(join(path, "C"), 3)

    def _parse_cell_volumes(self, path: str) -> pt.Tensor:
        """Read precomputed cell volumes from the constant directory."""
        return self._primitive_list(join(path, "V"))

    def _compute_face_centers_and_areas(
        self, points: pt.Tensor, faces: pt.Tensor, n_points_faces: pt.Tensor
    ) -> Tuple[pt.Tensor, pt.Tensor]:
        """Compute face centers and areas.

        The implemented algorithm is close to the one in makeFaceCentresAndAreas_.
        The main steps are:

        1. compute an estimate of the face center by averaging all face vertices
        2. decompose the face into triangles
        3. compute the sum over all area-weighted triangle centroids, triangle areas,
           and face area normal vectors
        4. compute the face centroid and face area normal from the (weighted) sums

        .. _makeFaceCentresAndAreas: https://www.openfoam.com/documentation/guides/latest/api/primitiveMeshFaceCentresAndAreas_8C_source.html

        :param points: list of points
        :type points: pt.Tensor
        :param faces: list point labels forming each face
        :type faces: pt.Tensor
        :param n_points_faces: number of points forming each face
        :type n_points_faces: pt.Tensor
        :return: tuple of two tensors; the first one holds the face centers
            and the second one holds the face areas
        :rtype: Tuple[pt.Tensor, pt.Tensor]
        """
        face_centers = pt.zeros((n_points_faces.size()[0], 3), dtype=self._dtype)
        face_areas = pt.zeros_like(face_centers, dtype=self._dtype)
        center_estimates = pt.zeros_like(face_centers, dtype=self._dtype)

        for i in range(faces.shape[1]):
            center_estimates += points[faces[:, i]] * pt.where(i < n_points_faces, 1, 0)
        center_estimates /= n_points_faces

        area_sums = pt.zeros_like(n_points_faces, dtype=self._dtype)
        for i in range(faces.shape[1]):
            this_point_mask = pt.where(i < n_points_faces, 1, 0)
            next_point_mask = pt.where(i + 1 < n_points_faces, 1, 0)
            last_point_mask = pt.where(i + 1 == n_points_faces, 1, 0)
            this_point = points[faces[:, i]] * this_point_mask
            if i + 1 < faces.shape[1]:
                next_point = points[faces[:, i + 1]] * next_point_mask
            else:
                next_point = pt.zeros_like(face_centers, dtype=self._dtype)
            next_point += points[faces[:, 0]] * last_point_mask
            c = center_estimates * this_point_mask + this_point + next_point
            n = pt.cross(
                next_point - this_point,
                center_estimates * this_point_mask - this_point,
                dim=1,
            )
            a = pt.norm(n, dim=1).unsqueeze(1)
            face_centers += c * a
            area_sums += a
            face_areas += n

        if not pt.all(pt.isfinite(area_sums) & (area_sums > 0)):
            raise ValueError("Mesh contains degenerate or invalid faces")
        face_centers /= area_sums * 3.0
        face_areas *= 0.5

        return face_centers, face_areas

    def _compute_cell_centers_and_volumes(
        self, mesh_path: str
    ) -> Tuple[pt.Tensor, pt.Tensor]:
        """Compute the cell centers and volumes of an OpenFOAM mesh.

        The implemented algorithm is close to the one in makeCellCentresAndVols_.
        The following steps are involved:

        1. compute an estimate of the cell center as the average over all face centers
        2. compute centroids and volumes of all pyramids formed by the cell faces and
           the center estimate
        3. the cell volume equals the sum over all pyramid volumes
        4. the cell center is the volume-weighted average of all pyramid centroids

        .. _makeCellCentresAndVols: https://www.openfoam.com/documentation/guides/latest/api/primitiveMeshCellCentresAndVols_8C_source.html

        :param mesh_path: mesh location
        :type mesh_path: str
        :return: tuple of two tensors; the first one holds the cell centers and
            the second one holds the cell volumes
        :rtype: Tuple[pt.Tensor, pt.Tensor]
        """
        points = self._parse_points(mesh_path)
        n_points_faces, faces = self._parse_faces(mesh_path)
        owners, neighbors = self._parse_owners_and_neighbors(mesh_path)
        face_centers, face_areas = self._compute_face_centers_and_areas(
            points, faces, n_points_faces
        )
        n_cells = self._get_n_cells(mesh_path)
        cell_centers = pt.zeros((n_cells, 3), dtype=self._dtype)
        cell_volumes = pt.zeros(n_cells, dtype=self._dtype)
        center_estimate = pt.zeros_like(cell_centers)
        n_faces_cell = pt.zeros(n_cells, dtype=self._itype)

        if (
            owners.numel() != faces.shape[0]
            or neighbors.numel() > owners.numel()
            or pt.any(owners < 0)
            or pt.any(owners >= n_cells)
            or pt.any(neighbors < 0)
            or pt.any(neighbors >= n_cells)
        ):
            raise ValueError("Invalid owner/neighbour addressing")
        center_estimate.index_add_(0, owners, face_centers)
        center_estimate.index_add_(0, neighbors, face_centers[: neighbors.numel()])
        n_faces_cell.index_add_(0, owners, pt.ones_like(owners))
        n_faces_cell.index_add_(0, neighbors, pt.ones_like(neighbors))
        if pt.any(n_faces_cell == 0):
            raise ValueError("Mesh contains cells without faces")
        center_estimate /= n_faces_cell.unsqueeze(-1)

        for addressing, sign in ((owners, 1), (neighbors, -1)):
            centers = face_centers[: addressing.numel()]
            areas = face_areas[: addressing.numel()]
            estimates = center_estimate[addressing]
            pyramid_volume = sign * pt.sum(areas * (centers - estimates), dim=1)
            pyramid_center = 0.75 * centers + 0.25 * estimates
            cell_centers.index_add_(
                0, addressing, pyramid_volume.unsqueeze(1) * pyramid_center
            )
            cell_volumes.index_add_(0, addressing, pyramid_volume)

        if not pt.all(pt.isfinite(cell_volumes) & (cell_volumes > 0)):
            raise ValueError("Mesh cell volumes must be finite and positive")
        cell_centers /= cell_volumes.unsqueeze(-1)
        cell_volumes /= 3.0

        return cell_centers, cell_volumes

    def _load_mesh(self):
        """Load or compute cell volumes and centers.

        .. warning:: For distributed cases, individual processor fields
        are simply concatenated. This reconstruction does not yield volumes
        and centers in the same order as the reconstructed OpenFOAM mesh.
        """
        if self._case.distributed:
            proc_data = []
            for proc in range(self._case.processors):
                if self._centers_and_volumes_computed(
                    join(self._case.path, f"processor{proc:d}", CONSTANT_PATH)
                ):
                    logger.info(
                        f"Loading precomputed cell centers and volumes from processor{proc:d}/{CONSTANT_PATH}"
                    )
                    mesh_location = join(
                        self._case.path, f"processor{proc:d}", CONSTANT_PATH
                    )
                    proc_data.append(
                        (
                            self._parse_cell_centers(mesh_location),
                            self._parse_cell_volumes(mesh_location),
                        )
                    )
                else:
                    mesh_location = join(
                        self._case.path, f"processor{proc:d}", POLYMESH_PATH
                    )
                    logger.warning(
                        f"Could not find precomputed cell centers and volumes (processor{proc:d}).\n"
                        + "Computing cell geometry from scratch (slow, not recommended for large meshes).\n"
                        + "To compute cell centers and volumes in OpenFOAM, use:\n\n"
                        + 'runParallel postProcess -func "writeCellCentres" -constant -time none\n'
                        + 'runParallel postProcess -func "writeCellVolumes" -constant -time none\n'
                    )
                    proc_data.append(
                        self._compute_cell_centers_and_volumes(mesh_location)
                    )
            centers = pt.cat(list(zip(*proc_data))[0])
            volumes = pt.cat(list(zip(*proc_data))[1])
        else:
            if self._centers_and_volumes_computed(join(self._case.path, CONSTANT_PATH)):
                logger.info(
                    f"Loading precomputed cell centers and volumes from {CONSTANT_PATH}"
                )
                centers = self._parse_cell_centers(join(self._case.path, CONSTANT_PATH))
                volumes = self._parse_cell_volumes(join(self._case.path, CONSTANT_PATH))
            else:
                mesh_location = join(self._case.path, POLYMESH_PATH)
                logger.warning(
                    "Could not find precomputed cell centers and volumes.\n"
                    + "Computing cell geometry from scratch (slow, not recommended for large meshes).\n"
                    + "To compute cell centers and volumes in OpenFOAM, run:\n\n"
                    + 'postProcess -func "writeCellCentres" -constant -time none\n'
                    + 'postProcess -func "writeCellVolumes" -constant -time none'
                )
                centers, volumes = self._compute_cell_centers_and_volumes(mesh_location)
        if not pt.all(pt.isfinite(centers)) or not pt.all(
            pt.isfinite(volumes) & (volumes > 0)
        ):
            raise ValueError("Cell geometry must be finite with positive volumes")
        self._cell_centers = centers
        self._cell_volumes = volumes

    def get_cell_centers(self) -> pt.Tensor:
        """Return or compute and return control volume centers.

        :return: control volume centers
        :rtype: pt.Tensor
        """
        if self._cell_centers is None:
            self._load_mesh()
        return self._cell_centers

    def get_cell_volumes(self) -> pt.Tensor:
        """Return or compute and return cell volumes.

        :return: cell volumes
        :rtype: pt.Tensor
        """
        if self._cell_volumes is None:
            self._load_mesh()
        return self._cell_volumes
