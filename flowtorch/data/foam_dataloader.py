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

from sys import exit
from glob import glob
from os import listdir
from os.path import isdir, isfile, join
from typing import List, Dict, Sequence, Tuple, Union

# third party packages
import torch as pt

# flowtorch packages
from flowtorch import DEFAULT_DTYPE
from .dataloader import Dataloader
from .utils import check_and_standardize_path, check_list_or_str

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# volSymmTensorField as 6 entries since it is symmetric:
# the order is: XX, XY, XZ, YY, YZ, ZZ
# volTensorField as 9 entries since:
# the order is: XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ
FIELD_TYPE_DIMENSION = {
    b"volScalarField": 1,
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
        """Parse field data from an OpenFOAM snapshot.

        :param data: full content of the snapshot file (including header
            and metadata)
        :type data: List[str]
        :raises ValueError: if the field type is not supported
        :return: numerical values stored in the snapshot
        :rtype: pt.Tensor
        """
        field_type = self._field_type(data[:MAX_LINE_HEADER])
        if not field_type in FIELD_TYPE_DIMENSION.keys():
            type_name = (
                field_type.decode("utf-8") if field_type is not None else "unknown"
            )
            raise ValueError(f"Field type {type_name} not supported.")
        try:
            if self._case.is_binary(data[:MAX_LINE_HEADER]):
                field_data = self._unpack_internal_field_binary(
                    data, FIELD_TYPE_DIMENSION[field_type]
                )
            else:
                field_data = self._unpack_internal_field_ascii(
                    data, FIELD_TYPE_DIMENSION[field_type]
                )
        except Exception as e:
            logger.error(e)
        else:
            return field_data

    @staticmethod
    def _find_nonuniform(data: Sequence[bytes]) -> Tuple[int, int]:
        """Determine the start index and size of a nonuniform field.

        OpenFOAM output files with nonuniform fields contain the line
            nonuniform
            #####
            (...
            )
        where ##### is an integer number indicating the number of list
        entries.
        This number is needed.

        :param data: content of an OpenFOAM field file
        :type data: List[str]
        :return: line index containing the keyword *nonuniform* and the size of the nonuniform list;
                 if the keyword is not found, two zeros are returned
        :rtype: Tuple[int, int]
        """
        for i, line in enumerate(data):
            if b"nonuniform" in line:
                return i, int(data[i + 1])
        return 0, 0

    @staticmethod
    def _field_type(data: Sequence[bytes]) -> bytes | None:
        """Determine the tensor type of a field.

        :param data: content of an OpenFoam output file
        :type data: List[str]
        :return: the field type; if the type is not supported or not found,
        `None` is returned
        :rtype: str
        """
        for line in data:
            if b"class" in line:
                return line.split()[-1].strip(b";")
        return None

    def _unpack_internal_field_ascii(self, data: List[bytes], dim: int) -> pt.Tensor:
        """Parse internal field given as ASCII-encoded text.

        :param data: content of output file
        :type data: List[str]
        :param dim: dimensionality of the field, e.g., 1 for a scalar or
            3 for a vector
        :type dim: int
        :return: tensor holding the field data as numerical values
        :rtype: pt.Tensor

        """
        start, n_values = self._find_nonuniform(data[:MAX_LINE_INTERNAL_FIELD])
        start += 3
        if dim == 1:
            return pt.tensor(
                [float(line) for line in data[start : start + n_values]],
                dtype=self._dtype,
            )
        else:
            return pt.tensor(
                [
                    list(map(float, line[1:-2].split()))
                    for line in data[start : start + n_values]
                ],
                dtype=self._dtype,
            )

    def _unpack_internal_field_binary(self, data: List[bytes], dim: int) -> pt.Tensor:
        """Parse internal field given in binary encoding.

        :param data: content of output file
        :type data: List[str]
        :param dim: dimensionality of the field, e.g., 1 for a scalar, 3 for a vector or 9 for a tensor
        :type dim: int
        :return: tensor holding the field data
        :rtype: pt.Tensor
        """
        start, n_values = self._find_nonuniform(data[:MAX_LINE_INTERNAL_FIELD])
        start += 2
        buffer = b"".join(data[start:])
        values = struct.unpack(
            "{}d".format(dim * n_values),
            buffer[SIZE_OF_CHAR : SIZE_OF_CHAR + SIZE_OF_DOUBLE * n_values * dim],
        )
        if dim > 1:
            return pt.tensor(values, dtype=self._dtype).reshape(n_values, dim)
        else:
            return pt.tensor(values, dtype=self._dtype)

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
        file_paths = []
        if self._case.distributed:
            for proc in range(self._case.processors):
                file_paths.append(self._case.build_file_path(field_name, time, proc))
        else:
            file_paths.append(self._case.build_file_path(field_name, time, 0))
        field_data = []
        for file_path in file_paths:
            with open(file_path, "rb") as file:
                field_data.append(self._parse_data(file.readlines()))
        return pt.cat(field_data)

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
        return pt.stack(
            [self._load_single_snapshot(field_name, time) for time in times], dim=-1
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
            exit("Error: could not find valid mesh in case {:s}".format(self._path))

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
                        join(self._path, f"processor{proc:d}", POLYMESH_PATH, mesh_file)
                    )
                    for mesh_file in MESH_FILES
                ]
        else:
            files_found = [
                isfile(join(self._path, POLYMESH_PATH, mesh_file))
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
        return len(proc_dirs) > 0

    def _eval_processors(self) -> int:
        """Get number of processor folders.

        :return: number of processor folders or 1 for serial runs
        :rtype: int
        """
        if self._distributed:
            return len(glob(join(self._path, "processor*")))
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
                field for field in listdir(folder) if isfile(join(folder, field))
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
        return file_path

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
            exit(
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
        n_cells = 0
        with open(join(mesh_path, "owner"), "rb") as file:
            found = False
            while not found:
                line = file.readline()
                if b"note" in line:
                    tokens = line.split()
                    for token in tokens:
                        if b"nCells" in token:
                            n_cells = int(token.split(b":")[1])
                            found = True
        return n_cells

    def _parse_points(self, mesh_path: str) -> pt.Tensor:
        """Parse mesh vertices defined in *constant/polyMesh/points*.

        :param mesh_path: mesh location
        :type mesh_path: str
        :return: tensor holding the mesh vertices
        :rtype: pt.Tensor
        """
        with open(join(mesh_path, "points"), "rb") as file:
            data = file.readlines()
            start, length = self._get_list_length(data[:MAX_LINE_HEADER])
            if self._case.is_binary(data[:MAX_LINE_HEADER]):
                start += 1
                buffer = b"".join(data[start:])
                values = struct.unpack(
                    "{}d".format(3 * length),
                    buffer[SIZE_OF_CHAR : SIZE_OF_CHAR + SIZE_OF_DOUBLE * length * 3],
                )
                return pt.tensor(values, dtype=self._dtype).reshape(length, 3)
            else:
                start += 2
                return pt.tensor(
                    [
                        list(map(float, line[1:-2].split()))
                        for line in data[start : start + length]
                    ],
                    dtype=self._dtype,
                )

    def _parse_faces(self, mesh_path: str) -> Tuple[pt.Tensor, pt.Tensor]:
        """Parse cell faces stored in *constant/polyMesh/faces*.

        :param mesh_path: mesh location
        :type mesh_path: str
        :return: tuple of tensors with the first tensor holding the
            number of points per face and the second one holding the
            point labels forming the face; the second dimension of the
            point label tensor is determined by the face with the highest
            number of points; faces with fewer points are padded with zeros
        :rtype: Tuple[pt.Tensor, pt.Tensor]
        """

        def zero_pad(tensor, new_size):
            """Increase the size of second tensor dimension."""
            diff = new_size - tensor.size()[1]
            pad = pt.zeros((tensor.size()[0], diff), dtype=self._itype)
            return pt.cat([tensor, pad], dim=1)

        with open(join(mesh_path, "faces"), "rb") as file:
            data = file.readlines()
            start, length = self._get_list_length(data[:MAX_LINE_HEADER])
            if self._case.is_binary(data[:MAX_LINE_HEADER]):
                n_points_faces = pt.zeros((length - 1, 1), dtype=self._itype)
                faces = pt.zeros_like(n_points_faces, dtype=self._itype)
                start += 1
                buffer = b"".join(data[start:])
                idx = struct.unpack(
                    "{}i".format(length),
                    buffer[SIZE_OF_CHAR : SIZE_OF_CHAR + SIZE_OF_INT * length],
                )

                # search for the next opening bracket to see where the second list starts
                # the length of the second list is interpreted as a sequence of characters,
                # so looking 50 characters ahead allows for a very large number of faces
                list_0_end = SIZE_OF_INT * length
                offset = 0
                for i, c in enumerate(
                    buffer[list_0_end : list_0_end + 50 * SIZE_OF_CHAR]
                ):
                    if chr(c) == r"(":
                        offset = i + 1
                values = struct.unpack(
                    "{}i".format(idx[-1]),
                    buffer[
                        offset
                        + SIZE_OF_INT * length : offset
                        + (length + idx[-1]) * SIZE_OF_INT
                    ],
                )
                for i in range(length - 1):
                    face_labels = pt.tensor(
                        values[idx[i] : idx[i + 1]], dtype=self._itype
                    )
                    n_points_faces[i] = len(face_labels)
                    if len(face_labels) > faces.size()[1]:
                        faces = zero_pad(faces, len(face_labels))
                    faces[i][: len(face_labels)] = face_labels
            else:
                n_points_faces = pt.zeros((length, 1), dtype=self._itype)
                faces = pt.zeros_like(n_points_faces, dtype=self._itype)
                start += 2
                for i, line in enumerate(data[start : start + length]):
                    n_points_faces[i] = int(line[:1])
                    face_labels = pt.tensor(
                        list(map(int, line[2:-2].split())), dtype=self._itype
                    )
                    if len(face_labels) > faces.size()[1]:
                        faces = zero_pad(faces, len(face_labels))
                    faces[i][: len(face_labels)] = face_labels
            return n_points_faces, faces

    def _parse_owners_and_neighbors(
        self, mesh_path: str
    ) -> Tuple[pt.Tensor, pt.Tensor]:
        """Parse face owners and neighbors.

        - owners are parsed from *constant/polyMesh/owner*
        - neighbors are parsed from *constant/polyMesh/neighbor*

        :param mesh_path: mesh location
        :type mesh_path: str
        :return: tensor holding the cell labels owning or neighboring
            a face
        :rtype: Tuple[pt.Tensor, pt.Tensor]
        """
        with open(join(mesh_path, "owner"), "rb") as file:
            data = file.readlines()
            start, length = self._get_list_length(data[:MAX_LINE_HEADER])
            owner_values: Sequence[int]
            if self._case.is_binary(data[:MAX_LINE_HEADER]):
                start += 1
                buffer = b"".join(data[start:])
                owner_values = struct.unpack(
                    "{}i".format(length),
                    buffer[SIZE_OF_CHAR : SIZE_OF_CHAR + SIZE_OF_INT * length],
                )
            else:
                start += 2
                owner_values = [int(line[:-1]) for line in data[start : start + length]]

        with open(join(mesh_path, "neighbour"), "rb") as file:
            data = file.readlines()
            start, length = self._get_list_length(data[:MAX_LINE_HEADER])
            neighbor_values: Sequence[int]
            if self._case.is_binary(data[:MAX_LINE_HEADER]):
                start += 1
                buffer = b"".join(data[start:])
                neighbor_values = struct.unpack(
                    "{}i".format(length),
                    buffer[SIZE_OF_CHAR : SIZE_OF_CHAR + SIZE_OF_INT * length],
                )
            else:
                start += 2
                neighbor_values = [
                    int(line[:-1]) for line in data[start : start + length]
                ]

        return (
            pt.tensor(owner_values, dtype=self._itype),
            pt.tensor(neighbor_values, dtype=self._itype),
        )

    @staticmethod
    def _centers_and_volumes_computed(path: str) -> bool:
        """Check if files *C* and *V* exist in the specified location.

        :param path: mesh location
        :type path: str
        :return: True if volumes and cell centers are available
        :rtype: bool
        """
        return isfile(join(path, "C")) and isfile(join(path, "V"))

    def _parse_cell_centers(self, path: str) -> pt.Tensor:
        """Parse cell centers from the constant directory.

        :param path: mesh location
        :type path: str
        :return: tensor holding the cell center coordinates
        :rtype: pt.Tensor
        """
        with open(join(path, "C"), "rb") as file:
            data = file.readlines()
            start, length = self._get_list_length(data[:MAX_LINE_HEADER])
            if self._case.is_binary(data[:MAX_LINE_HEADER]):
                start += 1
                buffer = b"".join(data[start:])
                values = struct.unpack(
                    "{}d".format(3 * length),
                    buffer[SIZE_OF_CHAR : SIZE_OF_CHAR + SIZE_OF_DOUBLE * length * 3],
                )
                return pt.tensor(values, dtype=self._dtype).reshape(length, 3)
            else:
                start += 2
                return pt.tensor(
                    [
                        list(map(float, line[1:-2].split()))
                        for line in data[start : start + length]
                    ],
                    dtype=self._dtype,
                )

    def _parse_cell_volumes(self, path: str) -> pt.Tensor:
        """Parse cell volumes from the constant directory.

        :param path: mesh location
        :type path: str
        :return: tensor holding the cell volumes
        :rtype: pt.Tensor
        """
        with open(join(path, "V"), "rb") as file:
            data = file.readlines()
            start, length = self._get_list_length(data[:MAX_LINE_HEADER])
            if self._case.is_binary(data[:MAX_LINE_HEADER]):
                start += 1
                buffer = b"".join(data[start:])
                values = struct.unpack(
                    "{}d".format(length),
                    buffer[SIZE_OF_CHAR : SIZE_OF_CHAR + SIZE_OF_DOUBLE * length],
                )
                return pt.tensor(values, dtype=self._dtype)
            else:
                start += 2
                return pt.tensor(
                    [float(line[:-1]) for line in data[start : start + length]],
                    dtype=self._dtype,
                )

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

        for i, owner in enumerate(owners):
            center_estimate[owner] += face_centers[i]
            n_faces_cell[owner] += 1

        for i, neigh in enumerate(neighbors):
            center_estimate[neigh] += face_centers[i]
            n_faces_cell[neigh] += 1
        center_estimate /= n_faces_cell.unsqueeze(-1)

        for i, owner in enumerate(owners):
            pyr_3vol = pt.dot(face_areas[i], face_centers[i] - center_estimate[owner])
            pyr_ctr = 3.0 / 4.0 * face_centers[i] + center_estimate[owner] / 4.0
            cell_centers[owner] += pyr_3vol * pyr_ctr
            cell_volumes[owner] += pyr_3vol

        for i, neigh in enumerate(neighbors):
            pyr_3vol = pt.dot(face_areas[i], center_estimate[neigh] - face_centers[i])
            pyr_ctr = 3.0 / 4.0 * face_centers[i] + center_estimate[neigh] / 4.0
            cell_centers[neigh] += pyr_3vol * pyr_ctr
            cell_volumes[neigh] += pyr_3vol

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
