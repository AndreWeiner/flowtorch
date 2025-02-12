"""Module to read and write the internal flowTorch data format.

The :class:`HDF5Writer` class allows to write field and mesh data
into an HDF5 file. It also creates an XDMF accessor file for postprocessing
in ParaView. The XDMF file creation is implemented in the :class:`XDMFWriter`
class. The :class:`HDF5Dataloader` class is the complementary writer for the
flowTorch data format. Moreover, the :class:`FOAM2HDF5` class allows conversion
of reconstructed OpenFOAM simulation cases into the HDF5-based flowTorch format.
"""

# standard library packages
from os.path import isfile, exists, join
from os import remove
from typing import List, Dict, Union

# third party packages
import torch as pt
from h5py import File
# flowtorch packages
from flowtorch import DEFAULT_DTYPE
from .dataloader import Dataloader
from .foam_dataloader import FOAMDataloader, POLYMESH_PATH, MAX_LINE_HEADER, FIELD_TYPE_DIMENSION
from .utils import check_list_or_str, check_and_standardize_path


CONST_GROUP = "constant"
VAR_GROUP = "variable"
VERTICES_DS = "vertices"
CONNECTIVITY_DS = "connectivity"
CENTERS_DS = "centers"
VOLUMES_DS = "volumes"
XDMF_HEADER = """<source lang="xml" line="1">
    <Domain>
        <Grid Name="flowTorch" GridType="Collection" CollectionType="Temporal">
"""
XDMF_FOOTER = """
        </Grid>
    </Domain>
</source> 
"""
TOPOLOGY = "topology"
GEOMETRY = "geometry"

dtype_conversion = {
    pt.float32: "f4",
    pt.float64: "f8",
    pt.int32: "i4",
    pt.int64: "i8",
    "float32": "f4",
    "float64": "f8",
    "int32": "i4",
    "int64": "i8"
}

xdmf_attributes = {
    1: "Scalar",
    2: "Vector",
    3: "Vector",
    6: "Tensor6",
    9: "Tensor"
}

xdmf_dtype_conversion = {
    "f4": ("Float", 4),
    "f8": ("Float", 8),
    "i4": ("Int", 4),
    "i8": ("Int", 8),
    "float32": ("Float", 4),
    "float64": ("Float", 8),
    "int32": ("Int", 4),
    "int64": ("Int", 8)
}


class HDF5Dataloader(Dataloader):
    """Load HDF5-based flowTorch data.

    Examples

    >>> from flowtorch import HDF5Dataloader
    >>> loader = HDF5Dataloader("flowtorch.hdf5")
    >>> times = loader.write_times
    >>> p, U = loader.load_snapshot(["p", "U"], write_times)
    >>> vertices = loader.vertices

    """

    def __init__(self, file_path: str, dtype: str = DEFAULT_DTYPE):
        """Create HDF5Dataloader instance.

        :param file_path: path to HDF5 file
        :type file_path: str
        :param dtype: tensor type of floating point values, defaults to DEFAULT_DTYPE
        :type dtype: str, optional
        """
        if not isfile(file_path):
            raise FileNotFoundError(f"Could not find file {file_path}")
        self._file_path = file_path
        self._dtype = dtype
        self._file = File(self._file_path, mode="a")

    def load_snapshot(self, field_name: Union[List[str], str],
                      time: Union[List[str], str]) -> Union[List[pt.Tensor], pt.Tensor]:
        check_list_or_str(field_name, "field_name")
        check_list_or_str(time, "time")
        # load multiple fields
        if isinstance(field_name, list):
            if isinstance(time, list):
                return [pt.stack([pt.tensor(
                        self._file[f"{VAR_GROUP}/{t}/{field}"][:].copy(), dtype=self._dtype
                        ).squeeze() for t in time], dim=-1) for field in field_name]
            else:
                return [pt.tensor(
                        self._file[f"{VAR_GROUP}/{time}/{field}"][:].copy(), dtype=self._dtype
                        ).squeeze() for field in field_name]
        # load single field
        else:
            if isinstance(time, list):
                return pt.stack([pt.tensor(
                    self._file[f"{VAR_GROUP}/{t}/{field_name}"][:].copy(), dtype=self._dtype
                ).squeeze() for t in time], dim=-1)
            else:
                return pt.tensor(
                    self._file[f"{VAR_GROUP}/{time}/{field_name}"][:].copy(), dtype=self._dtype
                ).squeeze()

    @property
    def write_times(self) -> List[str]:
        return list(self._file[VAR_GROUP].keys())

    @property
    def field_names(self) -> Dict[str, List[str]]:
        times = self.write_times
        field_names = dict.fromkeys(times, [])
        for time in times:
            field_names[time] = list(self._file[f"{VAR_GROUP}/{time}"].keys())
        return field_names

    @property
    def vertices(self) -> pt.Tensor:
        return pt.tensor(
            self._file[f"{CONST_GROUP}/{CENTERS_DS}"][:], dtype=self._dtype
        )

    @property
    def weights(self) -> pt.Tensor:
        return pt.tensor(
            self._file[f"{CONST_GROUP}/{VOLUMES_DS}"][:], dtype=self._dtype
        )

    @property
    def connectivity(self) -> pt.Tensor:
        return pt.tensor(
            self._file[f"{CONST_GROUP}/{CONNECTIVITY_DS}"][:], dtype=self._dtype
        )

    @property
    def edge_vertices(self) -> pt.Tensor:
        return pt.tensor(
            self._file[f"{CONST_GROUP}/{VERTICES_DS}"][:], dtype=self._dtype
        )


class HDF5Writer(object):
    """Class to write flowTorch data to HDF5 file.

    Two types of data are supported:
    - variable: (field) data that changes with times, e.g., snapshots
    - constant: constant data like mesh vertices or cell volumes

    An XDMF accessor file can be created to support visual post-processing
    with ParaView and other XDMF-compatible software packages.

    Examples

    >>> import torch as pt
    >>> from flowtorch.data import HDF5Writer
    >>> writer = HDF5Writer("test_file.hdf5")
    >>> writer.write("ones", (3, 2), pt.ones((3, 2)), "0.01")
    >>> int_data =  pt.ones((3, 2), dtype=pt.int32)
    >>> writer.write("ones_int", (3, 2), int_data, dtype=pt.int32)

    """

    def __init__(self, file: str):
        """Write flowTorch HDF5 files.

        :param file: path and file name to HDF5 file.
        :type file: str
        """
        self._file_path = file
        self._file = File(file, mode="a")

    def write(self,
              name: str,
              size: tuple,
              data: pt.Tensor = None,
              time: str = None,
              dtype: str = DEFAULT_DTYPE
              ):
        """Write data to HDF5 file.

        :param name: dataset name
        :type name: str
        :param size: dataset shape
        :type size: tuple
        :param data: data to write; if None, dataset is only allocated
        :type data: pt.Tensor, optional
        :param time: snapshot time, dataset if created in VAR_GROUP if present
        :type time: str, optional
        :param dtype: data type, defaults to pt.float32
        :type dtype: str, optional
        """
        if time is not None:
            ds_name = VAR_GROUP + "/{:s}/{:s}".format(time, name)
        else:
            ds_name = CONST_GROUP + "/{:s}".format(name)

        if dtype in dtype_conversion.keys():
            if ds_name in self._file:
                del self._file[ds_name]

            ds = self._file.create_dataset(
                ds_name,
                size,
                dtype=dtype_conversion[dtype]
            )
            if data is not None:
                shape_diff = len(size) - len(data.size())
                if shape_diff == 1:
                    data = data.unsqueeze(-1)
                elif shape_diff == -1:
                    data = data.squeeze()
                ds[:] = data.numpy()
        else:
            print(
                "Warning: invalid data type {:s} for field {:s}. Skipping field.".format(
                    str(dtype), name)
            )

    def write_xdmf(self):
        """Write XDMF wrapper to access flowTorch HDF5 files in ParaView.
        """
        writer = XDMFWriter(self._file_path, self._file)
        writer.create_xdmf("flowtorch.xdmf")


class FOAM2HDF5(object):
    """Convert reconstructed OpenFOAM cases to flowTorch HDF5 format.

    If the simulation case is decomposed into processor folders/domains,
    an info statement is displayed and no conversion is performed.

    Examples

    >>> from flowtorch import DATASETS
    >>> from flowtorch.data import FOAM2HDF5
    >>> converter = FOAM2HDF5(DATASETS["of_cavity_ascii"])
    >>> converter.convert("cavity.hdf5", ["U", "p"], ["0.1", "0.2", "0.3"])

    """

    def __init__(self, path: str, dtype=DEFAULT_DTYPE):
        """Create FOAM2HDF5 converter from a path.

        :param path: path to OpenFOAM case
        :type path: str
        :param dtype: tensor type, defaults to DEFAULT_DTYPE
        :type dtype: str, optional
        """
        self._loader = FOAMDataloader(path, dtype, False)
        self._dtype = dtype
        self._topology = None
        self._mesh_points = None

    def convert(self, filename: str, fields: List[str] = None,
                times: List[str] = None):
        """Convert OpenFOAM case to flowTorch HDF5 file.

        :param filename: name of the HDF5 file
        :type filename: str
        :param fields: list of fields to convert; if None, all available
            fields are converted
        :type fields: List[str], optional
        :param times: list of times to convert; if None, all available
            times are converted
        :type times: List[str], optional
        """
        file_path = self._loader._case._path + "/" + filename
        self._remove_file_if_present(file_path)
        # this is currently redundant since the loader is initialized
        # with distributed set to False
        if self._loader._case._distributed:
            message = """The direct conversion of distributed cases is currently not supported.\n
Workaround:
    1. run reconstructPar (OpenFOAM utility)
    1.1 if there is no reconstructed mesh, run also reconstructParMesh
    2. remove all processor* folders
    3. perform the conversion again (flowTorch)
            """
            print(message)
        else:
            print("Writing data to file {:s}".format(file_path))
            writer = HDF5Writer(file_path)
            print("Converting mesh.")
            self._convert_mesh(writer)
            print("Converting fields.")
            self._convert_fields(writer, fields, times)
            print("Conversion finished. Writing XDMF file.")
            writer.write_xdmf()

    def _remove_file_if_present(self, file_path: str):
        """Remove output file from previous runs if present

        :param file_path: path to file
        :type file_path: str
        """
        if exists(file_path):
            print("Removing old file {:s}".format(file_path))
            remove(file_path)

    def _convert_mesh(self, writer: HDF5Writer):
        """Convert OpenFOAM mesh to XDMF-based structure.

        :param writer: HDF5 file writer
        :type writer: HDF5Writer
        """
        mesh_path = self._loader._case._path + "/" + POLYMESH_PATH
        n_cells, n_points, n_top = self._gather_mesh_information(mesh_path)
        data = self._get_vertices(mesh_path, job=0)
        writer.write(VERTICES_DS, (n_points, 3), data, None, self._dtype)
        data = self._get_topology(job=0)
        writer.write(CONNECTIVITY_DS, (n_top,), data, None, pt.int32)
        data = self._get_cell_centers(job=1)
        writer.write(CENTERS_DS, (n_cells, 3), data, None, self._dtype)
        data = self._get_cell_volumes(job=1)
        writer.write(VOLUMES_DS, (n_cells,), data, None, self._dtype)

    def _gather_mesh_information(self, mesh_path: str):
        """Gather information for parallel writing of mesh data.

        :param mesh_path: path to polyMesh folder
        :type mesh_path: str
        """
        n_cells = self._loader._mesh._get_n_cells(mesh_path)
        n_points_faces, faces = self._loader._mesh._parse_faces(mesh_path)
        owners, neighbors = self._loader._mesh._parse_owners_and_neighbors(
            mesh_path)
        self._mesh_points = self._loader._mesh._parse_points(mesh_path)

        cell_faces = [[] for _ in range(n_cells)]
        n_faces_cell = pt.zeros(n_cells, dtype=pt.int32)
        n_face_labels = 0

        for i, owner in enumerate(owners):
            cell_faces[owner].append(faces[i][:n_points_faces[i]])
            n_face_labels += n_points_faces[i]
            n_faces_cell[owner] += 1
        for i, neigh in enumerate(neighbors):
            cell_faces[neigh].append(faces[i][:n_points_faces[i]])
            n_face_labels += n_points_faces[i]
            n_faces_cell[neigh] += 1

        topology_length = n_cells * 2 + \
            pt.sum(n_faces_cell).item() + n_face_labels
        self._topology = pt.zeros(topology_length, dtype=pt.int32)
        marker = 0
        for i, faces in enumerate(cell_faces):
            self._topology[marker] = 16
            self._topology[marker+1] = len(faces)
            marker += 2
            for j in range(len(faces)):
                n_labels = faces[j].size()[0]
                self._topology[marker] = n_labels
                self._topology[marker+1:marker+1+n_labels] = faces[j]
                marker += n_labels + 1
        return n_cells, self._mesh_points.size()[0], self._topology.size()[0]

    def _get_topology(self, job: int = 0):
        return self._topology

    def _get_vertices(self, mesh_path: str, job: int = 0):
        return self._mesh_points

    def _get_cell_centers(self, job: int = 0):
        return self._loader._mesh.get_cell_centers()

    def _get_cell_volumes(self, job: int = 0):
        return self._loader._mesh.get_cell_volumes()

    def _convert_fields(self, writer: HDF5Writer, fields: List[str],
                        times: List[str]):
        """Convert OpenFOAM fields to HDF5.

        :param writer: HDF5 writer
        :type writer: :class:`HDF5Writer`
        :param fields: list of fields to convert; if None, all available
            fields are converted
        :type fields: List[str]
        :param times: list of times to convert; if None, all available
            times are converted
        :type times: List[str]
        """
        field_info = self._gather_field_information(fields, times)
        for job, info in enumerate(field_info):
            print(
                f"Converting field {info[0]} at time {info[1]}, dimension {info[2]}")
            data = self._load_field(*info[:2], job=job)
            writer.write(info[0], info[2], data, info[1])

    def _gather_field_information(self, fields: List[str], times: List[str]
                                  ) -> List[tuple]:
        """Gather field information for parallel writing.

        - check if a field type is supported
        - determine data size

        :param fields: list of the field names
        :type fields: List[str]
        :param times: write times
        :type times: List[str]
        :return: list of all fields; each list element is a tuple
            with the entries (name, time, shape)
        :rtype: List[tuple]
        """
        def load_n_lines(file_name, n):
            lines = []
            with open(file_name, "rb") as file:
                for _ in range(n):
                    lines.append(file.readline())
            return lines

        field_info = []
        mesh_path = self._loader._case._path + "/" + POLYMESH_PATH
        n_cells = self._loader._mesh._get_n_cells(mesh_path)
        times_to_convert = self._loader.write_times
        if times is not None:
            times_to_convert = [t for t in times if t in times_to_convert]
        for time in times_to_convert:
            fields_to_convert = self._loader.field_names[time]
            if fields is not None:
                fields_to_convert = [
                    field for field in fields if field in fields_to_convert]
            for name in fields_to_convert:
                path = self._loader._case.build_file_path(name, time)
                header = load_n_lines(path, MAX_LINE_HEADER)
                field_type = self._loader._field_type(header)
                if field_type in FIELD_TYPE_DIMENSION.keys():
                    field_info.append(
                        (name, time, (n_cells, FIELD_TYPE_DIMENSION[field_type])))
        return field_info

    def _load_field(self, field: str, time: str, job: int = 0) -> pt.Tensor:
        return self._loader.load_snapshot(field, time)


class XDMFWriter(object):
    """Create XDMF file to open flowTorch HDF5 files in ParaView.

    Example

    >>> from flowtorch.data import XDMFWriter
    >>> writer = XDMFWriter("flowtorch.hdf5")
    >>> writer.create_xdmf("flowtorch.xdmf")

    """

    def __init__(self, file_path: str, hdf5_file: File):
        """Create XDMFWriter instance from path and file.

        :param file_path: path to HDF5 file
        :type file_path: str
        :param hdf5_file: HDF5 file instance
        :type hdf5_file: File
        """
        if "/" in file_path:
            self._path = file_path[:file_path.rfind("/")]
            self._hdf5_filename = file_path[file_path.rfind("/") + 1:]
        else:
            self._path = "."
            self._hdf5_filename = file_path
        self._file = hdf5_file
        self._n_cells = self._get_n_cells()

    @classmethod
    def from_filepath(cls, file_path: str):
        """Create XDMFWriter from a file path.

        :param file_path: path to HDF5 file
        :type file_path: str
        """
        return cls(
            file_path,
            File(file_path, mode="a")
        )

    def _get_n_cells(self) -> int:
        """Determine the number of mesh cells.

        The number of cells is currently determined based on the
        cell volume dataset.

        :return: number of mesh cells
        :rtype: int
        """
        n_cells = 0
        location = "/{:s}/{:s}".format(CONST_GROUP, VOLUMES_DS)
        if location in self._file:
            n_cells = self._file[location].shape[0]
        if n_cells == 0:
            print("XDMF warning: could not determine number of cells.")
        return n_cells

    def _add_grid(self, time: str, offset: str = "") -> str:
        """Create an XDMF grid element.

        :param time: snapshot write time
        :type time: str
        :param offset: number of spaces for indentation, defaults to ""
        :type offset: str, optional
        :return: XDMF grid element
        :rtype: str
        """
        grid = offset + "<Grid Name=\"Grid\" GridType=\"Uniform\">\n"
        grid += self._add_topology(offset + " "*4)
        grid += self._add_geometry(offset + " "*4)
        if time is not None:
            grid += offset + " "*4 + "<Time Value=\"{:s}\"/>\n".format(time)
            for key in self._find_attributes(time):
                grid += self._add_attribute(time, key, offset + " "*4)
        grid += offset + "</Grid>\n"
        return grid

    def _find_attributes(self, time: str) -> List[str]:
        """Find potential XDMF attributes in HDF5 file.

        A dataset is considered as XDMF attribute if the number
        of elements matches the number of cells in the mesh.

        :param time: snapshot write time
        :type time: str
        :return: a list of attributes to write in XDMF file
        :rtype: List[str]
        """
        location = "/{:s}/{:s}".format(VAR_GROUP, time)
        keys = self._file[location].keys()
        valid_attr = []
        for key in keys:
            loc = location + "/{:s}".format(key)
            first_dim = self._file[loc].shape[0]
            if first_dim == self._n_cells:
                valid_attr.append(key)
        return valid_attr

    def _add_topology(self, offset: str = "") -> str:
        """Create an XDMF topology element.

        :param offset: number of spaces for indentation, defaults to ""
        :type offset: str, optional
        :return: XDMF topology element
        :rtype: str
        """
        topology = offset + \
            "<Topology Name=\"{:s}\" TopologyType=\"Mixed\">\n".format(
                TOPOLOGY)
        location = self._hdf5_filename + \
            ":/{:s}/{:s}".format(CONST_GROUP, CONNECTIVITY_DS)
        topology += self._add_dataitem(location, offset + " "*4)
        topology += offset + "</Topology>\n"
        return topology

    def _add_geometry(self, offset: str = "") -> str:
        """Create XDMF geometry element.

        :param offset: number of spaces for indentation, defaults to ""
        :type offset: str, optional
        :return: XDMF geometry element
        :rtype: str
        """
        geometry = offset + "<Geometry GeometryType=\"XYZ\">\n"
        location = self._hdf5_filename + \
            ":/{:s}/{:s}".format(CONST_GROUP, VERTICES_DS)
        geometry += self._add_dataitem(location, offset + " "*4)
        geometry += offset + "</Geometry>\n"
        return geometry

    def _add_attribute(self, time: str, name: str, offset: str = "") -> str:
        """Create an XDMF attribute element.

        :param time: snapshot write time
        :type time: str
        :param name: name of the attribute
        :type name: str
        :param offset: number of spaces for indentation, defaults to ""
        :type offset: str, optional
        :return: XDMF attribute element
        :rtype: str
        """
        location = self._hdf5_filename + ":/{:s}/{:s}/{:s}".format(
            VAR_GROUP, time, name
        )
        shape = self._file[location.split(":")[-1]].shape
        tensor_type = xdmf_attributes[len(shape)]
        attribute = offset + "<Attribute Name=\"{:s}\" AttributeType=\"{:s}\" Center=\"Cell\">\n".format(
            name, tensor_type
        )
        attribute += self._add_dataitem(location, offset + " "*4)
        attribute += offset + "</Attribute>\n"
        return attribute

    def _add_dataitem(self, location: str,  offset: str = "") -> str:
        """Create an XDMF dataitem element.

        :param location: location of dataset in the HDF5 file
        :type location: str
        :param offset: number of spaces for indentation, defaults to ""
        :type offset: str, optional
        :return: XDMF dataitem element
        :rtype: str
        """
        path_in_file = location.split(":")[-1]
        shape = self._file[path_in_file].shape
        dimensions = " ".join(["{:d}".format(i) for i in shape])
        dtype, precision = xdmf_dtype_conversion[
            str(self._file[path_in_file].dtype)
        ]
        dataitem = offset + "<DataItem Dimensions=\"{:s}\" NumberType=\"{:s}\" Precision=\"{:d}\" Format=\"HDF\">\n".format(
            dimensions, dtype, precision
        )
        dataitem += offset + " "*4 + "{:s}\n".format(location)
        dataitem += offset + "</DataItem>\n"
        return dataitem

    def create_xdmf(self, filename: str = None) -> None:
        """Create XDMF wrapper to access flowTorch HDF5 file in ParaView.

        :param filename: name of the XDMF file, defaults to None
        :type filename: str, optional
        """
        xdmf_str = XDMF_HEADER
        times = list(self._file[VAR_GROUP].keys()
                     ) if VAR_GROUP in self._file else []
        if len(times) > 0:
            for time in times:
                xdmf_str += self._add_grid(time, " "*12)
        else:
            xdmf_str += self._add_grid(None, " "*12)
        xdmf_str = xdmf_str[:-1]  # remove last linebreak
        xdmf_str += XDMF_FOOTER

        if filename is None:
            if "." in self._hdf5_filename:
                filename = self._hdf5_filename[:self._hdf5_filename.rfind(
                    ".")] + ".xdmf"
            else:
                filename = self._hdf5_filename + ".xdmf"
        print(
            "Writing file {:s} as wrapper for {:s} at location {:s}".format(
                filename, self._hdf5_filename, self._path
            )
        )
        with open(self._path + "/" + filename, "w") as file:
            file.write(xdmf_str)


def copy_hdf5_mesh(path: str, from_file: str, to_file: str) -> None:
    """Create a copy of a flowTorch hdf5 file containing only the mesh.

    Sometimes, it is helpful to create a new copy of an existing hdf5 file
    that contains only the mesh, e.g., to create a separate file for
    POD or DMD modes.

    :param path: location of the flowtorch hdf5 file excluding the filename
    :type path: str
    :param from_file: name of the file from which to copy the mesh
    :type from_file: str
    :param to_file: name of the file to which to copy the mesh
    :type to_file: str
    """
    path = check_and_standardize_path(path)
    from_file_path = join(path, from_file)
    loader = HDF5Dataloader(from_file_path)
    to_file_path = join(path, to_file)
    print(f"Copying mesh from file {from_file_path} to {to_file_path}")
    writer = HDF5Writer(to_file_path)
    datasets = {
        VERTICES_DS: loader.edge_vertices,
        CENTERS_DS: loader.vertices,
        VOLUMES_DS: loader.weights,
        CONNECTIVITY_DS: loader.connectivity
    }
    for ds_key in datasets.keys():
        data = datasets[ds_key]
        writer.write(ds_key, data.shape, data, None, data.dtype)
    xdmf_name = f"{to_file.split('.')[0]}.xdmf"
    xdmf = XDMFWriter.from_filepath(to_file_path)
    xdmf.create_xdmf(xdmf_name)
