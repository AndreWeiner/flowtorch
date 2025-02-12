"""Read Tecplot data via the ParaView module.
"""

# standard library packages
from os.path import join
from os import sep
from glob import glob
from typing import Union, List, Dict
# third party packages
import torch as pt
from paraview import servermanager as sm
from paraview.vtk.numpy_interface import dataset_adapter as dsa
from paraview.simple import VisItTecplotBinaryReader
# flowtorch packages
from flowtorch import DEFAULT_DTYPE
from .dataloader import Dataloader
from .utils import check_and_standardize_path, check_list_or_str


class TecplotDataloader(Dataloader):
    """Dataloader for Tecplot binary format.

    The dataloader wraps around `VisItTecplotBinaryReader` available
    in the ParaView Python module. One level of blocks/zones is expected
    under the root block/zone

    Examples

    >>> from flowtorch import DATASETS
    >>> from flowtorch.data import TecplotDataloader
    >>> path = DATASETS["plt_naca2409_surface"]
    >>> loader = TecplotDataloader.from_tau(path, "alfa16.surface.pval.unsteady_")
    >>> loader.zone_names
    ["ls", "te", "us"]
    >>> loader.zone
    "le"
    >>> loader.zone = "us"
    >>> times = loader.write_times
    >>> density = loader.load_snapshot("density", times)
    >>> density.shape
    torch.Size([300, 3])

    """

    def __init__(self, path: str, file_names: Dict[str, str],
                 reader: VisItTecplotBinaryReader, dtype: str = DEFAULT_DTYPE):
        """Default constructor function.

        :param path: path to snapshot location
        :type path: str
        :param file_names: names of available snapshots
        :type file_names: Dict[str, str]
        :param reader: ParaView reader for Tecplot binary format
        :type reader: VisItTecplotBinaryReader
        :param dtype: tensor data type, defaults to DEFAULT_DTYPE
        :type dtype: str, optional
        """
        self._path = path
        self._file_names = file_names
        self._reader = reader
        self._dtype = dtype
        self._zone_names = None
        self._zone = self.zone_names[0]

    @classmethod
    def from_tau(cls, path: str, base_name: str = "", suffix: str = ".plt", dtype: str = DEFAULT_DTYPE):
        """Construct TecplotDataloader from TAU snapshots.

        :param path: path to snapshot location
        :type path: str
        :param base_name: common basename of all snapshots, defaults to ""
        :type base_name: str, optional
        :param suffix: snapshot file suffix, defaults to ".plt"
        :type suffix: str, optional
        :param dtype: tensor data type, defaults to DEFAULT_DTYPE
        :type dtype: str, optional
        :raises FileNotFoundError: if no snapshots are found
        :return: Tecplot dataloader object
        :rtype: TecplotDataloader
        """
        path = check_and_standardize_path(path)
        file_paths = glob(join(path,  f"{base_name}i=*t=*"))
        file_names = [f.split(sep)[-1] for f in file_paths]
        write_times = [name.split("t=")[-1].split(suffix)[0]
                       for name in file_names]
        sorted_names = sorted(zip(write_times, file_names),
                              key=lambda tup: float(tup[0]))
        file_names = {time: name for time, name in sorted_names}
        if len(file_names.keys()) < 1:
            raise FileNotFoundError(
                f"Could not find solution files in {self._sol_path}/")
        return cls(path, file_names, VisItTecplotBinaryReader, dtype)

    def _assemble_file_path(self, time: str) -> str:
        """Assemble a path to a single snapshot.

        :param time: snapshot write time
        :type time: str
        :return: snapshot path
        :rtype: str
        """
        return join(self._path, self._file_names[time])

    def _parse_block_name(self, meta_data: str) -> str:
        """Extract block name from a reader's metadata.

        :param meta_data: output of `GetMetaData()` as string
        :type meta_data: str
        :return: block name
        :rtype: str
        """
        lines = meta_data.split("\n")
        name = None
        for line in lines:
            if "NAME" in line:
                name = line.split(":")[-1].strip()
        return name

    def _create_tecplot_reader(self, time: str) -> VisItTecplotBinaryReader:
        """Create instance of `VisItTecplotBinaryReader`.

        :param time: snapshot write time
        :type time: str
        :return: reader for Tecplot binary data
        :rtype: VisItTecplotBinaryReader
        """
        return self._reader(
            registrationName=self._file_names[time],
            FileName=[self._assemble_file_path(time)]
        )

    def _load_single_snapshot(self, field_name: str, time: str) -> pt.Tensor:
        """Load a single snapshot of a single field.

        :param field_name: name of field to load
        :type field_name: str
        :param time: snapshot write time
        :type time: str
        :return: snapshot of the requested field
        :rtype: pt.Tensor
        """
        reader = self._create_tecplot_reader(time)
        field_names = self.field_names[self.write_times[0]]
        reader.PointArrayStatus = field_names
        reader = sm.Fetch(reader)
        wrapper = dsa.WrapDataObject(reader.GetBlock(0).GetBlock(
            self.zone_names.index(self.zone)
        ))
        return pt.from_numpy(wrapper.PointData[field_names.index(field_name)])

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
            [self._load_single_snapshot(field_name, time) for time in times],
            dim=-1
        )

    def load_snapshot(self,
                      field_name: Union[List[str], str],
                      time: Union[List[str], str]) -> Union[List[pt.Tensor], pt.Tensor]:
        """Load snapshots of single or multiple fields and write times.

        :param field_name: single field name or list of field names
        :type field_name: Union[List[str], str]
        :param time: single write time of list of write times
        :type time: Union[List[str], str]
        :return: snapshot(s) of one or multiple fields
        :rtype: Union[List[pt.Tensor], pt.Tensor]
        """
        check_list_or_str(field_name, "field_name")
        check_list_or_str(time, "time")
        # load multiple fields
        if isinstance(field_name, list):
            if isinstance(time, list):
                return [self._load_multiple_snapshots(name, time) for name in field_name]
            else:
                return [self._load_single_snapshot(name, time) for name in field_name]
        # load a single field
        else:
            if isinstance(time, list):
                return self._load_multiple_snapshots(field_name, time)
            else:
                return self._load_single_snapshot(field_name, time)

    @property
    def zone_names(self) -> List[str]:
        """Names of available blocks/zones.

        :return: block/zone names
        :rtype: List[str]
        """
        if self._zone_names is None:
            reader = sm.Fetch(self._create_tecplot_reader(self.write_times[0]))
            root_block = reader.GetBlock(0)
            self._zone_names = [
                self._parse_block_name(
                    str(root_block.GetMetaData(i))
                ) for i in range(root_block.GetNumberOfBlocks())
            ]
        return self._zone_names

    @property
    def zone(self) -> str:
        """Currently selected block/zone.

        :return: block/zone name
        :rtype: str
        """
        return self._zone

    @zone.setter
    def zone(self, value: str):
        """Select active block/zone.

        The selected block remains unchanged if an invalid
        block name is passed

        :param value: name of block to select
        :type value: str
        """
        if value in self.zone_names:
            self._zone = value
        else:
            print(f"{value} not found. Available zones are:")
            print(self.zone_names)

    @property
    def write_times(self) -> List[str]:
        """Available snapshot write times

        :return: list of available write times
        :rtype: List[str]
        """
        return list(self._file_names.keys())

    @property
    def field_names(self) -> Dict[str, List[str]]:
        """List of available field names.

        The field names are only determined once for the
        first available snapshot time.

        :return: available fields at first write time
        :rtype: Dict[str, List[str]]
        """
        time = self.write_times[0]
        reader = self._create_tecplot_reader(time)
        return {
            time: reader.GetProperty("PointArrayInfo")[::2]
        }

    @property
    def vertices(self) -> pt.Tensor:
        """Points in which field values are defined.

        :return: list of points
        :rtype: pt.Tensor
        """
        reader = sm.Fetch(self._create_tecplot_reader(self.write_times[0]))
        wrapper = dsa.WrapDataObject(reader.GetBlock(0).GetBlock(
            self.zone_names.index(self.zone)
        ))
        return pt.from_numpy(wrapper.Points)

    @property
    def weights(self) -> pt.Tensor:
        """Weight for POD/DMD analysis.

        This function returns currently a list of ones, since
        cell areas/volumes are not accessible via the reader.

        :return: list of ones
        :rtype: pt.Tensor
        """
        reader = sm.Fetch(self._create_tecplot_reader(self.write_times[0]))
        wrapper = dsa.WrapDataObject(reader.GetBlock(0).GetBlock(
            self.zone_names.index(self.zone)
        ))
        # volume or area are not contained in the file; therefore, a tensor
        # of ones is returned for now
        n_points = wrapper.Points.shape[0]
        return pt.ones(n_points, dtype=self._dtype)
