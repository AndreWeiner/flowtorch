"""Dataloader and accompanying tools to work with CSV files.

A lot of scientific data is exchanged as comma separated value (CSV) files.
While there are many Python packages available to read such data, one has
to understand how the data is organized in the CSV file before being able
to use the readers properly. Moreover, time series data sometimes come as
individual files in a single folder or as time folders with the respective
snapshot data inside that folder. This subpackages simplifies access to
common CSV-based time series data by trying to figure out appropriate reader
settings automatically.
"""

# standard library 
from os.path import join
from os import sep
from glob import glob
import re
from typing import List, Dict, Union
# third party packages
import torch as pt
from pandas import read_csv, DataFrame
# flowtorch packages
from flowtorch import DEFAULT_DTYPE
from .dataloader import Dataloader
from .utils import check_and_standardize_path, check_list_or_str

VERTEX_KEY = "vertices"
WEIGHT_KEY = "weights"
FIELD_KEY = "fields"

DAVIS_KEYS = {
    VERTEX_KEY: ["x", "y"],
    WEIGHT_KEY: "isValid",
}

FOAM_SURFACE_KEYS = {
    VERTEX_KEY: ["x", "y", "z"],
    WEIGHT_KEY: ["area_x", "area_y", "area_z"]
}

PANDAS_SKIPROWS = "skiprows"
PANDAS_SEP = "sep"
PANDAS_HEADER = "header"
PANDAS_NAMES = "names"
PANDAS_ARGS = (PANDAS_SKIPROWS, PANDAS_SEP, PANDAS_HEADER, PANDAS_NAMES)


def _parse_davis_header(header: str) -> List[str]:
    """Find column names in DaVis header line.

    :param header: header line containing column names
    :type header: str
    :return: list of column names
    :rtype: List[str]
    """
    return re.findall('"([^"]*)"', header)


def _parse_foam_surface_header(header: str) -> List[str]:
    """Find column names in OpenFOAM *surfaces* output file.

    :param header: header line containing column names
    :type header: str
    :return: list of column names
    :rtype: List[str]
    """
    return list(filter(None, header.replace("#", "").strip().split(" ")))


class CSVDataloader(Dataloader):
    """Load CSV files from different sources.

    This class allows to load generic CSV files based on Pandas's `load_csv` function.
    Multiple specific formats are supported via class methods.

    Examples

    >>> from flowtorch import DATASETS
    >>> from flowtorch.data import CSVDataloader
    >>> davis_data = DATASETS["csv_aoa8_beta0_xc100_stereopiv"]
    >>> loader = CSVDataloader.from_davis(davis_data, "B")
    >>> times = loader.write_times
    >>> times[:5]
    ['00001', '00002', '00003', '00004', '00005']
    >>> loader.field_names
    {'00001': ['Vx', 'Vy', 'Vz']}
    >>> Vx, Vy, Vz = loader.load_snapshot(['Vx', 'Vy', 'Vz'], times[:5])
    >>> Vx.shape
    torch.Size([3741, 5])

    >>> foam_data = DATASETS["csv_naca0012_alpha4_surface"]
    >>> loader = CSVDataloader.from_foam_surface(foam_data, "total(p)_coeff_airfoil.raw")
    >>> times = loader.write_times
    >>> times[:5]
    ['0.001', '0.002', '0.003', '0.004', '0.005']
    >>> loader.field_names
    {'0.001': ['total(p)_coeff']}
    >>> snapshots = loader.load_snapshot("total(p)_coeff", times[:10])
    >>> snapshots.shape
    torch.Size([28892, 10])
    >>> vertices = loader.vertices
    >>> vertices.shape
    torch.Size([28892, 3])
    >>> vertices[0, :]
    tensor([0.0000e+00, 0.0000e+00, 4.1706e-18])

    """

    def __init__(self, path: str, prefix: str, suffix: str, read_options: dict,
                 time_folders: bool, dtype: str = DEFAULT_DTYPE):
        """Create a CSVDataloader instance from files or folders.

        :param path: path to folder with CSV files or folder with time-subfolders
            containing individual CSV files
        :type path: str
        :param prefix: part of the file name before the time value; used to
            find CSV files
        :type prefix: str
        :param suffix: part of the file name after the time value
        :type suffix: str
        :param read_options: dictionary holding options for `pandas.read_csv`
        :type read_options: dict
        :param time_folders: `True` if CSV files are located in individual folders
            for each write time
        :type time_folders: bool
        :param dtype: tensor type, defaults to single precision `torch.float32`
        :type dtype: str

        """
        self._path = check_and_standardize_path(path)
        self._prefix = prefix
        self._suffix = suffix
        self._read_options = read_options
        self._time_folders = time_folders
        self._dtype = dtype
        self._write_times = self._determine_write_times()

    @classmethod
    def from_davis(cls, path: str, prefix: str = "", suffix: str = ".dat", dtype: str = DEFAULT_DTYPE):
        """Create CSVDataloader instance for DaVis_ output files.

        .. _DaVis: https://www.lavision.de/en/products/davis-software/

        :param path: path to location of time series data
        :type path: str
        :param prefix: part of the file name before the time/snapshot number; e.g.,
            if the file name *B00001.dat*, the prefix is *B*; defaults to empty string
        :type prefix: str
        :param suffix: part of the file name of the time/snapshot number; e.g.,
            if the file name is *B00001.dat*, the suffix is *.dat*; defaults to *.dat*
        :type suffix: str
        :param dtype: floating point precision; defaults to `pt.float32` (single precision)
        :type dtype: str

        """
        file_name = sorted(glob(join(path, f"{prefix}*{suffix}")))[0]
        with open(file_name, "r") as davis_file:
            content = davis_file.readlines()
            for line in content:
                if line.startswith("VARIABLES"):
                    column_names = _parse_davis_header(line)
                    break
        if not set(DAVIS_KEYS[VERTEX_KEY]).issubset(set(column_names)):
            print("Warning: DaVis files do not contain vertices")
        if not set([DAVIS_KEYS[WEIGHT_KEY]]).issubset(set(column_names)):
            print("Warning: DaVis files do not contain isValid mask")
        read_options = {
            PANDAS_SKIPROWS: [0, 1, 2],
            PANDAS_HEADER: None,
            PANDAS_SEP: " ",
            PANDAS_NAMES: column_names,
            VERTEX_KEY: DAVIS_KEYS[VERTEX_KEY],
            WEIGHT_KEY: DAVIS_KEYS[WEIGHT_KEY],
            FIELD_KEY: [col for col in column_names if col not in
                DAVIS_KEYS[VERTEX_KEY] + [DAVIS_KEYS[WEIGHT_KEY]]]
        }
        return cls(path, prefix, suffix, read_options, False, dtype)

    @classmethod
    def from_foam_surface(cls, path: str, file_name: str, header: int=1, dtype: str = DEFAULT_DTYPE):
        """Create CSVDataloader instance to load OpenFOAM surface sample data.

        The class method simplifies to load data generated by OpenFOAM's
        *sampling* function object if the type is set to *surfaces* and the
        *surfaceFormat* is set to *raw*. The time series data are stored in
        individual time folders. The file name remains the same.

        :param path: path to location of time folders
        :type path: str
        :param file_name: file name of individual CSV files, e.g.,
            *p_airfoil.raw*
        :type file_name: str
        :param header: line number in which to find the column names (starting from 0);
            defaults to 1
        :type header: int, optional
        :param dtype: floating point precision; defaults to `pt.float32` (single precision)
        :type dtype: str

        """
        folders = glob(join(path, "*"))
        times = sorted(
            [folder.split(sep)[-1] for folder in folders], key=float
        )
        filepath = join(path, times[0], file_name)
        with open(filepath, "r") as surface:
            content = surface.readlines()
            column_names = _parse_foam_surface_header(content[header])

        weight = FOAM_SURFACE_KEYS[WEIGHT_KEY] if all([key in column_names for key in FOAM_SURFACE_KEYS[WEIGHT_KEY]]) else None
        read_options = {
            PANDAS_SKIPROWS: list(range(header+1)),
            PANDAS_HEADER: None,
            PANDAS_SEP: " ",
            PANDAS_NAMES: column_names,
            VERTEX_KEY: FOAM_SURFACE_KEYS[VERTEX_KEY],
            WEIGHT_KEY: weight,
            FIELD_KEY: [col for col in column_names if col not in
                FOAM_SURFACE_KEYS[VERTEX_KEY] + FOAM_SURFACE_KEYS[WEIGHT_KEY]]
        }
        return cls(path, file_name, "", read_options, True, dtype)

    def _build_file_path(self, time: str) -> str:
        """Build the file path for a given time instance.

        :param time: snapshot time
        :type time: str
        :return: path to snapshot
        :rtype: str

        """
        if self._time_folders:
            return join(self._path, time, f"{self._prefix}{self._suffix}")
        else:
            return join(self._path, f"{self._prefix}{time}{self._suffix}")

    def _load_csv(self, time: str) -> DataFrame:
        """Load a single CSV snapshot.

        :param time: snapshot time
        :type time: str
        :return: content of the CSV file stored in a DataFrame
        :rtype: DataFrame

        """
        file_path = self._build_file_path(time)
        options = {key: self._read_options[key] for key in PANDAS_ARGS}
        return read_csv(file_path, **options)

    def _determine_write_times(self) -> List[str]:
        """Find available snapshot times.

        Depending on the storage format, the write times are
        determined from the folder names or the file names.

        :return: list of snapshot/write times
        :rtype: List[str]
        """
        if self._time_folders:
            folders = glob(f"{self._path}/*")
            return sorted(
                [folder.split(sep)[-1] for folder in folders], key=float
            )
        else:
            files = glob(self._build_file_path("*"))
            return sorted(
                [f.split(sep)[-1][len(self._prefix):-len(self._suffix)]
                 for f in files], key=float
            )

    def load_snapshot(self,
                      field_name: Union[List[str], str],
                      time: Union[List[str], str]) -> Union[List[pt.Tensor], pt.Tensor]:
        check_list_or_str(field_name, "field_name")
        check_list_or_str(time, "time")
        # load multiple fields
        if isinstance(field_name, list):
            if isinstance(time, list):
                snapshots = [self._load_csv(t) for t in time]
                return [
                    pt.stack(
                        [pt.tensor(snapshot[field].values, dtype=self._dtype)
                         for snapshot in snapshots], dim=-1
                    ) for field in field_name
                ]
            else:
                snapshot = self._load_csv(time)
                return [
                    pt.tensor(snapshot[field].values, dtype=self._dtype)
                    for field in field_name
                ]
        # load single field
        else:
            if isinstance(time, list):
                snapshots = [self._load_csv(t) for t in time]
                return pt.stack(
                    [pt.tensor(snapshot[field_name].values, dtype=self._dtype)
                     for snapshot in snapshots], dim=-1
                )
            else:
                snapshot = self._load_csv(time)
                return pt.tensor(
                    snapshot[field_name].values, dtype=self._dtype
                )

    @property
    def write_times(self) -> List[str]:
        return self._write_times

    @property
    def field_names(self) -> Dict[str, List[str]]:
        return dict(
            {self.write_times[0]: self._read_options[FIELD_KEY]}
        )

    @property
    def vertices(self) -> pt.Tensor:
        snapshot = self._load_csv(self._write_times[0])
        return pt.tensor(
            snapshot[self._read_options[VERTEX_KEY]].values, dtype=self._dtype
        )

    @property
    def weights(self) -> pt.Tensor:
        weight_key = self._read_options[WEIGHT_KEY]
        snapshot = self._load_csv(self._write_times[0])
        if weight_key is not None:
            if isinstance(weight_key, list):
                return pt.tensor(
                    snapshot[weight_key].values, dtype=self._dtype
                ).norm(dim=1)
            else:
                return pt.tensor(
                    snapshot[weight_key].values, dtype=self._dtype
                )
        else:
            shape = snapshot[self._read_options[FIELD_KEY]].values.shape
            return pt.ones(shape[0], dtype=self._dtype)
