"""Module to read and write the internal flowTorch data format.

The :class:`HDF5Writer` class allows to write field and mesh data
into an HDF5 file. It also creates an XDMF accessor file for postprocessing
in ParaView. The XDMF file creation is implemented in the :class:`XDMFWriter`
class. The :class:`HDF5Dataloader` class is the complementary writer for the
flowTorch data format. Moreover, the :class:`FOAM2HDF5` class allows conversion
of reconstructed OpenFOAM simulation cases into the HDF5-based flowTorch format.
"""

# standard library packages
from functools import partial
import logging
from os.path import isfile, join
from typing import List, Dict, Union, cast

# third party packages
import torch as pt
from h5py import File

# flowtorch packages
from flowtorch import DEFAULT_DTYPE
from .dataloader import Dataloader, _preallocate_time_series
from ._foam_stream import foam_open, header as foam_header
from .foam_dataloader import (
    FOAMDataloader,
    POLYMESH_PATH,
    FIELD_TYPE_DIMENSION,
)
from .utils import check_list_or_str, check_and_standardize_path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONST_GROUP = "constant"
VAR_GROUP = "variable"
VERTICES_DS = "vertices"
CONNECTIVITY_DS = "connectivity"
CENTERS_DS = "centers"
VOLUMES_DS = "volumes"
XDMF_HEADER = """<Xdmf Version="3.0">
    <Domain>
        <Grid Name="flowTorch" GridType="Collection" CollectionType="Temporal">
"""
XDMF_FOOTER = """
        </Grid>
    </Domain>
</Xdmf>
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
    "int64": "i8",
}

xdmf_attributes = {1: "Scalar", 2: "Vector", 3: "Vector", 6: "Tensor6", 9: "Tensor"}

xdmf_dtype_conversion = {
    "f4": ("Float", 4),
    "f8": ("Float", 8),
    "i4": ("Int", 4),
    "i8": ("Int", 8),
    "float32": ("Float", 4),
    "float64": ("Float", 8),
    "int32": ("Int", 4),
    "int64": ("Int", 8),
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

    def __init__(
        self,
        file_path: str,
        dtype: pt.dtype = DEFAULT_DTYPE,
        *,
        zones=None,
        geometry_file=None,
    ):
        """Create HDF5Dataloader instance.

        :param file_path: path to HDF5 file
        :type file_path: str
        :param dtype: tensor type of floating point values, defaults to DEFAULT_DTYPE
        :type dtype: pt.dtype, optional
        """
        from pathlib import Path

        path = Path(file_path).resolve()
        if path.is_dir():
            path = path / "fields.h5"
        if path.suffix.lower() in (".xdmf", ".xmf"):
            import xml.etree.ElementTree as ET

            candidates = set()
            for item in ET.parse(path).iter("DataItem"):
                if item.attrib.get("Format") == "HDF" and item.text:
                    reference = path.parent / item.text.strip().rsplit(":", 1)[0]
                    with File(reference, "r") as handle:
                        if "frames" in handle:
                            candidates.add(reference.resolve())
            if len(candidates) != 1:
                raise ValueError(
                    "Descriptor must reference exactly one versioned field container"
                )
            path = candidates.pop()
        file_path = str(path)
        if not isfile(file_path):
            raise FileNotFoundError(f"Could not find file {file_path}")
        self._file_path = file_path
        self._dtype = dtype
        self._series = None
        self._file = File(self._file_path, mode="r")
        if "schema" in self._file.attrs:
            self._file.close()
            from ._hdf5_schema import SeriesReader

            self._series = SeriesReader(file_path, dtype, zones, geometry_file)
            self._file = self._series.file
        else:
            try:
                if zones is not None or geometry_file is not None:
                    raise ValueError(
                        "Zone selection and shared geometry require the versioned schema"
                    )
                from ._hdf5_compression import prepare_read, prepare_dataset_read

                prepare_read(self._file)
                if CONST_GROUP in self._file:
                    for dataset in self._file[CONST_GROUP].values():
                        prepare_dataset_read(dataset)
            except Exception:
                self._file.close()
                raise

    def _load_single_snapshot(self, field_name: str, time: str) -> pt.Tensor:
        """Load one field at one time."""
        return self._load_single_snapshot_slice(field_name, time, slice(None))

    def _load_single_snapshot_slice(
        self, field_name: str, time: str, spatial_slice: slice
    ) -> pt.Tensor:
        """Load a first-axis slice directly from an HDF5 dataset."""
        if self._series is not None:
            return self._series.read(field_name, time, spatial_slice)
        from ._hdf5_compression import prepare_dataset_read

        dataset = self._file[f"{VAR_GROUP}/{time}/{field_name}"]
        prepare_dataset_read(dataset)
        values = pt.tensor(
            dataset[spatial_slice],
            dtype=self._dtype,
        )
        return (
            values.squeeze(-1) if values.ndim > 1 and values.shape[-1] == 1 else values
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
                    _preallocate_time_series(
                        partial(self._load_single_snapshot, field),
                        time,
                    )
                    for field in field_name
                ]
            else:
                return [self._load_single_snapshot(field, time) for field in field_name]
        # load single field
        else:
            if isinstance(time, list):
                return _preallocate_time_series(
                    partial(self._load_single_snapshot, field_name),
                    time,
                )
            else:
                return self._load_single_snapshot(field_name, time)

    def load_snapshot_slice(
        self,
        field_name: Union[List[str], str],
        time: Union[List[str], str],
        spatial_slice: slice,
    ) -> Union[List[pt.Tensor], pt.Tensor]:
        """Load one or more fields using direct first-axis HDF5 slicing."""
        check_list_or_str(field_name, "field_name")
        check_list_or_str(time, "time")
        fields = field_name if isinstance(field_name, list) else [field_name]
        times = time if isinstance(time, list) else [time]
        loaded = [
            _preallocate_time_series(
                partial(
                    self._load_single_snapshot_slice,
                    name,
                    spatial_slice=spatial_slice,
                ),
                times,
            )
            for name in fields
        ]
        if not isinstance(time, list):
            loaded = [value[..., 0] for value in loaded]
        return loaded if isinstance(field_name, list) else loaded[0]

    def snapshot_shape(self, field_name: str, time: str) -> tuple[int, ...]:
        """Return a variable dataset shape without loading it."""
        if self._series is not None:
            return self._series.shape(field_name, time)
        shape = tuple(self._file[f"{VAR_GROUP}/{time}/{field_name}"].shape)
        return shape[:-1] if len(shape) > 1 and shape[-1] == 1 else shape

    @property
    def write_times(self) -> List[str]:
        if self._series is not None:
            return list(self._series.frames)
        return sorted(self._file[VAR_GROUP].keys(), key=float)

    @property
    def field_names(self) -> Dict[str, List[str]]:
        if self._series is not None:
            return {
                time: list(self._series.datasets[time, self._series.keys[0]])
                for time in self.write_times
            }
        times = self.write_times
        field_names: Dict[str, List[str]] = {}
        for time in times:
            field_names[time] = list(self._file[f"{VAR_GROUP}/{time}"].keys())
        return field_names

    @property
    def vertices(self) -> pt.Tensor:
        if self._series is not None:
            return pt.tensor(
                self._series.geometry_values("locations"), dtype=pt.float64
            )
        return pt.tensor(
            self._file[f"{CONST_GROUP}/{CENTERS_DS}"][:], dtype=self._dtype
        )

    @property
    def weights(self) -> pt.Tensor:
        if self._series is not None:
            if len(set(self.zone_dimensions.values())) != 1:
                raise ValueError(
                    "Selected zones mix geometric dimensions. Select compatible zones "
                    "or use weights_by_zone to define explicit combined analysis weights."
                )
            return pt.tensor(self._series.geometry_values("weights"), dtype=self._dtype)
        return pt.tensor(
            self._file[f"{CONST_GROUP}/{VOLUMES_DS}"][:], dtype=self._dtype
        )

    @property
    def zone_dimensions(self) -> Dict[str, int]:
        """Physical mesh dimension of each selected zone (1, 2 or 3)."""
        if self._series is None:
            return {"mesh": 3}
        return {
            name: int(self._series.geometry["zones"][key].attrs["dimension"])
            for name, key in zip(self._series.names, self._series.keys)
        }

    @property
    def weights_by_zone(self) -> Dict[str, pt.Tensor]:
        """Raw measures per zone, without combining different physical dimensions."""
        if self._series is None:
            return {"mesh": self.weights}
        return {
            name: pt.tensor(
                self._series.geometry["zones"][key]["weights"][:], dtype=self._dtype
            )
            for name, key in zip(self._series.names, self._series.keys)
        }

    @property
    def connectivity(self) -> pt.Tensor:
        if self._series is not None:
            if len(self._series.keys) != 1:
                raise ValueError("Use geometry(zone) for multi-zone topology")
            return pt.tensor(self._series.geometry_values("topology"), dtype=pt.int64)
        return pt.tensor(
            self._file[f"{CONST_GROUP}/{CONNECTIVITY_DS}"][:], dtype=pt.int64
        )

    @property
    def edge_vertices(self) -> pt.Tensor:
        if self._series is not None:
            return pt.tensor(self._series.geometry_values("points"), dtype=pt.float64)
        return pt.tensor(
            self._file[f"{CONST_GROUP}/{VERTICES_DS}"][:], dtype=self._dtype
        )

    @property
    def supports_direct_slicing(self) -> bool:
        return True

    @property
    def zone_names(self):
        return self._series.names if self._series is not None else ["mesh"]

    @property
    def geometry_id(self):
        return self._series.geometry_id if self._series is not None else None

    @property
    def association(self):
        return self._series.association if self._series is not None else "cell"

    @property
    def locations(self):
        return self.vertices

    def geometry(self, zone=None):
        """Read one zone's explicit geometry from a versioned file."""
        from ._hdf5_schema import read_geometry

        if self._series is None:
            raise ValueError("Explicit zone geometry requires the versioned schema")
        zone = (
            self.zone_names[0] if zone is None and len(self.zone_names) == 1 else zone
        )
        if zone not in self.zone_names:
            raise ValueError("Select one of the loaded geometry zones")
        key = self._series.keys[self.zone_names.index(zone)]
        return read_geometry(self._series.geometry["zones"][key])

    def close(self):
        if self._series is not None:
            self._series.close()
        else:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


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

    def __init__(
        self,
        file: str,
        *,
        geometry_file=None,
        field_dtype=DEFAULT_DTYPE,
        spatial_chunk=65536,
        compression=None,
        compression_level=None,
        shuffle=None,
        geometry_compression=None,
        geometry_compression_level=None,
        geometry_shuffle=False,
        overwrite=False,
        append=False,
    ):
        """Write flowTorch HDF5 files.

        :param file: path and file name to HDF5 file.
        :type file: str
        :param compression: field codec: None (default), "gzip" or "zstd".
        :param compression_level: codec level; defaults to gzip 4 or Zstd 3.
        :param shuffle: reversible byte shuffle for fields. None disables it
            for new output and inherits the existing setting on append.
        :param geometry_compression: codec for legacy constant datasets.
        :param geometry_compression_level: level for legacy constant datasets.
        :param geometry_shuffle: byte shuffle for legacy constant datasets.
        :param spatial_chunk: maximum rows per compressed/chunked dataset chunk.
        :param geometry_file: existing read-only geometry for split output.
            Configure its compression when creating it with write_geometry.

        :param append: add new snapshots to an existing complete split dataset.
            Omitted compression settings inherit stored filters. Duplicate times,
            precision/schema changes and overwrite=True are rejected. Geometry
            can be discovered from the existing file when geometry_file is omitted.

        Zstd requires the optional compression extra. All filters are lossless;
        storage precision is selected separately. Defaults remain uncompressed.
        """
        from ._hdf5_compression import compression_settings, record_compression

        if append and overwrite:
            raise ValueError("append and overwrite are mutually exclusive")
        if append and geometry_file is None:
            from pathlib import Path
            from ._hdf5_schema import SCHEMA

            with File(file, "r") as existing:
                if existing.attrs.get("schema") != SCHEMA:
                    raise ValueError(
                        "Explicit append requires the split-geometry schema"
                    )
                geometry_file = (
                    Path(file).resolve().parent / existing.attrs["geometry_file"]
                )
        self._geometry_compression = compression_settings(
            geometry_compression, geometry_compression_level, geometry_shuffle
        )
        if spatial_chunk < 1:
            raise ValueError("spatial_chunk must be positive")
        self._spatial_chunk = spatial_chunk
        self._file_path = file
        self._series = None
        if geometry_file is None:
            self._compression = compression_settings(
                compression, compression_level, False if shuffle is None else shuffle
            )
            self._file = File(file, mode="a")
            # Preserve provenance for existing compressed datasets in append mode.
            if "compression_settings" not in self._file.attrs:
                record_compression(
                    self._file,
                    fields=self._compression,
                    geometry=self._geometry_compression,
                )
        else:
            from ._hdf5_schema import FieldFile

            if (
                geometry_compression is not None
                or geometry_shuffle
                or geometry_compression_level is not None
            ):
                raise ValueError(
                    "Configure new geometry with HDF5Writer.write_geometry; existing geometry is read-only"
                )
            self._series = FieldFile(
                file,
                geometry_file,
                field_dtype,
                spatial_chunk,
                compression,
                overwrite,
                compression_level=compression_level,
                shuffle=shuffle,
                append=append,
            )
            self._compression = self._series.compression
            self._file = self._series.file

    def write(
        self,
        name: str,
        size: tuple,
        data: pt.Tensor = None,
        time: str = None,
        dtype: pt.dtype = DEFAULT_DTYPE,
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
        :type dtype: pt.dtype, optional
        """
        if self._series is not None:
            raise ValueError("Use allocate_field/write_slice for versioned output")
        if time is not None:
            ds_name = VAR_GROUP + "/{:s}/{:s}".format(time, name)
        else:
            ds_name = CONST_GROUP + "/{:s}".format(name)

        if dtype in dtype_conversion.keys():
            settings = (
                self._compression if time is not None else self._geometry_compression
            )
            options = settings.dataset_options
            if settings.enabled:
                if not size or any(n < 1 for n in size):
                    raise ValueError("Compressed datasets require positive dimensions")
                options["chunks"] = (min(size[0], self._spatial_chunk), *size[1:])
            if ds_name in self._file:
                del self._file[ds_name]
            ds = self._file.create_dataset(
                ds_name, size, dtype=dtype_conversion[dtype], **options
            )
            # Per-dataset provenance also supports mixed settings in append mode.
            import json

            ds.attrs["compression_settings"] = json.dumps(settings.metadata)
            if data is not None:
                shape_diff = len(size) - len(data.size())
                if shape_diff == 1:
                    data = data.unsqueeze(-1)
                elif shape_diff == -1:
                    data = data.squeeze()
                ds[:] = data.numpy()
        else:
            logger.warning(
                "Invalid data type {:s} for field {:s}. Skipping field.".format(
                    str(dtype), name
                )
            )

    def write_xdmf(self, filename=None):
        """Write XDMF wrapper to access flowTorch HDF5 files in ParaView."""
        if self._series is not None:
            from pathlib import Path

            self._series.write_xdmf(
                filename or Path(self._file_path).with_suffix(".xdmf")
            )
        else:
            writer = XDMFWriter(self._file_path, self._file)
            writer.create_xdmf(filename or "flowtorch.xdmf")

    @staticmethod
    def write_geometry(
        file,
        zones,
        *,
        compression=None,
        compression_level=None,
        shuffle=False,
        spatial_chunk=65536,
    ):
        """Create or validate reusable geometry from ordered (name, geometry) pairs.

        Codec, level and shuffle affect new files only. Existing geometry is
        validated and reused read-only, retaining its original filter pipeline.
        Compressed arrays are written in spatial_chunk-sized row blocks.
        """
        from ._hdf5_schema import write_geometry

        return write_geometry(
            file,
            zones,
            compression=compression,
            compression_level=compression_level,
            shuffle=shuffle,
            spatial_chunk=spatial_chunk,
        )

    def allocate_field(self, name, component_shape, time, zone, metadata=None):
        """Allocate one field; values are supplied in increasing spatial chunks."""
        if self._series is None:
            raise ValueError("A shared geometry file is required")
        return self._series.allocate(name, component_shape, time, zone, metadata)

    def write_slice(self, name, time, zone, spatial_slice, values):
        if self._series is None:
            raise ValueError("A shared geometry file is required")
        self._series.write_slice(name, time, zone, spatial_slice, values)

    def finish(self):
        if self._series is not None:
            self._series.finish()
        else:
            self._file.flush()

    def close(self):
        if self._series is not None:
            self._series.close()
        else:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class FOAM2HDF5(object):
    """Convert reconstructed OpenFOAM cases to flowTorch HDF5 format.

    Decomposed cases and time-dependent meshes raise ValueError. Existing output
    requires explicit overwrite. Supply geometry_file for reusable separate geometry.

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
        self._loader = FOAMDataloader(path, pt.float64, None)
        self._dtype = dtype
        self._topology: pt.Tensor | None = None
        self._mesh_points: pt.Tensor | None = None

    def convert(
        self,
        filename: str,
        fields: List[str] | None = None,
        times: List[str] | None = None,
        *,
        geometry_file: str | None = None,
        spatial_chunk: int = 65536,
        compression: str | None = None,
        compression_level: int | None = None,
        shuffle: bool | None = None,
        geometry_compression: str | None = None,
        geometry_compression_level: int | None = None,
        geometry_shuffle: bool = False,
        overwrite: bool = False,
        append: bool = False,
    ):
        """Convert OpenFOAM case to flowTorch HDF5 file.

        :param filename: name of the HDF5 file
        :type filename: str
        :param fields: list of fields to convert; if None, all available
            fields are converted
        :type fields: List[str], optional
        :param times: list of times to convert; if None, all available
            times are converted
        :type times: List[str], optional

        compression, compression_level and shuffle configure time-dependent
        fields. geometry_compression, geometry_compression_level and
        geometry_shuffle independently configure newly written geometry.
        Both layouts support lossless gzip and optional Zstd. Reusing a shared
        geometry file preserves its compression regardless of these options.
        append=True adds unique new times to complete split output and inherits
        omitted field selections and compression. It rejects legacy single-file
        output and cannot be combined with overwrite=True. The converter dtype
        must match existing field storage precision.
        """
        from pathlib import Path
        from ._hdf5_schema import select_names
        from ._hdf5_compression import compression_settings

        if append and overwrite:
            raise ValueError("append and overwrite are mutually exclusive")
        if not append:
            compression_settings(
                compression, compression_level, False if shuffle is None else shuffle
            )
        compression_settings(
            geometry_compression, geometry_compression_level, geometry_shuffle
        )
        if self._loader._case.distributed:
            raise ValueError(
                "Decomposed OpenFOAM conversion is unsupported; reconstruct the case first"
            )
        if self._dtype not in (pt.float32, pt.float64) or spatial_chunk < 1:
            raise ValueError("Expected float32/float64 fields and positive chunks")
        selected_times = select_names(self._loader.write_times, times, "time")
        selected_times.sort(key=float)
        select_names(self._loader.field_names[selected_times[0]], fields, "field")
        supported = self._gather_field_information(fields, [selected_times[0]])
        selected_fields = select_names(
            [name for name, _, _ in supported], fields, "field"
        )
        case = Path(self._loader._case.path)
        for time in selected_times:
            select_names(self._loader.field_names[time], selected_fields, "field")
            if (case / time / "polyMesh").exists():
                raise ValueError(
                    "Time-dependent meshes are unsupported; map to a fixed mesh first"
                )
        file_path = case / filename
        if file_path.exists() and not overwrite and not append:
            raise FileExistsError(file_path)
        if append:
            from ._hdf5_schema import inspect_append, validate_append_selection

            info = inspect_append(
                file_path,
                self._dtype,
                geometry_file,
                compression=compression,
                compression_level=compression_level,
                shuffle=shuffle,
            )
            geometry_file = info["geometry_path"]
            if info["zone_names"] != ["mesh"]:
                raise ValueError("OpenFOAM append requires the existing mesh zone")
            if fields is None:
                selected_fields = select_names(
                    self._loader.field_names[selected_times[0]],
                    list(info["schema"]["mesh"]),
                    "field",
                )
            n_cells, _, _ = self._gather_mesh_information(str(case / POLYMESH_PATH))
            schemas = {
                "mesh": {
                    name: (
                        n_cells,
                        *cast(
                            pt.Tensor,
                            self._loader.load_snapshot(name, selected_times[0]),
                        ).shape[1:],
                    )
                    for name in selected_fields
                }
            }
            validate_append_selection(info, selected_times, schemas)
        self._gather_field_information(selected_fields, selected_times)
        if geometry_file is not None:
            from .unstructured import UnstructuredGeometry

            mesh_path = str(case / POLYMESH_PATH)
            n_cells, _, _ = self._gather_mesh_information(mesh_path)
            assert self._mesh_points is not None and self._topology is not None
            geometry = UnstructuredGeometry(
                self._mesh_points.numpy(),
                self._topology.numpy(),
                "Mixed",
                n_cells,
                "cell",
                self._get_cell_centers().numpy(),
                self._get_cell_volumes().numpy(),
            )
            geometry_path = Path(geometry_file).resolve()
            HDF5Writer.write_geometry(
                geometry_path,
                [("mesh", geometry)],
                compression=geometry_compression,
                compression_level=geometry_compression_level,
                shuffle=geometry_shuffle,
                spatial_chunk=spatial_chunk,
            )
            descriptor = file_path.with_suffix(".xdmf")
            if descriptor.exists() and not overwrite and not append:
                raise FileExistsError(descriptor)
            if descriptor.exists() and not append:
                descriptor.unlink()
            with HDF5Writer(
                str(file_path),
                geometry_file=geometry_path,
                field_dtype=self._dtype,
                spatial_chunk=spatial_chunk,
                compression=compression,
                compression_level=compression_level,
                shuffle=shuffle,
                overwrite=overwrite,
                append=append,
            ) as writer:
                for time in selected_times:
                    for name in selected_fields:
                        value = cast(pt.Tensor, self._loader.load_snapshot(name, time))
                        writer.allocate_field(
                            name, tuple(value.shape[1:]), time, "mesh"
                        )
                        for start in range(0, len(value), spatial_chunk):
                            part = slice(start, min(start + spatial_chunk, len(value)))
                            writer.write_slice(name, time, "mesh", part, value[part])
                writer.write_xdmf(descriptor)
        else:
            if file_path.exists():
                file_path.unlink()
            with HDF5Writer(
                str(file_path),
                compression=compression,
                compression_level=compression_level,
                shuffle=shuffle,
                geometry_compression=geometry_compression,
                geometry_compression_level=geometry_compression_level,
                geometry_shuffle=geometry_shuffle,
                spatial_chunk=spatial_chunk,
            ) as writer:
                self._convert_mesh(writer)
                self._convert_fields(writer, selected_fields, selected_times)
                writer.write_xdmf()
        return str(file_path)

    def _convert_mesh(self, writer: HDF5Writer):
        """Convert OpenFOAM mesh to XDMF-based structure.

        :param writer: HDF5 file writer
        :type writer: HDF5Writer
        """
        mesh_path = join(self._loader._case.path, POLYMESH_PATH)
        n_cells, n_points, n_top = self._gather_mesh_information(mesh_path)
        data = self._get_vertices(mesh_path)
        writer.write(VERTICES_DS, (n_points, 3), data, None, pt.float64)
        data = self._get_topology()
        writer.write(CONNECTIVITY_DS, (n_top,), data, None, pt.int64)
        data = self._get_cell_centers()
        writer.write(CENTERS_DS, (n_cells, 3), data, None, pt.float64)
        data = self._get_cell_volumes()
        if not bool(pt.isfinite(data).all() and (data > 0).all()):
            raise ValueError("Cell volumes must be finite and positive")
        writer.write(VOLUMES_DS, (n_cells,), data, None, pt.float64)

    def _gather_mesh_information(self, mesh_path: str):
        """Gather fixed mesh sizes and assemble XDMF polyhedral topology.

        :param mesh_path: path to polyMesh folder
        :type mesh_path: str
        """
        n_cells = self._loader._mesh._get_n_cells(mesh_path)
        n_points_faces, faces = self._loader._mesh._parse_faces(mesh_path)
        owners, neighbors = self._loader._mesh._parse_owners_and_neighbors(mesh_path)
        self._mesh_points = self._loader._mesh._parse_points(mesh_path)

        # Sort face incidences by owner cell and assemble the ragged XDMF
        # polyhedron stream in bulk, without Python loops over mesh cells.
        face_ids = pt.cat((pt.arange(len(owners)), pt.arange(len(neighbors))))
        cell_ids = pt.cat((owners.flatten(), neighbors.flatten())).long()
        order = pt.argsort(cell_ids, stable=True)
        face_ids, cell_ids = face_ids[order], cell_ids[order]
        counts = n_points_faces.flatten()[face_ids].long()
        n_faces_cell = pt.bincount(cell_ids, minlength=n_cells)
        point_counts = pt.zeros(n_cells, dtype=pt.int64)
        point_counts.scatter_add_(0, cell_ids, counts)
        lengths = 2 + n_faces_cell + point_counts
        cell_starts = pt.cumsum(lengths, 0) - lengths
        topology = pt.empty(int(lengths.sum()), dtype=pt.int64)
        topology[cell_starts] = 16
        topology[cell_starts + 1] = n_faces_cell
        face_lengths = counts + 1
        face_starts = pt.cumsum(face_lengths, 0) - face_lengths + 2 * (cell_ids + 1)
        topology[face_starts] = counts
        source_starts = pt.cumsum(counts, 0) - counts
        local_ids = pt.arange(int(counts.sum())) - pt.repeat_interleave(
            source_starts, counts
        )
        repeated_faces = pt.repeat_interleave(face_ids, counts)
        destinations = pt.repeat_interleave(face_starts + 1, counts) + local_ids
        topology[destinations] = faces[repeated_faces, local_ids].long()
        self._topology = topology
        return n_cells, self._mesh_points.size()[0], topology.size()[0]

    def _get_topology(self):
        return self._topology

    def _get_vertices(self, mesh_path: str):
        return self._mesh_points

    def _get_cell_centers(self):
        return self._loader._mesh.get_cell_centers()

    def _get_cell_volumes(self):
        return self._loader._mesh.get_cell_volumes()

    def _convert_fields(
        self,
        writer: HDF5Writer,
        fields: List[str] | None,
        times: List[str] | None,
    ):
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
        for info in field_info:
            logger.info(
                f"Converting field {info[0]} at time {info[1]}, dimension {info[2]}"
            )
            data = self._load_field(info[0], info[1])
            writer.write(info[0], info[2], data, info[1], self._dtype)

    def _gather_field_information(
        self, fields: List[str] | None, times: List[str] | None
    ) -> List[tuple]:
        """Gather selected supported field names and shapes.

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

        field_info = []
        mesh_path = join(self._loader._case.path, POLYMESH_PATH)
        n_cells = self._loader._mesh._get_n_cells(mesh_path)
        times_to_convert = self._loader.write_times
        if times is not None:
            times_to_convert = [t for t in times if t in times_to_convert]
        for time in times_to_convert:
            fields_to_convert = self._loader.field_names[time]
            if fields is not None:
                fields_to_convert = [
                    field for field in fields if field in fields_to_convert
                ]
            for name in fields_to_convert:
                path = self._loader._case.build_file_path(name, time)
                with foam_open(path) as stream:
                    field_type = foam_header(stream).get(b"class")
                if field_type not in FIELD_TYPE_DIMENSION:
                    if fields is None:
                        continue  # Default export contains supported cell fields only.
                    raise ValueError(
                        f"Unsupported OpenFOAM field type for {name}: {field_type!r}"
                    )
                field_info.append(
                    (name, time, (n_cells, FIELD_TYPE_DIMENSION[field_type]))
                )
        return field_info

    def _load_field(self, field: str, time: str) -> pt.Tensor:
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
        file_path = str(file_path)
        if "/" in file_path:
            self._path = file_path[: file_path.rfind("/")]
            self._hdf5_filename = file_path[file_path.rfind("/") + 1 :]
        else:
            self._path = "."
            self._hdf5_filename = file_path
        self._file = hdf5_file
        self._owns_file = False
        self._versioned = self._file.attrs.get("schema") == "flowtorch-unstructured"
        self._n_cells = 0 if self._versioned else self._get_n_cells()

    @classmethod
    def from_filepath(cls, file_path: str):
        """Create XDMFWriter from a file path.

        :param file_path: path to HDF5 file
        :type file_path: str
        """
        writer = cls(file_path, File(file_path, mode="r"))
        writer._owns_file = True
        return writer

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
            logger.warning("XDMF: could not determine number of cells.")
        return n_cells

    def _add_grid(self, time: Union[str, None], offset: str = "") -> str:
        """Create an XDMF grid element.

        :param time: snapshot write time
        :type time: str
        :param offset: number of spaces for indentation, defaults to ""
        :type offset: str, optional
        :return: XDMF grid element
        :rtype: str
        """
        grid = offset + '<Grid Name="Grid" GridType="Uniform">\n'
        grid += self._add_topology(offset + " " * 4)
        grid += self._add_geometry(offset + " " * 4)
        if time is not None:
            grid += offset + " " * 4 + '<Time Value="{:s}"/>\n'.format(time)
            for key in self._find_attributes(time):
                grid += self._add_attribute(time, key, offset + " " * 4)
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
        topology = offset + '<Topology Name="{:s}" TopologyType="Mixed">\n'.format(
            TOPOLOGY
        )
        location = self._hdf5_filename + ":/{:s}/{:s}".format(
            CONST_GROUP, CONNECTIVITY_DS
        )
        topology += self._add_dataitem(location, offset + " " * 4)
        topology += offset + "</Topology>\n"
        return topology

    def _add_geometry(self, offset: str = "") -> str:
        """Create XDMF geometry element.

        :param offset: number of spaces for indentation, defaults to ""
        :type offset: str, optional
        :return: XDMF geometry element
        :rtype: str
        """
        geometry = offset + '<Geometry GeometryType="XYZ">\n'
        location = self._hdf5_filename + ":/{:s}/{:s}".format(CONST_GROUP, VERTICES_DS)
        geometry += self._add_dataitem(location, offset + " " * 4)
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
        tensor_type = xdmf_attributes[__import__("math").prod(shape[1:])]
        attribute = (
            offset
            + '<Attribute Name="{:s}" AttributeType="{:s}" Center="Cell">\n'.format(
                name, tensor_type
            )
        )
        attribute += self._add_dataitem(location, offset + " " * 4)
        attribute += offset + "</Attribute>\n"
        return attribute

    def _add_dataitem(self, location: str, offset: str = "") -> str:
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
        dtype, precision = xdmf_dtype_conversion[str(self._file[path_in_file].dtype)]
        dataitem = (
            offset
            + '<DataItem Dimensions="{:s}" NumberType="{:s}" Precision="{:d}" Format="HDF">\n'.format(
                dimensions, dtype, precision
            )
        )
        dataitem += offset + " " * 4 + "{:s}\n".format(location)
        dataitem += offset + "</DataItem>\n"
        return dataitem

    def create_xdmf(self, filename: str = None) -> None:
        """Create XDMF wrapper to access flowTorch HDF5 file in ParaView.

        :param filename: name of the XDMF file, defaults to None
        :type filename: str, optional
        """
        if self._versioned:
            from pathlib import Path
            from ._hdf5_schema import FieldFile, SeriesReader

            # Reuse the same descriptor generator without opening output writable.
            reader = SeriesReader(
                join(self._path, self._hdf5_filename),
                pt.float32,
            )
            try:
                storage = object.__new__(FieldFile)
                storage.file = reader.file
                storage.frames = reader.frames
                storage.geometry = reader.geometry
                storage.path = reader.path
                storage.geometry_path = Path(reader.geometry.filename).resolve()
                storage.write_xdmf(
                    Path(self._path)
                    / (filename or Path(self._hdf5_filename).with_suffix(".xdmf").name)
                )
            finally:
                reader.close()
            return
        xdmf_str = XDMF_HEADER
        times = list(self._file[VAR_GROUP].keys()) if VAR_GROUP in self._file else []
        if len(times) > 0:
            for time in times:
                xdmf_str += self._add_grid(time, " " * 12)
        else:
            xdmf_str += self._add_grid(None, " " * 12)
        xdmf_str = xdmf_str[:-1]  # remove last linebreak
        xdmf_str += XDMF_FOOTER

        if filename is None:
            if "." in self._hdf5_filename:
                filename = (
                    self._hdf5_filename[: self._hdf5_filename.rfind(".")] + ".xdmf"
                )
            else:
                filename = self._hdf5_filename + ".xdmf"
        logger.info(
            "Writing file {:s} as wrapper for {:s} at location {:s}".format(
                filename, self._hdf5_filename, self._path
            )
        )
        with open(self._path + "/" + filename, "w") as file:
            file.write(xdmf_str)

    def close(self):
        """Close only handles owned by from_filepath()."""
        if self._owns_file:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


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
    to_file_path = join(path, to_file)
    with HDF5Dataloader(from_file_path, dtype=pt.float64) as loader:
        if loader._series is not None:
            HDF5Writer.write_geometry(
                to_file_path,
                [(zone, loader.geometry(zone)) for zone in loader.zone_names],
            )
            return
        with HDF5Writer(to_file_path) as writer:
            datasets = {
                VERTICES_DS: loader.edge_vertices,
                CENTERS_DS: loader.vertices,
                VOLUMES_DS: loader.weights,
                CONNECTIVITY_DS: loader.connectivity,
            }
            for name, data in datasets.items():
                writer.write(name, data.shape, data, None, data.dtype)
            writer.finish()
        with XDMFWriter.from_filepath(to_file_path) as xdmf:
            xdmf.create_xdmf(f"{to_file.rsplit('.', 1)[0]}.xdmf")
