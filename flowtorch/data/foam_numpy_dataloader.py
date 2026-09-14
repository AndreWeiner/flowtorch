"""Dataloader for the ``foamToNumpy`` function-object output.

The OpenFOAM function object writes processor-local NumPy arrays into restart
segments and fixed-size batches.  Segments are applied in numerical folder
order. As in the accompanying ``numpyToFoam`` importer, a snapshot from a later
segment or batch replaces an earlier snapshot at the same time.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Tuple, Union

import numpy as np
import torch as pt

from flowtorch import DEFAULT_DTYPE
from .dataloader import Dataloader
from .utils import check_list_or_str

_BATCH_PATTERN = re.compile(r"batch_(\d+)$")
_TIME_TOLERANCE = 1.0e-15
_FIELD_COMPONENTS = {
    "volScalarField": 1,
    "volVectorField": 3,
    "volSphericalTensorField": 1,
    "volSymmTensorField": 6,
    "volTensorField": 9,
}


@dataclass(frozen=True)
class _Dataset:
    """Validated processor-local arrays for one mesh or cell-zone dataset."""

    field_files: Dict[str, Tuple[Path, ...]]
    field_shapes: Dict[str, Tuple[int, ...]]
    processor_sizes: Tuple[int, ...]


@dataclass(frozen=True)
class _Batch:
    """Validated metadata and files belonging to one output batch."""

    path: Path
    times: Tuple[float, ...]
    mesh_revision: int
    fields: Tuple[str, ...]
    field_classes: Dict[str, str]
    cell_zones: Tuple[str, ...]
    datasets: Dict[Union[str, None], _Dataset]


@dataclass(frozen=True)
class _SegmentMetadata:
    """Storage-layout metadata read from one ``segmentInfo`` file."""

    path: Path
    zone_layout_version: Union[int, None]
    region: Union[str, None]
    n_processors: Union[int, None]


@dataclass(frozen=True)
class _ZoneMapping:
    """Processor-local cell addressing for one zone and mesh revision."""

    files: Tuple[Path, ...]
    processor_sizes: Tuple[int, ...]
    mesh_sizes: Tuple[int, ...]


@dataclass(frozen=True)
class _Snapshot:
    """Location of a committed snapshot in a batch."""

    time: float
    batch: _Batch
    index: int


def _entry(text: str, name: str) -> str:
    """Read one semicolon-terminated primitive dictionary entry."""
    match = re.search(rf"^\s*{re.escape(name)}\s+([^;]+);", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"Missing entry {name!r}")
    return match.group(1).strip()


def _optional_entry(text: str, name: str) -> Union[str, None]:
    match = re.search(rf"^\s*{re.escape(name)}\s+([^;]+);", text, re.MULTILINE)
    return None if match is None else match.group(1).strip()


def _word_list(text: str, name: str) -> Tuple[str, ...]:
    value = _entry(text, name)
    if not value.startswith("(") or not value.endswith(")"):
        raise ValueError(f"Entry {name!r} is not an OpenFOAM word list")
    return tuple(value[1:-1].split())


def _field_classes(text: str) -> Dict[str, str]:
    match = re.search(r"\bfieldClasses\s*\{(.*?)\}", text, re.DOTALL)
    if match is None:
        raise ValueError("Missing entry 'fieldClasses'")
    return dict(re.findall(r"([^\s;{}]+)\s+([^\s;{}]+)\s*;", match.group(1)))


def _same_time(first: float, second: float) -> bool:
    return abs(first - second) <= _TIME_TOLERANCE * max(1.0, abs(second))


def _format_time(value: float) -> str:
    return np.format_float_positional(value, unique=True, trim="-")


def _segment_sort_key(name: str) -> Tuple[float, int, str]:
    """Sort numeric segment names, including collision suffixes such as ``0_1``."""
    try:
        return float(name), 0, name
    except ValueError:
        base, separator, suffix = name.rpartition("_")
        if separator and suffix.isdigit():
            try:
                return float(base), int(suffix), name
            except ValueError:
                pass
    raise ValueError(
        f"NumPy segment folder {name!r} is not numeric and cannot be ordered"
    )


class FOAMNumpyDataloader(Dataloader):
    """Load batched output from the OpenFOAM ``foamToNumpy`` function object.

    ``foamToNumpy`` creates one segment for every solver or ``postProcess``
    invocation.  All segments are merged in numerical folder order unless
    excluded with ``ignore_segments``.  The catalog is sorted by time and
    snapshots from later segments replace duplicate times, matching
    ``numpyToFoam``.

    Examples
    --------
    Configure the function object to write time and geometry metadata:

    .. code-block:: text

        numpyExport
        {
            type                foamToNumpy;
            libs                (numpyFunctionObjects);
            fields              (p U);
            writeTimes          true;
            writeCellCentres    true;
            writeCellVolumes    true;
        }

    Load all automatically merged restart segments, except any explicitly
    ignored segments:

    >>> from flowtorch.data import FOAMNumpyDataloader
    >>> loader = FOAMNumpyDataloader(
    ...     "postProcessing/numpyExport", ignore_segments=["0.5_1"]
    ... )
    >>> pressure = loader.load_snapshot("p", loader.write_times)

    Cell-zone output uses the same interface after selecting one exported zone:

    .. code-block:: text

        numpyExport
        {
            type                foamToNumpy;
            libs                (numpyFunctionObjects);
            fields              (p U);
            cellZones           (fluidCore porous);
            writeCellCentres    true;
            writeCellVolumes    true;
        }

    >>> loader = FOAMNumpyDataloader(
    ...     "postProcessing/numpyExport", zone="fluidCore"
    ... )
    >>> loader.zone_names
    ['fluidCore', 'porous']

    :param path: function-object output directory or one segment directory
    :param ignore_segments: segment name or names to exclude from the catalog
    :param dtype: output tensor dtype, defaults to ``torch.float32``
    :param zone: active cell zone; defaults to the first exported zone
    """

    def __init__(
        self,
        path: str,
        ignore_segments: Union[List[str], str, None] = None,
        dtype: pt.dtype = DEFAULT_DTYPE,
        zone: Union[str, None] = None,
    ):
        self._path = Path(path)
        if not self._path.is_dir():
            raise ValueError(f"Directory does not exist: {path}")
        self._dtype = dtype
        self._segment_paths = self._resolve_segments(ignore_segments)
        self._segment_metadata: Dict[Path, _SegmentMetadata] = {}
        self._zone_mappings: Dict[Tuple[Path, int, str], _ZoneMapping] = {}
        snapshots: List[_Snapshot] = []

        for segment_path in self._segment_paths:
            metadata = self._read_segment_metadata(segment_path)
            self._segment_metadata[segment_path] = metadata
            segment_batches = self._load_segment(metadata)
            for batch in segment_batches:
                snapshots.extend(
                    _Snapshot(time, batch, index)
                    for index, time in enumerate(batch.times)
                )

        self._snapshots = self._normalise_snapshots(snapshots)
        if not self._snapshots:
            raise ValueError("No committed NumPy snapshots were found")

        self._validate_selected_layout()
        self._zone_names = self._snapshots[0].batch.cell_zones
        self._zone: Union[str, None]
        if self._zone_names:
            self._zone = self._zone_names[0] if zone is None else zone
            if self._zone not in self._zone_names:
                raise ValueError(
                    f"Cell zone {self._zone!r} not found; available zones are "
                    f"{list(self._zone_names)}"
                )
        else:
            if zone is not None:
                raise ValueError(
                    "A cell zone cannot be selected from a whole-mesh NumPy dataset"
                )
            self._zone = None

        self._write_times = [
            _format_time(snapshot.time) for snapshot in self._snapshots
        ]
        self._vertices: Dict[Union[str, None], pt.Tensor] = {}
        self._weights: Dict[Union[str, None], pt.Tensor] = {}

    def _resolve_segments(
        self, ignore_segments: Union[List[str], str, None]
    ) -> List[Path]:
        if (self._path / "segmentInfo").is_file():
            if ignore_segments is not None:
                raise ValueError(
                    "ignore_segments must not be provided when path is a segment "
                    "directory"
                )
            return [self._path]

        available = sorted(
            (
                child.name
                for child in self._path.iterdir()
                if child.is_dir() and (child / "segmentInfo").is_file()
            ),
            key=_segment_sort_key,
        )
        if ignore_segments is None:
            ignored: List[str] = []
        else:
            check_list_or_str(ignore_segments, "ignore_segments")
            ignored = (
                [ignore_segments]
                if isinstance(ignore_segments, str)
                else ignore_segments
            )
            if len(set(ignored)) != len(ignored):
                raise ValueError("ignore_segments must not contain duplicate names")
            unknown = sorted(set(ignored) - set(available))
            if unknown:
                raise ValueError(f"Cannot ignore unknown NumPy segments: {unknown}")

        selected = [name for name in available if name not in set(ignored)]

        if not selected:
            raise ValueError(f"No NumPy segments remain in {self._path}")

        paths = []
        for name in selected:
            segment_path = self._path / name
            if not (segment_path / "segmentInfo").is_file():
                raise ValueError(f"Cannot find NumPy segment {segment_path}")
            paths.append(segment_path)
        return paths

    def _read_segment_metadata(self, segment_path: Path) -> _SegmentMetadata:
        """Read and validate storage metadata shared by all segment batches."""
        info_path = segment_path / "segmentInfo"
        info = info_path.read_text()
        order = _optional_entry(info, "order")
        if order is not None and order != "fortran":
            raise ValueError(
                f"Unsupported NumPy storage order {order!r} in {info_path}"
            )

        version_entry = _optional_entry(info, "zoneLayoutVersion")
        if version_entry is None:
            if (
                _optional_entry(info, "region") is not None
                or _optional_entry(info, "nProcs") is not None
            ):
                raise ValueError(f"Incomplete cell-zone layout metadata in {info_path}")
            return _SegmentMetadata(segment_path, None, None, None)

        try:
            version = int(version_entry)
            n_processors = int(_entry(info, "nProcs"))
            region = _entry(info, "region")
        except ValueError as error:
            raise ValueError(
                f"Cannot parse cell-zone layout metadata in {info_path}: {error}"
            ) from error
        if version != 1:
            raise ValueError(
                f"Unsupported cell-zone layout version {version} in {info_path}"
            )
        if n_processors <= 0:
            raise ValueError(f"Invalid processor count {n_processors} in {info_path}")
        if not region:
            raise ValueError(f"Empty OpenFOAM region in {info_path}")
        return _SegmentMetadata(segment_path, version, region, n_processors)

    def _load_segment(self, metadata: _SegmentMetadata) -> List[_Batch]:
        segment_path = metadata.path

        batch_paths = sorted(
            (
                (int(match.group(1)), child)
                for child in segment_path.iterdir()
                if child.is_dir()
                and (match := _BATCH_PATTERN.fullmatch(child.name)) is not None
            ),
            key=lambda item: item[0],
        )
        batches = []
        revisions = set()
        for _, batch_path in batch_paths:
            state_path = batch_path / "state"
            if not state_path.is_file():
                continue
            batch = self._load_batch(batch_path, state_path, metadata)
            if batch is not None:
                batches.append(batch)
                revisions.add(batch.mesh_revision)

        if len(revisions) > 1:
            raise ValueError(
                f"Segment {segment_path.name!r} contains multiple mesh revisions; "
                "dynamic meshes are not supported"
            )
        return batches

    def _load_batch(
        self,
        batch_path: Path,
        state_path: Path,
        metadata: _SegmentMetadata,
    ) -> Union[_Batch, None]:
        try:
            state = state_path.read_text()
            count = int(_entry(state, "count"))
            mesh_revision = int(_entry(state, "meshRevision"))
            fields = _word_list(state, "fields")
            classes = _field_classes(state)
            state_version = _optional_entry(state, "zoneLayoutVersion")
            cell_zones_entry = _optional_entry(state, "cellZones")
            cell_zones = (
                _word_list(state, "cellZones") if cell_zones_entry is not None else ()
            )
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Cannot parse batch state {state_path}: {error}"
            ) from error

        if count < 0:
            raise ValueError(f"Invalid committed count {count} in {state_path}")
        if count == 0:
            return None
        if not fields:
            raise ValueError(f"No fields recorded in {state_path}")
        if set(fields) != set(classes):
            raise ValueError(
                f"Field class metadata does not match fields in {state_path}"
            )
        if metadata.zone_layout_version is None:
            if state_version is not None or cell_zones_entry is not None:
                raise ValueError(
                    f"Batch {state_path} declares cell zones but its segment does not"
                )
        else:
            try:
                parsed_state_version = (
                    None if state_version is None else int(state_version)
                )
            except ValueError as error:
                raise ValueError(
                    f"Invalid cell-zone layout version in {state_path}"
                ) from error
            if parsed_state_version != metadata.zone_layout_version:
                raise ValueError(
                    f"Cell-zone layout version in {state_path} does not match "
                    f"{metadata.path / 'segmentInfo'}"
                )
            if not cell_zones:
                raise ValueError(f"No cell zones recorded in {state_path}")
            if len(set(cell_zones)) != len(cell_zones):
                raise ValueError(f"Duplicate cell-zone names in {state_path}")

        times_path = batch_path / "times.npy"
        times_array = self._load_array(
            times_path, require_fortran_header=bool(cell_zones)
        )
        if times_array.ndim != 1 or times_array.shape[0] < count:
            raise ValueError(
                f"Invalid times array {times_path}; expected at least {count} values"
            )
        times = tuple(float(value) for value in times_array[:count])
        if any(
            current <= previous or _same_time(current, previous)
            for previous, current in zip(times, times[1:])
        ):
            raise ValueError(f"Times are not strictly increasing in {times_path}")

        datasets: Dict[Union[str, None], _Dataset] = {}
        if cell_zones:
            assert metadata.n_processors is not None
            for zone in cell_zones:
                mapping = self._zone_mapping(metadata, mesh_revision, zone)
                datasets[zone] = self._load_dataset(
                    batch_path / "cellZones" / zone,
                    fields,
                    classes,
                    count,
                    metadata.n_processors,
                    mapping.processor_sizes,
                )
        else:
            datasets[None] = self._load_dataset(
                batch_path, fields, classes, count, None, None
            )

        return _Batch(
            batch_path,
            times,
            mesh_revision,
            fields,
            classes,
            cell_zones,
            datasets,
        )

    def _load_dataset(
        self,
        directory: Path,
        fields: Tuple[str, ...],
        classes: Dict[str, str],
        count: int,
        expected_n_processors: Union[int, None],
        expected_sizes: Union[Tuple[int, ...], None],
    ) -> _Dataset:
        """Validate all field arrays belonging to one mesh or zone dataset."""
        if not directory.is_dir():
            raise ValueError(f"Missing NumPy dataset directory {directory}")

        field_files: Dict[str, Tuple[Path, ...]] = {}
        field_shapes: Dict[str, Tuple[int, ...]] = {}
        expected_processor_ids: Union[Tuple[int, ...], None] = None
        processor_sizes: Union[Tuple[int, ...], None] = None

        for field in fields:
            field_class = classes[field]
            if field_class not in _FIELD_COMPONENTS:
                raise ValueError(
                    f"Unsupported field class {field_class!r} for {field!r} in "
                    f"{directory}"
                )
            pattern = re.compile(rf"{re.escape(field)}_proc_(\d+)\.npy$")
            indexed_files = sorted(
                (
                    (int(match.group(1)), child)
                    for child in directory.iterdir()
                    if child.is_file()
                    and (match := pattern.fullmatch(child.name)) is not None
                ),
                key=lambda item: item[0],
            )
            if not indexed_files:
                raise ValueError(
                    f"No processor arrays found for {field!r} in {directory}"
                )
            processor_ids = tuple(index for index, _ in indexed_files)
            required_ids = tuple(
                range(
                    len(processor_ids)
                    if expected_n_processors is None
                    else expected_n_processors
                )
            )
            if processor_ids != required_ids:
                raise ValueError(
                    f"Invalid processor set for {field!r} in {directory}; "
                    f"expected {required_ids}, found {processor_ids}"
                )
            if expected_processor_ids is None:
                expected_processor_ids = processor_ids
            elif processor_ids != expected_processor_ids:
                raise ValueError(f"Processor sets differ between fields in {directory}")

            files = tuple(path for _, path in indexed_files)
            arrays = [
                self._load_array(
                    path,
                    mmap=True,
                    require_fortran_header=expected_sizes is not None,
                )
                for path in files
            ]
            components = _FIELD_COMPONENTS[field_class]
            expected_ndim = 2 if components == 1 else 3
            for path, array in zip(files, arrays):
                valid_components = components == 1 or (
                    array.ndim >= 2 and array.shape[1] == components
                )
                if (
                    array.ndim != expected_ndim
                    or not valid_components
                    or array.shape[-1] < count
                ):
                    raise ValueError(
                        f"Invalid shape {array.shape} for {field_class} array {path}"
                    )

            sizes = tuple(array.shape[0] for array in arrays)
            if processor_sizes is None:
                processor_sizes = sizes
            elif sizes != processor_sizes:
                raise ValueError(
                    f"Processor cell counts differ between fields in {directory}"
                )

            component_shape = tuple(arrays[0].shape[1:-1])
            if any(tuple(array.shape[1:-1]) != component_shape for array in arrays):
                raise ValueError(
                    f"Component shapes differ for {field!r} in {directory}"
                )
            field_files[field] = files
            field_shapes[field] = component_shape

        assert processor_sizes is not None
        if expected_sizes is not None and processor_sizes != expected_sizes:
            raise ValueError(
                f"Field entity counts in {directory} do not match its cell-zone "
                f"mapping; expected {expected_sizes}, found {processor_sizes}"
            )
        return _Dataset(field_files, field_shapes, processor_sizes)

    def _zone_mapping(
        self, metadata: _SegmentMetadata, mesh_revision: int, zone: str
    ) -> _ZoneMapping:
        """Load and cache the mandatory mapping for one exported cell zone."""
        key = (metadata.path, mesh_revision, zone)
        if key in self._zone_mappings:
            return self._zone_mappings[key]

        assert metadata.n_processors is not None
        geometry_path = metadata.path / f"geometry_{mesh_revision:06d}"
        if not geometry_path.is_dir():
            raise ValueError(
                f"Missing geometry revision {geometry_path} for cell zone {zone!r}"
            )

        mesh_pattern = re.compile(r"mesh_proc_(\d+)$")
        mesh_files = sorted(
            (
                (int(match.group(1)), child)
                for child in geometry_path.iterdir()
                if child.is_file()
                and (match := mesh_pattern.fullmatch(child.name)) is not None
            ),
            key=lambda item: item[0],
        )
        required_ids = tuple(range(metadata.n_processors))
        if tuple(index for index, _ in mesh_files) != required_ids:
            raise ValueError(
                f"Invalid mesh-metadata processor set in {geometry_path}; "
                f"expected {required_ids}"
            )

        mesh_sizes = []
        for _, path in mesh_files:
            try:
                text = path.read_text()
                n_cells = int(_entry(text, "nCells"))
                revision = int(_entry(text, "meshRevision"))
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"Cannot parse zone mesh metadata {path}: {error}"
                ) from error
            if n_cells < 0 or revision != mesh_revision:
                raise ValueError(f"Invalid zone mesh metadata in {path}")
            mesh_sizes.append(n_cells)

        zone_path = geometry_path / "cellZones" / zone
        if not zone_path.is_dir():
            raise ValueError(f"Missing cell-zone mapping directory {zone_path}")
        id_pattern = re.compile(r"cellIds_proc_(\d+)\.npy$")
        indexed_files = sorted(
            (
                (int(match.group(1)), child)
                for child in zone_path.iterdir()
                if child.is_file()
                and (match := id_pattern.fullmatch(child.name)) is not None
            ),
            key=lambda item: item[0],
        )
        if tuple(index for index, _ in indexed_files) != required_ids:
            raise ValueError(
                f"Invalid cell-ID processor set in {zone_path}; expected {required_ids}"
            )

        files = tuple(path for _, path in indexed_files)
        sizes = []
        for processor, path in enumerate(files):
            cell_ids = self._load_cell_ids(path)
            if cell_ids.size and int(cell_ids[-1]) >= mesh_sizes[processor]:
                raise ValueError(
                    f"Cell ID in {path} exceeds processor-local mesh size "
                    f"{mesh_sizes[processor]}"
                )
            sizes.append(int(cell_ids.size))

        mapping = _ZoneMapping(files, tuple(sizes), tuple(mesh_sizes))
        self._zone_mappings[key] = mapping
        return mapping

    @staticmethod
    def _array_header(path: Path) -> Tuple[Tuple[int, ...], bool, np.dtype]:
        """Read an ``.npy`` header without loading its payload."""
        try:
            with path.open("rb") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                        stream
                    )
                elif version == (2, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                        stream
                    )
                else:
                    raise ValueError(f"unsupported NumPy format version {version}")
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot read NumPy header {path}: {error}") from error
        return tuple(shape), bool(fortran_order), np.dtype(dtype)

    @staticmethod
    def _load_array(
        path: Path,
        mmap: bool = False,
        require_fortran_header: bool = False,
    ) -> np.ndarray:
        if not path.is_file():
            raise ValueError(f"Missing NumPy array {path}")
        mmap_mode = None
        if mmap or require_fortran_header:
            shape, fortran_order, _ = FOAMNumpyDataloader._array_header(path)
            if require_fortran_header and not fortran_order:
                raise ValueError(f"Array is not stored in Fortran order: {path}")
            # NumPy cannot memory-map an array with an empty payload. Empty
            # processor-local cell-zone arrays are nevertheless valid.
            if mmap and all(size > 0 for size in shape):
                mmap_mode = "r"
        try:
            array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot load NumPy array {path}: {error}") from error
        if array.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError(f"Unsupported dtype {array.dtype} in {path}")
        if not array.flags.f_contiguous:
            raise ValueError(f"Array is not Fortran-contiguous: {path}")
        return array

    @staticmethod
    def _load_cell_ids(path: Path) -> np.ndarray:
        """Load and validate processor-local cell-zone addressing."""
        if not path.is_file():
            raise ValueError(f"Missing cell-ID array {path}")
        shape, fortran_order, dtype = FOAMNumpyDataloader._array_header(path)
        if len(shape) != 1 or dtype.kind != "i" or dtype.itemsize != 8:
            raise ValueError(
                f"Cell IDs in {path} must be a one-dimensional signed int64 array"
            )
        if not fortran_order:
            raise ValueError(f"Cell-ID array is not stored in Fortran order: {path}")
        try:
            array = np.load(
                path,
                mmap_mode="r" if shape[0] > 0 else None,
                allow_pickle=False,
            )
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot load cell-ID array {path}: {error}") from error
        if array.size and (int(array[0]) < 0 or bool(np.any(array[1:] <= array[:-1]))):
            raise ValueError(
                f"Cell IDs in {path} must be nonnegative and strictly increasing"
            )
        return array

    def _validate_selected_layout(self) -> None:
        """Ensure every selected snapshot represents one stable state space."""
        reference_batch = self._snapshots[0].batch
        reference_zones = reference_batch.cell_zones
        reference_metadata = self._segment_metadata[reference_batch.path.parent]

        for snapshot in self._snapshots[1:]:
            batch = snapshot.batch
            metadata = self._segment_metadata[batch.path.parent]
            if batch.cell_zones != reference_zones:
                raise ValueError(
                    "Cell-zone selections differ across the selected NumPy snapshots"
                )
            if metadata.zone_layout_version != reference_metadata.zone_layout_version:
                raise ValueError(
                    "Whole-mesh and cell-zone NumPy segments cannot be combined"
                )
            if reference_zones and (
                metadata.region != reference_metadata.region
                or metadata.n_processors != reference_metadata.n_processors
            ):
                raise ValueError(
                    "OpenFOAM region or processor count changes across the selected "
                    "cell-zone segments"
                )

        dataset_keys: Tuple[Union[str, None], ...] = (
            tuple(reference_zones) if reference_zones else (None,)
        )
        for key in dataset_keys:
            reference_sizes = reference_batch.datasets[key].processor_sizes
            for snapshot in self._snapshots[1:]:
                sizes = snapshot.batch.datasets[key].processor_sizes
                if sizes != reference_sizes:
                    subject = "cell-zone" if key is not None else "processor cell"
                    raise ValueError(
                        f"{subject.capitalize()} counts change across the selected "
                        "snapshots; dynamic mappings or changed decompositions are "
                        "not supported"
                    )

        if not reference_zones:
            return

        for zone in reference_zones:
            reference_mapping = self._zone_mapping(
                reference_metadata, reference_batch.mesh_revision, zone
            )
            checked = set()
            for snapshot in self._snapshots[1:]:
                batch = snapshot.batch
                metadata = self._segment_metadata[batch.path.parent]
                mapping_key = (metadata.path, batch.mesh_revision, zone)
                if mapping_key in checked:
                    continue
                checked.add(mapping_key)
                mapping = self._zone_mapping(metadata, batch.mesh_revision, zone)
                if (
                    mapping.processor_sizes != reference_mapping.processor_sizes
                    or mapping.mesh_sizes != reference_mapping.mesh_sizes
                    or any(
                        not np.array_equal(
                            self._load_cell_ids(first), self._load_cell_ids(second)
                        )
                        for first, second in zip(reference_mapping.files, mapping.files)
                    )
                ):
                    raise ValueError(
                        f"Cell-zone mapping for {zone!r} changes across the selected "
                        "snapshots; changing state-vector mappings are not supported"
                    )

    def _dataset(self, batch: _Batch) -> _Dataset:
        """Return the whole-mesh or active-zone arrays for a batch."""
        return batch.datasets[self._zone]

    @staticmethod
    def _normalise_snapshots(snapshots: List[_Snapshot]) -> List[_Snapshot]:
        ordered = sorted(snapshots, key=lambda snapshot: snapshot.time)
        unique: List[_Snapshot] = []
        for snapshot in ordered:
            if unique and _same_time(snapshot.time, unique[-1].time):
                unique[-1] = snapshot
            else:
                unique.append(snapshot)
        return unique

    def _snapshot_at(self, time: str) -> _Snapshot:
        try:
            value = float(time)
        except ValueError as error:
            raise ValueError(f"Invalid snapshot time {time!r}") from error
        for snapshot in self._snapshots:
            if _same_time(value, snapshot.time):
                return snapshot
        raise ValueError(f"Snapshot time {time!r} is not available")

    def _load_field(self, field: str, snapshots: List[_Snapshot]) -> pt.Tensor:
        """Preallocate and populate one time-last tensor for a field."""
        for snapshot in snapshots:
            if field not in self._dataset(snapshot.batch).field_files:
                raise ValueError(
                    f"Field {field!r} is not available at time "
                    f"{_format_time(snapshot.time)!r}"
                )

        component_shape = self._dataset(snapshots[0].batch).field_shapes[field]
        if any(
            self._dataset(snapshot.batch).field_shapes[field] != component_shape
            for snapshot in snapshots[1:]
        ):
            raise ValueError(f"Component shape changes across snapshots for {field!r}")

        n_cells = sum(self._dataset(snapshots[0].batch).processor_sizes)
        result = pt.empty(
            (n_cells, *component_shape, len(snapshots)), dtype=self._dtype
        )
        grouped: Dict[Path, List[Tuple[int, _Snapshot]]] = {}
        for output_index, snapshot in enumerate(snapshots):
            grouped.setdefault(snapshot.batch.path, []).append((output_index, snapshot))

        for entries in grouped.values():
            batch = entries[0][1].batch
            dataset = self._dataset(batch)
            offset = 0
            for processor_size, path in zip(
                dataset.processor_sizes, dataset.field_files[field]
            ):
                array = self._load_array(path, mmap=True)
                target = result[offset : offset + processor_size]
                for output_index, snapshot in entries:
                    values = np.array(array[..., snapshot.index], copy=True, order="C")
                    target[..., output_index].copy_(
                        pt.as_tensor(values, dtype=self._dtype)
                    )
                offset += processor_size
        return result[..., 0] if len(snapshots) == 1 else result

    def _load_field_slice(
        self, field: str, snapshots: List[_Snapshot], spatial_slice: slice
    ) -> pt.Tensor:
        """Load a contiguous global cell slice from memory-mapped arrays."""
        for snapshot in snapshots:
            if field not in self._dataset(snapshot.batch).field_files:
                raise ValueError(
                    f"Field {field!r} is not available at time "
                    f"{_format_time(snapshot.time)!r}"
                )
        component_shape = self._dataset(snapshots[0].batch).field_shapes[field]
        if any(
            self._dataset(snapshot.batch).field_shapes[field] != component_shape
            for snapshot in snapshots[1:]
        ):
            raise ValueError(f"Component shape changes across snapshots for {field!r}")
        n_cells = sum(self._dataset(snapshots[0].batch).processor_sizes)
        start, stop, step = spatial_slice.indices(n_cells)
        if step != 1:
            raise ValueError("spatial_slice must have unit stride")
        result = pt.empty(
            (stop - start, *component_shape, len(snapshots)), dtype=self._dtype
        )
        grouped: Dict[Path, List[Tuple[int, _Snapshot]]] = {}
        for output_index, snapshot in enumerate(snapshots):
            grouped.setdefault(snapshot.batch.path, []).append((output_index, snapshot))
        for entries in grouped.values():
            batch = entries[0][1].batch
            dataset = self._dataset(batch)
            global_offset = 0
            for processor_size, path in zip(
                dataset.processor_sizes, dataset.field_files[field]
            ):
                overlap_start = max(start, global_offset)
                overlap_stop = min(stop, global_offset + processor_size)
                if overlap_start < overlap_stop:
                    local_start = overlap_start - global_offset
                    local_stop = overlap_stop - global_offset
                    target_start = overlap_start - start
                    target_stop = overlap_stop - start
                    array = self._load_array(path, mmap=True)
                    for output_index, snapshot in entries:
                        values = np.array(
                            array[local_start:local_stop, ..., snapshot.index],
                            copy=True,
                            order="C",
                        )
                        result[target_start:target_stop, ..., output_index].copy_(
                            pt.as_tensor(values, dtype=self._dtype)
                        )
                global_offset += processor_size
        return result[..., 0] if len(snapshots) == 1 else result

    def load_snapshot(
        self, field_name: Union[List[str], str], time: Union[List[str], str]
    ) -> Union[List[pt.Tensor], pt.Tensor]:
        """Load one or more fields at one or more output times."""
        check_list_or_str(field_name, "field_name")
        check_list_or_str(time, "time")
        fields = field_name if isinstance(field_name, list) else [field_name]
        times = time if isinstance(time, list) else [time]
        snapshots = [self._snapshot_at(value) for value in times]
        loaded = [self._load_field(field, snapshots) for field in fields]
        return loaded if isinstance(field_name, list) else loaded[0]

    def load_snapshot_slice(
        self,
        field_name: Union[List[str], str],
        time: Union[List[str], str],
        spatial_slice: slice,
    ) -> Union[List[pt.Tensor], pt.Tensor]:
        """Load a first-axis slice directly from processor NumPy arrays."""
        check_list_or_str(field_name, "field_name")
        check_list_or_str(time, "time")
        fields = field_name if isinstance(field_name, list) else [field_name]
        times = time if isinstance(time, list) else [time]
        snapshots = [self._snapshot_at(value) for value in times]
        loaded = [
            self._load_field_slice(field, snapshots, spatial_slice) for field in fields
        ]
        if isinstance(time, list) and len(time) == 1:
            loaded = [value.unsqueeze(-1) for value in loaded]
        return loaded if isinstance(field_name, list) else loaded[0]

    def snapshot_shape(self, field_name: str, time: str) -> tuple[int, ...]:
        """Return a field shape from batch metadata."""
        snapshot = self._snapshot_at(time)
        dataset = self._dataset(snapshot.batch)
        if field_name not in dataset.field_shapes:
            raise ValueError(f"Field {field_name!r} is not available at time {time!r}")
        return (sum(dataset.processor_sizes), *dataset.field_shapes[field_name])

    @property
    def zone_names(self) -> List[str]:
        """Cell zones exported by ``foamToNumpy`` in metadata order."""
        return list(self._zone_names)

    @property
    def zone(self) -> Union[str, None]:
        """Currently selected cell zone, or ``None`` for whole-mesh output."""
        return self._zone

    @zone.setter
    def zone(self, value: str) -> None:
        """Select the cell zone used by subsequent field and geometry reads."""
        if not self._zone_names:
            raise ValueError("A whole-mesh NumPy dataset has no selectable cell zones")
        if value not in self._zone_names:
            raise ValueError(
                f"Cell zone {value!r} not found; available zones are "
                f"{list(self._zone_names)}"
            )
        self._zone = value

    @property
    def write_times(self) -> List[str]:
        """Committed output times after applying segment precedence."""
        return self._write_times.copy()

    @property
    def field_names(self) -> Dict[str, List[str]]:
        """Available volume fields for each committed output time."""
        return {
            time: list(snapshot.batch.fields)
            for time, snapshot in zip(self._write_times, self._snapshots)
        }

    def _load_geometry(self, name: str, components: int) -> pt.Tensor:
        candidates: List[Tuple[Path, _Dataset]] = []
        seen = set()
        for snapshot in self._snapshots:
            batch = snapshot.batch
            key = (batch.path.parent, batch.mesh_revision)
            if key in seen:
                continue
            seen.add(key)
            geometry_path = batch.path.parent / f"geometry_{batch.mesh_revision:06d}"
            if self._zone is not None:
                geometry_path = geometry_path / "cellZones" / self._zone
            if geometry_path.is_dir():
                candidates.append((geometry_path, self._dataset(batch)))

        loaded = []
        for geometry_path, dataset in candidates:
            pattern = re.compile(rf"{re.escape(name)}_proc_(\d+)\.npy$")
            indexed_files = sorted(
                (
                    (int(match.group(1)), child)
                    for child in geometry_path.iterdir()
                    if child.is_file()
                    and (match := pattern.fullmatch(child.name)) is not None
                ),
                key=lambda item: item[0],
            )
            if not indexed_files:
                continue
            if tuple(index for index, _ in indexed_files) != tuple(
                range(len(dataset.processor_sizes))
            ):
                raise ValueError(f"Invalid processor set in {geometry_path}")

            processor_values = []
            for processor, (_, path) in enumerate(indexed_files):
                array = self._load_array(
                    path,
                    mmap=True,
                    require_fortran_header=self._zone is not None,
                )
                expected_shape = (
                    (dataset.processor_sizes[processor], 1)
                    if components == 1
                    else (dataset.processor_sizes[processor], components, 1)
                )
                if array.shape != expected_shape:
                    raise ValueError(
                        f"Invalid geometry shape {array.shape} in {path}; "
                        f"expected {expected_shape}"
                    )
                values = np.array(array[..., 0], copy=True, order="C")
                processor_values.append(pt.as_tensor(values, dtype=self._dtype))
            loaded.append(pt.cat(processor_values, dim=0))

        if not loaded:
            option = "writeCellCentres" if name == "cellCentres" else "writeCellVolumes"
            raise NotImplementedError(
                f"{name} were not exported; enable {option} in foamToNumpy"
            )
        reference = loaded[0]
        if any(
            value.shape != reference.shape or not pt.allclose(value, reference)
            for value in loaded[1:]
        ):
            raise ValueError(f"{name} change across the selected segments")
        return reference

    @property
    def vertices(self) -> pt.Tensor:
        """Cell-centre coordinates for the active dataset in processor order."""
        if self._zone not in self._vertices:
            self._vertices[self._zone] = self._load_geometry("cellCentres", 3)
        return self._vertices[self._zone]

    @property
    def weights(self) -> pt.Tensor:
        """Cell volumes for the active dataset in processor order."""
        if self._zone not in self._weights:
            self._weights[self._zone] = self._load_geometry("cellVolumes", 1)
        return self._weights[self._zone]
