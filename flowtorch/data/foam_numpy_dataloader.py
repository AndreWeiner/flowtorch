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
class _Batch:
    """Validated metadata and files belonging to one output batch."""

    path: Path
    times: Tuple[float, ...]
    mesh_revision: int
    fields: Tuple[str, ...]
    field_classes: Dict[str, str]
    field_files: Dict[str, Tuple[Path, ...]]
    field_shapes: Dict[str, Tuple[int, ...]]
    processor_sizes: Tuple[int, ...]


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

    :param path: function-object output directory or one segment directory
    :param ignore_segments: segment name or names to exclude from the catalog
    :param dtype: output tensor dtype, defaults to ``torch.float32``
    """

    def __init__(
        self,
        path: str,
        ignore_segments: Union[List[str], str, None] = None,
        dtype: pt.dtype = DEFAULT_DTYPE,
    ):
        self._path = Path(path)
        if not self._path.is_dir():
            raise ValueError(f"Directory does not exist: {path}")
        self._dtype = dtype
        self._segment_paths = self._resolve_segments(ignore_segments)
        self._batches: List[_Batch] = []
        snapshots: List[_Snapshot] = []

        for segment_path in self._segment_paths:
            segment_batches = self._load_segment(segment_path)
            self._batches.extend(segment_batches)
            for batch in segment_batches:
                snapshots.extend(
                    _Snapshot(time, batch, index)
                    for index, time in enumerate(batch.times)
                )

        self._snapshots = self._normalise_snapshots(snapshots)
        if not self._snapshots:
            raise ValueError("No committed NumPy snapshots were found")

        self._processor_sizes = self._snapshots[0].batch.processor_sizes
        for snapshot in self._snapshots[1:]:
            if snapshot.batch.processor_sizes != self._processor_sizes:
                raise ValueError(
                    "Processor cell counts change across the selected snapshots; "
                    "dynamic meshes or changed decompositions are not supported"
                )

        self._write_times = [
            _format_time(snapshot.time) for snapshot in self._snapshots
        ]
        self._vertices: Union[pt.Tensor, None] = None
        self._weights: Union[pt.Tensor, None] = None

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

    def _load_segment(self, segment_path: Path) -> List[_Batch]:
        info_path = segment_path / "segmentInfo"
        info = info_path.read_text()
        order = _optional_entry(info, "order")
        if order is not None and order != "fortran":
            raise ValueError(
                f"Unsupported NumPy storage order {order!r} in {info_path}"
            )

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
            batch = self._load_batch(batch_path, state_path)
            if batch is not None:
                batches.append(batch)
                revisions.add(batch.mesh_revision)

        if len(revisions) > 1:
            raise ValueError(
                f"Segment {segment_path.name!r} contains multiple mesh revisions; "
                "dynamic meshes are not supported"
            )
        return batches

    def _load_batch(self, batch_path: Path, state_path: Path) -> Union[_Batch, None]:
        try:
            state = state_path.read_text()
            count = int(_entry(state, "count"))
            mesh_revision = int(_entry(state, "meshRevision"))
            fields = _word_list(state, "fields")
            classes = _field_classes(state)
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

        times_path = batch_path / "times.npy"
        times_array = self._load_array(times_path)
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

        field_files: Dict[str, Tuple[Path, ...]] = {}
        field_shapes: Dict[str, Tuple[int, ...]] = {}
        expected_processor_ids: Union[Tuple[int, ...], None] = None
        processor_sizes: Union[Tuple[int, ...], None] = None

        for field in fields:
            field_class = classes[field]
            if field_class not in _FIELD_COMPONENTS:
                raise ValueError(
                    f"Unsupported field class {field_class!r} for {field!r} in "
                    f"{state_path}"
                )
            pattern = re.compile(rf"{re.escape(field)}_proc_(\d+)\.npy$")
            indexed_files = sorted(
                (
                    (int(match.group(1)), child)
                    for child in batch_path.iterdir()
                    if child.is_file()
                    and (match := pattern.fullmatch(child.name)) is not None
                ),
                key=lambda item: item[0],
            )
            if not indexed_files:
                raise ValueError(
                    f"No processor arrays found for {field!r} in {batch_path}"
                )
            processor_ids = tuple(index for index, _ in indexed_files)
            if processor_ids != tuple(range(len(processor_ids))):
                raise ValueError(
                    f"Processor indices for {field!r} are not contiguous in {batch_path}"
                )
            if expected_processor_ids is None:
                expected_processor_ids = processor_ids
            elif processor_ids != expected_processor_ids:
                raise ValueError(
                    f"Processor sets differ between fields in {batch_path}"
                )

            files = tuple(path for _, path in indexed_files)
            arrays = [self._load_array(path, mmap=True) for path in files]
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
                    f"Processor cell counts differ between fields in {batch_path}"
                )

            component_shape = tuple(arrays[0].shape[1:-1])
            if any(tuple(array.shape[1:-1]) != component_shape for array in arrays):
                raise ValueError(
                    f"Component shapes differ for {field!r} in {batch_path}"
                )
            field_files[field] = files
            field_shapes[field] = component_shape

        assert processor_sizes is not None
        return _Batch(
            batch_path,
            times,
            mesh_revision,
            fields,
            classes,
            field_files,
            field_shapes,
            processor_sizes,
        )

    @staticmethod
    def _load_array(path: Path, mmap: bool = False) -> np.ndarray:
        if not path.is_file():
            raise ValueError(f"Missing NumPy array {path}")
        try:
            array = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot load NumPy array {path}: {error}") from error
        if array.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError(f"Unsupported dtype {array.dtype} in {path}")
        if not array.flags.f_contiguous:
            raise ValueError(f"Array is not Fortran-contiguous: {path}")
        return array

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
            if field not in snapshot.batch.field_files:
                raise ValueError(
                    f"Field {field!r} is not available at time "
                    f"{_format_time(snapshot.time)!r}"
                )

        component_shape = snapshots[0].batch.field_shapes[field]
        if any(
            snapshot.batch.field_shapes[field] != component_shape
            for snapshot in snapshots[1:]
        ):
            raise ValueError(f"Component shape changes across snapshots for {field!r}")

        n_cells = sum(self._processor_sizes)
        result = pt.empty(
            (n_cells, *component_shape, len(snapshots)), dtype=self._dtype
        )
        grouped: Dict[Path, List[Tuple[int, _Snapshot]]] = {}
        for output_index, snapshot in enumerate(snapshots):
            grouped.setdefault(snapshot.batch.path, []).append((output_index, snapshot))

        for entries in grouped.values():
            batch = entries[0][1].batch
            offset = 0
            for processor_size, path in zip(
                batch.processor_sizes, batch.field_files[field]
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
        candidates = []
        for segment_path in self._segment_paths:
            revision = next(
                (
                    batch.mesh_revision
                    for batch in self._batches
                    if batch.path.parent == segment_path
                ),
                None,
            )
            if revision is None:
                continue
            geometry_path = segment_path / f"geometry_{revision:06d}"
            if geometry_path.is_dir():
                candidates.append(geometry_path)

        loaded = []
        for geometry_path in candidates:
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
                range(len(self._processor_sizes))
            ):
                raise ValueError(f"Invalid processor set in {geometry_path}")

            processor_values = []
            for processor, (_, path) in enumerate(indexed_files):
                array = self._load_array(path, mmap=True)
                expected_shape = (
                    (self._processor_sizes[processor], 1)
                    if components == 1
                    else (self._processor_sizes[processor], components, 1)
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
        """Cell-centre coordinates concatenated in processor order."""
        if self._vertices is None:
            self._vertices = self._load_geometry("cellCentres", 3)
        return self._vertices

    @property
    def weights(self) -> pt.Tensor:
        """Cell volumes concatenated in processor order."""
        if self._weights is None:
            self._weights = self._load_geometry("cellVolumes", 1)
        return self._weights
