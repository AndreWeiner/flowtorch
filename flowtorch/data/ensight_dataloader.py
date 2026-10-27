"""Load static EnSight Gold volume and surface time series with VTK."""

from pathlib import Path
import re
from typing import Dict, List, Optional, Union

import numpy as np
import torch as pt
from vtkmodules.vtkCommonDataModel import vtkCompositeDataSet
from vtkmodules.vtkFiltersCore import vtkCellCenters
from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter
from vtkmodules.vtkIOEnSight import vtkGenericEnSightReader
from vtkmodules.util.numpy_support import vtk_to_numpy

from flowtorch import DEFAULT_DTYPE
from ._ensight_slice import COMPONENTS, EnSightSliceReader
from .dataloader import Dataloader
from .utils import check_list_or_str


class EnSightDataloader(Dataloader):
    """Read a named part from an ASCII or C-binary EnSight Gold case.

    ``part`` is an EnSight part name, e.g. ``internalMesh``, ``movingWall`` or
    ``geometry`` for a sampled plane. It may be omitted for a single-part case.
    Parts are physical regions/surfaces, not OpenFOAM processor partitions.
    OpenFOAM's assembled parallel exports are supported without reconstruction.

    Cell fields use VTK parametric cell centers as :attr:`vertices` (vertex
    averages for arbitrary polygons) and volumes
    (3D) or areas (2D) as :attr:`weights`. Parametric centers may differ from
    OpenFOAM finite-volume centroids on irregular elements. Point fields use
    mesh points and conservative lumped weights: each element distributes its
    measure equally among its nodes. Geometry is cached on first access.

    Scalars have shape ``(N,)``, vectors ``(N,3)``, symmetric tensors ``(N,6)``
    in OpenFOAM order ``xx,xy,xz,yy,yz,zz``, and full tensors ``(N,9)`` in
    row-major order. Time series append time as the last dimension.
    Element order is the EnSight/VTK order, including mixed element blocks;
    it need not match OpenFOAM numbering or another decomposition's export.

    Static unstructured geometry, one time set and separate snapshot files
    are supported. Moving geometry, file sets, measured/cloud data, complex
    variables and mixed-dimensional parts are rejected explicitly.
    Requires the optional ``vtk`` dependency (``flowtorch-fluid[data]``).

    Examples
    --------
    >>> loader = EnSightDataloader("postProcessing/volume_binary/cavity.case",
    ...                            part="internalMesh")
    >>> p, U = loader.load_snapshot(["p", "U"], loader.write_times)
    >>> wall = EnSightDataloader("postProcessing/patch_binary/cavity.case",
    ...                          part="movingWall")
    >>> plane = EnSightDataloader("postProcessing/cuts_binary/oblique/oblique.case")

    :param case_file: path to an EnSight Gold .case descriptor
    :param part: named part; required if the descriptor contains multiple parts
    :param association: ``"cell"`` (default) or ``"point"``
    :param dtype: floating-point torch dtype for returned tensors
    :param slice_backend: ``"native"`` (default) or ``"vtk"`` for direct dense
        C-binary field slices; full snapshots and geometry always use VTK
    """

    def __init__(
        self,
        case_file: Union[str, Path],
        part: Optional[str] = None,
        association: str = "cell",
        dtype: pt.dtype = DEFAULT_DTYPE,
        slice_backend: str = "native",
    ):
        self._case_file = Path(case_file).expanduser().resolve()
        if not self._case_file.is_file():
            raise FileNotFoundError(f"EnSight case does not exist: {self._case_file}")
        if association not in ("cell", "point"):
            raise ValueError("association must be 'cell' or 'point'")
        if part is not None and not isinstance(part, str):
            raise ValueError("part must be a part name, not a block index or part ID")
        if (
            not isinstance(dtype, pt.dtype)
            or not pt.empty((), dtype=dtype).is_floating_point()
        ):
            raise ValueError("dtype must be a floating-point torch dtype")
        if slice_backend not in ("vtk", "native"):
            raise ValueError("slice_backend must be 'vtk' or 'native'")
        self._slice_backend = slice_backend
        self._slice_reader: Optional[EnSightSliceReader] = None
        self._association = association
        self._dtype = dtype
        self._vertices = None
        self._weights = None
        self._field_names: Optional[Dict[str, List[str]]] = None
        self._parse_case()
        self._reader = vtkGenericEnSightReader()
        self._reader.SetCaseFileName(str(self._case_file))
        self._reader.ReadAllVariablesOn()
        self._errors: List[str] = []
        self._reader.AddObserver("ErrorEvent", self._on_error)
        self._update(information=True)
        implementation = self._reader.GetReader()
        if implementation is not None:
            implementation.AddObserver("ErrorEvent", self._on_error)
        time_sets = self._reader.GetTimeSets()
        if self._transient:
            if time_sets is None or time_sets.GetNumberOfItems() != 1:
                raise ValueError("Expected a single EnSight time set")
            vtk_times = vtk_to_numpy(time_sets.GetItem(0))
            if len(vtk_times) != len(self._write_times) or not np.allclose(
                vtk_times, [float(t) for t in self._write_times], rtol=1e-6, atol=0
            ):
                raise ValueError("VTK time metadata does not match the case descriptor")
        # Read geometry before fields: invalid geometry must not be passed to
        # VTK's variable readers, which assume its parts have been created.
        self._reader.ReadAllVariablesOff()
        self._reader.GetCellDataArraySelection().DisableAllArrays()
        self._reader.GetPointDataArraySelection().DisableAllArrays()
        self._read(self._write_times[0])
        output = self._reader.GetOutput()
        self._parts = self._part_names(output)
        if part is None:
            if len(self._parts) != 1:
                raise ValueError(
                    f"Select a part by name; available parts: {self._parts}"
                )
            part = self._parts[0]
        if part not in self._parts:
            raise ValueError(f"Unknown part {part!r}; available parts: {self._parts}")
        self._part = part
        grid = self._selected_grid()
        self._geometry = grid.NewInstance()
        structure = grid.NewInstance()
        structure.CopyStructure(grid)
        self._geometry.DeepCopy(structure)
        types = self._cell_types(grid)
        dimensions = {
            grid.GetCell(int(np.flatnonzero(types == t)[0])).GetCellDimension()
            for t in np.unique(types)
        }
        if len(dimensions) != 1 or not dimensions.issubset({2, 3}):
            raise ValueError(
                "The selected part must contain only 2D or only 3D elements"
            )
        self._dimension = dimensions.pop()
        if slice_backend == "native" and self._geometry_is_binary:
            count = (
                grid.GetNumberOfCells()
                if association == "cell"
                else grid.GetNumberOfPoints()
            )
            self._field_shapes = {
                name: (count,) if COMPONENTS[kind] == 1 else (count, COMPONENTS[kind])
                for (assoc, name), (kind, _) in self._variables.items()
                if assoc == association
            }
        else:
            self._enable_fields()
            self._read(self._write_times[0])
            attributes = self._attributes()
            self._field_shapes = {
                name: tuple(self._array(name).shape)
                for assoc, name in self._variables
                if assoc == association and attributes.GetArray(name) is not None
            }

    def _native_slice_reader(self) -> EnSightSliceReader:
        if self._slice_reader is None:
            # VTK exposes topology but not original part IDs/block offsets.
            reader = EnSightSliceReader(
                self._geometry_file, self._reader.GetReader().GetByteOrderAsString()
            )
            output = self._reader.GetOutput()
            grids = {
                output.GetMetaData(i).Get(vtkCompositeDataSet.NAME()): output.GetBlock(
                    i
                )
                for i in range(output.GetNumberOfBlocks())
                if output.GetBlock(i) is not None
            }
            reader.validate(grids, self._cell_types)
            self._slice_reader = reader
        return self._slice_reader

    def _enable_fields(self):
        self._reader.ReadAllVariablesOn()
        self._reader.GetCellDataArraySelection().EnableAllArrays()
        self._reader.GetPointDataArraySelection().EnableAllArrays()

    def _field_path(self, field, time):
        if time not in self._numbers:
            raise ValueError(
                f"Unknown time {time!r}; available times: {self._write_times}"
            )
        if (self._association, field) not in self._variables:
            raise ValueError(f"Unknown {self._association} field {field!r}")
        _, pattern = self._variables[self._association, field]
        filename = re.sub(
            r"\*+", lambda m: str(self._numbers[time]).zfill(len(m[0])), pattern
        )
        path = self._case_file.parent / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing EnSight field file at time {time!r}: {path}"
            )
        return path

    def load_snapshot_slice(
        self,
        field_name: Union[List[str], str],
        time: Union[List[str], str],
        spatial_slice: slice,
    ) -> Union[List[pt.Tensor], pt.Tensor]:
        """Read a spatial slice, using direct C-binary reads by default.

        Native reads support unit stride and never invoke VTK field loading.
        Unsupported binary sections raise; select ``slice_backend="vtk"`` to
        use the full-read fallback. Geometry remains resident in VTK.
        """
        if self._slice_backend == "vtk":
            return super().load_snapshot_slice(field_name, time, spatial_slice)
        check_list_or_str(field_name, "field_name")
        check_list_or_str(time, "time")
        if not isinstance(spatial_slice, slice):
            raise ValueError("spatial_slice must be a slice with unit stride")
        fields = field_name if isinstance(field_name, list) else [field_name]
        times = time if isinstance(time, list) else [time]
        outputs = []
        for field in fields:
            count = self.snapshot_shape(field, times[0])[0]
            start, stop, step = spatial_slice.indices(count)
            if step != 1:
                raise ValueError("spatial_slice must have unit stride")
            stop = max(start, stop)
            kind = self._variables[self._association, field][0]
            output = None
            for index, t in enumerate(times):
                snapshot = pt.tensor(
                    self._native_slice_reader().read(
                        self._field_path(field, t),
                        self._part,
                        kind,
                        self._association,
                        start,
                        stop,
                    ),
                    dtype=self._dtype,
                )
                if isinstance(time, list):
                    if output is None:
                        output = snapshot.new_empty((*snapshot.shape, len(times)))
                    output[..., index].copy_(snapshot)
                else:
                    output = snapshot
            outputs.append(output)
        return outputs if isinstance(field_name, list) else outputs[0]

    def _parse_case(self):
        raw_sections: Dict[str, List[List[str]]] = {}
        section = None
        for raw in self._case_file.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line in ("FORMAT", "GEOMETRY", "VARIABLE", "TIME", "FILE"):
                section = line
                raw_sections.setdefault(section, []).append([])
            elif section is not None:
                raw_sections[section][-1].append(line)
        if any(len(entries) != 1 for entries in raw_sections.values()):
            raise ValueError("Repeated EnSight sections are not supported")
        sections = {name: entries[0] for name, entries in raw_sections.items()}
        if sections.get("FORMAT") != ["type: ensight gold"]:
            raise ValueError("Only EnSight Gold descriptors are supported")
        if "FILE" in sections:
            raise ValueError(
                "EnSight file sets are not supported; use separate snapshot files"
            )
        geometry = sections.get("GEOMETRY", [])
        if len(geometry) != 1 or not geometry[0].startswith("model:"):
            raise ValueError(
                "Expected one geometry model; measured/cloud data are unsupported"
            )
        model = geometry[0].split(":", 1)[1].split()
        if len(model) != 1 or "*" in model[0]:
            raise ValueError(
                "Time-dependent geometry is unsupported; export static geometry"
            )
        self._geometry_file = self._case_file.parent / model[0]
        if not self._geometry_file.is_file():
            raise FileNotFoundError(f"Missing EnSight geometry: {self._geometry_file}")
        self._validate_geometry_header()
        self._transient = "TIME" in sections
        time_metadata = {}
        key = None
        for line in sections.get("TIME", []):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                if key in time_metadata:
                    raise ValueError("Multiple EnSight time sets are not supported")
                time_metadata[key] = value.split()
            elif key:
                time_metadata[key].extend(line.split())
        if self._transient:
            try:
                time_set = int(time_metadata["time set"][0])
                steps = int(time_metadata["number of steps"][0])
                labels = time_metadata["time values"]
                times = np.array([float(t) for t in labels])
                if steps != len(labels) or steps < 1 or not np.isfinite(times).all():
                    raise ValueError("Invalid time values")
                if len(set(times)) != steps or np.any(np.diff(times) <= 0):
                    raise ValueError("Time values must be unique and increasing")
                if "filename numbers" in time_metadata:
                    numbers = [int(n) for n in time_metadata["filename numbers"]]
                else:
                    start = int(time_metadata.get("filename start number", ["0"])[0])
                    increment = int(time_metadata.get("filename increment", ["1"])[0])
                    numbers = [start + i * increment for i in range(steps)]
                if len(numbers) != steps or len(set(numbers)) != steps:
                    raise ValueError("Invalid filename numbers")
            except (KeyError, IndexError, ValueError) as error:
                raise ValueError(f"Invalid EnSight time metadata: {error}") from error
        else:
            time_set, labels, numbers = None, ["0"], [0]
        self._write_times = list(labels)
        self._numbers = dict(zip(labels, numbers))
        self._variables: Dict[tuple[str, str], tuple[str, str]] = {}
        for line in sections.get("VARIABLE", []):
            match = re.fullmatch(
                r"(scalar|vector|tensor symm|tensor asym) per (node|element):\s*(.+)",
                line,
            )
            if match is None:
                raise ValueError(f"Unsupported EnSight variable declaration: {line}")
            kind, location, declaration = match.groups()
            tokens = declaration.split()
            ids = []
            while tokens and re.fullmatch(r"\d+", tokens[0]):
                ids.append(int(tokens.pop(0)))
            if len(ids) > 1 or (ids and ids[0] != time_set):
                raise ValueError(
                    "Variable time/file sets must match the single case time set"
                )
            if len(tokens) < 2:
                raise ValueError(f"Invalid EnSight variable declaration: {line}")
            name, pattern = " ".join(tokens[:-1]), tokens[-1]
            assoc = "point" if location == "node" else "cell"
            if (assoc, name) in self._variables:
                raise ValueError(f"Duplicate EnSight variable: {name}")
            if "*" in pattern and not ids:
                raise ValueError("Transient variable filenames require a time set")
            self._variables[assoc, name] = (kind, pattern)

    def _validate_geometry_header(self):
        with self._geometry_file.open("rb") as stream:
            first = stream.read(80)
            self._geometry_is_binary = first.rstrip(b"\x00 ").lower() == b"c binary"
            if self._geometry_is_binary:
                header = [stream.read(80).rstrip(b"\x00 ") for _ in range(4)]
            else:
                stream.seek(0)
                header = [stream.readline().strip() for _ in range(4)]
        if not (
            header[2].startswith(b"node id ") and header[3].startswith(b"element id ")
        ):
            raise ValueError(
                f"Invalid EnSight Gold geometry header: {self._geometry_file}"
            )

    @staticmethod
    def _cell_types(grid):
        # VTK 9.6 renamed this accessor; retain compatibility with older wheels.
        try:
            array = grid.GetCellTypes()
        except TypeError:
            array = grid.GetCellTypesArray()
        return vtk_to_numpy(array)

    def _on_error(self, caller, event):
        self._errors.append(event)

    def _update(self, information=False):
        self._errors.clear()
        if information:
            self._reader.UpdateInformation()
        else:
            self._reader.Update()
        if self._errors:
            raise ValueError(f"VTK could not read EnSight data: {self._case_file}")

    def _read(self, time):
        if time not in self._numbers:
            raise ValueError(
                f"Unknown time {time!r}; available times: {self._write_times}"
            )
        for kind, pattern in self._variables.values():
            filename = re.sub(
                r"\*+", lambda m: str(self._numbers[time]).zfill(len(m[0])), pattern
            )
            path = self._case_file.parent / filename
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing EnSight field file at time {time!r}: {path}"
                )
        if self._transient:
            self._reader.SetTimeValue(float(time))
        self._update()

    @staticmethod
    def _part_names(output):
        names = []
        for index in range(output.GetNumberOfBlocks()):
            grid = output.GetBlock(index)
            if grid is None:
                continue
            if not grid.IsA("vtkUnstructuredGrid"):
                raise ValueError("Only unstructured EnSight parts are supported")
            name = output.GetMetaData(index).Get(vtkCompositeDataSet.NAME())
            if not name or name in names:
                raise ValueError("EnSight parts must have unique, nonempty names")
            names.append(name)
        return names

    def _selected_grid(self):
        output = self._reader.GetOutput()
        self._part_names(output)
        for index in range(output.GetNumberOfBlocks()):
            if output.GetMetaData(index).Get(vtkCompositeDataSet.NAME()) == self._part:
                return output.GetBlock(index)
        raise ValueError(f"Part {self._part!r} is absent from this snapshot")

    def _attributes(self):
        grid = self._selected_grid()
        return (
            grid.GetCellData() if self._association == "cell" else grid.GetPointData()
        )

    def _array(self, field):
        if (self._association, field) not in self._variables:
            raise ValueError(f"Unknown {self._association} field {field!r}")
        array = self._attributes().GetArray(field)
        if array is None:
            raise ValueError(f"Field {field!r} is absent from part {self._part!r}")
        expected = {"scalar": 1, "vector": 3, "tensor symm": 6, "tensor asym": 9}
        kind = self._variables[self._association, field][0]
        if array.GetNumberOfComponents() != expected[kind]:
            raise ValueError(f"Unexpected component count for field {field!r}")
        count = (
            self._geometry.GetNumberOfCells()
            if self._association == "cell"
            else self._geometry.GetNumberOfPoints()
        )
        if array.GetNumberOfTuples() != count:
            raise ValueError(f"Field {field!r} does not match the selected geometry")
        values = vtk_to_numpy(array)
        if kind == "tensor symm":
            values = values[:, [0, 3, 5, 1, 4, 2]]
        return values

    def load_snapshot(
        self, field_name: Union[List[str], str], time: Union[List[str], str]
    ) -> Union[List[pt.Tensor], pt.Tensor]:
        check_list_or_str(field_name, "field_name")
        check_list_or_str(time, "time")
        fields = field_name if isinstance(field_name, list) else [field_name]
        times = time if isinstance(time, list) else [time]
        for t in times:
            if t not in self._numbers:
                raise ValueError(
                    f"Unknown time {t!r}; available times: {self._write_times}"
                )
        self._enable_fields()
        outputs = None
        for index, t in enumerate(times):
            self._read(t)
            snapshots = [
                pt.tensor(self._array(field), dtype=self._dtype) for field in fields
            ]
            if not isinstance(time, list):
                outputs = snapshots
            elif outputs is None:
                outputs = [
                    snapshot.new_empty((*snapshot.shape, len(times)))
                    for snapshot in snapshots
                ]
            if isinstance(time, list):
                for output, snapshot in zip(outputs, snapshots):
                    if output.shape[:-1] != snapshot.shape:
                        raise ValueError(
                            "Snapshot shapes changed during the time series"
                        )
                    output[..., index].copy_(snapshot)
        assert outputs is not None  # Lists are validated as nonempty above.
        return outputs if isinstance(field_name, list) else outputs[0]

    def snapshot_shape(self, field_name: str, time: str):
        if time not in self._numbers:
            raise ValueError(
                f"Unknown time {time!r}; available times: {self._write_times}"
            )
        if field_name not in self._field_shapes:
            raise ValueError(f"Unknown {self._association} field {field_name!r}")
        if self._slice_backend == "native" and self._geometry_is_binary:
            kind = self._variables[self._association, field_name][0]
            index = self._native_slice_reader().index(
                self._field_path(field_name, time), COMPONENTS[kind], self._association
            )
            if self._part not in index:
                raise ValueError(
                    f"Field {field_name!r} is absent from part {self._part!r}"
                )
        return self._field_shapes[field_name]

    @property
    def write_times(self) -> List[str]:
        return list(self._write_times)

    @property
    def case_file(self) -> Path:
        """Source EnSight descriptor path."""
        return self._case_file

    @property
    def part(self) -> str:
        """Selected part name."""
        return self._part

    @property
    def association(self) -> str:
        """Field association: ``cell`` or ``point``."""
        return self._association

    @property
    def mesh(self):
        """Return an independent VTK mesh copy, including connectivity/faces.

        This is the actual mesh, unlike :attr:`vertices`, which gives field
        locations. The returned mesh contains no field arrays; editing it does
        not change the loader. Requires the same optional VTK dependency.
        """
        mesh = self._geometry.NewInstance()
        mesh.DeepCopy(self._geometry)
        return mesh

    @property
    def parts(self) -> List[str]:
        """Available part names; numeric EnSight IDs are not block indices."""
        return list(self._parts)

    @property
    def field_names(self) -> Dict[str, List[str]]:
        if self._field_names is None:
            self._field_names = {}
            for time in self._write_times:
                if self._slice_backend == "native" and self._geometry_is_binary:
                    names = []
                    for (assoc, name), (kind, _) in self._variables.items():
                        if (
                            assoc == self._association
                            and self._part
                            in self._native_slice_reader().index(
                                self._field_path(name, time), COMPONENTS[kind], assoc
                            )
                        ):
                            names.append(name)
                    self._field_names[time] = names
                    continue
                self._read(time)
                attributes = self._attributes()
                self._field_names[time] = [
                    name
                    for assoc, name in self._variables
                    if assoc == self._association
                    and attributes.GetArray(name) is not None
                ]
        return {time: list(names) for time, names in self._field_names.items()}

    def _load_geometry(self):
        if self._vertices is not None:
            return
        sizes = vtkCellSizeFilter()
        sizes.SetInputData(self._geometry)
        sizes.Update()
        measure = "Area" if self._dimension == 2 else "Volume"
        weights = vtk_to_numpy(sizes.GetOutput().GetCellData().GetArray(measure))
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError(
                "The selected geometry has nonpositive or invalid element measures"
            )
        if self._association == "cell":
            centers = vtkCellCenters()
            centers.SetInputData(self._geometry)
            centers.Update()
            vertices = vtk_to_numpy(centers.GetOutput().GetPoints().GetData()).copy()
            # vtkPolygon's parametric orientation can change under tiny rounding
            # differences. Vertex means are stable, fast representative points.
            polygon_ids = np.flatnonzero(self._cell_types(self._geometry) == 7)
            if len(polygon_ids):
                cells = self._geometry.GetCells()
                offsets = vtk_to_numpy(cells.GetOffsetsArray())
                connectivity = vtk_to_numpy(cells.GetConnectivityArray())
                counts = np.diff(offsets)[polygon_ids]
                starts = np.cumsum(counts) - counts
                indices = (
                    np.repeat(offsets[polygon_ids], counts)
                    + np.arange(counts.sum())
                    - np.repeat(starts, counts)
                )
                points = vtk_to_numpy(self._geometry.GetPoints().GetData())
                vertices[polygon_ids] = (
                    np.add.reduceat(points[connectivity[indices]], starts, axis=0)
                    / counts[:, None]
                )
        else:
            vertices = vtk_to_numpy(self._geometry.GetPoints().GetData())
            cells = self._geometry.GetCells()
            offsets = vtk_to_numpy(cells.GetOffsetsArray())
            connectivity = vtk_to_numpy(cells.GetConnectivityArray())
            counts = np.diff(offsets)
            weights = np.bincount(
                connectivity,
                weights=np.repeat(weights / counts, counts),
                minlength=len(vertices),
            )
        self._vertices = pt.tensor(vertices, dtype=self._dtype)
        self._weights = pt.tensor(weights, dtype=self._dtype)

    @property
    def vertices(self) -> pt.Tensor:
        self._load_geometry()
        return self._vertices

    @property
    def weights(self) -> pt.Tensor:
        self._load_geometry()
        return self._weights
