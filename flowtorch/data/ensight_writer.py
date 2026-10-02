"""Write static fields and time series on an EnSight or Tecplot mesh."""

from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import torch as pt
from vtkmodules.util.numpy_support import vtk_to_numpy  # type: ignore[import-not-found]

from .ensight_dataloader import EnSightDataloader
from .tecplot_dataloader import TecplotDataloader

# VTK type -> EnSight element name and fixed node count (None for ragged cells).
_ELEMENT_TYPES = {
    5: ("tria3", 3),
    7: ("nsided", None),
    9: ("quad4", 4),
    10: ("tetra4", 4),
    12: ("hexa8", 8),
    13: ("penta6", 6),
    14: ("pyramid5", 5),
    42: ("nfaced", None),
}
_KINDS = {1: "scalar", 3: "vector", 6: "tensor symm", 9: "tensor asym"}


def _ragged_rows(offsets, connectivity, rows):
    counts = np.diff(offsets)[rows]
    starts = np.cumsum(counts) - counts
    indices = (
        np.repeat(offsets[rows], counts)
        + np.arange(counts.sum())
        - np.repeat(starts, counts)
    )
    return counts, connectivity[indices]


class EnSightWriter:
    """Write a selected part to a separate, self-contained EnSight Gold dataset.

    Create with :meth:`from_loader` or :meth:`from_case`. The output directory
    must be new or empty and must not contain any source data. Its static mesh
    is copied once from an EnSight part or Tecplot zone; source files and loader
    tensors are never modified. The selected part keeps its name but receives
    output part ID 1.

    ``write_static`` writes a static field. ``write_snapshot`` writes a complete
    snapshot mapping, and ``write_series`` accepts tensors with time last.
    Supported snapshot shapes are ``(N,)``, ``(N,3)``, ``(N,6)`` and ``(N,9)``;
    symmetric tensors follow the loader's OpenFOAM component order. Fields must
    already have physical shapes/units; analysis normalization and spatial
    weighting are not undone implicitly. Complex values are written as separate
    ``NAME_real`` and ``NAME_imag`` fields by default, never silently discarded.

    Binary output uses little-endian 32-bit integers and single-precision floats.
    ASCII uses 17 significant digits, although readers may use single precision.
    Linear triangles, quads, polygons, tetrahedra, hexahedra, wedges, pyramids and
    polyhedra are supported. Higher-order cells are rejected. One part and one
    association are supported per writer, including assembled MPI input exports.
    Moving geometry and distributed writing are outside this API's scope.
    Connectivity is preserved in the files; VTK's ASCII reader reverses nsided
    winding on readback, which may affect surface normals. Prefer binary when
    winding must round-trip exactly through VTK/ParaView.

    Each successful operation publishes the .case descriptor atomically after
    the referenced files are complete. Existing outputs are never overwritten;
    interrupted writes can leave unreferenced files, not incomplete published
    snapshots. A context manager closes the writer automatically.

    Examples
    --------
    >>> with EnSightWriter.from_case("input.case", "reconstructed",
    ...                              case_name="velocity", part="internalMesh") as writer:
    ...     writer.write_series({"U_rec": reconstructed_velocity}, times)
    >>> with EnSightWriter.from_loader(loader, "modes") as writer:
    ...     writer.write_static("U_mode_001", velocity_mode)
    """

    def __init__(
        self,
        loader: Union[EnSightDataloader, TecplotDataloader],
        output_dir: Union[str, Path],
        case_name: Optional[str] = None,
        format: str = "binary",
    ):
        if not isinstance(loader, (EnSightDataloader, TecplotDataloader)):
            raise TypeError("loader must be an EnSightDataloader or TecplotDataloader")
        if format not in ("ascii", "binary"):
            raise ValueError("format must be 'ascii' or 'binary'")
        if isinstance(loader, TecplotDataloader):
            name = "results" if case_name is None else case_name
            part = loader.zone
            protected = [
                Path(loader._assemble_file_path(time)) for time in loader.write_times
            ]
        else:
            name = loader.case_file.stem if case_name is None else case_name
            part = loader.part
            protected = [loader.case_file, loader._geometry_file]
            protected.extend(
                loader.case_file.parent / pattern
                for _, pattern in loader._variables.values()
            )
        self._validate_name(name, "case_name")
        self._validate_record(part)
        output = Path(output_dir).expanduser().resolve()
        if any(path.resolve().is_relative_to(output) for path in protected):
            raise ValueError(
                "The output directory must not contain source EnSight or Tecplot files"
            )
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise FileExistsError("The output directory must be new or empty")
        self._output_dir = output
        self._case_file = output / f"{name}.case"
        self._format = format
        self._association = loader.association
        self._part = part
        self._mesh = loader.mesh
        self._blocks = []
        types = EnSightDataloader._cell_types(self._mesh)
        unique, first = np.unique(types, return_index=True)
        for cell_type in unique[np.argsort(first)]:
            if cell_type not in _ELEMENT_TYPES:
                raise ValueError(
                    f"Unsupported VTK element type {cell_type}; only linear cells are supported"
                )
            element, nodes = _ELEMENT_TYPES[int(cell_type)]
            self._blocks.append((element, nodes, np.flatnonzero(types == cell_type)))
        self._count = (
            self._mesh.GetNumberOfCells()
            if self._association == "cell"
            else self._mesh.GetNumberOfPoints()
        )
        self._schema: Optional[dict[str, int]] = None
        self._static: dict[str, int] = {}
        self._times: list[str] = []
        self._closed = False
        output.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".ensight-", dir=output) as temporary:
            geometry = Path(temporary) / "geometry"
            self._write_geometry(geometry)
            geometry.replace(output / "geometry")

    @classmethod
    def from_loader(
        cls,
        loader: Union[EnSightDataloader, TecplotDataloader],
        output_dir: Union[str, Path],
        case_name: Optional[str] = None,
        format: str = "binary",
    ):
        """Copy an EnSight part or Tecplot zone and retain its association.

        Tecplot export uses the first snapshot's fixed geometry and applies the
        loader's cell-type override. Source fields are not copied automatically.
        The default case name for Tecplot sources is ``results``.
        """
        return cls(loader, output_dir, case_name=case_name, format=format)

    @classmethod
    def from_case(
        cls,
        case_file: Union[str, Path],
        output_dir: Union[str, Path],
        case_name: Optional[str] = None,
        part: Optional[str] = None,
        association: str = "cell",
        format: str = "binary",
    ):
        """Read an existing .case and copy its selected part to a new dataset.

        ``case_name`` defaults to the source descriptor's stem. Part selection
        follows the loader: automatic for one part, explicit for multiple parts.
        """
        loader = EnSightDataloader(case_file, part=part, association=association)
        return cls.from_loader(loader, output_dir, case_name=case_name, format=format)

    @staticmethod
    def _validate_name(name, parameter="field name"):
        leading = r"[A-Za-z0-9_]" if parameter == "case_name" else r"[A-Za-z_]"
        if (
            not isinstance(name, str)
            or re.fullmatch(leading + r"[A-Za-z0-9_.-]*", name) is None
        ):
            raise ValueError(
                f"{parameter} must be a safe name containing only letters, digits, _, . or -; field names must start with a letter/underscore"
            )
        if len(name) > 79:
            raise ValueError(f"{parameter} must not exceed 79 ASCII characters")

    @staticmethod
    def _validate_record(value):
        try:
            encoded = value.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("EnSight record names must be ASCII") from error
        if not encoded or len(encoded) > 79 or b"\n" in encoded or b"\r" in encoded:
            raise ValueError(
                "EnSight record names must be nonempty, one line, and at most 79 characters"
            )

    def _record(self, stream, value):
        encoded = value.encode("ascii")
        stream.write(
            encoded.ljust(80, b"\x00") if self._format == "binary" else encoded + b"\n"
        )

    def _integers(self, stream, values, counts=None):
        array = np.asarray(values, dtype=np.int64)
        if array.size and (array.min() < 0 or array.max() > np.iinfo(np.int32).max):
            raise ValueError("EnSight connectivity exceeds the 32-bit integer range")
        if self._format == "binary":
            array.astype("<i4").tofile(stream)
        elif counts is None:
            np.savetxt(stream, array.reshape(-1, 1), fmt="%10d")
        else:
            start = 0
            for count in counts:
                row = array[start : start + count]
                stream.write(
                    (" ".join(str(int(v)) for v in row) + "\n").encode("ascii")
                )
                start += count

    def _floats(self, stream, values):
        array = np.asarray(values)
        if self._format == "binary":
            with np.errstate(over="ignore"):
                array = array.astype("<f4")
            if not np.isfinite(array).all():
                raise ValueError(
                    "Values cannot be represented as finite EnSight float32 data"
                )
            array.tofile(stream)
        else:
            np.savetxt(stream, array.reshape(-1, 1), fmt="%.17e")

    def _polyhedron_connectivity(self, rows):
        if hasattr(self._mesh, "GetPolyhedronFaces"):
            faces = self._mesh.GetPolyhedronFaces()
            locations = self._mesh.GetPolyhedronFaceLocations()
            face_counts, face_ids = _ragged_rows(
                vtk_to_numpy(locations.GetOffsetsArray()),
                vtk_to_numpy(locations.GetConnectivityArray()),
                rows,
            )
            node_counts, nodes = _ragged_rows(
                vtk_to_numpy(faces.GetOffsetsArray()),
                vtk_to_numpy(faces.GetConnectivityArray()),
                face_ids,
            )
            return face_counts, node_counts, nodes + 1
        # Older VTK versions expose polyhedron faces only through individual cells.
        face_counts, node_counts, nodes = [], [], []
        for index in rows:
            cell = self._mesh.GetCell(int(index))
            face_counts.append(cell.GetNumberOfFaces())
            for face_index in range(cell.GetNumberOfFaces()):
                face = cell.GetFace(face_index)
                node_counts.append(face.GetNumberOfPoints())
                nodes.extend(
                    face.GetPointId(i) + 1 for i in range(face.GetNumberOfPoints())
                )
        return np.array(face_counts), np.array(node_counts), np.array(nodes)

    def _write_geometry(self, path):
        points = vtk_to_numpy(self._mesh.GetPoints().GetData())
        cells = self._mesh.GetCells()
        offsets, connectivity = vtk_to_numpy(cells.GetOffsetsArray()), vtk_to_numpy(
            cells.GetConnectivityArray()
        )
        with path.open("wb") as stream:
            if self._format == "binary":
                self._record(stream, "C Binary")
            for record in (
                "EnSight Gold geometry",
                "Written by FlowTorch",
                "node id assign",
                "element id assign",
                "part",
            ):
                self._record(stream, record)
            self._integers(stream, [1])
            self._record(stream, self._part)
            self._record(stream, "coordinates")
            self._integers(stream, [len(points)])
            self._floats(stream, points.T.reshape(-1))
            for element, fixed_nodes, rows in self._blocks:
                self._record(stream, element)
                self._integers(stream, [len(rows)])
                if element == "nfaced":
                    face_counts, node_counts, nodes = self._polyhedron_connectivity(
                        rows
                    )
                    self._integers(stream, face_counts)
                    self._integers(stream, node_counts)
                    self._integers(stream, nodes, counts=node_counts)
                else:
                    counts, nodes = _ragged_rows(offsets, connectivity, rows)
                    if fixed_nodes is not None and np.any(counts != fixed_nodes):
                        raise ValueError(f"Invalid connectivity for {element}")
                    if element == "nsided":
                        self._integers(stream, counts)
                    self._integers(stream, nodes + 1, counts=counts)

    def _prepare_fields(self, fields, complex_mode):
        if complex_mode not in ("split", "error"):
            raise ValueError("complex_mode must be 'split' or 'error'")
        if not isinstance(fields, Mapping) or not fields:
            raise ValueError("fields must be a nonempty mapping of names to tensors")
        result = {}
        for name, data in fields.items():
            self._validate_name(name)
            if not isinstance(data, pt.Tensor) or not (
                data.is_floating_point() or data.is_complex()
            ):
                raise ValueError(
                    f"Field {name!r} must be a floating-point or complex tensor"
                )
            if tuple(data.shape) != (self._count,) and not (
                data.ndim == 2
                and data.shape[0] == self._count
                and data.shape[1] in (3, 6, 9)
            ):
                raise ValueError(
                    f"Field {name!r} must have shape ({self._count},) or ({self._count},3/6/9)"
                )
            if data.is_complex():
                if complex_mode == "error":
                    raise ValueError(
                        f"Field {name!r} is complex; choose complex_mode='split'"
                    )
                pieces = {name + "_real": data.real, name + "_imag": data.imag}
            else:
                pieces = {name: data}
            for output_name, piece in pieces.items():
                self._validate_name(output_name)
                if output_name in result:
                    raise ValueError(f"Duplicate output field name {output_name!r}")
                if not pt.isfinite(piece).all().item():
                    raise ValueError(f"Field {output_name!r} contains nonfinite values")
                dtype = pt.float32 if self._format == "binary" else pt.float64
                array = piece.detach().to(device="cpu", dtype=dtype).numpy()
                if not np.isfinite(array).all():
                    raise ValueError(
                        f"Field {output_name!r} exceeds the EnSight float32 range"
                    )
                result[output_name] = array
        return result

    def _write_variable(self, path, name, values):
        if values.ndim == 1:
            values = values[:, None]
        if values.shape[1] == 6:
            # OpenFOAM xx,xy,xz,yy,yz,zz -> EnSight xx,yy,zz,xy,xz,yz.
            values = values[:, [0, 3, 5, 1, 2, 4]]
        with path.open("wb") as stream:
            self._record(stream, name)
            self._record(stream, "part")
            self._integers(stream, [1])
            if self._association == "point":
                self._record(stream, "coordinates")
                self._floats(stream, values.T.reshape(-1))
            else:
                for element, _, rows in self._blocks:
                    self._record(stream, element)
                    self._floats(stream, values[rows].T.reshape(-1))

    @staticmethod
    def _components(values):
        return 1 if values.ndim == 1 else values.shape[1]

    def _publish_case(self, static, schema, times):
        location = "element" if self._association == "cell" else "node"
        lines = [
            "FORMAT",
            "type: ensight gold",
            "",
            "GEOMETRY",
            "model: geometry",
            "",
            "VARIABLE",
        ]
        for name, components in static.items():
            lines.append(
                f"{_KINDS[components]} per {location}: {name} data/static/{name}"
            )
        for name, components in (schema or {}).items():
            lines.append(
                f"{_KINDS[components]} per {location}: 1 {name} data/********/{name}"
            )
        if times:
            lines.extend(
                [
                    "",
                    "TIME",
                    "time set: 1",
                    f"number of steps: {len(times)}",
                    "filename start number: 0",
                    "filename increment: 1",
                    "time values:",
                    *times,
                ]
            )
        with TemporaryDirectory(prefix=".ensight-", dir=self._output_dir) as temporary:
            case = Path(temporary) / self._case_file.name
            case.write_text("\n".join(lines) + "\n", encoding="ascii")
            case.replace(self._case_file)

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("The EnSight writer is closed")

    def write_static(self, name: str, data: pt.Tensor, complex_mode: str = "split"):
        """Write one time-independent field on the copied mesh.

        ``data`` must follow the selected mesh row order and association, with
        shape ``(N,)``, ``(N,3)``, ``(N,6)`` or ``(N,9)``. This accepts any
        static field, including a mode or mean; no modal processing is applied.
        Complex tensors become ``NAME_real`` and ``NAME_imag`` fields when
        ``complex_mode="split"`` (default); ``"error"`` rejects them.

        The field is added to the same .case as transient snapshots and is
        available at every output time. This call adds no timestep. Names must
        be unique across static and transient fields; existing fields cannot
        be overwritten. The descriptor is published after the files are complete.
        """
        self._ensure_open()
        prepared = self._prepare_fields({name: data}, complex_mode)
        occupied = set(self._static) | set(self._schema or {})
        if occupied.intersection(prepared):
            raise ValueError("A field with this name has already been written")
        static = dict(self._static)
        static.update(
            {name: self._components(values) for name, values in prepared.items()}
        )
        destination = self._output_dir / "data/static"
        destination.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".ensight-", dir=self._output_dir) as temporary:
            for output_name, values in prepared.items():
                staged = Path(temporary) / output_name
                self._write_variable(staged, output_name, values)
            for output_name in prepared:
                target = destination / output_name
                if target.exists():
                    raise FileExistsError(f"Output field already exists: {target}")
                (Path(temporary) / output_name).replace(target)
            self._publish_case(static, self._schema, self._times)
        self._static = static

    def write_snapshot(
        self,
        fields: Mapping[str, pt.Tensor],
        time: Union[str, float],
        complex_mode: str = "split",
    ):
        """Write one complete snapshot; times must be finite and increasing.

        All snapshots must contain the same names, shapes and component counts.
        Static fields may coexist with transient fields under different names.
        GPU tensors are detached and transferred one snapshot at a time.
        """
        self._ensure_open()
        try:
            numeric_time = float(time)
        except (ValueError, TypeError) as error:
            raise ValueError("time must be a finite numeric value") from error
        label = time.strip() if isinstance(time, str) else format(numeric_time, ".17g")
        if (
            not np.isfinite(numeric_time)
            or not label.isascii()
            or len(label) > 79
            or any(char.isspace() for char in label)
        ):
            raise ValueError(
                "time must be a finite numeric value with an ASCII label of at most 79 characters"
            )
        if self._times and numeric_time <= float(self._times[-1]):
            raise ValueError("Snapshot times must be unique and increasing")
        prepared = self._prepare_fields(fields, complex_mode)
        schema = {name: self._components(values) for name, values in prepared.items()}
        if set(schema).intersection(self._static):
            raise ValueError("Transient fields must not reuse static field names")
        if self._schema is not None and schema != self._schema:
            raise ValueError("All snapshots must have the same field schema")
        destination = self._output_dir / "data" / f"{len(self._times):08d}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Output snapshot already exists: {destination}")
        times = self._times + [label]
        with TemporaryDirectory(prefix=".ensight-", dir=self._output_dir) as temporary:
            folder = Path(temporary) / "snapshot"
            folder.mkdir()
            for name, values in prepared.items():
                self._write_variable(folder / name, name, values)
            folder.rename(destination)
            self._publish_case(self._static, schema, times)
        self._schema, self._times = schema, times

    def write_series(
        self,
        fields: Mapping[str, pt.Tensor],
        times: Sequence[Union[str, float]],
        complex_mode: str = "split",
    ):
        """Stream a time-last tensor series through :meth:`write_snapshot`.

        Successfully written snapshots remain readable if a later snapshot fails.
        For lazy reconstructions, call write_snapshot directly instead of
        materializing an entire tensor series.
        """
        self._ensure_open()
        if not isinstance(fields, Mapping) or not fields or len(times) == 0:
            raise ValueError("fields and times must be nonempty")
        for data in fields.values():
            if (
                not isinstance(data, pt.Tensor)
                or data.ndim < 2
                or data.shape[-1] != len(times)
            ):
                raise ValueError(
                    "Series tensors must have time last, with len(times) snapshots"
                )
        for index, time in enumerate(times):
            self.write_snapshot(
                {name: data[..., index] for name, data in fields.items()},
                time,
                complex_mode,
            )

    @property
    def case_file(self) -> Path:
        """Path of the new .case descriptor."""
        return self._case_file

    @property
    def write_times(self) -> list[str]:
        """Written transient time labels; static fields add no entries."""
        return list(self._times)

    def close(self):
        """Finalize the descriptor; repeated calls are harmless."""
        if not self._closed:
            self._publish_case(self._static, self._schema, self._times)
            self._closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Successful writes already published valid descriptors; preserve them
        # without masking an exception from the user's analysis or IO operation.
        if exc_type is None:
            self.close()
        else:
            self._closed = True
