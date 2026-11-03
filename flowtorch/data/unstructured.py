"""Explicit unstructured geometry and raw geometric integration measures."""

from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass
class UnstructuredGeometry:
    """One zone, with zero-based integer topology and association-compatible data.

    ``topology`` is either an (n_cells, n_nodes) array for a uniform XDMF cell
    type, or an XDMF Mixed stream. Coordinates and locations are Float64.
    """

    points: np.ndarray
    topology: np.ndarray
    topology_type: str
    n_cells: int
    association: str = "point"
    locations: np.ndarray | None = None
    weights: np.ndarray | None = None
    dimension: int = 3

    def validate(self) -> None:
        """Reject inconsistent shapes, nonfinite geometry and invalid measures."""
        self.points = np.asarray(self.points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("Coordinates must have shape (n_points, 3)")
        if not np.isfinite(self.points).all() or not len(self.points):
            raise ValueError("Coordinates must be finite and nonempty")
        self.topology = np.asarray(self.topology)
        if self.topology.dtype.kind not in "iu" or self.n_cells <= 0:
            raise ValueError("Topology must contain integer indices and nonempty cells")
        if self.topology_type != "Mixed":
            nodes = {
                "Triangle": 3,
                "Quadrilateral": 4,
                "Tetrahedron": 4,
                "Hexahedron": 8,
                "Polyline": 2,
            }.get(self.topology_type)
            if nodes is None or self.topology.shape != (self.n_cells, nodes):
                raise ValueError("Topology shape does not match cell type")
            if self.topology.min() < 0 or self.topology.max() >= len(self.points):
                raise ValueError("Topology index outside coordinate array")
        if self.association not in ("point", "cell"):
            raise ValueError("Expected point or cell association")
        n = len(self.points) if self.association == "point" else self.n_cells
        if self.locations is None and self.association == "point":
            self.locations = self.points
        self.locations = np.asarray(self.locations, dtype=np.float64)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        if self.locations.shape != (n, 3) or not np.isfinite(self.locations).all():
            raise ValueError("Invalid association-compatible locations")
        if self.weights.shape != (n,) or not np.all(
            np.isfinite(self.weights) & (self.weights > 0)
        ):
            raise ValueError("Geometric weights must be raw, finite and positive")

    @property
    def signature(self) -> str:
        """Hash geometry, integration semantics and topology deterministically."""
        self.validate()
        digest = hashlib.sha256(
            str(
                (self.topology_type, self.n_cells, self.association, self.dimension)
            ).encode()
        )
        for value, dtype in (
            (self.points, "<f8"),
            (self.topology, "<i8"),
            (self.locations, "<f8"),
            (self.weights, "<f8"),
        ):
            assert value is not None  # Established by validate().
            digest.update(str(value.shape).encode())
            for start in range(0, len(value), 65536):
                digest.update(
                    np.asarray(value[start : start + 65536], dtype=dtype).tobytes()
                )
        return digest.hexdigest()


def geometric_weights(mesh, association: str = "point") -> np.ndarray:
    """Return raw cell measures or conservative lumped nodal measures.

    Uses VTK's compiled cell-size implementation. Mixed physical dimensions,
    invalid cells and isolated points are rejected rather than normalized away.
    """
    from ._tecplot_pvpython import _geometric_weights

    return _geometric_weights(mesh, association, np)


def geometry_from_mesh(mesh, association: str = "point") -> UnstructuredGeometry:
    """Extract a uniform or mixed linear-cell mesh using bulk VTK arrays."""
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from ._tecplot_pvpython import _cell_locations

    points = np.asarray(vtk_to_numpy(mesh.GetPoints().GetData()), dtype=np.float64)
    cells = mesh.GetCells()
    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    ids = vtk_to_numpy(cells.GetConnectivityArray()).astype(np.int64)
    try:
        vtk_types = mesh.GetCellTypes()
    except TypeError:  # VTK releases with only the output-argument overload.
        vtk_types = mesh.GetCellTypesArray()
    types = vtk_to_numpy(vtk_types)
    mapping = {
        3: ("Polyline", 2, 1),
        5: ("Triangle", 4, 2),
        9: ("Quadrilateral", 5, 2),
        10: ("Tetrahedron", 6, 3),
        12: ("Hexahedron", 9, 3),
        13: ("Wedge", 8, 3),
        14: ("Pyramid", 7, 3),
    }
    if any(int(t) not in mapping for t in np.unique(types)):
        raise ValueError("Unsupported mesh cell type for XDMF export")
    dimensions = {mapping[int(t)][2] for t in np.unique(types)}
    if len(dimensions) != 1:
        raise ValueError("Cannot combine different physical cell dimensions")
    kinds = np.unique(types)
    if len(kinds) == 1 and int(kinds[0]) in (3, 5, 9, 10, 12):
        topology_type = mapping[int(kinds[0])][0]
        topology = ids.reshape(len(types), -1)
    else:
        topology_type = "Mixed"
        counts = np.diff(offsets)
        extra = np.where(types == 3, 2, 1)
        starts = offsets[:-1] + np.cumsum(extra) - extra
        topology = np.empty(len(ids) + int(extra.sum()), dtype=np.int64)
        codes = np.zeros(256, dtype=np.int64)
        for key, (_, code, _) in mapping.items():
            codes[key] = code
        topology[starts] = codes[types]
        topology[starts[types == 3] + 1] = counts[types == 3]
        topology[np.arange(len(ids)) + np.repeat(np.cumsum(extra), counts)] = ids
    result = UnstructuredGeometry(
        points,
        topology,
        topology_type,
        len(types),
        association,
        points if association == "point" else _cell_locations(mesh, np),
        geometric_weights(mesh, association),
        dimensions.pop(),
    )
    result.validate()
    return result
