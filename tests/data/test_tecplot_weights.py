"""Geometric quadrature checks using real VTK meshes."""

import numpy as np
import pytest

pytest.importorskip("vtkmodules")
from vtkmodules.vtkCommonCore import vtkPoints, vtkUnsignedCharArray
from vtkmodules.vtkCommonDataModel import (
    vtkUnstructuredGrid,
    vtkTriangle,
    vtkQuad,
    vtkLine,
    vtkTetra,
    vtkStructuredGrid,
)
from flowtorch.data._tecplot_pvpython import _geometric_weights, _cell_locations


def mesh(points, cells):
    vtkpoints = vtkPoints()
    for point in points:
        vtkpoints.InsertNextPoint(*point)
    grid = vtkUnstructuredGrid()
    grid.SetPoints(vtkpoints)
    for cell_type, ids in cells:
        cell = cell_type()
        for j, i in enumerate(ids):
            cell.GetPointIds().SetId(j, i)
        grid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())
    return grid


def test_unequal_triangles_conserve_area_and_chunking():
    grid = mesh(
        [(0, 0, 0), (2, 0, 0), (0, 1, 0), (0, -3, 0)],
        [(vtkTriangle, [0, 1, 2]), (vtkTriangle, [0, 3, 1])],
    )
    np.testing.assert_allclose(_geometric_weights(grid, "cell", np), [1, 3])
    expected = [4 / 3, 4 / 3, 1 / 3, 1]
    np.testing.assert_allclose(_geometric_weights(grid, "point", np), expected)
    np.testing.assert_allclose(
        _geometric_weights(grid, "point", np, chunk_size=1), expected
    )
    np.testing.assert_allclose(
        _cell_locations(grid, np), [[2 / 3, 1 / 3, 0], [2 / 3, -1, 0]]
    )


@pytest.mark.parametrize(
    "cell_type,points,measure",
    [
        (vtkLine, [(0, 0, 0), (3, 0, 0)], 3),
        (vtkQuad, [(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)], 2),
        (vtkTetra, [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], 1 / 6),
    ],
)
def test_measures_by_dimension(cell_type, points, measure):
    grid = mesh(points, [(cell_type, list(range(len(points))))])
    np.testing.assert_allclose(_geometric_weights(grid, "cell", np), [measure])
    np.testing.assert_allclose(
        _geometric_weights(grid, "point", np),
        np.full(len(points), measure / len(points)),
    )


def test_structured_grid_preserves_vertex_order():
    points = vtkPoints()
    for point in [(0, 0, 0), (2, 0, 0), (0, 1, 0), (2, 1, 0), (0, 4, 0), (2, 4, 0)]:
        points.InsertNextPoint(*point)
    grid = vtkStructuredGrid()
    grid.SetDimensions(2, 3, 1)
    grid.SetPoints(points)
    np.testing.assert_allclose(_geometric_weights(grid, "cell", np), [2, 6])
    np.testing.assert_allclose(
        _geometric_weights(grid, "point", np), [0.5, 0.5, 2, 2, 1.5, 1.5]
    )


def test_mixed_triangles_and_quads():
    grid = mesh(
        [(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0), (0, -1, 0)],
        [(vtkQuad, [0, 1, 2, 3]), (vtkTriangle, [0, 4, 1])],
    )
    np.testing.assert_allclose(
        _geometric_weights(grid, "point", np), [5 / 6, 5 / 6, 0.5, 0.5, 1 / 3]
    )


@pytest.mark.parametrize(
    "kind,message",
    [
        ("isolated", "isolated points"),
        ("degenerate", "finite and positive"),
        ("mixed", "single dimension"),
        ("ghost", "Ghost cells"),
    ],
)
def test_invalid_geometry_is_reported(kind, message):
    points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    cells = [(vtkTriangle, [0, 1, 2])]
    if kind == "isolated":
        points.append((9, 9, 9))
    elif kind == "degenerate":
        points[2] = (2, 0, 0)
    elif kind == "mixed":
        cells.append((vtkLine, [0, 1]))
    grid = mesh(points, cells)
    if kind == "ghost":
        ghosts = vtkUnsignedCharArray()
        ghosts.SetName("vtkGhostType")
        ghosts.InsertNextValue(1)
        grid.GetCellData().AddArray(ghosts)
    with pytest.raises(ValueError, match=message):
        _geometric_weights(grid, "point", np)


def test_explicit_legacy_quad_override():
    grid = mesh(
        [(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)], [(vtkTetra, [0, 1, 2, 3])]
    )
    with pytest.raises(ValueError, match="finite and positive"):
        _geometric_weights(grid, "point", np)
    np.testing.assert_allclose(
        _geometric_weights(grid, "point", np, cell_type="quad"), [0.5] * 4
    )
    np.testing.assert_allclose(_cell_locations(grid, np, "quad"), [[1, 0.5, 0]])
    with pytest.raises(ValueError, match="requires 3 vertices"):
        _geometric_weights(grid, "point", np, cell_type="triangle")
    # Overrides must not mutate the input dataset.
    assert grid.GetCellType(0) == vtkTetra().GetCellType()
