import pytest
import torch as pt

from flowtorch.data import map_points_to_grid_2d, replace_masked_values


def _grid():
    return pt.meshgrid(pt.arange(5.0), pt.arange(5.0), indexing="ij")


def test_map_points_to_grid_2d_maps_scalar_field():
    grid_x, grid_y = _grid()
    x = grid_x.flatten()
    y = grid_y.flatten()
    field = x + 2.0 * y

    mapped = map_points_to_grid_2d(x, y, field, grid_x, grid_y)

    assert mapped.shape == grid_x.shape
    assert pt.allclose(mapped, grid_x + 2.0 * grid_y)


def test_map_points_to_grid_2d_maps_field_sequence():
    grid_x, grid_y = _grid()
    x = grid_x.flatten()
    y = grid_y.flatten()
    field = pt.stack((x + y, 2.0 * x - y), dim=-1)

    mapped = map_points_to_grid_2d(x, y, field, grid_x, grid_y)

    assert mapped.shape == (*grid_x.shape, 2)
    assert pt.allclose(mapped[..., 0], grid_x + grid_y)
    assert pt.allclose(mapped[..., 1], 2.0 * grid_x - grid_y)


def test_map_points_to_grid_2d_uses_nearest_outside_convex_hull():
    x = pt.tensor([0.0, 1.0, 0.0])
    y = pt.tensor([0.0, 0.0, 1.0])
    field = pt.tensor([0.0, 1.0, 1.0])
    grid_x = pt.tensor([[2.0]])
    grid_y = pt.tensor([[0.0]])

    mapped = map_points_to_grid_2d(x, y, field, grid_x, grid_y)

    assert mapped.item() == pytest.approx(1.0)


def test_map_points_to_grid_2d_uses_nearest_for_collinear_points():
    x = pt.tensor([0.0, 1.0, 2.0])
    y = pt.zeros(3)
    field = pt.tensor([0.0, 1.0, 2.0])
    grid_x = pt.tensor([[0.1, 1.9]])
    grid_y = pt.zeros_like(grid_x)

    mapped = map_points_to_grid_2d(x, y, field, grid_x, grid_y)

    assert pt.equal(mapped, pt.tensor([[0.0, 2.0]]))


def test_map_points_to_grid_2d_preserves_dtype():
    grid_x, grid_y = _grid()
    field = (grid_x + grid_y).flatten().to(dtype=pt.float64)

    mapped = map_points_to_grid_2d(
        grid_x.flatten(), grid_y.flatten(), field, grid_x, grid_y
    )

    assert mapped.dtype == pt.float64


@pytest.mark.parametrize(
    "x, y, field, grid_x, grid_y",
    [
        (
            pt.zeros((2, 2)),
            pt.zeros(4),
            pt.zeros(4),
            pt.zeros((2, 2)),
            pt.zeros((2, 2)),
        ),
        (pt.zeros(3), pt.zeros(4), pt.zeros(3), pt.zeros((2, 2)), pt.zeros((2, 2))),
        (pt.zeros(3), pt.zeros(3), pt.zeros(4), pt.zeros((2, 2)), pt.zeros((2, 2))),
        (pt.zeros(3), pt.zeros(3), pt.zeros(3), pt.zeros(4), pt.zeros((2, 2))),
        (
            pt.tensor([0.0, float("nan")]),
            pt.zeros(2),
            pt.zeros(2),
            pt.zeros((2, 2)),
            pt.zeros((2, 2)),
        ),
    ],
)
def test_map_points_to_grid_2d_rejects_invalid_inputs(x, y, field, grid_x, grid_y):
    with pytest.raises(ValueError):
        map_points_to_grid_2d(x, y, field, grid_x, grid_y)


def test_replace_masked_values_interpolates_single_field():
    x, y = _grid()
    field = x + 2.0 * y
    mask = pt.ones((5, 5), dtype=pt.bool)
    mask[2, 2] = False

    repaired = replace_masked_values(x, y, field, mask)

    assert repaired.shape == field.shape
    assert repaired[2, 2] == pytest.approx(field[2, 2].item())
    assert pt.equal(repaired[mask], field[mask])


def test_replace_masked_values_interpolates_sequence():
    x, y = _grid()
    field = pt.stack((x + y, 2.0 * x - y), dim=-1)
    mask = pt.ones((5, 5), dtype=pt.bool)
    mask[2, 2] = False

    repaired = replace_masked_values(x, y, field, mask)

    assert repaired.shape == field.shape
    assert pt.allclose(repaired[2, 2], field[2, 2])


def test_replace_masked_values_returns_copy_for_complete_mask():
    x, y = _grid()
    field = x + y
    mask = pt.ones((5, 5), dtype=pt.bool)

    repaired = replace_masked_values(x, y, field, mask)

    assert repaired is not field
    assert pt.equal(repaired, field)


def test_replace_masked_values_rejects_invalid_inputs():
    x, y = _grid()
    field = x + y
    mask = pt.ones((5, 5), dtype=pt.bool)

    with pytest.raises(ValueError):
        replace_masked_values(x[:-1], y, field, mask)
    with pytest.raises(ValueError):
        replace_masked_values(x, y, field, pt.zeros_like(mask))
