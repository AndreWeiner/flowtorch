import pytest
import torch as pt

from flowtorch.data import element_areas_to_node_weights, grid_element_areas


def test_grid_element_areas_on_unit_grid():
    x, y = pt.meshgrid(pt.arange(4.0), pt.arange(3.0), indexing="ij")

    areas = grid_element_areas(x, y)

    assert areas.shape == (3, 2)
    assert pt.equal(areas, pt.ones_like(areas))


def test_grid_element_areas_on_stretched_grid():
    x, y = pt.meshgrid(
        pt.tensor([0.0, 1.0, 3.0]),
        pt.tensor([0.0, 2.0, 5.0]),
        indexing="ij",
    )

    areas = grid_element_areas(x, y)

    assert pt.equal(areas, pt.tensor([[2.0, 3.0], [4.0, 6.0]]))


def test_zero_z_matches_planar_area():
    x, y = pt.meshgrid(pt.arange(4.0), pt.arange(3.0), indexing="ij")

    planar = grid_element_areas(x, y)
    embedded = grid_element_areas(x, y, pt.zeros_like(x))

    assert pt.equal(embedded, planar)


def test_grid_element_areas_on_tilted_surface():
    x, y = pt.meshgrid(pt.arange(3.0), pt.arange(3.0), indexing="ij")
    z = x

    areas = grid_element_areas(x, y, z)

    assert pt.allclose(areas, pt.full_like(areas, 2.0**0.5))


def test_surface_area_includes_out_of_plane_curvature():
    x, y = pt.meshgrid(pt.arange(3.0), pt.arange(3.0), indexing="ij")
    z = x**2

    planar = grid_element_areas(x, y)
    surface = grid_element_areas(x, y, z)

    assert pt.all(surface > planar)


def test_degenerate_element_has_zero_area():
    x = pt.zeros((2, 2))
    y = pt.zeros((2, 2))

    assert grid_element_areas(x, y).item() == 0.0
    assert grid_element_areas(x, y, pt.zeros_like(x)).item() == 0.0


def test_element_areas_to_node_weights_averages_adjacent_elements():
    areas = pt.tensor([[1.0, 3.0], [5.0, 7.0]])

    weights = element_areas_to_node_weights(areas, square_root=False, normalize=False)

    expected = pt.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]])
    assert pt.equal(weights, expected)


def test_element_areas_to_node_weights_default_raw_transformation():
    areas = pt.tensor([[1.0, 4.0]])

    weights = element_areas_to_node_weights(areas)

    expected_areas = pt.tensor([[1.0, 2.5, 4.0], [1.0, 2.5, 4.0]])
    assert pt.allclose(weights, expected_areas / 4.0)


def test_element_areas_to_node_weights_can_return_square_root_factors():
    areas = pt.tensor([[1.0, 4.0]])

    weights = element_areas_to_node_weights(areas, square_root=True)

    expected_areas = pt.tensor([[1.0, 2.5, 4.0], [1.0, 2.5, 4.0]])
    assert pt.allclose(weights, pt.sqrt(expected_areas) / 2.0)


@pytest.mark.parametrize(
    "x, y, z",
    [
        (pt.zeros(4), pt.zeros(4), None),
        (pt.zeros((2, 2)), pt.zeros((3, 2)), None),
        (pt.zeros((1, 2)), pt.zeros((1, 2)), None),
        (pt.zeros((2, 2)), pt.zeros((2, 2)), pt.zeros((3, 2))),
        (
            pt.tensor([[0.0, float("nan")], [0.0, 0.0]]),
            pt.zeros((2, 2)),
            None,
        ),
    ],
)
def test_grid_element_areas_rejects_invalid_inputs(x, y, z):
    with pytest.raises(ValueError):
        grid_element_areas(x, y, z)


@pytest.mark.parametrize(
    "areas",
    [
        pt.zeros(4),
        pt.empty((0, 2)),
        pt.ones((2, 2), dtype=pt.int64),
        pt.tensor([[1.0, -1.0]]),
        pt.tensor([[1.0, float("nan")]]),
    ],
)
def test_element_areas_to_node_weights_rejects_invalid_inputs(areas):
    with pytest.raises(ValueError):
        element_areas_to_node_weights(areas)
