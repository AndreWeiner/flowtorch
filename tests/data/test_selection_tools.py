# third party packages
import pytest
import torch as pt

# flowtorch packages
from flowtorch.data import mask_box, mask_polygon, mask_sphere


def test_mask_box_1d():
    vertices = pt.linspace(0.0, 5.0, 6)
    mask = mask_box(vertices, [2.0], [4.0])
    true_mask = pt.tensor([False, False, True, True, True, False])
    assert pt.all(mask == true_mask)


def test_mask_box_2d():
    vertices = pt.tensor([[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 2], [3, 4]])
    mask = mask_box(vertices, [1, 2], [2, 3])
    true_mask = pt.tensor([True, True, False, True, False, False, False])
    assert pt.all(mask == true_mask)


def test_mask_box_3d():
    vertices = pt.tensor(
        [[1, 2, 1], [1, 3, 2], [1, 4, 2], [2, 3, 2], [2, 4, 1], [3, 2, 1], [3, 4, 2]]
    )
    mask = mask_box(vertices, [1, 2, 0], [2, 3, 2])
    true_mask = pt.tensor([True, True, False, True, False, False, False])
    assert pt.all(mask == true_mask)


def test_mask_sphere_1d():
    vertices = pt.linspace(0.0, 5.0, 6)
    mask = mask_sphere(vertices, [2.0], 1.5)
    true_mask = pt.tensor([False, True, True, True, False, False])
    assert pt.all(mask == true_mask)


def test_mask_sphere_2d():
    vertices = pt.tensor([[1.0, 0.0], [0.3, 0.3], [1.5, 1.5]])
    mask = mask_sphere(vertices, [0, 0], 1.0)
    true_mask = pt.tensor([True, True, False])
    assert pt.all(mask == true_mask)


def test_mask_sphere_3d():
    vertices = pt.tensor([[-1, 0, 0], [0, 1, 0], [0.1, 0.1, 0.1], [1, 1, 1]])
    mask = mask_sphere(vertices, [0, 0, 0], 1.0)
    true_mask = pt.tensor([True, True, True, False])
    assert pt.all(mask == true_mask)


def test_mask_polygon():
    vertices = pt.tensor([[0.5, 0.5], [3.5, 0.5], [3.5, 3.5], [0.5, 3.5]])
    mask = mask_polygon((5, 5), vertices)

    assert mask.dtype == pt.bool
    assert mask.shape == (5, 5)
    assert mask[2, 2]
    assert not mask[0, 0]


def test_mask_polygon_with_too_few_vertices_is_empty():
    mask = mask_polygon((2, 3), pt.tensor([[0.0, 0.0], [1.0, 1.0]]))

    assert not mask.any()


@pytest.mark.parametrize(
    "shape, vertices",
    [
        ((2, 3, 4), pt.zeros((3, 2))),
        ((2, 3), pt.zeros((3, 3))),
    ],
)
def test_mask_polygon_rejects_invalid_inputs(shape, vertices):
    with pytest.raises(ValueError):
        mask_polygon(shape, vertices)
