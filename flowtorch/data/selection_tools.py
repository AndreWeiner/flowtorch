"""Helper tools for building data matrices."""

# standard library packages
from typing import List

# third party packages
import torch as pt


def mask_box(vertices: pt.Tensor, lower: List[float], upper: List[float]) -> pt.Tensor:
    """Create a boolean mask to select all vertices in a box.

    This function may be used in conjunction with torch.masked_select to select
    all field values in a box, e.g., when building data matrices.

    :param vertices: tensor of vertices, where each column corresponds to a coordinate
    :type vertices: pt.Tensor
    :param lower: lower bounds of box; one value for each coordinate must be given
    :type lower: List[float]
    :param upper: upper bounds of box; one value for each coordinate must be given
    :type upper: List[float]
    :return: boolean mask that's *True* for every vertex inside the box
    :rtype: pt.Tensor

    """
    assert (
        len(vertices.shape) < 3
    ), "The vertices tensor cannot have more than two axes."
    dim_message = "Exactly one lower and upper bound must be given for each coordinate."
    if len(vertices.shape) == 1:
        assert len(lower) == len(upper) == 1, dim_message
        return pt.logical_and(
            pt.where(vertices >= lower[0], True, False),
            pt.where(vertices <= upper[0], True, False),
        )
    else:
        assert vertices.shape[1] == len(lower) == len(upper), dim_message
        return pt.all(
            pt.logical_and(
                pt.where(vertices >= pt.tensor(lower), True, False),
                pt.where(vertices <= pt.tensor(upper), True, False),
            ),
            dim=1,
        )


def mask_sphere(vertices: pt.Tensor, center: List[float], radius: float) -> pt.Tensor:
    """Create a boolean mask to select all vertices in a sphere.

    This function may be used in conjunction with torch.masked_select to select
    all field values within a sphere, e.g., when building data matrices.

    :param vertices: tensor of vertices, where each column corresponds to a coordinate
    :type vertices: pt.Tensor
    :param center: the sphere's center
    :type center: List[float]
    :param radius: the sphere's radius
    :type radius: float
    :return: boolean mask that's *True* for every vertex inside the sphere
    :rtype: pt.Tensor
    """
    center_tensor = pt.tensor(center).type(vertices.dtype)
    assert (
        len(vertices.shape) < 3
    ), "The vertices tensor cannot have more than two axes."
    if len(vertices.shape) == 1:
        assert len(center_tensor) == 1
        radii = pt.abs(vertices - center_tensor)
    else:
        assert (
            vertices.shape[1] == center_tensor.shape[0]
        ), "Missmatch between number of vertices and center coordinates."
        radii = pt.linalg.norm(vertices - center_tensor, dim=1)
    return pt.where(radii <= radius, True, False)
