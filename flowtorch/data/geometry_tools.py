"""Geometry utilities for structured two-dimensional grids."""

# third party packages
import torch as pt


def grid_element_areas(
    x: pt.Tensor, y: pt.Tensor, z: pt.Tensor | None = None
) -> pt.Tensor:
    """Compute quadrilateral element areas on a structured 2D grid.

    If only ``x`` and ``y`` are supplied, element areas are computed in the
    coordinate plane using the shoelace formula. If ``z`` is supplied, the
    grid is treated as a surface embedded in three dimensions. Each possibly
    non-planar quadrilateral is triangulated along both diagonals, and the two
    resulting area estimates are averaged to avoid favoring one diagonal.

    The coordinates must have shape ``(nx, ny)``. The returned element areas
    have shape ``(nx - 1, ny - 1)`` and preserve the coordinates' dtype and
    device. For curved surfaces, these areas form a piecewise-planar
    approximation that converges as the grid is refined.

    :param x: first coordinate on the structured grid
    :type x: pt.Tensor
    :param y: second coordinate on the structured grid
    :type y: pt.Tensor
    :param z: optional third coordinate on the structured grid
    :type z: pt.Tensor, optional
    :return: area of every grid element
    :rtype: pt.Tensor

    **Examples**

    .. code-block:: python

        x, y = pt.meshgrid(pt.arange(5.0), pt.arange(4.0), indexing="ij")
        planar_areas = grid_element_areas(x, y)

        z = 0.25 * x**2
        surface_areas = grid_element_areas(x, y, z)
    """
    _validate_coordinates(x, y, z)
    if z is None:
        return _planar_element_areas(x, y)
    return _surface_element_areas(x, y, z)


def element_areas_to_node_weights(
    element_areas: pt.Tensor,
    square_root: bool = True,
    normalize: bool = True,
) -> pt.Tensor:
    """Convert structured-grid element areas to node weights.

    Each node receives the arithmetic mean of its adjacent element areas.
    Corner, edge, and interior nodes therefore average one, two, and four
    elements, respectively. By default, the square root is taken to produce
    weights suitable for weighted inner products, and the weights are
    normalized by their maximum value.

    :param element_areas: non-negative areas with shape ``(nx - 1, ny - 1)``
    :type element_areas: pt.Tensor
    :param square_root: take the square root of the nodal areas
    :type square_root: bool, optional
    :param normalize: normalize weights by their maximum value
    :type normalize: bool, optional
    :return: node weights with shape ``(nx, ny)``
    :rtype: pt.Tensor

    **Examples**

    .. code-block:: python

        areas = grid_element_areas(x, y, z)
        weights = element_areas_to_node_weights(areas)
    """
    if element_areas.ndim != 2 or min(element_areas.shape) < 1:
        raise ValueError("element_areas must be a non-empty 2D tensor")
    if not element_areas.is_floating_point():
        raise ValueError("element_areas must have a floating-point dtype")
    if not pt.isfinite(element_areas).all():
        raise ValueError("element_areas must contain only finite values")
    if (element_areas < 0).any():
        raise ValueError("element_areas must be non-negative")

    nx, ny = element_areas.shape[0] + 1, element_areas.shape[1] + 1
    nodal_areas = pt.zeros(
        (nx, ny), dtype=element_areas.dtype, device=element_areas.device
    )
    adjacent_elements = pt.zeros_like(nodal_areas)
    for row_slice, column_slice in (
        (slice(None, -1), slice(None, -1)),
        (slice(1, None), slice(None, -1)),
        (slice(1, None), slice(1, None)),
        (slice(None, -1), slice(1, None)),
    ):
        nodal_areas[row_slice, column_slice] += element_areas
        adjacent_elements[row_slice, column_slice] += 1
    weights = nodal_areas / adjacent_elements
    if square_root:
        weights = pt.sqrt(weights)
    if normalize:
        weights /= pt.max(weights)
    return weights


def _planar_element_areas(x: pt.Tensor, y: pt.Tensor) -> pt.Tensor:
    x00, y00 = x[:-1, :-1], y[:-1, :-1]
    x10, y10 = x[1:, :-1], y[1:, :-1]
    x11, y11 = x[1:, 1:], y[1:, 1:]
    x01, y01 = x[:-1, 1:], y[:-1, 1:]
    forward = x00 * y10 + x10 * y11 + x11 * y01 + x01 * y00
    backward = y00 * x10 + y10 * x11 + y11 * x01 + y01 * x00
    return 0.5 * pt.abs(forward - backward)


def _surface_element_areas(x: pt.Tensor, y: pt.Tensor, z: pt.Tensor) -> pt.Tensor:
    points = pt.stack((x, y, z), dim=-1)
    p00 = points[:-1, :-1]
    p10 = points[1:, :-1]
    p11 = points[1:, 1:]
    p01 = points[:-1, 1:]

    first_diagonal = _triangle_areas(p00, p10, p11) + _triangle_areas(p00, p11, p01)
    second_diagonal = _triangle_areas(p00, p10, p01) + _triangle_areas(p10, p11, p01)
    return 0.5 * (first_diagonal + second_diagonal)


def _triangle_areas(a: pt.Tensor, b: pt.Tensor, c: pt.Tensor) -> pt.Tensor:
    cross_product = pt.linalg.cross(b - a, c - a, dim=-1)
    return 0.5 * pt.linalg.vector_norm(cross_product, dim=-1)


def _validate_coordinates(x: pt.Tensor, y: pt.Tensor, z: pt.Tensor | None) -> None:
    coordinates = (x, y) if z is None else (x, y, z)
    if any(coordinate.ndim != 2 for coordinate in coordinates):
        raise ValueError("coordinates must be two-dimensional")
    if any(coordinate.shape != x.shape for coordinate in coordinates[1:]):
        raise ValueError("coordinates must have identical shapes")
    if min(x.shape) < 2:
        raise ValueError("each grid dimension must contain at least two points")
    if any(coordinate.device != x.device for coordinate in coordinates[1:]):
        raise ValueError("coordinates must be on the same device")
    if any(coordinate.dtype != x.dtype for coordinate in coordinates[1:]):
        raise ValueError("coordinates must have the same dtype")
    if not x.is_floating_point():
        raise ValueError("coordinates must have a floating-point dtype")
    if any(not pt.isfinite(coordinate).all() for coordinate in coordinates):
        raise ValueError("coordinates must contain only finite values")
