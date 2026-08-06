"""Interpolation tools for repairing incomplete field data."""

# third party packages
import numpy as np
import torch as pt
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import QhullError


def map_points_to_grid_2d(
    x: pt.Tensor,
    y: pt.Tensor,
    field: pt.Tensor,
    grid_x: pt.Tensor,
    grid_y: pt.Tensor,
) -> pt.Tensor:
    """Map field values at scattered points to a two-dimensional grid.

    Linear interpolation is used inside the convex hull of the source points.
    Target points outside the convex hull are filled by nearest-neighbor
    interpolation. If the source points cannot form a Delaunay triangulation,
    nearest-neighbor interpolation is used for the complete target grid.

    A scalar ``field`` with shape ``(n_points,)`` produces an output with shape
    ``(nx, ny)``. A field sequence with shape ``(n_points, n_snapshots)``
    produces an output with shape ``(nx, ny, n_snapshots)``. The result
    preserves the field's dtype and device.

    :param x: first coordinate of the scattered source points
    :type x: pt.Tensor
    :param y: second coordinate of the scattered source points
    :type y: pt.Tensor
    :param field: scalar field or field sequence at the source points
    :type field: pt.Tensor
    :param grid_x: first coordinate on the target grid
    :type grid_x: pt.Tensor
    :param grid_y: second coordinate on the target grid
    :type grid_y: pt.Tensor
    :return: field values mapped to the target grid
    :rtype: pt.Tensor

    **Examples**

    .. code-block:: python

        x = pt.tensor([0.0, 1.0, 0.0, 1.0])
        y = pt.tensor([0.0, 0.0, 1.0, 1.0])
        field = x + y
        grid_x, grid_y = pt.meshgrid(
            pt.linspace(0.0, 1.0, 20),
            pt.linspace(0.0, 1.0, 10),
            indexing="ij",
        )
        mapped = map_points_to_grid_2d(x, y, field, grid_x, grid_y)
    """
    _validate_mapping_inputs(x, y, field, grid_x, grid_y)
    is_scalar_field = field.ndim == 1
    values = field.reshape(field.shape[0], -1).detach().cpu().numpy()
    points = np.column_stack((x.detach().cpu().numpy(), y.detach().cpu().numpy()))
    targets = np.column_stack(
        (
            grid_x.detach().cpu().numpy().ravel(),
            grid_y.detach().cpu().numpy().ravel(),
        )
    )
    mapped = _linear_with_nearest(points, values, targets)
    output_shape = grid_x.shape if is_scalar_field else (*grid_x.shape, field.shape[1])
    return pt.as_tensor(mapped, dtype=field.dtype, device=field.device).reshape(
        output_shape
    )


def replace_masked_values(
    x: pt.Tensor,
    y: pt.Tensor,
    field: pt.Tensor,
    mask: pt.Tensor,
) -> pt.Tensor:
    """Replace invalid values in a sequence of structured 2D fields.

    Values at points where ``mask`` is ``False`` are interpolated from valid
    points. Linear interpolation is used inside the convex hull of valid
    points; nearest-neighbor interpolation fills points outside that hull.

    ``field`` may be a single 2D field with shape ``(nx, ny)`` or a sequence
    with shape ``(nx, ny, n_snapshots)``. Coordinates and the mask must have
    shape ``(nx, ny)``. The returned tensor preserves the field's dtype and
    device.

    :param x: first coordinate on the structured grid
    :type x: pt.Tensor
    :param y: second coordinate on the structured grid
    :type y: pt.Tensor
    :param field: field or sequence of fields to repair
    :type field: pt.Tensor
    :param mask: boolean validity mask; ``True`` denotes a valid point
    :type mask: pt.Tensor
    :return: field with invalid values replaced by interpolation
    :rtype: pt.Tensor

    **Examples**

    .. code-block:: python

        x, y = pt.meshgrid(pt.arange(5.0), pt.arange(5.0), indexing="ij")
        field = x + y
        mask = pt.ones((5, 5), dtype=pt.bool)
        mask[2, 2] = False
        repaired = replace_masked_values(x, y, field, mask)
    """
    _validate_inputs(x, y, field, mask)
    initial_shape = field.shape
    n_snapshots = 1 if field.ndim == 2 else field.shape[-1]
    flattened_field = field.reshape(-1, n_snapshots)
    flattened_mask = mask.reshape(-1).to(device=field.device, dtype=pt.bool)
    if flattened_mask.all():
        return field.clone()

    mask_numpy = flattened_mask.detach().cpu().numpy()
    points = np.column_stack(
        (x.detach().cpu().numpy().ravel(), y.detach().cpu().numpy().ravel())
    )
    points_valid = points[mask_numpy]
    points_invalid = points[~mask_numpy]
    values_valid = flattened_field[flattened_mask].detach().cpu().numpy()
    if points_valid.shape[0] == 0:
        raise ValueError("mask must contain at least one valid point")

    values_invalid = _linear_with_nearest(points_valid, values_valid, points_invalid)

    result = flattened_field.clone()
    replacement = pt.as_tensor(values_invalid, dtype=field.dtype, device=field.device)
    result[~flattened_mask] = replacement
    return result.reshape(initial_shape)


def _linear_with_nearest(
    points: np.ndarray, values: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Interpolate values linearly with nearest-neighbor fallback."""
    try:
        linear = LinearNDInterpolator(points, values, fill_value=np.nan)
        interpolated = np.asarray(linear(targets))
    except QhullError:
        # A linear simplex cannot be formed from too few or collinear points.
        interpolated = np.full(
            (targets.shape[0], values.shape[1]), np.nan, dtype=values.dtype
        )
    if interpolated.ndim == 1:
        interpolated = interpolated[:, None]
    missing = np.isnan(interpolated).any(axis=1)
    if missing.any():
        nearest = NearestNDInterpolator(points, values)
        interpolated[missing] = nearest(targets[missing])
    return interpolated


def _validate_mapping_inputs(
    x: pt.Tensor,
    y: pt.Tensor,
    field: pt.Tensor,
    grid_x: pt.Tensor,
    grid_y: pt.Tensor,
) -> None:
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional")
    if x.shape != y.shape:
        raise ValueError("x and y must have identical shapes")
    if x.numel() == 0:
        raise ValueError("at least one source point is required")
    if field.ndim not in (1, 2) or field.shape[0] != x.shape[0]:
        raise ValueError("field must have shape (n_points,) or (n_points, n_snapshots)")
    if grid_x.ndim != 2 or grid_y.ndim != 2 or grid_x.shape != grid_y.shape:
        raise ValueError("grid_x and grid_y must be identically shaped 2D tensors")
    if not pt.isfinite(x).all() or not pt.isfinite(y).all():
        raise ValueError("source coordinates must contain only finite values")
    if not pt.isfinite(grid_x).all() or not pt.isfinite(grid_y).all():
        raise ValueError("target coordinates must contain only finite values")


def _validate_inputs(
    x: pt.Tensor, y: pt.Tensor, field: pt.Tensor, mask: pt.Tensor
) -> None:
    if x.ndim != 2 or y.ndim != 2 or mask.ndim != 2:
        raise ValueError("x, y, and mask must be two-dimensional")
    if x.shape != y.shape or x.shape != mask.shape:
        raise ValueError("x, y, and mask must have identical shapes")
    if field.ndim not in (2, 3) or field.shape[:2] != x.shape:
        raise ValueError("field must have shape (nx, ny) or (nx, ny, n_snapshots)")
