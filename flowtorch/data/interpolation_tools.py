"""Interpolation tools for repairing incomplete field data."""

# third party packages
import numpy as np
import torch as pt
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import QhullError


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
    flattened_mask = mask.reshape(-1).to(dtype=pt.bool)
    if flattened_mask.all():
        return field.clone()

    points = np.column_stack(
        (x.detach().cpu().numpy().ravel(), y.detach().cpu().numpy().ravel())
    )
    points_valid = points[flattened_mask.cpu().numpy()]
    points_invalid = points[~flattened_mask].copy()
    values_valid = flattened_field[flattened_mask].detach().cpu().numpy()
    if points_valid.shape[0] == 0:
        raise ValueError("mask must contain at least one valid point")

    # SciPy's interpolators operate on NumPy arrays. Keep all conversions at
    # this boundary and restore the input dtype/device before returning.
    try:
        linear = LinearNDInterpolator(points_valid, values_valid, fill_value=np.nan)
        values_invalid = np.asarray(linear(points_invalid))
    except QhullError:
        # A linear simplex cannot be formed from too few or collinear points.
        values_invalid = np.full(
            (points_invalid.shape[0], n_snapshots), np.nan, dtype=values_valid.dtype
        )
    if values_invalid.ndim == 1:
        values_invalid = values_invalid[:, None]
    missing = np.isnan(values_invalid).any(axis=1)
    if missing.any():
        nearest = NearestNDInterpolator(points_valid, values_valid)
        values_invalid[missing] = nearest(points_invalid[missing])

    result = flattened_field.clone()
    replacement = pt.as_tensor(values_invalid, dtype=field.dtype, device=field.device)
    result[~flattened_mask] = replacement
    return result.reshape(initial_shape)


def _validate_inputs(
    x: pt.Tensor, y: pt.Tensor, field: pt.Tensor, mask: pt.Tensor
) -> None:
    if x.ndim != 2 or y.ndim != 2 or mask.ndim != 2:
        raise ValueError("x, y, and mask must be two-dimensional")
    if x.shape != y.shape or x.shape != mask.shape:
        raise ValueError("x, y, and mask must have identical shapes")
    if field.ndim not in (2, 3) or field.shape[:2] != x.shape:
        raise ValueError("field must have shape (nx, ny) or (nx, ny, n_snapshots)")
