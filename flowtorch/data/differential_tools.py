"""Differential operators for structured curvilinear grids."""

# standard library packages
from math import ceil, isfinite
from typing import Tuple

# third party packages
import torch as pt
from torch.nn import functional as F

# flowTorch packages
from .outlier_tools import replace_spatial_outliers


def curvilinear_gradient(
    field: pt.Tensor,
    x: pt.Tensor,
    y: pt.Tensor,
    z: pt.Tensor | None = None,
    smoothing_sigma: float | Tuple[float, float] | None = None,
    smoothing_mode: str = "reflect",
    smoothing_truncate: float = 3.0,
    edge_order: int = 2,
    metric_tolerance: float = 1.0e-12,
    outlier_threshold: float | None = None,
    outlier_window_size: int = 3,
) -> pt.Tensor:
    """Compute spatial gradients on a structured curvilinear grid.

    The field may be one snapshot with shape ``(nx, ny)`` or a sequence with
    shape ``(nx, ny, n_snapshots)``. Every snapshot is differentiated
    independently. For coordinates ``x`` and ``y``, the result contains two
    Cartesian components. If ``z`` is supplied, the result is the
    three-component ambient Cartesian representation of the surface-tangential
    gradient. A normal derivative cannot be inferred from surface data.

    Gaussian smoothing is disabled by default and is applied only along the
    spatial grid axes before differentiation. Sigma values are measured in
    grid-index units rather than physical distance. Values around ``0.5`` to
    ``0.75`` provide light smoothing, ``1.0`` is a sensible general-purpose
    starting value for noisy experimental data, and ``1.5`` to ``2.0`` provide
    strong smoothing. A two-tuple applies different widths along the two grid
    directions. With the default truncation, ``sigma=1.0`` uses a seven-point
    kernel. On strongly stretched grids, a fixed sigma represents a spatially
    varying physical smoothing width.

    Spatial outlier replacement is enabled by setting ``outlier_threshold``.
    It is applied after differentiation, independently to every snapshot and
    gradient component, using a local median and median absolute deviation.
    Invalid gradients caused by singular grid metrics remain ``NaN``.

    :param field: scalar field or sequence of scalar fields
    :type field: pt.Tensor
    :param x: first Cartesian coordinate on the structured grid
    :type x: pt.Tensor
    :param y: second Cartesian coordinate on the structured grid
    :type y: pt.Tensor
    :param z: optional third Cartesian coordinate on an embedded surface
    :type z: pt.Tensor, optional
    :param smoothing_sigma: Gaussian width in grid-index units
    :type smoothing_sigma: float or Tuple[float, float], optional
    :param smoothing_mode: boundary mode; ``"reflect"``, ``"replicate"``, or
        ``"circular"``
    :type smoothing_mode: str, optional
    :param smoothing_truncate: Gaussian kernel radius in multiples of sigma
    :type smoothing_truncate: float, optional
    :param edge_order: finite-difference boundary accuracy, either 1 or 2
    :type edge_order: int, optional
    :param metric_tolerance: relative threshold for singular metric tensors
    :type metric_tolerance: float, optional
    :param outlier_threshold: local robust-score threshold; disabled if ``None``
    :type outlier_threshold: float, optional
    :param outlier_window_size: odd spatial window size for outlier replacement
    :type outlier_window_size: int, optional
    :return: gradient with a final axis of length two or three
    :rtype: pt.Tensor

    **Examples**

    .. code-block:: python

        gradient = curvilinear_gradient(field, x, y)
        smooth_gradient = curvilinear_gradient(
            field_sequence,
            x,
            y,
            z,
            smoothing_sigma=1.0,
            outlier_threshold=3.5,
        )
    """
    coordinates = _validate_inputs(
        field,
        x,
        y,
        z,
        smoothing_sigma,
        smoothing_mode,
        smoothing_truncate,
        edge_order,
        metric_tolerance,
        outlier_threshold,
        outlier_window_size,
    )
    single_snapshot = field.ndim == 2
    sequence = field.unsqueeze(-1) if single_snapshot else field
    sequence = _gaussian_smooth_2d(
        sequence, smoothing_sigma, smoothing_mode, smoothing_truncate
    )

    effective_edge_order = edge_order if min(x.shape) >= edge_order + 1 else 1
    df_di, df_dj = pt.gradient(sequence, dim=(0, 1), edge_order=effective_edge_order)
    position = pt.stack(coordinates, dim=-1)
    dr_di, dr_dj = pt.gradient(position, dim=(0, 1), edge_order=effective_edge_order)

    metric_ii = pt.sum(dr_di * dr_di, dim=-1)
    metric_ij = pt.sum(dr_di * dr_dj, dim=-1)
    metric_jj = pt.sum(dr_dj * dr_dj, dim=-1)
    determinant = metric_ii * metric_jj - metric_ij.square()
    metric_scale = metric_ii * metric_jj
    invalid = determinant <= metric_tolerance * metric_scale
    safe_determinant = pt.where(invalid, pt.ones_like(determinant), determinant)

    coefficient_i = (
        metric_jj.unsqueeze(-1) * df_di - metric_ij.unsqueeze(-1) * df_dj
    ) / safe_determinant.unsqueeze(-1)
    coefficient_j = (
        -metric_ij.unsqueeze(-1) * df_di + metric_ii.unsqueeze(-1) * df_dj
    ) / safe_determinant.unsqueeze(-1)
    gradient = coefficient_i.unsqueeze(-1) * dr_di.unsqueeze(-2)
    gradient += coefficient_j.unsqueeze(-1) * dr_dj.unsqueeze(-2)
    gradient = gradient.masked_fill(invalid[..., None, None], float("nan"))

    if outlier_threshold is not None:
        gradient = replace_spatial_outliers(
            gradient,
            threshold=outlier_threshold,
            window_size=outlier_window_size,
        )
    return gradient[:, :, 0] if single_snapshot else gradient


def _gaussian_smooth_2d(
    field: pt.Tensor,
    sigma: float | Tuple[float, float] | None,
    mode: str,
    truncate: float,
) -> pt.Tensor:
    sigma_i, sigma_j = _standardize_sigma(sigma)
    if sigma_i == 0.0 and sigma_j == 0.0:
        return field

    smoothed = field.permute(2, 0, 1).unsqueeze(1)
    if sigma_i > 0.0:
        kernel_i = _gaussian_kernel_1d(sigma_i, truncate, field)
        radius_i = kernel_i.numel() // 2
        padding_mode = _safe_padding_mode(mode, field.shape[0], radius_i)
        smoothed = F.pad(smoothed, (0, 0, radius_i, radius_i), mode=padding_mode)
        smoothed = F.conv2d(smoothed, kernel_i.reshape(1, 1, -1, 1))
    if sigma_j > 0.0:
        kernel_j = _gaussian_kernel_1d(sigma_j, truncate, field)
        radius_j = kernel_j.numel() // 2
        padding_mode = _safe_padding_mode(mode, field.shape[1], radius_j)
        smoothed = F.pad(smoothed, (radius_j, radius_j, 0, 0), mode=padding_mode)
        smoothed = F.conv2d(smoothed, kernel_j.reshape(1, 1, 1, -1))
    return smoothed.squeeze(1).permute(1, 2, 0)


def _gaussian_kernel_1d(
    sigma: float, truncate: float, reference: pt.Tensor
) -> pt.Tensor:
    radius = max(1, ceil(truncate * sigma))
    positions = pt.arange(
        -radius, radius + 1, dtype=reference.dtype, device=reference.device
    )
    kernel = pt.exp(-0.5 * (positions / sigma).square())
    return kernel / pt.sum(kernel)


def _safe_padding_mode(mode: str, size: int, radius: int) -> str:
    if mode == "reflect" and radius >= size:
        return "replicate"
    return mode


def _standardize_sigma(
    sigma: float | Tuple[float, float] | None,
) -> Tuple[float, float]:
    if sigma is None:
        return 0.0, 0.0
    if isinstance(sigma, (float, int)):
        return float(sigma), float(sigma)
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise ValueError("smoothing_sigma must be a scalar or a two-tuple")
    return float(sigma[0]), float(sigma[1])


def _validate_inputs(
    field: pt.Tensor,
    x: pt.Tensor,
    y: pt.Tensor,
    z: pt.Tensor | None,
    smoothing_sigma: float | Tuple[float, float] | None,
    smoothing_mode: str,
    smoothing_truncate: float,
    edge_order: int,
    metric_tolerance: float,
    outlier_threshold: float | None,
    outlier_window_size: int,
) -> Tuple[pt.Tensor, ...]:
    coordinates = (x, y) if z is None else (x, y, z)
    if any(coordinate.ndim != 2 for coordinate in coordinates):
        raise ValueError("coordinates must be two-dimensional")
    if any(coordinate.shape != x.shape for coordinate in coordinates[1:]):
        raise ValueError("coordinates must have identical shapes")
    if min(x.shape) < 2:
        raise ValueError("each grid dimension must contain at least two points")
    if field.ndim not in (2, 3) or field.shape[:2] != x.shape:
        raise ValueError("field must have shape (nx, ny) or (nx, ny, n_snapshots)")
    if not field.is_floating_point():
        raise ValueError("field must have a floating-point dtype")
    for coordinate in coordinates:
        if coordinate.dtype != field.dtype or coordinate.device != field.device:
            raise ValueError("field and coordinates must share dtype and device")
        if not pt.isfinite(coordinate).all():
            raise ValueError("coordinates must contain only finite values")
    if edge_order not in (1, 2):
        raise ValueError("edge_order must be 1 or 2")
    if smoothing_mode not in ("reflect", "replicate", "circular"):
        raise ValueError('smoothing_mode must be "reflect", "replicate", or "circular"')
    sigma_i, sigma_j = _standardize_sigma(smoothing_sigma)
    if not isfinite(sigma_i) or not isfinite(sigma_j):
        raise ValueError("smoothing sigma values must be finite")
    if sigma_i < 0.0 or sigma_j < 0.0:
        raise ValueError("smoothing sigma values must be non-negative")
    if not isfinite(smoothing_truncate) or smoothing_truncate <= 0.0:
        raise ValueError("smoothing_truncate must be finite and positive")
    if not isfinite(metric_tolerance) or metric_tolerance < 0.0:
        raise ValueError("metric_tolerance must be finite and non-negative")
    if outlier_threshold is not None and (
        not isfinite(outlier_threshold) or outlier_threshold <= 0.0
    ):
        raise ValueError("outlier_threshold must be positive")
    if outlier_window_size < 1 or outlier_window_size % 2 == 0:
        raise ValueError("outlier_window_size must be a positive odd integer")
    return coordinates
