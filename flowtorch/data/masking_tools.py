"""Automatic masks for time-dependent image-like data."""

# standard library packages
from math import ceil, isfinite
from typing import Literal, NamedTuple, Sequence

# third party packages
import torch as pt

# flowtorch packages
from .outlier_tools import _spatial_median_inward


class ImageMaskDiagnostics(NamedTuple):
    """Intermediate results and effective settings of image-mask creation."""

    extreme_mask: pt.Tensor
    extreme_count: pt.Tensor
    spatial_mask: pt.Tensor
    spatial_score: pt.Tensor
    patch_size: tuple[int, int]
    grid_spacing: tuple[float, float] | None
    physical_half_width: float | None


def create_image_mask(
    data: pt.Tensor,
    x: pt.Tensor | None = None,
    y: pt.Tensor | None = None,
    z: pt.Tensor | None = None,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    patch_size: int | Sequence[int] = 5,
    physical_half_width: float | None = None,
    threshold: float = 10.0,
    temporal_reduction: Literal["max", "mean"] = "max",
    chunk_size: int = 16,
) -> tuple[pt.Tensor, ImageMaskDiagnostics]:
    """Create a keep mask for image-like data using extreme and spatial filters.

    ``data`` may be a single image with shape ``(nx, ny)`` or a time series
    with shape ``(nx, ny, n_snapshots)``. The returned mask has shape
    ``(nx, ny)`` and is ``True`` where a pixel should be retained. Non-finite
    values are always rejected. Finite extreme values are rejected only when
    the corresponding lower or upper bound is supplied.

    The spatial filter compares every value with the median of a surrounding
    patch. For each snapshot, the residual is centered by its spatial median
    and scaled by its median absolute deviation (MAD). The resulting robust
    scores are reduced over the snapshots and compared with ``threshold``.

    Without coordinates, ``patch_size`` is measured in logical pixels. If
    ``physical_half_width`` and ``x``, ``y`` and optionally ``z`` are supplied,
    robust neighbor spacings are used to choose an anisotropic logical patch
    with approximately the requested physical reach along both grid axes.
    Supplying ``z`` accounts for surface curvature when estimating spacing;
    the filter neighborhood remains rectangular in logical grid indices.

    :param data: image or image sequence with time along the last dimension
    :type data: pt.Tensor
    :param x: curvilinear x-coordinate
    :type x: pt.Tensor, optional
    :param y: curvilinear y-coordinate
    :type y: pt.Tensor, optional
    :param z: optional curvilinear z-coordinate
    :type z: pt.Tensor, optional
    :param lower_bound: reject finite values smaller than this bound
    :type lower_bound: float, optional
    :param upper_bound: reject finite values larger than this bound
    :type upper_bound: float, optional
    :param patch_size: odd pixel patch size, defaults to 5
    :type patch_size: int or Sequence[int], optional
    :param physical_half_width: physical half-width used to derive the patch
    :type physical_half_width: float, optional
    :param threshold: robust spatial anomaly-score threshold, defaults to 10
    :type threshold: float, optional
    :param temporal_reduction: reduce spatial scores using ``"max"`` or
        ``"mean"``, defaults to ``"max"``
    :type temporal_reduction: str, optional
    :param chunk_size: number of snapshots processed together, defaults to 16
    :type chunk_size: int, optional
    :return: keep mask and mask diagnostics
    :rtype: Tuple[pt.Tensor, ImageMaskDiagnostics]

    **Examples**

    .. code-block:: python

        keep, diagnostics = create_image_mask(
            images,
            lower_bound=-5.0,
            upper_bound=2.0,
            patch_size=5,
            threshold=10.0,
        )

        keep_surface, diagnostics = create_image_mask(
            images,
            x=x,
            y=y,
            z=z,
            physical_half_width=0.002,
        )
    """
    _validate_data(data)
    lower_bound, upper_bound = _validate_options(
        lower_bound,
        upper_bound,
        physical_half_width,
        threshold,
        temporal_reduction,
        chunk_size,
    )
    nominal_patch = _validate_patch_size(patch_size)
    coordinates = _validate_coordinates(data.shape[:2], data, x, y, z)
    effective_patch = nominal_patch
    spacing = None
    if physical_half_width is not None:
        if coordinates is None:
            raise ValueError("coordinates are required with physical_half_width")
        spacing = _estimate_grid_spacing(coordinates)
        effective_patch = (
            2 * max(1, ceil(physical_half_width / spacing[0])) + 1,
            2 * max(1, ceil(physical_half_width / spacing[1])) + 1,
        )
    if any(size > extent for size, extent in zip(effective_patch, data.shape[:2])):
        raise ValueError("patch dimensions cannot exceed the spatial dimensions")

    sequence = data.unsqueeze(-1) if data.ndim == 2 else data
    nx, ny, n_snapshots = sequence.shape
    extreme_mask = pt.zeros((nx, ny), dtype=pt.bool, device=data.device)
    extreme_count = pt.zeros((nx, ny), dtype=pt.int64, device=data.device)
    score_dtype = pt.float64 if temporal_reduction == "mean" else data.dtype
    spatial_score = pt.zeros((nx, ny), dtype=score_dtype, device=data.device)

    for start in range(0, n_snapshots, chunk_size):
        frames = sequence[..., start : start + chunk_size].permute(2, 0, 1)
        extreme = ~pt.isfinite(frames)
        if lower_bound is not None:
            extreme |= frames < lower_bound
        if upper_bound is not None:
            extreme |= frames > upper_bound
        extreme_mask |= pt.any(extreme, dim=0)
        extreme_count += pt.sum(extreme, dim=0, dtype=pt.int64)

        scores = _spatial_anomaly_scores(frames, effective_patch)
        if temporal_reduction == "max":
            spatial_score = pt.maximum(spatial_score, pt.amax(scores, dim=0))
        else:
            spatial_score += pt.sum(scores.to(pt.float64), dim=0)

    if temporal_reduction == "mean":
        spatial_score = (spatial_score / n_snapshots).to(data.dtype)
    spatial_mask = spatial_score > threshold
    keep_mask = ~(extreme_mask | spatial_mask)
    diagnostics = ImageMaskDiagnostics(
        extreme_mask,
        extreme_count,
        spatial_mask,
        spatial_score,
        effective_patch,
        spacing,
        physical_half_width,
    )
    return keep_mask, diagnostics


def _spatial_anomaly_scores(
    frames: pt.Tensor, patch_size: tuple[int, int]
) -> pt.Tensor:
    finite_frames = pt.where(
        pt.isfinite(frames), frames, pt.full_like(frames, float("nan"))
    )
    local_median = _spatial_median_inward(finite_frames, patch_size)
    residual = frames - local_median
    flattened = residual.flatten(start_dim=1)
    offset = pt.nanmedian(flattened, dim=1).values
    deviation = pt.abs(residual - offset[:, None, None])
    mad = pt.nanmedian(deviation.flatten(start_dim=1), dim=1).values
    scale = pt.clamp(1.4826 * mad, min=pt.finfo(frames.dtype).eps)
    scores = deviation / scale[:, None, None]
    return pt.where(pt.isfinite(scores), scores, pt.zeros_like(scores))


def _validate_data(data: pt.Tensor) -> None:
    if data.ndim not in (2, 3):
        raise ValueError("data must have shape (nx, ny) or (nx, ny, n_snapshots)")
    if not data.is_floating_point():
        raise ValueError("data must have a floating-point dtype")
    if min(data.shape[:2]) < 3:
        raise ValueError("each spatial dimension must contain at least three pixels")
    if data.ndim == 3 and data.shape[-1] < 1:
        raise ValueError("data must contain at least one snapshot")


def _validate_options(
    lower_bound: float | None,
    upper_bound: float | None,
    physical_half_width: float | None,
    threshold: float,
    temporal_reduction: str,
    chunk_size: int,
) -> tuple[float | None, float | None]:
    if lower_bound is not None and not isfinite(lower_bound):
        raise ValueError("lower_bound must be finite")
    if upper_bound is not None and not isfinite(upper_bound):
        raise ValueError("upper_bound must be finite")
    if lower_bound is not None and upper_bound is not None:
        if lower_bound >= upper_bound:
            raise ValueError("lower_bound must be smaller than upper_bound")
    if physical_half_width is not None:
        if not isfinite(physical_half_width) or physical_half_width <= 0.0:
            raise ValueError("physical_half_width must be positive and finite")
    if not isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be positive and finite")
    if temporal_reduction not in ("max", "mean"):
        raise ValueError('temporal_reduction must be "max" or "mean"')
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer")
    return lower_bound, upper_bound


def _validate_patch_size(patch_size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(patch_size, bool):
        raise ValueError("patch_size must be an integer or a pair")
    if isinstance(patch_size, int):
        patch = (patch_size, patch_size)
    else:
        values = tuple(patch_size)
        if len(values) != 2:
            raise ValueError("patch dimensions must be odd integers of at least 3")
        patch = (values[0], values[1])
    if len(patch) != 2 or any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 3
        or value % 2 == 0
        for value in patch
    ):
        raise ValueError("patch dimensions must be odd integers of at least 3")
    return patch[0], patch[1]


def _validate_coordinates(
    shape: pt.Size,
    data: pt.Tensor,
    x: pt.Tensor | None,
    y: pt.Tensor | None,
    z: pt.Tensor | None,
) -> tuple[pt.Tensor, ...] | None:
    if x is None and y is None and z is None:
        return None
    if x is None or y is None:
        raise ValueError("x and y must be supplied together")
    coordinates = (x, y) if z is None else (x, y, z)
    if any(coordinate.shape != shape for coordinate in coordinates):
        raise ValueError("coordinates must match the spatial data shape")
    if any(not coordinate.is_floating_point() for coordinate in coordinates):
        raise ValueError("coordinates must have floating-point dtypes")
    if any(coordinate.device != data.device for coordinate in coordinates):
        raise ValueError("coordinates and data must be on the same device")
    if any(not pt.isfinite(coordinate).all() for coordinate in coordinates):
        raise ValueError("coordinates must contain only finite values")
    return coordinates


def _estimate_grid_spacing(coordinates: tuple[pt.Tensor, ...]) -> tuple[float, float]:
    grid = pt.stack(coordinates, dim=-1)
    differences_i = pt.linalg.vector_norm(grid[1:] - grid[:-1], dim=-1)
    differences_j = pt.linalg.vector_norm(grid[:, 1:] - grid[:, :-1], dim=-1)
    spacing = (
        float(pt.median(differences_i).item()),
        float(pt.median(differences_j).item()),
    )
    if any(not isfinite(value) or value <= 0.0 for value in spacing):
        raise ValueError("median grid-neighbor spacings must be positive")
    return spacing
