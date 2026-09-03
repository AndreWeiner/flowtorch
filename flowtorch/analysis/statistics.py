"""Batched statistics and trend analysis for snapshot sequences."""

from math import isclose, isfinite
from numbers import Integral, Real
from typing import Any, Callable, Literal, NamedTuple, Optional, Sequence, Union

import torch as pt

SnapshotSource = Union[pt.Tensor, Callable[[int, int], pt.Tensor]]
MOMENT_NAMES = ("mean", "variance", "skewness", "kurtosis")
DEFAULT_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.95)
STATISTIC_NAMES = ("moments", "dependency", "trend", "spatial", "histogram")


class MomentFields(NamedTuple):
    """Mean, population variance, skewness, and Pearson kurtosis fields."""

    mean: pt.Tensor
    variance: pt.Tensor
    skewness: pt.Tensor
    kurtosis: pt.Tensor


class MomentDependencyResult(NamedTuple):
    """Fraction-dependent moments and consecutive spatial differences.

    ``reduced_moments`` has shape ``(n_fractions, 4)`` in the order given by
    :data:`MOMENT_NAMES`. ``field_difference_norms`` has shape
    ``(n_fractions - 1, 4)`` and compares each fraction with its predecessor.
    """

    fractions: pt.Tensor
    n_snapshots: pt.Tensor
    reduced_moments: pt.Tensor
    field_difference_norms: pt.Tensor
    final_fields: MomentFields
    intermediate_fields: Optional[MomentFields]


class LinearTrendResult(NamedTuple):
    """Spatial fields describing a least-squares linear temporal trend."""

    slope: pt.Tensor
    intercept: pt.Tensor
    r_squared: pt.Tensor
    normalized_change: pt.Tensor
    n_snapshots: int


class SpatialStatisticsResult(NamedTuple):
    """Spatially reduced statistics for every snapshot.

    ``minimum``, ``maximum``, and ``mean`` have shape ``(n_snapshots,)``.
    ``quantiles`` has shape ``(n_quantiles, n_snapshots)`` and follows the
    probabilities stored in ``quantile_levels``.
    """

    minimum: pt.Tensor
    maximum: pt.Tensor
    mean: pt.Tensor
    quantiles: pt.Tensor
    quantile_levels: pt.Tensor


class HistogramResult(NamedTuple):
    """Spatiotemporal histogram values and bin edges."""

    histogram: pt.Tensor
    bin_edges: pt.Tensor


class DistributedExecution(NamedTuple):
    """Select collective execution for a statistics call.

    The caller must initialize :mod:`torch.distributed` before invoking a
    collective statistics operation. ``root_rank`` is relative to the selected
    process group and is the only rank that receives a result. The collective
    implementation is compatible with Gloo, NCCL, and MPI process groups.
    """

    process_group: Optional[Any] = None
    root_rank: int = 0


class SnapshotStatisticsResult(NamedTuple):
    """Results produced by :func:`snapshot_statistics`.

    Fields excluded through ``compute`` are ``None``. Requesting dependency
    statistics also populates ``moments`` from the dependency's final fields.
    """

    moments: Optional[MomentFields]
    moment_dependency: Optional[MomentDependencyResult]
    trend: Optional[LinearTrendResult]
    spatial_statistics: Optional[SpatialStatisticsResult]
    histogram: Optional[HistogramResult]


class _MomentState(NamedTuple):
    n_snapshots: int
    mean: pt.Tensor
    m2: pt.Tensor
    m3: pt.Tensor
    m4: pt.Tensor


class _LinearTrendState(NamedTuple):
    n_snapshots: int
    mean_time: pt.Tensor
    mean_data: pt.Tensor
    time_variation: pt.Tensor
    covariation: pt.Tensor
    data_variation: pt.Tensor
    minimum_time: pt.Tensor
    maximum_time: pt.Tensor


class _LocalStatisticsResult(NamedTuple):
    moment_states: Optional[list[Optional[_MomentState]]]
    trend_state: Optional[_LinearTrendState]
    spatial_statistics: Optional[SpatialStatisticsResult]
    histogram: Optional[HistogramResult]
    range_minimum: Optional[pt.Tensor]
    range_maximum: Optional[pt.Tensor]
    reference: Optional[pt.Tensor]


def _normalize_snapshot_dim(snapshot_dim: int, ndim: int) -> int:
    if not isinstance(snapshot_dim, Integral) or isinstance(snapshot_dim, bool):
        raise ValueError("snapshot_dim must be an integer")
    normalized = int(snapshot_dim)
    if normalized < 0:
        normalized += ndim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(
            f"snapshot_dim {snapshot_dim} is invalid for data with {ndim} dimensions"
        )
    return normalized


def _validate_batch_size(batch_size: int) -> int:
    if (
        not isinstance(batch_size, Integral)
        or isinstance(batch_size, bool)
        or int(batch_size) < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    return int(batch_size)


def _source_size(
    source: SnapshotSource,
    n_snapshots: Optional[int],
    snapshot_dim: int,
) -> int:
    if n_snapshots is not None and (
        not isinstance(n_snapshots, Integral)
        or isinstance(n_snapshots, bool)
        or int(n_snapshots) < 1
    ):
        raise ValueError("n_snapshots must be a positive integer")
    if isinstance(source, pt.Tensor):
        if source.ndim < 1:
            raise ValueError("source must have at least one dimension")
        dim = _normalize_snapshot_dim(snapshot_dim, source.ndim)
        inferred = source.shape[dim]
        if n_snapshots is not None and n_snapshots != inferred:
            raise ValueError(
                f"n_snapshots ({n_snapshots}) does not match source ({inferred})"
            )
        n_total = inferred
    else:
        if not callable(source):
            raise ValueError("source must be a tensor or an indexed batch callable")
        if n_snapshots is None:
            raise ValueError("n_snapshots is required for a callable source")
        n_total = n_snapshots
    if int(n_total) < 1:
        raise ValueError("n_snapshots must be a positive integer")
    return int(n_total)


def _load_batch(
    source: SnapshotSource,
    start: int,
    stop: int,
    snapshot_dim: int,
) -> pt.Tensor:
    if isinstance(source, pt.Tensor):
        dim = _normalize_snapshot_dim(snapshot_dim, source.ndim)
        index = [slice(None)] * source.ndim
        index[dim] = slice(start, stop)
        batch = source[tuple(index)]
    else:
        batch = source(start, stop)
    if not isinstance(batch, pt.Tensor):
        raise ValueError("the snapshot source must return a tensor")
    if batch.ndim < 1:
        raise ValueError("snapshot batches must have at least one dimension")
    dim = _normalize_snapshot_dim(snapshot_dim, batch.ndim)
    expected = stop - start
    if batch.shape[dim] != expected:
        raise ValueError(
            f"the source returned {batch.shape[dim]} snapshots for [{start}:{stop}], "
            f"expected {expected}"
        )
    return batch


def _move_snapshots_last(
    data: pt.Tensor, snapshot_dim: int, validate_finite: bool = True
) -> pt.Tensor:
    if data.ndim < 1:
        raise ValueError("snapshot data must have at least one dimension")
    if not data.is_floating_point() or pt.is_complex(data):
        raise ValueError("snapshot data must have a real floating-point dtype")
    if validate_finite and not bool(pt.isfinite(data).all()):
        raise ValueError("snapshot data must contain only finite values")
    dim = _normalize_snapshot_dim(snapshot_dim, data.ndim)
    if data.shape[dim] < 1:
        raise ValueError("a batch must contain at least one snapshot")
    return pt.movedim(data, dim, -1)


class RunningMoments:
    r"""Accumulate the first four moments from batches of snapshots.

    Batches are combined using stable parallel central-moment formulas. The
    retained state consists of a count, mean, and the unnormalized central
    moments M2, M3, and M4.
    """

    def __init__(
        self,
        snapshot_dim: int = -1,
        accumulator_dtype: Optional[pt.dtype] = None,
    ):
        self._snapshot_dim = snapshot_dim
        self._accumulator_dtype = accumulator_dtype
        self._count = 0
        self._mean: Optional[pt.Tensor] = None
        self._m2: Optional[pt.Tensor] = None
        self._m3: Optional[pt.Tensor] = None
        self._m4: Optional[pt.Tensor] = None

    @property
    def count(self) -> int:
        """Number of accumulated snapshots."""
        return self._count

    def _merge_state(
        self,
        count: int,
        mean: pt.Tensor,
        m2: pt.Tensor,
        m3: pt.Tensor,
        m4: pt.Tensor,
    ) -> None:
        if self._count == 0:
            self._count = count
            self._mean = mean.clone()
            self._m2 = m2.clone()
            self._m3 = m3.clone()
            self._m4 = m4.clone()
            return

        assert self._mean is not None
        assert self._m2 is not None
        assert self._m3 is not None
        assert self._m4 is not None
        if mean.shape != self._mean.shape:
            raise ValueError("all batches must have matching spatial dimensions")
        if mean.device != self._mean.device:
            raise ValueError("all batches must be on the same device")
        if mean.dtype != self._mean.dtype:
            raise ValueError("all batches must use the same accumulator dtype")

        count_a = self._count
        count_b = count
        combined = count_a + count_b
        delta = mean - self._mean
        delta2 = delta.square()
        delta3 = delta2 * delta
        delta4 = delta2.square()
        scale = count_a * count_b / combined

        mean_combined = self._mean + delta * (count_b / combined)
        m2_combined = self._m2 + m2 + delta2 * scale
        m3_combined = (
            self._m3
            + m3
            + delta3 * scale * (count_a - count_b) / combined
            + 3.0 * delta * (count_a * m2 - count_b * self._m2) / combined
        )
        m4_combined = (
            self._m4
            + m4
            + delta4
            * scale
            * (count_a**2 - count_a * count_b + count_b**2)
            / combined**2
            + 6.0 * delta2 * (count_a**2 * m2 + count_b**2 * self._m2) / combined**2
            + 4.0 * delta * (count_a * m3 - count_b * self._m3) / combined
        )

        self._count = combined
        self._mean = mean_combined
        self._m2 = m2_combined
        self._m3 = m3_combined
        self._m4 = m4_combined

    def update(self, batch: pt.Tensor) -> None:
        """Merge a batch of snapshots into the running moments."""
        snapshots = _move_snapshots_last(batch, self._snapshot_dim)
        dtype = self._accumulator_dtype
        if dtype is not None:
            probe = pt.empty((), dtype=dtype)
            if not probe.is_floating_point() or pt.is_complex(probe):
                raise ValueError("accumulator_dtype must be a real floating type")
            snapshots = snapshots.to(dtype=dtype)
        elif self._mean is not None and snapshots.dtype != self._mean.dtype:
            raise ValueError("all batches must have the same dtype")

        count = snapshots.shape[-1]
        mean = snapshots.mean(dim=-1)
        centered = snapshots - mean.unsqueeze(-1)
        self._merge_state(
            count,
            mean,
            centered.square().sum(dim=-1),
            centered.pow(3).sum(dim=-1),
            centered.pow(4).sum(dim=-1),
        )

    def merge(self, other: "RunningMoments") -> None:
        """Merge another compatible accumulator into this instance."""
        if other.count == 0:
            return
        assert other._mean is not None
        assert other._m2 is not None
        assert other._m3 is not None
        assert other._m4 is not None
        self._merge_state(other.count, other._mean, other._m2, other._m3, other._m4)

    def _state(self) -> _MomentState:
        """Return a detached copy of the accumulator state."""
        if self._count == 0:
            raise ValueError("at least one snapshot must be accumulated")
        assert self._mean is not None
        assert self._m2 is not None
        assert self._m3 is not None
        assert self._m4 is not None
        return _MomentState(
            self._count,
            self._mean.clone(),
            self._m2.clone(),
            self._m3.clone(),
            self._m4.clone(),
        )

    def finalize(self) -> MomentFields:
        """Return the four conventional moment fields."""
        return _finalize_moment_state(self._state())


def _finalize_moment_state(state: _MomentState) -> MomentFields:
    """Convert an accumulated central-moment state to conventional moments."""
    variance = (state.m2 / state.n_snapshots).clamp_min(0.0)
    valid = variance > 0.0
    invalid = pt.full_like(variance, float("nan"))
    skewness = pt.where(
        valid, (state.m3 / state.n_snapshots) / variance.pow(1.5), invalid
    )
    kurtosis = pt.where(
        valid, (state.m4 / state.n_snapshots) / variance.square(), invalid
    )
    return MomentFields(state.mean.clone(), variance, skewness, kurtosis)


def statistical_moments(
    data: SnapshotSource,
    snapshot_dim: int = -1,
    batch_size: int = 32,
    accumulator_dtype: Optional[pt.dtype] = None,
    *,
    n_snapshots: Optional[int] = None,
    spatial_mask: Optional[pt.Tensor] = None,
    execution: Optional[DistributedExecution] = None,
) -> Optional[MomentFields]:
    """Compute mean, variance, skewness, and kurtosis in batches.

    A spatial mask preserves the field shape and fills excluded locations with
    ``NaN``. Non-finite snapshot values are permitted only outside the mask.

    A distributed call uses the same interface::

        result = statistical_moments(
            source,
            n_snapshots=100_000,
            execution=DistributedExecution(root_rank=0),
        )
        if result is not None:
            print(result.variance)

    :param data: snapshot tensor or globally indexed batch callable
    :type data: Union[pt.Tensor, Callable[[int, int], pt.Tensor]]
    :param snapshot_dim: dimension containing snapshots, defaults to ``-1``
    :type snapshot_dim: int, optional
    :param batch_size: maximum snapshots processed together, defaults to 32
    :type batch_size: int, optional
    :param accumulator_dtype: optional floating-point accumulation dtype
    :type accumulator_dtype: pt.dtype, optional
    :param n_snapshots: global count required for a callable source
    :type n_snapshots: int, optional
    :param spatial_mask: boolean mask selecting spatial locations
    :type spatial_mask: pt.Tensor, optional
    :param execution: optional collective execution policy
    :type execution: DistributedExecution, optional
    :return: four spatial moment fields
    :rtype: MomentFields
    """
    result = snapshot_statistics(
        data,
        n_snapshots,
        batch_size=batch_size,
        snapshot_dim=snapshot_dim,
        accumulator_dtype=accumulator_dtype,
        spatial_mask=spatial_mask,
        compute=("moments",),
        execution=execution,
    )
    return None if result is None else result.moments


def _prepare_fractions(
    fractions: Sequence[float], n_snapshots: int
) -> tuple[list[float], list[int]]:
    if isinstance(fractions, pt.Tensor):
        if fractions.ndim != 1:
            raise ValueError("fractions must be one-dimensional")
        values = fractions.detach().cpu().tolist()
    else:
        values = list(fractions)
    if len(values) == 0:
        raise ValueError("fractions must contain at least one value")
    if any(
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not isfinite(float(value))
        or float(value) <= 0.0
        or float(value) > 1.0
        for value in values
    ):
        raise ValueError("fractions must lie in the interval (0, 1]")
    normalized = [float(value) for value in values]
    if any(first >= second for first, second in zip(normalized, normalized[1:])):
        raise ValueError("fractions must be strictly increasing")
    if not isclose(normalized[-1], 1.0, rel_tol=0.0, abs_tol=1.0e-7):
        raise ValueError("the last fraction must be 1.0")
    normalized[-1] = 1.0
    counts = [max(1, int(value * n_snapshots)) for value in normalized]
    if any(first >= second for first, second in zip(counts, counts[1:])):
        raise ValueError("fractions must select distinct snapshot counts")
    return normalized, counts


def _prepare_spatial_weight(
    weight: Optional[pt.Tensor], field: pt.Tensor
) -> Optional[pt.Tensor]:
    if weight is None:
        return None
    if not isinstance(weight, pt.Tensor) or weight.numel() < 1:
        raise ValueError("spatial_weight must be a non-empty tensor")
    if weight.device != field.device:
        raise ValueError("spatial_weight and snapshots must be on the same device")
    if pt.is_complex(weight):
        raise ValueError("spatial_weight must be real")
    converted = weight.to(dtype=field.dtype)
    if converted.numel() == field.numel() and converted.shape != field.shape:
        converted = converted.reshape(field.shape)
    try:
        expanded = pt.broadcast_to(converted, field.shape)
    except RuntimeError as error:
        raise ValueError(
            "spatial_weight must match or broadcast to the spatial field shape"
        ) from error
    if not bool(pt.isfinite(expanded).all()) or bool((expanded < 0.0).any()):
        raise ValueError("spatial_weight values must be finite and non-negative")
    if not bool((expanded > 0.0).any()):
        raise ValueError("spatial_weight must contain at least one positive value")
    return expanded


def _prepare_spatial_mask(
    spatial_mask: Optional[pt.Tensor], field: pt.Tensor
) -> Optional[pt.Tensor]:
    """Validate and broadcast a boolean spatial-domain mask."""
    if spatial_mask is None:
        return None
    if not isinstance(spatial_mask, pt.Tensor) or spatial_mask.numel() < 1:
        raise ValueError("spatial_mask must be a non-empty tensor")
    if spatial_mask.dtype != pt.bool:
        raise ValueError("spatial_mask must have boolean dtype")
    if spatial_mask.device != field.device:
        raise ValueError("spatial_mask and snapshots must be on the same device")
    candidate = spatial_mask
    if candidate.numel() == field.numel() and candidate.shape != field.shape:
        candidate = candidate.reshape(field.shape)
    try:
        expanded = pt.broadcast_to(candidate, field.shape)
    except RuntimeError as error:
        raise ValueError(
            "spatial_mask must match or broadcast to the spatial field shape"
        ) from error
    if not bool(expanded.any()):
        raise ValueError("spatial_mask must select at least one spatial location")
    return expanded


def _prepare_analysis_weight(
    spatial_weight: Optional[pt.Tensor],
    spatial_mask: Optional[pt.Tensor],
    field: pt.Tensor,
) -> pt.Tensor:
    """Combine optional integration weights and a spatial mask."""
    weight = _prepare_spatial_weight(spatial_weight, field)
    if weight is None:
        weight = field.new_ones(field.shape)
    if spatial_mask is not None:
        weight = pt.where(spatial_mask, weight, pt.zeros_like(weight))
    if not bool((weight > 0.0).any()):
        raise ValueError(
            "spatial_weight must be positive at a location selected by spatial_mask"
        )
    return weight


def _mask_field(field: pt.Tensor, spatial_mask: Optional[pt.Tensor]) -> pt.Tensor:
    """Preserve a field shape while marking excluded locations as undefined."""
    if spatial_mask is None:
        return field
    mask = _prepare_spatial_mask(spatial_mask, field)
    assert mask is not None
    return pt.where(mask, field, pt.full_like(field, float("nan")))


def _mask_moment_fields(
    fields: MomentFields, spatial_mask: Optional[pt.Tensor]
) -> MomentFields:
    """Apply a spatial mask to every conventional moment field."""
    return MomentFields(*(_mask_field(field, spatial_mask) for field in fields))


def _mask_linear_trend(
    result: LinearTrendResult, spatial_mask: Optional[pt.Tensor]
) -> LinearTrendResult:
    """Apply a spatial mask to every field in a trend result."""
    return LinearTrendResult(
        _mask_field(result.slope, spatial_mask),
        _mask_field(result.intercept, spatial_mask),
        _mask_field(result.r_squared, spatial_mask),
        _mask_field(result.normalized_change, spatial_mask),
        result.n_snapshots,
    )


def _prepare_quantiles(quantiles: Sequence[float]) -> list[float]:
    """Validate spatial quantile probabilities."""
    if isinstance(quantiles, pt.Tensor):
        if quantiles.ndim != 1:
            raise ValueError("quantiles must be one-dimensional")
        values = quantiles.detach().cpu().tolist()
    else:
        values = list(quantiles)
    if len(values) == 0:
        raise ValueError("quantiles must contain at least one probability")
    if any(
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not isfinite(float(value))
        or float(value) < 0.0
        or float(value) > 1.0
        for value in values
    ):
        raise ValueError("quantiles must lie in the interval [0, 1]")
    normalized = [float(value) for value in values]
    if any(first >= second for first, second in zip(normalized, normalized[1:])):
        raise ValueError("quantiles must be strictly increasing")
    return normalized


def _weighted_quantiles(
    values: pt.Tensor,
    weight: pt.Tensor,
    quantiles: pt.Tensor,
) -> pt.Tensor:
    """Compute linearly interpolated weighted quantiles for a batch."""
    sorted_values, indices = values.sort(dim=0)
    expanded_weight = weight.unsqueeze(-1).expand_as(values)
    sorted_weight = expanded_weight.gather(0, indices)
    total_weight = sorted_weight.sum(dim=0, keepdim=True)
    positions = (sorted_weight.cumsum(dim=0) - 0.5 * sorted_weight) / total_weight

    if sorted_values.shape[0] == 1:
        return sorted_values.expand(quantiles.numel(), -1)

    values_by_snapshot = sorted_values.T.contiguous()
    positions_by_snapshot = positions.T.contiguous()
    queries = (
        quantiles.unsqueeze(0).expand(values_by_snapshot.shape[0], -1).contiguous()
    )
    upper = pt.searchsorted(positions_by_snapshot, queries)
    upper = upper.clamp(1, values_by_snapshot.shape[1] - 1)
    lower = upper - 1
    lower_position = positions_by_snapshot.gather(1, lower)
    upper_position = positions_by_snapshot.gather(1, upper)
    lower_value = values_by_snapshot.gather(1, lower)
    upper_value = values_by_snapshot.gather(1, upper)
    fraction = (queries - lower_position) / (upper_position - lower_position)
    interpolated = lower_value + fraction * (upper_value - lower_value)
    interpolated = pt.where(
        queries <= positions_by_snapshot[:, :1],
        values_by_snapshot[:, :1],
        interpolated,
    )
    interpolated = pt.where(
        queries >= positions_by_snapshot[:, -1:],
        values_by_snapshot[:, -1:],
        interpolated,
    )
    return interpolated.T


def spatial_statistics(
    source: SnapshotSource,
    n_snapshots: Optional[int] = None,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    batch_size: int = 32,
    snapshot_dim: int = -1,
    spatial_weight: Optional[pt.Tensor] = None,
    spatial_mask: Optional[pt.Tensor] = None,
    execution: Optional[DistributedExecution] = None,
) -> Optional[SpatialStatisticsResult]:
    r"""Compute weighted spatial statistics for every snapshot in batches.

    Quantiles use linear interpolation over weighted sample midpoints

    .. math::

        p_i = \frac{\sum_{j \leq i} w_j - w_i/2}{\sum_j w_j}.

    Values outside the midpoint range are clamped to the spatial minimum or
    maximum. Locations with zero weight or a false spatial mask do not
    contribute to any statistic.
    A callable source follows the indexed ``source(start, stop)`` contract of
    :func:`moment_data_dependency`.

    In a distributed run, the callback remains globally indexed and only the
    configured root receives the concatenated time series::

        result = spatial_statistics(
            source,
            n_snapshots=100_000,
            execution=DistributedExecution(root_rank=0),
        )

    :param source: snapshot tensor or indexed batch callable
    :type source: Union[pt.Tensor, Callable[[int, int], pt.Tensor]]
    :param n_snapshots: total count required for a callable source
    :type n_snapshots: int, optional
    :param quantiles: increasing probabilities in ``[0, 1]``
    :type quantiles: Sequence[float], optional
    :param batch_size: maximum snapshots loaded together, defaults to 32
    :type batch_size: int, optional
    :param snapshot_dim: dimension containing snapshots, defaults to ``-1``
    :type snapshot_dim: int, optional
    :param spatial_weight: non-negative spatial weights
    :type spatial_weight: pt.Tensor, optional
    :param spatial_mask: boolean mask selecting spatial locations
    :type spatial_mask: pt.Tensor, optional
    :param execution: optional collective execution policy
    :type execution: DistributedExecution, optional
    :return: spatial statistic time series
    :rtype: SpatialStatisticsResult
    """
    result = snapshot_statistics(
        source,
        n_snapshots,
        quantiles=quantiles,
        batch_size=batch_size,
        snapshot_dim=snapshot_dim,
        spatial_weight=spatial_weight,
        spatial_mask=spatial_mask,
        compute=("spatial",),
        execution=execution,
    )
    return None if result is None else result.spatial_statistics


def _prepare_histogram_bins(
    bins: Union[int, Sequence[float], pt.Tensor],
) -> Union[int, list[float]]:
    """Validate a bin count or explicit bin edges."""
    if isinstance(bins, Integral) and not isinstance(bins, bool):
        if int(bins) < 1:
            raise ValueError("bins must be a positive integer")
        return int(bins)
    if isinstance(bins, pt.Tensor):
        if bins.ndim != 1:
            raise ValueError("bin edges must be one-dimensional")
        values = bins.detach().cpu().tolist()
    elif isinstance(bins, Sequence):
        values = list(bins)
    else:
        raise ValueError("bins must be a positive integer or bin edges")
    if len(values) < 2:
        raise ValueError("bin edges must contain at least two values")
    if any(
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not isfinite(float(value))
        for value in values
    ):
        raise ValueError("bin edges must contain only finite real values")
    normalized = [float(value) for value in values]
    if any(first >= second for first, second in zip(normalized, normalized[1:])):
        raise ValueError("bin edges must be strictly increasing")
    return normalized


def _prepare_histogram_range(
    value_range: Optional[Sequence[float]],
) -> Optional[tuple[float, float]]:
    """Validate an optional histogram value range."""
    if value_range is None:
        return None
    try:
        values = list(value_range)
    except TypeError as error:
        raise ValueError("value_range must contain a minimum and maximum") from error
    if len(values) != 2:
        raise ValueError("value_range must contain a minimum and maximum")
    if any(
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not isfinite(float(value))
        for value in values
    ):
        raise ValueError("value_range values must be finite real numbers")
    minimum, maximum = (float(value) for value in values)
    if minimum >= maximum:
        raise ValueError("value_range minimum must be smaller than maximum")
    return minimum, maximum


def spatiotemporal_histogram(
    source: SnapshotSource,
    n_snapshots: Optional[int] = None,
    bins: Union[int, Sequence[float], pt.Tensor] = 50,
    value_range: Optional[Sequence[float]] = None,
    batch_size: int = 32,
    snapshot_dim: int = -1,
    spatial_weight: Optional[pt.Tensor] = None,
    spatial_mask: Optional[pt.Tensor] = None,
    density: bool = False,
    execution: Optional[DistributedExecution] = None,
) -> Optional[HistogramResult]:
    r"""Compute a histogram reduced across space and time in batches.

    Every snapshot has equal temporal weight. If ``spatial_weight`` is given,
    the weight of each spatial location is applied once per snapshot.
    Locations with zero weight or a false spatial mask are excluded. Counts
    are returned by default.
    With ``density=True``, bin values are normalized so that

    .. math::

        \sum_i h_i \Delta x_i = 1.

    Integer bins with no ``value_range`` require two batched passes over the
    source: one to discover the exact global range and one to accumulate the
    histogram. Explicit edges or a value range require only one pass. Bins are
    left-inclusive and right-exclusive, except for the final bin, which also
    includes its upper edge.

    For distributed execution, all ranks call the function and only the root
    receives the global histogram::

        result = spatiotemporal_histogram(
            source,
            n_snapshots=100_000,
            bins=50,
            value_range=(-5.0, 5.0),
            execution=DistributedExecution(root_rank=0),
        )

    :param source: snapshot tensor or indexed batch callable
    :type source: Union[pt.Tensor, Callable[[int, int], pt.Tensor]]
    :param n_snapshots: total count required for a callable source
    :type n_snapshots: int, optional
    :param bins: positive bin count or strictly increasing edges, defaults to 50
    :type bins: Union[int, Sequence[float], pt.Tensor], optional
    :param value_range: minimum and maximum for integer bins
    :type value_range: Sequence[float], optional
    :param batch_size: maximum snapshots loaded together, defaults to 32
    :type batch_size: int, optional
    :param snapshot_dim: dimension containing snapshots, defaults to ``-1``
    :type snapshot_dim: int, optional
    :param spatial_weight: non-negative spatial weights
    :type spatial_weight: pt.Tensor, optional
    :param spatial_mask: boolean mask selecting spatial locations
    :type spatial_mask: pt.Tensor, optional
    :param density: normalize by total weight and bin width, defaults to False
    :type density: bool, optional
    :param execution: optional collective execution policy
    :type execution: DistributedExecution, optional
    :return: histogram values and bin edges
    :rtype: HistogramResult
    """
    result = snapshot_statistics(
        source,
        n_snapshots,
        histogram_bins=bins,
        histogram_range=value_range,
        batch_size=batch_size,
        snapshot_dim=snapshot_dim,
        spatial_weight=spatial_weight,
        spatial_mask=spatial_mask,
        density=density,
        compute=("histogram",),
        execution=execution,
    )
    return None if result is None else result.histogram


def _spatial_reduce(
    field: pt.Tensor,
    weight: Optional[pt.Tensor],
    reduction: Literal["mean", "rms"],
) -> pt.Tensor:
    valid = pt.isfinite(field)
    safe = pt.where(valid, field, pt.zeros_like(field))
    valid_weight = (
        valid.to(field.dtype)
        if weight is None
        else pt.where(valid, weight, pt.zeros_like(weight))
    )
    denominator = valid_weight.sum()
    if not bool(denominator > 0.0):
        return field.new_tensor(float("nan"))
    if reduction == "mean":
        return (safe * valid_weight).sum() / denominator
    return ((safe.square() * valid_weight).sum() / denominator).sqrt()


def _field_difference_norm(
    first: pt.Tensor,
    second: pt.Tensor,
    weight: Optional[pt.Tensor],
) -> pt.Tensor:
    valid = pt.isfinite(first) & pt.isfinite(second)
    difference = pt.where(valid, second - first, pt.zeros_like(first))
    valid_weight = (
        valid.to(first.dtype)
        if weight is None
        else pt.where(valid, weight, pt.zeros_like(weight))
    )
    if not bool(valid_weight.sum() > 0.0):
        return first.new_tensor(float("nan"))
    return (difference.square() * valid_weight).sum().sqrt()


def _stack_moment_fields(
    fields: list[MomentFields], reference: MomentFields
) -> MomentFields:
    if fields:
        columns = list(zip(*fields))
        return MomentFields(*(pt.stack(values, dim=0) for values in columns))
    shape = (0, *reference.mean.shape)
    return MomentFields(*(field.new_empty(shape) for field in reference))


def moment_data_dependency(
    source: SnapshotSource,
    n_snapshots: Optional[int] = None,
    fractions: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    batch_size: int = 32,
    snapshot_dim: int = -1,
    spatial_weight: Optional[pt.Tensor] = None,
    spatial_mask: Optional[pt.Tensor] = None,
    spatial_reduction: Literal["mean", "rms"] = "mean",
    keep_intermediate_fields: bool = False,
    accumulator_dtype: Optional[pt.dtype] = None,
    execution: Optional[DistributedExecution] = None,
) -> Optional[MomentDependencyResult]:
    r"""Compute fraction-dependent moments in one forward batched pass.

    Every checkpoint field is reduced spatially. Weighted spatial L2 norms
    monitor field changes between consecutive fractions. Only the final fields
    are retained unless ``keep_intermediate_fields`` is enabled. Masked field
    locations remain in the output as ``NaN`` but do not enter any reduction.
    A callable source accepts integer ``start`` and ``stop`` indices and
    returns the corresponding snapshot batch.

    Distributed fractions refer to prefixes of the global snapshot sequence::

        result = moment_data_dependency(
            source,
            n_snapshots=100_000,
            fractions=(0.25, 0.5, 0.75, 1.0),
            execution=DistributedExecution(root_rank=0),
        )

    :param source: snapshot tensor or indexed batch callable
    :type source: Union[pt.Tensor, Callable[[int, int], pt.Tensor]]
    :param n_snapshots: total count required for a callable source
    :type n_snapshots: int, optional
    :param fractions: increasing data fractions ending in 1.0
    :type fractions: Sequence[float], optional
    :param batch_size: maximum snapshots loaded together, defaults to 32
    :type batch_size: int, optional
    :param snapshot_dim: dimension containing snapshots, defaults to ``-1``
    :type snapshot_dim: int, optional
    :param spatial_weight: non-negative spatial integration weights
    :type spatial_weight: pt.Tensor, optional
    :param spatial_mask: boolean mask selecting spatial locations
    :type spatial_mask: pt.Tensor, optional
    :param spatial_reduction: weighted ``"mean"`` or ``"rms"``
    :type spatial_reduction: str, optional
    :param keep_intermediate_fields: retain all non-final checkpoint fields
    :type keep_intermediate_fields: bool, optional
    :param accumulator_dtype: optional floating-point accumulation dtype
    :type accumulator_dtype: pt.dtype, optional
    :param execution: optional collective execution policy
    :type execution: DistributedExecution, optional
    :return: reduced moments, consecutive difference norms, and fields
    :rtype: MomentDependencyResult
    """
    result = snapshot_statistics(
        source,
        n_snapshots,
        fractions=fractions,
        batch_size=batch_size,
        snapshot_dim=snapshot_dim,
        spatial_weight=spatial_weight,
        spatial_reduction=spatial_reduction,
        keep_intermediate_fields=keep_intermediate_fields,
        accumulator_dtype=accumulator_dtype,
        spatial_mask=spatial_mask,
        compute=("dependency",),
        execution=execution,
    )
    return None if result is None else result.moment_dependency


class _RunningLinearTrend:
    """Accumulate centered sums for a linear least-squares fit."""

    def __init__(
        self, snapshot_dim: int, accumulator_dtype: Optional[pt.dtype]
    ) -> None:
        self.snapshot_dim = snapshot_dim
        self.accumulator_dtype = accumulator_dtype
        self.count = 0
        self.mean_time: Optional[pt.Tensor] = None
        self.mean_data: Optional[pt.Tensor] = None
        self.time_variation: Optional[pt.Tensor] = None
        self.covariation: Optional[pt.Tensor] = None
        self.data_variation: Optional[pt.Tensor] = None
        self.minimum_time: Optional[pt.Tensor] = None
        self.maximum_time: Optional[pt.Tensor] = None

    def update(self, batch: pt.Tensor, time: pt.Tensor) -> None:
        snapshots = _move_snapshots_last(batch, self.snapshot_dim)
        dtype = self.accumulator_dtype or snapshots.dtype
        probe = pt.empty((), dtype=dtype)
        if not probe.is_floating_point() or pt.is_complex(probe):
            raise ValueError("accumulator_dtype must be a real floating type")
        snapshots = snapshots.to(dtype=dtype)
        time = time.to(device=snapshots.device, dtype=dtype)
        if time.ndim != 1 or time.numel() != snapshots.shape[-1]:
            raise ValueError("each time batch must match the snapshot batch")

        count_b = snapshots.shape[-1]
        mean_time_b = time.mean()
        mean_data_b = snapshots.mean(dim=-1)
        centered_time = time - mean_time_b
        centered_data = snapshots - mean_data_b.unsqueeze(-1)
        time_variation_b = centered_time.square().sum()
        covariation_b = (centered_data * centered_time).sum(dim=-1)
        data_variation_b = centered_data.square().sum(dim=-1)

        if self.count == 0:
            self.count = count_b
            self.mean_time = mean_time_b
            self.mean_data = mean_data_b
            self.time_variation = time_variation_b
            self.covariation = covariation_b
            self.data_variation = data_variation_b
            self.minimum_time = time.min()
            self.maximum_time = time.max()
            return

        assert self.mean_time is not None
        assert self.mean_data is not None
        assert self.time_variation is not None
        assert self.covariation is not None
        assert self.data_variation is not None
        assert self.minimum_time is not None
        assert self.maximum_time is not None
        if mean_data_b.shape != self.mean_data.shape:
            raise ValueError("all batches must have matching spatial dimensions")
        if mean_data_b.device != self.mean_data.device:
            raise ValueError("all batches must be on the same device")
        if mean_data_b.dtype != self.mean_data.dtype:
            raise ValueError("all batches must use the same accumulator dtype")

        count_a = self.count
        combined = count_a + count_b
        delta_time = mean_time_b - self.mean_time
        delta_data = mean_data_b - self.mean_data
        scale = count_a * count_b / combined
        self.time_variation = (
            self.time_variation + time_variation_b + delta_time.square() * scale
        )
        self.covariation = (
            self.covariation + covariation_b + delta_time * delta_data * scale
        )
        self.data_variation = (
            self.data_variation + data_variation_b + delta_data.square() * scale
        )
        self.mean_time = self.mean_time + delta_time * (count_b / combined)
        self.mean_data = self.mean_data + delta_data * (count_b / combined)
        self.minimum_time = pt.minimum(self.minimum_time, time.min())
        self.maximum_time = pt.maximum(self.maximum_time, time.max())
        self.count = combined

    def _state(self) -> _LinearTrendState:
        """Return a detached copy of the trend accumulator state."""
        if self.count == 0:
            raise ValueError("at least one snapshot must be accumulated")
        assert self.mean_time is not None
        assert self.mean_data is not None
        assert self.time_variation is not None
        assert self.covariation is not None
        assert self.data_variation is not None
        assert self.minimum_time is not None
        assert self.maximum_time is not None
        return _LinearTrendState(
            self.count,
            self.mean_time.clone(),
            self.mean_data.clone(),
            self.time_variation.clone(),
            self.covariation.clone(),
            self.data_variation.clone(),
            self.minimum_time.clone(),
            self.maximum_time.clone(),
        )

    def finalize(self) -> LinearTrendResult:
        return _finalize_linear_trend_state(self._state())


def _finalize_linear_trend_state(state: _LinearTrendState) -> LinearTrendResult:
    if state.n_snapshots < 2:
        raise ValueError("at least two snapshots are required for a trend")
    if not bool(state.time_variation > 0.0):
        raise ValueError("time values must span a non-zero interval")

    slope = state.covariation / state.time_variation
    intercept = state.mean_data - slope * state.mean_time
    varying = state.data_variation > 0.0
    r_squared = pt.where(
        varying,
        state.covariation.square() / (state.time_variation * state.data_variation),
        pt.zeros_like(state.data_variation),
    ).clamp(0.0, 1.0)
    standard_deviation = (
        (state.data_variation / state.n_snapshots).clamp_min(0.0).sqrt()
    )
    normalized_change = pt.where(
        standard_deviation > 0.0,
        slope * (state.maximum_time - state.minimum_time) / standard_deviation,
        pt.zeros_like(slope),
    )
    return LinearTrendResult(
        slope,
        intercept,
        r_squared,
        normalized_change,
        state.n_snapshots,
    )


def linear_trend(
    source: SnapshotSource,
    n_snapshots: Optional[int] = None,
    time: Optional[Union[pt.Tensor, Sequence[float]]] = None,
    batch_size: int = 32,
    snapshot_dim: int = -1,
    accumulator_dtype: Optional[pt.dtype] = None,
    spatial_mask: Optional[pt.Tensor] = None,
    execution: Optional[DistributedExecution] = None,
) -> Optional[LinearTrendResult]:
    """Fit a batched linear temporal trend at every spatial location.

    ``normalized_change`` is the fitted change across the complete time span
    divided by the temporal population standard deviation. Masked locations
    are returned as ``NaN``. A callable source follows the same indexed
    ``source(start, stop)`` contract as :func:`moment_data_dependency`.

    Global time values are sliced with the distributed snapshot ranges::

        result = linear_trend(
            source,
            n_snapshots=100_000,
            time=time,
            execution=DistributedExecution(root_rank=0),
        )
        if result is not None:
            mask = detect_linear_trend(result, 1.0)

    :param source: snapshot tensor or indexed batch callable
    :type source: Union[pt.Tensor, Callable[[int, int], pt.Tensor]]
    :param n_snapshots: total count required for a callable source
    :type n_snapshots: int, optional
    :param time: strictly increasing snapshot times; indices are the default
    :type time: Union[pt.Tensor, Sequence[float]], optional
    :param batch_size: maximum snapshots loaded together, defaults to 32
    :type batch_size: int, optional
    :param snapshot_dim: dimension containing snapshots, defaults to ``-1``
    :type snapshot_dim: int, optional
    :param accumulator_dtype: optional floating-point accumulation dtype
    :type accumulator_dtype: pt.dtype, optional
    :param spatial_mask: boolean mask selecting spatial locations
    :type spatial_mask: pt.Tensor, optional
    :param execution: optional collective execution policy
    :type execution: DistributedExecution, optional
    :return: spatial fields describing the linear fit
    :rtype: LinearTrendResult
    """
    result = snapshot_statistics(
        source,
        n_snapshots,
        time=time,
        batch_size=batch_size,
        snapshot_dim=snapshot_dim,
        accumulator_dtype=accumulator_dtype,
        spatial_mask=spatial_mask,
        compute=("trend",),
        execution=execution,
    )
    return None if result is None else result.trend


def _prepare_compute(compute: Sequence[str]) -> frozenset[str]:
    """Validate requested joint statistics."""
    values = [compute] if isinstance(compute, str) else list(compute)
    if not values:
        raise ValueError("compute must request at least one statistic")
    if len(set(values)) != len(values):
        raise ValueError("compute must not contain duplicate statistics")
    invalid = set(values).difference(STATISTIC_NAMES)
    if invalid:
        raise ValueError(f"unknown statistics requested: {sorted(invalid)}")
    return frozenset(values)


def _prepare_time_values(
    time: Optional[Union[pt.Tensor, Sequence[float]]], n_snapshots: int
) -> pt.Tensor:
    """Validate or construct global snapshot times."""
    if time is None:
        values = pt.arange(n_snapshots, dtype=pt.float64)
    elif isinstance(time, pt.Tensor):
        values = time
    else:
        values = pt.tensor(list(time), dtype=pt.float64)
    if values.ndim != 1 or values.numel() != n_snapshots:
        raise ValueError("time must be one-dimensional and match n_snapshots")
    if pt.is_complex(values) or values.dtype == pt.bool:
        raise ValueError("time must have a real numeric dtype")
    if not bool(pt.isfinite(values).all()):
        raise ValueError("time must contain only finite values")
    if not bool((values[1:] > values[:-1]).all()):
        raise ValueError("time must be strictly increasing")
    return values


def _make_histogram_edges(
    bins: Union[int, list[float]],
    value_range: Optional[tuple[float, float]],
    reference: pt.Tensor,
    automatic_range: Optional[tuple[pt.Tensor, pt.Tensor]] = None,
) -> pt.Tensor:
    """Construct histogram edges using a snapshot field as dtype/device reference."""
    if isinstance(bins, int):
        if value_range is not None:
            lower: Union[float, pt.Tensor] = value_range[0]
            upper: Union[float, pt.Tensor] = value_range[1]
        else:
            if automatic_range is None:
                raise ValueError("an automatic histogram range is not available")
            lower, upper = automatic_range
            if bool(lower == upper):
                lower = lower - 0.5
                upper = upper + 0.5
        edges = pt.linspace(
            lower,
            upper,
            bins + 1,
            device=reference.device,
            dtype=reference.dtype,
        )
    else:
        edges = reference.new_tensor(bins)
    if not bool((edges[1:] > edges[:-1]).all()):
        raise ValueError("bin edges must remain strictly increasing in source dtype")
    return edges


def _accumulate_histogram(
    histogram: pt.Tensor,
    edges: pt.Tensor,
    flat: pt.Tensor,
    flat_weight: pt.Tensor,
) -> None:
    """Accumulate one flattened snapshot batch into fixed histogram edges."""
    values = flat.reshape(-1)
    weights = flat_weight.unsqueeze(-1).expand_as(flat).reshape(-1)
    inside = (values >= edges[0]) & (values <= edges[-1])
    selected_values = values[inside]
    selected_weights = weights[inside]
    indices = pt.bucketize(selected_values, edges, right=True) - 1
    indices = indices.clamp_max(histogram.numel() - 1)
    histogram.scatter_add_(0, indices, selected_weights)


def _run_local_statistics(
    source: SnapshotSource,
    global_start: int,
    global_stop: int,
    batch_size: int,
    snapshot_dim: int,
    compute: frozenset[str],
    checkpoint_counts: Optional[list[int]],
    time_values: Optional[pt.Tensor],
    quantile_values: Optional[list[float]],
    histogram_bins: Optional[Union[int, list[float]]],
    histogram_range: Optional[tuple[float, float]],
    spatial_weight: Optional[pt.Tensor],
    spatial_mask: Optional[pt.Tensor],
    accumulator_dtype: Optional[pt.dtype],
) -> _LocalStatisticsResult:
    """Run enabled collectors over one contiguous global snapshot interval."""
    local_count = global_stop - global_start
    needs_moments = bool({"moments", "dependency"}.intersection(compute))
    moment_accumulator = (
        RunningMoments(-1, accumulator_dtype) if needs_moments and local_count else None
    )
    trend_accumulator = (
        _RunningLinearTrend(-1, accumulator_dtype)
        if "trend" in compute and local_count
        else None
    )
    targets = (
        [min(max(count - global_start, 0), local_count) for count in checkpoint_counts]
        if checkpoint_counts is not None
        else ([local_count] if needs_moments else [])
    )
    positive_targets = sorted(set(target for target in targets if target > 0))
    captured_states: dict[int, _MomentState] = {}

    minimum: list[pt.Tensor] = []
    maximum: list[pt.Tensor] = []
    means: list[pt.Tensor] = []
    quantile_batches: list[pt.Tensor] = []
    levels: Optional[pt.Tensor] = None
    prepared_mask: Optional[pt.Tensor] = None
    prepared_weight: Optional[pt.Tensor] = None
    positive_weight: Optional[pt.Tensor] = None
    flat_weight: Optional[pt.Tensor] = None
    edges: Optional[pt.Tensor] = None
    histogram: Optional[pt.Tensor] = None
    range_minimum: Optional[pt.Tensor] = None
    range_maximum: Optional[pt.Tensor] = None
    reference: Optional[pt.Tensor] = None
    expected_shape: Optional[pt.Size] = None
    expected_device: Optional[pt.device] = None
    expected_dtype: Optional[pt.dtype] = None
    processed = 0

    for start in range(global_start, global_stop, batch_size):
        stop = min(start + batch_size, global_stop)
        snapshots = _move_snapshots_last(
            _load_batch(source, start, stop, snapshot_dim),
            snapshot_dim,
            validate_finite=False,
        )
        spatial_shape = snapshots.shape[:-1]
        if snapshots[..., 0].numel() < 1:
            raise ValueError("snapshots must contain at least one spatial value")
        if expected_shape is None:
            expected_shape = spatial_shape
            expected_device = snapshots.device
            expected_dtype = snapshots.dtype
            reference = snapshots[..., 0].clone()
            prepared_mask = _prepare_spatial_mask(spatial_mask, snapshots[..., 0])
            if {"dependency", "spatial", "histogram"}.intersection(compute):
                prepared_weight = _prepare_analysis_weight(
                    spatial_weight, prepared_mask, snapshots[..., 0]
                )
            if {"spatial", "histogram"}.intersection(compute):
                assert prepared_weight is not None
                positive_weight = prepared_weight.reshape(-1) > 0.0
                flat_weight = prepared_weight.reshape(-1)[positive_weight]
            if "spatial" in compute:
                assert quantile_values is not None
                levels = snapshots.new_tensor(quantile_values)
            if "histogram" in compute:
                assert histogram_bins is not None
                if not isinstance(histogram_bins, int) or histogram_range is not None:
                    edges = _make_histogram_edges(
                        histogram_bins, histogram_range, snapshots[..., 0]
                    )
                    histogram = edges.new_zeros(edges.numel() - 1)
        else:
            if spatial_shape != expected_shape:
                raise ValueError("all batches must have matching spatial dimensions")
            if snapshots.device != expected_device:
                raise ValueError("all batches must be on the same device")
            if snapshots.dtype != expected_dtype:
                raise ValueError("all batches must have the same dtype")

        analysis_snapshots = (
            snapshots
            if prepared_mask is None
            else pt.where(
                prepared_mask.unsqueeze(-1), snapshots, pt.zeros_like(snapshots)
            )
        )
        if not bool(pt.isfinite(analysis_snapshots).all()):
            raise ValueError(
                "snapshot data selected by spatial_mask must contain only finite values"
            )

        batch_count = snapshots.shape[-1]
        if moment_accumulator is not None:
            local_stop = processed + batch_count
            boundaries = [
                target for target in positive_targets if processed < target < local_stop
            ] + [local_stop]
            segment_start = processed
            for boundary in boundaries:
                first = segment_start - processed
                last = boundary - processed
                moment_accumulator.update(analysis_snapshots[..., first:last])
                if boundary in positive_targets:
                    captured_states[boundary] = moment_accumulator._state()
                segment_start = boundary

        if trend_accumulator is not None:
            assert time_values is not None
            trend_accumulator.update(analysis_snapshots, time_values[start:stop])

        if {"spatial", "histogram"}.intersection(compute):
            assert positive_weight is not None
            assert flat_weight is not None
            flat = snapshots.reshape(-1, batch_count)[positive_weight]
            if "spatial" in compute:
                assert levels is not None
                denominator = flat_weight.sum()
                minimum.append(flat.min(dim=0).values)
                maximum.append(flat.max(dim=0).values)
                means.append(
                    (flat * flat_weight.unsqueeze(-1)).sum(dim=0) / denominator
                )
                quantile_batches.append(_weighted_quantiles(flat, flat_weight, levels))
            if "histogram" in compute:
                if edges is None:
                    batch_minimum = flat.min()
                    batch_maximum = flat.max()
                    range_minimum = (
                        batch_minimum
                        if range_minimum is None
                        else pt.minimum(range_minimum, batch_minimum)
                    )
                    range_maximum = (
                        batch_maximum
                        if range_maximum is None
                        else pt.maximum(range_maximum, batch_maximum)
                    )
                else:
                    assert histogram is not None
                    _accumulate_histogram(histogram, edges, flat, flat_weight)
        processed += batch_count

    moment_states: Optional[list[Optional[_MomentState]]] = None
    if needs_moments:
        if local_count == 0:
            moment_states = [None for _ in targets]
        else:
            assert moment_accumulator is not None
            final_state = moment_accumulator._state()
            captured_states.setdefault(local_count, final_state)
            moment_states = [
                None if target == 0 else captured_states[target] for target in targets
            ]

    spatial_result = None
    if "spatial" in compute and local_count:
        assert levels is not None
        spatial_result = SpatialStatisticsResult(
            pt.cat(minimum),
            pt.cat(maximum),
            pt.cat(means),
            pt.cat(quantile_batches, dim=1),
            levels,
        )
    histogram_result = (
        HistogramResult(histogram, edges)
        if histogram is not None and edges is not None
        else None
    )
    trend_state = trend_accumulator._state() if trend_accumulator is not None else None
    return _LocalStatisticsResult(
        moment_states,
        trend_state,
        spatial_result,
        histogram_result,
        range_minimum,
        range_maximum,
        reference,
    )


def _build_dependency_result(
    fields: list[MomentFields],
    fraction_values: list[float],
    checkpoint_counts: list[int],
    spatial_weight: Optional[pt.Tensor],
    spatial_mask: Optional[pt.Tensor],
    spatial_reduction: Literal["mean", "rms"],
    keep_intermediate_fields: bool,
) -> MomentDependencyResult:
    """Build public dependency output from global checkpoint fields."""
    final = fields[-1]
    mask = _prepare_spatial_mask(spatial_mask, final.mean)
    weight = _prepare_analysis_weight(spatial_weight, mask, final.mean)
    reduced = pt.stack(
        [
            pt.stack(
                [_spatial_reduce(field, weight, spatial_reduction) for field in current]
            )
            for current in fields
        ]
    )
    differences = [
        pt.stack(
            [
                _field_difference_norm(first, second, weight)
                for first, second in zip(previous, current)
            ]
        )
        for previous, current in zip(fields, fields[1:])
    ]
    difference_tensor = (
        pt.stack(differences)
        if differences
        else final.mean.new_empty((0, len(MOMENT_NAMES)))
    )
    retained = fields[:-1] if keep_intermediate_fields else []
    intermediate = (
        _stack_moment_fields(retained, final) if keep_intermediate_fields else None
    )
    return MomentDependencyResult(
        final.mean.new_tensor(fraction_values),
        pt.tensor(checkpoint_counts, dtype=pt.int64, device=final.mean.device),
        reduced,
        difference_tensor,
        final,
        intermediate,
    )


def _normalize_histogram(result: HistogramResult, density: bool) -> HistogramResult:
    """Apply optional probability-density normalization."""
    if not density:
        return result
    total_weight = result.histogram.sum()
    if not bool(total_weight > 0.0):
        raise ValueError("density is undefined when no values fall inside the range")
    widths = result.bin_edges[1:] - result.bin_edges[:-1]
    return HistogramResult(result.histogram / (total_weight * widths), result.bin_edges)


def _serial_snapshot_statistics(
    source: SnapshotSource,
    n_snapshots: int,
    batch_size: int,
    snapshot_dim: int,
    compute: frozenset[str],
    fraction_values: Optional[list[float]],
    checkpoint_counts: Optional[list[int]],
    time_values: Optional[pt.Tensor],
    quantile_values: Optional[list[float]],
    histogram_bins: Optional[Union[int, list[float]]],
    histogram_range: Optional[tuple[float, float]],
    spatial_weight: Optional[pt.Tensor],
    spatial_mask: Optional[pt.Tensor],
    spatial_reduction: Literal["mean", "rms"],
    keep_intermediate_fields: bool,
    density: bool,
    accumulator_dtype: Optional[pt.dtype],
) -> SnapshotStatisticsResult:
    """Execute a joint statistics request in one process."""
    local = _run_local_statistics(
        source,
        0,
        n_snapshots,
        batch_size,
        snapshot_dim,
        compute,
        checkpoint_counts,
        time_values,
        quantile_values,
        histogram_bins,
        histogram_range,
        spatial_weight,
        spatial_mask,
        accumulator_dtype,
    )
    moments = None
    dependency = None
    if local.moment_states is not None:
        fields = [
            _mask_moment_fields(_finalize_moment_state(state), spatial_mask)
            for state in local.moment_states
            if state is not None
        ]
        moments = fields[-1]
        if "dependency" in compute:
            assert fraction_values is not None
            assert checkpoint_counts is not None
            dependency = _build_dependency_result(
                fields,
                fraction_values,
                checkpoint_counts,
                spatial_weight,
                spatial_mask,
                spatial_reduction,
                keep_intermediate_fields,
            )
            moments = dependency.final_fields

    trend = (
        _finalize_linear_trend_state(local.trend_state)
        if local.trend_state is not None
        else None
    )
    histogram = local.histogram
    if "histogram" in compute and histogram is None:
        assert local.reference is not None
        assert local.range_minimum is not None
        assert local.range_maximum is not None
        assert histogram_bins is not None
        edges = _make_histogram_edges(
            histogram_bins,
            None,
            local.reference,
            (local.range_minimum, local.range_maximum),
        )
        second = _run_local_statistics(
            source,
            0,
            n_snapshots,
            batch_size,
            snapshot_dim,
            frozenset(("histogram",)),
            None,
            None,
            None,
            _prepare_histogram_bins(edges),
            None,
            spatial_weight,
            spatial_mask,
            accumulator_dtype,
        )
        histogram = second.histogram
    if histogram is not None:
        histogram = _normalize_histogram(histogram, density)
    if trend is not None:
        trend = _mask_linear_trend(trend, spatial_mask)
    return SnapshotStatisticsResult(
        moments,
        dependency,
        trend,
        local.spatial_statistics,
        histogram,
    )


def snapshot_statistics(
    source: SnapshotSource,
    n_snapshots: Optional[int] = None,
    *,
    time: Optional[Union[pt.Tensor, Sequence[float]]] = None,
    fractions: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    histogram_bins: Union[int, Sequence[float], pt.Tensor] = 50,
    histogram_range: Optional[Sequence[float]] = None,
    batch_size: int = 32,
    snapshot_dim: int = -1,
    spatial_weight: Optional[pt.Tensor] = None,
    spatial_mask: Optional[pt.Tensor] = None,
    spatial_reduction: Literal["mean", "rms"] = "mean",
    keep_intermediate_fields: bool = False,
    density: bool = False,
    accumulator_dtype: Optional[pt.dtype] = None,
    compute: Sequence[str] = STATISTIC_NAMES,
    execution: Optional[DistributedExecution] = None,
) -> Optional[SnapshotStatisticsResult]:
    r"""Compute selected snapshot statistics through a shared batch stream.

    With explicit histogram edges or ``histogram_range``, every enabled
    statistic consumes the same loaded batch and the source is traversed once.
    An exact automatically ranged histogram requires one additional pass that
    updates only histogram counts.

    ``spatial_mask`` restricts every collector to its true locations. Spatial
    field outputs retain their original shape and contain ``NaN`` outside the
    mask. Non-finite data outside the mask is ignored.

    Distributed execution partitions globally indexed snapshots among the
    selected process-group ranks. Only ``execution.root_rank`` returns a
    result. For example, after initializing :mod:`torch.distributed`::

        execution = DistributedExecution(root_rank=0)
        result = snapshot_statistics(
            source,
            n_snapshots=100_000,
            batch_size=16,
            histogram_range=(-5.0, 5.0),
            execution=execution,
        )
        if result is not None:
            print(result.moments.mean)

    :param source: snapshot tensor or globally indexed batch callable
    :param n_snapshots: total snapshot count required for a callable source
    :param time: strictly increasing global snapshot times
    :param fractions: increasing dependency fractions ending in 1.0
    :param quantiles: increasing spatial quantile probabilities
    :param histogram_bins: histogram bin count or explicit edges
    :param histogram_range: optional range for integer histogram bins
    :param batch_size: maximum snapshots loaded by each rank at once
    :param snapshot_dim: dimension containing snapshots
    :param spatial_weight: non-negative spatial weights
    :param spatial_mask: boolean mask selecting spatial locations
    :param spatial_reduction: dependency reduction, ``"mean"`` or ``"rms"``
    :param keep_intermediate_fields: retain dependency checkpoint fields
    :param density: normalize histogram counts to a density
    :param accumulator_dtype: optional moment and trend accumulation dtype
    :param compute: selected statistics; all are enabled by default
    :param execution: optional collective execution policy
    :return: selected results on the serial or distributed root process
    """
    size = _validate_batch_size(batch_size)
    n_total = _source_size(source, n_snapshots, snapshot_dim)
    selected = _prepare_compute(compute)
    if not isinstance(keep_intermediate_fields, bool):
        raise ValueError("keep_intermediate_fields must be a boolean")
    if spatial_reduction not in ("mean", "rms"):
        raise ValueError("spatial_reduction must be 'mean' or 'rms'")

    fraction_values = None
    checkpoint_counts = None
    if "dependency" in selected:
        fraction_values, checkpoint_counts = _prepare_fractions(fractions, n_total)
    elif "moments" in selected:
        checkpoint_counts = [n_total]
    time_values = _prepare_time_values(time, n_total) if "trend" in selected else None
    quantile_values = _prepare_quantiles(quantiles) if "spatial" in selected else None
    prepared_bins = None
    prepared_range = None
    if "histogram" in selected:
        prepared_bins = _prepare_histogram_bins(histogram_bins)
        prepared_range = _prepare_histogram_range(histogram_range)
        if not isinstance(prepared_bins, int) and prepared_range is not None:
            raise ValueError("histogram_range cannot be used with explicit bin edges")
        if not isinstance(density, bool):
            raise ValueError("density must be a boolean")
    if execution is not None and not isinstance(execution, DistributedExecution):
        raise ValueError("execution must be a DistributedExecution instance")

    arguments = (
        source,
        n_total,
        size,
        snapshot_dim,
        selected,
        fraction_values,
        checkpoint_counts,
        time_values,
        quantile_values,
        prepared_bins,
        prepared_range,
        spatial_weight,
        spatial_mask,
        spatial_reduction,
        keep_intermediate_fields,
        density,
        accumulator_dtype,
    )
    if execution is None:
        return _serial_snapshot_statistics(*arguments)
    from ._distributed_statistics import _distributed_snapshot_statistics

    return _distributed_snapshot_statistics(*arguments, execution)


def detect_linear_trend(
    result: LinearTrendResult,
    min_normalized_change: float,
    min_r_squared: float = 0.5,
) -> pt.Tensor:
    """Classify trends using explicit effect-size and fit thresholds.

    :param result: output of :func:`linear_trend`
    :type result: LinearTrendResult
    :param min_normalized_change: minimum absolute fitted total change in
        temporal standard deviations
    :type min_normalized_change: float
    :param min_r_squared: minimum explained variance, defaults to 0.5
    :type min_r_squared: float, optional
    :return: boolean spatial trend mask
    :rtype: pt.Tensor
    """
    if (
        not isinstance(min_normalized_change, Real)
        or isinstance(min_normalized_change, bool)
        or not isfinite(float(min_normalized_change))
        or float(min_normalized_change) <= 0.0
    ):
        raise ValueError("min_normalized_change must be finite and positive")
    if (
        not isinstance(min_r_squared, Real)
        or isinstance(min_r_squared, bool)
        or not isfinite(float(min_r_squared))
        or float(min_r_squared) < 0.0
        or float(min_r_squared) > 1.0
    ):
        raise ValueError("min_r_squared must lie in the interval [0, 1]")
    return (result.normalized_change.abs() >= float(min_normalized_change)) & (
        result.r_squared >= float(min_r_squared)
    )


__all__ = [
    "DEFAULT_QUANTILES",
    "detect_linear_trend",
    "DistributedExecution",
    "HistogramResult",
    "linear_trend",
    "LinearTrendResult",
    "MOMENT_NAMES",
    "moment_data_dependency",
    "MomentDependencyResult",
    "MomentFields",
    "RunningMoments",
    "snapshot_statistics",
    "SnapshotStatisticsResult",
    "spatial_statistics",
    "SpatialStatisticsResult",
    "statistical_moments",
    "STATISTIC_NAMES",
    "spatiotemporal_histogram",
]
