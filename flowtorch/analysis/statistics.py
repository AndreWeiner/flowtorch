"""Batched statistics and trend analysis for snapshot sequences."""

from math import isclose, isfinite
from numbers import Integral, Real
from typing import Callable, Literal, NamedTuple, Optional, Sequence, Union

import torch as pt

SnapshotSource = Union[pt.Tensor, Callable[[int, int], pt.Tensor]]
MOMENT_NAMES = ("mean", "variance", "skewness", "kurtosis")
DEFAULT_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.95)


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


def _move_snapshots_last(data: pt.Tensor, snapshot_dim: int) -> pt.Tensor:
    if data.ndim < 1:
        raise ValueError("snapshot data must have at least one dimension")
    if not data.is_floating_point() or pt.is_complex(data):
        raise ValueError("snapshot data must have a real floating-point dtype")
    if not bool(pt.isfinite(data).all()):
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

    def finalize(self) -> MomentFields:
        """Return the four conventional moment fields."""
        if self._count == 0:
            raise ValueError("at least one snapshot must be accumulated")
        assert self._mean is not None
        assert self._m2 is not None
        assert self._m3 is not None
        assert self._m4 is not None
        variance = (self._m2 / self._count).clamp_min(0.0)
        valid = variance > 0.0
        invalid = pt.full_like(variance, float("nan"))
        skewness = pt.where(
            valid, (self._m3 / self._count) / variance.pow(1.5), invalid
        )
        kurtosis = pt.where(
            valid, (self._m4 / self._count) / variance.square(), invalid
        )
        return MomentFields(self._mean.clone(), variance, skewness, kurtosis)


def statistical_moments(
    data: pt.Tensor,
    snapshot_dim: int = -1,
    batch_size: int = 32,
    accumulator_dtype: Optional[pt.dtype] = None,
) -> MomentFields:
    """Compute mean, variance, skewness, and kurtosis in batches.

    :param data: snapshot sequence
    :type data: pt.Tensor
    :param snapshot_dim: dimension containing snapshots, defaults to ``-1``
    :type snapshot_dim: int, optional
    :param batch_size: maximum snapshots processed together, defaults to 32
    :type batch_size: int, optional
    :param accumulator_dtype: optional floating-point accumulation dtype
    :type accumulator_dtype: pt.dtype, optional
    :return: four spatial moment fields
    :rtype: MomentFields
    """
    size = _validate_batch_size(batch_size)
    n_total = _source_size(data, None, snapshot_dim)
    accumulator = RunningMoments(snapshot_dim, accumulator_dtype)
    for start in range(0, n_total, size):
        stop = min(start + size, n_total)
        accumulator.update(_load_batch(data, start, stop, snapshot_dim))
    return accumulator.finalize()


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
) -> SpatialStatisticsResult:
    r"""Compute weighted spatial statistics for every snapshot in batches.

    Quantiles use linear interpolation over weighted sample midpoints

    .. math::

        p_i = \frac{\sum_{j \leq i} w_j - w_i/2}{\sum_j w_j}.

    Values outside the midpoint range are clamped to the spatial minimum or
    maximum. Locations with zero weight do not contribute to any statistic.
    A callable source follows the indexed ``source(start, stop)`` contract of
    :func:`moment_data_dependency`.

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
    :return: spatial statistic time series
    :rtype: SpatialStatisticsResult
    """
    size = _validate_batch_size(batch_size)
    n_total = _source_size(source, n_snapshots, snapshot_dim)
    quantile_values = _prepare_quantiles(quantiles)

    minimum: list[pt.Tensor] = []
    maximum: list[pt.Tensor] = []
    mean: list[pt.Tensor] = []
    quantile_batches: list[pt.Tensor] = []
    expected_shape: Optional[pt.Size] = None
    expected_device: Optional[pt.device] = None
    expected_dtype: Optional[pt.dtype] = None
    positive_weight: Optional[pt.Tensor] = None
    flat_weight: Optional[pt.Tensor] = None
    levels: Optional[pt.Tensor] = None

    for start in range(0, n_total, size):
        stop = min(start + size, n_total)
        batch = _load_batch(source, start, stop, snapshot_dim)
        snapshots = _move_snapshots_last(batch, snapshot_dim)
        spatial_shape = snapshots.shape[:-1]
        spatial_size = snapshots[..., 0].numel()
        if spatial_size < 1:
            raise ValueError("snapshots must contain at least one spatial value")

        if expected_shape is None:
            expected_shape = spatial_shape
            expected_device = snapshots.device
            expected_dtype = snapshots.dtype
            prepared_weight = _prepare_spatial_weight(spatial_weight, snapshots[..., 0])
            if prepared_weight is None:
                prepared_weight = snapshots.new_ones(spatial_shape)
            positive_weight = prepared_weight.reshape(-1) > 0.0
            flat_weight = prepared_weight.reshape(-1)[positive_weight]
            levels = snapshots.new_tensor(quantile_values)
        else:
            if spatial_shape != expected_shape:
                raise ValueError("all batches must have matching spatial dimensions")
            if snapshots.device != expected_device:
                raise ValueError("all batches must be on the same device")
            if snapshots.dtype != expected_dtype:
                raise ValueError("all batches must have the same dtype")

        assert positive_weight is not None
        assert flat_weight is not None
        assert levels is not None
        flat = snapshots.reshape(-1, snapshots.shape[-1])[positive_weight]
        denominator = flat_weight.sum()
        minimum.append(flat.min(dim=0).values)
        maximum.append(flat.max(dim=0).values)
        mean.append((flat * flat_weight.unsqueeze(-1)).sum(dim=0) / denominator)
        quantile_batches.append(_weighted_quantiles(flat, flat_weight, levels))

    assert levels is not None
    return SpatialStatisticsResult(
        minimum=pt.cat(minimum),
        maximum=pt.cat(maximum),
        mean=pt.cat(mean),
        quantiles=pt.cat(quantile_batches, dim=1),
        quantile_levels=levels,
    )


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
    spatial_reduction: Literal["mean", "rms"] = "mean",
    keep_intermediate_fields: bool = False,
    accumulator_dtype: Optional[pt.dtype] = None,
) -> MomentDependencyResult:
    r"""Compute fraction-dependent moments in one forward batched pass.

    Every checkpoint field is reduced spatially. Weighted spatial L2 norms
    monitor field changes between consecutive fractions. Only the final fields
    are retained unless ``keep_intermediate_fields`` is enabled. A callable
    source accepts integer ``start`` and ``stop`` indices and returns the
    corresponding snapshot batch.

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
    :param spatial_reduction: weighted ``"mean"`` or ``"rms"``
    :type spatial_reduction: str, optional
    :param keep_intermediate_fields: retain all non-final checkpoint fields
    :type keep_intermediate_fields: bool, optional
    :param accumulator_dtype: optional floating-point accumulation dtype
    :type accumulator_dtype: pt.dtype, optional
    :return: reduced moments, consecutive difference norms, and fields
    :rtype: MomentDependencyResult
    """
    size = _validate_batch_size(batch_size)
    if spatial_reduction not in ("mean", "rms"):
        raise ValueError("spatial_reduction must be 'mean' or 'rms'")
    if not isinstance(keep_intermediate_fields, bool):
        raise ValueError("keep_intermediate_fields must be a boolean")
    n_total = _source_size(source, n_snapshots, snapshot_dim)
    fraction_values, checkpoint_counts = _prepare_fractions(fractions, n_total)

    accumulator = RunningMoments(snapshot_dim, accumulator_dtype)
    previous: Optional[MomentFields] = None
    retained: list[MomentFields] = []
    reduced: list[pt.Tensor] = []
    differences: list[pt.Tensor] = []
    prepared_weight: Optional[pt.Tensor] = None
    start = 0

    for checkpoint_index, checkpoint in enumerate(checkpoint_counts):
        while start < checkpoint:
            stop = min(start + size, checkpoint)
            accumulator.update(_load_batch(source, start, stop, snapshot_dim))
            start = stop

        current = accumulator.finalize()
        if checkpoint_index == 0:
            prepared_weight = _prepare_spatial_weight(spatial_weight, current.mean)
        current_values = tuple(current)
        reduced.append(
            pt.stack(
                [
                    _spatial_reduce(field, prepared_weight, spatial_reduction)
                    for field in current_values
                ]
            )
        )
        if previous is not None:
            differences.append(
                pt.stack(
                    [
                        _field_difference_norm(first, second, prepared_weight)
                        for first, second in zip(previous, current)
                    ]
                )
            )
        if keep_intermediate_fields and checkpoint_index < len(checkpoint_counts) - 1:
            retained.append(current)
        previous = current

    assert previous is not None
    difference_tensor = (
        pt.stack(differences, dim=0) if differences else previous.mean.new_empty((0, 4))
    )
    intermediate = (
        _stack_moment_fields(retained, previous) if keep_intermediate_fields else None
    )
    return MomentDependencyResult(
        fractions=previous.mean.new_tensor(fraction_values),
        n_snapshots=pt.tensor(
            checkpoint_counts, dtype=pt.int64, device=previous.mean.device
        ),
        reduced_moments=pt.stack(reduced, dim=0),
        field_difference_norms=difference_tensor,
        final_fields=previous,
        intermediate_fields=intermediate,
    )


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

    def finalize(self) -> LinearTrendResult:
        if self.count < 2:
            raise ValueError("at least two snapshots are required for a trend")
        assert self.mean_time is not None
        assert self.mean_data is not None
        assert self.time_variation is not None
        assert self.covariation is not None
        assert self.data_variation is not None
        assert self.minimum_time is not None
        assert self.maximum_time is not None
        if not bool(self.time_variation > 0.0):
            raise ValueError("time values must span a non-zero interval")

        slope = self.covariation / self.time_variation
        intercept = self.mean_data - slope * self.mean_time
        varying = self.data_variation > 0.0
        r_squared = pt.where(
            varying,
            self.covariation.square() / (self.time_variation * self.data_variation),
            pt.zeros_like(self.data_variation),
        ).clamp(0.0, 1.0)
        standard_deviation = (self.data_variation / self.count).clamp_min(0.0).sqrt()
        normalized_change = pt.where(
            standard_deviation > 0.0,
            slope * (self.maximum_time - self.minimum_time) / standard_deviation,
            pt.zeros_like(slope),
        )
        return LinearTrendResult(
            slope,
            intercept,
            r_squared,
            normalized_change,
            self.count,
        )


def linear_trend(
    source: SnapshotSource,
    n_snapshots: Optional[int] = None,
    time: Optional[Union[pt.Tensor, Sequence[float]]] = None,
    batch_size: int = 32,
    snapshot_dim: int = -1,
    accumulator_dtype: Optional[pt.dtype] = None,
) -> LinearTrendResult:
    """Fit a batched linear temporal trend at every spatial location.

    ``normalized_change`` is the fitted change across the complete time span
    divided by the temporal population standard deviation. A callable source
    follows the same indexed ``source(start, stop)`` contract as
    :func:`moment_data_dependency`.

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
    :return: spatial fields describing the linear fit
    :rtype: LinearTrendResult
    """
    size = _validate_batch_size(batch_size)
    n_total = _source_size(source, n_snapshots, snapshot_dim)
    if time is None:
        time_values = pt.arange(n_total, dtype=pt.float64)
    elif isinstance(time, pt.Tensor):
        time_values = time
    else:
        time_values = pt.tensor(list(time), dtype=pt.float64)
    if time_values.ndim != 1 or time_values.numel() != n_total:
        raise ValueError("time must be one-dimensional and match n_snapshots")
    if pt.is_complex(time_values) or time_values.dtype == pt.bool:
        raise ValueError("time must have a real numeric dtype")
    if not bool(pt.isfinite(time_values).all()):
        raise ValueError("time must contain only finite values")
    if not bool((time_values[1:] > time_values[:-1]).all()):
        raise ValueError("time must be strictly increasing")

    accumulator = _RunningLinearTrend(snapshot_dim, accumulator_dtype)
    for start in range(0, n_total, size):
        stop = min(start + size, n_total)
        accumulator.update(
            _load_batch(source, start, stop, snapshot_dim),
            time_values[start:stop],
        )
    return accumulator.finalize()


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
    "linear_trend",
    "LinearTrendResult",
    "MOMENT_NAMES",
    "moment_data_dependency",
    "MomentDependencyResult",
    "MomentFields",
    "RunningMoments",
    "spatial_statistics",
    "SpatialStatisticsResult",
    "statistical_moments",
]
