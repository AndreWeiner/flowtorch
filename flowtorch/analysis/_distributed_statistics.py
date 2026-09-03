"""Collective implementation for the public snapshot statistics interface."""

from typing import Optional, Union

import torch as pt
import torch.distributed as dist

from .statistics import (
    DistributedExecution,
    HistogramResult,
    LinearTrendResult,
    MomentFields,
    SnapshotSource,
    SnapshotStatisticsResult,
    SpatialStatisticsResult,
    _LinearTrendState,
    _LocalStatisticsResult,
    _MomentState,
    _build_dependency_result,
    _finalize_linear_trend_state,
    _finalize_moment_state,
    _make_histogram_edges,
    _mask_linear_trend,
    _mask_moment_fields,
    _normalize_histogram,
    _prepare_histogram_bins,
    _run_local_statistics,
)


def _partition(n_snapshots: int, rank: int, world_size: int) -> tuple[int, int]:
    """Return a balanced contiguous interval for one group-local rank."""
    common, remainder = divmod(n_snapshots, world_size)
    start = rank * common + min(rank, remainder)
    stop = start + common + (1 if rank < remainder else 0)
    return start, stop


def _all_gather_objects(value, group):
    """Gather trusted, compact control metadata on every rank."""
    values = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    return values


def _collective_error(local_error: Optional[str], group) -> None:
    """Raise the same local-processing failure on every participating rank."""
    errors = _all_gather_objects(local_error, group)
    failures = [error for error in errors if error is not None]
    if failures:
        raise RuntimeError(f"distributed statistics failed: {failures[0]}")


def _reference_metadata(local: _LocalStatisticsResult, group) -> pt.Tensor:
    """Validate active-rank metadata and construct an idle-rank reference."""
    metadata = (
        None
        if local.reference is None
        else (
            tuple(local.reference.shape),
            local.reference.dtype,
            local.reference.device.type,
        )
    )
    gathered = _all_gather_objects(metadata, group)
    active = [value for value in gathered if value is not None]
    if not active:
        raise RuntimeError("distributed statistics received no snapshot data")
    expected = active[0]
    if any(value != expected for value in active[1:]):
        raise ValueError("all ranks must return matching spatial shapes and dtypes")
    if local.reference is not None:
        return local.reference
    shape, dtype, device_type = expected
    if device_type == "cuda":
        device = pt.device("cuda", pt.cuda.current_device())
    else:
        device = pt.device(device_type)
    return pt.empty(shape, dtype=dtype, device=device)


def _distributed_moment_fields(
    state: Optional[_MomentState],
    reference: pt.Tensor,
    accumulator_dtype: Optional[pt.dtype],
    group,
) -> MomentFields:
    """Merge centered moment states with numerically stable collective sums."""
    dtype = (
        state.mean.dtype if state is not None else accumulator_dtype or reference.dtype
    )
    template = reference.to(dtype=dtype)
    local_count = 0 if state is None else state.n_snapshots
    count = pt.tensor(local_count, dtype=pt.int64, device=reference.device)
    dist.all_reduce(count, op=dist.ReduceOp.SUM, group=group)
    global_count = int(count.item())
    if global_count < 1:
        raise ValueError("at least one snapshot must contribute to moments")

    local_mean = pt.zeros_like(template) if state is None else state.mean
    mean_numerator = local_mean * local_count
    dist.all_reduce(mean_numerator, op=dist.ReduceOp.SUM, group=group)
    global_mean = mean_numerator / global_count
    delta = local_mean - global_mean if local_count else pt.zeros_like(global_mean)
    if state is None:
        local_m2 = pt.zeros_like(template)
        local_m3 = pt.zeros_like(template)
        local_m4 = pt.zeros_like(template)
    else:
        local_m2, local_m3, local_m4 = state.m2, state.m3, state.m4
    corrected = pt.stack(
        (
            local_m2 + local_count * delta.square(),
            local_m3 + 3.0 * delta * local_m2 + local_count * delta.pow(3),
            local_m4
            + 4.0 * delta * local_m3
            + 6.0 * delta.square() * local_m2
            + local_count * delta.pow(4),
        )
    )
    dist.all_reduce(corrected, op=dist.ReduceOp.SUM, group=group)
    return _finalize_moment_state(
        _MomentState(
            global_count,
            global_mean,
            corrected[0],
            corrected[1],
            corrected[2],
        )
    )


def _distributed_trend(
    state: Optional[_LinearTrendState],
    reference: pt.Tensor,
    accumulator_dtype: Optional[pt.dtype],
    group,
) -> LinearTrendResult:
    """Merge centered least-squares states across the process group."""
    dtype = (
        state.mean_data.dtype
        if state is not None
        else accumulator_dtype or reference.dtype
    )
    template = reference.to(dtype=dtype)
    local_count = 0 if state is None else state.n_snapshots
    count = pt.tensor(local_count, dtype=pt.int64, device=reference.device)
    dist.all_reduce(count, op=dist.ReduceOp.SUM, group=group)
    global_count = int(count.item())
    if global_count < 2:
        raise ValueError("at least two snapshots are required for a trend")

    local_mean_time = template.new_zeros(()) if state is None else state.mean_time
    local_mean_data = pt.zeros_like(template) if state is None else state.mean_data
    mean_time = local_mean_time * local_count
    mean_data = local_mean_data * local_count
    dist.all_reduce(mean_time, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(mean_data, op=dist.ReduceOp.SUM, group=group)
    mean_time = mean_time / global_count
    mean_data = mean_data / global_count
    delta_time = local_mean_time - mean_time if local_count else mean_time.new_zeros(())
    delta_data = (
        local_mean_data - mean_data if local_count else pt.zeros_like(mean_data)
    )

    time_variation = (
        mean_time.new_zeros(()) if state is None else state.time_variation
    ) + local_count * delta_time.square()
    covariation = (
        pt.zeros_like(template) if state is None else state.covariation
    ) + local_count * delta_time * delta_data
    data_variation = (
        pt.zeros_like(template) if state is None else state.data_variation
    ) + local_count * delta_data.square()
    dist.all_reduce(time_variation, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(covariation, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(data_variation, op=dist.ReduceOp.SUM, group=group)

    minimum = (
        mean_time.new_tensor(float("inf")) if state is None else state.minimum_time
    ).clone()
    maximum = (
        mean_time.new_tensor(float("-inf")) if state is None else state.maximum_time
    ).clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    return _finalize_linear_trend_state(
        _LinearTrendState(
            global_count,
            mean_time,
            mean_data,
            time_variation,
            covariation,
            data_variation,
            minimum,
            maximum,
        )
    )


def _gather_spatial_statistics(
    local: Optional[SpatialStatisticsResult],
    reference: pt.Tensor,
    counts: list[int],
    quantile_values: list[float],
    root_rank: int,
    rank: int,
    group,
) -> Optional[SpatialStatisticsResult]:
    """Gather padded per-rank time series and concatenate them on the root."""
    rows = 3 + len(quantile_values)
    local_count = counts[rank]
    maximum_count = max(counts)
    padded = reference.new_zeros((rows, maximum_count))
    if local is not None:
        values = pt.cat(
            (
                local.minimum.unsqueeze(0),
                local.maximum.unsqueeze(0),
                local.mean.unsqueeze(0),
                local.quantiles,
            )
        )
        padded[:, :local_count] = values
    gathered = [pt.empty_like(padded) for _ in counts]
    dist.all_gather(gathered, padded, group=group)
    if rank != root_rank:
        return None
    combined = pt.cat(
        [values[:, :count] for values, count in zip(gathered, counts)], dim=1
    )
    return SpatialStatisticsResult(
        combined[0],
        combined[1],
        combined[2],
        combined[3:],
        reference.new_tensor(quantile_values),
    )


def _distributed_snapshot_statistics(
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
    spatial_reduction,
    keep_intermediate_fields: bool,
    density: bool,
    accumulator_dtype: Optional[pt.dtype],
    execution: DistributedExecution,
) -> Optional[SnapshotStatisticsResult]:
    """Execute a normalized joint request with PyTorch collectives."""
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    group = execution.process_group
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    config = (
        n_snapshots,
        snapshot_dim,
        tuple(sorted(compute)),
        None if fraction_values is None else tuple(fraction_values),
        None if quantile_values is None else tuple(quantile_values),
        (
            histogram_bins
            if isinstance(histogram_bins, int)
            else tuple(histogram_bins or ())
        ),
        histogram_range,
        spatial_weight is None,
        spatial_mask is None,
        spatial_reduction,
        keep_intermediate_fields,
        density,
        str(accumulator_dtype),
        execution.root_rank,
    )
    configs = _all_gather_objects(config, group)
    if any(value != configs[0] for value in configs[1:]):
        raise ValueError("all ranks must use matching distributed statistics controls")
    if execution.root_rank < 0 or execution.root_rank >= world_size:
        raise ValueError("root_rank must identify a rank in the process group")

    ranges = [_partition(n_snapshots, index, world_size) for index in range(world_size)]
    start, stop = ranges[rank]
    local_error = None
    try:
        local = _run_local_statistics(
            source,
            start,
            stop,
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
    except Exception as error:  # synchronize recoverable source failures
        local_error = f"{type(error).__name__}: {error}"
        local = _LocalStatisticsResult(None, None, None, None, None, None, None)
    _collective_error(local_error, group)
    reference = _reference_metadata(local, group)

    moments = None
    dependency = None
    if local.moment_states is not None:
        checkpoint_fields = [
            _distributed_moment_fields(state, reference, accumulator_dtype, group)
            for state in local.moment_states
        ]
        if rank == execution.root_rank:
            checkpoint_fields = [
                _mask_moment_fields(fields, spatial_mask)
                for fields in checkpoint_fields
            ]
            moments = checkpoint_fields[-1]
            if "dependency" in compute:
                assert fraction_values is not None
                assert checkpoint_counts is not None
                dependency = _build_dependency_result(
                    checkpoint_fields,
                    fraction_values,
                    checkpoint_counts,
                    spatial_weight,
                    spatial_mask,
                    spatial_reduction,
                    keep_intermediate_fields,
                )
                moments = dependency.final_fields

    trend = None
    if "trend" in compute:
        global_trend = _distributed_trend(
            local.trend_state, reference, accumulator_dtype, group
        )
        if rank == execution.root_rank:
            trend = _mask_linear_trend(global_trend, spatial_mask)

    spatial = None
    if "spatial" in compute:
        assert quantile_values is not None
        counts = [end - begin for begin, end in ranges]
        spatial = _gather_spatial_statistics(
            local.spatial_statistics,
            reference,
            counts,
            quantile_values,
            execution.root_rank,
            rank,
            group,
        )

    histogram = None
    if "histogram" in compute:
        assert histogram_bins is not None
        automatic = isinstance(histogram_bins, int) and histogram_range is None
        if automatic:
            lower = (
                reference.new_tensor(float("inf"))
                if local.range_minimum is None
                else local.range_minimum.clone()
            )
            upper = (
                reference.new_tensor(float("-inf"))
                if local.range_maximum is None
                else local.range_maximum.clone()
            )
            dist.all_reduce(lower, op=dist.ReduceOp.MIN, group=group)
            dist.all_reduce(upper, op=dist.ReduceOp.MAX, group=group)
            edges = _make_histogram_edges(
                histogram_bins, None, reference, (lower, upper)
            )
            second_error = None
            if stop > start:
                try:
                    second = _run_local_statistics(
                        source,
                        start,
                        stop,
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
                except Exception as error:
                    second_error = f"{type(error).__name__}: {error}"
                    second = _LocalStatisticsResult(
                        None, None, None, None, None, None, None
                    )
            else:
                second = _LocalStatisticsResult(
                    None, None, None, None, None, None, None
                )
            _collective_error(second_error, group)
            local_histogram = (
                edges.new_zeros(edges.numel() - 1)
                if second.histogram is None
                else second.histogram.histogram
            )
        else:
            edges = _make_histogram_edges(histogram_bins, histogram_range, reference)
            local_histogram = (
                edges.new_zeros(edges.numel() - 1)
                if local.histogram is None
                else local.histogram.histogram
            )
        dist.all_reduce(local_histogram, op=dist.ReduceOp.SUM, group=group)
        global_histogram = _normalize_histogram(
            HistogramResult(local_histogram, edges), density
        )
        if rank == execution.root_rank:
            histogram = global_histogram

    if rank != execution.root_rank:
        return None
    return SnapshotStatisticsResult(
        moments,
        dependency,
        trend,
        spatial,
        histogram,
    )
