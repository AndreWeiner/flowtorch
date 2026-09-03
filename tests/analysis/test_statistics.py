"""Tests for batched statistical moment and trend analysis."""

import pytest
import torch as pt

from flowtorch.analysis.statistics import (
    DEFAULT_QUANTILES,
    DistributedExecution,
    RunningMoments,
    detect_linear_trend,
    linear_trend,
    moment_data_dependency,
    snapshot_statistics,
    spatial_statistics,
    spatiotemporal_histogram,
    statistical_moments,
)


def _direct_moments(data, dim=-1):
    mean = data.mean(dim=dim)
    centered = data - mean.unsqueeze(dim)
    variance = centered.square().mean(dim=dim)
    skewness = centered.pow(3).mean(dim=dim) / variance.pow(1.5)
    kurtosis = centered.pow(4).mean(dim=dim) / variance.square()
    return mean, variance, skewness, kurtosis


def test_running_moments_matches_direct_computation_across_batches():
    generator = pt.Generator().manual_seed(4)
    data = pt.randn((3, 17), generator=generator, dtype=pt.float64)
    accumulator = RunningMoments()

    accumulator.update(data[:, :2])
    accumulator.update(data[:, 2:11])
    accumulator.update(data[:, 11:])

    assert accumulator.count == data.shape[-1]
    for actual, expected in zip(accumulator.finalize(), _direct_moments(data)):
        pt.testing.assert_close(actual, expected)


def test_running_moments_can_merge_accumulators():
    data = pt.arange(1.0, 25.0, dtype=pt.float64).reshape(3, 8)
    first = RunningMoments()
    second = RunningMoments()
    first.update(data[:, :3])
    second.update(data[:, 3:])

    first.merge(second)

    for actual, expected in zip(first.finalize(), _direct_moments(data)):
        pt.testing.assert_close(actual, expected)


def test_statistical_moments_supports_nonfinal_snapshot_dimension():
    data = pt.arange(1.0, 25.0, dtype=pt.float64).reshape(8, 3)

    result = statistical_moments(data, snapshot_dim=0, batch_size=3)

    for actual, expected in zip(result, _direct_moments(data, dim=0)):
        pt.testing.assert_close(actual, expected)


def test_constant_data_has_undefined_standardized_moments():
    result = statistical_moments(pt.ones((2, 5)))

    pt.testing.assert_close(result.variance, pt.zeros(2))
    assert bool(pt.isnan(result.skewness).all())
    assert bool(pt.isnan(result.kurtosis).all())


def test_moment_dependency_reduces_and_compares_checkpoint_fields():
    data = pt.tensor(
        [
            [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
            [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0],
            [2.0, 7.0, 1.0, 8.0, 2.0, 8.0, 1.0, 8.0],
        ],
        dtype=pt.float64,
    )
    weight = pt.tensor([1.0, 2.0, 3.0], dtype=pt.float64)
    half = _direct_moments(data[:, :4])
    full = _direct_moments(data)

    result = moment_data_dependency(
        data,
        fractions=(0.5, 1.0),
        batch_size=3,
        spatial_weight=weight,
    )

    expected_reduced = pt.stack(
        [
            pt.stack([(field * weight).sum() / weight.sum() for field in fields])
            for fields in (half, full)
        ]
    )
    expected_difference = pt.stack(
        [
            ((second - first).square() * weight).sum().sqrt()
            for first, second in zip(half, full)
        ]
    ).unsqueeze(0)
    pt.testing.assert_close(result.reduced_moments, expected_reduced)
    pt.testing.assert_close(result.field_difference_norms, expected_difference)
    assert pt.equal(result.n_snapshots, pt.tensor([4, 8]))
    for actual, expected in zip(result.final_fields, full):
        pt.testing.assert_close(actual, expected)
    assert result.intermediate_fields is None


def test_moment_dependency_loads_each_snapshot_once_and_retains_fields():
    data = pt.arange(1.0, 31.0, dtype=pt.float64).reshape(3, 10)
    calls = []

    def source(start, stop):
        calls.append((start, stop))
        return data[:, start:stop]

    result = moment_data_dependency(
        source,
        n_snapshots=10,
        fractions=(0.5, 1.0),
        batch_size=3,
        keep_intermediate_fields=True,
    )

    assert calls == [(0, 3), (3, 6), (6, 9), (9, 10)]
    assert result.intermediate_fields is not None
    half = _direct_moments(data[:, :5])
    for actual, expected in zip(result.intermediate_fields, half):
        assert actual.shape == (1, 3)
        pt.testing.assert_close(actual[0], expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fractions": (0.5, 0.8)}, "last fraction"),
        ({"fractions": (0.1, 0.2, 1.0)}, "distinct snapshot counts"),
        ({"batch_size": 0}, "batch_size"),
        ({"spatial_reduction": "median"}, "spatial_reduction"),
    ],
)
def test_moment_dependency_validates_controls(kwargs, message):
    with pytest.raises(ValueError, match=message):
        moment_data_dependency(pt.rand((2, 5)), **kwargs)


def test_callable_moment_source_requires_snapshot_count():
    with pytest.raises(ValueError, match="n_snapshots"):
        moment_data_dependency(lambda start, stop: pt.ones((2, stop - start)))


def test_spatial_statistics_uses_default_quantiles():
    data = pt.tensor(
        [
            [0.0, 3.0],
            [10.0, 2.0],
            [20.0, 1.0],
            [30.0, 0.0],
        ],
        dtype=pt.float64,
    )

    result = spatial_statistics(data, batch_size=1)

    pt.testing.assert_close(
        result.quantile_levels, pt.tensor(DEFAULT_QUANTILES, dtype=pt.float64)
    )
    pt.testing.assert_close(result.minimum, data.new_tensor([0.0, 0.0]))
    pt.testing.assert_close(result.maximum, data.new_tensor([30.0, 3.0]))
    pt.testing.assert_close(result.mean, data.new_tensor([15.0, 1.5]))
    pt.testing.assert_close(
        result.quantiles[:, 0],
        pt.tensor([0.0, 0.0, 5.0, 15.0, 25.0, 30.0], dtype=pt.float64),
    )


def test_spatial_statistics_weights_all_reductions():
    data = pt.tensor(
        [
            [-100.0, 100.0],
            [0.0, 30.0],
            [10.0, 20.0],
            [20.0, 10.0],
        ],
        dtype=pt.float64,
    )
    weight = pt.tensor([0.0, 1.0, 2.0, 1.0], dtype=pt.float64)

    result = spatial_statistics(
        data,
        quantiles=(0.0, 0.25, 0.5, 0.75, 1.0),
        spatial_weight=weight,
    )

    third = 10.0 / 3.0
    pt.testing.assert_close(result.minimum, data.new_tensor([0.0, 10.0]))
    pt.testing.assert_close(result.maximum, data.new_tensor([20.0, 30.0]))
    pt.testing.assert_close(result.mean, data.new_tensor([10.0, 20.0]))
    expected = pt.tensor(
        [
            [0.0, 10.0],
            [third, 10.0 + third],
            [10.0, 20.0],
            [20.0 - third, 30.0 - third],
            [20.0, 30.0],
        ],
        dtype=pt.float64,
    )
    pt.testing.assert_close(result.quantiles, expected)


def test_spatial_statistics_loads_each_snapshot_once():
    data = pt.arange(1.0, 29.0, dtype=pt.float64).reshape(4, 7)
    calls = []

    def source(start, stop):
        calls.append((start, stop))
        return data[:, start:stop]

    result = spatial_statistics(
        source,
        n_snapshots=7,
        quantiles=(0.5,),
        batch_size=3,
    )

    assert calls == [(0, 3), (3, 6), (6, 7)]
    pt.testing.assert_close(result.minimum, data.min(dim=0).values)
    pt.testing.assert_close(result.maximum, data.max(dim=0).values)
    pt.testing.assert_close(result.mean, data.mean(dim=0))
    assert result.quantiles.shape == (1, 7)


def test_spatial_statistics_supports_nonfinal_snapshot_dimension():
    data = pt.tensor([[0.0, 10.0, 20.0], [3.0, 2.0, 1.0]], dtype=pt.float64)

    result = spatial_statistics(
        data,
        quantiles=(0.5,),
        snapshot_dim=0,
        batch_size=1,
    )

    pt.testing.assert_close(result.minimum, data.new_tensor([0.0, 1.0]))
    pt.testing.assert_close(result.maximum, data.new_tensor([20.0, 3.0]))
    pt.testing.assert_close(result.mean, data.new_tensor([10.0, 2.0]))
    pt.testing.assert_close(result.quantiles, data.new_tensor([[10.0, 2.0]]))


def test_spatial_statistics_handles_one_spatial_value():
    data = pt.tensor([2.0, 4.0, 8.0])

    result = spatial_statistics(data, quantiles=(0.1, 0.9), batch_size=2)

    pt.testing.assert_close(result.minimum, data)
    pt.testing.assert_close(result.maximum, data)
    pt.testing.assert_close(result.mean, data)
    pt.testing.assert_close(result.quantiles, data.expand(2, -1))


@pytest.mark.parametrize(
    ("quantiles", "message"),
    [
        ((), "at least one"),
        ((0.5, 0.25), "strictly increasing"),
        ((0.25, 0.25), "strictly increasing"),
        ((-0.1, 0.5), "interval"),
        ((0.5, 1.1), "interval"),
    ],
)
def test_spatial_statistics_validates_quantiles(quantiles, message):
    with pytest.raises(ValueError, match=message):
        spatial_statistics(pt.rand((3, 4)), quantiles=quantiles)


def test_spatial_statistics_validates_callable_batches():
    def source(start, stop):
        spatial_size = 2 if start == 0 else 3
        return pt.ones((spatial_size, stop - start))

    with pytest.raises(ValueError, match="spatial dimensions"):
        spatial_statistics(source, n_snapshots=4, batch_size=2)


def test_spatiotemporal_histogram_uses_50_bins_by_default():
    result = spatiotemporal_histogram(pt.arange(12.0).reshape(3, 4), batch_size=2)

    assert result.histogram.shape == (50,)
    assert result.bin_edges.shape == (51,)
    pt.testing.assert_close(result.histogram.sum(), pt.tensor(12.0))


def test_spatiotemporal_histogram_accumulates_spatial_weights_across_batches():
    data = pt.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]], dtype=pt.float64)

    result = spatiotemporal_histogram(
        data,
        bins=(0.0, 1.0, 2.0, 3.0),
        batch_size=2,
        spatial_weight=pt.tensor([1.0, 2.0], dtype=pt.float64),
    )

    pt.testing.assert_close(result.histogram, data.new_tensor([1.0, 3.0, 5.0]))
    pt.testing.assert_close(result.bin_edges, data.new_tensor([0.0, 1.0, 2.0, 3.0]))


def test_spatiotemporal_histogram_discovers_exact_range_in_two_passes():
    data = pt.arange(5.0).reshape(1, 5)
    calls = []

    def source(start, stop):
        calls.append((start, stop))
        return data[:, start:stop]

    result = spatiotemporal_histogram(source, n_snapshots=5, bins=2, batch_size=2)

    assert calls == [(0, 2), (2, 4), (4, 5)] * 2
    pt.testing.assert_close(result.bin_edges, data.new_tensor([0.0, 2.0, 4.0]))
    pt.testing.assert_close(result.histogram, data.new_tensor([2.0, 3.0]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bins": (0.0, 2.0, 4.0)},
        {"bins": 2, "value_range": (0.0, 4.0)},
    ],
)
def test_spatiotemporal_histogram_known_edges_load_once(kwargs):
    data = pt.arange(5.0).reshape(1, 5)
    calls = []

    def source(start, stop):
        calls.append((start, stop))
        return data[:, start:stop]

    result = spatiotemporal_histogram(source, n_snapshots=5, batch_size=2, **kwargs)

    assert calls == [(0, 2), (2, 4), (4, 5)]
    pt.testing.assert_close(result.histogram, data.new_tensor([2.0, 3.0]))


def test_spatiotemporal_histogram_excludes_zero_weight_from_automatic_range():
    data = pt.tensor([[-100.0, 100.0], [0.0, 2.0]], dtype=pt.float64)

    result = spatiotemporal_histogram(
        data, bins=2, spatial_weight=data.new_tensor([0.0, 3.0])
    )

    pt.testing.assert_close(result.bin_edges, data.new_tensor([0.0, 1.0, 2.0]))
    pt.testing.assert_close(result.histogram, data.new_tensor([3.0, 3.0]))


def test_spatiotemporal_histogram_expands_constant_automatic_range():
    data = pt.full((2, 3), 4.0)

    result = spatiotemporal_histogram(data, bins=2)

    pt.testing.assert_close(result.bin_edges, data.new_tensor([3.5, 4.0, 4.5]))
    pt.testing.assert_close(result.histogram, data.new_tensor([0.0, 6.0]))


def test_spatiotemporal_histogram_density_integrates_to_one():
    data = pt.tensor([[0.25, 1.0, 3.0]], dtype=pt.float64)

    result = spatiotemporal_histogram(data, bins=(0.0, 0.5, 2.0, 4.0), density=True)

    widths = result.bin_edges[1:] - result.bin_edges[:-1]
    pt.testing.assert_close((result.histogram * widths).sum(), data.new_tensor(1.0))
    pt.testing.assert_close(
        result.histogram, data.new_tensor([2.0 / 3.0, 2.0 / 9.0, 1.0 / 6.0])
    )


def test_spatiotemporal_histogram_supports_nonfinal_snapshot_dimension():
    data = pt.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])

    result = spatiotemporal_histogram(
        data, bins=(0.0, 2.0, 4.0, 5.0), snapshot_dim=0, batch_size=2
    )

    pt.testing.assert_close(result.histogram, data.new_tensor([2.0, 2.0, 2.0]))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bins": 0}, "positive integer"),
        ({"bins": (0.0,)}, "at least two"),
        ({"bins": (0.0, 1.0, 1.0)}, "strictly increasing"),
        ({"bins": (0.0, float("inf"))}, "finite real"),
        ({"bins": 2, "value_range": (1.0, 1.0)}, "smaller"),
        (
            {"bins": (0.0, 1.0), "value_range": (0.0, 1.0)},
            "cannot be used",
        ),
        ({"density": 1}, "boolean"),
    ],
)
def test_spatiotemporal_histogram_validates_controls(kwargs, message):
    with pytest.raises(ValueError, match=message):
        spatiotemporal_histogram(pt.rand((2, 3)), **kwargs)


def test_spatiotemporal_histogram_rejects_empty_density_mass():
    with pytest.raises(ValueError, match="no values"):
        spatiotemporal_histogram(
            pt.arange(4.0).reshape(2, 2),
            bins=2,
            value_range=(10.0, 20.0),
            density=True,
        )


def test_spatiotemporal_histogram_validates_callable_batches():
    def source(start, stop):
        spatial_size = 2 if start == 0 else 3
        return pt.ones((spatial_size, stop - start))

    with pytest.raises(ValueError, match="spatial dimensions"):
        spatiotemporal_histogram(
            source,
            n_snapshots=4,
            bins=2,
            value_range=(0.0, 1.0),
            batch_size=2,
        )


def test_spatiotemporal_histogram_rejects_nonfinite_snapshots():
    with pytest.raises(ValueError, match="finite values"):
        spatiotemporal_histogram(pt.tensor([[0.0, float("nan")]]), bins=2)


def test_snapshot_statistics_reuses_one_fixed_range_pass():
    data = pt.arange(1.0, 31.0, dtype=pt.float64).reshape(3, 10)
    calls = []

    def source(start, stop):
        calls.append((start, stop))
        return data[:, start:stop]

    result = snapshot_statistics(
        source,
        n_snapshots=10,
        fractions=(0.5, 1.0),
        quantiles=(0.25, 0.5, 0.75),
        histogram_bins=5,
        histogram_range=(1.0, 30.0),
        batch_size=3,
    )

    assert result is not None
    assert calls == [(0, 3), (3, 6), (6, 9), (9, 10)]
    assert result.moment_dependency is not None
    assert result.moments is result.moment_dependency.final_fields
    expected_moments = statistical_moments(data, batch_size=3)
    assert expected_moments is not None
    for actual, expected in zip(result.moments, expected_moments):
        pt.testing.assert_close(actual, expected)
    expected_trend = linear_trend(data, batch_size=3)
    assert expected_trend is not None
    assert result.trend is not None
    for actual, expected in zip(result.trend[:-1], expected_trend[:-1]):
        pt.testing.assert_close(actual, expected)
    expected_spatial = spatial_statistics(data, quantiles=(0.25, 0.5, 0.75))
    assert expected_spatial is not None
    assert result.spatial_statistics is not None
    for actual, expected in zip(result.spatial_statistics, expected_spatial):
        pt.testing.assert_close(actual, expected)


def test_snapshot_statistics_auto_histogram_only_reloads_histogram():
    data = pt.arange(12.0).reshape(3, 4)
    calls = []

    def source(start, stop):
        calls.append((start, stop))
        return data[:, start:stop]

    result = snapshot_statistics(
        source,
        n_snapshots=4,
        fractions=(0.5, 1.0),
        histogram_bins=3,
        batch_size=3,
    )

    assert result is not None
    assert calls == [(0, 3), (3, 4)] * 2
    assert result.moments is not None
    assert result.histogram is not None
    pt.testing.assert_close(result.histogram.histogram.sum(), data.new_tensor(12.0))


def test_snapshot_statistics_supports_selective_collectors():
    result = snapshot_statistics(
        pt.arange(12.0).reshape(3, 4),
        compute=("moments", "histogram"),
        histogram_bins=(0.0, 6.0, 11.0),
    )

    assert result is not None
    assert result.moments is not None
    assert result.moment_dependency is None
    assert result.trend is None
    assert result.spatial_statistics is None
    assert result.histogram is not None


def test_snapshot_statistics_restricts_all_collectors_with_spatial_mask():
    data = pt.tensor(
        [
            [float("nan"), float("nan"), float("nan"), float("nan")],
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ],
        dtype=pt.float64,
    )
    spatial_mask = pt.tensor([False, True, True])
    weight = pt.tensor([100.0, 1.0, 3.0], dtype=pt.float64)

    result = snapshot_statistics(
        data,
        fractions=(0.5, 1.0),
        quantiles=(0.5,),
        histogram_bins=4,
        histogram_range=(1.0, 5.0),
        spatial_weight=weight,
        spatial_mask=spatial_mask,
        keep_intermediate_fields=True,
        batch_size=2,
    )

    assert result is not None
    assert result.moments is not None
    assert result.moment_dependency is not None
    assert result.trend is not None
    assert result.spatial_statistics is not None
    assert result.histogram is not None
    for field in result.moments:
        assert bool(pt.isnan(field[~spatial_mask]).all())
    for field in result.trend[:-1]:
        assert bool(pt.isnan(field[~spatial_mask]).all())
    assert result.moment_dependency.intermediate_fields is not None
    for field in result.moment_dependency.intermediate_fields:
        assert bool(pt.isnan(field[..., ~spatial_mask]).all())
    expected_mean = (data[1] + 3.0 * data[2]) / 4.0
    pt.testing.assert_close(result.spatial_statistics.mean, expected_mean)
    pt.testing.assert_close(
        result.spatial_statistics.minimum, data[1:].min(dim=0).values
    )
    pt.testing.assert_close(
        result.spatial_statistics.maximum, data[1:].max(dim=0).values
    )
    pt.testing.assert_close(result.histogram.histogram.sum(), data.new_tensor(16.0))


@pytest.mark.parametrize(
    ("spatial_mask", "message"),
    [
        (pt.tensor([1.0, 0.0]), "boolean dtype"),
        (pt.tensor([False, False]), "select at least one"),
        (pt.tensor([True, False, True]), "broadcast"),
    ],
)
def test_snapshot_statistics_validates_spatial_mask(spatial_mask, message):
    with pytest.raises(ValueError, match=message):
        snapshot_statistics(
            pt.ones((2, 4)),
            spatial_mask=spatial_mask,
            compute=("moments",),
        )


def test_snapshot_statistics_rejects_weights_outside_spatial_mask():
    with pytest.raises(ValueError, match="positive at a location"):
        snapshot_statistics(
            pt.ones((2, 4)),
            spatial_weight=pt.tensor([0.0, 1.0]),
            spatial_mask=pt.tensor([True, False]),
            compute=("histogram",),
        )


def test_individual_statistics_functions_forward_spatial_mask():
    data = pt.tensor(
        [
            [float("nan"), float("nan"), float("nan"), float("nan")],
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ]
    )
    spatial_mask = pt.tensor([False, True, True])

    moments = statistical_moments(data, spatial_mask=spatial_mask)
    dependency = moment_data_dependency(
        data, fractions=(0.5, 1.0), spatial_mask=spatial_mask
    )
    trend = linear_trend(data, spatial_mask=spatial_mask)
    spatial = spatial_statistics(data, quantiles=(0.5,), spatial_mask=spatial_mask)
    histogram = spatiotemporal_histogram(
        data, bins=4, value_range=(1.0, 5.0), spatial_mask=spatial_mask
    )

    assert moments is not None
    assert dependency is not None
    assert trend is not None
    assert spatial is not None
    assert histogram is not None
    assert bool(pt.isnan(moments.mean[0]))
    assert bool(pt.isnan(dependency.final_fields.mean[0]))
    assert bool(pt.isnan(trend.slope[0]))
    pt.testing.assert_close(spatial.mean, data[1:].mean(dim=0))
    pt.testing.assert_close(histogram.histogram.sum(), data.new_tensor(8.0))


@pytest.mark.parametrize(
    ("compute", "message"),
    [
        ((), "at least one"),
        (("moments", "moments"), "duplicate"),
        (("unknown",), "unknown statistics"),
    ],
)
def test_snapshot_statistics_validates_compute(compute, message):
    with pytest.raises(ValueError, match=message):
        snapshot_statistics(pt.rand((2, 3)), compute=compute)


def test_snapshot_statistics_validates_execution_policy():
    with pytest.raises(ValueError, match="DistributedExecution"):
        snapshot_statistics(
            pt.rand((2, 3)), compute=("moments",), execution="distributed"
        )

    with pytest.raises(RuntimeError, match="must be initialized"):
        snapshot_statistics(
            pt.rand((2, 3)),
            compute=("moments",),
            execution=DistributedExecution(root_rank=0),
        )


def test_linear_trend_matches_exact_fit_with_irregular_times():
    time = pt.tensor([0.0, 0.5, 2.0, 3.0, 5.0], dtype=pt.float64)
    data = pt.stack((2.0 * time + 1.0, -3.0 * time + 4.0, pt.ones_like(time)))

    result = linear_trend(data, time=time, batch_size=2)

    pt.testing.assert_close(result.slope, pt.tensor([2.0, -3.0, 0.0], dtype=pt.float64))
    pt.testing.assert_close(
        result.intercept, pt.tensor([1.0, 4.0, 1.0], dtype=pt.float64)
    )
    pt.testing.assert_close(
        result.r_squared, pt.tensor([1.0, 1.0, 0.0], dtype=pt.float64)
    )
    assert result.n_snapshots == 5
    mask = detect_linear_trend(result, min_normalized_change=1.0, min_r_squared=0.9)
    assert pt.equal(mask, pt.tensor([True, True, False]))


def test_linear_trend_supports_an_indexed_source():
    time = pt.linspace(0.0, 1.0, 9, dtype=pt.float64)
    data = (4.0 * time - 2.0).reshape(1, -1)
    calls = []

    def source(start, stop):
        calls.append((start, stop))
        return data[:, start:stop]

    result = linear_trend(source, n_snapshots=9, time=time, batch_size=4)

    assert calls == [(0, 4), (4, 8), (8, 9)]
    pt.testing.assert_close(result.slope, pt.tensor([4.0], dtype=pt.float64))
    pt.testing.assert_close(result.intercept, pt.tensor([-2.0], dtype=pt.float64))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"time": [0.0, 1.0]}, "match n_snapshots"),
        ({"time": [0.0, 1.0, 1.0]}, "strictly increasing"),
    ],
)
def test_linear_trend_validates_time(kwargs, message):
    with pytest.raises(ValueError, match=message):
        linear_trend(pt.rand((2, 3)), **kwargs)


@pytest.mark.parametrize(
    ("change", "r_squared", "message"),
    [
        (0.0, 0.5, "positive"),
        (1.0, 1.1, "interval"),
    ],
)
def test_detect_linear_trend_validates_thresholds(change, r_squared, message):
    result = linear_trend(pt.tensor([[0.0, 1.0, 2.0]]))
    with pytest.raises(ValueError, match=message):
        detect_linear_trend(result, change, r_squared)
