"""Multi-process tests for distributed snapshot statistics."""

import os

import pytest
import torch as pt
import torch.distributed as dist
import torch.multiprocessing as mp

from flowtorch.analysis import (
    DistributedExecution,
    snapshot_statistics,
    spatiotemporal_histogram,
    statistical_moments,
)

pytestmark = pytest.mark.integration


def _distributed_worker(rank, world_size, store_path):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        pt.set_num_threads(2)
        data = pt.tensor(
            [[0.0, 2.0], [float("nan"), float("nan")], [2.0, 4.0]],
            dtype=pt.float64,
        )
        weight = pt.tensor([1.0, 2.0, 3.0], dtype=pt.float64)
        spatial_mask = pt.tensor([True, False, True])
        calls = []

        def source(start, stop):
            calls.append((start, stop))
            return data[:, start:stop]

        execution = DistributedExecution(root_rank=0)
        result = snapshot_statistics(
            source,
            n_snapshots=2,
            fractions=(0.5, 1.0),
            quantiles=(0.25, 0.5, 0.75),
            histogram_bins=2,
            histogram_range=(0.0, 4.0),
            batch_size=1,
            spatial_weight=weight,
            spatial_mask=spatial_mask,
            keep_intermediate_fields=True,
            execution=execution,
        )
        expected_calls = [(rank, rank + 1)] if rank < 2 else []
        assert calls == expected_calls
        if rank == 0:
            expected = snapshot_statistics(
                data,
                fractions=(0.5, 1.0),
                quantiles=(0.25, 0.5, 0.75),
                histogram_bins=2,
                histogram_range=(0.0, 4.0),
                batch_size=1,
                spatial_weight=weight,
                spatial_mask=spatial_mask,
                keep_intermediate_fields=True,
            )
            assert expected is not None
            assert result is not None
            assert result.moments is not None
            assert expected.moments is not None
            for actual, serial in zip(result.moments, expected.moments):
                pt.testing.assert_close(actual, serial, equal_nan=True)
            assert result.moment_dependency is not None
            assert expected.moment_dependency is not None
            pt.testing.assert_close(
                result.moment_dependency.fractions,
                expected.moment_dependency.fractions,
            )
            pt.testing.assert_close(
                result.moment_dependency.n_snapshots,
                expected.moment_dependency.n_snapshots,
            )
            pt.testing.assert_close(
                result.moment_dependency.reduced_moments,
                expected.moment_dependency.reduced_moments,
                equal_nan=True,
            )
            pt.testing.assert_close(
                result.moment_dependency.field_difference_norms,
                expected.moment_dependency.field_difference_norms,
                equal_nan=True,
            )
            assert result.moment_dependency.intermediate_fields is not None
            assert expected.moment_dependency.intermediate_fields is not None
            for actual, serial in zip(
                result.moment_dependency.intermediate_fields,
                expected.moment_dependency.intermediate_fields,
            ):
                pt.testing.assert_close(actual, serial, equal_nan=True)
            assert result.trend is not None
            assert expected.trend is not None
            for actual, serial in zip(result.trend[:-1], expected.trend[:-1]):
                pt.testing.assert_close(actual, serial, equal_nan=True)
            assert result.trend.n_snapshots == expected.trend.n_snapshots
            assert result.spatial_statistics is not None
            assert expected.spatial_statistics is not None
            for actual, serial in zip(
                result.spatial_statistics, expected.spatial_statistics
            ):
                pt.testing.assert_close(actual, serial)
            assert result.histogram is not None
            assert expected.histogram is not None
            for actual, serial in zip(result.histogram, expected.histogram):
                pt.testing.assert_close(actual, serial)
        else:
            assert result is None

        calls.clear()
        root_two_moments = statistical_moments(
            source,
            n_snapshots=2,
            batch_size=1,
            spatial_mask=spatial_mask,
            execution=DistributedExecution(root_rank=2),
        )
        assert calls == expected_calls
        if rank == 2:
            expected_moments = statistical_moments(
                data, batch_size=1, spatial_mask=spatial_mask
            )
            assert root_two_moments is not None
            assert expected_moments is not None
            for actual, serial in zip(root_two_moments, expected_moments):
                pt.testing.assert_close(actual, serial, equal_nan=True)
        else:
            assert root_two_moments is None

        with pytest.raises(ValueError, match="root_rank"):
            statistical_moments(
                source,
                n_snapshots=2,
                execution=DistributedExecution(root_rank=world_size),
            )

        mismatched_compute = ("moments",) if rank == 0 else ("trend",)
        with pytest.raises(ValueError, match="matching"):
            snapshot_statistics(
                source,
                n_snapshots=2,
                compute=mismatched_compute,
                execution=execution,
            )

        calls.clear()
        histogram = spatiotemporal_histogram(
            source,
            n_snapshots=2,
            bins=2,
            batch_size=1,
            spatial_weight=weight,
            spatial_mask=spatial_mask,
            execution=execution,
        )
        assert calls == expected_calls * 2
        if rank == 0:
            expected_histogram = spatiotemporal_histogram(
                data,
                bins=2,
                batch_size=1,
                spatial_weight=weight,
                spatial_mask=spatial_mask,
            )
            assert histogram is not None
            assert expected_histogram is not None
            pt.testing.assert_close(histogram, expected_histogram)
        else:
            assert histogram is None
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_distributed_statistics_match_serial_with_idle_rank(tmp_path):
    store_path = os.fspath(tmp_path / "distributed-statistics-store")
    mp.spawn(_distributed_worker, args=(3, store_path), nprocs=3, join=True)
