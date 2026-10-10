"""Bounded host preparation, cleanup, and no speculative GPU transfers."""

from contextlib import closing
from threading import Event, enumerate as threads, current_thread

import pytest
import torch as pt

from flowtorch.analysis._chunk_prefetch import iter_prepared_chunks
from flowtorch.analysis.state_vector import (
    StateVectorSource,
    StateVectorLayout,
    FieldSpec,
)


def test_one_ahead_overlaps_consumer_and_is_bounded():
    started, release = Event(), Event()
    calls = []

    def read(index):
        calls.append(index)
        if index == 1:
            started.set()
            assert release.wait(5)
        return pt.tensor([index])

    with closing(iter_prepared_chunks(range(4), read)) as chunks:
        assert next(chunks).item() == 0
        assert started.wait(5)
        assert calls == [0, 1]
        release.set()
        assert [item.item() for item in chunks] == [1, 2, 3]
    assert not any(t.name.startswith("flowtorch-read") for t in threads())


def test_early_close_drains_only_pending_read():
    calls = []
    with closing(
        iter_prepared_chunks(range(10), lambda i: (calls.append(i), pt.tensor(i))[1])
    ) as chunks:
        assert next(chunks).item() == 0
    assert calls == [0, 1]
    assert not any(t.name.startswith("flowtorch-read") for t in threads())


def test_read_failure_propagates_and_cleans_up():
    def read(i):
        if i == 1:
            raise OSError("reader failed")
        return pt.tensor(i)

    with closing(iter_prepared_chunks(range(3), read)) as chunks:
        assert next(chunks).item() == 0
        with pytest.raises(OSError, match="reader failed"):
            next(chunks)
    assert not any(t.name.startswith("flowtorch-read") for t in threads())


def test_disabled_prefetch_does_not_read_ahead():
    calls = []
    with closing(
        iter_prepared_chunks(
            range(3), lambda i: (calls.append(i), pt.tensor(i))[1], prefetch=False
        )
    ) as chunks:
        next(chunks)
        assert calls == [0]


class HostSource(StateVectorSource):
    _device = pt.device("cuda")

    @property
    def layout(self):
        return StateVectorLayout((FieldSpec("q"),), (8,))

    @property
    def n_snapshots(self):
        return 3

    def read(self, spatial_slice, snapshot_slice):
        pytest.fail("native GPU transfer was requested")

    def read_cpu(self, spatial_slice, snapshot_slice):
        return pt.ones(spatial_slice.stop - spatial_slice.start, 3)


def test_declared_gpu_source_iterator_prepares_cpu_only():
    source = HostSource()
    with closing(source.iter_chunks(slice(0, 8), 3)) as chunks:
        packets = list(chunks)
    assert [part for part, _, _ in packets] == [slice(0, 3), slice(3, 6), slice(6, 8)]
    assert all(data.device.type == "cpu" for _, data, _ in packets)


@pytest.mark.skipif(not pt.cuda.is_available(), reason="CUDA unavailable")
def test_legacy_gpu_reader_stays_on_consumer_thread():
    owner = current_thread()

    def read(i):
        assert current_thread() is owner
        return pt.tensor(i, device="cuda")

    assert len(list(iter_prepared_chunks(range(3), read))) == 3


def test_dispatch_does_not_change_random_state_or_thread_limits():
    state = pt.random.get_rng_state().clone()
    count = pt.get_num_threads()
    list(iter_prepared_chunks(range(3), lambda i: pt.zeros(2)))
    pt.testing.assert_close(pt.random.get_rng_state(), state)
    assert pt.get_num_threads() == count


def test_gpu_queue_capacity_is_bounded_by_shared_host_memory():
    from flowtorch.analysis.svd import _initial_memory_capacity

    reports = [
        dict(
            hostname="node",
            available=10**9,
            device_type="cuda",
            budget=None,
            scalar_bytes=4,
            rows_per_value=1,
            prefetch=True,
            host_available=6000,
        )
        for _ in range(2)
    ]
    assert _initial_memory_capacity(reports, 8) == 5
