"""Multi-process tests for spatially distributed TSQR SVD."""

import os
from math import ceil
from unittest.mock import patch

import pytest
import torch as pt
import torch.distributed as dist
import torch.multiprocessing as mp

from flowtorch.analysis import DistributedExecution, PAMSPOD, SVD, subspace_similarity
from flowtorch.analysis.spod import mode_similarity
from flowtorch.analysis.state_vector import (
    FieldSpec,
    StateVectorLayout,
    StateVectorSource,
)

pytestmark = pytest.mark.integration


class _DistributedMatrixSource(StateVectorSource):
    def __init__(self, data, weight=None):
        self.data = data
        self.weight = weight
        self._layout = StateVectorLayout((FieldSpec("q"),), (data.shape[0],))

    @property
    def n_snapshots(self):
        return self.data.shape[1]

    @property
    def layout(self):
        return self._layout

    def read(self, spatial_slice, snapshot_slice):
        return self.data[spatial_slice, snapshot_slice]

    def read_weight(self, spatial_slice):
        return None if self.weight is None else self.weight[spatial_slice]


def _worker(rank, world_size, store_path):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        pt.manual_seed(12)
        data = pt.rand((17, 5), dtype=pt.float64)
        weight = pt.linspace(0.5, 1.5, data.shape[0], dtype=data.dtype)
        execution = DistributedExecution(root_rank=1)
        svd = SVD(
            _DistributedMatrixSource(data, weight),
            rank=4,
            subtract_mean=True,
            spatial_batch_size=3,
            execution=execution,
        )
        direct = SVD(
            data - data.mean(dim=1, keepdim=True),
            rank=4,
            mode="svd",
            weight=weight,
        )
        pt.testing.assert_close(svd.s, direct.s)
        # Host B is the tightest rank even though host A has two peers.
        fixed = (world_size + 6) * 10**2 * 8 + 4 * 5 * 5 * 8
        workspace = 14 * 5 * 8
        available = (
            ceil(2 * (fixed + 6 * workspace) / 0.6)
            if rank < 2
            else ceil((fixed + 4 * workspace) / 0.6)
        )
        with (
            patch(
                "flowtorch.analysis.svd.gethostname",
                return_value="a" if rank < 2 else "b",
            ),
            patch(
                "flowtorch.analysis.svd._available_memory_bytes", return_value=available
            ),
        ):
            bounded = SVD(
                _DistributedMatrixSource(data, weight),
                rank=5,
                subtract_mean=True,
                execution=execution,
            )
            assert bounded._spatial_batch_size == 4
        pt.testing.assert_close(bounded.s, direct.s)
        sizes = [None] * world_size
        dist.all_gather_object(sizes, bounded._spatial_batch_size)
        assert len(set(sizes)) == 1
        with pytest.raises(ValueError, match="too small"):
            SVD(
                _DistributedMatrixSource(data),
                memory_budget_bytes=1 if rank == 2 else 1024**3,
                execution=execution,
            )
        gathered_modes = svd.U.gather(root_rank=1)
        gathered_reconstruction = svd.reconstruct().gather(root_rank=1)
        if rank == 1:
            assert gathered_modes is not None
            assert gathered_reconstruction is not None
            pt.testing.assert_close(
                subspace_similarity(gathered_modes, direct.U, ranks=4, weight=weight),
                pt.ones(1, dtype=data.dtype),
            )
            pt.testing.assert_close(gathered_reconstruction, direct.reconstruct())
        else:
            assert gathered_modes is None
            assert gathered_reconstruction is None

        spod = PAMSPOD.from_svd(
            svd,
            dt=0.1,
            adaptive=False,
            max_tapers=3,
            keep_n_modes=2,
        )
        direct_spod = PAMSPOD(
            data,
            dt=0.1,
            rank=4,
            weight=weight,
            adaptive=False,
            max_tapers=3,
            keep_n_modes=2,
        )
        pt.testing.assert_close(spod.eigvals, direct_spod.eigvals)
        gathered_mode = spod.get_mode(2).gather(root_rank=1)
        if rank == 1:
            assert gathered_mode is not None
            similarity = mode_similarity(
                gathered_mode.unsqueeze(-1),
                direct_spod.get_mode(2).unsqueeze(-1),
            )
            pt.testing.assert_close(similarity, pt.ones_like(similarity))
        else:
            assert gathered_mode is None

        initial = _DistributedMatrixSource(data[:, :3])
        appended = _DistributedMatrixSource(data[:, 3:])
        updated = SVD(
            initial,
            rank=3,
            spatial_batch_size=3,
            execution=execution,
        )
        updated.update(appended)
        direct_updated = SVD(data, rank=3, mode="svd")
        pt.testing.assert_close(updated.s, direct_updated.s)
        gathered_updated = updated.U.gather(root_rank=1)
        if rank == 1:
            assert gathered_updated is not None
            pt.testing.assert_close(
                subspace_similarity(gathered_updated, direct_updated.U, ranks=3),
                pt.ones(1, dtype=data.dtype),
            )
        else:
            assert gathered_updated is None

        # Centered rank growth uses stored local modes, including on tail ranks.
        for storage in ("disk", "memory"):
            centered_source = _DistributedMatrixSource(data[:, :2], weight)
            centered = SVD(
                centered_source,
                rank=10,
                subtract_mean=True,
                basis_storage=storage,
                spatial_batch_size=3,
                execution=execution,
            )
            centered_source.data = None
            centered.update(_DistributedMatrixSource(data[:, 2:], weight))
            assert centered.rank == 4
            assert centered.maximum_rank == 10
            full = SVD(data, rank=4, subtract_mean=True, weight=weight, mode="svd")
            pt.testing.assert_close(centered.s, full.s)
            reconstructed = centered.reconstruct().gather(root_rank=1)
            if rank == 1:
                pt.testing.assert_close(reconstructed, full.reconstruct())
            basis_path = f"{store_path}-{storage}"
            centered.save(basis_path)
            restored_centered = SVD.load(
                basis_path, execution=execution, basis_storage=storage
            )
            pt.testing.assert_close(
                restored_centered.U.materialize_local(), centered.U.materialize_local()
            )
            # A local weight error must reach all ranks, without deadlocking.
            invalid = _DistributedMatrixSource(
                data[:, :1], weight * (2 if rank == 2 else 1)
            )
            with pytest.raises(RuntimeError, match="weights are incompatible"):
                centered.update(invalid)
            pt.testing.assert_close(centered.s, full.s)
            centered.close()
            restored_centered.close()

        # Per-rank update budgets select one common size without timing trials.
        update_fixed = (world_size + 6) * 5**2 * 8 + 2 * 5 * 5 * 8
        update_per_value = 8 * (6 * 2 + 4 * (3 + 5)) + 2 * 8 * (2 + 3 + 2)
        own_budget = (
            update_fixed + (2 if rank == 2 else 5) * update_per_value + (3 + 3 * 3) * 8
        )
        bounded = SVD(
            _DistributedMatrixSource(data[:, :3]),
            rank=5,
            spatial_batch_size=5,
            memory_budget_bytes=own_budget,
            execution=execution,
        )
        with patch.object(
            SVD,
            "_select_spatial_batch",
            side_effect=AssertionError("update reselected"),
        ):
            bounded.update(_DistributedMatrixSource(data[:, 3:]))
        assert bounded._spatial_batch_size == 2
        pt.testing.assert_close(bounded.s, SVD(data, rank=5, mode="svd").s)
        bounded_path = f"{store_path}-budget"
        bounded.save(bounded_path)
        loaded_bounded = SVD.load(bounded_path, execution=execution)
        assert loaded_bounded._memory_budget_bytes == own_budget
        assert loaded_bounded._spatial_batch_size == 2
        bounded.close()
        loaded_bounded.close()

        class FailedReadSource(_DistributedMatrixSource):
            def read(self, spatial_slice, snapshot_slice):
                if rank == 2 and spatial_slice.start >= 15:
                    raise OSError("prefetched read failed")
                return super().read(spatial_slice, snapshot_slice)

        with pytest.raises(RuntimeError, match="prefetched read failed"):
            SVD(
                FailedReadSource(data),
                rank=3,
                spatial_batch_size=3,
                execution=execution,
            )

        checkpoint_path = f"{store_path}-svd.pt"
        updated.save(checkpoint_path)
        restored = SVD.load(
            checkpoint_path,
            source=_DistributedMatrixSource(data),
            execution=execution,
        )
        pt.testing.assert_close(restored.s, updated.s)
        pt.testing.assert_close(restored.V, updated.V)
        gathered_restored = restored.reconstruct().gather(root_rank=1)
        gathered_original = updated.reconstruct().gather(root_rank=1)
        if rank == 1:
            assert gathered_restored is not None
            assert gathered_original is not None
            pt.testing.assert_close(gathered_restored, gathered_original)
        else:
            assert gathered_restored is None
            assert gathered_original is None
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_spatially_distributed_tsqr_matches_serial(tmp_path):
    store_path = os.fspath(tmp_path / "distributed-svd-store")
    mp.spawn(_worker, args=(3, store_path), nprocs=3, join=True)
