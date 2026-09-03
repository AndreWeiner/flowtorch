"""Multi-process tests for spatially distributed TSQR SVD."""

import os

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
            snapshot_batch_size=2,
            execution=execution,
        )
        direct = SVD(
            data - data.mean(dim=1, keepdim=True),
            rank=4,
            mode="svd",
            weight=weight,
        )
        pt.testing.assert_close(svd.s, direct.s)
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
            snapshot_batch_size=2,
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
