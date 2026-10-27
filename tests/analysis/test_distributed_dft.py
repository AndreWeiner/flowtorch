"""Distributed PDFT projections using source-backed SVD factors."""

import pytest
import torch as pt
import torch.distributed as dist
import torch.multiprocessing as mp

from flowtorch.analysis import DistributedExecution, SVD
from flowtorch.analysis.dft import DFT, PDFT
from flowtorch.analysis.state_vector import (
    FieldSpec,
    StateVectorLayout,
    StateVectorSource,
)

pytestmark = pytest.mark.integration


class _Source(StateVectorSource):
    def __init__(self, data):
        self.data = data
        self._layout = StateVectorLayout((FieldSpec("q"),), (data.shape[0],))

    @property
    def layout(self):
        return self._layout

    @property
    def n_snapshots(self):
        return self.data.shape[1]

    def read(self, spatial_slice, snapshot_slice):
        return self.data[spatial_slice, snapshot_slice]


def _worker(rank, store_path):
    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=2
    )
    try:
        pt.manual_seed(33)
        data = pt.randn(17, 8, dtype=pt.float64) + 4
        weight = pt.linspace(0.5, 2.0, 17, dtype=pt.float64)
        svd = SVD(
            _Source(data),
            rank=7,
            weight=weight,
            subtract_mean=True,
            spatial_batch_size=3,
            execution=DistributedExecution(root_rank=1),
        )
        result = PDFT.from_svd(svd, 0.2)
        direct = DFT(data, 0.2, weight=weight)
        local = result.modes.local_spatial_slice
        pt.testing.assert_close(result.amplitude, direct.amplitude)
        pt.testing.assert_close(result.modes.materialize_local(), direct.modes[local])
        pt.testing.assert_close(
            result.get_mode(1).materialize_local(), direct.get_mode(1)[local]
        )
        pt.testing.assert_close(result.reconstruction.materialize_local(), data[local])
        pt.testing.assert_close(
            result.partial_reconstruction({1, 2}).materialize_local(),
            direct.partial_reconstruction({1, 2})[local],
        )
    finally:
        dist.destroy_process_group()


def test_distributed_pdft_matches_serial(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "dft-store"),), nprocs=2, join=True)
