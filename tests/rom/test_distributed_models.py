"""Multi-process equivalence tests for source-backed ROMs."""

import os

import pytest
import torch as pt
import torch.distributed as dist
import torch.multiprocessing as mp

from flowtorch.analysis import DistributedExecution
from flowtorch.analysis.state_vector import (
    FieldSpec,
    StateVectorLayout,
    StateVectorSource,
)
from flowtorch.rom import (
    ControlledTrajectory,
    DMD,
    DMDc,
    MonomialEmbedding,
    ParametricSnapshots,
    PODI,
    TimeDelayEmbedding,
    Trajectory,
)

pytestmark = pytest.mark.integration


class MatrixSource(StateVectorSource):
    def __init__(self, data):
        self.data = data
        self._layout = StateVectorLayout((FieldSpec("q"),), (data.shape[0],))

    @property
    def n_snapshots(self):
        return self.data.shape[1]

    @property
    def layout(self):
        return self._layout

    def read(self, spatial_slice, snapshot_slice):
        return self.data[spatial_slice, snapshot_slice]


def _worker(rank, world_size, store_path):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        execution = DistributedExecution(root_rank=0)
        time = pt.arange(30, dtype=pt.float64) * 0.1
        state = pt.stack(
            (
                pt.cos(1.7 * time),
                pt.sin(1.7 * time),
                0.5 * pt.cos(1.7 * time),
                0.5 * pt.sin(1.7 * time),
            )
        )
        options = {
            "execution": execution,
            "spatial_batch_size": 1,
            "snapshot_batch_size": 5,
        }
        distributed_dmd = DMD(rank=2, subtract_mean=False, **options).fit(
            Trajectory(MatrixSource(state), time)
        )
        result = distributed_dmd.predict(MatrixSource(state[:, :1]), time=time[:8]).mean
        gathered = result.gather(root_rank=0)
        if rank == 0:
            serial = DMD(rank=2, subtract_mean=False).fit(Trajectory(state, time))
            expected = serial.predict(state[:, 0], time=time[:8]).mean
            pt.testing.assert_close(gathered, expected, rtol=1.0e-6, atol=1.0e-7)

        distributed_hodmd = DMD(
            rank=2,
            subtract_mean=False,
            embedding=TimeDelayEmbedding(2),
        ).fit(Trajectory(MatrixSource(state), time))
        result = distributed_hodmd.predict(
            MatrixSource(state[:, :2]), time=time[1:8]
        ).mean
        gathered = result.gather(root_rank=0)
        if rank == 0:
            serial = DMD(
                rank=2,
                subtract_mean=False,
                embedding=TimeDelayEmbedding(2),
            ).fit(Trajectory(state, time))
            expected = serial.predict(state[:, :2], time=time[1:8]).mean
            pt.testing.assert_close(gathered, expected, rtol=1.0e-6, atol=1.0e-7)

        exponential_scalar = pt.exp(-0.4 * time)
        exponential = pt.stack((exponential_scalar, 0.5 * exponential_scalar))
        distributed_edmd = DMD(
            rank=1,
            subtract_mean=False,
            stationary_initialization=False,
            embedding=MonomialEmbedding(2),
            **options,
        ).fit(Trajectory(MatrixSource(exponential), time))
        result = distributed_edmd.predict(
            MatrixSource(exponential[:, :1]), time=time[:8]
        ).mean
        gathered = result.gather(root_rank=0)
        if rank == 0:
            serial = DMD(
                rank=1,
                subtract_mean=False,
                stationary_initialization=False,
                embedding=MonomialEmbedding(2),
            ).fit(Trajectory(exponential, time))
            expected = serial.predict(exponential[:, 0], time=time[:8]).mean
            pt.testing.assert_close(gathered, expected, rtol=1.0e-6, atol=1.0e-7)

        forcing = pt.sin(time[:-1]).unsqueeze(0)
        controlled = state.clone()
        distributed_dmdc = DMDc(rank=2, subtract_mean=False, **options).fit(
            ControlledTrajectory(MatrixSource(controlled), time, forcing)
        )
        result = distributed_dmdc.predict(
            MatrixSource(controlled[:, :1]),
            time=time[:6],
            forcing=forcing[:, :5],
        ).mean
        gathered = result.gather(root_rank=0)
        if rank == 0:
            assert gathered is not None
            assert gathered.shape == (4, 6)

        distributed_hodmdc = DMDc(
            rank=2,
            subtract_mean=False,
            embedding=TimeDelayEmbedding(2),
            control_embedding=TimeDelayEmbedding(2),
        ).fit(ControlledTrajectory(MatrixSource(controlled), time, forcing))
        result = distributed_hodmdc.predict(
            MatrixSource(controlled[:, :2]),
            time=time[1:6],
            forcing=forcing[:, 1:5],
            forcing_history=forcing[:, :1],
        ).mean
        gathered = result.gather(root_rank=0)
        if rank == 0:
            serial = DMDc(
                rank=2,
                subtract_mean=False,
                embedding=TimeDelayEmbedding(2),
                control_embedding=TimeDelayEmbedding(2),
            ).fit(ControlledTrajectory(controlled, time, forcing))
            expected = serial.predict(
                controlled[:, :2],
                time=time[1:6],
                forcing=forcing[:, 1:5],
                forcing_history=forcing[:, :1],
            ).mean
            pt.testing.assert_close(gathered, expected, rtol=1.0e-5, atol=1.0e-6)

        parameters = pt.linspace(
            -1.0, 1.0, state.shape[1], dtype=state.dtype
        ).unsqueeze(0)
        distributed_podi = PODI(rank=2, subtract_mean=False, **options).fit(
            ParametricSnapshots(MatrixSource(state), parameters)
        )
        result = distributed_podi.predict(parameters=parameters[:, :3]).mean
        gathered = result.gather(root_rank=0)
        if rank == 0:
            serial = PODI(rank=2, subtract_mean=False).fit(
                ParametricSnapshots(state, parameters)
            )
            expected = serial.predict(parameters=parameters[:, :3]).mean
            pt.testing.assert_close(gathered, expected, rtol=1.0e-5, atol=1.0e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_distributed_roms_match_serial(tmp_path):
    store_path = os.fspath(tmp_path / "distributed-rom-store")
    mp.spawn(_worker, args=(2, store_path), nprocs=2, join=True)
