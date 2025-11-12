"""Unit tests for the HODMD class."""

# third party packages
import pytest
import torch as pt

# flowtorch packages
from .hodmd import HODMD, _create_time_delays


def test_create_time_delays():
    dm = pt.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=pt.float32)
    assert pt.allclose(_create_time_delays(dm, 1), dm)
    expected = pt.tensor([[1, 2, 3], [5, 6, 7], [2, 3, 4], [6, 7, 8]], dtype=pt.float32)
    assert pt.allclose(_create_time_delays(dm, 2), expected)
    expected = pt.tensor(
        [[1, 2], [5, 6], [2, 3], [6, 7], [3, 4], [7, 8]], dtype=pt.float32
    )
    assert pt.allclose(_create_time_delays(dm, 3), expected)


class TestHODMD:
    def test_init(self):
        dm = pt.rand((50, 20))
        # minimal construction
        dmd = HODMD(dm, 1.0)
        dmd.delay == int(20 // 3)
        # explicit rank for reduction
        dmd = HODMD(dm, 1.0, rank_dr=10)
        assert dmd.svd_dr.rank == 10
        # explicit delay, no rank
        dmd = HODMD(dm, 1.0, delay=10)
        assert dmd.delay == 10
        # invalid delay
        with pytest.raises(ValueError):
            dmd = HODMD(dm, 1.0, delay=0)
        with pytest.raises(ValueError):
            dmd = HODMD(dm, 1.0, delay=20)
        # optional dmd options
        dmd = HODMD(dm, 1.0, optimal=True, rank_dr=10, rank=5)
        assert dmd._optimal == True
        assert dmd.svd_dr.rank == 10
        assert dmd.svd.rank == 5

    def test_modes(self):
        dm = pt.rand((50, 20))
        dmd = HODMD(dm, 1.0)
        modes = dmd.modes
        assert modes.shape == (50, dmd.svd.rank)

    def test_reconstruction_error(self):
        dm = pt.rand((50, 20))
        dmd = HODMD(dm, 1.0)
        assert dmd.reconstruction_error.shape == (50, 20)

    def test_tlsq_error(self):
        dm = pt.rand((50, 20))
        dmd = HODMD(dm, 1.0, rank_dr=15, rank=10, tlsq=True)
        _, cols = dmd._dm[0].shape
        dx, dy = dmd.tlsq_error
        assert dx.shape == (50, cols - 1)
        assert dy.shape == (50, cols - 1)

    def test_projection_error(self):
        dm = pt.rand((50, 20))
        dmd = HODMD(dm, 1.0, rank_dr=15, rank=10)
        _, cols = dmd._dm[0].shape
        err = dmd.projection_error
        assert err.shape == (50, cols - 1)

    def test_predict(self):
        dm = pt.rand((50, 20))
        dmd = HODMD(dm, 1.0, rank_dr=15, rank=10, delay=5)
        prediction = dmd.predict(pt.rand((50, 5)), 12)
        assert prediction.shape == (50, 13)
        assert prediction.dtype == dm.dtype
        # test special case of 1D signal
        dm = pt.rand((1, 20))
        dmd = HODMD(dm, 1.0, rank_dr=15, rank=10, delay=5)
        prediction = dmd.predict(pt.rand((1, 5)), 12)
        assert prediction.shape == (1, 13)
        assert prediction.dtype == dm.dtype
