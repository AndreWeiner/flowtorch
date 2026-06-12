"""Unit tests for MSSA and PMSSA.
"""

import torch as pt
from flowtorch.analysis.mssa import MSSA, PMSSA


class TestMSSA:
    def test_init(self):
        rows, cols = 20, 13
        dm = pt.rand((rows, cols))
        # minimal construction
        mssa = MSSA(dm)
        assert mssa.window_size == cols // 2
        assert mssa.delay == cols - (cols // 2) + 1
        # prescript window size
        mssa = MSSA(dm, 5)
        assert mssa.window_size == 5
        assert mssa.delay == cols - 5 + 1
        assert mssa.svd._rows == rows * mssa.delay
        assert mssa.svd._cols == mssa.window_size
        # prescript window size and reconstruction rank
        mssa = MSSA(dm, 5, 5)
        assert mssa.svd.rank == 5
        # rank cannot be larger than window size
        mssa = MSSA(dm, 5, 6)
        assert mssa.svd.rank == 5

    def test_reconstruction(self):
        rows, cols = 20, 13
        dm = pt.rand((rows, cols))
        mssa = MSSA(dm)
        rec = mssa.reconstruction
        assert rec.shape == dm.shape
        dm = pt.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=pt.float32)
        mssa = MSSA(dm, 3, 2)
        rec = mssa.reconstruction
        assert pt.allclose(rec, dm)
        err = mssa.reconstruction_error
        assert pt.isclose(err.norm(), pt.tensor(0.0), atol=1e-5)


class TestPMSSA:
    def test_init(self):
        rows, cols, rank = 20, 13, 4
        dm = pt.rand((rows, cols))
        mssa = PMSSA(dm, rank=rank)
        assert mssa.window_size == cols // 2
        assert mssa.delay == cols - (cols // 2) + 1
        assert mssa.svd._rows == rank * mssa.delay
        assert mssa.svd._cols == mssa.window_size

    def test_reconstruction(self):
        rows, cols, rank = 20, 13, 4
        dm = pt.rand((rows, cols))
        mssa = PMSSA(dm, rank=rank)
        rec = mssa.reconstruction
        assert rec.shape == dm.shape
        err = mssa.reconstruction_error
        assert err.shape == dm.shape
