"""Unittests for the SVD class.
"""

# third party libraries
import pytest
import torch as pt

# flowtorch packages
from .svd import SVD


class TestSVD:
    def test_init_cpu(self):
        M, N = 100, 15
        dm = pt.rand((M, N), dtype=pt.float32)
        svd = SVD(dm, rank=N)
        assert svd.U.shape == (M, N)
        assert svd.V.shape == (N, N)
        assert svd.s.shape == (N,)
        assert svd.s_rel.shape == (N,)
        assert svd.s_cum.shape == (N,)
        assert svd.mode == "evd"
        svd = SVD(dm, mode="svd")
        assert svd.mode == "svd"
        assert svd.opt_rank == svd.rank
        svd = SVD(pt.rand((N, N)))
        assert svd.mode == "svd"
        svd = SVD(dm, rank=M)
        assert svd.rank == N
        dm = pt.rand((N, M), dtype=pt.float32)
        svd = SVD(dm, rank=N)
        assert svd.mode == "evd"
        with pytest.raises(ValueError):
            _ = SVD(dm.unsqueeze(-1))
        with pytest.raises(ValueError):
            _ = SVD(dm, mode="abc")

    @pytest.mark.skipif(not pt.cuda.is_available(), reason="CUDA not available")
    def test_init_gpu(self):
        M, N = 100, 15
        dm = pt.rand((M, N), dtype=pt.float32, device="cuda")
        svd = SVD(dm, rank=N)
        assert svd.U.shape == (M, N)
        assert "cuda" in str(svd.U.device)
        assert svd.V.shape == (N, N)
        assert svd.s.shape == (N,)
        assert svd.s_rel.shape == (N,)
        assert svd.s_cum.shape == (N,)
        assert svd.mode == "evd"
        svd = SVD(dm, mode="svd")
        assert svd.mode == "svd"
        assert svd.opt_rank == svd.rank
        svd = SVD(pt.rand((N, N)))
        assert svd.mode == "svd"
        svd = SVD(dm, rank=M)
        assert svd.rank == N

    def test_reconstruct(self):
        M, N = 100, 15
        dm = pt.rand((M, N), dtype=pt.float32)
        svd = SVD(dm, N)
        pt.testing.assert_close(dm, svd.reconstruct())
        err_r1 = pt.linalg.norm(dm - svd.reconstruct(rank=1)).item()
        err_r2 = pt.linalg.norm(dm - svd.reconstruct(rank=2)).item()
        assert err_r2 <= err_r1
