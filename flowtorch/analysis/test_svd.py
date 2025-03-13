# third party libraries
import pytest
import torch as pt
# flowtorch packages
from flowtorch import DATASETS
from flowtorch.data import FOAMDataloader
from flowtorch.analysis import SVD, inexact_alm_matrix_completion


def create_noisy_low_rank_data():
    pt.manual_seed(0)
    L = pt.ones((100, 20)) * pt.randint(-5, 5, (20,))
    L[:50, :] = pt.ones((50, 20)) * pt.randint(-5, 5, (20,))
    S = 20 * pt.bernoulli(pt.ones_like(L)*0.1)
    S -= 20 * pt.bernoulli(pt.ones_like(L)*0.1)
    return L+S, L, S


class TestSVD:
    def setup_method(self, test_method):
        loader = FOAMDataloader(DATASETS["of_cavity_ascii"])
        self.data = loader.load_snapshot("p", loader.write_times[1:])
        self.rows, self.cols = self.data.shape

    def test_init(self):
        svd = SVD(self.data, rank=self.cols)
        assert svd.U.shape == self.data.shape
        assert svd.V.shape == (self.cols, self.cols)
        assert svd.s.shape == (self.cols,)
        assert svd.s_rel.shape == (self.cols,)
        assert svd.s_cum.shape == (self.cols,)
        assert pt.allclose(svd.s_cum[-1], pt.tensor(100.0))

    def test_optimal_rank(self):
        svd = SVD(self.data)
        assert 1 <= svd.opt_rank <= self.cols

    def test_reconstruct(self):
        data = self.data.type(pt.float64)
        svd = SVD(data, self.cols*2)
        assert pt.allclose(data, svd.reconstruct())
        assert pt.allclose(data, svd.reconstruct(self.cols*2))
        err_r1 = pt.linalg.norm(data - svd.reconstruct(rank=1)).item()
        err_r2 = pt.linalg.norm(data - svd.reconstruct(rank=2)).item()
        assert err_r2 <= err_r1

    def test_robust(self):
        X, low, noise = create_noisy_low_rank_data()
        # test if robust is True
        svd = SVD(X, 20, True)
        assert svd.robust
        assert svd.S.shape == X.shape
        assert svd.L.shape == X.shape
        assert svd.U.shape == X.shape
        assert pt.linalg.norm(low-svd.L) < 20.0
        # test if robust is False/default
        svd = SVD(X, 20)
        assert not svd.robust
        assert svd.L is None
        assert svd.S is None
        # test if empty dictionary is passed
        svd = SVD(X, 20, robust={})
        assert not svd.robust
        # test passing arguments to Inexact ALM
        svd = SVD(X, 20, robust={"sparsity": 1.0, "verbose": True})
        assert bool(svd.robust)
        assert pt.linalg.norm(low-svd.L) < 20.0


def test_inexact_alm_matrix_completion():
    X, low, noise = create_noisy_low_rank_data()
    L, S = inexact_alm_matrix_completion(X)
    assert L.shape == low.shape
    assert S.shape == noise.shape
    # very rough test to see is low rank tensor was found
    assert pt.linalg.norm(low-L) < 20.0
