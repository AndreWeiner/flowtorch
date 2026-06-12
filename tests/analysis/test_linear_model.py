"""Unit tests for the LinearModel class.
"""

import pytest
from pytest import raises
import torch as pt
from flowtorch.analysis.utils import trajectory_train_test_split
from flowtorch.analysis.linear_model import _data_consistent, _least_squares_operator, LinearModel


def test_data_consistent():
    # single data matrix
    assert _data_consistent([pt.rand((20, 10))])
    # two data matrices
    assert _data_consistent([pt.rand((20, 10)), pt.rand((20, 10))])
    # two data matrices with different shapes
    assert not _data_consistent([pt.rand((20, 10)), pt.rand((20, 15))])
    # two data matrices with different data types
    assert not _data_consistent(
        [pt.rand((20, 10), dtype=pt.float32), pt.rand((20, 15), dtype=pt.complex64)]
    )
    # too few snapshots
    assert not _data_consistent([pt.rand((10, 1)), pt.rand(10, 1)])


def test_least_squares_operator():
    # single data matrix - under-determined
    A = _least_squares_operator([pt.rand((20, 10))], False)
    assert A.shape == (20, 20)
    # single data_matrix - over-determined
    A = _least_squares_operator([pt.rand((20, 30))], False)
    assert A.shape == (20, 20)
    # two data matrices - under-determined, consistent
    A = _least_squares_operator([pt.rand((20, 5)), pt.rand((20, 5))], True)
    assert A.shape == (20, 20)
    # two data matrices - over-determined, consistent
    A = _least_squares_operator([pt.rand((20, 15)), pt.rand((20, 15))], True)
    assert A.shape == (20, 20)


class TestLinearModel:
    def test_init(self):
        # construction using a single data matrix
        dm = pt.rand((20, 10))
        lm = LinearModel(dm, 1.0)
        assert len(lm._dm) == 1
        # construction with invalid data
        with raises(ValueError):
            lm = LinearModel([dm, pt.rand((20, 1))], 1.0)
        # construction with eigendecomposition
        lm = LinearModel(dm, 1.0, decompose_operator=True, rank=5)
        assert lm._eigvals.shape == (5,)
        assert lm._eigvecs.shape == (20, 5)
        assert lm._A is None

    def test_forward(self):
        # test batch size > 1
        dm = pt.rand((10, 22))
        lm = LinearModel(dm, 1.0)
        ds_train, _ = trajectory_train_test_split(dm, 1.0, 0.0, horizon=2)
        fi, f, _, l = ds_train[:]
        lm._A = pt.nn.Parameter(pt.eye(10))
        lm._noise = pt.nn.Parameter(pt.zeros_like(lm._noise))
        p = lm.forward(f, fi, l.shape[-1], False)
        assert p.shape == l.shape
        assert pt.allclose(p[0, :, 0], f[0])
        assert pt.allclose(p[2, :, 0], f[2])
        ds_train, _ = trajectory_train_test_split(
            dm, 1.0, 0.0, horizon=2, forward_backward=True
        )
        *_, fbi, fb, _, lb = ds_train[:]
        p = lm.forward(fb, fbi, lb.shape[-1], True)
        assert p.shape == lb.shape
        assert pt.allclose(p[0, :, 0], fb[0])
        assert pt.allclose(p[2, :, 0], fb[2])
        # test batch size = 1
        ds_train, _ = trajectory_train_test_split(dm, 1.0, 0.0, horizon=21)
        fi, f, _, l = ds_train[:]
        p = lm.forward(f, fi, l.shape[-1], False)
        assert p.shape == l.shape
        assert pt.allclose(p[0, :, 0], f[0])
        # same tests for decomposed operator
        lm = LinearModel(dm, 1.0, decompose_operator=True, rank=5)
        ds_train, _ = trajectory_train_test_split(
            dm, 1.0, 0.0, horizon=2, forward_backward=True
        )
        ffi, ff, _, lf, fbi, fb, _, lb = ds_train[:]
        p = lm.forward(ff, ffi, lf.shape[-1], False)
        assert p.shape == lf.shape
        p = lm.forward(fb, fbi, lb.shape[-1], True)
        assert p.shape == lb.shape
        lm = LinearModel(dm, 1.0, decompose_operator=True, rank=5)
        ds_train, _ = trajectory_train_test_split(
            dm, 1.0, 0.0, horizon=21, forward_backward=True
        )
        ffi, ff, _, lf, fbi, fb, _, lb = ds_train[:]
        p = lm.forward(ff, ffi, lf.shape[-1], False)
        assert p.shape == lf.shape
        p = lm.forward(fb, fbi, lb.shape[-1], True)
        assert p.shape == lb.shape

    def test_train(self):
        # training with three data matrices, gradient descent
        dm = [pt.rand((10, 8)) for _ in range(12)]
        lm = LinearModel(dm, 1.0)
        lm.train(10, split_options={"forward_backward": True, "horizon": 4})
        assert len(lm.log["train_loss"]) == 10
        assert len(lm.log["val_loss"]) == 10
        assert len(lm.log["train_loss_f"]) == 10
        assert len(lm.log["val_loss_f"]) == 10
        assert len(lm.log["train_loss_b"]) == 10
        assert len(lm.log["val_loss_b"]) == 10
        assert len(lm.log["full_loss"]) == 10
        # batch size = 1 (stochastic gradient descent)
        lm = LinearModel(dm, 1.0)
        lm.train(10, batch_size=1, optimizer_options={"lr": 1.0e-2})
        assert len(lm.log["train_loss"]) == 10
        assert len(lm.log["val_loss"]) == 10
        assert lm.log["lr"][0] == 1.0e-2
        # batch size larger than possible
        lm = LinearModel(dm, 1.0)
        lm.train(10, batch_size=10)
        assert len(lm.log["train_loss"]) == 10
        assert len(lm.log["val_loss"]) == 10
        # decomposed operator
        lm = LinearModel(dm, 1.0, decompose_operator=True, rank=3)
        lm.train(10, split_options={"forward_backward": True, "horizon": 4})
        assert len(lm.log["train_loss"]) == 10
        assert len(lm.log["val_loss"]) == 10

    @pytest.mark.skipif(not pt.cuda.is_available(), reason="no GPU - skipping test")
    def test_train_gpu(self):
        dm = [pt.rand((10, 8)) for _ in range(12)]
        lm = LinearModel(dm, 1.0)
        lm.train(
            10, split_options={"forward_backward": True, "horizon": 4}, device="cuda"
        )
        assert len(lm.log["train_loss"]) == 10
        assert len(lm.log["val_loss"]) == 10
        assert len(lm.frequency) == 10
        lm = LinearModel(dm, 1.0, decompose_operator=True, rank=4)
        lm.train(
            10, split_options={"forward_backward": True, "horizon": 4}, device="cuda"
        )
        assert len(lm.log["train_loss"]) == 10
        assert len(lm.log["val_loss"]) == 10
        assert lm.A.shape == (10, 10)

    def test_predict(self):
        dm = pt.rand((10, 12))
        lm = LinearModel(dm, 1.0)
        lm.train(10)
        pred = lm.predict(dm[:, 0], 5)
        assert pred.dtype == dm.dtype
        assert pred.shape == (10, 6)
        lm = LinearModel(dm, 1.0, decompose_operator=True, rank=4)
        lm.train(10)
        pred = lm.predict(dm[:, 0], 5)
        assert pred.dtype == dm.dtype
        assert pred.shape == (10, 6)

    def test_properties(self):
        dm = [pt.rand((10, 12)) for _ in range(3)]
        lm = LinearModel(dm, 1.0)
        lm.train(10)
        assert lm.A.shape == (10, 10)
        assert lm.eigvecs.shape == (10, 10)
        assert lm.eigvals.shape == (10,)
        assert lm.frequency.shape == (10,)
        assert lm.growth_rate.shape == (10,)
        a = lm.amplitude
        assert a[0].shape == (10,)
        assert len(a) == 3
        d = lm.dynamics
        assert d[0].shape == (10, 12)
        assert len(d) == 3
        i = lm.integral_contribution
        assert i[0].shape == (10,)
        assert len(i) == 3
        rec = lm.reconstruction
        assert rec[0].shape == (10, 12)
        assert rec[0].dtype == dm[0].dtype
        assert len(rec) == 3
        err = lm.reconstruction_error
        assert err[0].shape == (10, 12)
        assert len(err) == 3
        lm = LinearModel(dm, 1.0, decompose_operator=True, rank=3)
        lm.train(10)
        assert lm.A.shape == (10, 10)
        assert lm.eigvecs.shape == (10, 3)
        assert lm.eigvals.shape == (3,)
        assert lm.frequency.shape == (3,)
