"""Unit tests for the OptDMD class.
"""

# third party packages
import torch as pt

# flowtorch packages
from flowtorch.analysis.optdmd import OptDMD, _create_conj_complex_pairs


def test_create_conj_complex_pairs():
    # odd number of eigenvalues, unique real eigenvalue
    ev = pt.tensor([1.0 + 1.0j, 0.5 - 0.5j, 1.0 - 1.0j, 0.5 + 0.5j, 1.0 + 0.0j])
    keep, pairs = _create_conj_complex_pairs(ev)
    assert all(keep == pt.tensor((0, 3, 4), dtype=pt.int64))
    assert all(pairs == pt.tensor((0, 1), dtype=pt.int64))
    # odd number of eigenvalues, several real eigenvalues
    ev = pt.tensor(
        [
            1.0 + 1.0j,
            0.1 + 0.0j,
            0.5 - 0.5j,
            1.0 - 1.0j,
            0.5 + 0.5j,
            1.0 + 0.0j,
            2.0 + 0.0j,
        ]
    )
    keep, pairs = _create_conj_complex_pairs(ev)
    assert all(keep == pt.tensor((0, 4, 1, 5), dtype=pt.int64))
    assert all(pairs == pt.tensor((0, 1, 2), dtype=pt.int64))
    # even number of eigenvalues
    ev = pt.tensor([1.0 + 1.0j, 0.5 - 0.5j, 1.0 - 1.0j, 0.5 + 0.5j])
    keep, pairs = _create_conj_complex_pairs(ev)
    assert all(keep == pt.tensor((0, 3), dtype=pt.int64))
    assert all(pairs == pt.tensor((0, 1), dtype=pt.int64))
    # even number of eigenvalues, multiple real eigenvalues
    ev = pt.tensor(
        [
            1.0 + 1.0j,
            0.1 + 0.0j,
            0.5 - 0.5j,
            1.0 - 1.0j,
            0.5 + 0.5j,
            1.0 + 0.0j,
            2.0 + 0.0j,
            2.0 - 0.0j,
        ]
    )
    keep, pairs = _create_conj_complex_pairs(ev)
    assert all(keep == pt.tensor((0, 4, 1, 5), dtype=pt.int64))
    assert all(pairs == pt.tensor((0, 1, 2, 3), dtype=pt.int64))


class TestOptDMD:
    def test_init(self):
        dm = pt.rand((50, 20))
        # construction with even rank
        rank = 10
        dmd = OptDMD(dm, 1.0, rank)
        assert dmd.dmd_init.svd.rank == rank
        assert dmd.eigvecs.shape == (50, rank)
        assert dmd.eigvals.size(0) == rank
        # construction with odd rank
        rank = 11
        dmd = OptDMD(dm, 1.0, rank)
        assert dmd.dmd_init.svd.rank == rank
        assert dmd.eigvecs.shape == (50, rank)
        assert dmd.eigvals.size(0) == rank
        # construction with complex data matrix
        dm = pt.rand((50, 20), dtype=pt.complex64)
        rank = 11
        dmd = OptDMD(dm, 1.0, rank)
        assert dmd.dmd_init.svd.rank == rank
        assert dmd.eigvecs.shape == (50, rank)
        assert dmd.eigvals.size(0) == rank
        assert dmd.modes.shape == (50, rank)

    def test_create_train_val_split(self):
        dm = pt.rand((50, 21))
        dmd = OptDMD(dm, 1.0, 10)
        train, val = dmd._create_train_val_split(0.5, 0.5)
        assert len(train) == 10
        assert len(val) == 11
        assert train[:][0].tolist() == list(range(10))
        assert val[:][0].tolist() == list(range(10, 21))

    def test_forward(self):
        dm = pt.rand((50, 20))
        dmd = OptDMD(dm, 1.0, 10)
        test_input = (pt.tensor(range(5), dtype=pt.int64),)
        pred = dmd(test_input)
        assert pred.shape == (50, 5)
        assert pred.requires_grad

    def test_train(self):
        dm = pt.rand((50, 20))
        dmd = OptDMD(dm, 1.0, rank=10)
        dmd.train(5)
        assert len(dmd.log["train_loss"]) == 5
        assert isinstance(dmd.log["train_loss"][-1], float)
        assert len(dmd.log["val_loss"]) == 5
        assert isinstance(dmd.log["val_loss"][-1], float)
        assert len(dmd.log["full_loss"]) == 5
        assert isinstance(dmd.log["full_loss"][-1], float)
        # no validation data
        dmd = OptDMD(dm, 1.0, 10)
        dmd.train(5, val_size=0, loss_key="train_loss")
        assert "val_loss" not in dmd.log
        # batch size larger than possible
        dmd.train(5, batch_size=200)
        # train with complex data matrix
        dm = pt.rand((50, 20), dtype=pt.complex64)
        dmd = OptDMD(dm, 1.0, rank=10)
        dmd.train(5)
        assert True

    def test_partial_reconstruction(self):
        dm = pt.rand((50, 20))
        dmd = OptDMD(dm, 1.0, rank=10)
        rec = dmd.partial_reconstruction((0, 1))
        assert rec.shape == (50, 20)
        assert rec.dtype == dm.dtype

    def test_top_modes(self):
        dm = pt.rand((50, 20))
        dmd = OptDMD(dm, 1.0, rank=5)
        top = dmd.top_modes(1, True, 0.0)
        assert len(top) == 1
        im = dmd.integral_contribution
        assert im[top] == im.max()

    def test_reconstruction(self):
        dm = pt.rand((50, 20))
        dmd = OptDMD(dm, 1.0, rank=5)
        rec = dmd.reconstruction
        assert rec.shape == (50, 20)
        assert rec.dtype == dm.dtype

    def test_predict(self):
        dm = pt.rand((50, 20))
        dmd = OptDMD(dm, 1.0, rank=5)
        dmd.train(10)
        pred = dmd.predict(dm[:, -1], 10)
        assert pred.shape == (50, 11)
        assert pred.dtype == dm.dtype
