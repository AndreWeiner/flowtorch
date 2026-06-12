"""Unit tests for `flowtorch.analysis.utils` module.
"""

from os.path import join, isfile
from os import remove
from pytest import raises
import torch as pt
from flowtorch.analysis.optdmd import OptDMD
from flowtorch.analysis.utils import unsqueeze_if_1d, trajectory_train_test_split, EarlyStopping


def test_trajectory_train_test_split():
    # single trajectory
    ## default options
    dm = pt.rand((5, 21))
    train, test = trajectory_train_test_split(dm)
    assert len(train) == 15
    assert len(test) == 5
    ## no test data
    train, test = trajectory_train_test_split(dm, train_size=1.0, test_size=0.0)
    assert len(train) == 20
    assert len(test) == 0
    ## increased shift
    train, test = trajectory_train_test_split(dm, n_shift=2)
    assert len(train) == 7
    assert len(test) == 3
    ## increased horizon
    dm = pt.rand((5, 22))
    train, test = trajectory_train_test_split(dm, horizon=2)
    assert len(train) == 15
    assert len(test) == 5
    ## forward-backward pairs
    train, test = trajectory_train_test_split(
        dm, horizon=2, forward_backward=True, shuffle=True
    )
    assert len(train) == 15
    assert len(test) == 5
    ffi, ff, lfi, lf, fbi, fb, lbi, lb = train[:]
    assert ffi.shape == (15,)
    assert ff.shape == (15, 5)
    assert lfi.shape == (15, 2)
    assert lf.shape == (15, 5, 2)
    assert fbi.shape == (15,)
    assert fb.shape == (15, 5)
    assert lbi.shape == (15, 2)
    assert lb.shape == (15, 5, 2)
    assert pt.allclose(ff[0], lb[0, :, -1])
    assert pt.allclose(dm[:, ffi[0]], ff[0])
    assert pt.allclose(dm[:, fbi[-1]], fb[-1])

    # multiple trajectories
    ## defaults options
    dm = [pt.rand((5, 21)) for _ in range(2)]
    train, test = trajectory_train_test_split(dm)
    ffi, *_ = train[:]
    assert len(train) == 15 * 2
    assert len(test) == 5 * 2
    ## invalid horizon
    with raises(ValueError):
        train, test = trajectory_train_test_split(dm, horizon=21)
    ## invalid trajectories
    with raises(ValueError):
        train, test = trajectory_train_test_split([pt.rand((5, 5)), pt.rand((4, 5))])


class TestEarlyStopping:
    def test_init(self):
        stopper = EarlyStopping()
        stop = stopper(1.0)
        assert not stop
        assert stopper._best_loss == 1.0
        _ = stopper(2.0)
        assert stopper._counter == 1

    def test_checkpoint(self):
        dm = pt.rand((50, 20))
        dmd = OptDMD(dm, 1.0, 5)
        chp = join("/tmp", "optDMDchp.pt")
        stopper = EarlyStopping(checkpoint=chp, model=dmd)
        _ = stopper(1.0)
        assert isfile(chp)
        eigs_before = dmd.eigvals
        dmd.train(3)
        dmd.load_state_dict(pt.load(chp))
        eigs_after = dmd.eigvals
        assert all(pt.isclose(eigs_after, eigs_before))
        if isfile(chp):
            remove(chp)


def test_unsqueeze_if_1d():
    tmp = unsqueeze_if_1d(pt.empty(10))
    assert tmp.shape == (1, 10)
    tmp = unsqueeze_if_1d(pt.empty(10), True)
    assert tmp.shape == (10, 1)
    tmp = unsqueeze_if_1d(pt.empty(10, 10))
    assert tmp.shape == (10, 10)
