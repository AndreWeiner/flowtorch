"""Unit tests for linear controller.
"""

from pytest import raises
import torch as pt
from .linear_control import LinearControlModel


class TestLinearControl:
    def test_init(self):
        dm = pt.rand((10, 20))
        cm = pt.rand(19)
        with raises(ValueError) as e:
            lc = LinearControlModel(dm, cm[:-1], 1.0)
        lc = LinearControlModel(dm, cm, 1.0)
        assert lc.A.shape == (10, 10)
        assert lc.A.dtype == dm.dtype
        assert lc.B.shape == (10, 1)
        assert lc.B.dtype == cm.dtype
        assert lc.eigvals.shape == (10,)
        assert lc.eigvecs.shape == (10, 10)
        assert lc.amplitude.shape == (10,)
        assert lc.eigvals_cont.shape == (10,)
        assert lc.frequency.shape == (10,)
        assert lc.growth_rate.shape == (10,)
        assert lc.modes.shape == (10, 10)
        assert len(lc.dft_properties) == 3

    def test_dynamics(self):
        dm = pt.rand((10, 20))
        cm = pt.rand(19)
        lc = LinearControlModel(dm, cm, 1.0)
        assert lc.unforced_dynamics.shape == (10, 20)
        assert lc.forced_dynamics.shape == (10, 20)
        assert lc.dynamics.shape == (10, 20)

    def test_reconstruction(self):
        dm = pt.rand((10, 20))
        cm = pt.rand(19)
        lc = LinearControlModel(dm, cm, 1.0)
        assert lc.reconstruction.shape == (10, 20)
        assert lc.reconstruction.dtype == dm.dtype
        assert lc.reconstruction_error.shape == (10, 20)
        assert lc.projection_error.shape == (10, 19)

    def test_predict(self):
        dm = pt.rand((10, 20))
        cm = pt.rand(19)
        lc = LinearControlModel(dm, cm, 1.0)
        with raises(ValueError) as e:
            lc.predict(dm[:, 0], dm[:2, :])
        pred = lc.predict(dm[:, 0], cm)
        assert pred.shape == (10, 20)
        assert pred.dtype == dm.dtype
