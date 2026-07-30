"""Unit tests for the DMD class."""

# third party packages
from pytest import raises
import torch as pt

# flowtorch packages
from flowtorch.analysis.dmd import _dft_properties, _fb_consistent_operator, DMD


def test_dft_properties():
    props = _dft_properties(0.1, 500)
    assert len(props) == 3
    assert props == (10.0, 5.0, 10.0 / 500)


def test_fb_consistent_operator():
    X, Y = pt.rand((10, 4)), pt.rand((10, 4))
    log, op = _fb_consistent_operator(X, Y)
    assert op.dtype == X.dtype
    assert op.shape == (10, 10)
    keys = ["p_res", "d_res", "p_eps", "d_eps", "f_err", "b_err", "rho"]
    assert all([k in log.keys() for k in keys])


def test_DMD_single_matrix():
    data = pt.rand((10, 8))
    rows, cols = data.shape
    rank = 3
    dmd = DMD(data, dt=0.1, rank=rank)
    assert dmd.eigvals.shape == (rank,)
    assert dmd.eigvals.dtype == pt.complex64
    assert dmd.eigvecs.shape == (rank, rank)
    assert dmd.eigvecs.dtype == pt.complex64
    assert dmd.modes.dtype == pt.complex64
    assert dmd.modes.shape == (rows, rank)
    assert dmd.frequency.shape == (rank,)
    assert dmd.growth_rate.shape == (rank,)
    a = dmd.amplitude
    assert isinstance(a, pt.Tensor)
    assert a.shape == (rank,)
    assert a.dtype == pt.complex64
    dyn = dmd.dynamics
    assert isinstance(dyn, pt.Tensor)
    assert dyn.shape == (rank, cols)
    assert dyn.dtype == pt.complex64
    a = dmd.integral_contribution
    assert isinstance(a, pt.Tensor)
    assert a.shape == (rank,)
    assert a.dtype == pt.float32
    rec = dmd.reconstruction
    assert isinstance(rec, pt.Tensor)
    assert rec.shape == (rows, cols)
    assert rec.dtype == data.dtype
    partial = dmd.partial_reconstruction({0})
    assert isinstance(partial, pt.Tensor)
    assert partial.dtype == data.dtype
    assert partial.shape == (rows, cols)
    partial = dmd.partial_reconstruction({0, 2})
    assert isinstance(partial, pt.Tensor)
    assert partial.dtype == data.dtype
    assert partial.shape == (rows, cols)
    top = dmd.top_modes(10)
    top = dmd.top_modes(10, True)
    assert top.shape == (min(rank, 10),)
    assert top.dtype == pt.int64
    err = dmd.reconstruction_error
    assert isinstance(err, pt.Tensor)
    assert err.shape == (rows, cols)
    assert dmd.projection_error.shape == (rows, cols - 1)
    dft = dmd.dft_properties
    assert len(dft) == 3
    assert dft == (10.0, 5.0, 10.0 / (cols - 1))
    # unitary operator
    dmd = DMD(data, dt=0.1, rank=rank, unitary=True)
    assert dmd.operator.shape == (rank, rank)
    diag = dmd.operator.conj().T @ dmd.operator
    assert pt.allclose(diag, pt.diag(pt.ones(rank)), atol=1e-6)
    # forward-backward consistent operator
    dmd = DMD(data, dt=0.1, rank=rank, forward_backward=True)
    assert dmd.operator.shape == (rank, rank)
    assert dmd.fb_log is not None
    dmd = DMD(data, dt=0.1, rank=rank, forward_backward={"steps": 10})
    assert dmd.operator.shape == (rank, rank)
    assert dmd.fb_log is not None
    # optimal mode amplitudes
    dmd = DMD(data, dt=0.1, rank=rank, optimal=True)
    dmd = DMD(data, dt=0.1, rank=rank, unitary=True, optimal=True)
    a = dmd.amplitude
    assert isinstance(a, pt.Tensor)
    assert a.shape == (rank,)
    assert a.dtype == pt.complex64
    # total least-squares
    dmd = DMD(data, dt=0.1, tlsq=True)
    a = dmd.amplitude
    assert isinstance(a, pt.Tensor)
    assert a.dtype == pt.complex64
    dmd = DMD(data, dt=0.1, rank=rank, optimal=True, tlsq=True)
    a = dmd.amplitude
    assert isinstance(a, pt.Tensor)
    assert a.shape == (rank,)
    DX, DY = dmd.tlsq_error
    assert DX.shape == (data.shape[0], data.shape[1] - 1)
    assert DY.shape == (data.shape[0], data.shape[1] - 1)
    # predict member function
    dmd = DMD(data, dt=0.1, rank=3)
    prediction = dmd.predict(data[:, -1], 10)
    assert prediction.shape == (rows, 11)
    assert prediction.dtype == data.dtype
    data = data.type(pt.complex64)
    dmd = DMD(data, dt=0.1, rank=3)
    prediction = dmd.predict(data[:, -1], 10)
    assert prediction.shape == (rows, 11)
    assert prediction.dtype == data.dtype


def test_DMD_matrix_ensemble():
    data = [pt.rand((10, 8)), pt.rand((10, 6))]
    rows = 10
    rank = 3
    dmd = DMD(data, dt=0.1, rank=rank)
    assert dmd.eigvals.shape == (rank,)
    assert dmd.eigvals.dtype == pt.complex64
    assert dmd.eigvecs.shape == (rank, rank)
    assert dmd.eigvecs.dtype == pt.complex64
    assert dmd.modes.dtype == pt.complex64
    assert dmd.modes.shape == (rows, rank)
    assert dmd.frequency.shape == (rank,)
    assert dmd.growth_rate.shape == (rank,)
    amp = dmd.amplitude
    assert len(amp) == 2
    assert all([a.shape == (rank,) for a in amp])
    dyn = dmd.dynamics
    assert len(dyn) == 2
    assert all([d.shape == (rank, dm.shape[1]) for d, dm in zip(dyn, data)])
    imp = dmd.integral_contribution
    assert len(imp) == 2
    assert all([i.shape == (rank,) for i in imp])
    rec = dmd.reconstruction
    assert len(rec) == 2
    assert all([r.shape == dm.shape for r, dm in zip(rec, data)])
    partial = dmd.partial_reconstruction({0})
    assert all([r.shape == dm.shape for r, dm in zip(partial, data)])
    top = dmd.top_modes(10)
    top = dmd.top_modes(10, True)
    assert top.shape == (min(rank, 10),)
    assert top.dtype == pt.int64
    assert len(dmd.reconstruction_error) == 2
    assert dmd.projection_error.shape == (rows, 14 - 2)
    # unitary operator
    dmd = DMD(data, dt=0.1, rank=rank, unitary=True)
    assert dmd.operator.shape == (rank, rank)
    diag = dmd.operator.conj().T @ dmd.operator
    assert pt.allclose(diag, pt.diag(pt.ones(rank)), atol=1e-6)
    # forward-backward consistent operator
    dmd = DMD(data, dt=0.1, rank=rank, forward_backward=True)
    assert dmd.operator.shape == (rank, rank)
    assert dmd.fb_log is not None
    dmd = DMD(data, dt=0.1, rank=rank, forward_backward={"steps": 10})
    assert dmd.operator.shape == (rank, rank)
    assert dmd.fb_log is not None
    # optimal mode amplitudes
    dmd = DMD(data, dt=0.1, rank=rank, unitary=True, optimal=True)
    assert len(dmd.amplitude) == 2
    # total least-squares
    dmd = DMD(data, dt=0.1, tlsq=True)
    assert len(dmd.amplitude) == 2
    dmd = DMD(data, dt=0.1, rank=rank, optimal=True, tlsq=True)
    assert len(dmd.amplitude) == 2
    DX, DY = dmd.tlsq_error
    assert DX.shape == (10, 14 - 2)
    assert DY.shape == (10, 14 - 2)
