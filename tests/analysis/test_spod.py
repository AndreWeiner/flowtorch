"""Unittests for the adaptive multitaper SPOD (AMSPOD)."""

import pytest
from pytest import raises
import torch as pt
from flowtorch.analysis.spod import AMSPOD, PAMSPOD, _prepare_weights, _calc_mode_similarity, _free_memory


def test_prepare_weights():
    w = pt.tensor([1, 2, 3])
    # default behavior
    w_out = _prepare_weights(w, len(w))
    assert w_out.shape == (len(w), 1)
    # multiple stacked fields
    w_out = _prepare_weights(w, 2 * len(w))
    assert w_out.shape == (2 * len(w), 1)
    pt.testing.assert_close(w_out.squeeze(), pt.tensor([1, 2, 3, 1, 2, 3]).sqrt())
    # invalid combination of weights and desired length
    with raises(ValueError):
        _ = _prepare_weights(w, 5)


def test_calc_mode_similarity():
    m = pt.rand(3, dtype=pt.complex64)
    # modes aligned -> similarity of unity
    assert _calc_mode_similarity(m, m) - 1.0 < 1.0e-5
    assert _calc_mode_similarity(m, -m) - 1.0 < 1.0e-5
    # modes orthogonal -> similarity of zero
    m3 = -(m[0] * m[0].conj() + m[1] * m[1].conj()) / m[2].conj()
    m_orth = pt.tensor([m[0], m[1], m3])
    assert _calc_mode_similarity(m, m_orth) < 1.0e-5


def test_free_memory():
    _free_memory("cpu")
    if pt.cuda.is_available():
        _free_memory("cuda")
        _free_memory("cuda:0")
    assert True


def test_AMSPOD_cpu():
    # real input data, even number of snapshots
    M, N, K = 20, 30, 5
    dm_even_real = pt.rand((M, N), dtype=pt.float32)
    ## no adaptive selection, fewer tapers than rows
    spod = AMSPOD(dm_even_real, 1.0, adaptive=False, max_tapers=K, keep_n_modes=K)
    n_freq = N / 2 + 1
    assert spod.frequency.shape == (n_freq,)
    m = spod.modes
    assert m.dtype == pt.complex64
    assert m.shape == (n_freq, M, K)
    e = spod.eigvals
    assert e.dtype == pt.float32
    assert e.shape == (n_freq, K)
    top = spod.top_modes(5)
    assert len(top) == 5
    ## adaptive selection, keep only first mode
    spod = AMSPOD(dm_even_real, 1.0, max_tapers=K, keep_n_modes=1)
    m = spod.modes
    assert m.shape == (n_freq, M, 1)
    assert "convergence" in spod._log.keys()
    assert len(spod._log["convergence"]) == n_freq
    # real input data, odd number of snapshots, K > M
    M, N, K = 5, 31, 9
    dm_odd_real = pt.rand((M, N), dtype=pt.float32)
    spod = AMSPOD(dm_odd_real, 1.0, adaptive=False, max_tapers=K, keep_n_modes=K)
    n_freq = (N + 1) // 2
    assert spod.frequency.shape == (n_freq,)
    m = spod.modes
    assert m.dtype == pt.complex64
    assert m.shape == (n_freq, M, min(M, K))
    e = spod.eigvals
    assert e.dtype == pt.float32
    assert e.shape == (n_freq, min(M, K))
    # complex input data, even number of snapshots
    M, N, K = 20, 30, 5
    dm_even_complex = pt.rand((M, N), dtype=pt.complex64)
    ## no adaptive selection, fewer tapers than rows
    spod = AMSPOD(dm_even_complex, 1.0, adaptive=False, max_tapers=K, keep_n_modes=K)
    n_freq = N
    assert spod.frequency.shape == (n_freq,)
    m = spod.modes
    assert m.dtype == pt.complex64
    assert m.shape == (n_freq, M, K)
    e = spod.eigvals
    assert e.dtype == pt.float32
    assert e.shape == (n_freq, K)
    assert isinstance(spod.half_bandwidth, float)
    ## adaptive selection, keep only first mode
    spod = AMSPOD(dm_even_complex, 1.0, max_tapers=K, keep_n_modes=1)
    m = spod.modes
    assert m.shape == (n_freq, M, 1)
    bw = spod.half_bandwidth
    assert isinstance(bw, pt.Tensor)
    assert len(bw) == n_freq
    assert "convergence" in spod._log.keys()
    assert len(spod._log["convergence"]) == n_freq
    res = spod.residual
    assert res is not None
    assert res.shape == (n_freq, K - 2)
    rec = spod.mode_reconstruction(N // 2, 0)
    assert rec.shape == dm_even_complex.shape
    assert rec.dtype == dm_even_complex.dtype
    rec = spod.mode_reconstruction(N // 2, 0, 1.0, 10)
    assert rec.shape == (M, 10)
    assert rec.dtype == dm_even_complex.dtype
    with raises(ValueError):
        _ = spod.mode_reconstruction(N, 0)
    with raises(ValueError):
        _ = spod.mode_reconstruction(N // 2, 1)
    # complex input data, odd number of snapshots
    M, N, K = 20, 31, 5
    dm_odd_complex = pt.rand((M, N), dtype=pt.complex64)
    ## no adaptive selection, fewer tapers than rows
    spod = AMSPOD(dm_odd_complex, 1.0, adaptive=False, max_tapers=K, keep_n_modes=K)
    n_freq = N
    assert spod.frequency.shape == (n_freq,)
    m = spod.modes
    assert m.dtype == pt.complex64
    assert m.shape == (n_freq, M, K)
    e = spod.eigvals
    assert e.dtype == pt.float32
    assert e.shape == (n_freq, K)
    ## adaptive selection, keep only first mode
    spod = AMSPOD(dm_odd_complex, 1.0, max_tapers=K, keep_n_modes=1)
    m = spod.modes
    assert m.shape == (n_freq, M, 1)
    assert "convergence" in spod._log.keys()
    assert len(spod._log["convergence"]) == n_freq


@pytest.mark.skipif(not pt.cuda.is_available(), reason="CUDA not available")
def test_AMSPOD_cuda():
    # real input data, even number of snapshots
    M, N, K = 20, 30, 5
    dm_even_real = pt.rand((M, N), dtype=pt.float32)
    w = pt.rand(M, dtype=pt.float32)
    ## adaptive selection, fewer tapers than rows
    spod = AMSPOD(
        dm_even_real,
        1.0,
        adaptive=True,
        weight=w,
        max_tapers=K,
        keep_n_modes=K,
        device="cuda",
    )
    n_freq = N / 2 + 1
    assert spod.frequency.shape == (n_freq,)
    m = spod.modes
    assert m.dtype == pt.complex64
    assert m.shape == (n_freq, M, K)
    assert str(m.device) == "cpu"
    e = spod.eigvals
    assert e.dtype == pt.float32
    assert e.shape == (n_freq, K)
    assert str(e.device) == "cpu"


def test_PAMSPOD():
    M, N = 30, 20
    n_freq = N // 2 + 1
    dm = pt.rand((M, N), dtype=pt.float32)
    w = pt.rand(M, dtype=pt.float32)
    spod = PAMSPOD(dm, 1.0, weight=w, max_tapers=5, keep_n_modes=3)
    assert spod.svd.rank == N
    assert spod._dm.shape == (N, N)
    assert spod.modes.shape == (n_freq, M, 3)
    r = spod.mode_reconstruction(0, 0)
    assert r.shape == dm.shape
    assert r.dtype == dm.dtype
    r = spod.mode_reconstruction(0, 0, 1.0, 35, scale=True)
    assert r.shape == (M, 35)
    assert r.dtype == dm.dtype
    m = spod.get_mode(5, 0)
    assert m.shape == (M,)
