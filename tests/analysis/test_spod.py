"""Unittests for the adaptive multitaper SPOD (AMSPOD)."""

import pytest
from pytest import raises
import torch as pt
from flowtorch.analysis.spod import (
    AMSPOD,
    PAMSPOD,
    _prepare_weights,
    _calc_mode_similarity,
    _free_memory,
)


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
    top = spod.top_modes(5, eig_idx="sum")
    assert len(top) == 5
    top = spod.top_modes(5, f_min=0.1, f_max=0.4)
    assert 0 < len(top) <= 5
    assert bool((spod.frequency[top] >= 0.1).all())
    assert bool((spod.frequency[top] < 0.4).all())
    with raises(ValueError):
        _ = spod.top_modes(0)
    with raises(ValueError):
        _ = spod.top_modes(5, eig_idx=K)
    with raises(ValueError):
        _ = spod.top_modes(5, eig_idx="invalid")
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


def test_AMSPOD_top_modes_log_segments():
    spod = AMSPOD.__new__(AMSPOD)
    spod._frequency = pt.tensor([0.0, 1.0, 2.0, 5.0, 10.0, 20.0])
    spod._eigvals = pt.tensor(
        [[0.0, 0.0], [1.0, 3.0], [5.0, 1.0], [2.0, 8.0], [7.0, 2.0], [4.0, 10.0]]
    )
    spod._complex = True
    spod._adaptive = False

    pt.testing.assert_close(spod.top_modes(2), pt.tensor([2, 4]))
    pt.testing.assert_close(spod.top_modes(2, eig_idx=1), pt.tensor([1, 5]))
    pt.testing.assert_close(spod.top_modes(2, eig_idx="sum"), pt.tensor([2, 5]))
    pt.testing.assert_close(spod.top_modes(2, f_min=2.1, f_max=15.0), pt.tensor([3, 4]))


def test_AMSPOD_top_modes_negative_frequencies():
    spod = AMSPOD.__new__(AMSPOD)
    spod._frequency = pt.tensor([0.0, 1.0, 2.0, -20.0, -10.0, -5.0, -2.0, -1.0])
    spod._eigvals = pt.tensor(
        [[0.0], [1.0], [2.0], [4.0], [10.0], [3.0], [8.0], [5.0]]
    )
    spod._complex = True
    spod._adaptive = False

    pt.testing.assert_close(
        spod.top_modes(2, f_min=-float("inf"), f_max=0.0), pt.tensor([4, 6])
    )
    pt.testing.assert_close(spod.top_modes(3), pt.tensor([4, 6, 2]))


def test_AMSPOD_top_modes_adaptive_eig_idx_limit():
    spod = AMSPOD.__new__(AMSPOD)
    spod._frequency = pt.tensor([0.0, 1.0, 2.0, 5.0, 10.0, 20.0])
    spod._eigvals = pt.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 3.0, 9.0, 0.0],
            [5.0, 1.0, 8.0, 0.0],
            [2.0, 8.0, 7.0, 0.0],
            [7.0, 2.0, 6.0, 0.0],
            [4.0, 10.0, 5.0, 0.0],
        ]
    )
    spod._complex = True
    spod._adaptive = True
    spod._log = {"n_tapers": pt.tensor([2, 2, 3, 4, 3, 4])}

    with pytest.warns(UserWarning, match="using eig_idx=1"):
        top_modes = spod.top_modes(2, eig_idx=3)
    pt.testing.assert_close(top_modes, pt.tensor([1, 5]))


def test_AMSPOD_temporal_coefficients():
    M, N, K = 12, 14, 4
    dm = pt.rand((M, N), dtype=pt.float32)
    w = pt.rand(M, dtype=pt.float32)
    spod = AMSPOD(
        dm,
        1.0,
        adaptive=False,
        weight=w,
        max_tapers=K,
        keep_n_modes=2,
    )
    coeffs = spod.temporal_coefficients()
    assert coeffs.shape == (N // 2 + 1, 1, N)
    assert coeffs.dtype == pt.complex64
    modes = spod.modes[:, :, :1].permute(1, 0, 2).reshape(M, -1)
    weight = spod._weight.type(modes.dtype)
    modes_weighted = modes * weight
    snapshots = dm - dm.mean(dim=1).unsqueeze(-1)
    snapshots_weighted = snapshots.type(modes.dtype) * weight
    gram = modes_weighted.conj().T @ modes_weighted
    rhs = modes_weighted.conj().T @ snapshots_weighted
    expected = (pt.linalg.pinv(gram) @ rhs).reshape(N // 2 + 1, 1, N)
    pt.testing.assert_close(coeffs, expected)
    with pytest.warns(UserWarning, match="exceeds keep_n_modes"):
        coeffs = spod.temporal_coefficients(n_modes=3)
    assert coeffs.shape == (N // 2 + 1, 2, N)
    with raises(ValueError):
        _ = spod.temporal_coefficients(n_modes=0)


def test_AMSPOD_temporal_coefficients_adaptive_mode_limit():
    M, N = 12, 14
    dm = pt.rand((M, N), dtype=pt.float32)
    spod = AMSPOD(dm, 1.0, adaptive=True, max_tapers=2, keep_n_modes=5)
    with pytest.warns(UserWarning, match="minimum number of available modes"):
        coeffs = spod.temporal_coefficients(n_modes=3)
    assert coeffs.shape == (N // 2 + 1, 2, N)


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
