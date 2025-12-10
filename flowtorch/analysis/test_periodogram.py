"""Unittests for periodogram estimators."""

from pytest import raises
import torch as pt
from .periodogram import AMPS


def test_AMPS():
    # real input data, even number of samples
    N, K = 30, 5
    signal_even_real = pt.rand(N, dtype=pt.float32)
    ## no adaptive selection
    ps = AMPS(signal_even_real, 1.0, adaptive=False, max_tapers=K)
    n_freq = N / 2 + 1
    assert ps.frequency.shape == (n_freq,)
    p = ps.spectral_power
    assert p.dtype == pt.float32
    assert p.shape == (n_freq,)
    psd = ps.spectral_density
    assert psd.dtype == pt.float32
    assert psd.shape == (n_freq,)
    top = ps.top_power(5)
    assert len(top) == 5
    ## adaptive selection
    ps = AMPS(signal_even_real.unsqueeze(0), 1.0, max_tapers=K)
    assert "convergence" in ps.log.keys()
    assert len(ps.log["convergence"]) == n_freq
    bw = ps.half_bandwidth
    assert isinstance(bw, pt.Tensor)
    assert len(bw) == n_freq
    res = ps.residual
    assert res is not None
    assert res.shape == (n_freq, K - 2)
    ## invalid input signal
    with raises(ValueError):
        _ = AMPS(signal_even_real.unsqueeze(-1), 1.0, max_tapers=K)
    # real input signal, odd number of samples
    N, K = 31, 9
    signal_odd_real = pt.rand(N, dtype=pt.float32)
    ps = AMPS(signal_odd_real, 1.0, adaptive=False, max_tapers=K)
    n_freq = (N + 1) // 2
    assert ps.frequency.shape == (n_freq,)
    p = ps.spectral_power
    assert p.dtype == pt.float32
    assert p.shape == (n_freq,)
    # complex input signal, even number of samples
    N, K = 30, 5
    signal_odd_complex = pt.rand(N, dtype=pt.complex64)
    ## no adaptive selection, zero-padding
    ps = AMPS(signal_odd_complex, 1.0, nfft=2 * N, adaptive=False, max_tapers=K)
    n_freq = 2 * N
    assert ps.frequency.shape == (n_freq,)
    ## adaptive selection, zero-padding
    ps = AMPS(signal_odd_complex, 1.0, nfft=2 * N, max_tapers=K)
    assert "convergence" in ps.log.keys()
    assert len(ps.log["convergence"]) == n_freq
