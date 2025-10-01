"""Unittests for computing the DFT."""

from pytest import raises
import torch as pt
from .dft import DFT, PDFT


def test_DFT():
    dm_complex_even = pt.tensor(
        [
            [0.0 + 1.0j, 2.0 + 3.0j, 4.0 + 5.0j, 6.0 + 7.0j],
            [0.1 + 1.0j, 2.0 + 3.0j, 4.0 + 5.0j, 6.0 + 7.0j],
        ]
    )
    dm_complex_odd = pt.tensor(
        [
            [0.0 + 1.0j, 2.0 + 3.0j, 4.0 + 5.0j],
            [0.1 + 1.0j, 2.0 + 3.0j, 4.0 + 5.0j],
        ]
    )
    dm_real_even = pt.tensor(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [0.1, 1.0, 2.0, 3.0, 4.0, 5.0],
        ]
    )
    dm_real_odd = pt.tensor(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.1, 1.0, 2.0, 3.0, 4.0],
        ]
    )
    # complex inputs, default arguments
    dft = DFT(dm_complex_even, 1.0)
    nfreq = dm_complex_even.shape[1] - 1
    assert dft.amplitude.shape == (nfreq,)
    assert dft.frequency.shape == (nfreq,)
    assert pt.is_complex(dft.modes)
    assert dft.modes.shape == (2, nfreq)
    pt.testing.assert_close(dft.modes.norm(dim=0), pt.ones(nfreq))
    pt.testing.assert_close(dft.reconstruction, dm_complex_even)
    # complex inputs, GPU
    if pt.cuda.is_available():
        dft = DFT(dm_complex_odd, 1.0, device="cuda")
        nfreq = dm_complex_odd.shape[1] - 1
        assert dft.amplitude.shape == (nfreq,)
        assert dft.frequency.shape == (nfreq,)
        assert pt.is_complex(dft.modes)
        assert dft.modes.shape == (2, nfreq)
        pt.testing.assert_close(dft.modes.norm(dim=0), pt.ones(nfreq))
        pt.testing.assert_close(dft.reconstruction, dm_complex_odd)
    # real inputs, default arguments
    dft = DFT(dm_real_even, 1.0)
    nfreq = dm_real_even.shape[1] // 2
    assert dft.amplitude.shape == (nfreq,)
    assert dft.frequency.shape == (nfreq,)
    assert pt.is_complex(dft.modes)
    assert dft.modes.shape == (2, nfreq)
    pt.testing.assert_close(dft.modes.norm(dim=0), pt.ones(nfreq))
    pt.testing.assert_close(dft.reconstruction, dm_real_even)
    part = dft.partial_reconstruction(0)
    assert part.shape == dm_real_even.shape
    assert part.dtype == dm_real_even.dtype
    part = dft.partial_reconstruction({0, 2})
    assert part.shape == dm_real_even.shape
    assert part.dtype == dm_real_even.dtype
    assert len(dft.top_modes(2, False)) == 2
    with raises(Exception):
        _ = dft.partial_reconstruction(99)
    # real inputs, GPU
    if pt.cuda.is_available():
        nfft = 12
        dft = DFT(dm_real_odd, 1.0, nfft=nfft, device="cuda")
        nfreq = nfft // 2
        assert dft.amplitude.shape == (nfreq,)
        assert dft.frequency.shape == (nfreq,)
        assert pt.is_complex(dft.modes)
        assert dft.modes.shape == (2, nfreq)
        pt.testing.assert_close(dft.modes.norm(dim=0), pt.ones(nfreq))
        rec = dft.reconstruction
        assert not pt.is_complex(rec)
        assert rec.shape == (2, dm_real_odd.shape[-1])
    # windowing
    dft = DFT(dm_real_even, 1.0, window="hann")
    rec = dft.reconstruction
    assert pt.isfinite(rec).all()
    assert dft.amplitude.shape == (nfreq // 2,)
    with raises(ValueError):
        _ = DFT(dm_complex_even, 1.0, window="hanning")


def test_PDFT():
    M, N = 100, 40
    dm = pt.rand((M, N))
    dft = PDFT(dm, 1.0, N)
    modes = dft.modes
    assert modes.shape == (M, N // 2)
    assert pt.is_complex(modes)
    rec = dft.reconstruction
    pt.testing.assert_close(rec, dm)
    assert len(dft.top_modes()) == 1
    # rank truncation
    dft = PDFT(dm, 1.0, 20)
    assert dft.svd.rank == 20
    modes = dft.modes
    assert modes.shape == (M, N // 2)
    rec = dft.reconstruction
    pt.testing.assert_close(rec, dft.svd.reconstruct(20))
