"""Unittests for computing the DFT."""

from pytest import raises
import pytest
from flowtorch.analysis import SVD
from flowtorch.analysis.spod import AMSPOD
from flowtorch.analysis.state_vector import (
    FieldSpec,
    StateVectorLayout,
    StateVectorSource,
)
import torch as pt
from flowtorch.analysis.dft import DFT, PDFT


def test_weighted_norm_uses_bounded_batches(monkeypatch):
    data = pt.randn(1024, 2050)
    weight = pt.linspace(0.5, 2, 1024)
    expected = (pt.fft.rfft(data, norm="ortho") * weight.sqrt()[:, None]).norm(dim=0)
    norm = pt.Tensor.norm
    shapes = []

    def record_norm(tensor, *args, **kwargs):
        shapes.append(tuple(tensor.shape))
        return norm(tensor, *args, **kwargs)

    monkeypatch.setattr(pt.Tensor, "norm", record_norm)
    result = DFT(data, 0.1, weight=weight, subtract_mean=False)
    assert shapes == [(1024, 1024), (1024, 2)]
    pt.testing.assert_close(result._amplitude, expected)


@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("window", ["boxcar", "hamming"])
@pytest.mark.parametrize("grad", [False, True])
def test_dft_preprocessing_preserves_input_and_gradients(
    complex_data, center, window, grad
):
    pt.manual_seed(42)
    data = pt.randn(5, 16, dtype=pt.float64)
    if complex_data:
        data = data + 1j * pt.randn_like(data)
    data.requires_grad_(grad)
    original = data.detach().clone()
    reference = data.detach().clone().requires_grad_(grad)
    weight = pt.linspace(0.5, 2, 5, dtype=pt.float64)
    result = DFT(data, 0.1, window=window, subtract_mean=center, weight=weight)
    centered = reference - reference.mean(dim=1, keepdim=True) if center else reference
    taper = (
        pt.ones(16, dtype=pt.float64)
        if window == "boxcar"
        else pt.hamming_window(16, dtype=pt.float64)
    )
    fft = pt.fft.fft if complex_data else pt.fft.rfft
    expected = fft(centered * taper, dim=-1, norm="ortho")
    expected_norms = (expected * weight.sqrt()[:, None]).norm(dim=0)
    pt.testing.assert_close(result._modes, expected)
    pt.testing.assert_close(result._amplitude, expected_norms)
    pt.testing.assert_close(data, original)
    if grad:
        result.amplitude.sum().backward()
        energy = expected_norms.square()
        if not complex_data:
            energy = energy * pt.tensor([1] + [2] * 7 + [1])
        (energy[int(center) :].sum() / taper.square().sum()).backward()
        pt.testing.assert_close(data.grad, reference.grad)


@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("null", [False, True])
def test_single_mode_avoids_normalizing_full_spectrum(monkeypatch, reduced, null):
    data = (
        pt.ones(5, 16, dtype=pt.float64) if null else pt.randn(5, 16, dtype=pt.float64)
    )
    result = PDFT(data, 0.1, rank=3) if reduced else DFT(data, 0.1)
    expected = result.modes[:, 2]
    expected_animation = result.mode_reconstruction(2)
    normalize = result._normalized_modes
    requested = []

    def selected_only(indices):
        assert isinstance(indices, int)
        requested.append(indices)
        return normalize(indices)

    monkeypatch.setattr(result, "_normalized_modes", selected_only)
    pt.testing.assert_close(result.get_mode(2), expected)
    pt.testing.assert_close(result.mode_reconstruction(2), expected_animation)
    assert requested == [3, 3]


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
    assert len(dft.top_modes(2)) == 2
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
    nfreq = dm_real_even.shape[1] // 2
    assert dft.amplitude.shape == (nfreq,)
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
    assert len(dft.top_modes(n=1)) == 1
    # rank truncation
    dft = PDFT(dm, 1.0, 20)
    assert dft.svd.rank == 20
    modes = dft.modes
    assert modes.shape == (M, N // 2)
    rec = dft.reconstruction
    pt.testing.assert_close(rec, dft.svd.reconstruct(20) + dft.svd.mean[:, None])


class _DFTSource(StateVectorSource):
    def __init__(self, data):
        self.data = data
        self.calls = 0
        self._layout = StateVectorLayout((FieldSpec("q"),), (data.shape[0],))

    @property
    def layout(self):
        return self._layout

    @property
    def n_snapshots(self):
        return self.data.shape[1]

    def read(self, spatial_slice, snapshot_slice):
        self.calls += 1
        return self.data[spatial_slice, snapshot_slice]


@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("window", ["boxcar", "hamming"])
def test_weighted_pdft_matches_dft(complex_data, center, window):
    pt.manual_seed(17)
    data = pt.randn(7, 24, dtype=pt.float64) + 5
    if complex_data:
        data = data + 1j * pt.randn_like(data)
    weight = pt.linspace(0.3, 2, 7, dtype=pt.float64)
    direct = DFT(data, 0.2, nfft=32, weight=weight, subtract_mean=center, window=window)
    reduced = PDFT(
        data, 0.2, rank=7, nfft=32, weight=weight, subtract_mean=center, window=window
    )
    pt.testing.assert_close(reduced.amplitude, direct.amplitude)
    pt.testing.assert_close(reduced.modes, direct.modes)
    pt.testing.assert_close(reduced.reconstruction, data)
    pt.testing.assert_close(direct.reconstruction, data)
    pt.testing.assert_close(
        (reduced.modes.abs().square() * weight[:, None]).sum(0),
        pt.ones(len(reduced.frequency), dtype=pt.float64),
    )
    pt.testing.assert_close(reduced.mode_similarity(), direct.mode_similarity())
    pt.testing.assert_close(reduced.get_mode(2), direct.get_mode(2))
    pt.testing.assert_close(
        reduced.mode_reconstruction(2), direct.mode_reconstruction(2)
    )
    pt.testing.assert_close(
        reduced.partial_reconstruction({1, 3}), direct.partial_reconstruction({1, 3})
    )


@pytest.mark.parametrize("nfft", [15, 16, 32])
@pytest.mark.parametrize("complex_data", [False, True])
def test_bin_energy_parseval(nfft, complex_data):
    pt.manual_seed(18)
    data = pt.randn(6, 15, dtype=pt.float64)
    if complex_data:
        data = data + 1j * pt.randn_like(data)
    weight = pt.tensor([1.0, 2.0, 3.0], dtype=pt.float64)
    dft = DFT(data, 0.1, nfft=nfft, weight=weight)
    expected = (
        ((data - data.mean(1, keepdim=True)).abs().square() * weight.repeat(2)[:, None])
        .sum(0)
        .mean()
    )
    pt.testing.assert_close(dft.amplitude.sum(), expected)
    pt.testing.assert_close(dft.spectral_density.sum() / (nfft * 0.1), expected)


def test_null_short_and_dc():
    dft = DFT(pt.ones(3, 8), 1.0)
    assert (dft.modes == 0).all()
    assert pt.isnan(dft.mode_similarity()).all()
    short = DFT(pt.tensor([[0.0, 1.0]]), 1.0)
    pt.testing.assert_close(short.spectral_density, short.amplitude * 2)
    uncentered = DFT(pt.ones(3, 8), 1.0, subtract_mean=False)
    assert uncentered.frequency[0] == 0
    pt.testing.assert_close(uncentered.amplitude.sum(), pt.tensor(3.0))
    pt.testing.assert_close(uncentered.partial_reconstruction(0), pt.ones(3, 8))
    pt.testing.assert_close(
        uncentered.mode_reconstruction(0, N=4), pt.ones(3, 4) / 3**0.5
    )
    for index in (-1, 99):
        with raises(ValueError):
            dft.partial_reconstruction(index)


@pytest.mark.parametrize("center", [False, True])
def test_lazy_pdft_svd_reuse(center):
    pt.manual_seed(19)
    data = pt.randn(9, 20, dtype=pt.float64) + 3
    source = _DFTSource(data)
    weight = pt.linspace(0.5, 2, 9, dtype=pt.float64)
    svd = SVD(
        source,
        rank=9,
        weight=weight,
        subtract_mean=center,
        spatial_batch_size=3,
        snapshot_batch_size=4,
    )
    reads = source.calls
    result = PDFT.from_svd(svd, 0.1)
    assert source.calls == reads
    direct = DFT(data, 0.1, weight=weight, subtract_mean=center)
    pt.testing.assert_close(result.amplitude, direct.amplitude)
    pt.testing.assert_close(result.get_mode(2).materialize_local(), direct.get_mode(2))
    pt.testing.assert_close(result.modes.materialize_local(), direct.modes)
    pt.testing.assert_close(result.reconstruction.materialize_local(), data)
    pt.testing.assert_close(
        result.partial_reconstruction({1, 3}).materialize_local(),
        direct.partial_reconstruction({1, 3}),
    )
    pt.testing.assert_close(
        result.mode_reconstruction(2).materialize_local(), direct.mode_reconstruction(2)
    )
    constructed = PDFT(
        source, 0.1, rank=9, weight=weight, subtract_mean=center, spatial_batch_size=3
    )
    pt.testing.assert_close(constructed.amplitude, direct.amplitude)


def test_truncated_centering_preserves_original_mean():
    pt.manual_seed(20)
    data = 100 + pt.randn(9, 20, dtype=pt.float64)
    result = PDFT(data, 0.1, rank=2)
    pt.testing.assert_close(result.reconstruction.mean(1), data.mean(1))
    pt.testing.assert_close(
        result.reconstruction, result.svd.reconstruct() + data.mean(1, keepdim=True)
    )


def test_segment_selection_matches_spod():
    pt.manual_seed(21)
    dft = DFT(pt.randn(3, 40, dtype=pt.float64), 0.1)
    spod = AMSPOD.__new__(AMSPOD)
    spod._frequency = dft.frequency
    spod._eigvals = dft.eigvals
    spod._complex = True
    spod._adaptive = False
    for spacing in ("log", "linear"):
        pt.testing.assert_close(
            dft.top_modes(spacing=spacing), spod.top_modes(spacing=spacing)
        )
    assert dft.top_modes(f_min=100).numel() == 0


def test_plotting_helpers():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dft = DFT(pt.randn(4, 20), 0.1)
    figure, axes = dft.show_spectrum(energy_density=True, reference_timescale=2.0)
    assert axes.figure is figure
    plt.close(figure)
    figure, axes = dft.show_mode_similarity()
    assert axes.figure is figure
    plt.close(figure)


@pytest.mark.parametrize("shape", [(3, 20), (20, 3)])
@pytest.mark.parametrize("lazy", [False, True])
def test_pdft_null_centered_data(shape, lazy):
    data = pt.ones(shape, dtype=pt.float64)
    result = PDFT(_DFTSource(data) if lazy else data, 1.0, rank=1)
    modes = result.modes.materialize_local() if lazy else result.modes
    reconstruction = (
        result.reconstruction.materialize_local() if lazy else result.reconstruction
    )
    assert (modes == 0).all()
    assert (result.amplitude == 0).all()
    pt.testing.assert_close(reconstruction, data)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0},
        {"dt": float("nan")},
        {"nfft": 0},
        {"nfft": 3.5},
        {"window": "unknown"},
        {"weight": pt.tensor([0.0, 1.0])},
    ],
)
def test_invalid_configuration(kwargs):
    options = {"dt": 1.0}
    options.update(kwargs)
    with raises(ValueError):
        DFT(pt.randn(2, 6), **options)
