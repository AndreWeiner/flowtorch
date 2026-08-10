import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch as pt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from flowtorch.analysis import AMSPOD
from flowtorch.visualization import plot_spod_spectra, plot_spod_spectrum


def _spectrum():
    frequency = pt.tensor([0.0, 1.0, 2.0, 3.0])
    eigenvalues = pt.tensor(
        [
            [4.0, 2.0, 1.0],
            [3.0, 1.0, 0.0],
            [2.0, 0.5, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    return frequency, eigenvalues


def test_plot_spod_spectrum_returns_modifiable_objects():
    frequency, eigenvalues = _spectrum()

    figure, axes = plot_spod_spectrum(frequency, eigenvalues)

    assert isinstance(figure, Figure)
    assert isinstance(axes, Axes)
    assert axes.get_yscale() == "log"
    assert axes.get_xlabel() == r"$f\;[\mathrm{Hz}]$"
    assert axes.get_ylabel() == r"$\lambda$"
    assert axes.get_xlim() == pytest.approx((0.0, 3.0))
    assert len(axes.lines) == 4
    assert np.allclose(axes.lines[0].get_ydata(), eigenvalues.sum(dim=1))
    plt.close(figure)


def test_modes_share_color_and_have_decreasing_opacity():
    frequency, eigenvalues = _spectrum()

    figure, axes = plot_spod_spectrum(
        frequency, eigenvalues, show_sum=False, mode_color="C2"
    )

    assert all(line.get_color() == "C2" for line in axes.lines)
    opacity = [line.get_alpha() for line in axes.lines]
    assert opacity == sorted(opacity, reverse=True)
    assert opacity[-1] == pytest.approx(0.25)
    assert np.isnan(axes.lines[2].get_ydata()[1:]).all()
    plt.close(figure)


def test_legend_has_only_sum_and_generic_mode_entries():
    frequency, eigenvalues = _spectrum()
    figure, axes = plot_spod_spectrum(frequency, eigenvalues)

    _, labels = axes.get_legend_handles_labels()

    assert labels == [r"$\lambda_{\mathrm{sum}}$", r"$\lambda_i$"]
    plt.close(figure)


def test_energy_density_uses_displayed_coordinate_spacing():
    frequency, eigenvalues = _spectrum()

    figure, axes = plot_spod_spectrum(
        frequency,
        eigenvalues,
        n_show=1,
        show_sum=False,
        reference_timescale=2.0,
        energy_density=True,
    )

    assert np.allclose(axes.lines[0].get_xdata(), 2.0 * frequency)
    assert np.allclose(axes.lines[0].get_ydata(), eigenvalues[:, 0] / 2.0)
    assert axes.get_xlabel() == r"$St$"
    assert axes.get_ylabel() == r"$\lambda/\Delta St$"
    plt.close(figure)


def test_normalize_energy_uses_each_spectrums_total_energy():
    frequency, eigenvalues = _spectrum()
    total_energy = eigenvalues.sum()

    figure, axes = plot_spod_spectrum(
        frequency,
        eigenvalues,
        normalize_energy=True,
    )

    assert np.allclose(axes.lines[0].get_ydata(), eigenvalues.sum(dim=1) / total_energy)
    assert axes.lines[0].get_ydata().sum() == pytest.approx(1.0)
    assert axes.get_ylabel() == r"$\lambda/E$"
    plt.close(figure)


def test_normalized_strouhal_density_integrates_to_one():
    frequency, eigenvalues = _spectrum()

    figure, axes = plot_spod_spectrum(
        frequency,
        eigenvalues,
        reference_timescale=2.0,
        energy_density=True,
        normalize_energy=True,
    )

    density_sum = axes.lines[0].get_ydata()
    assert np.sum(density_sum * 2.0) == pytest.approx(1.0)
    assert axes.get_ylabel() == r"$\lambda/(E\,\Delta St)$"
    plt.close(figure)


def test_half_bandwidth_is_scaled_with_reference_timescale():
    frequency, eigenvalues = _spectrum()

    figure, axes = plot_spod_spectrum(
        frequency,
        eigenvalues,
        reference_timescale=2.0,
        half_bandwidth=0.25,
    )

    segments = axes.collections[0].get_segments()
    assert segments[0][:, 0] == pytest.approx((-0.5, 0.5))
    assert axes.collections[0].get_edgecolor()[0] == pytest.approx(
        matplotlib.colors.to_rgba("C3", alpha=0.18)
    )
    _, labels = axes.get_legend_handles_labels()
    assert labels[-1] == r"$\Delta St_{1/2}$"
    plt.close(figure)


def test_plot_spod_spectra_uses_individual_timescales_and_hides_unused_axes():
    frequency, eigenvalues = _spectrum()

    figure, axes = plot_spod_spectra(
        [frequency, frequency],
        [eigenvalues, eigenvalues],
        [r"$Re=100$", r"$Re=200$"],
        n_rows=2,
        n_cols=2,
        n_show=1,
        reference_timescale=[1.0, 2.0],
        energy_density=True,
    )

    assert axes.shape == (2, 2)
    assert axes[0, 0].get_title() == r"$Re=100$"
    assert axes[0, 1].get_title() == r"$Re=200$"
    assert np.allclose(axes[0, 1].lines[0].get_xdata(), 2.0 * frequency)
    assert axes[0, 0].get_xlim() == pytest.approx((0.0, 6.0))
    assert axes[0, 1].get_xlim() == pytest.approx((0.0, 6.0))
    assert not axes[1, 0].get_visible()
    assert not axes[1, 1].get_visible()
    assert figure._supxlabel.get_text() == r"$St$"
    assert figure._supylabel.get_text() == r"$\lambda/\Delta St$"
    plt.close(figure)


def test_plot_spod_spectra_rejects_insufficient_grid():
    frequency, eigenvalues = _spectrum()

    with pytest.raises(ValueError, match="grid must contain"):
        plot_spod_spectra(
            [frequency, frequency],
            [eigenvalues, eigenvalues],
            ["one", "two"],
            n_rows=1,
            n_cols=1,
        )


def test_amspod_show_spectrum_marks_half_bandwidth():
    frequency, eigenvalues = _spectrum()
    spod = AMSPOD.__new__(AMSPOD)
    spod._frequency = frequency
    spod._eigvals = eigenvalues
    spod._complex = True
    spod._adaptive = False
    spod._max_tapers = 3
    spod._nfft = 4
    spod._dt = 0.25

    figure, axes = spod.show_spectrum(n_show=2)

    assert len(axes.lines) == 3
    assert len(axes.collections) == 1
    _, labels = axes.get_legend_handles_labels()
    assert labels == [
        r"$\lambda_{\mathrm{sum}}$",
        r"$\lambda_i$",
        r"$\Delta f_{1/2}$",
    ]
    plt.close(figure)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_show": 0}, "n_show"),
        ({"reference_timescale": 0.0}, "reference_timescale"),
        ({"min_opacity": 0.0}, "min_opacity"),
        ({"energy_density": True}, "uniformly spaced"),
    ],
)
def test_plot_spod_spectrum_rejects_invalid_options(kwargs, message):
    frequency, eigenvalues = _spectrum()
    if kwargs == {"energy_density": True}:
        frequency = pt.tensor([0.0, 1.0, 2.5, 3.0])
    with pytest.raises(ValueError, match=message):
        plot_spod_spectrum(frequency, eigenvalues, **kwargs)


def test_plot_spod_spectrum_rejects_normalizing_zero_energy():
    frequency, eigenvalues = _spectrum()

    with pytest.raises(ValueError, match="zero total energy"):
        plot_spod_spectrum(
            frequency,
            pt.zeros_like(eigenvalues),
            normalize_energy=True,
        )
