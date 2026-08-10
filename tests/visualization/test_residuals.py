import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch as pt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from flowtorch.analysis import AMPS, AMSPOD
from flowtorch.visualization import (
    plot_adaptive_residual,
    plot_adaptive_residuals,
)


def _residual_data():
    frequency = pt.tensor([0.0, 1.0, 2.0, 4.0, 8.0])
    residual = pt.tensor(
        [
            [1.0e-1, 1.0e-2, 1.0e-3],
            [1.0e-2, 1.0e-3, float("nan")],
            [1.0e-3, 1.0e-4, 0.0],
            [1.0e-4, float("nan"), float("nan")],
            [1.0e-5, 1.0e-6, 1.0e-7],
        ]
    )
    return frequency, residual


def test_individual_residual_coordinates_labels_and_mask():
    frequency, residual = _residual_data()

    figure, axes = plot_adaptive_residual(frequency, residual)

    assert isinstance(figure, Figure)
    assert isinstance(axes, Axes)
    assert axes.get_xscale() == "log"
    assert axes.get_xlabel() == r"$f\;[\mathrm{Hz}]$"
    assert axes.get_ylabel() == r"$N_{\mathrm{tapers}}$"
    assert axes.get_ylim() == pytest.approx((3.0, 5.0))
    plotted = axes.collections[0].get_array()
    assert plotted.shape == (3, 4)
    assert np.ma.count_masked(plotted) == 3
    assert figure.axes[1].get_ylabel() == r"$\log_{10}(r)$"
    plt.close(figure)


def test_automatic_floor_uses_smallest_positive_selected_residual():
    frequency, residual = _residual_data()

    figure, axes = plot_adaptive_residual(
        frequency,
        residual,
        frequency_limits=(1.0, 4.0),
        colorbar=False,
    )

    plotted = axes.collections[0].get_array()
    assert np.nanmin(plotted) == pytest.approx(-4.0)
    assert len(figure.axes) == 1
    plt.close(figure)


def test_explicit_floor_and_color_limits():
    frequency, residual = _residual_data()
    mesh_kwargs = {"rasterized": True}

    figure, axes = plot_adaptive_residual(
        frequency,
        residual,
        residual_floor=1.0e-3,
        color_limits=(-3.0, -1.0),
        mesh_kwargs=mesh_kwargs,
    )

    assert np.nanmin(axes.collections[0].get_array()) == pytest.approx(-3.0)
    assert axes.collections[0].norm.vmin == -3.0
    assert axes.collections[0].norm.vmax == -1.0
    assert mesh_kwargs == {"rasterized": True}
    plt.close(figure)


def test_strouhal_and_linear_frequency_axis():
    frequency, residual = _residual_data()

    figure, axes = plot_adaptive_residual(
        frequency,
        residual,
        reference_timescale=0.25,
        log_frequency=False,
    )

    assert axes.get_xscale() == "linear"
    assert axes.get_xlabel() == r"$St$"
    assert axes.get_xlim() == pytest.approx((0.25, 2.0))
    plt.close(figure)


def test_existing_axes_and_custom_colorbar_label():
    frequency, residual = _residual_data()
    figure, supplied_axes = plt.subplots()
    options = {"label": r"$R$"}

    returned_figure, returned_axes = plot_adaptive_residual(
        frequency,
        residual,
        ax=supplied_axes,
        colorbar_kwargs=options,
    )

    assert returned_figure is figure
    assert returned_axes is supplied_axes
    assert figure.axes[1].get_ylabel() == r"$R$"
    assert options == {"label": r"$R$"}
    plt.close(figure)


def test_comparative_grid_has_shared_limits_and_hides_unused_axes():
    frequency, residual = _residual_data()
    residual_2 = residual[:, :2] * 10.0
    residual_3 = residual * 0.1

    figure, axes = plot_adaptive_residuals(
        [frequency, frequency, frequency],
        [residual, residual_2, residual_3],
        [r"$\epsilon=10^{-3}$", r"$\epsilon=10^{-4}$", r"$\epsilon=10^{-5}$"],
        n_rows=2,
        n_cols=2,
    )

    assert axes.shape == (2, 2)
    assert not axes[1, 1].get_visible()
    limits = [
        (
            axes.flat[index].collections[0].norm.vmin,
            axes.flat[index].collections[0].norm.vmax,
        )
        for index in range(3)
    ]
    assert limits[0] == limits[1] == limits[2]
    assert len(figure.axes) == 5
    assert figure.axes[-1].get_ylabel() == r"$\log_{10}(r)$"
    assert axes[0, 0].get_ylim() == pytest.approx((3.0, 5.0))
    plt.close(figure)


def test_comparative_reference_timescales_use_common_extent():
    frequency, residual = _residual_data()

    figure, axes = plot_adaptive_residuals(
        [frequency, frequency],
        [residual, residual],
        ["first", "second"],
        n_rows=1,
        n_cols=2,
        reference_timescale=[0.1, 0.2],
        colorbar=False,
    )

    assert axes[0, 0].get_xlabel() == r"$St$"
    assert axes[0, 1].get_xlabel() == r"$St$"
    assert axes[0, 0].get_xlim() == pytest.approx((0.1, 1.6))
    assert axes[0, 1].get_xlim() == pytest.approx((0.1, 1.6))
    assert len(figure.axes) == 2
    plt.close(figure)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"reference_timescale": 0.0}, "reference_timescale"),
        ({"residual_floor": 0.0}, "residual_floor"),
        ({"frequency_limits": (2.0, 1.0)}, "frequency_limits"),
        ({"color_limits": (-1.0, -2.0)}, "color_limits"),
    ],
)
def test_invalid_individual_options(kwargs, message):
    frequency, residual = _residual_data()
    with pytest.raises(ValueError, match=message):
        plot_adaptive_residual(frequency, residual, **kwargs)


def test_invalid_residual_and_comparative_layout():
    frequency, residual = _residual_data()
    with pytest.raises(ValueError, match="shape"):
        plot_adaptive_residual(frequency, residual[:-1])
    with pytest.raises(ValueError, match="two bins"):
        plot_adaptive_residual(frequency, residual, frequency_limits=(7.0, 9.0))
    with pytest.raises(ValueError, match="title"):
        plot_adaptive_residuals([frequency], [residual], [""], n_rows=1, n_cols=1)
    with pytest.raises(ValueError, match="subplot grid"):
        plot_adaptive_residuals(
            [frequency, frequency],
            [residual, residual],
            ["first", "second"],
            n_rows=1,
            n_cols=1,
        )


def test_amspod_and_amps_show_residual():
    signal = pt.sin(pt.linspace(0.0, 6.0 * pt.pi, 24))
    amps = AMPS(signal, 0.1, adaptive=True, max_tapers=5)
    spod_data = pt.stack(
        [pt.sin((index + 1) * pt.linspace(0.0, 6.0 * pt.pi, 24)) for index in range(6)]
    )
    spod = AMSPOD(
        spod_data,
        0.1,
        adaptive=True,
        max_tapers=5,
    )

    amps_figure, amps_axes = amps.show_residual(colorbar=False)
    spod_figure, spod_axes = spod.show_residual(colorbar=False)

    assert amps_axes.get_ylabel() == r"$N_{\mathrm{tapers}}$"
    assert spod_axes.get_ylabel() == r"$N_{\mathrm{tapers}}$"
    plt.close(amps_figure)
    plt.close(spod_figure)


def test_nonadaptive_show_residual_raises():
    signal = pt.rand(16)
    amps = AMPS(signal, 0.1, adaptive=False, max_tapers=4)
    spod = AMSPOD(signal.unsqueeze(0), 0.1, adaptive=False, max_tapers=4)

    with pytest.raises(RuntimeError, match="adaptive"):
        amps.show_residual()
    with pytest.raises(RuntimeError, match="adaptive"):
        spod.show_residual()
