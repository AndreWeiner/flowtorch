import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch as pt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from flowtorch.analysis import AMSPOD
from flowtorch.visualization import plot_spod_time_coefficients


def _coefficient_data():
    time = pt.linspace(0.0, 0.5, 6)
    frequency = pt.tensor([0.0, 2.0, 4.0, 6.0])
    coefficients = pt.zeros((4, 3, 6), dtype=pt.complex64)
    coefficients[:, 0] = 3.0 + 4.0j
    coefficients[:, 1] = 12.0j
    coefficients[:, 2] = 2.0
    return time, frequency, coefficients


def test_returns_modifiable_objects_and_single_mode_amplitude():
    time, frequency, coefficients = _coefficient_data()

    figure, axes = plot_spod_time_coefficients(
        time, frequency, coefficients, color_percentile=100.0
    )

    assert isinstance(figure, Figure)
    assert isinstance(axes, Axes)
    assert axes.get_xlabel() == r"$t\;[\mathrm{s}]$"
    assert axes.get_ylabel() == r"$f\;[\mathrm{Hz}]$"
    assert np.allclose(axes.collections[0].get_array(), 5.0)
    assert figure.axes[1].get_ylabel() == r"$|a_1(f,t)|$"
    plt.close(figure)


def test_multiple_modes_use_l2_amplitude_and_label():
    time, frequency, coefficients = _coefficient_data()

    figure, axes = plot_spod_time_coefficients(
        time,
        frequency,
        coefficients,
        n_modes=2,
        color_percentile=100.0,
    )

    assert np.allclose(axes.collections[0].get_array(), 13.0)
    assert figure.axes[1].get_ylabel() == (
        r"$\left(\sum_{i=1}^{2}|a_i(f,t)|^2\right)^{1/2}$"
    )
    plt.close(figure)


def test_reference_timescale_transforms_both_axes():
    time, frequency, coefficients = _coefficient_data()

    figure, axes = plot_spod_time_coefficients(
        time,
        frequency,
        coefficients,
        reference_timescale=0.25,
        colorbar=False,
    )

    assert axes.get_xlabel() == r"$t/\tau$"
    assert axes.get_ylabel() == r"$St$"
    assert axes.get_xlim() == pytest.approx((0.0, 2.0))
    assert axes.get_ylim() == pytest.approx((0.5, 1.5))
    assert len(figure.axes) == 1
    plt.close(figure)


def test_limits_select_input_coordinates_before_scaling():
    time, frequency, coefficients = _coefficient_data()

    figure, axes = plot_spod_time_coefficients(
        time,
        frequency,
        coefficients,
        reference_timescale=0.5,
        time_limits=(0.1, 0.4),
        frequency_limits=(3.0, 6.0),
    )

    assert axes.get_xlim() == pytest.approx((0.2, 0.8))
    assert axes.get_ylim() == pytest.approx((2.0, 3.0))
    assert axes.collections[0].get_array().shape == (2, 4)
    plt.close(figure)


def test_robust_color_limit_and_explicit_override():
    time, frequency, coefficients = _coefficient_data()
    coefficients[2, 0, 3] = 1.0e6
    mesh_kwargs = {"vmax": 20.0}

    figure, axes = plot_spod_time_coefficients(
        time,
        frequency,
        coefficients,
        color_percentile=90.0,
        mesh_kwargs=mesh_kwargs,
    )

    assert axes.collections[0].norm.vmin == 0.0
    assert axes.collections[0].norm.vmax == 20.0
    assert mesh_kwargs == {"vmax": 20.0}
    plt.close(figure)


def test_existing_axes_and_colorbar_options():
    time, frequency, coefficients = _coefficient_data()
    figure, supplied_axes = plt.subplots()
    colorbar_kwargs = {"label": r"$A(f,t)$"}

    returned_figure, returned_axes = plot_spod_time_coefficients(
        time,
        frequency,
        coefficients,
        ax=supplied_axes,
        colorbar_kwargs=colorbar_kwargs,
    )

    assert returned_figure is figure
    assert returned_axes is supplied_axes
    assert figure.axes[1].get_ylabel() == r"$A(f,t)$"
    assert colorbar_kwargs == {"label": r"$A(f,t)$"}
    plt.close(figure)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"n_modes": 0}, "n_modes"),
        ({"n_modes": 4}, "n_modes"),
        ({"reference_timescale": 0.0}, "reference_timescale"),
        ({"color_percentile": 0.0}, "color_percentile"),
        ({"time_limits": (0.4, 0.1)}, "time_limits"),
        ({"frequency_limits": (4.0, 3.0)}, "frequency_limits"),
    ],
)
def test_invalid_options(update, message):
    time, frequency, coefficients = _coefficient_data()
    with pytest.raises(ValueError, match=message):
        plot_spod_time_coefficients(time, frequency, coefficients, **update)


def test_invalid_shapes_and_empty_selections():
    time, frequency, coefficients = _coefficient_data()
    with pytest.raises(ValueError, match="shape"):
        plot_spod_time_coefficients(time, frequency, coefficients[:, 0])
    with pytest.raises(ValueError, match="frequency dimensions"):
        plot_spod_time_coefficients(time, frequency[:-1], coefficients)
    with pytest.raises(ValueError, match="frequency selection"):
        plot_spod_time_coefficients(
            time, frequency, coefficients, frequency_limits=(7.0, 8.0)
        )
    with pytest.raises(ValueError, match="time selection"):
        plot_spod_time_coefficients(
            time, frequency, coefficients, time_limits=(1.0, 2.0)
        )


def test_amspod_show_time_coefficients():
    data = pt.rand((8, 12))
    spod = AMSPOD(
        data,
        0.1,
        adaptive=False,
        max_tapers=3,
        keep_n_modes=2,
    )

    figure, axes = spod.show_time_coefficients(
        n_modes=2,
        reference_timescale=0.2,
        colorbar=False,
    )

    assert isinstance(figure, Figure)
    assert axes.get_xlabel() == r"$t/\tau$"
    assert axes.get_ylabel() == r"$St$"
    assert axes.collections[0].get_array().shape == (6, 12)
    plt.close(figure)
