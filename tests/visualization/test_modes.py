import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch as pt
from matplotlib.collections import QuadMesh
from matplotlib.image import AxesImage

from flowtorch.visualization import plot_spod_mode_2d, plot_spod_modes_2d


def _mode(n_components=2):
    base = pt.arange(20.0).reshape(4, 5) + 1.0
    components = [
        base * pt.exp(1j * pt.tensor(0.4 + index)) for index in range(n_components)
    ]
    return pt.cat([component.reshape(-1) for component in components])


def test_single_mode_plots_all_components_and_quantities():
    mode = _mode()

    figure, axes = plot_spod_mode_2d(mode, shape=(4, 5), colorbar=False)

    assert axes.shape == (2, 4)
    assert all(isinstance(axes.flat[index].images[0], AxesImage) for index in range(8))
    assert axes[0, 0].get_aspect() == 1.0
    assert axes[0, 0].get_title() == r"$\operatorname{Re}(\phi)$"
    assert axes[0, 3].images[0].get_cmap().name == "twilight"
    assert axes[0, 3].images[0].get_clim() == pytest.approx((-np.pi, np.pi))
    assert not axes[0, 0].axison
    plt.close(figure)


def test_single_mode_can_show_axes():
    figure, axes = plot_spod_mode_2d(
        _mode(1),
        shape=(4, 5),
        hide_axes=False,
        colorbar=False,
    )

    assert axes[0, 0].axison
    plt.close(figure)


def test_single_mode_uses_individual_component_limits_and_colorbars():
    first = pt.ones((4, 5), dtype=pt.complex64)
    second = 10.0 * first
    mode = pt.cat((first.ravel(), second.ravel()))

    figure, axes = plot_spod_mode_2d(
        mode,
        shape=(4, 5),
        show_imaginary=False,
        show_absolute=False,
        show_phase=False,
    )

    assert axes[0, 0].images[0].get_clim() == pytest.approx((-1.0, 1.0))
    assert axes[1, 0].images[0].get_clim() == pytest.approx((-10.0, 10.0))
    assert len(figure.axes) == 4
    plt.close(figure)


def test_single_mode_places_colorbar_below_wide_data():
    figure, _ = plot_spod_mode_2d(
        pt.ones(16),
        shape=(8, 2),
        show_imaginary=False,
        show_absolute=False,
        show_phase=False,
    )
    figure.canvas.draw()
    colorbar_axes = figure.axes[1]

    assert colorbar_axes.get_position().width > colorbar_axes.get_position().height
    plt.close(figure)


def test_single_mode_places_colorbar_beside_tall_data():
    figure, _ = plot_spod_mode_2d(
        pt.ones(16),
        shape=(2, 8),
        show_imaginary=False,
        show_absolute=False,
        show_phase=False,
    )
    figure.canvas.draw()
    colorbar_axes = figure.axes[1]

    assert colorbar_axes.get_position().height > colorbar_axes.get_position().width
    plt.close(figure)


def test_component_selection_and_labels():
    mode = _mode(3)

    figure, axes = plot_spod_mode_2d(
        mode,
        shape=(4, 5),
        components=[2, 0],
        component_labels=[r"$w$", r"$u$"],
        show_imaginary=False,
        show_absolute=False,
        show_phase=False,
        colorbar=False,
    )

    assert axes.shape == (2, 1)
    assert axes[0, 0].get_ylabel() == r"$w$"
    assert axes[1, 0].get_ylabel() == r"$u$"
    plt.close(figure)


def test_curvilinear_mode_uses_quad_mesh():
    x, y = pt.meshgrid(pt.arange(4.0), pt.arange(5.0), indexing="ij")

    figure, axes = plot_spod_mode_2d(
        _mode(1),
        x=x,
        y=y,
        show_imaginary=False,
        show_absolute=False,
        show_phase=False,
        colorbar=False,
    )

    assert isinstance(axes[0, 0].collections[0], QuadMesh)
    plt.close(figure)


def test_phase_alignment_rotates_all_components_from_reference_component():
    first = pt.full((4, 5), 1j, dtype=pt.complex64)
    second = pt.full((4, 5), -2.0 + 0j, dtype=pt.complex64)
    mode = pt.cat((first.ravel(), second.ravel()))

    figure, axes = plot_spod_mode_2d(
        mode,
        shape=(4, 5),
        reference_index=(0, 0),
        reference_component=0,
        show_real=False,
        show_absolute=False,
        show_phase=False,
        colorbar=False,
    )

    assert np.allclose(axes[0, 0].images[0].get_array(), 0.0, atol=1.0e-6)
    assert np.allclose(axes[1, 0].images[0].get_array(), 2.0, atol=1.0e-6)
    plt.close(figure)


def test_reference_point_uses_nearest_curvilinear_node():
    x, y = pt.meshgrid(pt.arange(4.0), pt.arange(5.0), indexing="ij")
    phase = 0.2 * x + 0.3 * y
    mode = pt.exp(1j * phase).reshape(-1)

    figure, axes = plot_spod_mode_2d(
        mode,
        x=x,
        y=y,
        reference_point=(0.1, 0.1),
        show_real=False,
        show_imaginary=False,
        show_absolute=False,
        colorbar=False,
    )

    assert isinstance(axes[0, 0].collections[0], QuadMesh)
    plt.close(figure)


def test_comparison_shares_limits_and_one_colorbar_per_quantity():
    first = _mode(1)
    second = 5.0 * first

    figure, axes = plot_spod_modes_2d(
        [first, second],
        [r"$St=0.1$", r"$St=0.2$"],
        n_rows=2,
        n_cols=2,
        shape=(4, 5),
        show_imaginary=False,
        show_phase=False,
    )

    assert axes.shape == (2, 2, 2)
    assert axes[0, 0, 0].images[0].get_clim() == axes[0, 0, 1].images[0].get_clim()
    assert axes[0, 0, 0].get_title() == r"$St=0.1$"
    assert not axes[0, 1, 0].get_visible()
    assert not axes[1, 1, 1].get_visible()
    assert not axes[0, 0, 0].axison
    assert len(figure.axes) == 10
    plt.close(figure)


def test_comparison_defaults_to_first_component():
    mode = _mode(2)

    figure, axes = plot_spod_modes_2d(
        [mode],
        ["mode"],
        n_rows=1,
        n_cols=1,
        shape=(4, 5),
        show_imaginary=False,
        show_absolute=False,
        show_phase=False,
        colorbar=False,
    )

    expected = mode[:20].reshape(4, 5).real.T
    assert np.allclose(axes[0, 0, 0].images[0].get_array(), expected)
    plt.close(figure)


def test_comparison_requires_nonempty_title_for_every_mode():
    with pytest.raises(ValueError, match="nonempty string"):
        plot_spod_modes_2d(
            [_mode(1)],
            [""],
            n_rows=1,
            n_cols=1,
            shape=(4, 5),
        )


def test_automatic_figure_size_is_capped():
    figure, _ = plot_spod_mode_2d(_mode(5), shape=(4, 5), colorbar=False)

    assert max(figure.get_size_inches()) <= 20.0
    plt.close(figure)


def test_automatic_size_cap_preserves_figure_ratio():
    mode = _mode(1)
    figure, _ = plot_spod_modes_2d(
        [mode] * 4,
        ["one", "two", "three", "four"],
        n_rows=1,
        n_cols=4,
        shape=(4, 5),
    )

    width, height = figure.get_size_inches()
    panel_width = np.sqrt(24.0 * (4.0 / 5.0))
    panel_height = np.sqrt(24.0 / (4.0 / 5.0))
    panel_scale = 3.0 / max(panel_width, panel_height)
    uncapped_width = panel_width * panel_scale * 4 + 0.8
    uncapped_height = panel_height * panel_scale * 4
    assert width / height == pytest.approx(uncapped_width / uncapped_height)
    plt.close(figure)


def test_single_mode_automatic_layout_is_compact_for_wide_panels():
    figure, _ = plot_spod_mode_2d(
        pt.ones(64, dtype=pt.complex64),
        shape=(8, 4),
    )

    width, _ = figure.get_size_inches()
    assert width < 14.0
    plt.close(figure)


def test_plot_keyword_dictionaries_are_not_mutated():
    plot_kwargs = {"real": {"interpolation": "nearest"}}
    colorbar_kwargs = {"shrink": 0.8}

    figure, _ = plot_spod_mode_2d(
        _mode(1),
        shape=(4, 5),
        plot_kwargs=plot_kwargs,
        colorbar_kwargs=colorbar_kwargs,
    )

    assert plot_kwargs == {"real": {"interpolation": "nearest"}}
    assert colorbar_kwargs == {"shrink": 0.8}
    plt.close(figure)


def test_requires_at_least_one_quantity():
    with pytest.raises(ValueError, match="at least one mode quantity"):
        plot_spod_mode_2d(
            _mode(1),
            shape=(4, 5),
            show_real=False,
            show_imaginary=False,
            show_absolute=False,
            show_phase=False,
        )


def test_rejects_unresolvable_reference_phase():
    with pytest.raises(ValueError, match="no resolvable mean phase"):
        plot_spod_mode_2d(
            pt.zeros(20, dtype=pt.complex64),
            shape=(4, 5),
            reference_index=(0, 0),
        )
