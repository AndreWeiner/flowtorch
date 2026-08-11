import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch as pt
from matplotlib.collections import QuadMesh
from matplotlib.image import AxesImage

from flowtorch.visualization import plot_scalar_fields, plot_vector_fields


def _data():
    x, y = pt.meshgrid(pt.arange(4.0), pt.arange(5.0), indexing="ij")
    scalar = pt.stack((x + y, 10.0 * (x - y)), dim=-1)
    vector = pt.stack((scalar, -scalar), dim=-1)
    return x, y, scalar, vector


def test_scalar_grid_accepts_stack_and_shares_robust_limits():
    _, _, scalar, _ = _data()
    scalar[-1, -1, 1] = 10000.0

    figure, axes = plot_scalar_fields(
        scalar,
        [r"$p_1$", r"$p_2$"],
        2,
        2,
        color_percentile=95.0,
    )

    assert axes.shape == (2, 2)
    assert isinstance(axes[0, 0].images[0], AxesImage)
    assert axes[0, 0].images[0].get_clim() == axes[0, 1].images[0].get_clim()
    assert axes[0, 0].images[0].norm.vmax < 10000.0
    assert axes[0, 0].get_title() == r"$p_1$"
    assert axes[0, 0].get_aspect() == 1.0
    assert not axes[0, 0].axison
    assert not axes[1, 0].get_visible()
    assert len(figure.axes) == 5
    plt.close(figure)


def test_scalar_sequence_curvilinear_grid_and_quiver_overlay():
    x, y, scalar, vector = _data()

    figure, axes = plot_scalar_fields(
        list(scalar.unbind(dim=2)),
        ["one", "two"],
        1,
        2,
        x,
        y,
        vector_fields=list(vector.unbind(dim=2)),
        quiver_step=2,
        quiver_normalize=False,
        quiver_scale=1.0,
        colorbar=False,
    )

    assert isinstance(axes[0, 0].collections[0], QuadMesh)
    assert np.allclose(
        axes[0, 1].collections[1].U, vector[::2, ::2, 1, 0].numpy().ravel()
    )
    plt.close(figure)


def test_scalar_limits_support_explicit_bounds_and_constant_fields():
    field = pt.ones((3, 4))

    figure, axes = plot_scalar_fields(
        field, ["constant"], 1, 1, vmin=0.0, vmax=2.0, colorbar=False
    )
    assert axes[0, 0].images[0].get_clim() == (0.0, 2.0)
    plt.close(figure)

    figure, axes = plot_scalar_fields(field, ["constant"], 1, 1, colorbar=False)
    lower, upper = axes[0, 0].images[0].get_clim()
    assert lower < 1.0 < upper
    plt.close(figure)


def test_vector_quiver_accepts_stack_and_has_no_colorbar():
    _, _, _, vector = _data()

    figure, axes = plot_vector_fields(vector, ["one", "two"], 1, 2, colorbar=True)

    assert axes.shape == (1, 2)
    assert len(figure.axes) == 2
    assert len(axes[0, 0].collections) == 1
    plt.close(figure)


def test_vector_lic_uses_index_grid_and_shared_magnitude_colorbar():
    _, _, _, vector = _data()

    figure, axes = plot_vector_fields(
        vector,
        ["one", "two"],
        1,
        2,
        method="lic",
        show_magnitude=True,
        lic_steps=2,
        seed=0,
        colorbar_kwargs={"label": r"$|\mathbf{u}|$"},
    )

    assert len(figure.axes) == 3
    assert len(axes[0, 0].collections) == 2
    assert axes[0, 0].collections[0].get_clim() == axes[0, 1].collections[0].get_clim()
    assert figure.axes[-1].get_xlabel() == r"$|\mathbf{u}|$"
    plt.close(figure)


def test_wide_grid_uses_horizontal_shared_colorbar():
    field = pt.ones((4, 5, 2))

    figure, _ = plot_scalar_fields(field, ["one", "two"], 1, 2)
    figure.canvas.draw()
    colorbar_axes = figure.axes[-1]

    assert colorbar_axes.get_position().width > colorbar_axes.get_position().height
    plt.close(figure)


def test_grid_supports_shared_latex_axis_labels():
    figure, _ = plot_scalar_fields(
        pt.ones((3, 4, 2)),
        ["one", "two"],
        1,
        2,
        hide_axes=False,
        xlabel=r"$x/L$",
        ylabel=r"$y/L$",
        colorbar=False,
    )

    xlabel = next(text for text in figure.texts if text.get_text() == r"$x/L$")
    ylabel = next(text for text in figure.texts if text.get_text() == r"$y/L$")
    assert ylabel.get_fontsize() == xlabel.get_fontsize()
    assert ylabel.get_fontweight() == xlabel.get_fontweight()
    plt.close(figure)


def test_inputs_and_keyword_dictionaries_are_validated_and_preserved():
    scalar_options = {"interpolation": "nearest"}
    colorbar_options = {"shrink": 0.8}
    figure, _ = plot_scalar_fields(
        pt.ones((3, 4)),
        ["one"],
        1,
        1,
        scalar_kwargs=scalar_options,
        colorbar_kwargs=colorbar_options,
    )
    assert scalar_options == {"interpolation": "nearest"}
    assert colorbar_options == {"shrink": 0.8}
    plt.close(figure)

    with pytest.raises(ValueError, match="one nonempty string"):
        plot_scalar_fields(pt.ones((3, 4)), [""], 1, 1)
    with pytest.raises(ValueError, match="at least one finite"):
        plot_scalar_fields(pt.full((3, 4), float("nan")), ["one"], 1, 1)
    with pytest.raises(ValueError, match="same number"):
        plot_scalar_fields(
            pt.ones((3, 4, 2)),
            ["one", "two"],
            1,
            2,
            vector_fields=pt.ones((3, 4, 1, 2)),
        )
    with pytest.raises(ValueError, match="only available"):
        plot_vector_fields(pt.ones((3, 4, 2)), ["one"], 1, 1, show_magnitude=True)
