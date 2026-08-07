import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch as pt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import QuadMesh
from matplotlib.image import AxesImage

from flowtorch.visualization import animate_scalar_field


def _field():
    x, y = pt.meshgrid(pt.arange(4.0), pt.arange(5.0), indexing="ij")
    return x, y, pt.stack((x + y, 2.0 * x - y), dim=-1)


def _finish(animation):
    animation._draw_was_started = True
    plt.close(animation._fig)


def test_animate_scalar_field_uses_image_without_coordinates():
    _, _, field = _field()

    animation = animate_scalar_field(field, colorbar=False)

    assert isinstance(animation, FuncAnimation)
    assert isinstance(animation._fig.axes[0].images[0], AxesImage)
    _finish(animation)


def test_animate_scalar_field_uses_mesh_with_coordinates():
    x, y, field = _field()

    animation = animate_scalar_field(field, x, y, colorbar=False)

    assert isinstance(animation._fig.axes[0].collections[0], QuadMesh)
    _finish(animation)


def test_animation_updates_scalar_frame():
    _, _, field = _field()
    animation = animate_scalar_field(field, colorbar=False)

    artists = animation._func(1)

    assert np.array_equal(artists[0].get_array(), field[:, :, 1].T.numpy())
    _finish(animation)


def test_animation_adds_and_updates_vector_arrows():
    _, _, field = _field()
    vector = pt.stack((field, -field), dim=-1)
    animation = animate_scalar_field(
        field,
        vector_field=vector,
        quiver_step=2,
        quiver_normalize=False,
        quiver_scale=1.0,
        colorbar=False,
    )

    artists = animation._func(1)

    assert len(artists) == 2
    assert np.allclose(artists[1].U, vector[::2, ::2, 1, 0].numpy().ravel())
    _finish(animation)


def test_animation_uses_robust_color_limits():
    field = pt.arange(100.0).reshape(10, 10, 1)
    field[-1, -1, 0] = 10000.0

    animation = animate_scalar_field(field, color_percentile=99.0, colorbar=False)
    image = animation._fig.axes[0].images[0]

    assert image.norm.vmax < 10000.0
    assert image.norm.vmin == pytest.approx(np.percentile(field.numpy(), 1.0))
    _finish(animation)


def test_animation_centers_color_limits_about_zero():
    _, _, field = _field()

    animation = animate_scalar_field(field, center_zero=True, colorbar=False)
    image = animation._fig.axes[0].images[0]

    assert image.norm.vmin == -image.norm.vmax
    assert image.get_cmap().name == "coolwarm"
    _finish(animation)


def test_animation_optionally_adds_colorbar_and_background():
    _, _, field = _field()

    animation = animate_scalar_field(
        field,
        colorbar=True,
        background="#112233",
        colorbar_kwargs={"label": "value"},
    )

    assert len(animation._fig.axes) == 2
    assert animation._fig.get_facecolor()[:3] == pytest.approx(
        matplotlib.colors.to_rgb("#112233")
    )
    assert animation._fig.axes[1].get_ylabel() == "value"
    _finish(animation)


def test_wide_image_uses_horizontal_colorbar_and_automatic_size():
    field = pt.ones((8, 2, 1))

    animation = animate_scalar_field(field)
    animation._fig.canvas.draw()
    width, height = animation._fig.get_size_inches()
    colorbar_axes = animation._fig.axes[1]

    assert width == pytest.approx(np.sqrt(72.0))
    assert height == pytest.approx(np.sqrt(8.0) + 0.6)
    assert colorbar_axes.get_position().width > colorbar_axes.get_position().height
    _finish(animation)


def test_tall_image_uses_vertical_colorbar():
    field = pt.ones((2, 8, 1))

    animation = animate_scalar_field(field)
    animation._fig.canvas.draw()
    colorbar_axes = animation._fig.axes[1]

    assert colorbar_axes.get_position().height > colorbar_axes.get_position().width
    _finish(animation)


def test_curvilinear_size_uses_coordinate_extents():
    x, y, field = _field()
    x = 100.0 * x

    animation = animate_scalar_field(field, x, y)
    width, height = animation._fig.get_size_inches()

    assert width == pytest.approx(np.sqrt(72.0))
    assert height == pytest.approx(np.sqrt(8.0) + 0.6)
    _finish(animation)


def test_explicit_figure_size_and_colorbar_orientation_override_automatic_values():
    field = pt.ones((8, 2, 1))

    animation = animate_scalar_field(
        field,
        figsize=(5.0, 7.0),
        colorbar_orientation="vertical",
    )
    animation._fig.canvas.draw()
    colorbar_axes = animation._fig.axes[1]

    assert animation._fig.get_size_inches() == pytest.approx((5.0, 7.0))
    assert colorbar_axes.get_position().height > colorbar_axes.get_position().width
    _finish(animation)


def test_layout_padding_depends_on_axes_visibility():
    _, _, field = _field()

    compact = animate_scalar_field(field, colorbar=False)
    labeled = animate_scalar_field(field, colorbar=False, show_axes=True)

    assert compact._fig.get_layout_engine().get()["w_pad"] == pytest.approx(0.1)
    assert labeled._fig.get_layout_engine().get()["w_pad"] == pytest.approx(0.4)
    _finish(compact)
    _finish(labeled)


def test_animation_does_not_mutate_keyword_dictionaries():
    _, _, field = _field()
    scalar_kwargs = {"interpolation": "nearest"}
    colorbar_kwargs = {"label": "field"}

    animation = animate_scalar_field(
        field,
        scalar_kwargs=scalar_kwargs,
        colorbar_kwargs=colorbar_kwargs,
    )

    assert scalar_kwargs == {"interpolation": "nearest"}
    assert colorbar_kwargs == {"label": "field"}
    _finish(animation)


def test_animation_rejects_one_coordinate():
    x, _, field = _field()
    with pytest.raises(ValueError):
        animate_scalar_field(field, x=x)


def test_animation_rejects_inconsistent_color_limits():
    _, _, field = _field()

    with pytest.raises(ValueError, match="color limits are inconsistent"):
        animate_scalar_field(field, vmin=100.0, colorbar=False)
