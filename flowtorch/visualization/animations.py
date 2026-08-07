"""Matplotlib animations for scalar and vector field sequences."""

# standard library packages
from math import isfinite, sqrt
from typing import Any, Dict, Literal, Tuple

# third party packages
import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from matplotlib.animation import FuncAnimation


def animate_scalar_field(
    field: pt.Tensor,
    x: pt.Tensor | None = None,
    y: pt.Tensor | None = None,
    vector_field: pt.Tensor | None = None,
    interval: int = 30,
    cmap: Any = None,
    colorbar: bool = True,
    center_zero: bool = False,
    color_percentile: float = 99.0,
    vmin: float | None = None,
    vmax: float | None = None,
    background: str = "white",
    figsize: Tuple[float, float] | None = None,
    colorbar_orientation: Literal["auto", "vertical", "horizontal"] = "auto",
    layout: Literal["constrained", "compressed"] | None = "constrained",
    padding: float | None = None,
    show_axes: bool = False,
    aspect: str | float = "equal",
    quiver_step: int = 10,
    quiver_normalize: bool = True,
    quiver_scale: float = 0.02,
    scalar_kwargs: Dict[str, Any] | None = None,
    quiver_kwargs: Dict[str, Any] | None = None,
    colorbar_kwargs: Dict[str, Any] | None = None,
    blit: bool = True,
    repeat: bool = True,
) -> FuncAnimation:
    """Animate a scalar field sequence with optional vector arrows.

    ``field`` must have shape ``(nx, ny, n_snapshots)``. If ``x`` and ``y``
    are omitted, frames are shown in array-index coordinates using
    :func:`matplotlib.pyplot.imshow`. If both coordinates are supplied, the
    field is drawn on the curvilinear grid using
    :func:`matplotlib.pyplot.pcolormesh`.

    Color limits are fixed across the complete animation. By default, the 1st
    and 99th percentiles of all finite values are used so isolated extremes do
    not compress the useful color range. Set ``color_percentile=100`` to use
    the exact range. With ``center_zero=True``, the selected percentile of the
    absolute field values defines symmetric limits about zero. Explicit
    ``vmin`` and ``vmax`` override their respective estimated limits.

    An optional ``vector_field`` has shape
    ``(nx, ny, n_snapshots, 2)``. Arrows are spatially downsampled, optionally
    normalized, and multiplied by ``quiver_scale`` before plotting.

    :param field: scalar field sequence
    :type field: pt.Tensor
    :param x: optional first coordinate on a curvilinear grid
    :type x: pt.Tensor, optional
    :param y: optional second coordinate on a curvilinear grid
    :type y: pt.Tensor, optional
    :param vector_field: optional two-component vector-field sequence
    :type vector_field: pt.Tensor, optional
    :param interval: delay between frames in milliseconds, defaults to 30
    :type interval: int, optional
    :param cmap: Matplotlib colormap name or object; defaults to ``"viridis"``
        or ``"coolwarm"`` when centered about zero
    :param colorbar: add a fixed scalar colorbar, defaults to ``True``
    :type colorbar: bool, optional
    :param center_zero: use symmetric color limits about zero
    :type center_zero: bool, optional
    :param color_percentile: upper robust percentile in ``(50, 100]``
    :type color_percentile: float, optional
    :param vmin: explicit lower color limit
    :type vmin: float, optional
    :param vmax: explicit upper color limit
    :type vmax: float, optional
    :param background: figure and axes background color
    :type background: str, optional
    :param figsize: figure size in inches; inferred from the image shape or
        coordinate extents when omitted
    :type figsize: Tuple[float, float], optional
    :param colorbar_orientation: colorbar orientation; ``"auto"`` places a
        horizontal colorbar below wide data and a vertical one beside other
        data, defaults to ``"auto"``
    :type colorbar_orientation: {"auto", "vertical", "horizontal"}, optional
    :param layout: Matplotlib figure layout, defaults to ``"constrained"``;
        use ``"compressed"`` for a still tighter fixed-aspect layout or
        ``None`` to disable automatic layout
    :type layout: {"constrained", "compressed"}, optional
    :param padding: outer layout padding in inches; defaults to ``0.1`` with
        hidden axes and ``0.4`` with visible axes
    :type padding: float, optional
    :param show_axes: show axes decorations, defaults to ``False``
    :type show_axes: bool, optional
    :param aspect: Matplotlib axes aspect, defaults to ``"equal"``
    :type aspect: str or float, optional
    :param quiver_step: plot every nth arrow along both grid axes
    :type quiver_step: int, optional
    :param quiver_normalize: normalize nonzero arrows before scaling
    :type quiver_normalize: bool, optional
    :param quiver_scale: multiplier applied to plotted arrow components
    :type quiver_scale: float, optional
    :param scalar_kwargs: additional ``imshow`` or ``pcolormesh`` arguments
    :type scalar_kwargs: Dict[str, Any], optional
    :param quiver_kwargs: additional ``Axes.quiver`` arguments
    :type quiver_kwargs: Dict[str, Any], optional
    :param colorbar_kwargs: additional ``Figure.colorbar`` arguments
    :type colorbar_kwargs: Dict[str, Any], optional
    :param blit: use Matplotlib blitting, defaults to ``True``
    :type blit: bool, optional
    :param repeat: repeat the animation, defaults to ``True``
    :type repeat: bool, optional
    :return: Matplotlib animation
    :rtype: matplotlib.animation.FuncAnimation

    **Examples**

    .. code-block:: python

        animation = animate_scalar_field(
            pressure,
            x,
            y,
            cmap="coolwarm",
            center_zero=True,
            colorbar_kwargs={"label": "pressure coefficient"},
        )

        animation = animate_scalar_field(
            magnitude,
            vector_field=velocity,
            color_percentile=99.5,
            quiver_step=8,
        )
    """
    _validate_inputs(
        field,
        x,
        y,
        vector_field,
        interval,
        color_percentile,
        vmin,
        vmax,
        figsize,
        colorbar_orientation,
        layout,
        padding,
        quiver_step,
        quiver_scale,
    )
    scalar_options = {} if scalar_kwargs is None else scalar_kwargs.copy()
    quiver_options = {} if quiver_kwargs is None else quiver_kwargs.copy()
    colorbar_options = {} if colorbar_kwargs is None else colorbar_kwargs.copy()
    field_data = field.detach().cpu().numpy()
    lower, upper = _color_limits(field_data, color_percentile, vmin, vmax, center_zero)
    selected_cmap = (
        cmap if cmap is not None else ("coolwarm" if center_zero else "viridis")
    )
    scalar_options.update(cmap=selected_cmap, vmin=lower, vmax=upper)

    coordinates = None
    if x is not None and y is not None:
        coordinates = (x.detach().cpu().numpy(), y.detach().cpu().numpy())
    vectors = None
    if vector_field is not None:
        vectors = _prepare_vectors(
            vector_field.detach().cpu().numpy(),
            quiver_step,
            quiver_normalize,
            quiver_scale,
        )

    data_ratio = _data_aspect_ratio(field.shape[:2], coordinates)
    orientation = _colorbar_orientation(data_ratio, colorbar_orientation)
    colorbar_options["orientation"] = orientation
    colorbar_options.setdefault("pad", _default_colorbar_pad(orientation, show_axes))
    selected_figsize = (
        _automatic_figure_size(data_ratio, colorbar, orientation)
        if figsize is None
        else figsize
    )
    figure, axes = plt.subplots(figsize=selected_figsize, layout=layout)
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None and hasattr(layout_engine, "set"):
        outer_padding = (0.4 if show_axes else 0.1) if padding is None else padding
        layout_engine.set(w_pad=outer_padding, h_pad=outer_padding)
    figure.patch.set_facecolor(background)
    axes.set_facecolor(background)
    if coordinates is None:
        scalar_options.setdefault("origin", "lower")
        scalar_artist = axes.imshow(field_data[:, :, 0].T, **scalar_options)
    else:
        scalar_options.setdefault("shading", "auto")
        scalar_artist = axes.pcolormesh(
            coordinates[0], coordinates[1], field_data[:, :, 0], **scalar_options
        )

    quiver_artist = None
    if vectors is not None:
        arrow_x, arrow_y = _arrow_coordinates(coordinates, field.shape[:2], quiver_step)
        quiver_options.setdefault("angles", "xy")
        quiver_options.setdefault("scale_units", "xy")
        quiver_options.setdefault("scale", 1.0)
        quiver_artist = axes.quiver(
            arrow_x,
            arrow_y,
            vectors[:, :, 0, 0],
            vectors[:, :, 0, 1],
            **quiver_options,
        )

    if colorbar:
        figure.colorbar(scalar_artist, ax=axes, **colorbar_options)
    axes.set_aspect(aspect, adjustable="box")
    if not show_axes:
        axes.set_axis_off()
        axes.margins(0.0)
        if not colorbar and layout is None:
            figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    def update(frame: int):
        if coordinates is None:
            scalar_artist.set_data(field_data[:, :, frame].T)
        else:
            scalar_artist.set_array(field_data[:, :, frame].ravel())
        if quiver_artist is None or vectors is None:
            return (scalar_artist,)
        quiver_artist.set_UVC(vectors[:, :, frame, 0], vectors[:, :, frame, 1])
        return scalar_artist, quiver_artist

    return FuncAnimation(
        figure,
        update,
        frames=field.shape[-1],
        interval=interval,
        blit=blit,
        repeat=repeat,
    )


def _color_limits(
    field: np.ndarray,
    percentile: float,
    vmin: float | None,
    vmax: float | None,
    center_zero: bool,
) -> tuple[float, float]:
    finite = field[np.isfinite(field)]
    if finite.size == 0:
        raise ValueError("field must contain at least one finite value")
    lower_estimate = float(np.percentile(finite, 100.0 - percentile))
    upper_estimate = float(np.percentile(finite, percentile))
    lower = lower_estimate if vmin is None else vmin
    upper = upper_estimate if vmax is None else vmax
    if vmin is not None and vmax is not None and vmin >= vmax:
        raise ValueError("vmin must be smaller than vmax")
    if center_zero:
        if vmin is None and vmax is None:
            limit = float(np.percentile(np.abs(finite), percentile))
        else:
            limit = max(abs(lower), abs(upper))
        lower, upper = -limit, limit
    if lower > upper:
        raise ValueError("estimated and explicit color limits are inconsistent")
    if lower == upper:
        epsilon = np.finfo(field.dtype).eps
        delta = max(abs(lower) * epsilon**0.5, epsilon)
        lower, upper = lower - delta, upper + delta
    return lower, upper


def _prepare_vectors(
    vector_field: np.ndarray,
    step: int,
    normalize: bool,
    scale: float,
) -> np.ndarray:
    vectors = vector_field[::step, ::step].astype(float, copy=True)
    if normalize:
        magnitude = np.linalg.norm(vectors, axis=-1)
        valid = np.isfinite(magnitude) & (magnitude > 0.0)
        vectors[valid] /= magnitude[valid, None]
        vectors[~np.isfinite(vectors)] = 0.0
    return vectors * scale


def _arrow_coordinates(
    coordinates: tuple[np.ndarray, np.ndarray] | None,
    shape: pt.Size,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    if coordinates is not None:
        return coordinates[0][::step, ::step], coordinates[1][::step, ::step]
    index_i, index_j = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), indexing="ij"
    )
    return index_i[::step, ::step], index_j[::step, ::step]


def _data_aspect_ratio(
    shape: pt.Size,
    coordinates: tuple[np.ndarray, np.ndarray] | None,
) -> float:
    if coordinates is None:
        return float(shape[0] / shape[1])
    x_extent = float(np.ptp(coordinates[0]))
    y_extent = float(np.ptp(coordinates[1]))
    if x_extent <= 0.0 or y_extent <= 0.0:
        return 1.5
    return x_extent / y_extent


def _colorbar_orientation(
    data_ratio: float,
    orientation: Literal["auto", "vertical", "horizontal"],
) -> Literal["vertical", "horizontal"]:
    if orientation == "auto":
        return "horizontal" if data_ratio >= 1.25 else "vertical"
    return orientation


def _automatic_figure_size(
    data_ratio: float,
    colorbar: bool,
    orientation: str,
) -> tuple[float, float]:
    ratio = min(max(data_ratio, 1.0 / 3.0), 3.0)
    width = sqrt(24.0 * ratio)
    height = sqrt(24.0 / ratio)
    if colorbar and orientation == "vertical":
        width += 0.8
    elif colorbar and orientation == "horizontal":
        height += 0.6
    return width, height


def _default_colorbar_pad(orientation: str, show_axes: bool) -> float:
    if orientation == "horizontal":
        return 0.08 if show_axes else 0.04
    return 0.03 if show_axes else 0.025


def _validate_inputs(
    field: pt.Tensor,
    x: pt.Tensor | None,
    y: pt.Tensor | None,
    vector_field: pt.Tensor | None,
    interval: int,
    color_percentile: float,
    vmin: float | None,
    vmax: float | None,
    figsize: Tuple[float, float] | None,
    colorbar_orientation: str,
    layout: str | None,
    padding: float | None,
    quiver_step: int,
    quiver_scale: float,
) -> None:
    if field.ndim != 3 or min(field.shape) < 1:
        raise ValueError("field must have shape (nx, ny, n_snapshots)")
    if not field.is_floating_point():
        raise ValueError("field must have a floating-point dtype")
    if (x is None) != (y is None):
        raise ValueError("x and y must either both be supplied or both be omitted")
    if x is not None and y is not None:
        if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
            raise ValueError("x and y must be identically shaped 2D tensors")
        if x.shape != field.shape[:2]:
            raise ValueError("coordinate and field spatial shapes must match")
        if not pt.isfinite(x).all() or not pt.isfinite(y).all():
            raise ValueError("coordinates must contain only finite values")
    if vector_field is not None:
        expected_shape = (*field.shape, 2)
        if vector_field.shape != expected_shape:
            raise ValueError("vector_field must have shape (nx, ny, n_snapshots, 2)")
        if not vector_field.is_floating_point():
            raise ValueError("vector_field must have a floating-point dtype")
    if not isinstance(interval, int) or interval < 1:
        raise ValueError("interval must be a positive integer")
    if not isfinite(color_percentile) or not 50.0 < color_percentile <= 100.0:
        raise ValueError("color_percentile must be in the interval (50, 100]")
    if vmin is not None and not isfinite(vmin):
        raise ValueError("vmin must be finite")
    if vmax is not None and not isfinite(vmax):
        raise ValueError("vmax must be finite")
    if figsize is not None and (len(figsize) != 2 or min(figsize) <= 0.0):
        raise ValueError("figsize must contain two positive values")
    if colorbar_orientation not in {"auto", "vertical", "horizontal"}:
        raise ValueError(
            "colorbar_orientation must be 'auto', 'vertical', or 'horizontal'"
        )
    if layout not in {"constrained", "compressed", None}:
        raise ValueError("layout must be 'constrained', 'compressed', or None")
    if padding is not None and (not isfinite(padding) or padding < 0.0):
        raise ValueError("padding must be finite and non-negative")
    if not isinstance(quiver_step, int) or quiver_step < 1:
        raise ValueError("quiver_step must be a positive integer")
    if not isfinite(quiver_scale) or quiver_scale < 0.0:
        raise ValueError("quiver_scale must be finite and non-negative")
