"""Matplotlib animations for scalar and vector field sequences."""

# standard library packages
from math import isfinite, sqrt
from typing import Any, Dict, Literal, Tuple

# third party packages
import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from matplotlib.animation import FuncAnimation

# local packages
from flowtorch.visualization.lic import line_integral_convolution


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
    figure, axes = _create_figure(
        data_ratio,
        colorbar,
        colorbar_orientation,
        figsize,
        layout,
        padding,
        show_axes,
        background,
        colorbar_options,
    )
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
            _update_mesh(scalar_artist, field_data[:, :, frame])
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


def animate_line_integral_convolution(
    vector_field: pt.Tensor,
    x: pt.Tensor,
    y: pt.Tensor,
    z: pt.Tensor | None = None,
    interval: int = 30,
    steps: int = 30,
    step_size: float = 0.5,
    texture: pt.Tensor | None = None,
    seed: int | None = None,
    normalize_lic: bool = True,
    show_magnitude: bool = False,
    cmap: Any = "viridis",
    lic_cmap: Any = "gray",
    lic_alpha: float | None = None,
    colorbar: bool = True,
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
    scalar_kwargs: Dict[str, Any] | None = None,
    lic_kwargs: Dict[str, Any] | None = None,
    colorbar_kwargs: Dict[str, Any] | None = None,
    blit: bool = True,
    repeat: bool = True,
) -> FuncAnimation:
    r"""Animate a vector-field sequence using line integral convolution.

    The LIC images are computed by :func:`line_integral_convolution` before
    rendering. Computation therefore remains on the input tensor's device,
    and only the completed image sequence is transferred to the CPU. The same
    input texture is used for every snapshot to avoid artificial flicker.

    With ``show_magnitude=True``, vector magnitude is displayed beneath the
    LIC texture. Its color limits use robust global percentiles across the
    sequence, as in :func:`animate_scalar_field`. The ``colorbar`` option only
    applies to this magnitude layer and is ignored when magnitude is hidden.

    For three-component fields, ``z`` contributes to both LIC trajectories
    and vector magnitude; the resulting LIC image is plotted in the ``x`` and
    ``y`` projection.

    :param vector_field: vector-field sequence with shape
        ``(nx, ny, n_snapshots, 2)`` or, with ``z``, a final dimension of 3
    :type vector_field: pt.Tensor
    :param x: first coordinate on the structured curvilinear grid
    :type x: pt.Tensor
    :param y: second coordinate on the structured curvilinear grid
    :type y: pt.Tensor
    :param z: optional third coordinate of an embedded surface
    :type z: pt.Tensor, optional
    :param interval: delay between frames in milliseconds, defaults to 30
    :type interval: int, optional
    :param steps: LIC steps in each streamline direction, defaults to 30
    :type steps: int, optional
    :param step_size: LIC step length in grid-index units, defaults to 0.5
    :type step_size: float, optional
    :param texture: optional LIC input texture shared across snapshots
    :type texture: pt.Tensor, optional
    :param seed: local random seed used to generate the texture
    :type seed: int, optional
    :param normalize_lic: normalize each LIC image to ``[0, 1]``
    :type normalize_lic: bool, optional
    :param show_magnitude: display vector magnitude below the LIC texture
    :type show_magnitude: bool, optional
    :param cmap: magnitude colormap, defaults to ``"viridis"``
    :param lic_cmap: LIC colormap, defaults to ``"gray"``
    :param lic_alpha: LIC opacity; defaults to ``0.45`` over magnitude and
        ``1.0`` otherwise
    :type lic_alpha: float, optional
    :param colorbar: display the magnitude colorbar, defaults to ``True``
    :type colorbar: bool, optional
    :param color_percentile: upper robust magnitude percentile in ``(50, 100]``
    :type color_percentile: float, optional
    :param vmin: explicit lower magnitude color limit
    :type vmin: float, optional
    :param vmax: explicit upper magnitude color limit
    :type vmax: float, optional
    :param background: figure and axes background color
    :type background: str, optional
    :param figsize: explicit figure size; inferred from coordinate extents when
        omitted
    :type figsize: Tuple[float, float], optional
    :param colorbar_orientation: automatic or explicit magnitude-colorbar
        orientation
    :type colorbar_orientation: {"auto", "vertical", "horizontal"}, optional
    :param layout: Matplotlib figure layout, defaults to ``"constrained"``
    :type layout: {"constrained", "compressed"}, optional
    :param padding: outer layout padding in inches
    :type padding: float, optional
    :param show_axes: show axes decorations, defaults to ``False``
    :type show_axes: bool, optional
    :param aspect: Matplotlib axes aspect, defaults to ``"equal"``
    :type aspect: str or float, optional
    :param scalar_kwargs: additional magnitude ``pcolormesh`` arguments
    :type scalar_kwargs: Dict[str, Any], optional
    :param lic_kwargs: additional LIC ``pcolormesh`` arguments
    :type lic_kwargs: Dict[str, Any], optional
    :param colorbar_kwargs: additional magnitude-colorbar arguments
    :type colorbar_kwargs: Dict[str, Any], optional
    :param blit: use Matplotlib blitting, defaults to ``True``
    :type blit: bool, optional
    :param repeat: repeat the animation, defaults to ``True``
    :type repeat: bool, optional
    :return: Matplotlib animation
    :rtype: matplotlib.animation.FuncAnimation

    **Example**

    .. code-block:: python

        animation = animate_line_integral_convolution(
            velocity,
            x,
            y,
            seed=0,
            show_magnitude=True,
            colorbar_kwargs={"label": r"$|\mathbf{u}|$"},
        )
    """
    if vector_field.ndim != 4:
        raise ValueError(
            "vector_field must have shape (nx, ny, n_snapshots, n_components)"
        )
    _validate_animation_options(
        interval,
        color_percentile,
        vmin,
        vmax,
        figsize,
        colorbar_orientation,
        layout,
        padding,
    )
    if lic_alpha is not None and (
        not isfinite(lic_alpha) or not 0.0 <= lic_alpha <= 1.0
    ):
        raise ValueError("lic_alpha must be finite and in the interval [0, 1]")

    lic = line_integral_convolution(
        vector_field,
        x,
        y,
        z,
        steps,
        step_size,
        texture,
        seed,
        normalize_lic,
    )
    lic_data = lic.detach().cpu().numpy()
    magnitude_data = (
        pt.linalg.vector_norm(vector_field, dim=-1).detach().cpu().numpy()
        if show_magnitude
        else None
    )
    scalar_options = {} if scalar_kwargs is None else scalar_kwargs.copy()
    lic_options = {} if lic_kwargs is None else lic_kwargs.copy()
    colorbar_options = {} if colorbar_kwargs is None else colorbar_kwargs.copy()
    lic_options.setdefault("shading", "auto")
    lic_options.setdefault("cmap", lic_cmap)
    lic_options.setdefault(
        "alpha", (0.45 if show_magnitude else 1.0) if lic_alpha is None else lic_alpha
    )
    if normalize_lic:
        lic_options.setdefault("vmin", 0.0)
        lic_options.setdefault("vmax", 1.0)
    else:
        lic_lower, lic_upper = _color_limits(lic_data, 100.0, None, None, False)
        lic_options.setdefault("vmin", lic_lower)
        lic_options.setdefault("vmax", lic_upper)

    coordinates = (x.detach().cpu().numpy(), y.detach().cpu().numpy())
    data_ratio = _data_aspect_ratio(x.shape, coordinates)
    effective_colorbar = colorbar and show_magnitude
    figure, axes = _create_figure(
        data_ratio,
        effective_colorbar,
        colorbar_orientation,
        figsize,
        layout,
        padding,
        show_axes,
        background,
        colorbar_options,
    )

    magnitude_artist = None
    if magnitude_data is not None:
        lower, upper = _color_limits(
            magnitude_data, color_percentile, vmin, vmax, False
        )
        scalar_options.setdefault("shading", "auto")
        scalar_options.update(cmap=cmap, vmin=lower, vmax=upper)
        magnitude_artist = axes.pcolormesh(
            coordinates[0], coordinates[1], magnitude_data[:, :, 0], **scalar_options
        )
    lic_artist = axes.pcolormesh(
        coordinates[0], coordinates[1], lic_data[:, :, 0], **lic_options
    )
    if effective_colorbar and magnitude_artist is not None:
        figure.colorbar(magnitude_artist, ax=axes, **colorbar_options)
    axes.set_aspect(aspect, adjustable="box")
    if not show_axes:
        axes.set_axis_off()
        axes.margins(0.0)
        if not effective_colorbar and layout is None:
            figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    def update(frame: int):
        _update_mesh(lic_artist, lic_data[:, :, frame])
        if magnitude_artist is None or magnitude_data is None:
            return (lic_artist,)
        _update_mesh(magnitude_artist, magnitude_data[:, :, frame])
        return magnitude_artist, lic_artist

    return FuncAnimation(
        figure,
        update,
        frames=vector_field.shape[2],
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


def _update_mesh(mesh: Any, frame: np.ndarray) -> None:
    mesh.set_array(frame if mesh.get_array().ndim == 2 else frame.ravel())


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


def _create_figure(
    data_ratio: float,
    colorbar: bool,
    colorbar_orientation: Literal["auto", "vertical", "horizontal"],
    figsize: tuple[float, float] | None,
    layout: Literal["constrained", "compressed"] | None,
    padding: float | None,
    show_axes: bool,
    background: str,
    colorbar_options: Dict[str, Any],
):
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
    return figure, axes


def _validate_animation_options(
    interval: int,
    color_percentile: float,
    vmin: float | None,
    vmax: float | None,
    figsize: Tuple[float, float] | None,
    colorbar_orientation: str,
    layout: str | None,
    padding: float | None,
) -> None:
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
    _validate_animation_options(
        interval,
        color_percentile,
        vmin,
        vmax,
        figsize,
        colorbar_orientation,
        layout,
        padding,
    )
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
    if not isinstance(quiver_step, int) or quiver_step < 1:
        raise ValueError("quiver_step must be a positive integer")
    if not isfinite(quiver_scale) or quiver_scale < 0.0:
        raise ValueError("quiver_scale must be finite and non-negative")
