"""Comparative plots for two-dimensional scalar and vector fields."""

from collections.abc import Sequence
from math import isfinite, sqrt
from typing import Any, Dict, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from matplotlib.figure import Figure

from flowtorch.visualization.lic import line_integral_convolution


def plot_scalar_fields(
    fields: Sequence[pt.Tensor] | pt.Tensor,
    titles: Sequence[str],
    n_rows: int,
    n_cols: int,
    x: pt.Tensor | None = None,
    y: pt.Tensor | None = None,
    vector_fields: Sequence[pt.Tensor] | pt.Tensor | None = None,
    cmap: Any = None,
    colorbar: bool = True,
    center_zero: bool = False,
    color_percentile: float = 99.0,
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] | None = None,
    colorbar_orientation: Literal["auto", "vertical", "horizontal"] = "auto",
    aspect: str | float = "equal",
    hide_axes: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    quiver_step: int = 10,
    quiver_normalize: bool = True,
    quiver_scale: float = 0.02,
    scalar_kwargs: Dict[str, Any] | None = None,
    quiver_kwargs: Dict[str, Any] | None = None,
    colorbar_kwargs: Dict[str, Any] | None = None,
) -> tuple[Figure, np.ndarray]:
    r"""Compare scalar fields, optionally overlaying vector-field arrows.

    A stacked scalar tensor has shape ``(nx, ny, n_fields)``; alternatively,
    a sequence of ``(nx, ny)`` tensors may be supplied. A vector overlay has
    shape ``(nx, ny, n_fields, 2)`` or is a sequence of ``(nx, ny, 2)``
    tensors. Color limits and the optional colorbar are shared by every panel.
    Titles and colorbar labels are passed unchanged to Matplotlib, so mathtext
    or LaTeX strings may be used.

    :return: modifiable figure and axes with shape ``(n_rows, n_cols)``

    **Examples**

    Compare three fields in one row with hidden axes and a shared colorbar:

    .. code-block:: python

        figure, axes = plot_scalar_fields(
            fields,
            [r"$q_1$", r"$q_2$", r"$q_3$"],
            n_rows=1,
            n_cols=3,
            center_zero=True,
            colorbar_kwargs={"label": r"$q/q_\infty$"},
        )

    Show a curvilinear 2-by-2 comparison with shared coordinate labels and no
    colorbar:

    .. code-block:: python

        figure, axes = plot_scalar_fields(
            fields,
            ["Case A", "Case B", "Case C", "Case D"],
            n_rows=2,
            n_cols=2,
            x=x,
            y=y,
            colorbar=False,
            hide_axes=False,
            xlabel=r"$x/L$",
            ylabel=r"$y/L$",
        )

    Overlay one vector field per scalar field:

    .. code-block:: python

        figure, axes = plot_scalar_fields(
            pressure,
            titles,
            n_rows=2,
            n_cols=2,
            x=x,
            y=y,
            vector_fields=velocity,
            quiver_step=8,
            quiver_scale=0.05,
        )
    """
    scalar_list = _scalar_sequence(fields)
    shape = _validate_collection(scalar_list, titles, n_rows, n_cols, "scalar")
    coordinates = _coordinates(x, y, shape)
    vectors = None
    if vector_fields is not None:
        vectors = _vector_sequence(vector_fields)
        _validate_vectors(vectors, len(scalar_list), shape)
    _validate_options(
        color_percentile,
        vmin,
        vmax,
        figsize,
        colorbar_orientation,
        quiver_step,
        quiver_scale,
    )

    scalar_data = [field.detach().cpu().numpy() for field in scalar_list]
    lower, upper = _color_limits(
        np.stack(scalar_data), color_percentile, vmin, vmax, center_zero
    )
    scalar_options = {} if scalar_kwargs is None else scalar_kwargs.copy()
    scalar_options.update(
        cmap=cmap if cmap is not None else ("coolwarm" if center_zero else "viridis"),
        vmin=lower,
        vmax=upper,
    )
    quiver_options = {} if quiver_kwargs is None else quiver_kwargs.copy()
    colorbar_options = {} if colorbar_kwargs is None else colorbar_kwargs.copy()
    figure, axes, orientation = _figure(
        shape,
        coordinates,
        n_rows,
        n_cols,
        colorbar,
        figsize,
        colorbar_orientation,
        colorbar_options,
        hide_axes,
    )

    artists = []
    flat_axes = axes.ravel()
    for index, data in enumerate(scalar_data):
        current_axes = flat_axes[index]
        if coordinates is None:
            options = scalar_options.copy()
            options.setdefault("origin", "lower")
            artist = current_axes.imshow(data.T, **options)
        else:
            options = scalar_options.copy()
            options.setdefault("shading", "auto")
            artist = current_axes.pcolormesh(*coordinates, data, **options)
        artists.append(artist)
        if vectors is not None:
            prepared = _prepare_vector(
                vectors[index], quiver_step, quiver_normalize, quiver_scale
            )
            arrow_x, arrow_y = _arrow_coordinates(coordinates, shape, quiver_step)
            options = quiver_options.copy()
            options.setdefault("angles", "xy")
            options.setdefault("scale_units", "xy")
            options.setdefault("scale", 1.0)
            current_axes.quiver(
                arrow_x, arrow_y, prepared[..., 0], prepared[..., 1], **options
            )
        _finish_axes(current_axes, titles[index], aspect, hide_axes)
    _hide_unused(flat_axes, len(scalar_data))
    if colorbar:
        colorbar_options.setdefault("orientation", orientation)
        colorbar_options.setdefault(
            "pad",
            (
                (0.14 if xlabel is not None else 0.04)
                if orientation == "horizontal"
                else 0.025
            ),
        )
        colorbar_options.setdefault("shrink", 0.8)
        colorbar_options.setdefault("aspect", 40 if orientation == "horizontal" else 25)
        figure.colorbar(
            artists[0], ax=list(flat_axes[: len(artists)]), **colorbar_options
        )
    _set_shared_labels(figure, flat_axes[: len(artists)], xlabel, ylabel)
    return figure, axes


def plot_vector_fields(
    fields: Sequence[pt.Tensor] | pt.Tensor,
    titles: Sequence[str],
    n_rows: int,
    n_cols: int,
    x: pt.Tensor | None = None,
    y: pt.Tensor | None = None,
    method: Literal["quiver", "lic"] = "quiver",
    show_magnitude: bool = False,
    cmap: Any = "viridis",
    lic_cmap: Any = "gray",
    lic_alpha: float | None = None,
    colorbar: bool = True,
    color_percentile: float = 99.0,
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] | None = None,
    colorbar_orientation: Literal["auto", "vertical", "horizontal"] = "auto",
    aspect: str | float = "equal",
    hide_axes: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    quiver_step: int = 10,
    quiver_normalize: bool = True,
    quiver_scale: float = 0.02,
    lic_steps: int = 30,
    lic_step_size: float = 0.5,
    texture: pt.Tensor | None = None,
    seed: int | None = None,
    normalize_lic: bool = True,
    quiver_kwargs: Dict[str, Any] | None = None,
    magnitude_kwargs: Dict[str, Any] | None = None,
    lic_kwargs: Dict[str, Any] | None = None,
    colorbar_kwargs: Dict[str, Any] | None = None,
) -> tuple[Figure, np.ndarray]:
    r"""Compare vector fields as quiver or line-integral-convolution plots.

    A stacked tensor has shape ``(nx, ny, n_fields, 2)``; a sequence of
    ``(nx, ny, 2)`` tensors is also accepted. Plain LIC has no meaningful
    colorbar. With ``show_magnitude=True``, magnitude is drawn below LIC with
    shared limits and one shared colorbar.

    **Examples**

    Compare vector fields as downsampled arrows:

    .. code-block:: python

        figure, axes = plot_vector_fields(
            velocity,
            [r"$\mathbf{u}_1$", r"$\mathbf{u}_2$"],
            n_rows=1,
            n_cols=2,
            x=x,
            y=y,
            quiver_step=6,
            quiver_scale=0.08,
        )

    Display LIC over a shared vector-magnitude layer and colorbar:

    .. code-block:: python

        figure, axes = plot_vector_fields(
            velocity,
            titles,
            n_rows=2,
            n_cols=2,
            x=x,
            y=y,
            method="lic",
            show_magnitude=True,
            seed=0,
            colorbar_kwargs={"label": r"$|\mathbf{u}|$"},
        )
    """
    vectors = _vector_sequence(fields)
    if not vectors:
        raise ValueError("at least one vector field must be supplied")
    shape = tuple(vectors[0].shape[:2])
    _validate_collection(vectors, titles, n_rows, n_cols, "vector")
    _validate_vectors(vectors, len(vectors), shape)
    coordinates = _coordinates(x, y, shape)
    _validate_options(
        color_percentile,
        vmin,
        vmax,
        figsize,
        colorbar_orientation,
        quiver_step,
        quiver_scale,
    )
    if method not in {"quiver", "lic"}:
        raise ValueError("method must be 'quiver' or 'lic'")
    if show_magnitude and method != "lic":
        raise ValueError("show_magnitude is only available with method='lic'")
    if lic_alpha is not None and (not isfinite(lic_alpha) or not 0 <= lic_alpha <= 1):
        raise ValueError("lic_alpha must be finite and in the interval [0, 1]")

    effective_colorbar = colorbar and method == "lic" and show_magnitude
    colorbar_options = {} if colorbar_kwargs is None else colorbar_kwargs.copy()
    figure, axes, orientation = _figure(
        shape,
        coordinates,
        n_rows,
        n_cols,
        effective_colorbar,
        figsize,
        colorbar_orientation,
        colorbar_options,
        hide_axes,
    )
    flat_axes = axes.ravel()
    magnitude_artists = []
    magnitudes = [pt.linalg.vector_norm(vector, dim=-1) for vector in vectors]
    magnitude_limits = None
    if show_magnitude:
        magnitude_limits = _color_limits(
            np.stack([value.detach().cpu().numpy() for value in magnitudes]),
            color_percentile,
            vmin,
            vmax,
            False,
        )
    plot_coordinates = coordinates or _index_coordinates(shape)

    for index, vector in enumerate(vectors):
        current_axes = flat_axes[index]
        if method == "quiver":
            prepared = _prepare_vector(
                vector, quiver_step, quiver_normalize, quiver_scale
            )
            arrow_x, arrow_y = _arrow_coordinates(coordinates, shape, quiver_step)
            options = {} if quiver_kwargs is None else quiver_kwargs.copy()
            options.setdefault("angles", "xy")
            options.setdefault("scale_units", "xy")
            options.setdefault("scale", 1.0)
            current_axes.quiver(
                arrow_x, arrow_y, prepared[..., 0], prepared[..., 1], **options
            )
        else:
            if show_magnitude and magnitude_limits is not None:
                options = {} if magnitude_kwargs is None else magnitude_kwargs.copy()
                options.setdefault("shading", "auto")
                options.update(
                    cmap=cmap, vmin=magnitude_limits[0], vmax=magnitude_limits[1]
                )
                magnitude_artists.append(
                    current_axes.pcolormesh(
                        *plot_coordinates,
                        magnitudes[index].detach().cpu().numpy(),
                        **options,
                    )
                )
            lic = line_integral_convolution(
                vector,
                pt.as_tensor(
                    plot_coordinates[0], dtype=vector.dtype, device=vector.device
                ),
                pt.as_tensor(
                    plot_coordinates[1], dtype=vector.dtype, device=vector.device
                ),
                steps=lic_steps,
                step_size=lic_step_size,
                texture=texture,
                seed=seed,
                normalize=normalize_lic,
            )
            options = {} if lic_kwargs is None else lic_kwargs.copy()
            options.setdefault("shading", "auto")
            options.setdefault("cmap", lic_cmap)
            options.setdefault(
                "alpha",
                (0.45 if show_magnitude else 1.0) if lic_alpha is None else lic_alpha,
            )
            if normalize_lic:
                options.setdefault("vmin", 0.0)
                options.setdefault("vmax", 1.0)
            current_axes.pcolormesh(
                *plot_coordinates, lic.detach().cpu().numpy(), **options
            )
        _finish_axes(current_axes, titles[index], aspect, hide_axes)
    _hide_unused(flat_axes, len(vectors))
    if effective_colorbar:
        colorbar_options.setdefault("orientation", orientation)
        colorbar_options.setdefault(
            "pad",
            (
                (0.14 if xlabel is not None else 0.04)
                if orientation == "horizontal"
                else 0.025
            ),
        )
        colorbar_options.setdefault("shrink", 0.8)
        colorbar_options.setdefault("aspect", 40 if orientation == "horizontal" else 25)
        figure.colorbar(
            magnitude_artists[0],
            ax=list(flat_axes[: len(vectors)]),
            **colorbar_options,
        )
    _set_shared_labels(figure, flat_axes[: len(vectors)], xlabel, ylabel)
    return figure, axes


def _scalar_sequence(fields: Sequence[pt.Tensor] | pt.Tensor) -> list[pt.Tensor]:
    if isinstance(fields, pt.Tensor):
        if fields.ndim == 2:
            return [fields]
        if fields.ndim == 3:
            return list(fields.unbind(dim=2))
        raise ValueError("fields must have shape (nx, ny) or (nx, ny, n_fields)")
    return list(fields)


def _vector_sequence(fields: Sequence[pt.Tensor] | pt.Tensor) -> list[pt.Tensor]:
    if isinstance(fields, pt.Tensor):
        if fields.ndim == 3:
            return [fields]
        if fields.ndim == 4:
            return list(fields.unbind(dim=2))
        raise ValueError(
            "vector fields must have shape (nx, ny, 2) or (nx, ny, n_fields, 2)"
        )
    return list(fields)


def _validate_collection(fields, titles, n_rows, n_cols, kind):
    if not fields:
        raise ValueError(f"at least one {kind} field must be supplied")
    if len(titles) != len(fields) or any(
        not isinstance(title, str) or not title for title in titles
    ):
        raise ValueError("titles must contain one nonempty string per field")
    if not isinstance(n_rows, int) or not isinstance(n_cols, int):
        raise ValueError("n_rows and n_cols must be integers")
    if n_rows < 1 or n_cols < 1 or n_rows * n_cols < len(fields):
        raise ValueError("the subplot grid must contain every field")
    expected_dimensions = 2 if kind == "scalar" else 3
    if any(field.ndim != expected_dimensions for field in fields):
        raise ValueError(
            f"each {kind} field must have {expected_dimensions} dimensions"
        )
    shape = tuple(fields[0].shape[:2])
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("fields must have two positive spatial dimensions")
    if any(tuple(field.shape[:2]) != shape for field in fields):
        raise ValueError("all fields must have the same spatial shape")
    if any(not field.is_floating_point() for field in fields):
        raise ValueError("fields must have a floating-point dtype")
    return shape


def _validate_vectors(vectors, expected_count, shape):
    if len(vectors) != expected_count:
        raise ValueError(
            "scalar and vector collections must contain the same number of fields"
        )
    if any(vector.ndim != 3 or vector.shape != (*shape, 2) for vector in vectors):
        raise ValueError("each vector field must have shape (nx, ny, 2)")
    if any(not vector.is_floating_point() for vector in vectors):
        raise ValueError("vector fields must have a floating-point dtype")


def _coordinates(x, y, shape):
    if (x is None) != (y is None):
        raise ValueError("x and y must either both be supplied or both be omitted")
    if x is None or y is None:
        return None
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape or tuple(x.shape) != shape:
        raise ValueError("x and y must match the two-dimensional field shape")
    if not pt.isfinite(x).all() or not pt.isfinite(y).all():
        raise ValueError("coordinates must contain only finite values")
    return x.detach().cpu().numpy(), y.detach().cpu().numpy()


def _validate_options(percentile, vmin, vmax, figsize, orientation, step, scale):
    if not isfinite(percentile) or not 50 < percentile <= 100:
        raise ValueError("color_percentile must be in the interval (50, 100]")
    if vmin is not None and not isfinite(vmin):
        raise ValueError("vmin must be finite")
    if vmax is not None and not isfinite(vmax):
        raise ValueError("vmax must be finite")
    if figsize is not None and (len(figsize) != 2 or min(figsize) <= 0):
        raise ValueError("figsize must contain two positive values")
    if orientation not in {"auto", "vertical", "horizontal"}:
        raise ValueError("invalid colorbar_orientation")
    if not isinstance(step, int) or step < 1:
        raise ValueError("quiver_step must be a positive integer")
    if not isfinite(scale) or scale < 0:
        raise ValueError("quiver_scale must be finite and non-negative")


def _color_limits(data, percentile, vmin, vmax, center_zero):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("fields must contain at least one finite value")
    lower = float(np.percentile(finite, 100 - percentile)) if vmin is None else vmin
    upper = float(np.percentile(finite, percentile)) if vmax is None else vmax
    if center_zero:
        if vmin is None and vmax is None:
            limit = float(np.percentile(np.abs(finite), percentile))
        else:
            limit = max(abs(lower), abs(upper))
        lower, upper = -limit, limit
    if lower > upper or (vmin is not None and vmax is not None and vmin >= vmax):
        raise ValueError("estimated and explicit color limits are inconsistent")
    if lower == upper:
        epsilon = np.finfo(data.dtype).eps
        delta = max(abs(lower) * sqrt(epsilon), epsilon)
        lower, upper = lower - delta, upper + delta
    return lower, upper


def _data_ratio(shape, coordinates):
    if coordinates is None:
        return shape[0] / shape[1]
    x_extent, y_extent = np.ptp(coordinates[0]), np.ptp(coordinates[1])
    return 1.5 if x_extent <= 0 or y_extent <= 0 else float(x_extent / y_extent)


def _figure(
    shape,
    coordinates,
    n_rows,
    n_cols,
    colorbar,
    figsize,
    orientation,
    options,
    hide_axes,
):
    panel_ratio = _data_ratio(shape, coordinates)
    grid_ratio = panel_ratio * n_cols / n_rows
    selected_orientation = (
        ("horizontal" if grid_ratio >= 1.25 else "vertical")
        if orientation == "auto"
        else orientation
    )
    if figsize is None:
        ratio = min(max(panel_ratio, 1 / 3), 3)
        panel_width = sqrt(24 * ratio)
        panel_height = sqrt(24 / ratio)
        panel_scale = min(1.0, 3.0 / max(panel_width, panel_height))
        width = panel_width * panel_scale * n_cols
        height = panel_height * panel_scale * n_rows
        if colorbar and selected_orientation == "vertical":
            width += 0.8
        elif colorbar:
            height += 0.6
        scale = min(1.0, 20 / width, 20 / height)
        figsize = width * scale, height * scale
    figure, axes = plt.subplots(
        n_rows, n_cols, squeeze=False, figsize=figsize, layout="compressed"
    )
    engine = figure.get_layout_engine()
    if engine is not None and hasattr(engine, "set"):
        engine.set(w_pad=0.05 if hide_axes else 0.12, h_pad=0.05 if hide_axes else 0.12)
    return figure, axes, selected_orientation


def _prepare_vector(vector, step, normalize, scale):
    values = vector[::step, ::step].detach().cpu().numpy().astype(float, copy=True)
    if normalize:
        magnitude = np.linalg.norm(values, axis=-1)
        valid = np.isfinite(magnitude) & (magnitude > 0)
        values[valid] /= magnitude[valid, None]
        values[~np.isfinite(values)] = 0
    return values * scale


def _index_coordinates(shape):
    return np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")


def _arrow_coordinates(coordinates, shape, step):
    x, y = coordinates or _index_coordinates(shape)
    return x[::step, ::step], y[::step, ::step]


def _finish_axes(axes, title, aspect, hide_axes):
    axes.set_title(title)
    axes.set_aspect(aspect, adjustable="box")
    if hide_axes:
        axes.set_axis_off()
        axes.margins(0)


def _hide_unused(axes, used):
    for current_axes in axes[used:]:
        current_axes.set_visible(False)


def _set_shared_labels(figure, axes, xlabel, ylabel):
    if xlabel is None and ylabel is None:
        return
    # Fixed-aspect grids may occupy only part of the canvas. Resolve their
    # final layout before placing shared labels relative to the data axes,
    # rather than relative to the outer figure edges.
    figure.canvas.draw()
    left = min(current_axes.get_position().x0 for current_axes in axes)
    right = max(current_axes.get_position().x1 for current_axes in axes)
    bottom = min(current_axes.get_position().y0 for current_axes in axes)
    top = max(current_axes.get_position().y1 for current_axes in axes)
    if xlabel is not None:
        figure.text(
            0.5 * (left + right),
            bottom - 0.09,
            xlabel,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=plt.rcParams["figure.labelsize"],
            fontweight=plt.rcParams["figure.labelweight"],
        )
    if ylabel is not None:
        figure.text(
            max(0.01, left - 0.055),
            0.5 * (bottom + top),
            ylabel,
            rotation="vertical",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=plt.rcParams["figure.labelsize"],
            fontweight=plt.rcParams["figure.labelweight"],
        )
