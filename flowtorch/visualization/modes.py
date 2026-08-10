"""Plots for image-like two-dimensional modal data."""

# standard library packages
from collections.abc import Sequence
from math import isfinite, sqrt
from typing import Any, Dict, Tuple

# third party packages
import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

_QUANTITIES = ("real", "imaginary", "absolute", "phase")
_TITLES = {
    "real": r"$\operatorname{Re}(\phi)$",
    "imaginary": r"$\operatorname{Im}(\phi)$",
    "absolute": r"$|\phi|$",
    "phase": r"$\arg(\phi)$",
}


def plot_spod_mode_2d(
    mode: pt.Tensor,
    shape: tuple[int, int] | None = None,
    x: pt.Tensor | None = None,
    y: pt.Tensor | None = None,
    components: Sequence[int] | None = None,
    component_labels: Sequence[str] | None = None,
    show_real: bool = True,
    show_imaginary: bool = True,
    show_absolute: bool = True,
    show_phase: bool = True,
    reference_index: tuple[int, int] | None = None,
    reference_point: tuple[float, float] | None = None,
    reference_component: int = 0,
    color_percentile: float = 100.0,
    real_cmap: Any = "coolwarm",
    imaginary_cmap: Any = "coolwarm",
    absolute_cmap: Any = "viridis",
    phase_cmap: Any = "twilight",
    colorbar: bool = True,
    aspect: str | float = "equal",
    hide_axes: bool = True,
    figsize: tuple[float, float] | None = None,
    plot_kwargs: Dict[str, Dict[str, Any]] | None = None,
    colorbar_kwargs: Dict[str, Any] | None = None,
) -> tuple[Figure, np.ndarray]:
    r"""Plot an image-like, potentially multicomponent SPOD mode.

    The flattened mode consists of contiguous component blocks, each containing
    ``nx * ny`` pixels. The spatial shape is inferred from ``x`` and ``y`` or
    supplied explicitly through ``shape``. Rows represent selected components;
    columns represent the enabled real, imaginary, absolute, and phase fields.

    A reference index or physical point optionally fixes the arbitrary complex
    phase. The phase of an amplitude-weighted ``4 x 4`` patch in
    ``reference_component`` is rotated to zero for every component. Colorbars
    are placed below wide fields and beside square or tall fields. Built-in
    labels use LaTeX-compatible math text, and user labels are passed through
    unchanged so raw LaTeX strings may be supplied.

    :param mode: flattened one- or multicomponent spatial mode
    :type mode: pt.Tensor
    :param shape: spatial ``(nx, ny)`` shape when coordinates are absent
    :type shape: Tuple[int, int], optional
    :param x: curvilinear x-coordinate
    :type x: pt.Tensor, optional
    :param y: curvilinear y-coordinate
    :type y: pt.Tensor, optional
    :param components: component indices to display; defaults to all
    :type components: Sequence[int], optional
    :param component_labels: optional label for each displayed component
    :type component_labels: Sequence[str], optional
    :param show_real: show real parts, defaults to ``True``
    :type show_real: bool, optional
    :param show_imaginary: show imaginary parts, defaults to ``True``
    :type show_imaginary: bool, optional
    :param show_absolute: show absolute values, defaults to ``True``
    :type show_absolute: bool, optional
    :param show_phase: show phases, defaults to ``True``
    :type show_phase: bool, optional
    :param reference_index: logical-grid reference index ``(i, j)``
    :type reference_index: Tuple[int, int], optional
    :param reference_point: physical reference point ``(x, y)``
    :type reference_point: Tuple[float, float], optional
    :param reference_component: component used for phase alignment, defaults to 0
    :type reference_component: int, optional
    :param color_percentile: robust upper percentile in ``(50, 100]``
    :type color_percentile: float, optional
    :param real_cmap: real-part colormap, defaults to ``"coolwarm"``
    :param imaginary_cmap: imaginary-part colormap, defaults to ``"coolwarm"``
    :param absolute_cmap: magnitude colormap, defaults to ``"viridis"``
    :param phase_cmap: cyclic phase colormap, defaults to ``"twilight"``
    :param colorbar: add one colorbar per panel, defaults to ``True``
    :type colorbar: bool, optional
    :param aspect: Matplotlib axes aspect, defaults to ``"equal"``
    :type aspect: str or float, optional
    :param hide_axes: hide coordinate axes, ticks, and spines, defaults to
        ``True``
    :type hide_axes: bool, optional
    :param figsize: explicit figure size in inches
    :type figsize: Tuple[float, float], optional
    :param plot_kwargs: quantity-specific plotting options keyed by ``real``,
        ``imaginary``, ``absolute``, or ``phase``
    :type plot_kwargs: Dict[str, Dict[str, Any]], optional
    :param colorbar_kwargs: additional ``Figure.colorbar`` options
    :type colorbar_kwargs: Dict[str, Any], optional
    :return: modifiable figure and axes of shape
        ``(n_components, n_quantities)``
    :rtype: Tuple[matplotlib.figure.Figure, np.ndarray]

    **Example**

    .. code-block:: python

        figure, axes = plot_spod_mode_2d(
            mode,
            x=x,
            y=y,
            reference_point=(0.25, 0.0),
            reference_component=0,
        )
        axes[0, 0].set_title(r"$\operatorname{Re}(\phi_1)$")
    """
    spatial_shape, coordinates = _spatial_data(shape, x, y)
    fields = _reshape_mode(mode, spatial_shape)
    quantities = _selected_quantities(
        show_real, show_imaginary, show_absolute, show_phase
    )
    selected_components = _selected_components(components, fields.shape[0])
    labels = _component_labels(component_labels, selected_components)
    _validate_plot_options(
        color_percentile,
        figsize,
        reference_index,
        reference_point,
        reference_component,
        fields.shape[0],
        spatial_shape,
        coordinates,
    )
    fields = _align_mode(
        fields,
        reference_index,
        reference_point,
        reference_component,
        coordinates,
    )
    options = _plot_options(plot_kwargs)
    colorbar_options = {} if colorbar_kwargs is None else colorbar_kwargs.copy()
    colorbar_orientation = _colorbar_orientation(spatial_shape, coordinates)
    selected_figsize = figsize or _automatic_size(
        spatial_shape,
        coordinates,
        len(quantities),
        len(selected_components),
        colorbar,
        colorbar_orientation,
        True,
    )
    figure, axes = plt.subplots(
        len(selected_components),
        len(quantities),
        squeeze=False,
        figsize=selected_figsize,
        layout="constrained",
    )
    for row, component in enumerate(selected_components):
        for column, quantity in enumerate(quantities):
            current_axes = axes[row, column]
            data = _quantity_data(fields[component], quantity)
            limits = _color_limits(data, quantity, color_percentile)
            artist = _plot_field(
                current_axes,
                data,
                coordinates,
                quantity,
                limits,
                real_cmap,
                imaginary_cmap,
                absolute_cmap,
                phase_cmap,
                options[quantity],
            )
            current_axes.set_aspect(aspect, adjustable="box")
            if row == 0:
                current_axes.set_title(_TITLES[quantity])
            if column == 0:
                current_axes.set_ylabel(labels[row])
            if hide_axes:
                current_axes.set_axis_off()
            if colorbar:
                current_colorbar_options = colorbar_options.copy()
                current_colorbar_options.setdefault("orientation", colorbar_orientation)
                current_colorbar_options.setdefault(
                    "pad", 0.08 if colorbar_orientation == "horizontal" else 0.03
                )
                current_colorbar_options.setdefault(
                    "shrink", 0.85 if colorbar_orientation == "horizontal" else 1.0
                )
                figure.colorbar(artist, ax=current_axes, **current_colorbar_options)
    return figure, axes


def plot_spod_modes_2d(
    modes: Sequence[pt.Tensor] | pt.Tensor,
    titles: Sequence[str],
    n_rows: int,
    n_cols: int,
    shape: tuple[int, int] | None = None,
    x: pt.Tensor | None = None,
    y: pt.Tensor | None = None,
    component: int = 0,
    show_real: bool = True,
    show_imaginary: bool = True,
    show_absolute: bool = True,
    show_phase: bool = True,
    reference_index: tuple[int, int] | None = None,
    reference_point: tuple[float, float] | None = None,
    reference_component: int = 0,
    color_percentile: float = 100.0,
    real_cmap: Any = "coolwarm",
    imaginary_cmap: Any = "coolwarm",
    absolute_cmap: Any = "viridis",
    phase_cmap: Any = "twilight",
    colorbar: bool = True,
    aspect: str | float = "equal",
    hide_axes: bool = True,
    figsize: tuple[float, float] | None = None,
    plot_kwargs: Dict[str, Dict[str, Any]] | None = None,
    colorbar_kwargs: Dict[str, Any] | None = None,
) -> tuple[Figure, np.ndarray]:
    r"""Compare one component of multiple image-like two-dimensional SPOD modes.

    The user-defined mode grid is repeated vertically for every selected
    quantity. Color limits and one optional colorbar are shared by all modes
    within a quantity. One reference location is shared across modes, while
    each mode is aligned using its own ``reference_component`` patch. A
    nonempty, optionally LaTeX-formatted title is required for every mode.

    :param modes: flattened modes or a tensor with one mode per row
    :type modes: Sequence[pt.Tensor] or pt.Tensor
    :param titles: LaTeX-compatible title for each mode
    :type titles: Sequence[str]
    :param n_rows: number of rows in each mode grid
    :type n_rows: int
    :param n_cols: number of columns in each mode grid
    :type n_cols: int
    :param shape: spatial shape when coordinates are absent
    :type shape: Tuple[int, int], optional
    :param x: curvilinear x-coordinate
    :type x: pt.Tensor, optional
    :param y: curvilinear y-coordinate
    :type y: pt.Tensor, optional
    :param component: component to display, defaults to 0
    :type component: int, optional
    :param show_real: show real parts, defaults to ``True``
    :type show_real: bool, optional
    :param show_imaginary: show imaginary parts, defaults to ``True``
    :type show_imaginary: bool, optional
    :param show_absolute: show absolute values, defaults to ``True``
    :type show_absolute: bool, optional
    :param show_phase: show phases, defaults to ``True``
    :type show_phase: bool, optional
    :param reference_index: shared logical-grid reference index
    :type reference_index: Tuple[int, int], optional
    :param reference_point: shared physical reference point
    :type reference_point: Tuple[float, float], optional
    :param reference_component: component used to align every mode
    :type reference_component: int, optional
    :param color_percentile: shared robust percentile in ``(50, 100]``
    :type color_percentile: float, optional
    :param real_cmap: real-part colormap
    :param imaginary_cmap: imaginary-part colormap
    :param absolute_cmap: magnitude colormap
    :param phase_cmap: cyclic phase colormap
    :param colorbar: add one shared colorbar per quantity, defaults to ``True``
    :type colorbar: bool, optional
    :param aspect: Matplotlib axes aspect, defaults to ``"equal"``
    :type aspect: str or float, optional
    :param hide_axes: hide coordinate axes, ticks, and spines, defaults to
        ``True``
    :type hide_axes: bool, optional
    :param figsize: explicit figure size in inches
    :type figsize: Tuple[float, float], optional
    :param plot_kwargs: quantity-specific plotting options
    :type plot_kwargs: Dict[str, Dict[str, Any]], optional
    :param colorbar_kwargs: additional shared-colorbar options
    :type colorbar_kwargs: Dict[str, Any], optional
    :return: figure and axes with shape
        ``(n_quantities, n_rows, n_cols)``
    :rtype: Tuple[matplotlib.figure.Figure, np.ndarray]

    **Example**

    .. code-block:: python

        figure, axes = plot_spod_modes_2d(
            modes,
            [r"$St=0.1$", r"$St=0.2$", r"$St=0.3$"],
            n_rows=2,
            n_cols=2,
            shape=(128, 256),
            component=0,
            reference_index=(64, 32),
        )
    """
    mode_list = _mode_sequence(modes)
    if len(mode_list) == 0:
        raise ValueError("at least one mode must be supplied")
    if len(titles) != len(mode_list):
        raise ValueError("titles must contain one entry per mode")
    if any(not isinstance(title, str) or not title for title in titles):
        raise ValueError("every mode title must be a nonempty string")
    if not isinstance(n_rows, int) or not isinstance(n_cols, int):
        raise ValueError("n_rows and n_cols must be integers")
    if n_rows < 1 or n_cols < 1 or n_rows * n_cols < len(mode_list):
        raise ValueError("the mode grid must contain every mode")
    spatial_shape, coordinates = _spatial_data(shape, x, y)
    fields = [_reshape_mode(mode, spatial_shape) for mode in mode_list]
    n_components = fields[0].shape[0]
    if any(current.shape[0] != n_components for current in fields[1:]):
        raise ValueError("all modes must contain the same number of components")
    quantities = _selected_quantities(
        show_real, show_imaginary, show_absolute, show_phase
    )
    _validate_plot_options(
        color_percentile,
        figsize,
        reference_index,
        reference_point,
        reference_component,
        n_components,
        spatial_shape,
        coordinates,
    )
    if not isinstance(component, int) or not 0 <= component < n_components:
        raise ValueError("component is outside the available component range")
    aligned = [
        _align_mode(
            current,
            reference_index,
            reference_point,
            reference_component,
            coordinates,
        )
        for current in fields
    ]
    options = _plot_options(plot_kwargs)
    colorbar_options = {} if colorbar_kwargs is None else colorbar_kwargs.copy()
    selected_figsize = figsize or _automatic_size(
        spatial_shape,
        coordinates,
        n_cols,
        n_rows * len(quantities),
        colorbar,
        "vertical",
        False,
    )
    figure, flat_axes = plt.subplots(
        n_rows * len(quantities),
        n_cols,
        squeeze=False,
        figsize=selected_figsize,
        layout="constrained",
    )
    axes = flat_axes.reshape(len(quantities), n_rows, n_cols)
    for quantity_index, quantity in enumerate(quantities):
        quantity_data = [
            _quantity_data(current[component], quantity) for current in aligned
        ]
        limits = _color_limits(pt.stack(quantity_data), quantity, color_percentile)
        artists = []
        quantity_axes = axes[quantity_index].ravel()
        for mode_index, data in enumerate(quantity_data):
            current_axes = quantity_axes[mode_index]
            artist = _plot_field(
                current_axes,
                data,
                coordinates,
                quantity,
                limits,
                real_cmap,
                imaginary_cmap,
                absolute_cmap,
                phase_cmap,
                options[quantity],
            )
            artists.append(artist)
            current_axes.set_aspect(aspect, adjustable="box")
            current_axes.set_title(titles[mode_index])
            if hide_axes:
                current_axes.set_axis_off()
        for unused_axes in quantity_axes[len(mode_list) :]:
            unused_axes.set_visible(False)
        if colorbar:
            current_colorbar_options = colorbar_options.copy()
            current_colorbar_options.setdefault("label", _TITLES[quantity])
            figure.colorbar(
                artists[0],
                ax=list(quantity_axes[: len(mode_list)]),
                **current_colorbar_options,
            )
    return figure, axes


def _mode_sequence(modes: Sequence[pt.Tensor] | pt.Tensor) -> list[pt.Tensor]:
    if isinstance(modes, pt.Tensor):
        if modes.ndim != 2:
            raise ValueError("a modes tensor must have shape (n_modes, n_state)")
        return list(modes.unbind(dim=0))
    return list(modes)


def _spatial_data(
    shape: tuple[int, int] | None,
    x: pt.Tensor | None,
    y: pt.Tensor | None,
) -> tuple[tuple[int, int], tuple[np.ndarray, np.ndarray] | None]:
    if (x is None) != (y is None):
        raise ValueError("x and y must either both be supplied or both be omitted")
    if x is not None and y is not None:
        if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
            raise ValueError("x and y must be identically shaped 2D tensors")
        if not pt.isfinite(x).all() or not pt.isfinite(y).all():
            raise ValueError("coordinates must contain only finite values")
        inferred = (x.shape[0], x.shape[1])
        if shape is not None and tuple(shape) != inferred:
            raise ValueError("shape must agree with the coordinate shape")
        return inferred, (x.detach().cpu().numpy(), y.detach().cpu().numpy())
    if shape is None or len(shape) != 2 or min(shape) < 1:
        raise ValueError("shape must contain two positive dimensions")
    return (shape[0], shape[1]), None


def _reshape_mode(mode: pt.Tensor, shape: tuple[int, int]) -> pt.Tensor:
    if mode.ndim != 1:
        raise ValueError("each mode must be one-dimensional")
    if not (mode.is_floating_point() or mode.is_complex()):
        raise ValueError("modes must have a floating-point or complex dtype")
    if not pt.isfinite(mode).all():
        raise ValueError("modes must contain only finite values")
    n_pixels = shape[0] * shape[1]
    if mode.numel() % n_pixels != 0:
        raise ValueError("mode length must be divisible by the number of pixels")
    return mode.reshape(mode.numel() // n_pixels, *shape)


def _selected_quantities(
    show_real: bool,
    show_imaginary: bool,
    show_absolute: bool,
    show_phase: bool,
) -> list[str]:
    enabled = (show_real, show_imaginary, show_absolute, show_phase)
    quantities = [name for name, show in zip(_QUANTITIES, enabled) if show]
    if not quantities:
        raise ValueError("at least one mode quantity must be displayed")
    return quantities


def _selected_components(
    components: Sequence[int] | None, n_components: int
) -> list[int]:
    selected = list(range(n_components)) if components is None else list(components)
    if not selected:
        raise ValueError("at least one component must be selected")
    if len(set(selected)) != len(selected) or any(
        not isinstance(index, int) or not 0 <= index < n_components
        for index in selected
    ):
        raise ValueError("components must be unique valid component indices")
    return selected


def _component_labels(
    labels: Sequence[str] | None, components: Sequence[int]
) -> list[str]:
    if labels is None:
        return [rf"$\phi_{{{index + 1}}}$" for index in components]
    if len(labels) != len(components):
        raise ValueError("component_labels must match the selected components")
    return list(labels)


def _validate_plot_options(
    percentile: float,
    figsize: tuple[float, float] | None,
    reference_index: tuple[int, int] | None,
    reference_point: tuple[float, float] | None,
    reference_component: int,
    n_components: int,
    shape: tuple[int, int],
    coordinates: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    if not isfinite(percentile) or not 50.0 < percentile <= 100.0:
        raise ValueError("color_percentile must be in the interval (50, 100]")
    if figsize is not None and (len(figsize) != 2 or min(figsize) <= 0.0):
        raise ValueError("figsize must contain two positive values")
    if reference_index is not None and reference_point is not None:
        raise ValueError("supply either reference_index or reference_point, not both")
    if (
        not isinstance(reference_component, int)
        or not 0 <= reference_component < n_components
    ):
        raise ValueError("reference_component is outside the available range")
    if reference_index is not None:
        if len(reference_index) != 2 or any(
            not isinstance(index, int) for index in reference_index
        ):
            raise ValueError("reference_index must contain two integers")
        if (
            not 0 <= reference_index[0] < shape[0]
            or not 0 <= reference_index[1] < shape[1]
        ):
            raise ValueError("reference_index is outside the spatial grid")
    if reference_point is not None:
        if coordinates is None:
            raise ValueError("reference_point requires x and y coordinates")
        if len(reference_point) != 2 or not all(
            isfinite(value) for value in reference_point
        ):
            raise ValueError("reference_point must contain two finite values")


def _align_mode(
    fields: pt.Tensor,
    reference_index: tuple[int, int] | None,
    reference_point: tuple[float, float] | None,
    reference_component: int,
    coordinates: tuple[np.ndarray, np.ndarray] | None,
) -> pt.Tensor:
    if reference_index is None and reference_point is None:
        return fields
    index = reference_index
    if reference_point is not None and coordinates is not None:
        distance = (coordinates[0] - reference_point[0]) ** 2
        distance += (coordinates[1] - reference_point[1]) ** 2
        index = tuple(np.unravel_index(np.argmin(distance), distance.shape))
    assert index is not None
    i_slice = _patch_slice(index[0], fields.shape[1])
    j_slice = _patch_slice(index[1], fields.shape[2])
    patch = fields[reference_component, i_slice, j_slice]
    patch_sum = patch.sum()
    real_dtype = patch.real.dtype
    tolerance = pt.finfo(real_dtype).eps * patch.abs().sum()
    if patch_sum.abs() <= tolerance:
        raise ValueError("reference patch has no resolvable mean phase")
    phase = pt.angle(patch_sum)
    return fields * pt.exp(-1j * phase)


def _patch_slice(center: int, size: int) -> slice:
    width = min(4, size)
    start = min(max(center - width // 2, 0), size - width)
    return slice(start, start + width)


def _quantity_data(field: pt.Tensor, quantity: str) -> pt.Tensor:
    if quantity == "real":
        return field.real
    if quantity == "imaginary":
        return field.imag if field.is_complex() else pt.zeros_like(field)
    if quantity == "absolute":
        return field.abs()
    return pt.angle(field)


def _color_limits(
    data: pt.Tensor, quantity: str, percentile: float
) -> tuple[float, float]:
    if quantity == "phase":
        return -float(np.pi), float(np.pi)
    if quantity in {"real", "imaginary"}:
        limit = float(pt.quantile(data.abs().reshape(-1), percentile / 100.0))
        limit = _positive_limit(limit, data.dtype)
        return -limit, limit
    limit = float(pt.quantile(data.reshape(-1), percentile / 100.0))
    return 0.0, _positive_limit(limit, data.dtype)


def _positive_limit(limit: float, dtype: pt.dtype) -> float:
    return limit if limit > 0.0 else pt.finfo(dtype).eps


def _plot_options(
    options: Dict[str, Dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {quantity: {} for quantity in _QUANTITIES}
    if options is None:
        return result
    invalid = set(options) - set(_QUANTITIES)
    if invalid:
        raise ValueError(f"unknown plot_kwargs quantities: {sorted(invalid)}")
    for quantity, values in options.items():
        result[quantity] = values.copy()
    return result


def _plot_field(
    axes: Axes,
    data: pt.Tensor,
    coordinates: tuple[np.ndarray, np.ndarray] | None,
    quantity: str,
    limits: tuple[float, float],
    real_cmap: Any,
    imaginary_cmap: Any,
    absolute_cmap: Any,
    phase_cmap: Any,
    options: dict[str, Any],
):
    selected_options = options.copy()
    cmap = {
        "real": real_cmap,
        "imaginary": imaginary_cmap,
        "absolute": absolute_cmap,
        "phase": phase_cmap,
    }[quantity]
    selected_options.update(cmap=cmap, vmin=limits[0], vmax=limits[1])
    values = data.detach().cpu().numpy()
    if coordinates is None:
        selected_options.setdefault("origin", "lower")
        return axes.imshow(values.T, **selected_options)
    selected_options.setdefault("shading", "auto")
    return axes.pcolormesh(coordinates[0], coordinates[1], values, **selected_options)


def _automatic_size(
    shape: tuple[int, int],
    coordinates: tuple[np.ndarray, np.ndarray] | None,
    n_columns: int,
    n_rows: int,
    colorbar: bool,
    colorbar_orientation: str,
    individual_colorbars: bool,
) -> tuple[float, float]:
    if coordinates is None:
        ratio = shape[0] / shape[1]
    else:
        x_extent = float(np.ptp(coordinates[0]))
        y_extent = float(np.ptp(coordinates[1]))
        ratio = 1.5 if x_extent <= 0.0 or y_extent <= 0.0 else x_extent / y_extent
    ratio = min(max(ratio, 1.0 / 3.0), 3.0)
    panel_width = sqrt(24.0 * ratio)
    panel_height = sqrt(24.0 / ratio)
    panel_scale = min(1.0, 3.0 / max(panel_width, panel_height))
    panel_width *= panel_scale
    panel_height *= panel_scale
    width = panel_width * n_columns
    height = panel_height * n_rows
    if colorbar and colorbar_orientation == "horizontal":
        height += 0.6 * n_rows
    elif colorbar:
        width += 0.8 * n_columns if individual_colorbars else 0.8
    scale = min(1.0, 20.0 / width, 20.0 / height)
    return width * scale, height * scale


def _colorbar_orientation(
    shape: tuple[int, int],
    coordinates: tuple[np.ndarray, np.ndarray] | None,
) -> str:
    if coordinates is None:
        ratio = shape[0] / shape[1]
    else:
        x_extent = float(np.ptp(coordinates[0]))
        y_extent = float(np.ptp(coordinates[1]))
        ratio = 1.0 if x_extent <= 0.0 or y_extent <= 0.0 else x_extent / y_extent
    return "horizontal" if ratio >= 1.25 else "vertical"
