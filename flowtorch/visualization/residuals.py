"""Plots for adaptive multi-taper convergence residuals."""

# standard library packages
from collections.abc import Sequence
from math import isfinite
from typing import Any, Dict, Tuple

# third party packages
import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_adaptive_residual(
    frequency: pt.Tensor,
    residual: pt.Tensor,
    reference_timescale: float | None = None,
    positive_frequencies_only: bool = True,
    frequency_limits: tuple[float, float] | None = None,
    log_frequency: bool = True,
    residual_floor: float | None = None,
    color_limits: tuple[float, float] | None = None,
    cmap: Any = "viridis",
    colorbar: bool = True,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    mesh_kwargs: Dict[str, Any] | None = None,
    colorbar_kwargs: Dict[str, Any] | None = None,
) -> Tuple[Figure, Axes]:
    r"""Plot adaptive multi-taper convergence residuals.

    ``residual`` has shape ``(n_frequency, max_tapers - 2)``. Its first
    column measures the change from two to three tapers and is therefore
    plotted at ``N_tapers=3``. Unavailable values represented by ``NaN`` are
    masked. The displayed quantity is ``log10(r)``.

    If ``residual_floor`` is omitted, zeros are replaced by the smallest
    positive finite residual in the selected data. If no positive residual
    exists, the smallest positive normal number of the residual dtype is used.
    Explicit ``color_limits`` apply to the displayed logarithmic values.

    :param frequency: frequency-bin coordinates in Hz
    :type frequency: pt.Tensor
    :param residual: adaptive residuals arranged by frequency and taper count
    :type residual: pt.Tensor
    :param reference_timescale: reference time used to plot Strouhal number
    :type reference_timescale: float, optional
    :param positive_frequencies_only: omit zero and negative frequencies,
        defaults to ``True``
    :type positive_frequencies_only: bool, optional
    :param frequency_limits: selected limits in dimensional frequency units
    :type frequency_limits: Tuple[float, float], optional
    :param log_frequency: use a logarithmic frequency axis, defaults to ``True``
    :type log_frequency: bool, optional
    :param residual_floor: positive floor applied before taking the logarithm
    :type residual_floor: float, optional
    :param color_limits: explicit limits for ``log10(r)``
    :type color_limits: Tuple[float, float], optional
    :param cmap: Matplotlib colormap, defaults to ``"viridis"``
    :param colorbar: add a colorbar, defaults to ``True``
    :type colorbar: bool, optional
    :param ax: existing axes; a new figure is created when omitted
    :type ax: matplotlib.axes.Axes, optional
    :param figsize: figure size used only when creating new axes
    :type figsize: Tuple[float, float], optional
    :param mesh_kwargs: additional options for ``Axes.pcolormesh``
    :type mesh_kwargs: Dict[str, Any], optional
    :param colorbar_kwargs: additional options for ``Figure.colorbar``
    :type colorbar_kwargs: Dict[str, Any], optional
    :return: modifiable Matplotlib figure and axes
    :rtype: Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

    **Example**

    .. code-block:: python

        figure, axes = plot_adaptive_residual(
            spod.frequency,
            spod.residual,
            reference_timescale=0.1,
        )
    """
    _validate_residual_inputs(
        frequency,
        residual,
        reference_timescale,
        frequency_limits,
        residual_floor,
        color_limits,
        figsize,
    )
    coordinate, values, taper_count = _residual_plot_data(
        frequency,
        residual,
        reference_timescale,
        positive_frequencies_only,
        frequency_limits,
        residual_floor,
    )
    if ax is None:
        figure, axes = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        if figsize is not None:
            raise ValueError("figsize cannot be supplied with ax")
        axes = ax
        figure = axes.figure
    mesh_options = _mesh_options(values, color_limits, cmap, mesh_kwargs)
    mesh = axes.pcolormesh(
        coordinate,
        taper_count,
        values.T,
        **mesh_options,
    )
    _format_residual_axes(
        axes,
        coordinate,
        taper_count,
        reference_timescale,
        log_frequency,
    )
    if colorbar:
        bar_options = {} if colorbar_kwargs is None else colorbar_kwargs.copy()
        bar_options.setdefault("label", r"$\log_{10}(r)$")
        figure.colorbar(mesh, ax=axes, **bar_options)
    return figure, axes


def plot_adaptive_residuals(
    frequencies: Sequence[pt.Tensor],
    residuals: Sequence[pt.Tensor],
    titles: Sequence[str],
    n_rows: int,
    n_cols: int,
    reference_timescale: float | Sequence[float] | None = None,
    positive_frequencies_only: bool = True,
    frequency_limits: tuple[float, float] | None = None,
    log_frequency: bool = True,
    residual_floor: float | None = None,
    color_limits: tuple[float, float] | None = None,
    cmap: Any = "viridis",
    colorbar: bool = True,
    sharex: bool = True,
    sharey: bool = True,
    figsize: tuple[float, float] | None = None,
    mesh_kwargs: Dict[str, Any] | None = None,
    colorbar_kwargs: Dict[str, Any] | None = None,
) -> tuple[Figure, np.ndarray]:
    r"""Compare adaptive convergence residuals in a subplot grid.

    All panels always share color limits so that equal colors represent equal
    residuals. A scalar ``reference_timescale`` is shared by every panel; a
    sequence supplies one scale per panel. Unused grid cells are hidden, and
    one optional colorbar is shared by the complete figure.

    :param frequencies: frequency coordinate for every residual
    :type frequencies: Sequence[pt.Tensor]
    :param residuals: residual tensor for every panel
    :type residuals: Sequence[pt.Tensor]
    :param titles: nonempty, optionally LaTeX-formatted panel titles
    :type titles: Sequence[str]
    :param n_rows: number of subplot rows
    :type n_rows: int
    :param n_cols: number of subplot columns
    :type n_cols: int
    :param reference_timescale: shared or panel-specific reference time
    :type reference_timescale: float or Sequence[float], optional
    :param positive_frequencies_only: omit zero and negative frequencies
    :type positive_frequencies_only: bool, optional
    :param frequency_limits: shared limits in dimensional frequency units
    :type frequency_limits: Tuple[float, float], optional
    :param log_frequency: use logarithmic frequency axes, defaults to ``True``
    :type log_frequency: bool, optional
    :param residual_floor: shared positive floor before taking logarithms
    :type residual_floor: float, optional
    :param color_limits: shared explicit limits for ``log10(r)``
    :type color_limits: Tuple[float, float], optional
    :param cmap: shared Matplotlib colormap
    :param colorbar: add one shared colorbar, defaults to ``True``
    :type colorbar: bool, optional
    :param sharex: share horizontal axes, defaults to ``True``
    :type sharex: bool, optional
    :param sharey: share vertical axes, defaults to ``True``
    :type sharey: bool, optional
    :param figsize: explicit figure size
    :type figsize: Tuple[float, float], optional
    :param mesh_kwargs: additional options shared by all meshes
    :type mesh_kwargs: Dict[str, Any], optional
    :param colorbar_kwargs: additional shared-colorbar options
    :type colorbar_kwargs: Dict[str, Any], optional
    :return: figure and axes with shape ``(n_rows, n_cols)``
    :rtype: Tuple[matplotlib.figure.Figure, np.ndarray]

    **Example**

    .. code-block:: python

        figure, axes = plot_adaptive_residuals(
            frequencies,
            residuals,
            [r"$\epsilon=10^{-3}$", r"$\epsilon=10^{-4}$"],
            n_rows=1,
            n_cols=2,
        )
    """
    count = len(residuals)
    if count == 0:
        raise ValueError("at least one residual must be supplied")
    if len(frequencies) != count or len(titles) != count:
        raise ValueError("frequencies, residuals, and titles must have equal lengths")
    if any(not isinstance(title, str) or not title for title in titles):
        raise ValueError("every title must be a nonempty string")
    if not isinstance(n_rows, int) or not isinstance(n_cols, int):
        raise ValueError("n_rows and n_cols must be integers")
    if n_rows < 1 or n_cols < 1 or n_rows * n_cols < count:
        raise ValueError("the subplot grid must contain every residual")
    timescales = _reference_timescales(reference_timescale, count)
    for frequency, residual, timescale in zip(
        frequencies, residuals, timescales, strict=True
    ):
        _validate_residual_inputs(
            frequency,
            residual,
            timescale,
            frequency_limits,
            residual_floor,
            color_limits,
            figsize,
        )

    selected_residuals = [
        _selected_residual(
            frequency,
            residual,
            positive_frequencies_only,
            frequency_limits,
        )
        for frequency, residual in zip(frequencies, residuals, strict=True)
    ]
    shared_floor = _residual_floor(selected_residuals, residual_floor)
    plot_data = [
        _residual_plot_data(
            frequency,
            residual,
            timescale,
            positive_frequencies_only,
            frequency_limits,
            shared_floor,
        )
        for frequency, residual, timescale in zip(
            frequencies, residuals, timescales, strict=True
        )
    ]
    all_values = pt.cat([values[pt.isfinite(values)] for _, values, _ in plot_data])
    shared_limits = (
        color_limits
        if color_limits is not None
        else (float(all_values.min()), float(all_values.max()))
    )
    if shared_limits[0] == shared_limits[1]:
        shared_limits = (shared_limits[0] - 0.5, shared_limits[1] + 0.5)

    if figsize is None:
        figsize = (min(16.0, 3.6 * n_cols), min(12.0, 2.8 * n_rows))
    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        constrained_layout=True,
    )
    flat_axes = axes.ravel()
    mesh = None
    active_axes = []
    for index, (coordinate, values, taper_count) in enumerate(plot_data):
        current_axes = flat_axes[index]
        options = _mesh_options(values, shared_limits, cmap, mesh_kwargs)
        mesh = current_axes.pcolormesh(
            coordinate,
            taper_count,
            values.T,
            **options,
        )
        _format_residual_axes(
            current_axes,
            coordinate,
            taper_count,
            timescales[index],
            log_frequency,
        )
        current_axes.set_title(titles[index])
        active_axes.append(current_axes)
    for unused_axes in flat_axes[count:]:
        unused_axes.set_visible(False)
    if sharex:
        coordinate_min = min(float(coordinate[0]) for coordinate, _, _ in plot_data)
        coordinate_max = max(float(coordinate[-1]) for coordinate, _, _ in plot_data)
        active_axes[0].set_xlim(coordinate_min, coordinate_max)
    if sharey:
        taper_max = max(float(taper_count[-1]) for _, _, taper_count in plot_data)
        active_axes[0].set_ylim(3.0, taper_max if taper_max > 3.0 else 3.5)
    if colorbar and mesh is not None:
        bar_options = {} if colorbar_kwargs is None else colorbar_kwargs.copy()
        bar_options.setdefault("label", r"$\log_{10}(r)$")
        figure.colorbar(mesh, ax=active_axes, **bar_options)
    return figure, axes


def _residual_plot_data(
    frequency: pt.Tensor,
    residual: pt.Tensor,
    reference_timescale: float | None,
    positive_frequencies_only: bool,
    frequency_limits: tuple[float, float] | None,
    residual_floor: float | None,
) -> tuple[np.ndarray, pt.Tensor, np.ndarray]:
    frequency_mask = _frequency_mask(
        frequency, positive_frequencies_only, frequency_limits
    )
    selected = residual[frequency_mask]
    floor = _residual_floor([selected], residual_floor)
    finite = pt.isfinite(selected)
    values = pt.where(finite, pt.clamp(selected, min=floor), selected)
    values = pt.log10(values)
    coordinate = frequency[frequency_mask]
    if reference_timescale is not None:
        coordinate = coordinate * reference_timescale
    taper_count = pt.arange(3, residual.shape[1] + 3)
    return (
        coordinate.detach().cpu().numpy(),
        values.detach().cpu(),
        taper_count.numpy(),
    )


def _selected_residual(
    frequency: pt.Tensor,
    residual: pt.Tensor,
    positive_frequencies_only: bool,
    frequency_limits: tuple[float, float] | None,
) -> pt.Tensor:
    return residual[
        _frequency_mask(frequency, positive_frequencies_only, frequency_limits)
    ]


def _frequency_mask(
    frequency: pt.Tensor,
    positive_frequencies_only: bool,
    frequency_limits: tuple[float, float] | None,
) -> pt.Tensor:
    mask = pt.ones_like(frequency, dtype=pt.bool)
    if positive_frequencies_only:
        mask &= frequency > 0.0
    if frequency_limits is not None:
        mask &= frequency >= frequency_limits[0]
        mask &= frequency <= frequency_limits[1]
    if int(mask.sum()) < 2:
        raise ValueError("frequency selection must contain at least two bins")
    return mask


def _residual_floor(
    residuals: Sequence[pt.Tensor], residual_floor: float | None
) -> float:
    if residual_floor is not None:
        return residual_floor
    positive = [
        residual[(residual > 0.0) & pt.isfinite(residual)] for residual in residuals
    ]
    positive = [values for values in positive if values.numel() > 0]
    if positive:
        return float(pt.cat(positive).min())
    dtype = residuals[0].dtype
    return float(pt.finfo(dtype).tiny)


def _mesh_options(
    values: pt.Tensor,
    color_limits: tuple[float, float] | None,
    cmap: Any,
    mesh_kwargs: Dict[str, Any] | None,
) -> Dict[str, Any]:
    options = {} if mesh_kwargs is None else mesh_kwargs.copy()
    finite = values[pt.isfinite(values)]
    limits = (
        color_limits
        if color_limits is not None
        else (float(finite.min()), float(finite.max()))
    )
    if limits[0] == limits[1]:
        limits = (limits[0] - 0.5, limits[1] + 0.5)
    options.setdefault("shading", "nearest")
    options.setdefault("cmap", cmap)
    options.setdefault("vmin", limits[0])
    options.setdefault("vmax", limits[1])
    return options


def _format_residual_axes(
    axes: Axes,
    coordinate: np.ndarray,
    taper_count: np.ndarray,
    reference_timescale: float | None,
    log_frequency: bool,
) -> None:
    if log_frequency:
        axes.set_xscale("log")
    axes.set_xlim(float(coordinate[0]), float(coordinate[-1]))
    if taper_count.size == 1:
        axes.set_ylim(float(taper_count[0] - 0.5), float(taper_count[0] + 0.5))
    else:
        axes.set_ylim(float(taper_count[0]), float(taper_count[-1]))
    axes.set_xlabel(r"$f\;[\mathrm{Hz}]$" if reference_timescale is None else r"$St$")
    axes.set_ylabel(r"$N_{\mathrm{tapers}}$")


def _reference_timescales(
    reference_timescale: float | Sequence[float] | None,
    count: int,
) -> list[float | None]:
    if reference_timescale is None or isinstance(reference_timescale, (int, float)):
        return [reference_timescale] * count
    timescales: list[float | None] = list(reference_timescale)
    if len(timescales) != count:
        raise ValueError("reference_timescale must contain one value per residual")
    return timescales


def _validate_residual_inputs(
    frequency: pt.Tensor,
    residual: pt.Tensor,
    reference_timescale: float | None,
    frequency_limits: tuple[float, float] | None,
    residual_floor: float | None,
    color_limits: tuple[float, float] | None,
    figsize: tuple[float, float] | None,
) -> None:
    if frequency.ndim != 1 or not frequency.is_floating_point():
        raise ValueError("frequency must be a one-dimensional floating-point tensor")
    if frequency.numel() < 2 or not bool(pt.all(frequency[1:] > frequency[:-1])):
        raise ValueError("frequency must contain at least two increasing values")
    if residual.ndim != 2 or residual.shape[0] != frequency.numel():
        raise ValueError("residual must have shape (n_frequency, n_taper_steps)")
    if residual.shape[1] < 1:
        raise ValueError("residual must contain at least one taper step")
    if not residual.is_floating_point():
        raise ValueError("residual must have a floating-point dtype")
    finite = residual[pt.isfinite(residual)]
    if finite.numel() == 0 or bool((finite < 0.0).any()):
        raise ValueError("residual must contain finite non-negative values")
    if reference_timescale is not None and (
        not isfinite(reference_timescale) or reference_timescale <= 0.0
    ):
        raise ValueError("reference_timescale must be positive and finite")
    _validate_limits(frequency_limits, "frequency_limits")
    _validate_limits(color_limits, "color_limits")
    if residual_floor is not None and (
        not isfinite(residual_floor) or residual_floor <= 0.0
    ):
        raise ValueError("residual_floor must be positive and finite")
    if figsize is not None and (
        len(figsize) != 2
        or any(not isfinite(value) or value <= 0.0 for value in figsize)
    ):
        raise ValueError("figsize entries must be positive and finite")


def _validate_limits(limits: tuple[float, float] | None, name: str) -> None:
    if limits is not None and (
        len(limits) != 2
        or any(not isfinite(value) for value in limits)
        or limits[0] >= limits[1]
    ):
        raise ValueError(f"{name} must contain two increasing finite values")
