"""Plots for modal-decomposition eigenvalue spectra."""

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


def plot_spod_spectrum(
    frequency: pt.Tensor,
    eigenvalues: pt.Tensor,
    n_show: int = 3,
    show_sum: bool = True,
    reference_timescale: float | None = None,
    energy_density: bool = False,
    normalize_energy: bool = False,
    half_bandwidth: float | pt.Tensor | None = None,
    ax: Axes | None = None,
    show_legend: bool = True,
    mode_color: Any = "C0",
    sum_color: Any = "black",
    min_opacity: float = 0.25,
    mode_kwargs: Dict[str, Any] | None = None,
    sum_kwargs: Dict[str, Any] | None = None,
    bandwidth_kwargs: Dict[str, Any] | None = None,
) -> Tuple[Figure, Axes]:
    r"""Plot leading SPOD eigenvalues and their sum over frequency bins.

    Eigenvalues must have shape ``(n_frequency, n_eigenvalues)`` and be sorted
    in descending order along the second dimension. Values no larger than
    :func:`torch.finfo` ``.tiny`` are treated as unavailable. This masks the
    zero padding used by adaptive AMSPOD without discarding small representable
    eigenvalues.

    If ``reference_timescale`` is supplied, the horizontal coordinate is the
    Strouhal number ``St = f * reference_timescale``. With
    ``energy_density=True``, bin energies are divided by ``delta_f`` or by
    ``delta_St`` so that integration over the displayed coordinate recovers
    total energy. With ``normalize_energy=True``, the complete spectrum is
    divided by its own total resolved energy.

    ``half_bandwidth`` is shown as faint horizontal error bars on the summed
    spectrum, or on the leading eigenvalue if the sum is hidden.

    :param frequency: frequency-bin coordinates in Hz
    :type frequency: pt.Tensor
    :param eigenvalues: SPOD eigenvalues arranged by frequency and rank
    :type eigenvalues: pt.Tensor
    :param n_show: maximum number of leading eigenvalues to show, defaults to 3
    :type n_show: int, optional
    :param show_sum: plot the sum over all eigenvalues, defaults to ``True``
    :type show_sum: bool, optional
    :param reference_timescale: reference time used to convert frequency to
        Strouhal number
    :type reference_timescale: float, optional
    :param energy_density: convert bin energy to density in the displayed
        horizontal coordinate, defaults to ``False``
    :type energy_density: bool, optional
    :param normalize_energy: divide by the total energy of the spectrum,
        defaults to ``False``
    :type normalize_energy: bool, optional
    :param half_bandwidth: scalar or frequency-dependent half-bandwidth in Hz
    :type half_bandwidth: float or pt.Tensor, optional
    :param ax: existing Matplotlib axes; a new figure is created when omitted
    :type ax: matplotlib.axes.Axes, optional
    :param show_legend: show the sum and generic eigenvalue labels, defaults to
        ``True``
    :type show_legend: bool, optional
    :param mode_color: shared color of the individual eigenvalue curves,
        defaults to ``"C0"``
    :param sum_color: color of the summed spectrum, defaults to ``"black"``
    :param min_opacity: opacity of the last displayed eigenvalue
    :type min_opacity: float, optional
    :param mode_kwargs: additional ``Axes.plot`` options for eigenvalues
    :type mode_kwargs: Dict[str, Any], optional
    :param sum_kwargs: additional ``Axes.plot`` options for the sum
    :type sum_kwargs: Dict[str, Any], optional
    :param bandwidth_kwargs: additional ``Axes.errorbar`` options
    :type bandwidth_kwargs: Dict[str, Any], optional
    :return: modifiable Matplotlib figure and axes
    :rtype: Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

    **Example**

    .. code-block:: python

        figure, axes = plot_spod_spectrum(
            spod.frequency,
            spod.eigvals,
            n_show=4,
            reference_timescale=0.1,
            energy_density=True,
        )
        axes.set_xlim(0.0, 1.0)
    """
    _validate_spectrum_inputs(
        frequency,
        eigenvalues,
        n_show,
        reference_timescale,
        min_opacity,
        half_bandwidth,
    )
    mode_options = {} if mode_kwargs is None else mode_kwargs.copy()
    sum_options = {} if sum_kwargs is None else sum_kwargs.copy()
    bandwidth_options = {} if bandwidth_kwargs is None else bandwidth_kwargs.copy()
    if ax is None:
        figure, axes = plt.subplots()
    else:
        axes = ax
        figure = axes.figure

    coordinate, values, bandwidth, coordinate_label, value_label = _plot_data(
        frequency,
        eigenvalues,
        reference_timescale,
        energy_density,
        normalize_energy,
        half_bandwidth,
    )
    threshold = pt.finfo(values.dtype).tiny
    available = values > threshold
    n_curves = min(n_show, values.shape[1])
    opacity = pt.linspace(1.0, min_opacity, n_curves).tolist()

    plotted_sum = None
    summed = values.sum(dim=1)
    if show_sum:
        sum_options.setdefault("color", sum_color)
        sum_options.setdefault("linewidth", 1.8)
        sum_options.setdefault("label", r"$\lambda_{\mathrm{sum}}$")
        sum_values = _masked_values(summed, summed > threshold)
        (plotted_sum,) = axes.plot(coordinate, sum_values, **sum_options)

    leading_values = None
    for index in range(n_curves):
        curve_options = mode_options.copy()
        curve_options.setdefault("color", mode_color)
        curve_options.setdefault("alpha", opacity[index])
        curve_options.setdefault(
            "label", r"$\lambda_i$" if index == 0 else "_nolegend_"
        )
        leading_values = _masked_values(values[:, index], available[:, index])
        axes.plot(coordinate, leading_values, **curve_options)

    if bandwidth is not None:
        target = summed if show_sum else values[:, 0]
        target_available = target > threshold
        bandwidth_options.setdefault("fmt", "none")
        bandwidth_options.setdefault("ecolor", "C3")
        bandwidth_options.setdefault("alpha", 0.18)
        bandwidth_options.setdefault("elinewidth", 0.8)
        bandwidth_options.setdefault("capsize", 0.0)
        bandwidth_options.setdefault("errorevery", max(1, coordinate.size // 20))
        bandwidth_options.setdefault(
            "label",
            (
                r"$\Delta f_{1/2}$"
                if reference_timescale is None
                else r"$\Delta St_{1/2}$"
            ),
        )
        axes.errorbar(
            coordinate[target_available],
            target[target_available].detach().cpu().numpy(),
            xerr=bandwidth[target_available],
            **bandwidth_options,
        )

    axes.set_yscale("log")
    axes.set_xlim(float(np.min(coordinate)), float(np.max(coordinate)))
    axes.set_xlabel(coordinate_label)
    axes.set_ylabel(value_label)
    if show_legend:
        axes.legend()
    return figure, axes


def plot_spod_spectra(
    frequencies: Sequence[pt.Tensor],
    eigenvalues: Sequence[pt.Tensor],
    titles: Sequence[str],
    n_rows: int,
    n_cols: int,
    n_show: int = 3,
    show_sum: bool = True,
    reference_timescale: float | Sequence[float] | None = None,
    energy_density: bool = False,
    normalize_energy: bool = False,
    sharex: bool = True,
    sharey: bool = True,
    figsize: tuple[float, float] | None = None,
    mode_color: Any = "C0",
    sum_color: Any = "black",
    min_opacity: float = 0.25,
    mode_kwargs: Dict[str, Any] | None = None,
    sum_kwargs: Dict[str, Any] | None = None,
) -> tuple[Figure, np.ndarray]:
    r"""Compare multiple SPOD eigenvalue spectra in a shared-axes grid.

    A scalar ``reference_timescale`` is shared by every spectrum. A sequence
    supplies one timescale per subplot. Unused grid cells are hidden, and only
    the first subplot contains the legend labels ``lambda_sum`` and
    ``lambda_i``.

    :param frequencies: frequency coordinate for each spectrum
    :type frequencies: Sequence[pt.Tensor]
    :param eigenvalues: eigenvalue tensor for each spectrum
    :type eigenvalues: Sequence[pt.Tensor]
    :param titles: subplot title for each spectrum
    :type titles: Sequence[str]
    :param n_rows: number of subplot rows
    :type n_rows: int
    :param n_cols: number of subplot columns
    :type n_cols: int
    :param n_show: maximum number of leading eigenvalues, defaults to 3
    :type n_show: int, optional
    :param show_sum: plot the sum over all eigenvalues, defaults to ``True``
    :type show_sum: bool, optional
    :param reference_timescale: shared reference time or one value per spectrum
    :type reference_timescale: float or Sequence[float], optional
    :param energy_density: convert energy to density in frequency or Strouhal
        coordinates, defaults to ``False``
    :type energy_density: bool, optional
    :param normalize_energy: normalize each spectrum by its own total energy,
        defaults to ``False``
    :type normalize_energy: bool, optional
    :param sharex: share horizontal axes, defaults to ``True``
    :type sharex: bool, optional
    :param sharey: share vertical axes, defaults to ``True``
    :type sharey: bool, optional
    :param figsize: complete figure size in inches
    :type figsize: Tuple[float, float], optional
    :param mode_color: shared color of individual eigenvalue curves
    :param sum_color: color of summed spectra
    :param min_opacity: opacity of the last displayed eigenvalue
    :type min_opacity: float, optional
    :param mode_kwargs: additional eigenvalue ``Axes.plot`` options
    :type mode_kwargs: Dict[str, Any], optional
    :param sum_kwargs: additional sum ``Axes.plot`` options
    :type sum_kwargs: Dict[str, Any], optional
    :return: modifiable Matplotlib figure and two-dimensional axes array
    :rtype: Tuple[matplotlib.figure.Figure, np.ndarray]

    **Example**

    .. code-block:: python

        spectra = [spod_a.eigvals, spod_b.eigvals, spod_c.eigvals]
        frequencies = [spod_a.frequency, spod_b.frequency, spod_c.frequency]
        titles = [r"$M=0.3$", r"$M=0.5$", r"$M=0.7$"]

        figure, axes = plot_spod_spectra(
            frequencies,
            spectra,
            titles,
            n_rows=2,
            n_cols=2,
            n_show=3,
            reference_timescale=[0.12, 0.10, 0.08],
            energy_density=True,
            normalize_energy=True,
        )
        axes[0, 0].grid(True, which="both", alpha=0.2)
        figure.suptitle(r"Normalized SPOD spectra")
    """
    n_spectra = len(frequencies)
    if n_spectra == 0:
        raise ValueError("at least one spectrum must be supplied")
    if len(eigenvalues) != n_spectra or len(titles) != n_spectra:
        raise ValueError("frequencies, eigenvalues, and titles must have equal length")
    if not isinstance(n_rows, int) or not isinstance(n_cols, int):
        raise ValueError("n_rows and n_cols must be integers")
    if n_rows < 1 or n_cols < 1 or n_rows * n_cols < n_spectra:
        raise ValueError("the subplot grid must contain every spectrum")
    timescales = _reference_timescales(reference_timescale, n_spectra)
    selected_figsize = (4.0 * n_cols, 3.0 * n_rows) if figsize is None else figsize
    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        sharex=sharex,
        sharey=sharey,
        figsize=selected_figsize,
        squeeze=False,
        layout="constrained",
    )
    flat_axes = axes.ravel()
    x_label = ""
    y_label = ""
    x_min = float("inf")
    x_max = -float("inf")
    for index, (frequency, values, title, timescale) in enumerate(
        zip(frequencies, eigenvalues, titles, timescales)
    ):
        _, current_axes = plot_spod_spectrum(
            frequency,
            values,
            n_show=n_show,
            show_sum=show_sum,
            reference_timescale=timescale,
            energy_density=energy_density,
            normalize_energy=normalize_energy,
            ax=flat_axes[index],
            show_legend=index == 0,
            mode_color=mode_color,
            sum_color=sum_color,
            min_opacity=min_opacity,
            mode_kwargs=mode_kwargs,
            sum_kwargs=sum_kwargs,
        )
        current_axes.set_title(title)
        displayed_frequency = frequency * (1.0 if timescale is None else timescale)
        x_min = min(x_min, float(displayed_frequency.min()))
        x_max = max(x_max, float(displayed_frequency.max()))
        x_label = current_axes.get_xlabel()
        y_label = current_axes.get_ylabel()
        current_axes.set_xlabel("")
        current_axes.set_ylabel("")
    for unused_axes in flat_axes[n_spectra:]:
        unused_axes.set_visible(False)
    for current_axes in flat_axes[:n_spectra]:
        current_axes.set_xlim(x_min, x_max)
    figure.supxlabel(x_label)
    figure.supylabel(y_label)
    return figure, axes


def _validate_spectrum_inputs(
    frequency: pt.Tensor,
    eigenvalues: pt.Tensor,
    n_show: int,
    reference_timescale: float | None,
    min_opacity: float,
    half_bandwidth: float | pt.Tensor | None,
) -> None:
    if frequency.ndim != 1 or eigenvalues.ndim != 2:
        raise ValueError("frequency must be 1D and eigenvalues must be 2D")
    if eigenvalues.shape[0] != frequency.shape[0] or frequency.numel() < 2:
        raise ValueError("frequency and eigenvalue frequency dimensions must match")
    if not frequency.is_floating_point() or not eigenvalues.is_floating_point():
        raise ValueError("frequency and eigenvalues must have floating-point dtypes")
    if not pt.isfinite(frequency).all() or not pt.isfinite(eigenvalues).all():
        raise ValueError("frequency and eigenvalues must contain only finite values")
    if (eigenvalues < 0.0).any():
        raise ValueError("eigenvalues must be non-negative")
    if not isinstance(n_show, int) or n_show < 1:
        raise ValueError("n_show must be a positive integer")
    if reference_timescale is not None and (
        not isfinite(reference_timescale) or reference_timescale <= 0.0
    ):
        raise ValueError("reference_timescale must be finite and positive")
    if not isfinite(min_opacity) or not 0.0 < min_opacity <= 1.0:
        raise ValueError("min_opacity must be in the interval (0, 1]")
    if half_bandwidth is not None:
        bandwidth = pt.as_tensor(half_bandwidth)
        if bandwidth.ndim > 1 or (
            bandwidth.ndim == 1 and bandwidth.shape != frequency.shape
        ):
            raise ValueError("half_bandwidth must be scalar or match frequency")
        if not pt.isfinite(bandwidth).all() or (bandwidth < 0.0).any():
            raise ValueError("half_bandwidth must be finite and non-negative")


def _plot_data(
    frequency: pt.Tensor,
    eigenvalues: pt.Tensor,
    reference_timescale: float | None,
    energy_density: bool,
    normalize_energy: bool,
    half_bandwidth: float | pt.Tensor | None,
) -> tuple[np.ndarray, pt.Tensor, np.ndarray | None, str, str]:
    scale = 1.0 if reference_timescale is None else reference_timescale
    coordinate = (frequency * scale).detach().cpu().numpy()
    values = eigenvalues.detach().cpu().clone()
    total_energy = values.sum()
    if normalize_energy:
        if total_energy <= pt.finfo(values.dtype).tiny:
            raise ValueError("cannot normalize a spectrum with zero total energy")
        values /= total_energy
    if energy_density:
        spacing = _frequency_spacing(frequency) * scale
        values /= spacing.to(dtype=values.dtype, device=values.device)
    bandwidth = None
    if half_bandwidth is not None:
        bandwidth_tensor = pt.as_tensor(half_bandwidth).expand_as(frequency) * scale
        bandwidth = bandwidth_tensor.detach().cpu().numpy()
    coordinate_label = r"$f\;[\mathrm{Hz}]$" if reference_timescale is None else r"$St$"
    if not energy_density and not normalize_energy:
        value_label = r"$\lambda$"
    elif not energy_density:
        value_label = r"$\lambda/E$"
    elif reference_timescale is None and not normalize_energy:
        value_label = r"$\lambda/\Delta f$"
    elif reference_timescale is None:
        value_label = r"$\lambda/(E\,\Delta f)$"
    elif not normalize_energy:
        value_label = r"$\lambda/\Delta St$"
    else:
        value_label = r"$\lambda/(E\,\Delta St)$"
    return coordinate, values, bandwidth, coordinate_label, value_label


def _frequency_spacing(frequency: pt.Tensor) -> pt.Tensor:
    differences = pt.diff(frequency).abs()
    positive = differences[differences > pt.finfo(frequency.dtype).tiny]
    if positive.numel() == 0:
        raise ValueError("frequency must contain at least two distinct bins")
    spacing = positive.min()
    tolerance = 100.0 * pt.finfo(frequency.dtype).eps * spacing
    irregular = (positive - spacing).abs() > tolerance
    if int(irregular.sum()) > 1:
        raise ValueError("energy density requires uniformly spaced frequency bins")
    return spacing


def _masked_values(values: pt.Tensor, available: pt.Tensor) -> np.ndarray:
    masked = values.detach().cpu().clone()
    masked[~available.detach().cpu()] = float("nan")
    return masked.numpy()


def _reference_timescales(
    reference_timescale: float | Sequence[float] | None,
    n_spectra: int,
) -> list[float | None]:
    if reference_timescale is None:
        return [None] * n_spectra
    if isinstance(reference_timescale, (float, int)):
        return [float(reference_timescale)] * n_spectra
    values: list[float | None] = [float(value) for value in reference_timescale]
    if len(values) != n_spectra:
        raise ValueError("reference_timescale must have one value per spectrum")
    return values
