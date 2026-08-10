"""Plots for SPOD temporal coefficients."""

# standard library packages
from math import isfinite
from typing import Any, Dict, Tuple

# third party packages
import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_spod_time_coefficients(
    time: pt.Tensor,
    frequency: pt.Tensor,
    coefficients: pt.Tensor,
    n_modes: int = 1,
    reference_timescale: float | None = None,
    positive_frequencies_only: bool = True,
    time_limits: tuple[float, float] | None = None,
    frequency_limits: tuple[float, float] | None = None,
    color_percentile: float = 99.0,
    cmap: Any = "viridis",
    colorbar: bool = True,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    mesh_kwargs: Dict[str, Any] | None = None,
    colorbar_kwargs: Dict[str, Any] | None = None,
) -> Tuple[Figure, Axes]:
    r"""Plot the L2 amplitude of SPOD time coefficients over frequency and time.

    ``coefficients`` must have shape
    ``(n_frequency, n_modes, n_snapshots)`` as returned by
    :meth:`flowtorch.analysis.AMSPOD.temporal_coefficients`. The displayed
    quantity is

    .. math::

        A(f,t) = \left(\sum_{i=1}^{n_\mathrm{modes}}
        |a_i(f,t)|^2\right)^{1/2}.

    The default ``n_modes=1`` therefore displays ``|a_1(f,t)|``. The upper
    color limit is estimated from a robust percentile of all finite displayed
    amplitudes. Explicit ``vmin`` or ``vmax`` entries in ``mesh_kwargs``
    override the automatically selected limits.

    If ``reference_timescale`` is supplied, the same scale is used for both
    axes: time is displayed as ``t / reference_timescale`` and frequency as
    the Strouhal number ``St = f * reference_timescale``. The function does
    not save or show the figure, and the returned objects remain modifiable.

    :param time: snapshot times
    :type time: pt.Tensor
    :param frequency: SPOD frequency-bin coordinates in Hz
    :type frequency: pt.Tensor
    :param coefficients: complex or real SPOD temporal coefficients
    :type coefficients: pt.Tensor
    :param n_modes: number of leading modes included in the L2 amplitude,
        defaults to 1
    :type n_modes: int, optional
    :param reference_timescale: common scale for nondimensional time and
        frequency
    :type reference_timescale: float, optional
    :param positive_frequencies_only: omit zero and negative frequencies,
        defaults to ``True``
    :type positive_frequencies_only: bool, optional
    :param time_limits: displayed time limits in dimensional input units
    :type time_limits: Tuple[float, float], optional
    :param frequency_limits: displayed frequency limits in Hz
    :type frequency_limits: Tuple[float, float], optional
    :param color_percentile: robust upper color-limit percentile, defaults to 99
    :type color_percentile: float, optional
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
    :return: modifiable Matplotlib figure and coefficient axes
    :rtype: Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

    **Examples**

    .. code-block:: python

        coefficients = spod.temporal_coefficients(n_modes=3)
        time = pt.arange(coefficients.shape[-1]) * dt
        figure, axes = plot_spod_time_coefficients(
            time,
            spod.frequency,
            coefficients,
            n_modes=3,
            reference_timescale=0.1,
        )
        axes.set_title(r"$\mathrm{SPOD\ coefficient\ amplitude}$")
    """
    _validate_inputs(
        time,
        frequency,
        coefficients,
        n_modes,
        reference_timescale,
        time_limits,
        frequency_limits,
        color_percentile,
        figsize,
    )
    mesh_options = {} if mesh_kwargs is None else mesh_kwargs.copy()
    bar_options = {} if colorbar_kwargs is None else colorbar_kwargs.copy()

    frequency_mask = pt.ones_like(frequency, dtype=pt.bool)
    if positive_frequencies_only:
        frequency_mask &= frequency > 0.0
    if frequency_limits is not None:
        frequency_mask &= frequency >= frequency_limits[0]
        frequency_mask &= frequency <= frequency_limits[1]
    time_mask = pt.ones_like(time, dtype=pt.bool)
    if time_limits is not None:
        time_mask &= time >= time_limits[0]
        time_mask &= time <= time_limits[1]
    if not bool(frequency_mask.any()):
        raise ValueError("frequency selection must contain at least one bin")
    if not bool(time_mask.any()):
        raise ValueError("time selection must contain at least one snapshot")

    selected_coefficients = coefficients[frequency_mask, :n_modes]
    selected_coefficients = selected_coefficients[..., time_mask]
    amplitude = pt.linalg.vector_norm(selected_coefficients, dim=1)
    selected_time = time[time_mask]
    selected_frequency = frequency[frequency_mask]
    if reference_timescale is None:
        horizontal = selected_time
        vertical = selected_frequency
        horizontal_label = r"$t\;[\mathrm{s}]$"
        vertical_label = r"$f\;[\mathrm{Hz}]$"
    else:
        horizontal = selected_time / reference_timescale
        vertical = selected_frequency * reference_timescale
        horizontal_label = r"$t/\tau$"
        vertical_label = r"$St$"

    finite_amplitude = amplitude[pt.isfinite(amplitude)]
    if finite_amplitude.numel() == 0:
        raise ValueError("selected coefficients must contain a finite value")
    robust_max = float(pt.quantile(finite_amplitude, color_percentile / 100.0).item())
    if robust_max <= 0.0:
        robust_max = float(pt.finfo(amplitude.dtype).eps)
    mesh_options.setdefault("shading", "nearest")
    mesh_options.setdefault("cmap", cmap)
    mesh_options.setdefault("vmin", 0.0)
    mesh_options.setdefault("vmax", robust_max)

    if ax is None:
        figure, axes = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        if figsize is not None:
            raise ValueError("figsize cannot be supplied with ax")
        axes = ax
        figure = axes.figure
    mesh = axes.pcolormesh(
        horizontal.detach().cpu().numpy(),
        vertical.detach().cpu().numpy(),
        amplitude.detach().cpu().numpy(),
        **mesh_options,
    )
    axes.set_xlabel(horizontal_label)
    axes.set_ylabel(vertical_label)
    axes.set_xlim(float(horizontal[0]), float(horizontal[-1]))
    axes.set_ylim(float(vertical[0]), float(vertical[-1]))
    if colorbar:
        bar_options.setdefault(
            "label",
            (
                r"$|a_1(f,t)|$"
                if n_modes == 1
                else rf"$\left(\sum_{{i=1}}^{{{n_modes}}}" r"|a_i(f,t)|^2\right)^{1/2}$"
            ),
        )
        figure.colorbar(mesh, ax=axes, **bar_options)
    return figure, axes


def _validate_inputs(
    time: pt.Tensor,
    frequency: pt.Tensor,
    coefficients: pt.Tensor,
    n_modes: int,
    reference_timescale: float | None,
    time_limits: tuple[float, float] | None,
    frequency_limits: tuple[float, float] | None,
    color_percentile: float,
    figsize: tuple[float, float] | None,
) -> None:
    if time.ndim != 1 or frequency.ndim != 1:
        raise ValueError("time and frequency must be one-dimensional")
    if not time.is_floating_point() or not frequency.is_floating_point():
        raise ValueError("time and frequency must have floating-point dtypes")
    if time.numel() < 2 or frequency.numel() < 1:
        raise ValueError("time needs two entries and frequency needs one entry")
    if coefficients.ndim != 3:
        raise ValueError(
            "coefficients must have shape (n_frequency, n_modes, n_snapshots)"
        )
    if coefficients.shape[0] != frequency.numel():
        raise ValueError("coefficient and frequency dimensions must match")
    if coefficients.shape[2] != time.numel():
        raise ValueError("coefficient and time dimensions must match")
    if not coefficients.is_floating_point() and not coefficients.is_complex():
        raise ValueError("coefficients must have a floating-point or complex dtype")
    if not isinstance(n_modes, int) or isinstance(n_modes, bool) or n_modes < 1:
        raise ValueError("n_modes must be a positive integer")
    if n_modes > coefficients.shape[1]:
        raise ValueError("n_modes cannot exceed the available coefficient modes")
    if reference_timescale is not None:
        if not isfinite(reference_timescale) or reference_timescale <= 0.0:
            raise ValueError("reference_timescale must be positive and finite")
    _validate_limits(time_limits, "time_limits")
    _validate_limits(frequency_limits, "frequency_limits")
    if not isfinite(color_percentile) or not 0.0 < color_percentile <= 100.0:
        raise ValueError("color_percentile must be in (0, 100]")
    if figsize is not None and (
        len(figsize) != 2
        or any(not isfinite(value) or value <= 0.0 for value in figsize)
    ):
        raise ValueError("figsize entries must be positive and finite")
    if not bool(pt.all(time[1:] > time[:-1])):
        raise ValueError("time must be strictly increasing")
    if frequency.numel() > 1 and not bool(pt.all(frequency[1:] > frequency[:-1])):
        raise ValueError("frequency must be strictly increasing")


def _validate_limits(limits: tuple[float, float] | None, name: str) -> None:
    if limits is not None and (
        len(limits) != 2
        or any(not isfinite(value) for value in limits)
        or limits[0] >= limits[1]
    ):
        raise ValueError(f"{name} must contain two increasing finite values")
