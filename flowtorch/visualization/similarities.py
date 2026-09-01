"""Plots for mode similarity across frequency spectra."""

# standard library packages
from math import isfinite
from typing import Any, Dict, Literal, Tuple

# third party packages
import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_mode_similarity(
    first_frequency: pt.Tensor,
    second_frequency: pt.Tensor,
    similarity: pt.Tensor,
    reference_timescale: float | None = None,
    triangle: Literal["auto", "full", "lower"] = "auto",
    ax: Axes | None = None,
    show_colorbar: bool = True,
    cmap: Any = "viridis",
    mesh_kwargs: Dict[str, Any] | None = None,
) -> Tuple[Figure, Axes]:
    r"""Plot a frequency-by-frequency mode similarity matrix.

    The first frequency coordinate is shown on the horizontal axis and the
    second on the vertical axis. Both coordinates are sorted in ascending
    order for display, including the wrapped ordering produced by
    :func:`torch.fft.fftfreq` for complex signals.

    With ``triangle="auto"``, square symmetric matrices defined on matching
    frequency coordinates are displayed using only their lower triangle and
    diagonal. ``triangle="lower"`` requests this behavior explicitly and
    raises an error when the matrix or coordinates are not symmetric.
    Unavailable values represented by NAN are masked.

    :param first_frequency: frequency coordinate for the first mode set
    :type first_frequency: pt.Tensor
    :param second_frequency: frequency coordinate for the second mode set
    :type second_frequency: pt.Tensor
    :param similarity: similarity matrix arranged as
        ``(n_first_frequency, n_second_frequency)``
    :type similarity: pt.Tensor
    :param reference_timescale: reference time used to convert frequency to
        Strouhal number
    :type reference_timescale: float, optional
    :param triangle: display the full matrix, its lower triangle, or select
        automatically based on symmetry; defaults to ``"auto"``
    :type triangle: Literal["auto", "full", "lower"], optional
    :param ax: existing Matplotlib axes; a new figure is created when omitted
    :type ax: matplotlib.axes.Axes, optional
    :param show_colorbar: add a similarity colorbar, defaults to ``True``
    :type show_colorbar: bool, optional
    :param cmap: Matplotlib colormap, defaults to ``"viridis"``
    :param mesh_kwargs: additional ``Axes.pcolormesh`` options
    :type mesh_kwargs: Dict[str, Any], optional
    :raises ValueError: for incompatible coordinates, similarity values, or an
        invalid triangle selection
    :return: modifiable Matplotlib figure and axes
    :rtype: Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    _validate_similarity_plot_inputs(
        first_frequency,
        second_frequency,
        similarity,
        reference_timescale,
        triangle,
    )
    first = first_frequency.detach().cpu()
    second = second_frequency.detach().cpu()
    values = similarity.detach().cpu()
    first_order = pt.argsort(first)
    second_order = pt.argsort(second)
    first = first[first_order]
    second = second[second_order]
    values = values[first_order][:, second_order]
    if reference_timescale is not None:
        first = first * reference_timescale
        second = second * reference_timescale

    symmetric = _is_symmetric(first, second, values)
    show_lower = triangle == "lower" or (triangle == "auto" and symmetric)
    if triangle == "lower" and not symmetric:
        raise ValueError(
            "triangle='lower' requires matching coordinates and a symmetric matrix"
        )

    display_values = values.T
    display_invalid = ~pt.isfinite(display_values)
    if show_lower:
        display_invalid = pt.logical_or(
            display_invalid,
            pt.triu(pt.ones_like(display_values, dtype=pt.bool), diagonal=1),
        )
    masked_values = np.ma.masked_where(display_invalid.numpy(), display_values.numpy())

    if ax is None:
        figure, axes = plt.subplots()
    else:
        axes = ax
        figure = axes.figure
    options = {} if mesh_kwargs is None else mesh_kwargs.copy()
    options.setdefault("shading", "auto")
    options.setdefault("vmin", 0.0)
    options.setdefault("vmax", 1.0)
    color_map = plt.get_cmap(cmap).copy()
    color_map.set_bad(alpha=0.0)
    options.setdefault("cmap", color_map)
    mesh = axes.pcolormesh(first.numpy(), second.numpy(), masked_values, **options)

    if reference_timescale is None:
        axes.set_xlabel(r"$f_1\;[\mathrm{Hz}]$")
        axes.set_ylabel(r"$f_2\;[\mathrm{Hz}]$")
    else:
        axes.set_xlabel(r"$St_1$")
        axes.set_ylabel(r"$St_2$")
    if show_lower:
        axes.set_aspect("equal")
    if show_colorbar:
        colorbar = figure.colorbar(mesh, ax=axes)
        colorbar.set_label(r"$\rho$")
    return figure, axes


def _is_symmetric(
    first_coordinate: pt.Tensor,
    second_coordinate: pt.Tensor,
    values: pt.Tensor,
) -> bool:
    """Check whether coordinates and values define a symmetric matrix."""
    if values.shape[0] != values.shape[1]:
        return False
    if first_coordinate.shape != second_coordinate.shape:
        return False
    coordinate_dtype = pt.promote_types(first_coordinate.dtype, second_coordinate.dtype)
    coordinates_match = pt.allclose(
        first_coordinate.to(coordinate_dtype),
        second_coordinate.to(coordinate_dtype),
    )
    values_match = pt.allclose(values, values.T, equal_nan=True)
    return bool(coordinates_match and values_match)


def _validate_similarity_plot_inputs(
    first_frequency: pt.Tensor,
    second_frequency: pt.Tensor,
    similarity: pt.Tensor,
    reference_timescale: float | None,
    triangle: str,
) -> None:
    """Validate frequency coordinates and a mode similarity matrix."""
    if first_frequency.ndim != 1 or second_frequency.ndim != 1:
        raise ValueError("frequency coordinates must be one-dimensional")
    if similarity.ndim != 2:
        raise ValueError("similarity must be a two-dimensional tensor")
    expected = (first_frequency.shape[0], second_frequency.shape[0])
    if similarity.shape != expected:
        raise ValueError(f"similarity must have shape {expected}")
    if pt.is_complex(similarity):
        raise ValueError("similarity must be real-valued")
    if not bool(pt.isfinite(first_frequency).all()) or not bool(
        pt.isfinite(second_frequency).all()
    ):
        raise ValueError("frequency coordinates must be finite")
    finite_values = similarity[pt.isfinite(similarity)]
    if finite_values.numel() > 0 and (
        bool((finite_values < 0.0).any()) or bool((finite_values > 1.0).any())
    ):
        raise ValueError("finite similarity values must lie in the range [0, 1]")
    if reference_timescale is not None:
        if not isfinite(reference_timescale) or reference_timescale <= 0.0:
            raise ValueError("reference_timescale must be finite and positive")
    if triangle not in ("auto", "full", "lower"):
        raise ValueError("triangle must be 'auto', 'full', or 'lower'")
