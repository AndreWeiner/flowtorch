"""Propagation quantities inferred from complex harmonic spatial modes."""

# standard library packages
from math import isfinite, pi
from numbers import Real
from typing import Tuple

# third party packages
import torch as pt

# flowTorch packages
from flowtorch.data.differential_tools import (
    _gaussian_smooth_2d,
    _standardize_sigma,
    curvilinear_gradient,
)


def curvilinear_surface_phase_velocity(
    modes: pt.Tensor,
    frequencies: float | pt.Tensor,
    x: pt.Tensor,
    y: pt.Tensor,
    z: pt.Tensor | None = None,
    amplitude_threshold: float = 0.05,
    min_wavenumber: float = 0.0,
    smoothing_sigma: float | Tuple[float, float] | None = None,
    smoothing_mode: str = "reflect",
    smoothing_truncate: float = 3.0,
    edge_order: int = 2,
    metric_tolerance: float = 1.0e-12,
) -> pt.Tensor:
    r"""Infer local phase velocity on a structured curvilinear surface.

    A single complex mode has shape ``(nx, ny)``. Multiple modes have shape
    ``(nx, ny, n_modes)`` and must be accompanied by a one-dimensional
    frequency tensor containing exactly one frequency per mode. A scalar
    frequency is accepted only for a single mode.

    For a mode :math:`\phi=a+ib=A\exp(i\theta)`, the phase gradient is

    .. math::

        \nabla\theta = \frac{a\nabla b-b\nabla a}{a^2+b^2}.

    Assuming time dependence :math:`\exp(i 2\pi f t)`, the local phase
    velocity normal to the phase fronts is

    .. math::

        \boldsymbol{c}_p = -2\pi f
        \frac{\nabla\theta}{\lVert\nabla\theta\rVert^2}.

    The result has shape ``(nx, ny, 2)`` for a planar grid and
    ``(nx, ny, 3)`` for a surface embedded using ``z``. For multiple modes,
    the mode axis precedes the final vector-component axis. The result is a
    surface-tangential normal phase velocity, not a material or group
    velocity, and this function does not support three-dimensional volume
    grids.

    Real and imaginary parts are smoothed consistently before differentiation.
    Locations with weak modal amplitude, insufficient wavenumber, singular
    grid metrics, or non-finite intermediate values are filled with NAN.

    :param modes: one complex mode with shape ``(nx, ny)`` or multiple modes
        with shape ``(nx, ny, n_modes)``
    :type modes: pt.Tensor
    :param frequencies: scalar nonzero frequency for a single mode or one
        nonzero frequency per mode with shape ``(n_modes,)``
    :type frequencies: float or pt.Tensor
    :param x: first Cartesian coordinate on the structured surface
    :type x: pt.Tensor
    :param y: second Cartesian coordinate on the structured surface
    :type y: pt.Tensor
    :param z: optional third Cartesian coordinate of an embedded surface
    :type z: pt.Tensor, optional
    :param amplitude_threshold: mask amplitudes below this fraction of each
        mode's maximum amplitude; defaults to 0.05
    :type amplitude_threshold: float, optional
    :param min_wavenumber: mask phase-gradient magnitudes no larger than this
        value; defaults to 0
    :type min_wavenumber: float, optional
    :param smoothing_sigma: Gaussian width in grid-index units, applied to real
        and imaginary parts before differentiation
    :type smoothing_sigma: float or Tuple[float, float], optional
    :param smoothing_mode: Gaussian boundary mode; ``"reflect"``,
        ``"replicate"``, or ``"circular"``
    :type smoothing_mode: str, optional
    :param smoothing_truncate: Gaussian kernel radius in multiples of sigma
    :type smoothing_truncate: float, optional
    :param edge_order: finite-difference boundary accuracy, either 1 or 2
    :type edge_order: int, optional
    :param metric_tolerance: relative threshold for singular grid metrics
    :type metric_tolerance: float, optional
    :raises ValueError: for invalid mode, frequency, grid, or option inputs
    :return: local phase-velocity vectors
    :rtype: pt.Tensor
    """
    single_mode = _validate_modes(modes, x)
    mode_sequence = modes.unsqueeze(-1) if single_mode else modes
    frequency_sequence = _prepare_frequencies(
        frequencies, mode_sequence.shape[-1], single_mode, modes
    )
    _validate_options(
        amplitude_threshold,
        min_wavenumber,
        smoothing_sigma,
        smoothing_mode,
        smoothing_truncate,
    )

    real = _smooth_mode_part(
        mode_sequence.real, smoothing_sigma, smoothing_mode, smoothing_truncate
    )
    imaginary = _smooth_mode_part(
        mode_sequence.imag, smoothing_sigma, smoothing_mode, smoothing_truncate
    )
    gradient_real = curvilinear_gradient(
        real,
        x,
        y,
        z,
        edge_order=edge_order,
        metric_tolerance=metric_tolerance,
    )
    gradient_imaginary = curvilinear_gradient(
        imaginary,
        x,
        y,
        z,
        edge_order=edge_order,
        metric_tolerance=metric_tolerance,
    )

    amplitude_squared = real.square() + imaginary.square()
    nonzero_amplitude = amplitude_squared > pt.finfo(real.dtype).tiny
    safe_amplitude_squared = pt.where(
        nonzero_amplitude, amplitude_squared, pt.ones_like(amplitude_squared)
    )
    phase_gradient = (
        real.unsqueeze(-1) * gradient_imaginary
        - imaginary.unsqueeze(-1) * gradient_real
    ) / safe_amplitude_squared.unsqueeze(-1)

    amplitude = amplitude_squared.sqrt()
    maximum_amplitude = amplitude.amax(dim=(0, 1), keepdim=True)
    sufficient_amplitude = amplitude >= amplitude_threshold * maximum_amplitude
    sufficient_amplitude = pt.logical_and(sufficient_amplitude, nonzero_amplitude)
    wavenumber_squared = phase_gradient.square().sum(dim=-1)
    sufficient_wavenumber = wavenumber_squared.sqrt() > min_wavenumber
    finite = pt.logical_and(
        pt.isfinite(phase_gradient).all(dim=-1), pt.isfinite(wavenumber_squared)
    )
    valid = pt.logical_and(sufficient_amplitude, sufficient_wavenumber)
    valid = pt.logical_and(valid, finite)
    safe_wavenumber_squared = pt.where(
        valid, wavenumber_squared, pt.ones_like(wavenumber_squared)
    )

    angular_frequency = 2.0 * pi * frequency_sequence.reshape(1, 1, -1, 1)
    velocity = (
        -angular_frequency * phase_gradient / safe_wavenumber_squared.unsqueeze(-1)
    )
    velocity = velocity.masked_fill(~valid.unsqueeze(-1), float("nan"))
    return velocity[:, :, 0] if single_mode else velocity


def _validate_modes(modes: pt.Tensor, x: pt.Tensor) -> bool:
    """Validate mode shape and values and return the single-mode flag."""
    if modes.ndim not in (2, 3) or modes.shape[:2] != x.shape:
        raise ValueError(
            "modes must have shape (nx, ny) or (nx, ny, n_modes) matching x"
        )
    if modes.ndim == 3 and modes.shape[-1] < 1:
        raise ValueError("modes must contain at least one mode")
    if not pt.is_complex(modes):
        raise ValueError("modes must have a complex dtype")
    if not bool(pt.isfinite(modes).all()):
        raise ValueError("modes must contain only finite values")
    return modes.ndim == 2


def _prepare_frequencies(
    frequencies: float | pt.Tensor,
    n_modes: int,
    single_mode: bool,
    reference: pt.Tensor,
) -> pt.Tensor:
    """Validate and standardize one frequency per mode."""
    if single_mode:
        if isinstance(frequencies, pt.Tensor):
            if frequencies.ndim != 0 or pt.is_complex(frequencies):
                raise ValueError("frequencies must be scalar for a single mode")
        elif not isinstance(frequencies, Real) or isinstance(frequencies, bool):
            raise ValueError("frequencies must be scalar for a single mode")
        frequency_sequence = pt.as_tensor(
            frequencies, dtype=reference.real.dtype, device=reference.device
        ).reshape(1)
    else:
        if not isinstance(frequencies, pt.Tensor) or frequencies.ndim != 1:
            raise ValueError(
                "frequencies must be a one-dimensional tensor for multiple modes"
            )
        if pt.is_complex(frequencies):
            raise ValueError("frequencies must be real-valued")
        if frequencies.shape[0] != n_modes:
            raise ValueError("frequencies must contain exactly one value per mode")
        frequency_sequence = frequencies.to(
            dtype=reference.real.dtype, device=reference.device
        )
    if not bool(pt.isfinite(frequency_sequence).all()):
        raise ValueError("frequencies must contain only finite values")
    if bool((frequency_sequence == 0.0).any()):
        raise ValueError("frequencies must be nonzero")
    return frequency_sequence


def _validate_options(
    amplitude_threshold: float,
    min_wavenumber: float,
    smoothing_sigma: float | Tuple[float, float] | None,
    smoothing_mode: str,
    smoothing_truncate: float,
) -> None:
    """Validate masking and smoothing options."""
    if not isfinite(amplitude_threshold) or not 0.0 <= amplitude_threshold <= 1.0:
        raise ValueError("amplitude_threshold must be in the interval [0, 1]")
    if not isfinite(min_wavenumber) or min_wavenumber < 0.0:
        raise ValueError("min_wavenumber must be finite and non-negative")
    if smoothing_mode not in ("reflect", "replicate", "circular"):
        raise ValueError('smoothing_mode must be "reflect", "replicate", or "circular"')
    if not isfinite(smoothing_truncate) or smoothing_truncate <= 0.0:
        raise ValueError("smoothing_truncate must be finite and positive")
    sigma_i, sigma_j = _standardize_sigma(smoothing_sigma)
    if not isfinite(sigma_i) or not isfinite(sigma_j):
        raise ValueError("smoothing sigma values must be finite")
    if sigma_i < 0.0 or sigma_j < 0.0:
        raise ValueError("smoothing sigma values must be non-negative")


def _smooth_mode_part(
    values: pt.Tensor,
    smoothing_sigma: float | Tuple[float, float] | None,
    smoothing_mode: str,
    smoothing_truncate: float,
) -> pt.Tensor:
    """Smooth one real-valued part of a complex mode sequence."""
    return _gaussian_smooth_2d(
        values, smoothing_sigma, smoothing_mode, smoothing_truncate
    )
