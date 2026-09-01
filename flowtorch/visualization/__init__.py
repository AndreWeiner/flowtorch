"""Visualization algorithms and plotting tools."""

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "plot_spod_time_coefficients": (
        "coefficients",
        "plot_spod_time_coefficients",
    ),
    "plot_adaptive_residual": ("residuals", "plot_adaptive_residual"),
    "plot_adaptive_residuals": ("residuals", "plot_adaptive_residuals"),
    "animate_line_integral_convolution": (
        "animations",
        "animate_line_integral_convolution",
    ),
    "animate_scalar_field": ("animations", "animate_scalar_field"),
    "line_integral_convolution": ("lic", "line_integral_convolution"),
    "plot_scalar_fields": ("fields", "plot_scalar_fields"),
    "plot_vector_fields": ("fields", "plot_vector_fields"),
    "plot_spod_mode_2d": ("modes", "plot_spod_mode_2d"),
    "plot_spod_modes_2d": ("modes", "plot_spod_modes_2d"),
    "plot_mode_similarity": ("similarities", "plot_mode_similarity"),
    "plot_spod_spectra": ("spectra", "plot_spod_spectra"),
    "plot_spod_spectrum": ("spectra", "plot_spod_spectrum"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Load a public visualization object when it is first accessed."""
    try:
        module_name, object_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(f".{module_name}", __name__), object_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
