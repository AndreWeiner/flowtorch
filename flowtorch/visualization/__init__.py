"""Visualization algorithms and plotting tools."""

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "line_integral_convolution": ("lic", "line_integral_convolution"),
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
