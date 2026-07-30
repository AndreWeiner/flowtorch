"""Utilities shared by multiple flowTorch subpackages."""

from typing import Tuple


def format_byte_size(size: int) -> Tuple[float, str]:
    """Convert a number of bytes into a human-readable value and unit."""
    exponent_labels = {0: "b", 1: "Kb", 2: "Mb", 3: "Gb", 4: "Tb", 5: "Pt"}
    exponent = 0
    converted_size = float(size)
    while converted_size > 1024:
        converted_size /= 1024
        exponent += 1
    return converted_size, exponent_labels[exponent]
