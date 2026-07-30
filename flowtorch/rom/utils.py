"""Collection of utilities for reduced-order modeling."""

# standars library packages
from time import time
import functools
from typing import Callable, Union, List

# third party libraries
import numpy as np


def log_time(func) -> dict:
    """Measure and log a function's execution time.

    :param func: function to be executed; the function is expected
        to return a dictionary
    :type func: Callable
    :return: dictionary returned by the wrapped function with additional
        entry for execution time
    :rtype: dict
    """

    @functools.wraps(func)
    def measure_time(*args, **kwargs) -> dict:
        start_time = time()
        log = func(*args, **kwargs)
        return {**log, "execution_time": time() - start_time}

    return measure_time


def check_larger_than(value: Union[int, float], limit: Union[int, float], name: str):
    """Check if a scalar value is larger than a given lower limit.

    :param value: scalar value to check
    :type value: Union[int, float]
    :param value: lower limit to check against
    :type value: Union[int, float]
    :param name: name of the parameter
    :type name: str
    :raises ValueError: if the argument is less than or equal
        to the lower limit
    """
    if value <= limit:
        raise ValueError(f"The argument for {name} must be larger than {limit}")


def check_int_larger_than(value: int, limit: int, name: str):
    """Check if input is an integer larger than a given lower limit.

    :param value: input value to check
    :type value: int
    :param limit: the value must be larger than the limit
    :type limit: int
    :param name: name of the parameter
    :type name: str
    :raises ValueError: if the argument is not an integer
    """
    message = f"The argument of {name} must be an integer larger than {limit}"
    if not isinstance(value, int):
        raise ValueError(message)
    check_larger_than(value, limit, name)


def remove_sequential_duplicates(sequence: np.ndarray) -> np.ndarray:
    """Get sequence of integers without sequential duplicates.

    :param sequence: input sequence to check
    :type sequence: np.ndarray
    :return: sequence without sequential duplicates
    :rtype: np.ndarray
    """
    is_different = np.diff(sequence).astype(bool)
    return sequence[np.insert(is_different, 0, True)]
