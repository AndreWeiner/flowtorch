"""Collection of utilities related to data and dataloaders."""

# standard library packages
from os.path import exists
from os import sep
from typing import Tuple, List, Union


def format_byte_size(size: int) -> Tuple[float, str]:
    """Convert a number of bytes into human-readable format.

    The function is based on `this <https://stackoverflow.com/questions/12523586/python-format-size-application-converting-b-to-kb-mb-gb-tb>` Stackoverflow question.

    :param size: size in bytes
    :type size: int
    :return: converted size corresponding unit
    :rtype: tuple(float, str)

    """
    exponent_labels = {0: "b", 1: "Kb", 2: "Mb", 3: "Gb", 4: "Tb", 5: "Pt"}
    exponent = 0
    while size > 1024:
        size /= 1024
        exponent += 1
    return size, exponent_labels[exponent]


def check_and_standardize_path(path: str, folder: bool = True):
    """Check if the path exists and remove trailing slash if present.

    :param path: path to folder or file
    :type path: str
    :param folder: True if path points to folder; False if path points to file
    :type folder: bool
    :return: standardized path to file or folder
    :rtype: str

    """
    if exists(path):
        if folder and path[-1] == sep:
            return path[:-1]
        else:
            return path
    else:
        raise ValueError(f"Could not find {path}")


def check_list_or_str(arg_value: Union[List[str], str], arg_name: str):
    """Check if argument is of type list or string.

    If the input is a list, an additional check is performed to ensure that
    the list has at list one entry and that all entries are strings.

    :param arg_value: object to perform the check on
    :type arg_value: Union[List[str], str]
    :param arg_name: additional argument name to provide informative error message
    :param arg_name: str

    """
    message = f"Argument {arg_name} must be a string or a list of strings."
    if isinstance(arg_value, list):
        if len(arg_value) < 1:
            raise ValueError(f"The list {arg_name} must not be empty.")
        if not all([isinstance(arg, str) for arg in arg_value]):
            raise ValueError(message)
    else:
        if not isinstance(arg_value, str):
            raise ValueError(message)
