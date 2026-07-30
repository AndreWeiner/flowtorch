"""Global constants for all subpackages."""

# standard library packages
from os import environ, listdir
from os.path import isdir, join
from typing import Dict

# third party packages
from torch import float32


def find_datasets(path: str) -> Dict[str, str]:
    """Find all available datasets/folders in a given location.

    If the path does not exist, an empty dictionary is returned.

    :param path: path to the folder containing the datasets
    :type path: str
    :return: dictionary with the keys being the folder names holding
        the data and the values being the full paths to the datasets
    :rtype: Dict[str, str]
    """
    if not isinstance(path, str):
        return dict()
    if isdir(path):
        datasets = listdir(path)
        dataset_paths = [join(path, folder) for folder in datasets]
        return dict(zip(datasets, dataset_paths))
    else:
        return dict()


# default data type for all PyTorch tensors
DEFAULT_DTYPE = float32
# default tolerance for comparison of floating point numbers
FLOAT_TOLERANCE = 1.0e-7
# path to flowTorch datasets
DATASET_PATH = environ.get("FLOWTORCH_DATASETS", "")
# available datasets
DATASETS = find_datasets(DATASET_PATH)
