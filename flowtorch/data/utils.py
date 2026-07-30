"""Collection of utilities related to data and dataloaders."""

# standard library packages
import logging
from os.path import exists
from os import sep
from typing import Tuple, List, Union

from scipy.spatial import KDTree, ConvexHull
from torch import Tensor, float32, from_numpy, where

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)


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
    converted_size = float(size)
    while converted_size > 1024:
        converted_size /= 1024
        exponent += 1
    return converted_size, exponent_labels[exponent]


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
    the list has at least one entry and that all entries are strings.

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


class CellWeightEstimator:
    """
    Estimate cell volumes based on a point cloud of coordinates.

    The algorithm works in three main steps:
        1. Build a KD-tree to compute the mean distances between each point and its
           :math:`k` nearest neighbors.
        2. Approximate local volumes as the mean neighbor distance raised to the
           power of the dimension (:math:`d`).
        3. Calculate a scaling factor :math:`\\alpha` such that the sum of all
           estimated volumes matches the true bounding-box volume.
        4. Normalization of weights to [0, 1] using min-max scaling and :math:\\sqrt(weights) (optional)

    Example
    -------
        >>> import torch
        >>> from flowtorch.data.utils import CellWeightEstimator
        >>>
        >>> # Generate a synthetic point cloud inside a unit square
        >>> coords = torch.rand(100, 2)
        >>> estimator = CellWeightEstimator(coords, n_workers=4, normalize=True)
        >>>
        >>> # Access estimated volumes per point
        >>> print(estimator.weights.shape)
        >>> # Access unscaled weights
        >>> estimator.normalize = False
        >>> print(estimator.weights.shape)


    :param coordinates: Input tensor of coordinates with shape
        ``(n_points, n_dims)``.
    :type coordinates: torch.Tensor
    :param k: Number of nearest neighbors to use for distance estimation
        (default: 8 for 3D, 4 for 2D).
    :type k: Union[int, None]
    :param n_workers: Number of parallel workers for KD-tree queries.
    :type n_workers: int
    :param normalize: Normalize the computed volumes to [0, 1].
    :type normalize: bool
    """

    def __init__(
        self,
        coordinates: Tensor,
        n_workers: int = 4,
        normalize: bool = True,
        k: Union[int, None] = None,
    ):
        self._normalize = normalize
        self._vertices = coordinates
        self._n_workers = n_workers

        # get the dimensions which are not zero, since e.g. in OpenFOAM we have 3D coordinates even for 2D simulations
        _tmp = (self._vertices == self._vertices[0]).all(dim=0)
        self._dims = where(_tmp == False)[0]

        # compute the overall volume of the domain
        self._hull = ConvexHull(self._vertices[:, self._dims])
        self._volume_original = self._hull.volume

        # distances
        if k is not None:
            self._k = k
        else:
            self._k = 8 if len(self._dims) == 3 else 4

        # initialization
        self._n_dims = self._vertices.size(1)
        self._sum_weights = None
        self._time_start = 0
        self._max_sqrt = 1

        # compute the mean distance for each point
        self._compute_mean_distance()
        self._alpha = self._volume_original / self._dist.sum()
        self._max_sqrt = (self._alpha * self._dist).sqrt().max()

    def _compute_mean_distance(self) -> None:
        """
        Compute the mean distances between points using a KD-tree.

        The mean neighbor distance is then raised to the power of the
        dimensionality to obtain an initial approximation of local cell volumes.

        :return: None
        :rtype: None
        """
        logger.info("Computing mean distances between the coordinates.")
        # TODO: - large memory requirement -> how can we get this cheaper?
        tree = KDTree(self._vertices)
        dists, _ = tree.query(self._vertices, k=self._k, workers=self._n_workers)

        # free up some memory
        self._vertices = None

        # mean distance to nearest neighbor
        self._dist = from_numpy(dists.mean(axis=1)) ** self._n_dims
        self._sum_weights = self._dist.sum().type(float32)
        logger.info("Done.")

    def _normalize_weights(self) -> Tensor:
        """
        Normalize weights by the maximum square root value.

        :return: None
        :rtype; None
        """
        return (self._alpha * self._dist).sqrt() / self._max_sqrt

    @property
    def alpha(self) -> float:
        """
        Returns the scaling factor :math:`\\alpha` such that the total sum of
        estimated volumes matches the convex hull of the original volume.

        :return: None
        :rtype: float
        """
        return self._alpha

    @property
    def weights(self) -> Tensor:
        """
        Returns the estimated cell volumes. If ``normalize = True``, the
        square roots of the volumes are normalized by the maximum square root
        value.

        :return: Estimated cell volume, either the absolute value or scaled with the maximum value.
        :rtype: Tensor
        """
        if self._normalize:
            return self._normalize_weights()
        else:
            return self.alpha * self._dist

    @property
    def normalize(self) -> bool:
        """
        Get the current status for normalization

        :return: Flag if normalization is turned on or off
        :rtype: bool
        """
        return self._normalize

    @normalize.setter
    def normalize(self, value: bool) -> None:
        """
        Set a new value for normalizing the cell volumes.

        :param value: bool
        :return: None
        """
        self._normalize = value

    @property
    def max_sqrt(self) -> Tensor:
        """
        Get the max. value of the sqrt() of the estimated cell volumes.

        :return: max(sqrt(cell volumes))
        :rtype: Tensor
        """
        return self._max_sqrt


if __name__ == "__main__":
    pass
