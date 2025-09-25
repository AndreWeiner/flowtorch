"""Collection of utilities related to data and dataloaders."""

# standard library packages
import logging
from os.path import exists
from os import sep
from typing import Tuple, List, Union

from time import time
from torch.nn import Parameter
from scipy.spatial import KDTree
from torch.optim import Adam, lr_scheduler
from torch import Tensor, tensor, float32, from_numpy, where, manual_seed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    force=True)

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


class CellVolumeEstimator:
    """
    Estimate cell volumes based on a point cloud of coordinates.

    The algorithm works in three main steps:
        1. Build a KD-tree to compute the mean distances between each point and its
           :math:`k` nearest neighbors.
        2. Approximate local volumes as the mean neighbor distance raised to the
           power of the dimension (:math:`d`).
        3. Optimize a scaling factor :math:`\\alpha` such that the sum of all
           estimated volumes matches the true bounding-box volume.

    Example
    -------
        >>> import torch
        >>> from flowtorch.data.utils import CellVolumeEstimator
        >>>
        >>> # Generate a synthetic point cloud inside a unit square
        >>> coords = torch.rand(100, 2)
        >>> estimator = CellVolumeEstimator(coords, k=4, lr=1e-2, max_iterations=1000)
        >>> estimator.estimate_cell_volume()
        >>>
        >>> # Access estimated volumes per point
        >>> print(estimator.weights.shape)
        >>> # Access the optimization loss history
        >>> print(estimator.loss[-5:])

    :param coordinates: Input tensor of coordinates with shape
        ``(n_points, n_dims)``.
    :type coordinates: torch.Tensor
    :param k: Number of nearest neighbors to use for distance estimation
        (default: 8 for 3D, 4 for 2D).
    :type k: int
    :param lr: Learning rate for optimizing :math:`\\alpha`.
    :type lr: float
    :param max_iterations: Maximum number of optimization iterations.
    :type max_iterations: int
    :param stop_at: Stopping criterion for relative error in optimization.
    :type stop_at: float
    :param n_workers: Number of parallel workers for KD-tree queries.
    :type n_workers: int
    """

    def __init__(self, coordinates: Tensor, k: int = 8, lr: float = 1e-2, max_iterations: int = 2500,
                 stop_at: float = 1e-8, n_workers: int = 4, seed: int = 0):
        self._vertices = coordinates
        self._n_workers = n_workers

        # ensure reproducibility
        manual_seed(seed)

        # get the dimensions which are not zero, since e.g. in OpenFOAM we have 3D coordinates even for 2D simulations
        _tmp = (self._vertices == self._vertices[0]).all(dim=0)
        self._dims = where(_tmp == False)[0]
        self._volume_original = (coordinates.max(dim=0).values -
                                 coordinates.min(dim=0).values)[self._dims].prod().type(float32)

        # optimization
        self._k = k if len(self._dims) == 3 else 4
        self._max_steps = max_iterations if max_iterations > 0 else 1
        self._stop = stop_at
        self._alpha = Parameter(tensor(2, dtype=float32))
        self._optimizer = Adam([self._alpha], lr=lr, weight_decay=1e-3)
        self._scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=self._optimizer, min_lr=1e-4, patience=10)

        # initialization
        self._n_dims = self._vertices.size(1)
        self._sum_weights = None
        self._time_start = 0

        # public
        self.weights = None
        self.loss = []

    def estimate_cell_volume(self) -> None:
        """
        Main entry point for estimating cell volumes.

        Runs the full pipeline:
            - Computes mean distances to neighbors.
            - Optimizes scaling factor :math:`\\alpha`.
            - Scales the weights accordingly.
            - Prints information summary to the logger.
        """
        self._time_start = time()
        self._compute_mean_distance()
        self._optimize_alpha()
        self.weights *= self._alpha.detach()
        self._print_info()

    def _compute_mean_distance(self) -> None:
        """
        Compute the mean distances between points using a KD-tree.

        The mean neighbor distance is then raised to the power of the
        dimensionality to obtain an initial approximation of local cell volumes.

        :return: None
        """
        logger.info("Computing mean distances between the coordinates.")
        # TODO: large memory requirement -> how can we get this cheaper?
        tree = KDTree(self._vertices)
        dists, _ = tree.query(self._vertices, k=self._k, workers=self._n_workers)

        # free up some memory
        self._vertices = None

        # mean distance to nearest neighbor
        self.weights = from_numpy(dists.mean(axis=1)) ** self._n_dims
        self._sum_weights = self.weights.sum().type(float32)
        logger.info("Done.")

    def _optimize_alpha(self) -> None:
        """
        Optimize the scaling factor :math:`\\alpha` such that the total sum of
        estimated volumes matches the original bounding-box volume.

        :return: None
        """
        logger.info("Optimizing cell volumes.")
        converged = False
        for step in range(self._max_steps):
            self._optimizer.zero_grad()
            loss = abs(self._alpha * self._sum_weights - self._volume_original) / self._volume_original
            loss.backward()
            self._optimizer.step()
            self._scheduler.step(loss)

            if step % 100 == 0:
                logger.info(f"Epoch {step:04d}, alpha = {self._alpha.item():.3f}, loss = {loss.item():.4e}")

            if loss.item() < self._stop:
                logger.info(f"Found optimal alpha = {self._alpha.item():.3f} after {step + 1} iterations with an "
                            f"error of {loss.item():.3e}")
                converged = True
                break
            self.loss.append(loss.detach().item())

        if not converged:
            logger.warning(f"Optimization of alpha did not converge after {step + 1} iterations. "
                           f"Current alpha = {self._alpha.item():.3f}")

    def _print_info(self) -> None:
        """
        Print a summary of the volume estimation before and after optimization.

        The summary includes:
            - Bounding box volume
            - Approximated volume before optimization
            - Approximated volume after optimization
            - Min/max local cell volumes
            - Total runtime

        :return: None
        """
        msg = f"""

                Overall cell volume estimated from coordinates:\t{self._volume_original.item():.5f}
                Overall cell volume estimated before optimization:\t{self._sum_weights.item():.5f}
                \t\t -> approximation of {self._sum_weights / self._volume_original.item() * 100:.3f} %.
                Overall cell volume estimated after optimization:\t{self._alpha * self._sum_weights.item():.5f}
                \t\t -> approximation of {self._sum_weights * self._alpha / self._volume_original.item() * 100:.3f} %.
                Computed cell volumes (min. / max.):\t{self.weights.min().item():.3e}, {self.weights.max().item():.3e}

                Approximation took {time() - self._time_start:.3f} s.
        """
        logger.info(msg)

if __name__ == "__main__":
    pass
