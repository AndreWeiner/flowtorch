"""Implementation of multivariate singular spectrum analysis (MSSA).

The MSSA may be used as a preprocessing tool to reduce noise in time
series data. For large state vectors, the memory requirements due to
time delay embedding can be reduced by projecting the data onto a
truncated POD basis before performing MSSA. This variant is referred to
as projected MSSA (PMSSA). The implementation is based on the work_
of Ohmichi et al.

.. _work: https://doi.org/10.1007/s00348-022-03523-5
"""

import torch as pt
from flowtorch.provenance import ProvenanceMixin, tracked
from .svd import SVD
from .hodmd import _create_time_delays


class MSSA(ProvenanceMixin):
    """Multivariate singular spectrum analysis (MSSA).

    Examples

    >>> import torch as pt
    >>> from flowtorch.analysis import MSSA
    >>> dm = pt.rand((10, 100))
    >>> mssa = MSSA(dm, rank=20)
    >>> rec = mssa.reconstruction
    >>> err = mssa.reconstruction_error.norm() / dm.norm()
    """

    def _effective_settings(self) -> dict:
        settings = {
            "window_size": self._window_size,
            "delay": self._delay,
            "rows": self._rows,
            "cols": self._cols,
            "hankel_svd": self._svd,
        }
        if "_svd_dr" in vars(self):
            settings["projection_svd"] = self._svd_dr
        return settings

    @tracked
    def __init__(
        self, data_matrix: pt.Tensor, window_size: int = None, rank: int = None
    ):
        """Create a MSSA instance.

        :param data_matrix: data matrix whose columns are formed by the individual snapshots
        :type data_matrix: pt.Tensor
        :param window_size: window size (number of snapshots) to shift over the full snapshot
            sequence; the window size is inversely proportional to the number of time delays,
            i.e., D = N - L +1 (D - delays, N - number of snapshots, L - window size); if L is
            not specified, N // 2 is chosen as default
        :type window_size: int, optional
        :param rank: truncation rank for the SVD of the Hankel matrix; if unspecified, a suitable
            value is determined via Optimal Singular Value Hard Thresholding (SVHT)
        :type rank: int, optional
        """
        self._dm = data_matrix
        self._rows, self._cols = data_matrix.shape
        if window_size is None:
            self._window_size = self._cols // 2
        else:
            self._window_size = min(window_size, self._cols - 1)
        self._delay = self._cols - self._window_size + 1
        self._svd = SVD(_create_time_delays(data_matrix, self._delay, 1), rank)

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def delay(self) -> int:
        return self._delay

    @property
    def reconstruction(self) -> pt.Tensor:
        """Compute reconstruction of the original data matrix.

        In the rank-r approximation of the Hankel matrix, multiple instances
        of individual snapshots are reconstructed. To reconstruct the original
        data matrix, averaging over the anti-diagonals is performed.

        :return: rank-r reconstruction of the original data matrix
        :rtype: pt.Tensor
        """
        rec = self._svd.reconstruct()
        m, n, d, l = self._rows, self._cols, self._delay, self._window_size
        rec_mean = pt.zeros(m * n, dtype=rec.dtype)
        count = pt.zeros(m * n, dtype=pt.int64)
        for i in range(l):
            rec_mean[m * i : m * (i + d)] += rec[:, i]
            count[m * i : m * (i + d)] += 1
        return pt.vstack((rec_mean / count).split(m)).T

    @property
    def reconstruction_error(self) -> pt.Tensor:
        """Compute elementwise reconstruction error.

        :return: reconstruction error with one value for each element in the
            data matrix
        :rtype: pt.Tensor
        """
        return self._dm - self.reconstruction

    @property
    def svd(self) -> SVD:
        """SVD of the Hankel matrix.

        :return: truncated SVD of the Hankel matrix
        :rtype: SVD
        """
        return self._svd


class PMSSA(MSSA):
    """Projected multivariate singular spectrum analysis (PMSSA).

    Examples

    >>> import torch as pt
    >>> from flowtorch.analysis import PMSSA
    >>> dm = pt.rand((1000, 200))
    >>> pmssa = PMSSA(dm, rank=20)
    >>> rec = pmssa.reconstruction
    >>> err = pmssa.reconstruction_error.norm() / dm.norm()
    """

    @tracked
    def __init__(
        self, data_matrix: pt.Tensor, window_size: int = None, rank: int = None
    ):
        """Create a PMSSA instance

        :param data_matrix: data matrix whose columns are formed by the individual snapshots
        :type data_matrix: pt.Tensor
        :param window_size: window size (number of snapshots) to shift over the full snapshot
            sequence; the window size is inversely proportional to the number of time delays,
            i.e., D = N - L +1 (D - delays, N - number of snapshots, L - window size); if L is
            not specified, N // 2 is chosen as default
        :type window_size: int, optional
        :param rank: number of POD modes on which to project the data before running MSSA; the
            truncation rank for the SVD of the Hankel matrix is identical; if unspecified, a suitable
            value is determined via Optimal Singular Value Hard Thresholding (SVHT)
        :type rank: int, optional
        """
        self._dm_org = data_matrix
        self._svd_dr = SVD(data_matrix, rank)
        super(PMSSA, self).__init__(
            self._svd_dr.U.T @ data_matrix, window_size, self._svd_dr.rank
        )

    @property
    def reconstruction(self) -> pt.Tensor:
        """Compute reconstruction error in the original basis

        :return: reconstruction of the original data matrix
        :rtype: pt.Tensor
        """
        return self._svd_dr.U @ super().reconstruction

    @property
    def reconstruction_error(self) -> pt.Tensor:
        """Compute elementwise reconstruction error."""
        return self._dm_org - self.reconstruction

    @property
    def svd_dr(self) -> SVD:
        """SVD used for dimensionality reduction.

        :return: SVD used for the initial projection step in PMSSA
        :rtype: SVD
        """
        return self._svd_dr
