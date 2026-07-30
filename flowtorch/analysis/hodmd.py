"""Implementation of the higher-order DMD (HODMD)."""

from typing import Any, Tuple, Union
import torch as pt
from .svd import SVD
from .dmd import DMD


def _check_time_delays(delay: int, columns: int, min_cols: int):
    """Check if the number of time delays is valid.

    :param delay: how many time delays to use
    :type delay: int
    :param columns: number of columns in the data matrix
    :type columns: int
    :param min_cols: minimum number of columns the resulting data matrix
        must have, e.g., DMD is only possible if there are at least two
        columns; defaults to 2
    :type delay: int
    :raises ValueError: if delay is less than 1
    :raises ValueError: if there are not enough snapshots for the
        requested delays in combination with the minimum number of columns
    """
    if delay < 1:
        raise ValueError(
            f"The 'delay' parameter must be a positive integer. Got {delay}"
        )
    if columns - delay < min_cols - 1:
        raise ValueError(
            f"The number of snapshots ({columns:d}) must be larger than ({delay+min_cols-1:d})"
        )


def _create_time_delays(
    data_matrix: pt.Tensor, delay: int, min_cols: int = 2
) -> pt.Tensor:
    """Create data matrix enriched with time delays (Hankel matrix).
    :param data_matrix: 2D data matrix with (reduced) snapshots as column
        vectors
    :type data_matrix: pt.Tensor
    :param delay: number of time levels (delay coordinates) to use
    :type delay: int
    :param min_cols: minimum number of columns the resulting data matrix
        must have, e.g., DMD is only possible if there are at least two
        columns; defaults to 2
    :type delay: int
    :return: data matrix enriched with time delays
    :rtype: pt.Tensor
    """
    _, cols = data_matrix.shape
    _check_time_delays(delay, cols, min_cols)
    return pt.cat([data_matrix[:, i : cols - (delay - i - 1)] for i in range(delay)])


class HODMD(DMD):
    """Higher-order dynamic mode decomposition (HODMD).

    For the theoretical background, refer to Clainche and Vega (link_).
    The HODMD wraps around the standard DMD by adding an initial dimensionality
    reduction step and an enrichment of the data matrix with delays. To reconstruct
    snapshots and modes in the original space, a few properties of the base class
    are overwritten.
    .. _link: https://doi.org/10.1137/15M1054924

    Examples

    >>> from flowtorch.analysis import HODMD
    >>> dmd = HODMD(data_matrix, dt)
    set time delay explicitly to 5 time levels
    >>> dmd = HODMD(data_matrix, dt, delay=5)
    set the rank for the initial dimensionality reduction
    >>> dmd = HODMD(data_matrix, dt, delay=5, rank_dr=100)
    use optimal mode coefficients
    >>> dmd = HODMD(data_matrix, dt, delay=5, rank_dr=100, optimal=True)

    """

    def __init__(
        self,
        data_matrix: pt.Tensor,
        dt: float,
        delay: Union[int, None] = None,
        rank_dr: Union[int, None] = None,
        **dmd_options: Any,
    ):
        """Create a HODMD instance from data matrix and time step.

        :param data_matrix: data matrix whose columns are formed by
            individual snapshots
        :type data_matrix: pt.Tensor
        :param dt: time step between two snapshots
        :type dt: float
        :param delay: number of time levels (delay coordinates) to use;
            a value of 1 corresponds to using only one time level; if the
            default value is not specified, delay is set to one third of
            the data matrix's columns (the number of snapshots) as suggested
            by Clainche and Vega (link_); defaults to None
        :type delay: int, optional
        :param rank_dr: SVD rank of the initial dimensionality reduction step; if
            the default value is not specified, the rank is automatically determined as
            described in :class:`flowtorch.analysis.svd.SVD`; defaults to None
        :type rank_dr: int, optional

        """
        self._dm_org = data_matrix
        self._rows_org, self._cols_org = data_matrix.shape
        if delay is None:
            self._delay = int(self._cols_org / 3)
        else:
            self._delay = delay
        self._svd_dr = SVD(data_matrix, rank_dr)
        super(HODMD, self).__init__(
            _create_time_delays(self._svd_dr.U.T @ self._dm_org, self._delay),
            dt,
            **dmd_options,
        )

    def predict(self, initial_condition: pt.Tensor, n_steps: int) -> pt.Tensor:
        """Predict evolution over N steps starting from user-defined initial conditions.

        The prediction is performed as follows:
        1) the initial conditions are projected on the first r POD modes
        2) the time delay embedding is computed in the reduced space
        3) the prediction is computed in the reduced space
        4) the first r rows of the prediction are reconstructed

        :param initial_condition: initial sequence of state vectors without time delay
            embedding; for d delays and a state vector of size M, the initial conditions
            should be given as a M x d matrix sorted such that the most recent state
            is contained in the last column
        :type initial_condition: pt.Tensor
        :param n_steps: number of future steps to predict
        :type n_steps: int
        :return: predicted states; due to the structure of the linear operator
            some states are predicted multiple times; only the first M rows of the
            reconstruction are returned
        :rtype: pt.Tensor
        """
        ic = self.svd_dr.U.T @ initial_condition
        ic = _create_time_delays(ic, self._delay, 1).squeeze()
        prediction = super().predict(ic, n_steps)
        r = self.svd_dr.rank
        return self.svd_dr.U @ prediction[-r:]

    @property
    def svd_dr(self) -> SVD:
        return self._svd_dr

    @property
    def delay(self) -> int:
        return self._delay

    @property
    def modes(self) -> pt.Tensor:
        """Get DMD modes in the input space.

        As suggested by Clainche and Vega, only the first set of modes
        corresponding to the first r rows of the reduced DMD modes are
        kept (r is the dimension after the initial dimensionality reduction).

        :return: DMD modes in the input space
        :rtype: pt.Tensor
        """
        r = self.svd_dr.rank
        return self.svd_dr.U.type(self._modes.dtype) @ super().modes[:r]

    @property
    def dynamics(self) -> pt.Tensor:
        """Get mode dynamics for the original data matrix.

        :return: mode dynamics for the original snapshot sequence
        :rtype: pt.Tensor
        """
        return pt.diag(self.amplitude) @ pt.linalg.vander(
            self.eigvals, N=self._cols_org
        )

    @property
    def reconstruction(self) -> pt.Tensor:
        """Compute reconstruction of original data matrix.

        Due to the time delays, some states are contained multiple times
        in a reconstruction. Only the first occurrence of a state is kept
        when looping over the rows in steps of size r (rank).

        :return: reconstruction of original data matrix
        :rtype: pt.Tensor
        """
        rec = self.modes @ self.dynamics
        if not self._complex:
            rec = rec.real
        return rec

    @property
    def reconstruction_error(self) -> pt.Tensor:
        """Compute the point-wise reconstruction error.

        Due to the time delay, not all snapshots from the input data
        matrix are reconstructed, so the error is only computed for the
        available reconstructed snapshots

        :return: reconstruction error
        :rtype: pt.Tensor
        """
        return self.reconstruction - self._dm_org

    @property
    def projection_error(self) -> pt.Tensor:
        """Compute the difference between Y and AX.

        :return: projection error
        :rtype: pt.Tensor
        """
        r = self.svd_dr.rank
        return self.svd_dr.U @ super().projection_error[:r]

    @property
    def tlsq_error(self) -> Tuple[pt.Tensor, pt.Tensor]:
        """Compute the *noise* in X and Y.

        :return: noise in X and Y
        :rtype: Tuple[pt.Tensor, pt.Tensor]
        """
        dx, dy = super().tlsq_error
        r = self.svd_dr.rank
        return self.svd_dr.U @ dx[:r], self.svd_dr.U @ dy[:r]
