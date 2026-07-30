"""Encoder based on the singular value decomposition (SVD)."""

# third party packages
import torch as pt

# flowtorch packages
from flowtorch import DEFAULT_DTYPE
from flowtorch.analysis import SVD
from .base import Encoder
from .utils import log_time


class SVDEncoder(Encoder):
    """SVD-based dimensionality reduction for ROM.

    Examples

    >>> from flowtorch import DATASETS
    >>> from flowtorch.data import FOAMDataloader
    >>> from flowtorch.rom import SVDEncoder
    >>> loader = FOAMDataloader(DATASETS["of_cylinder2D_binary"])
    >>> data = loader.load_snapshot("p", loader.write_times[1:11])
    >>> data.shape
    torch.Size([13678, 10])
    >>> encoder = SVDEncoder(rank=10)
    >>> info = encoder.train(data)
    >>> reduced_state = encoder.encode(data)
    >>> reduced_state.shape
    torch.Size([10, 10])
    >>> full_state = encoder.decode(reduced_state)
    >>> full_state.shape
    torch.Size([13678, 10])

    """

    def __init__(self, rank: int = None):
        """Derived class constructor.

        :param rank: rank to truncate the SVD
        :type rank: int, optional
        """
        super(SVDEncoder, self).__init__()
        self._rank = rank
        self._modes = None
        self._state_size = None

    @log_time
    def train(self, data: pt.Tensor) -> dict:
        """Compute the POD modes of a given data matrix.

        :param data: data matrix containing a sequence of snapshots,
            where each snapshot corresponds to a column vector of the
            data matrix
        :type data: pt.Tensor
        :return: empty dictionary since there is no real training process
        :rtype: dict
        """
        svd = SVD(data, self._rank)
        self._modes = svd.U.clone()
        self._state_size = self._modes.shape[0]
        self._rank = svd.rank
        del svd
        self.trained = True
        return dict()

    def encode(self, full_state: pt.Tensor) -> pt.Tensor:
        """Project one or multiple state vectors onto the POD modes.

        This function computes the scalar projections of one or more
        state vectors onto each POD mode. The result is vector (single state)
        or a matrix (sequence of states) of scalar projections, in which the
        row index corresponds to the associated POD mode and the column index
        corresponds to the associated snapshot in the sequence.


        :param full_state: full state vector or sequence of state vectors
        :type full_state: pt.Tensor
        :raises ValueError: if the full state shape does not match the trained encoder
        :return: reduced state vector or sequence of reduced state vectors
        :rtype: pt.Tensor
        """
        self._check_state_shape(full_state.shape)
        if not self.trained:
            raise Exception("Encoding not possible: the encoder has not been trained")
        return self._modes.conj().T @ full_state

    def decode(self, reduced_state: pt.Tensor) -> pt.Tensor:
        """Compute the full projection onto the POD modes.

        :param reduced_state: 1D or 2D tensor, in which each column
            holds the mode coefficients of a given state; if the input
            has two dimensions, the second dimension is considered as the
            batch dimension
        :type reduced_state: pt.Tensor
        :return: full state vector in the subspace spanned by the POD modes
        :rtype: pt.Tensor
        """
        self._check_reduced_state_size(reduced_state.shape)
        if not self.trained:
            raise Exception("Decoding not possible: the encoder has not been trained")
        return self._modes @ reduced_state

    @property
    def state_shape(self) -> pt.Size:
        """Get the size of the full state.

        :return: size of the full state
        :rtype: pt.Size
        """
        return pt.Size((self._state_size,))

    @property
    def reduced_state_size(self) -> int:
        """Get the size of the reduced state.

        :return: size of the reduced state.
        :rtype: int
        """
        return self._rank
