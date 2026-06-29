"""Classes and functions wrapping around *torch.linalg.svd*."""

# standard library packages
import logging
from math import sqrt
from typing import Tuple, Union

# third party packages
import torch as pt

# flowtorch packages
from flowtorch.data.utils import format_byte_size

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


MODES = ("auto", "svd", "evd")


class SVD(object):
    """Compute and analyze the SVD of a data matrix.

    Examples

    >>> import torch as pt
    >>> from flowtorch.analysis import SVD
    >>> data = pt.rand((400, 5), dtype=pt.float32)
    >>> svd = SVD(data, rank=100)
    >>> print(svd)
    SVD of a 400x5 data matrix
    Selected/optimal rank: 5/2
    data type: torch.float32 (4b)
    truncated SVD size: 7.9297Kb
    >>> svd.s_rel
    tensor([9.9969e+01, 3.0860e-02, 3.0581e-04, 7.8097e-05, 3.2241e-05])
    >>> svd.s_cum
    tensor([ 99.9687,  99.9996,  99.9999, 100.0000, 100.0000])
    >>> svd.U.shape
    torch.Size([100, 5])
    """

    def __init__(
        self, data_matrix: pt.Tensor, rank: Union[int, None] = None, mode: str = "auto"
    ):
        """Compute the truncated singular value decomposition of a data matrix.

        :param data_matrix: data matrix of shape M x N, typically with M being the
            number of spatial points and N being the number of time steps
        :type data_matrix: pt.Tensor
        :param rank: rank at which to truncated the SVD; if no rank is given, the 'optimal'
            rank is determined via singular value hard thresholding; defaults to None
        :type rank: Union[int, None], optional
        :param mode: compute path; can be one of:
            'svd' - most accurate but slow for non-square matrices
            'evd' - most efficient but less robust and accurate
            'auto' - switches between 'svd' and 'evd' depending on the shape of the data matrix
            defaults to 'auto'
        :type mode: str, optional
        :raises ValueError:
            - if the data matrix does not have exactly two dimensions
            - if an invalid compute mode is specified
        """
        shape = data_matrix.shape
        if len(shape) != 2:
            raise ValueError(
                f"The data matrix must be a 2D tensor. Found shape {shape}"
            )
        self._rows, self._cols = shape
        if mode == "auto":
            if self._rows > 1.5 * self._cols or self._cols > 1.5 * self._rows:
                self._mode = "evd"
            else:
                self._mode = "svd"
        else:
            self._mode = mode
        if self._mode == "svd":
            U, s, V = self._svd(data_matrix)
        elif self._mode == "evd":
            U, s, V = self._gram_evd(data_matrix)
        else:
            raise ValueError(f"'mode' must be one of {MODES}. Got '{mode}'")
        self._opt_rank = self._optimal_rank(s)
        self.rank = self.opt_rank if rank is None else rank
        self._s_full = s
        logger.info(f"Truncating SVD at index {self.rank}/{min(self._cols, self._rows)}")
        self._U = U[:, : self.rank].contiguous()
        self._s = s[: self.rank].clone()
        self._V = V[:, : self.rank].contiguous()

    def _svd(self, X: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        """Compute the economy via the native SVD implementation.

        :param X: data matrix of shape M x N
        :type X: pt.Tensor
        :return: economy SVD of X with shapes:
            L = min(M, N)
            U: M x L
            s: L
            V: N x L
        :rtype: Tuple[pt.Tensor, pt.Tensor, pt.Tensor]
        """
        logger.info("Computing economy SVD via torch.linalg.svd()")
        U, s, V = pt.linalg.svd(X, full_matrices=False)
        V = V.conj().T
        return U, s, V

    def _gram_evd(self, X: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        """Compute the SVD via Gram matrix and eigendecomposition.

        :param X: data matrix of shape M x N
        :type X: pt.Tensor
        :return: economy SVD of X with shapes:
            L = min(M, N)
            U: M x L
            s: L
            V: N x L
        :rtype: Tuple[pt.Tensor, pt.Tensor, pt.Tensor]
        """
        logger.info("Computing economy SVD via Gram matrix and eigendecomposition")
        eps = pt.finfo(X.dtype).eps
        if self._rows > self._cols:
            C = X.conj().T @ X
            evals, V = pt.linalg.eigh(C)
            s = evals.clamp(0.0).sqrt().flip(0)
            V = V.flip(1)
            U = (X @ V) / s.clamp(eps * s[0])
        else:
            C = X @ X.conj().T
            evals, U = pt.linalg.eigh(C)
            s = evals.clamp(0.0).sqrt().flip(0)
            U = U.flip(1)
            V = (U.conj().T @ X).conj().T / s.clamp(eps * s[0])
        return U, s, V

    def _optimal_rank(self, s: pt.Tensor) -> int:
        """Compute the optimal singular value hard threshold.

        This function implements the svht_ rank estimation.

        .. _svht: https://doi.org/10.1109/TIT.2014.2323359

        :param s: sorted singular values
        :type s: pt.Tensor
        :return: optimal rank for truncation
        :rtype: int
        """
        beta = min(self._rows, self._cols) / max(self._rows, self._cols)
        omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
        tau_star = omega * pt.median(s)
        closest = int(pt.argmin((s - tau_star).abs()).item())
        if s[closest] > tau_star:
            return closest + 1
        else:
            return closest

    def reconstruct(self, rank: Union[int, None] = None) -> pt.Tensor:
        """Reconstruct the data matrix for a given rank.

        :param rank: rank used to compute a truncated reconstruction
        :type rank: int, optional
        :return: reconstruction of the input data matrix
        :rtype: pt.Tensor
        """
        r_rank = self.rank if rank is None else max(min(rank, self.rank), 1)
        return (
            self.U[:, :r_rank] @ pt.diag(self.s[:r_rank]) @ self.V[:, :r_rank].conj().T
        )

    @property
    def U(self) -> pt.Tensor:
        """Truncated matrix of left-singular vectors.

        :return: left-singular vectors truncated to specified rank
        :rtype: pt.Tensor
        """
        return self._U

    @property
    def s(self) -> pt.Tensor:
        """Truncated singular values.

        :return: singular values truncated to specified rank
        :rtype: pt.Tensor
        """
        return self._s

    @property
    def s_full(self) -> pt.Tensor:
        """All singular values of economy SVD.

        :return: singular values (not truncated)
        :rtype: pt.Tensor
        """
        return self._s_full

    @property
    def s_rel(self) -> pt.Tensor:
        """Relative truncated singular values.

        :return: contribution of singular values to their sum; given in percent
        :rtype: pt.Tensor
        """
        return self._s / self._s_full.sum() * 100.0

    @property
    def s_cum(self) -> pt.Tensor:
        """Relative cumulative contribution of singular values.

        :return: relative cumulative contribution given in percent
        :rtype: pt.Tensor
        """
        s_sum = self._s_full.sum().item()
        return pt.tensor(
            [
                self._s[:i].sum().item() / s_sum * 100.0
                for i in range(1, self._s.shape[0] + 1)
            ],
            dtype=self._s.dtype,
        )

    @property
    def V(self) -> pt.Tensor:
        """Truncated matrix of right-singular vectors.

        :return: right-singular vectors truncated at specified rank
        :rtype: pt.Tensor
        """
        return self._V

    @property
    def rank(self) -> int:
        """Truncation rank.

        :return: truncation rank
        :rtype: int
        """
        return self._rank

    @rank.setter
    def rank(self, value: int):
        """Set the truncation rank within a valid range.

        :param value: truncation rank to set
        :type value: int
        """
        self._rank = max(min(self._rows, self._cols, value), 1)

    @property
    def opt_rank(self) -> int:
        """Optimal truncation rank according to singular value hard threshold.

        :return: optimal rank
        :rtype: int
        """
        return self._opt_rank
    
    @property
    def mode(self) -> str:
        """SVD compute path

        :return: can be 'svd' or 'evd'
        :rtype: str
        """
        return self._mode

    @property
    def required_memory(self) -> int:
        """Compute the memory size in bytes of the truncated SVD.

        :return: cumulative size of truncated U, s, and V tensors in byte
        :rtype: int
        """
        return (
            self.U.element_size() * self.U.nelement()
            + self.s.element_size() * self.s.nelement()
            + self.V.element_size() * self.V.nelement()
        )

    def __str__(self) -> str:
        size, unit = format_byte_size(self.required_memory)
        ms = (
            f"SVD of a {self._rows}x{self._cols} data matrix",
            f"Selected/optimal rank: {self.rank}/{self.opt_rank}",
            f"data type: {self.U.dtype} ({self.U.element_size()}b)",
            "truncated SVD size: {:1.4f}{:s}".format(size, unit),
        )
        return "\n".join(ms)
