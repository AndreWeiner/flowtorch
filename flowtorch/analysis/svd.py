"""Classes and functions wrapping around *torch.linalg.svd*.
"""

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


def inexact_alm_matrix_completion(data_matrix: pt.Tensor, sparsity: float = 1.0,
                                  tol: float = 1.0e-6, max_iter: int = 100,
                                  verbose: bool = False) -> Tuple[pt.Tensor, pt.Tensor]:
    """Split a data matrix in low rank and sparse contributions.

    This function implements the *inexact augmented Lagrange multiplier
    matrix completion* algorithm to solve the *principal component pursuit*
    problem. The implementation is based on the Matlab code of Isabel Scherl
    (link_), which is in turn based on the LRSLibrary_.

    .. _link: https://github.com/ischerl/RPCA-PIV/blob/master/functions/inexact_alm_rpca.m
    .. _LRSLibrary: https://github.com/andrewssobral/lrslibrary

    :param data_matrix: input data matrix; snapshots must be
        organized as column vectors
    :type data_matrix: pt.Tensor
    :param sparsity: factor to compute Lagrangian multiplier for sparsity
        (typically named *lambda*); lower values lead to more aggressive
        filtering
    :type sparsity: float, optional
    :param tol: tolerance for the normalized Frobenius norm of the difference
        between original data and the sum of low rank and sparse contributions;
        defaults to 1.0e-6
    :type tol: float, optional
    :param max_iter: maximum number of iteration before to give up, defaults to 100
    :type max_iter: int, optional
    :param verbose: residual is printed for every iteration if True;
        defaults to False
    :type verbose: bool, optional
    :return: tuple holding the low rank and the sparse matrices
    :rtype: Tuple[pt.Tensor, pt.Tensor]
    """
    logger.info("Performing inexact augmented Lagrange multiplier matrix completion.")
    # make sure we do at least one iteration
    if max_iter <= 0:
        logger.warning(f"A value of max_iter={max_iter} is invalid. Changed max_iter to 1.")
        max_iter = 1
    row, col = data_matrix.shape
    lambda_0 = sparsity / sqrt(row)
    # low rank and sparse matrices
    L, S = pt.zeros_like(data_matrix), pt.zeros_like(data_matrix)
    # matrix of Lagrange multipliers
    Y = data_matrix.detach().clone()
    norm_two = pt.linalg.svdvals(Y)[0].item()
    norm_inf = pt.linalg.norm(Y, float("inf")).item()
    dual_norm = max(norm_two, norm_inf)
    Y /= dual_norm
    # more hyperparameters
    mu = 1.25 / norm_two
    norm_data = pt.linalg.norm(data_matrix)
    sv = 10
    rho = 1.5

    for i in range(max_iter):
        temp = data_matrix - L + Y/mu
        S = pt.maximum(temp - lambda_0/mu, pt.tensor(0.0))
        S += pt.minimum(temp + lambda_0/mu, pt.tensor(0.0))
        U, s, VH = pt.linalg.svd(data_matrix - S + Y/mu, full_matrices=False)
        # truncate SVD
        svp = s[s > 1.0/mu].shape[0]
        if svp < sv:
            sv = min(svp+1, col)
        else:
            sv = min(svp + round(0.05*col), col)
        L = U[:, :svp] @ pt.diag(s[:svp] - 1.0/mu) @ VH[:svp, :]
        # print(L[0,:])
        # Z is the residual matrix
        Z = data_matrix - L - S
        # update Lagrange multipliers
        Y += mu*Z
        mu = min(mu*rho, mu*1.0e7)
        # check convergence
        residual = pt.linalg.norm(Z) / norm_data
        if residual < tol:
            logger.info(f"Inexact ALM converged after {i+1} iterations")
            logger.info("Final residual: {:2.4e}".format(residual))
            return L, S
        if verbose:
            logger.info("Residual after iteration {:5d}: {:10.4e}".format(i+1, residual.item()))
    logger.warning(f"Inexact ALM did not converge within {max_iter} iterations")
    logger.info("Final residual: {:10.4e}".format(residual))
    return L, S


class SVD(object):
    """Compute and analyze the SVD of a data matrix.

    :param U: left singular vectors
    :type U: pt.Tensor
    :param s: singular values
    :type s: pt.Tensor
    :param s_rel: singular values normalized with their sum in percent
    :type s_rel: pt.Tensor
    :param s_cum: cumulative normalized singular values in percent
    :type s_cum: pt.Tensor
    :param V: right singular values
    :type V: pt.Tensor
    :param L: low rank contribution to data matrix
    :type L: pt.Tensor
    :param S: sparse contribution to data matrix
    :type S: pt.Tensor
    :param robust: data_matrix is split into low rank and sparse contributions
        if True or if dictionary with options for Inexact ALM algorithm; the SVD
        is computed only on the low rank matrix; defaults to False
    :type robust: Union[bool,dict]
    :param rank: rank used for truncation
    :type rank: int
    :param opt_rank: optimal rank according to SVHT
    :type opt_rank: int
    :param required_memory: memory required to store the truncation U, s, and V
    :type required_memory: int

    Examples

    >>> from flowtorch import DATASETS
    >>> from flowtorch.data import FOAMDataloader
    >>> from flowtorch.analysis import SVD
    >>> loader = FOAMDataloader(DATASETS["of_cavity_ascii"])
    >>> data = loader.load_snapshot("p", loader.write_times[1:])
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
    torch.Size([400, 5])
    >>> svd = SVD(data, rank=100, robust=True)
    >>> svd.L.shape
    torch.Size([400, 5])
    >>> svd = SVD(data, rank=100, robust={"sparsity" : 1.0, "verbose" : True, "max_iter" : 100})
    >>> svd.S.shape
    torch.Size([400, 5])

    """

    def __init__(self, data_matrix: pt.Tensor, rank: Union[int, None] = None,
                 robust: Union[bool, dict] = False):
        shape = data_matrix.shape
        assert len(shape) == 2, (
            f"The data matrix must be a 2D tensor.\
            The provided data matrix has {len(shape)} dimensions."
        )
        self._rows, self._cols = shape
        self._robust = robust
        if bool(self._robust):
            if isinstance(robust, dict):
                L, S = inexact_alm_matrix_completion(data_matrix, **robust)
            else:
                L, S = inexact_alm_matrix_completion(data_matrix)
            self._L, self._S = L, S
            U, s, VH = pt.linalg.svd(L, full_matrices=False)
        else:
            self._L, self._S = None, None
            U, s, VH = pt.linalg.svd(data_matrix, full_matrices=False)
        self._opt_rank = self._optimal_rank(pt.from_numpy(s))
        self.rank = self.opt_rank if rank is None else rank
        self._U = U[:, :self.rank]
        self._s = s[:self.rank]
        self._V = VH.conj().T[:, :self.rank]

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
        omega = 0.56*beta**3 - 0.95*beta**2 + 1.82*beta + 1.43
        tau_star = omega * pt.median(s)
        closest = pt.argmin((s - tau_star).abs()).item()
        if s[closest] > tau_star:
            return closest + 1
        else:
            return closest

    def reconstruct(self, rank: int = None) -> pt.Tensor:
        """Reconstruct the data matrix for a given rank.

        :param rank: rank used to compute a truncated reconstruction
        :type rank: int, optional
        :return: reconstruction of the input data matrix
        :rtype: pt.Tensor
        """
        r_rank = self.rank if rank is None else max(min(rank, self.rank), 1)
        return self.U[:, :r_rank] @ pt.diag(self.s[:r_rank]) @ self.V[:, :r_rank].conj().T

    @property
    def U(self) -> pt.Tensor:
        return self._U

    @property
    def s(self) -> pt.Tensor:
        return self._s

    @property
    def s_rel(self) -> pt.Tensor:
        return self._s / self._s.sum() * 100.0

    @property
    def s_cum(self) -> pt.Tensor:
        s_sum = self._s.sum().item()
        return pt.tensor(
            [self._s[:i].sum().item() / s_sum *
             100.0 for i in range(1, self._s.shape[0]+1)],
            dtype=self._s.dtype
        )

    @property
    def V(self) -> pt.Tensor:
        return self._V

    @property
    def L(self) -> pt.Tensor:
        return self._L

    @property
    def S(self) -> pt.Tensor:
        return self._S

    @property
    def robust(self) -> Union[bool, dict]:
        return self._robust

    @property
    def rank(self) -> int:
        return self._rank

    @rank.setter
    def rank(self, value: int):
        self._rank = max(min(self._rows, self._cols, value), 1)

    @property
    def opt_rank(self) -> int:
        return self._opt_rank

    @property
    def required_memory(self) -> int:
        """Compute the memory size in bytes of the truncated SVD.

        :return: cumulative size of truncated U, s, and V tensors in byte
        :rtype: int
        """
        return (self.U.element_size() * self.U.nelement() +
                self.s.element_size() * self.s.nelement() +
                self.V.element_size() * self.V.nelement())

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(data_matrix, rank={self.rank})"

    def __str__(self) -> str:
        size, unit = format_byte_size(self.required_memory)
        ms = [f"SVD of a {self._rows}x{self._cols} data matrix", f"Selected/optimal rank: {self.rank}/{self.opt_rank}",
              f"data type: {self.U.dtype} ({self.U.element_size()}b)",
              "truncated SVD size: {:1.4f}{:s}".format(size, unit)]
        return "\n".join(ms)
