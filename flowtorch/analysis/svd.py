"""Classes and functions wrapping around *torch.linalg.svd*."""

# standard library packages
import logging
from math import isfinite, sqrt
from numbers import Integral, Real
from typing import NamedTuple, Sequence, Tuple, Union

# third party packages
import torch as pt

# flowtorch packages
from flowtorch.utils import format_byte_size

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


MODES = ("auto", "svd", "evd")


class PODSubspaceDependencyResult(NamedTuple):
    """Results of a POD subspace data-dependency sweep.

    ``similarity`` has shape ``(n_fractions, n_strides, n_ranks)``. The
    ``n_snapshots`` and ``optimal_ranks`` tensors have shape
    ``(n_fractions, n_strides)``.
    """

    similarity: pt.Tensor
    ranks: pt.Tensor
    sequence_fractions: pt.Tensor
    snapshot_strides: pt.Tensor
    n_snapshots: pt.Tensor
    optimal_ranks: pt.Tensor
    reference_optimal_rank: int


def _prepare_subspace_ranks(
    ranks: Union[int, Sequence[int], pt.Tensor, None],
    maximum_rank: int,
    device: pt.device,
) -> pt.Tensor:
    """Validate and normalize one or more subspace ranks."""
    if maximum_rank < 1:
        raise ValueError("at least one subspace direction must be available")
    if ranks is None:
        return pt.arange(1, maximum_rank + 1, dtype=pt.int64, device=device)
    if isinstance(ranks, pt.Tensor):
        if ranks.ndim != 1 or ranks.is_floating_point() or pt.is_complex(ranks):
            raise ValueError("ranks must be a one-dimensional integer tensor")
        values = ranks.detach().cpu().tolist()
    elif isinstance(ranks, Integral) and not isinstance(ranks, bool):
        values = [int(ranks)]
    else:
        if not isinstance(ranks, Sequence):
            raise ValueError("ranks must contain positive integers")
        values = list(ranks)
    if len(values) == 0:
        raise ValueError("ranks must contain at least one value")
    if any(
        not isinstance(value, Integral) or isinstance(value, bool) for value in values
    ):
        raise ValueError("ranks must contain positive integers")
    normalized = [int(value) for value in values]
    if any(value < 1 for value in normalized):
        raise ValueError("ranks must contain positive integers")
    if any(first >= second for first, second in zip(normalized, normalized[1:])):
        raise ValueError("ranks must be strictly increasing")
    if normalized[-1] > maximum_rank:
        raise ValueError(
            f"the largest rank ({normalized[-1]}) exceeds the available rank "
            f"({maximum_rank})"
        )
    return pt.tensor(normalized, dtype=pt.int64, device=device)


def _prepare_weight(
    weight: Union[pt.Tensor, None],
    n_state: int,
    device: pt.device,
    dtype: pt.dtype,
) -> tuple[Union[pt.Tensor, None], Union[pt.Tensor, None]]:
    """Return expanded raw weights and their columnwise square-root factors."""
    if weight is None:
        return None, None
    if weight.ndim != 1 or weight.numel() < 1:
        raise ValueError("weight must be a non-empty one-dimensional tensor")
    if weight.device != device:
        raise ValueError("weight and data must be on the same device")
    if pt.is_complex(weight) or weight.dtype == pt.bool:
        raise ValueError("weight must have a real numeric dtype")
    if not bool(pt.isfinite(weight).all()) or bool((weight <= 0.0).any()):
        raise ValueError("weight values must be finite and positive")
    n_weight = weight.shape[0]
    if n_weight != n_state:
        if n_state % n_weight != 0:
            raise ValueError(
                f"weight length ({n_weight}) must divide the state dimension "
                f"({n_state})"
            )
        weight = weight.repeat(n_state // n_weight)
    expanded_weight = weight.to(dtype=dtype)
    sqrt_weight = expanded_weight.sqrt().unsqueeze(-1)
    return expanded_weight, sqrt_weight


def _orthonormalize_modes(modes: pt.Tensor) -> pt.Tensor:
    """Return a basis spanning the nested column spaces of ``modes``."""
    basis, triangular = pt.linalg.qr(modes, mode="reduced")
    diagonal = pt.diagonal(triangular).abs()
    tolerance = (
        pt.finfo(modes.real.dtype).eps
        * max(modes.shape)
        * diagonal.max().clamp_min(1.0)
    )
    if bool((diagonal <= tolerance).any()):
        raise ValueError("the selected mode columns must be linearly independent")
    return basis


def subspace_similarity(
    first_modes: pt.Tensor,
    second_modes: pt.Tensor,
    ranks: Union[int, Sequence[int], pt.Tensor, None] = None,
    weight: Union[pt.Tensor, None] = None,
) -> pt.Tensor:
    r"""Compare nested subspaces spanned by two columnwise mode sets.

    For each requested rank ``r``, the returned normalized projection overlap is

    .. math::

        S_r = \frac{1}{r}\lVert Q_{1,r}^* Q_{2,r}\rVert_F^2
            = \frac{1}{r}\sum_{i=1}^r \cos^2(\theta_i),

    where ``Q`` are orthonormal bases and ``theta`` are the principal angles.
    The score lies in ``[0, 1]`` and is invariant to sign, phase, ordering, and
    rotations within either rank-``r`` subspace. When ``weight`` is provided,
    the bases are constructed in the corresponding diagonal weighted inner
    product.

    If ``ranks`` is omitted, similarities for every available rank are
    returned. The result is always one-dimensional, including when a single
    integer rank is supplied.

    :param first_modes: first columnwise mode set
    :type first_modes: pt.Tensor
    :param second_modes: second columnwise mode set
    :type second_modes: pt.Tensor
    :param ranks: positive, strictly increasing subspace ranks; defaults to all
        available ranks
    :type ranks: Union[int, Sequence[int], pt.Tensor, None], optional
    :param weight: positive diagonal spatial weight
    :type weight: pt.Tensor, optional
    :raises ValueError: for incompatible modes, ranks, or weights
    :return: normalized similarity for every requested rank
    :rtype: pt.Tensor
    """
    if first_modes.ndim != 2 or second_modes.ndim != 2:
        raise ValueError("mode sets must be two-dimensional tensors")
    if first_modes.shape[0] != second_modes.shape[0]:
        raise ValueError("mode sets must have the same state dimension")
    if first_modes.device != second_modes.device:
        raise ValueError("mode sets must be on the same device")
    if not (first_modes.is_floating_point() or pt.is_complex(first_modes)):
        raise ValueError("mode sets must have a floating-point or complex dtype")
    if not (second_modes.is_floating_point() or pt.is_complex(second_modes)):
        raise ValueError("mode sets must have a floating-point or complex dtype")

    maximum_rank = min(
        first_modes.shape[0], first_modes.shape[1], second_modes.shape[1]
    )
    selected_ranks = _prepare_subspace_ranks(ranks, maximum_rank, first_modes.device)
    maximum_selected = int(selected_ranks[-1].item())
    dtype = pt.promote_types(first_modes.dtype, second_modes.dtype)
    first = first_modes[:, :maximum_selected].to(dtype=dtype)
    second = second_modes[:, :maximum_selected].to(dtype=dtype)
    if not bool(pt.isfinite(first).all()) or not bool(pt.isfinite(second).all()):
        raise ValueError("selected modes must contain only finite values")
    _, sqrt_weight = _prepare_weight(
        weight,
        first_modes.shape[0],
        first_modes.device,
        first.real.dtype,
    )
    if sqrt_weight is not None:
        first = first * sqrt_weight
        second = second * sqrt_weight

    first_basis = _orthonormalize_modes(first)
    second_basis = _orthonormalize_modes(second)
    overlap_squared = (first_basis.conj().T @ second_basis).abs().square()
    cumulative_overlap = overlap_squared.cumsum(dim=0).cumsum(dim=1)
    indices = selected_ranks - 1
    similarities = cumulative_overlap[indices, indices] / selected_ranks.to(
        cumulative_overlap.dtype
    )
    return similarities.clamp(0.0, 1.0)


def _prepare_sweep_values(
    values: Sequence[Union[int, float]],
    name: str,
) -> list[Union[int, float]]:
    """Convert a sweep coordinate to a validated Python list."""
    if isinstance(values, pt.Tensor):
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        normalized = values.detach().cpu().tolist()
    else:
        normalized = list(values)
    if len(normalized) == 0:
        raise ValueError(f"{name} must contain at least one value")
    return normalized


def pod_subspace_data_dependency(
    data_matrix: pt.Tensor,
    ranks: Union[int, Sequence[int], pt.Tensor, None] = None,
    sequence_fractions: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    snapshot_strides: Sequence[int] = (1, 2, 3, 4),
    weight: Union[pt.Tensor, None] = None,
    subtract_mean: bool = True,
    mode: str = "auto",
) -> PODSubspaceDependencyResult:
    r"""Measure POD-subspace dependency on sequence length and sampling.

    The POD of the complete sequence at snapshot stride one is used as the
    reference. For every sequence fraction ``p`` and snapshot stride ``n``,
    the sampled matrix is ``data_matrix[:, :floor(p*N):n]``. Its leading POD
    subspaces are compared with the reference using
    :func:`subspace_similarity`.

    If ``ranks`` is omitted, all ranks up to the automatically selected rank
    of the reference SVD are evaluated, subject to the rank supported by the
    smallest sampled matrix. Automatic ranks of all sampled matrices are
    returned as diagnostics but do not alter the common comparison ranks.

    ``similarity`` in the returned result has shape
    ``(n_fractions, n_strides, n_ranks)``. A stride ``n`` corresponds to an
    effective sampling interval ``n*dt`` if the original snapshots are spaced
    by ``dt``.

    :param data_matrix: state-by-snapshot data matrix
    :type data_matrix: pt.Tensor
    :param ranks: positive, strictly increasing comparison ranks; defaults to
        all feasible ranks up to the reference SVD's automatic rank
    :type ranks: Union[int, Sequence[int], pt.Tensor, None], optional
    :param sequence_fractions: fractions of the sequence retained before
        subsampling
    :type sequence_fractions: Sequence[float], optional
    :param snapshot_strides: positive snapshot strides
    :type snapshot_strides: Sequence[int], optional
    :param weight: positive diagonal spatial weight used for the POD inner
        product
    :type weight: pt.Tensor, optional
    :param subtract_mean: subtract the temporal mean independently from every
        sampled matrix, defaults to ``True``
    :type subtract_mean: bool, optional
    :param mode: SVD compute mode passed to :class:`SVD`, defaults to ``"auto"``
    :type mode: str, optional
    :raises ValueError: for invalid data, sweep coordinates, or ranks
    :return: dependency sweep and rank diagnostics
    :rtype: PODSubspaceDependencyResult
    """
    if data_matrix.ndim != 2:
        raise ValueError("data_matrix must be a two-dimensional tensor")
    if not (data_matrix.is_floating_point() or pt.is_complex(data_matrix)):
        raise ValueError("data_matrix must have a floating-point or complex dtype")
    if not bool(pt.isfinite(data_matrix).all()):
        raise ValueError("data_matrix must contain only finite values")
    if not isinstance(subtract_mean, bool):
        raise ValueError("subtract_mean must be a boolean")
    n_state, n_total = data_matrix.shape
    if n_state < 1 or n_total < 1:
        raise ValueError("data_matrix dimensions must be non-empty")

    fraction_values = _prepare_sweep_values(sequence_fractions, "sequence_fractions")
    if any(
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not isfinite(float(value))
        or float(value) <= 0.0
        or float(value) > 1.0
        for value in fraction_values
    ):
        raise ValueError("sequence_fractions must lie in the interval (0, 1]")
    fractions = [float(value) for value in fraction_values]

    stride_values = _prepare_sweep_values(snapshot_strides, "snapshot_strides")
    if any(
        not isinstance(value, Integral) or isinstance(value, bool) or int(value) < 1
        for value in stride_values
    ):
        raise ValueError("snapshot_strides must contain positive integers")
    strides = [int(value) for value in stride_values]

    end_indices = [int(fraction * n_total) for fraction in fractions]
    counts = [
        [(end_index + stride - 1) // stride for stride in strides]
        for end_index in end_indices
    ]
    degrees_removed = 1 if subtract_mean else 0
    feasible_rank = min(
        n_state,
        min(count - degrees_removed for row in counts for count in row),
    )
    if feasible_rank < 1:
        raise ValueError(
            "every sampled matrix must retain at least one independent snapshot"
        )

    def prepare_sample(sample: pt.Tensor) -> pt.Tensor:
        if subtract_mean:
            sample = sample - sample.mean(dim=1, keepdim=True)
        return sample

    reference_matrix = prepare_sample(data_matrix)
    if ranks is None:
        reference_svd = SVD(reference_matrix, rank=None, mode=mode, weight=weight)
        maximum_rank = min(reference_svd.rank, feasible_rank)
        selected_ranks = _prepare_subspace_ranks(None, maximum_rank, data_matrix.device)
    else:
        selected_ranks = _prepare_subspace_ranks(
            ranks, feasible_rank, data_matrix.device
        )
        maximum_rank = int(selected_ranks[-1].item())
        reference_svd = SVD(
            reference_matrix, rank=maximum_rank, mode=mode, weight=weight
        )

    n_fractions = len(fractions)
    n_strides = len(strides)
    n_ranks = selected_ranks.numel()
    real_dtype = data_matrix.real.dtype
    similarity = pt.empty(
        (n_fractions, n_strides, n_ranks),
        dtype=real_dtype,
        device=data_matrix.device,
    )
    optimal_ranks = pt.empty(
        (n_fractions, n_strides), dtype=pt.int64, device=data_matrix.device
    )
    snapshot_counts = pt.tensor(counts, dtype=pt.int64, device=data_matrix.device)

    for fraction_idx, end_index in enumerate(end_indices):
        for stride_idx, stride in enumerate(strides):
            if end_index == n_total and stride == 1:
                similarity[fraction_idx, stride_idx] = 1.0
                optimal_ranks[fraction_idx, stride_idx] = reference_svd.opt_rank
                continue
            sample = prepare_sample(data_matrix[:, :end_index:stride])
            sampled_svd = SVD(sample, rank=maximum_rank, mode=mode, weight=weight)
            similarity[fraction_idx, stride_idx] = subspace_similarity(
                reference_svd.U,
                sampled_svd.U,
                ranks=selected_ranks,
                weight=weight,
            )
            optimal_ranks[fraction_idx, stride_idx] = sampled_svd.opt_rank

    return PODSubspaceDependencyResult(
        similarity=similarity,
        ranks=selected_ranks,
        sequence_fractions=pt.tensor(
            fractions, dtype=real_dtype, device=data_matrix.device
        ),
        snapshot_strides=pt.tensor(strides, dtype=pt.int64, device=data_matrix.device),
        n_snapshots=snapshot_counts,
        optimal_ranks=optimal_ranks,
        reference_optimal_rank=reference_svd.opt_rank,
    )


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
        self,
        data_matrix: pt.Tensor,
        rank: Union[int, None] = None,
        mode: str = "auto",
        weight: Union[pt.Tensor, None] = None,
    ):
        r"""Compute a truncated, optionally weighted SVD of a data matrix.

        With a diagonal spatial weight ``W``, the decomposition is computed for
        ``W**0.5 @ data_matrix``. :attr:`U` contains physical modes satisfying
        ``U.conj().T @ W @ U = I``, while :attr:`U_weighted` contains their
        Euclidean-orthonormal weighted-coordinate representation. Consequently,
        :meth:`reconstruct` always returns data in the original coordinates.

        :param data_matrix: data matrix of shape M x N, typically with M being the
            number of spatial points and N being the number of time steps
        :type data_matrix: pt.Tensor
        :param rank: rank at which to truncate the SVD; if no rank is given, the 'optimal'
            rank is determined via singular value hard thresholding; defaults to None
        :type rank: Union[int, None], optional
        :param mode: compute path; can be one of:
            'svd' - most accurate but slow for non-square matrices
            'evd' - most efficient but less robust and accurate
            'auto' - switches between 'svd' and 'evd' depending on the shape of the data matrix
            defaults to 'auto'
        :type mode: str, optional
        :param weight: positive diagonal inner-product weight; a shorter vector
            is repeated when its length divides the state dimension
        :type weight: pt.Tensor, optional
        :raises ValueError:
            - if the data matrix does not have exactly two dimensions
            - if an invalid compute mode is specified
            - if the weight is invalid or incompatible with the state dimension
        """
        shape = data_matrix.shape
        if len(shape) != 2:
            raise ValueError(
                f"The data matrix must be a 2D tensor. Found shape {shape}"
            )
        self._rows, self._cols = shape
        self._weight, self._sqrt_weight = _prepare_weight(
            weight,
            self._rows,
            data_matrix.device,
            data_matrix.real.dtype,
        )
        weighted_data = data_matrix
        if self._sqrt_weight is not None:
            weighted_data = data_matrix * self._sqrt_weight
        if mode == "auto":
            if self._rows > 1.5 * self._cols or self._cols > 1.5 * self._rows:
                self._mode = "evd"
            else:
                self._mode = "svd"
        else:
            self._mode = mode
        if self._mode == "svd":
            U_weighted, s, V = self._svd(weighted_data)
        elif self._mode == "evd":
            U_weighted, s, V = self._gram_evd(weighted_data)
        else:
            raise ValueError(f"'mode' must be one of {MODES}. Got '{mode}'")
        self._opt_rank = self._optimal_rank(s)
        self.rank = self.opt_rank if rank is None else rank
        self._s_full = s
        logger.info(
            f"Truncating SVD at index {self.rank}/{min(self._cols, self._rows)}"
        )
        U = U_weighted[:, : self.rank]
        if self._sqrt_weight is not None:
            U = U / self._sqrt_weight
        self._U = U.contiguous()
        self._s = s[: self.rank].clone()
        self._V = V[:, : self.rank].contiguous()

    def _svd(self, X: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        """Compute the economy SVD via the native SVD implementation.

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
        return (self.U[:, :r_rank] * self.s[:r_rank]) @ self.V[:, :r_rank].conj().T

    @property
    def U(self) -> pt.Tensor:
        r"""Physical left-singular vectors.

        For a weighted decomposition, these modes are orthonormal in the
        weighted inner product, ``U.conj().T @ W @ U = I``.

        :return: physical left-singular vectors truncated to specified rank
        :rtype: pt.Tensor
        """
        return self._U

    @property
    def U_weighted(self) -> pt.Tensor:
        r"""Euclidean-orthonormal left-singular vectors.

        For a weighted decomposition, this is ``W**0.5 @ U``. Without a
        weight, it is identical to :attr:`U`.

        :return: left-singular vectors in weighted coordinates
        :rtype: pt.Tensor
        """
        if self._sqrt_weight is None:
            return self._U
        return self._U * self._sqrt_weight

    @property
    def weight(self) -> Union[pt.Tensor, None]:
        """Expanded raw diagonal inner-product weight.

        :return: expanded weight, or ``None`` for the Euclidean inner product
        :rtype: Union[pt.Tensor, None]
        """
        return self._weight

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

        :return: cumulative size of truncated U, s, and V tensors in bytes
        :rtype: int
        """
        memory = (
            self.U.element_size() * self.U.nelement()
            + self.s.element_size() * self.s.nelement()
            + self.V.element_size() * self.V.nelement()
        )
        if self.weight is not None:
            memory += self.weight.element_size() * self.weight.nelement()
        if self._sqrt_weight is not None:
            memory += self._sqrt_weight.element_size() * self._sqrt_weight.nelement()
        return memory

    def __str__(self) -> str:
        size, unit = format_byte_size(self.required_memory)
        ms = (
            f"SVD of a {self._rows}x{self._cols} data matrix",
            f"Selected/optimal rank: {self.rank}/{self.opt_rank}",
            f"data type: {self.U.dtype} ({self.U.element_size()}b)",
            "truncated SVD size: {:1.4f}{:s}".format(size, unit),
        )
        return "\n".join(ms)
