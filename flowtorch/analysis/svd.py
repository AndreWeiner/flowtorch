"""Classes and functions wrapping around *torch.linalg.svd*."""

# standard library packages
import logging
from contextlib import closing
from math import isfinite, sqrt, prod
from numbers import Integral, Real
from os import PathLike, sysconf
from pathlib import Path
from socket import gethostname
from tempfile import TemporaryDirectory
import shutil
from typing import Any, NamedTuple, Optional, Sequence, Tuple, Union

# third party packages
import numpy as np
import torch as pt
import torch.distributed as dist

# flowtorch packages
from flowtorch.utils import format_byte_size
from .state_vector import (
    FieldSpec,
    StateVectorLayout,
    StateVectorResult,
    StateVectorSource,
)
from .statistics import DistributedExecution
from ._chunk_prefetch import iter_prepared_chunks

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


_UNSET_WEIGHT = object()

MODES = ("auto", "svd", "evd", "tsqr")
SVD_CHECKPOINT_VERSION = 2
SVD_DEFAULT_INPUT_BYTES = 256 * 1024**2


class _BasisStore:
    """Owned chunk storage; disk reads materialize only overlapping spatial rows."""

    def __init__(self, layout, mode="disk", directory=None):
        if mode not in ("memory", "disk"):
            raise ValueError("basis_storage must be 'memory' or 'disk'")
        self.layout = layout
        self.mode = mode
        self.directory = directory
        self._temp = TemporaryDirectory(dir=directory) if mode == "disk" else None
        self.chunks = {}

    def put(self, spatial_slice, **values):
        key = (spatial_slice.start, spatial_slice.stop)
        if self.mode == "memory":
            self.chunks[key] = {
                name: value.detach().clone() if value is not None else None
                for name, value in values.items()
            }
        else:
            record: dict[str, Any] = {}
            assert self._temp is not None
            for name, value in values.items():
                if value is None:
                    record[name] = None
                else:
                    path = Path(self._temp.name) / f"{key[0]}-{key[1]}-{name}.npy"
                    np.save(path, value.detach().cpu().numpy())
                    record[name] = path
            self.chunks[key] = record

    def restore_chunk(self, spatial_slice, files, device):
        """Copy checkpoint arrays without loading disk-backed modes into RAM."""
        if self.mode == "disk":
            assert self._temp is not None
            record: dict[str, Any] = {}
            for name, source_path in files.items():
                if source_path is None:
                    record[name] = None
                else:
                    target = (
                        Path(self._temp.name)
                        / f"{spatial_slice.start}-{spatial_slice.stop}-{name}.npy"
                    )
                    shutil.copyfile(source_path, target)
                    record[name] = target
            self.chunks[(spatial_slice.start, spatial_slice.stop)] = record
        else:
            values = {
                name: (
                    None
                    if path is None
                    else pt.from_numpy(
                        np.array(np.load(path, mmap_mode="r"), copy=True)
                    ).to(device)
                )
                for name, path in files.items()
            }
            self.put(spatial_slice, **values)

    def read(self, spatial_slice, name, device):
        start, stop, step = spatial_slice.indices(self.layout.spatial_shape[0])
        if step != 1:
            raise ValueError("basis reads require contiguous spatial slices")
        components = sum(field.n_components for field in self.layout.fields)
        remaining = prod(self.layout.spatial_shape[1:])
        pieces = []
        covered = 0
        for (left, right), record in sorted(self.chunks.items()):
            lo, hi = max(left, start), min(right, stop)
            if lo >= hi:
                continue
            value = record[name]
            if value is None:
                return None
            if self.mode == "disk":
                array = np.load(value, mmap_mode="r")
                trailing = array.shape[1:]
                selected = array.reshape(
                    components, right - left, remaining, *trailing
                )[:, lo - left : hi - left]
                tensor = pt.from_numpy(np.array(selected, copy=True)).to(device)
            else:
                trailing = value.shape[1:]
                tensor = value.reshape(components, right - left, remaining, *trailing)[
                    :, lo - left : hi - left
                ].to(device)
            pieces.append(tensor)
            covered += hi - lo
        if covered != stop - start:
            raise ValueError("requested basis rows are outside the local partition")
        return pt.cat(pieces, dim=1).reshape(
            components * covered * remaining, *trailing
        )

    def close(self):
        if self._temp is not None:
            self._temp.cleanup()
        self.chunks.clear()


class _FactorSource(StateVectorSource):
    """Layout-only source: factors own the data needed after construction."""

    def __init__(self, layout, count):
        self._layout = layout
        self._count = count

    @property
    def layout(self):
        return self._layout

    @property
    def n_snapshots(self):
        return self._count

    def read(self, spatial_slice, snapshot_slice):
        raise RuntimeError("original snapshots are not retained by this SVD")


def _partition_spatial_axis(size: int, rank: int, world_size: int) -> tuple[int, int]:
    """Return a balanced contiguous first-axis partition."""
    common, remainder = divmod(size, world_size)
    start = rank * common + min(rank, remainder)
    stop = start + common + (1 if rank < remainder else 0)
    return start, stop


def _iter_slices(start: int, stop: int, batch_size: int):
    for current in range(start, stop, batch_size):
        yield slice(current, min(current + batch_size, stop))


def _available_memory_bytes(device: pt.device) -> int:
    """Available device memory, respecting Linux container RAM limits."""
    if device.type == "cuda":
        return pt.cuda.mem_get_info(device)[0]
    try:
        info = Path("/proc/meminfo").read_text()
        available = next(
            int(line.split()[1]) * 1024
            for line in info.splitlines()
            if line.startswith("MemAvailable:")
        )
    except (OSError, StopIteration):
        available = sysconf("SC_AVPHYS_PAGES") * sysconf("SC_PAGE_SIZE")
    # Locate this process's cgroup rather than assuming the mount root.
    try:
        entry = next(
            line
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        )
        directory = Path("/sys/fs/cgroup") / entry.split("::", 1)[1].lstrip("/")
        root = Path("/sys/fs/cgroup")
        while directory == root or root in directory.parents:
            limit = (directory / "memory.max").read_text().strip()
            if limit != "max":
                used = int((directory / "memory.current").read_text())
                available = min(available, max(0, int(limit) - used))
            if directory == root:
                break
            directory = directory.parent
    except (OSError, StopIteration, ValueError):
        pass
    return available


def _initial_memory_capacity(reports: Sequence[dict], columns: int) -> int:
    """Common spatial capacity from each rank's per-process allowance.

    Host RAM is shared only among ranks on that host. Cgroup allowances are
    conservatively treated as shared host memory; explicit budgets bypass
    both sharing and the automatic safety factor.
    """
    capacities = []
    for report in reports:
        budget = report["budget"]
        if budget is None:
            available = report["available"]
            if report["device_type"] == "cpu":
                local_ranks = sum(
                    other["hostname"] == report["hostname"] for other in reports
                )
                available //= local_ranks
            budget = int(0.6 * available)
        scalar_bytes = report["scalar_bytes"]
        retained = report.get("maximum_rank", columns)
        fixed_bytes = (len(reports) + 6) * (retained + columns) ** 2 * scalar_bytes
        fixed_bytes += 4 * columns * retained * scalar_bytes
        fixed_bytes += report.get("basis_reserve", 0)
        block_bytes = report["rows_per_value"] * columns * scalar_bytes
        capacities.append(
            (budget - fixed_bytes)
            // (
                6 * block_bytes
                + 8 * report["rows_per_value"] * retained * scalar_bytes
                + (
                    2
                    * report["rows_per_value"]
                    * scalar_bytes
                    * (columns + retained + 2)
                    if report.get("prefetch", False) and report["device_type"] == "cpu"
                    else 0
                )
            )
        )
        if report.get("prefetch", False) and report["device_type"] == "cuda":
            host = report.get("host_available")
            if host is not None:
                peers = sum(
                    other["hostname"] == report["hostname"] for other in reports
                )
                capacities.append(
                    int(0.6 * host / peers)
                    // (
                        4
                        * report["rows_per_value"]
                        * scalar_bytes
                        * (columns + retained + 4)
                    )
                )
    return min(capacities)


def _merge_triangular(first: Optional[pt.Tensor], second: pt.Tensor) -> pt.Tensor:
    """Merge two TSQR triangular factors."""
    if first is None:
        return pt.linalg.qr(second, mode="r").R
    return pt.linalg.qr(pt.cat((first, second), dim=0), mode="r").R


def _reduce_triangular(
    local: pt.Tensor,
    execution: Optional[DistributedExecution],
) -> pt.Tensor:
    """Merge rank-local factors identically on every rank.

    The factors are only snapshot-by-snapshot in size.  Using ``all_gather``
    keeps this collective portable across Gloo, MPI, and NCCL while the tall
    spatial factors remain distributed and out of core.
    """
    if execution is None:
        return local
    group = execution.process_group
    world_size = dist.get_world_size(group)
    gathered = [pt.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local, group=group)
    factor = None
    for current in gathered:
        factor = _merge_triangular(factor, current)
    assert factor is not None
    return factor


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
    torch.Size([400, 5])

    A :class:`~flowtorch.analysis.state_vector.StateVectorSource` selects the
    out-of-core TSQR path.  The same call is spatially distributed when an
    initialized :class:`~flowtorch.analysis.statistics.DistributedExecution`
    is supplied::

        source = DataloaderStateVectorSource(
            loader,
            (FieldSpec("U", (3,)), FieldSpec("p")),
            normalization={"U": u_ref, "p": p_ref},
        )
        svd = SVD(
            source,
            rank=20,
            subtract_mean=True,
            spatial_batch_size=100_000,
            execution=execution,  # omit for shared-memory execution
        )
        for spatial_slice, fields in svd.U.iter_chunks(split=True):
            consume(fields["U"], fields["p"])
    """

    def __init__(
        self,
        data_matrix: Union[pt.Tensor, StateVectorSource],
        rank: Union[int, None] = None,
        mode: str = "auto",
        weight: Union[pt.Tensor, None] = None,
        *,
        subtract_mean: bool = False,
        spatial_batch_size: Optional[int] = None,
        execution: Optional[DistributedExecution] = None,
        memory_budget_bytes: Optional[int] = None,
        prefetch_chunks: bool = False,
        basis_storage: str = "disk",
        basis_directory: Optional[Union[str, PathLike[str]]] = None,
    ):
        r"""Compute a truncated, optionally weighted SVD of a data matrix.

        With a diagonal spatial weight ``W``, the decomposition is computed for
        ``W**0.5 @ data_matrix``. :attr:`U` contains physical modes satisfying
        ``U.conj().T @ W @ U = I``, while :attr:`U_weighted` contains their
        Euclidean-orthonormal weighted-coordinate representation. Consequently,
        :meth:`reconstruct` always returns data in the original coordinates.

        :param data_matrix: data matrix of shape M x N or a lazy state-vector
            source. Lazy sources use out-of-core TSQR and partition the first
            spatial dimension when ``execution`` is supplied.
        :type data_matrix: Union[pt.Tensor, StateVectorSource]
        :param rank: truncation rank for tensor inputs; maximum retained rank
            across updates for source-backed TSQR. None selects a rank by
            singular value hard thresholding after each factorization.
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
        :param subtract_mean: subtract and retain the temporal mean, defaults to
            False
        :param spatial_batch_size: maximum number of first-axis spatial values
            loaded per rank and batch for a lazy source. If omitted, target
            256 MiB of input, bounded by estimated memory and the local partition.
        :param execution: distributed execution configuration. Every rank must
            construct the same source; the spatial dimension is partitioned,
            while time is retained on every rank.
        :param memory_budget_bytes: per-process construction/update memory budget, including
            QR buffers and spatial chunk workspace. Each incoming temporal batch
            must fit the loader's memory requirements independently.
            If omitted, uses 60% of available device memory (CPU RAM shared
            among ranks on the same host), bounded by the tightest rank's
            allowance. This is an estimate, not a hard memory limit;
            loaders may allocate additional buffers or read larger files.
        :param prefetch_chunks: prepare one CPU spatial chunk ahead, default False.
            True enables prefetching; otherwise preparation is sequential. GPU-only custom readers remain
            synchronous unless they implement CPU preparation hooks.
        :param basis_storage: "disk" (default) or "memory" for local TSQR modes
        :param basis_directory: parent directory for owned temporary basis files;
            defaults to the system temporary directory. Checkpoints are independent.
        :raises ValueError:
            - if the data matrix does not have exactly two dimensions
            - if an invalid compute mode is specified
            - if the weight is invalid or incompatible with the state dimension
        """
        if isinstance(data_matrix, StateVectorSource):
            self._init_source(
                data_matrix,
                rank,
                mode,
                weight,
                subtract_mean,
                spatial_batch_size,
                execution,
                memory_budget_bytes,
                basis_storage,
                basis_directory,
                prefetch_chunks,
            )
            return
        if execution is not None:
            raise ValueError(
                "distributed SVD requires a StateVectorSource on every rank"
            )
        input_mean = data_matrix.mean(dim=1) if subtract_mean else None
        if input_mean is not None:
            data_matrix = data_matrix - input_mean.unsqueeze(-1)
        self._source: Optional[StateVectorSource] = None
        self._subtract_mean = subtract_mean
        self._execution: Optional[DistributedExecution] = None
        self._mean = input_mean
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
            if input_mean is not None and not (
                pt.is_grad_enabled()
                and (data_matrix.requires_grad or self._sqrt_weight.requires_grad)
            ):
                # Centering owns this buffer; weighting can reuse it without
                # modifying the caller's input.
                weighted_data.mul_(self._sqrt_weight)
            else:
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
            U_weighted, s, V = self._gram_evd(weighted_data, rank)
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

    @staticmethod
    def _validate_batch_size(value: Optional[int], default: int, name: str) -> int:
        selected = default if value is None else value
        if (
            not isinstance(selected, Integral)
            or isinstance(selected, bool)
            or selected < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
        return int(selected)

    def _init_source(
        self,
        source: StateVectorSource,
        rank: Optional[int],
        mode: str,
        weight: Optional[pt.Tensor],
        subtract_mean: bool,
        spatial_batch_size: Optional[int],
        execution: Optional[DistributedExecution],
        memory_budget_bytes: Optional[int],
        basis_storage: str,
        basis_directory: Optional[Union[str, PathLike[str]]],
        prefetch_chunks: bool,
    ) -> None:
        """Compute an out-of-core tall-skinny SVD."""
        if mode not in ("auto", "tsqr"):
            raise ValueError("StateVectorSource inputs require mode 'auto' or 'tsqr'")
        if not isinstance(subtract_mean, bool):
            raise ValueError("subtract_mean must be a boolean")
        if memory_budget_bytes is not None:
            memory_budget_bytes = self._validate_batch_size(
                memory_budget_bytes, 4 * 1024**3, "memory_budget_bytes"
            )
        if execution is not None:
            if not isinstance(execution, DistributedExecution):
                raise ValueError("execution must be a DistributedExecution instance")
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("torch.distributed must be initialized")
            group = execution.process_group
            process_rank = dist.get_rank(group)
            world_size = dist.get_world_size(group)
            if execution.root_rank < 0 or execution.root_rank >= world_size:
                raise ValueError("root_rank must identify a rank in the process group")
        else:
            process_rank = 0
            world_size = 1
        if world_size > source.spatial_shape[0]:
            raise ValueError(
                "the process group cannot contain more ranks than first-axis spatial values"
            )
        if weight is not None:
            if tuple(weight.shape) != source.spatial_shape:
                raise ValueError(
                    "a source-backed weight must match the source spatial shape"
                )
            if weight.device.type != "cpu":
                raise ValueError("source-backed global weights must be stored on CPU")
        if not isinstance(prefetch_chunks, bool):
            raise ValueError("prefetch_chunks must be a boolean")
        self._prefetch_chunks = prefetch_chunks
        self._basis = _BasisStore(source.layout, basis_storage, basis_directory)
        self._memory_budget_bytes = memory_budget_bytes
        self._maximum_rank = (
            None if rank is None else self._validate_batch_size(rank, 1, "rank")
        )
        self._source = source
        self._source_weight = weight
        self._external_weight = weight is not None
        self._subtract_mean = subtract_mean
        self._execution = execution
        self._rows = source.layout.state_size
        self._cols = source.n_snapshots
        self._mode = "tsqr"
        self._spatial_batch_size = self._validate_batch_size(
            spatial_batch_size, min(1024, source.spatial_shape[0]), "spatial_batch_size"
        )
        start, stop = _partition_spatial_axis(
            source.spatial_shape[0], process_rank, world_size
        )
        self._local_spatial_slice = slice(start, stop)
        source.fit_normalization(
            self._local_spatial_slice,
            self._spatial_batch_size,
            source.n_snapshots,
            execution,
        )
        self._basis.layout = source.layout
        if spatial_batch_size is None:
            self._select_spatial_batch(memory_budget_bytes, world_size)
        self._fit_source_factors(rank)
        self._source = _FactorSource(source.layout, self._cols)
        self._source_weight = None
        self._mean = self._mean_result() if subtract_mean else None

    def _select_spatial_batch(self, budget: Optional[int], world_size: int) -> None:
        """Select an input-byte target without timing trials or extra factorization."""
        assert self._source is not None
        group = None if self._execution is None else self._execution.process_group
        report: dict[str, Any] = {}
        try:
            start = self._local_spatial_slice.start
            probe = self._source.read(slice(start, start + 1), slice(0, 1))
            device = probe.device
            scalar_bytes = probe.element_size()
            packed = probe.shape[0]
            maximum = min(self._rows, self._maximum_rank or self._cols)
            report = dict(
                hostname=gethostname(),
                device_type=device.type,
                scalar_bytes=scalar_bytes,
                rows_per_value=packed,
                maximum_rank=maximum,
                budget=budget,
                prefetch=self._prefetch_chunks,
                available=_available_memory_bytes(device) if budget is None else None,
                local_size=self._local_spatial_slice.stop - start,
            )
            if self._basis.mode == "memory":
                report["basis_reserve"] = (
                    report["local_size"]
                    * packed
                    * scalar_bytes
                    * (2 * maximum + self._cols + 3 * int(self._subtract_mean) + 1)
                )
            if self._prefetch_chunks and device.type == "cuda":
                report["host_available"] = _available_memory_bytes(pt.device("cpu"))
            del probe
        except Exception as error:
            report["error"] = str(error)
        reports: list[Any] = [report]
        if self._execution is not None:
            reports = [None] * world_size
            dist.all_gather_object(reports, report, group=group)
        for index, item in enumerate(reports):
            if "error" in item:
                raise ValueError(f"rank {index}: {item['error']}")
        capacity = _initial_memory_capacity(reports, self._cols)
        if capacity < 1:
            raise ValueError(
                "memory_budget_bytes is too small for factors and spatial workspace"
            )
        targets = [
            max(
                1,
                SVD_DEFAULT_INPUT_BYTES
                // (item["rows_per_value"] * self._cols * item["scalar_bytes"]),
            )
            for item in reports
        ]
        self._spatial_batch_size = int(
            min(capacity, min(targets), min(item["local_size"] for item in reports))
        )

    def _source_block(
        self,
        source: StateVectorSource,
        spatial_slice: slice,
        *,
        subtract_mean: Optional[bool] = None,
    ) -> tuple[pt.Tensor, Optional[pt.Tensor]]:
        """Load all temporal columns for one spatial chunk."""
        # Own a contiguous block and force lazy/mapped pages to materialize
        # here so source-loading timings include the actual data access.
        block = source.read(spatial_slice, slice(0, source.n_snapshots)).clone(
            memory_format=pt.contiguous_format
        )
        center = self._subtract_mean if subtract_mean is None else subtract_mean
        mean = block.mean(dim=1) if center else None
        if mean is not None:
            block = block - mean.unsqueeze(-1)
        return block, mean

    def _weight_block(
        self,
        source: StateVectorSource,
        spatial_slice: slice,
        reference: pt.Tensor,
        *,
        raw_weight: Any = _UNSET_WEIGHT,
    ) -> Optional[pt.Tensor]:
        if isinstance(source, _FactorSource):
            return self._basis.read(spatial_slice, "weight", reference.device)
        weight = (
            source.read_weight(spatial_slice)
            if raw_weight is _UNSET_WEIGHT
            else raw_weight
        )
        if weight is not None and self._source_weight is not None:
            raise ValueError("weight is defined by both the source and SVD")
        if weight is None and self._source_weight is not None:
            weight = self._source_weight[spatial_slice].to(reference.device)
        if weight is None:
            return None
        start, stop, _ = spatial_slice.indices(source.spatial_shape[0])
        expected = (stop - start, *source.spatial_shape[1:])
        if tuple(weight.shape) != expected:
            raise ValueError(
                f"spatial weight has shape {tuple(weight.shape)}; expected {expected}"
            )
        if pt.is_complex(weight) or weight.dtype == pt.bool:
            raise ValueError("weight must have a real numeric dtype")
        if not bool(pt.isfinite(weight).all()) or bool((weight <= 0.0).any()):
            raise ValueError("weight values must be finite and positive")
        repeats = sum(field.n_components for field in source.layout.fields)
        return (
            weight.reshape(-1)
            .repeat(repeats)
            .to(device=reference.device, dtype=reference.real.dtype)
        )

    def _weighted_block(
        self, source: StateVectorSource, spatial_slice: slice, block: pt.Tensor
    ) -> pt.Tensor:
        weight = self._weight_block(source, spatial_slice, block)
        return block if weight is None else block * weight.sqrt().unsqueeze(-1)

    def _source_batches(
        self,
        source: StateVectorSource,
        size: Optional[int] = None,
        stop: Optional[int] = None,
    ):
        """CPU preparation; transfer and processing occur only on consumption."""
        region = (
            self._local_spatial_slice
            if stop is None
            else slice(self._local_spatial_slice.start, stop)
        )
        with closing(
            source.iter_chunks(
                region,
                self._spatial_batch_size if size is None else size,
                prefetch=self._prefetch_chunks,
            )
        ) as chunks:
            for part, host, raw_weight in chunks:
                device = source.device or host.device
                block = host.to(device)
                mean = block.mean(dim=1) if self._subtract_mean else None
                if mean is not None:
                    block = block - mean[:, None]
                weight = self._weight_block(source, part, block, raw_weight=raw_weight)
                yield part, block, mean, weight

    def _update_batches(self, staged=None, source=None):
        """Prepare disk basis and staged data together; GPU-resident stores stay direct."""
        resident_gpu = self._basis.mode == "memory" and self.V.device.type == "cuda"
        read_device = self.V.device if resident_gpu else pt.device("cpu")

        def read(part):
            packet = dict(part=part)
            if not resident_gpu:
                packet.update(
                    U=self._basis.read(part, "U", read_device),
                    weight=self._basis.read(part, "weight", read_device),
                    mean=(
                        self._basis.read(part, "mean", read_device)
                        if self._subtract_mean
                        else None
                    ),
                )
            if source is not None:
                packet["added"] = source.read_cpu(
                    part, slice(0, source.n_snapshots)
                ).clone(memory_format=pt.contiguous_format)
                packet["new_weight"] = source.read_weight_cpu(part)
            elif not resident_gpu:
                packet["added"] = staged.read(part, "added", read_device)
                packet["next_mean"] = staged.read(part, "mean", read_device)
            return packet

        with closing(
            iter_prepared_chunks(
                self._basis_slices(),
                read,
                prefetch=self._prefetch_chunks
                and (not resident_gpu or source is not None),
            )
        ) as chunks:
            for packet in chunks:
                part = packet["part"]
                if resident_gpu:
                    packet.update(
                        U=self._basis.read(part, "U", read_device),
                        weight=self._basis.read(part, "weight", read_device),
                        mean=(
                            self._basis.read(part, "mean", read_device)
                            if self._subtract_mean
                            else None
                        ),
                    )
                    if staged is not None:
                        packet["added"] = staged.read(part, "added", read_device)
                        packet["next_mean"] = staged.read(part, "mean", read_device)
                yield {
                    name: (
                        value.to(self.V.device)
                        if isinstance(value, pt.Tensor)
                        else value
                    )
                    for name, value in packet.items()
                }

    def _local_triangular_factor(
        self,
        source: StateVectorSource,
        columns: int,
        block_transform=None,
    ) -> pt.Tensor:
        """Accumulate one rank-local TSQR factor from spatial chunks."""
        triangular = None
        reference = None
        with closing(self._source_batches(source)) as chunks:
            for spatial_slice, block, _, weight in chunks:
                if block_transform is not None:
                    block = block_transform(spatial_slice, block)
                if weight is not None:
                    block = block * weight.sqrt()[:, None]
                reference = block
                triangular = _merge_triangular(triangular, block)
        if reference is None or triangular is None:
            raise RuntimeError("the local spatial partition is empty")
        padded = reference.new_zeros((columns, columns))
        padded[: triangular.shape[0]] = triangular
        return padded

    def _fit_source_factors(self, requested_rank: Optional[int]) -> None:
        assert self._source is not None
        error = None
        local = None
        try:
            local = self._local_triangular_factor(self._source, self._cols)
        except Exception as exc:
            if self._execution is None:
                raise
            error = str(exc)
        self._collective_error(error)
        assert local is not None
        triangular = _reduce_triangular(local, self._execution)
        root = (
            self._execution is None
            or dist.get_rank(self._execution.process_group) == self._execution.root_rank
        )
        if root:
            _, singular_values, right_h = pt.linalg.svd(triangular, full_matrices=False)
            right = right_h.conj().T
        else:
            singular_values = triangular.real.new_empty(self._cols)
            right = triangular.new_empty((self._cols, self._cols))
        if self._execution is not None:
            group = self._execution.process_group
            global_ranks: list[Any] = [None] * dist.get_world_size(group)
            dist.all_gather_object(global_ranks, dist.get_rank(), group=group)
            root_global = global_ranks[self._execution.root_rank]
            dist.broadcast(singular_values, src=root_global, group=group)
            dist.broadcast(right, src=root_global, group=group)
        self._opt_rank = self._optimal_rank(singular_values)
        self.rank = self.opt_rank if requested_rank is None else requested_rank
        # Null singular directions cannot be recovered through X V / s.
        # Retain only resolved directions; the requested ceiling is unchanged.
        tolerance = (
            pt.finfo(singular_values.dtype).eps
            * max(triangular.shape)
            * singular_values[0]
        )
        self._rank = max(1, min(self.rank, int((singular_values > tolerance).sum())))
        self._s_full = singular_values
        self._s = singular_values[: self.rank].clone().contiguous()
        self._V = right[:, : self.rank].contiguous()
        self._weight = None
        self._sqrt_weight = None
        source = self._source
        error = None
        try:
            with closing(self._source_batches(source)) as chunks:
                for spatial_slice, block, mean, weight_chunk in chunks:
                    denominator = self.s.clamp_min(
                        pt.finfo(self.s.dtype).eps * self.s[0]
                    ).clamp_min(pt.finfo(self.s.dtype).tiny)
                    modes = block @ (self.V / denominator)
                    self._basis.put(
                        spatial_slice, U=modes, mean=mean, weight=weight_chunk
                    )
        except Exception as exc:
            error = str(exc)
        self._collective_error(error)
        self._mean = self._mean_result() if self._subtract_mean else None
        logger.info(
            f"Truncating TSQR SVD at index {self.rank}/{min(self._cols, self._rows)}"
        )

    def _left_chunk(
        self, spatial_slice: slice, rank: Optional[int] = None
    ) -> pt.Tensor:
        modes = self._basis.read(spatial_slice, "U", self.V.device)
        r_rank = self.rank if rank is None else max(min(rank, self.rank), 1)
        return modes[:, :r_rank]

    def _basis_slices(self):
        yield from _iter_slices(
            self._local_spatial_slice.start,
            self._local_spatial_slice.stop,
            self._spatial_batch_size,
        )

    def _mean_result(self) -> StateVectorResult:
        assert self._source is not None
        return StateVectorResult(
            self._source.layout,
            self._local_spatial_slice,
            (),
            lambda spatial_slice: self._basis.read(
                spatial_slice, "mean", self.V.device
            ),
            self._spatial_batch_size,
            self._execution,
        )

    @property
    def maximum_rank(self) -> Optional[int]:
        """Requested TSQR rank ceiling; None selects the rank automatically."""
        return getattr(self, "_maximum_rank", None)

    @property
    def basis_storage(self) -> Optional[str]:
        """Local TSQR basis storage mode, or None for a tensor decomposition."""
        return self._basis.mode if self._source is not None else None

    def close(self) -> None:
        """Release owned temporary basis files. Saved checkpoints are independent."""
        if hasattr(self, "_basis"):
            self._basis.close()

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

    def _gram_evd(
        self, X: pt.Tensor, rank: Optional[int] = None
    ) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        """Compute the SVD via Gram matrix and eigendecomposition.

        :param X: data matrix of shape M x N
        :type X: pt.Tensor
        :return: truncated factors and the complete singular-value spectrum:
            L = min(M, N)
            U: M x rank
            s: L
            V: N x rank
        :rtype: Tuple[pt.Tensor, pt.Tensor, pt.Tensor]
        """
        logger.info("Computing economy SVD via Gram matrix and eigendecomposition")
        eps = pt.finfo(X.dtype).eps
        if self._rows > self._cols:
            C = X.conj().T @ X
            evals, V = pt.linalg.eigh(C)
        else:
            C = X @ X.conj().T
            evals, U = pt.linalg.eigh(C)
        del C
        s = evals.clamp(0.0).sqrt().flip(0)
        self._opt_rank = self._optimal_rank(s)
        self.rank = self.opt_rank if rank is None else rank
        # eigh orders eigenvectors by increasing eigenvalue. Select only the
        # retained columns before reversing and forming the large factor.
        start = s.numel() - self.rank
        denominator = s[: self.rank].clamp(eps * s[0]).clamp_min(pt.finfo(s.dtype).tiny)
        if self._rows > self._cols:
            V = V[:, start:].flip(1).contiguous()
            U = X @ (V / denominator)
        else:
            U = U[:, start:].flip(1).contiguous()
            V = X.conj().T @ (U / denominator)
        return U, s, V

    def _optimal_rank(self, s: pt.Tensor, columns: Optional[int] = None) -> int:
        """Compute the optimal singular value hard threshold.

        This function implements the svht_ rank estimation.

        .. _svht: https://doi.org/10.1109/TIT.2014.2323359

        :param s: sorted singular values
        :type s: pt.Tensor
        :return: optimal rank for truncation
        :rtype: int
        """
        columns = self._cols if columns is None else columns
        beta = min(self._rows, columns) / max(self._rows, columns)
        omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
        tau_star = omega * pt.median(s)
        closest = int(pt.argmin((s - tau_star).abs()).item())
        if s[closest] > tau_star:
            return closest + 1
        else:
            return closest

    def reconstruct(self, rank: Union[int, None] = None) -> Any:
        """Reconstruct the data matrix for a given rank.

        :param rank: rank used to compute a truncated reconstruction
        :type rank: int, optional
        :return: reconstruction of the input data matrix
        :rtype: pt.Tensor
        """
        r_rank = self.rank if rank is None else max(min(rank, self.rank), 1)
        if self._source is not None:

            def produce(spatial_slice: slice) -> pt.Tensor:
                modes = self._left_chunk(spatial_slice, r_rank)
                right_h = self.V[:, :r_rank].conj().T
                if modes.shape[0] > self._cols:
                    return modes @ (self.s[:r_rank, None] * right_h)
                return (modes * self.s[:r_rank]) @ right_h

            return StateVectorResult(
                self._source.layout,
                self._local_spatial_slice,
                (self._cols,),
                produce,
                self._spatial_batch_size,
                self._execution,
            )
        modes = self.U[:, :r_rank]
        right_h = self.V[:, :r_rank].conj().T
        if self._rows > self._cols:
            return modes @ (self.s[:r_rank, None] * right_h)
        return (modes * self.s[:r_rank]) @ right_h

    def encode(self, state: Union[pt.Tensor, StateVectorSource]) -> pt.Tensor:
        r"""Project one or more physical states onto the retained POD modes.

        Centering and the weighted inner product used to compute the SVD are
        applied consistently.  Source-backed projections are reduced across
        the configured process group, so every rank receives the same modal
        coefficients.
        """
        if self._source is None:
            if not isinstance(state, pt.Tensor):
                raise ValueError("a tensor-backed SVD expects tensor states")
            if state.ndim not in (1, 2) or state.shape[0] != self._rows:
                raise ValueError("state has an incompatible shape")
            centered = state
            trailing = (1,) if state.ndim == 2 else ()
            if self.mean is not None:
                assert isinstance(self.mean, pt.Tensor)
                centered = state - self.mean.reshape(-1, *trailing)
            if self.weight is not None:
                centered = centered * self.weight.reshape(-1, *trailing)
            assert isinstance(self.U, pt.Tensor)
            return self.U.conj().T @ centered

        if not isinstance(state, StateVectorSource):
            raise ValueError("a source-backed SVD expects a StateVectorSource")
        if state.layout.signature != self._source.layout.signature:
            raise ValueError("state source layout is incompatible with the SVD")
        modes = self.U
        assert isinstance(modes, StateVectorResult)
        coefficient: Optional[pt.Tensor] = None
        for spatial_slice, mode_chunk in modes.iter_chunks():
            assert isinstance(mode_chunk, pt.Tensor)
            data_chunk = state.read(spatial_slice, slice(0, state.n_snapshots))
            if self.mean is not None:
                assert isinstance(self.mean, StateVectorResult)
                mean_chunk = self.mean.read_chunk(spatial_slice)
                assert isinstance(mean_chunk, pt.Tensor)
                data_chunk = data_chunk - mean_chunk.unsqueeze(-1)
            weight = self._weight_block(self._source, spatial_slice, mode_chunk)
            if weight is not None:
                data_chunk = data_chunk * weight.unsqueeze(-1)
            local = mode_chunk.conj().T @ data_chunk
            coefficient = local if coefficient is None else coefficient + local
        if coefficient is None:
            raise RuntimeError("the local SVD partition is empty")
        if self._execution is not None:
            dist.all_reduce(coefficient, group=self._execution.process_group)
        return coefficient.squeeze(-1) if state.n_snapshots == 1 else coefficient

    def decode(self, coefficients: pt.Tensor) -> Union[pt.Tensor, StateVectorResult]:
        r"""Expand arbitrary POD coefficients into the physical state space."""
        if coefficients.ndim < 1 or coefficients.shape[0] != self.rank:
            raise ValueError("coefficients have an incompatible shape")
        trailing = coefficients.shape[1:]
        flattened = coefficients.reshape(self.rank, -1)
        if self._source is None:
            assert isinstance(self.U, pt.Tensor)
            result = (self.U @ flattened).reshape(self._rows, *trailing)
            if self.mean is not None:
                assert isinstance(self.mean, pt.Tensor)
                result = result + self.mean.reshape(-1, *((1,) * len(trailing)))
            return result

        modes = self.U
        assert isinstance(modes, StateVectorResult)
        source = self._source

        def produce(spatial_slice: slice) -> pt.Tensor:
            value = (modes.read_chunk(spatial_slice) @ flattened).reshape(-1, *trailing)
            if self.mean is not None:
                assert isinstance(self.mean, StateVectorResult)
                mean_chunk = self.mean.read_chunk(spatial_slice)
                assert isinstance(mean_chunk, pt.Tensor)
                value = value + mean_chunk.reshape(-1, *((1,) * len(trailing)))
            return value

        return StateVectorResult(
            source.layout,
            modes.local_spatial_slice,
            trailing,
            produce,
            modes.spatial_batch_size,
            self._execution,
        )

    def _apply_tensor_transform(self, transform) -> "SVD":
        """Apply a device/dtype transform to a tensor-backed decomposition.

        This mirrors the tensor handling used by ``torch.nn.Module._apply``
        without making an analytical SVD a trainable module.
        """
        if self._source is not None:
            raise RuntimeError(
                "device and dtype transforms are not supported for source-backed SVDs"
            )
        for name in (
            "_U",
            "_s",
            "_s_full",
            "_V",
            "_mean",
            "_weight",
            "_sqrt_weight",
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, transform(value))
        return self

    @property
    def U(self) -> Any:
        r"""Physical left-singular vectors.

        For a weighted decomposition, these modes are orthonormal in the
        weighted inner product, ``U.conj().T @ W @ U = I``.

        :return: physical left-singular vectors truncated to specified rank
        :rtype: pt.Tensor
        """
        if self._source is None:
            return self._U
        return StateVectorResult(
            self._source.layout,
            self._local_spatial_slice,
            (self.rank,),
            self._left_chunk,
            self._spatial_batch_size,
            self._execution,
        )

    @property
    def U_weighted(self) -> Union[pt.Tensor, StateVectorResult]:
        r"""Euclidean-orthonormal left-singular vectors.

        For a weighted decomposition, this is ``W**0.5 @ U``. Without a
        weight, it is identical to :attr:`U`.

        :return: left-singular vectors in weighted coordinates
        :rtype: pt.Tensor
        """
        if self._source is not None:
            source = self._source

            def produce(spatial_slice: slice) -> pt.Tensor:
                modes = self._left_chunk(spatial_slice)
                weight = self._weight_block(source, spatial_slice, modes)
                return modes if weight is None else modes * weight.sqrt().unsqueeze(-1)

            return StateVectorResult(
                source.layout,
                self._local_spatial_slice,
                (self.rank,),
                produce,
                self._spatial_batch_size,
                self._execution,
            )
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
    def mean(self) -> Optional[Union[pt.Tensor, StateVectorResult]]:
        """Temporal mean retained by a centered decomposition."""
        return self._mean

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

        :return: persistent factor tensor bytes (local for TSQR), excluding
            disk arrays, filesystem cache, and temporary workspace
        :rtype: int
        """
        if self._source is None:
            memory = self._U.element_size() * self._U.nelement()
        else:
            memory = 0
            if self._basis.mode == "memory":
                memory = sum(
                    value.element_size() * value.nelement()
                    for record in self._basis.chunks.values()
                    for value in record.values()
                    if value is not None
                )
        memory += self.s.element_size() * self.s.nelement()
        memory += self.V.element_size() * self.V.nelement()
        if self.weight is not None:
            memory += self.weight.element_size() * self.weight.nelement()
        if self._sqrt_weight is not None:
            memory += self._sqrt_weight.element_size() * self._sqrt_weight.nelement()
        return memory

    def _collective_error(self, error: Optional[str]) -> None:
        """Propagate local I/O failures before entering the next collective."""
        if self._execution is not None:
            group = self._execution.process_group
            errors: list[Any] = [None] * dist.get_world_size(group)
            dist.all_gather_object(errors, error, group=group)
            error = next((value for value in errors if value is not None), None)
        if error is not None:
            raise RuntimeError(error)

    def _small_svd(self, matrix: pt.Tensor):
        """Factor a compact update matrix on the execution root."""
        root = (
            self._execution is None
            or dist.get_rank(self._execution.process_group) == self._execution.root_rank
        )
        error = None
        size = min(matrix.shape)
        if root:
            try:
                left, values, right_h = pt.linalg.svd(matrix, full_matrices=False)
                left, values, right_h = (
                    left.contiguous(),
                    values.contiguous(),
                    right_h.contiguous(),
                )
            except Exception as exc:
                error = str(exc)
        self._collective_error(error)
        if self._execution is not None:
            group = self._execution.process_group
            ranks: list[Any] = [None] * dist.get_world_size(group)
            dist.all_gather_object(ranks, dist.get_rank(), group=group)
            global_root = ranks[self._execution.root_rank]
            if not root:
                left = matrix.new_empty((matrix.shape[0], size))
                values = matrix.real.new_empty(size)
                right_h = matrix.new_empty((size, matrix.shape[1]))
            for tensor in (left, values, right_h):
                dist.broadcast(tensor, src=global_root, group=group)
        return left, values, right_h

    def _update_spatial_capacity(self, batch_count: int) -> int:
        """Reuse the selected size, shrinking only for estimated memory needs."""
        assert self._source is not None
        group = None if self._execution is None else self._execution.process_group
        world_size = 1 if self._execution is None else dist.get_world_size(group)
        columns = batch_count + int(self._subtract_mean)
        next_rank = min(
            self._rows,
            self._cols + batch_count,
            self._maximum_rank or self.rank + columns,
            self.rank + columns,
        )
        packed_per_value = prod(self._source.spatial_shape[1:]) * sum(
            field.n_components for field in self._source.layout.fields
        )
        local_rows = (
            self._local_spatial_slice.stop - self._local_spatial_slice.start
        ) * packed_per_value
        scalar_bytes = self.V.element_size()
        error = None
        report: dict[str, Any] = {}
        try:
            explicit = self._memory_budget_bytes
            report = dict(
                hostname=gethostname(),
                device_type=self.V.device.type,
                available=(
                    _available_memory_bytes(self.V.device) if explicit is None else None
                ),
                budget=None if explicit is None else explicit - self.required_memory,
                # Existing factors already reduce automatically probed
                # free memory. Reserve only newly allocated arrays.
                fixed=(world_size + 6) * (self.rank + columns) ** 2 * scalar_bytes
                + 2 * (self._cols + batch_count) * next_rank * scalar_bytes,
                block_bytes=packed_per_value
                * scalar_bytes
                * (6 * columns + 4 * (self.rank + next_rank)),
            )
            if self._prefetch_chunks and self.V.device.type == "cuda":
                report["host_available"] = _available_memory_bytes(pt.device("cpu"))
                report["host_block_bytes"] = (
                    4 * packed_per_value * scalar_bytes * (columns + self.rank + 4)
                )
            if self._prefetch_chunks and self.V.device.type == "cpu":
                report["block_bytes"] += (
                    2 * packed_per_value * scalar_bytes * (columns + self.rank + 2)
                )
            if self._basis.mode == "memory":
                report["fixed"] += (
                    local_rows
                    * scalar_bytes
                    * (next_rank + batch_count + 3 * int(self._subtract_mean) + 1)
                )
        except Exception as exc:
            error = str(exc)
        self._collective_error(error)
        reports: list[Any] = [report]
        if self._execution is not None:
            reports = [None] * world_size
            dist.all_gather_object(reports, report, group=group)
        capacities = []
        for item in reports:
            allowance = item["budget"]
            if allowance is None:
                available = item["available"]
                if available is None:
                    raise ValueError(
                        "cannot determine update memory; supply memory_budget_bytes"
                    )
                if item["device_type"] == "cpu":
                    available //= sum(
                        other["hostname"] == item["hostname"] for other in reports
                    )
                allowance = int(0.6 * available)
            capacities.append((allowance - item["fixed"]) // item["block_bytes"])
            if item.get("host_available") is not None:
                peers = sum(other["hostname"] == item["hostname"] for other in reports)
                capacities.append(
                    int(0.6 * item["host_available"] / peers)
                    // item["host_block_bytes"]
                )
        capacity = min(capacities)
        if capacity < 1:
            raise ValueError(
                "update memory budget is too small for factors and one spatial value; use a smaller temporal batch or increase memory_budget_bytes"
            )
        return min(self._spatial_batch_size, capacity)

    def update(self, new_source: StateVectorSource) -> "SVD":
        """Append snapshots using stored local modes, including mean corrections.

        Explicit source-backed ranks are ceilings. Intermediate truncation is
        approximate and may differ from truncating a single full factorization.
        The initial spatial size is reused, reduced only if the current memory
        estimate requires it. Updates never run timing trials.
        Old snapshots are never read. The new batch is staged once in owned
        chunk storage and remains available only for the duration of the update.
        """
        if self._source is None:
            raise RuntimeError("update requires a source-backed SVD")
        if not isinstance(new_source, StateVectorSource):
            raise ValueError("new_source must be a StateVectorSource")
        new_source.restore_normalization(self._source.layout.normalization_factors)
        if new_source.layout.signature != self._source.layout.signature:
            raise ValueError("new source layout or normalization is incompatible")
        old_count, batch_count, old_rank = self._cols, new_source.n_snapshots, self.rank
        total = old_count + batch_count
        columns = batch_count + int(self._subtract_mean)
        previous_size = self._spatial_batch_size
        self._spatial_batch_size = self._update_spatial_capacity(batch_count)
        if self._spatial_batch_size < previous_size:
            logger.info(
                "Reducing update spatial_batch_size from %s to %s for memory capacity",
                previous_size,
                self._spatial_batch_size,
            )
        cross = self.V.new_zeros((old_rank, columns))
        staged = _BasisStore(
            self._source.layout, self._basis.mode, self._basis.directory
        )
        replacement = _BasisStore(
            self._source.layout, self._basis.mode, self._basis.directory
        )
        try:
            error = None
            try:
                with closing(self._update_batches(source=new_source)) as chunks:
                    for packet in chunks:
                        spatial_slice = packet["part"]
                        added = packet["added"]
                        batch_mean = added.mean(dim=1) if self._subtract_mean else None
                        if batch_mean is not None:
                            added = added - batch_mean[:, None]
                        modes, weight = packet["U"], packet["weight"]
                        supplied_weight = self._weight_block(
                            new_source,
                            spatial_slice,
                            modes,
                            raw_weight=packet["new_weight"],
                        )
                        if self._external_weight and supplied_weight is None:
                            supplied_weight = weight
                        if (weight is None) != (supplied_weight is None) or (
                            weight is not None
                            and not pt.allclose(weight, supplied_weight)
                        ):
                            raise ValueError(
                                "new source spatial weights are incompatible"
                            )
                        mean = None
                        if self._subtract_mean:
                            old_mean = packet["mean"]
                            delta = batch_mean - old_mean
                            mean = old_mean + delta * (batch_count / total)
                            added = pt.cat(
                                (
                                    added,
                                    delta[:, None]
                                    * sqrt(old_count * batch_count / total),
                                ),
                                dim=1,
                            )
                        staged.put(spatial_slice, added=added, mean=mean)
                        cross += modes.conj().T @ (
                            added if weight is None else added * weight[:, None]
                        )
            except Exception as exc:
                error = str(exc)
            self._collective_error(error)
            if self._execution is not None:
                dist.all_reduce(cross, group=self._execution.process_group)
            triangular = None
            error = None
            try:
                with closing(self._update_batches(staged=staged)) as chunks:
                    for packet in chunks:
                        spatial_slice = packet["part"]
                        residual = packet["added"] - packet["U"] @ cross
                        weight = packet["weight"]
                        if weight is not None:
                            residual = residual * weight.sqrt()[:, None]
                        triangular = _merge_triangular(triangular, residual)
            except Exception as exc:
                error = str(exc)
            self._collective_error(error)
            padded = cross.new_zeros((columns, columns))
            assert triangular is not None
            padded[: triangular.shape[0]] = triangular
            residual_r = _reduce_triangular(padded, self._execution)
            # Residual SVD provides a stable spatial basis, including when the
            # residual is rank deficient (e.g. a centered batch).
            _, residual_s, residual_vh = self._small_svd(residual_r)
            tolerance = (
                pt.finfo(residual_s.dtype).eps
                * max(residual_r.shape)
                * max(float(self.s[0]), float(residual_s[0]))
            )
            active = residual_s > tolerance
            residual_s = residual_s[active]
            residual_vh = residual_vh[active]
            core = cross.new_zeros((old_rank + residual_s.numel(), old_rank + columns))
            core[:old_rank, :old_rank] = pt.diag(self.s).to(core.dtype)
            core[:old_rank, old_rank:] = cross
            core[old_rank:, old_rank:] = residual_s[:, None] * residual_vh
            core_u, updated_s, updated_vh = self._small_svd(core)
            optimal = self._optimal_rank(updated_s, columns=total)
            ceiling = optimal if self._maximum_rank is None else self._maximum_rank
            new_rank = max(1, min(ceiling, self._rows, total, updated_s.numel()))
            core_u = core_u[:, :new_rank]
            updated_v = updated_vh[:new_rank].conj().T
            new_v = self.V.new_empty((total, new_rank))
            new_v[:old_count] = self.V @ updated_v[:old_rank]
            batch_v = updated_v[old_rank : old_rank + batch_count]
            if self._subtract_mean:
                correction = updated_v[-1]
                new_v[:old_count] -= (
                    sqrt(batch_count / (old_count * total)) * correction
                )
                new_v[old_count:] = (
                    batch_v
                    - batch_v.mean(dim=0)
                    + sqrt(old_count / (batch_count * total)) * correction
                )
            else:
                new_v[old_count:] = batch_v
            error = None
            try:
                with closing(self._update_batches(staged=staged)) as chunks:
                    for packet in chunks:
                        spatial_slice = packet["part"]
                        old_modes, added = packet["U"], packet["added"]
                        residual = added - old_modes @ cross
                        residual_modes = residual @ (residual_vh.conj().T / residual_s)
                        modes = (
                            old_modes @ core_u[:old_rank]
                            + residual_modes @ core_u[old_rank:]
                        )
                        replacement.put(
                            spatial_slice,
                            U=modes,
                            mean=packet["next_mean"],
                            weight=packet["weight"],
                        )
            except Exception as exc:
                error = str(exc)
            self._collective_error(error)
            old_basis = self._basis
            self._basis = replacement
            old_basis.close()
            self._cols = total
            self._source = _FactorSource(self._source.layout, total)
            self._rank = new_rank
            self._s_full = updated_s
            self._s = updated_s[:new_rank].clone().contiguous()
            self._V = new_v.contiguous()
            self._opt_rank = optimal
            self._mean = self._mean_result() if self._subtract_mean else None
        except BaseException:
            self._spatial_batch_size = previous_size
            replacement.close()
            raise
        finally:
            staged.close()
        return self

    @staticmethod
    def _layout_metadata(source: StateVectorSource) -> dict[str, Any]:
        """Convert state-vector metadata to checkpoint-safe built-in values."""
        return {
            "spatial_shape": list(source.layout.spatial_shape),
            "fields": [
                {
                    "name": field.name,
                    "component_shape": list(field.component_shape),
                    "component_names": list(field.component_names),
                }
                for field in source.layout.fields
            ],
            "normalization_factors": source.layout.normalization_factors,
        }

    def save(self, path: Union[str, PathLike[str]]) -> None:
        """Save self-contained factors, including the retained physical basis.

        Tensor decompositions use one torch file. Source-backed decompositions
        use a new directory with metadata and one chunk-file shard per rank.
        Distributed saves are collective and require a shared filesystem.
        Existing checkpoint directories are never overwritten.
        """
        checkpoint = {
            "format": "flowtorch.svd",
            "version": SVD_CHECKPOINT_VERSION,
            "kind": "source" if self._source is not None else "tensor",
            "rows": self._rows,
            "cols": self._cols,
            "rank": self.rank,
            "opt_rank": self.opt_rank,
            "mode": self.mode,
            "subtract_mean": self._subtract_mean,
            "s": self.s.detach().cpu(),
            "s_full": self.s_full.detach().cpu(),
            "V": self.V.detach().cpu(),
        }
        if self._source is None:
            checkpoint.update(
                U=self._U.detach().cpu(),
                weight=None if self._weight is None else self._weight.detach().cpu(),
                sqrt_weight=(
                    None
                    if self._sqrt_weight is None
                    else self._sqrt_weight.detach().cpu()
                ),
                mean=None if self._mean is None else self._mean.detach().cpu(),
            )
            pt.save(checkpoint, path)
            return
        group = None if self._execution is None else self._execution.process_group
        process_rank = 0 if self._execution is None else dist.get_rank(group)
        world_size = 1 if self._execution is None else dist.get_world_size(group)
        destination = Path(path)
        # Every rank checks before any rank creates the destination.
        exists = destination.exists()
        if self._execution is not None:
            reports = [None] * world_size
            dist.all_gather_object(reports, exists, group=group)
            exists = any(reports)
        if exists:
            raise FileExistsError(
                f"checkpoint destination already exists: {destination}"
            )
        error = None
        try:
            shard = destination / f"rank-{process_rank}"
            shard.mkdir(parents=True)
            chunks = []
            for (start, stop), record in sorted(self._basis.chunks.items()):
                names: dict[str, Optional[str]] = {}
                for name, value in record.items():
                    if value is None:
                        names[name] = None
                        continue
                    filename = f"{start}-{stop}-{name}.npy"
                    if self._basis.mode == "disk":
                        shutil.copyfile(value, shard / filename)
                    else:
                        np.save(shard / filename, value.detach().cpu().numpy())
                    names[name] = filename
                chunks.append((start, stop, names))
            pt.save(
                {
                    "chunks": chunks,
                    "start": self._local_spatial_slice.start,
                    "stop": self._local_spatial_slice.stop,
                    "memory_budget_bytes": self._memory_budget_bytes,
                },
                shard / "index.pt",
            )
        except Exception as exc:
            error = str(exc)
        if self._execution is not None:
            errors = [None] * world_size
            dist.all_gather_object(errors, error, group=group)
            error = next((item for item in errors if item is not None), None)
        if error is not None:
            raise RuntimeError(f"checkpoint shard write failed: {error}")
        if process_rank == 0:
            checkpoint.update(
                layout=self._layout_metadata(self._source),
                maximum_rank=self._maximum_rank,
                prefetch_chunks=self._prefetch_chunks,
                external_weight=self._external_weight,
                spatial_batch_size=self._spatial_batch_size,
                world_size=world_size,
            )
            try:
                # Metadata is written last and marks a complete checkpoint.
                pt.save(checkpoint, destination / "metadata.pt")
            except Exception as exc:
                error = str(exc)
        if self._execution is not None:
            errors = [None] * world_size
            dist.all_gather_object(errors, error, group=group)
            error = next((item for item in errors if item is not None), None)
        if error is not None:
            raise RuntimeError(f"checkpoint metadata write failed: {error}")

    @classmethod
    def load(
        cls,
        path: Union[str, PathLike[str]],
        *,
        source: Optional[StateVectorSource] = None,
        execution: Optional[DistributedExecution] = None,
        map_location: Any = "cpu",
        spatial_batch_size: Optional[int] = None,
        prefetch_chunks: Optional[bool] = None,
        basis_storage: str = "disk",
        basis_directory: Optional[Union[str, PathLike[str]]] = None,
    ) -> "SVD":
        """Restore self-contained factors without accessing original snapshots.

        Source-backed directory checkpoints restore using the same rank count.
        ``source`` is optional and, when supplied, only validates the layout.
        ``basis_storage`` selects memory or owned temporary disk storage.
        """
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "metadata.pt"
        try:
            checkpoint = pt.load(
                checkpoint_path,
                map_location=map_location,
                weights_only=True,
            )
        except TypeError:  # PyTorch versions before ``weights_only``
            checkpoint = pt.load(checkpoint_path, map_location=map_location)
        if not isinstance(checkpoint, dict):
            raise ValueError("invalid SVD checkpoint")
        if checkpoint.get("format") != "flowtorch.svd":
            raise ValueError("file is not a flowTorch SVD checkpoint")
        if checkpoint.get("version") != SVD_CHECKPOINT_VERSION:
            raise ValueError(
                "unsupported SVD checkpoint version " f"{checkpoint.get('version')!r}"
            )
        kind = checkpoint.get("kind")
        if kind not in ("tensor", "source"):
            raise ValueError("invalid SVD checkpoint kind")
        instance = cls.__new__(cls)
        instance._rows = int(checkpoint["rows"])
        instance._cols = int(checkpoint["cols"])
        instance._rank = int(checkpoint["rank"])
        instance._opt_rank = int(checkpoint["opt_rank"])
        instance._mode = str(checkpoint["mode"])
        instance._subtract_mean = bool(checkpoint["subtract_mean"])
        instance._s = checkpoint["s"]
        instance._s_full = checkpoint["s_full"]
        instance._V = checkpoint["V"]
        if kind == "tensor":
            if source is not None or execution is not None:
                raise ValueError(
                    "source and execution are only valid for source-backed checkpoints"
                )
            instance._source = None
            instance._execution = None
            instance._U = checkpoint["U"]
            instance._weight = checkpoint["weight"]
            instance._sqrt_weight = checkpoint["sqrt_weight"]
            instance._mean = checkpoint["mean"]
            return instance
        saved_layout = checkpoint["layout"]
        layout = StateVectorLayout(
            tuple(
                FieldSpec(
                    field["name"],
                    tuple(field["component_shape"]),
                    field["component_names"],
                )
                for field in saved_layout["fields"]
            ),
            saved_layout["spatial_shape"],
            saved_layout["normalization_factors"],
        )
        if source is not None:
            source.restore_normalization(layout.normalization_factors)
            if cls._layout_metadata(source) != saved_layout:
                raise ValueError("source layout is incompatible with checkpoint")
        source = _FactorSource(layout, instance._cols)
        if execution is not None:
            if not isinstance(execution, DistributedExecution):
                raise ValueError("execution must be a DistributedExecution instance")
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("torch.distributed must be initialized")
            group = execution.process_group
            process_rank = dist.get_rank(group)
            world_size = dist.get_world_size(group)
            if execution.root_rank < 0 or execution.root_rank >= world_size:
                raise ValueError("root_rank must identify a rank in the process group")
        else:
            process_rank = 0
            world_size = 1
        if world_size > source.spatial_shape[0]:
            raise ValueError(
                "the process group cannot contain more ranks than first-axis "
                "spatial values"
            )
        instance._source = source
        instance._execution = execution
        instance._source_weight = None
        instance._maximum_rank = checkpoint["maximum_rank"]
        if prefetch_chunks is not None and not isinstance(prefetch_chunks, bool):
            raise ValueError("prefetch_chunks must be a boolean")
        instance._prefetch_chunks = (
            checkpoint.get("prefetch_chunks", False)
            if prefetch_chunks is None
            else prefetch_chunks
        )
        instance._memory_budget_bytes = None
        instance._external_weight = checkpoint["external_weight"]
        if world_size != checkpoint["world_size"]:
            raise ValueError("checkpoint requires the original number of ranks")
        instance._basis = _BasisStore(layout, basis_storage, basis_directory)
        instance._weight = None
        instance._sqrt_weight = None
        instance._spatial_batch_size = cls._validate_batch_size(
            spatial_batch_size,
            int(checkpoint["spatial_batch_size"]),
            "spatial_batch_size",
        )
        start, stop = _partition_spatial_axis(
            source.spatial_shape[0], process_rank, world_size
        )
        instance._local_spatial_slice = slice(start, stop)
        shard = Path(path) / f"rank-{process_rank}"
        index = pt.load(shard / "index.pt", weights_only=True)
        instance._memory_budget_bytes = index.get("memory_budget_bytes")
        if (index["start"], index["stop"]) != (start, stop):
            raise ValueError("checkpoint spatial partition is incompatible")
        for left, right, names in index["chunks"]:
            files = {
                name: None if filename is None else shard / filename
                for name, filename in names.items()
            }
            instance._basis.restore_chunk(slice(left, right), files, instance.V.device)
        instance._mean = instance._mean_result() if instance._subtract_mean else None
        return instance

    def __str__(self) -> str:
        size, unit = format_byte_size(self.required_memory)
        ms = (
            f"SVD of a {self._rows}x{self._cols} data matrix",
            f"Selected/optimal rank: {self.rank}/{self.opt_rank}",
            f"data type: {self.V.dtype} ({self.V.element_size()}b)",
            "truncated SVD size: {:1.4f}{:s}".format(size, unit),
        )
        return "\n".join(ms)
