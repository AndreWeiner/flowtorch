"""Lazy construction and decomposition outputs for physical state vectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import prod
from numbers import Real
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Union,
    cast,
)

import torch as pt
import torch.distributed as dist

if TYPE_CHECKING:
    from flowtorch.data.dataloader import Dataloader


@dataclass(frozen=True)
class FieldSpec:
    """Describe one physical field in a flat state vector.

    Component dimensions must trail the common spatial dimensions. All spatial
    values of one component are stored contiguously before the next component.

    >>> velocity = FieldSpec("U", component_shape=(3,), component_names=("u", "v", "w"))
    >>> pressure = FieldSpec("p")
    """

    name: str
    component_shape: tuple[int, ...] = ()
    component_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("field name must not be empty")
        shape = tuple(int(value) for value in self.component_shape)
        if any(value < 1 for value in shape):
            raise ValueError("component_shape must contain positive dimensions")
        names = tuple(self.component_names)
        if names and len(names) != prod(shape):
            raise ValueError(
                "component_names must contain one name per tensor component"
            )
        object.__setattr__(self, "component_shape", shape)
        object.__setattr__(self, "component_names", names)

    @property
    def n_components(self) -> int:
        """Number of scalar components in the field."""
        return prod(self.component_shape) if self.component_shape else 1


class StateVectorLayout:
    """Pack and restore fields using a stable component-major layout."""

    def __init__(
        self,
        fields: Sequence[FieldSpec],
        spatial_shape: Sequence[int],
        normalization_factors: Optional[Mapping[str, float]] = None,
    ) -> None:
        self._fields = tuple(fields)
        if not self._fields:
            raise ValueError("at least one field is required")
        if len({field.name for field in self._fields}) != len(self._fields):
            raise ValueError("field names must be unique")
        self._spatial_shape = tuple(int(value) for value in spatial_shape)
        if not self._spatial_shape or any(value < 1 for value in self._spatial_shape):
            raise ValueError("spatial_shape must contain positive dimensions")
        factors = (
            {field.name: 1.0 for field in self._fields}
            if normalization_factors is None
            else normalization_factors
        )
        if set(factors) != {field.name for field in self._fields}:
            raise ValueError("normalization must define exactly one factor per field")
        normalized: dict[str, float] = {}
        for name, value in factors.items():
            if not isinstance(value, Real) or isinstance(value, bool):
                raise ValueError("normalization factors must be real scalars")
            factor = float(value)
            if not pt.isfinite(pt.tensor(factor)) or factor <= 0.0:
                raise ValueError("normalization factors must be finite and positive")
            normalized[name] = factor
        self._normalization_factors = normalized

    @property
    def fields(self) -> tuple[FieldSpec, ...]:
        return self._fields

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return self._spatial_shape

    @property
    def normalization_factors(self) -> dict[str, float]:
        return self._normalization_factors.copy()

    @property
    def state_size(self) -> int:
        points = prod(self.spatial_shape)
        return points * sum(field.n_components for field in self.fields)

    @property
    def signature(self) -> tuple[Any, ...]:
        """Metadata used to validate compatible sources."""
        return (
            self.fields,
            self.spatial_shape,
            tuple(self._normalization_factors.items()),
        )

    def with_normalization(self, factors: Mapping[str, float]) -> "StateVectorLayout":
        return StateVectorLayout(self.fields, self.spatial_shape, factors)

    def pack(
        self,
        values: Mapping[str, pt.Tensor],
        *,
        spatial_shape: Optional[Sequence[int]] = None,
        normalize: bool = True,
    ) -> pt.Tensor:
        """Concatenate scalar components into state-by-snapshot form."""
        shape = self.spatial_shape if spatial_shape is None else tuple(spatial_shape)
        packed = []
        trailing_shape = None
        reference = None
        for field in self.fields:
            if field.name not in values:
                raise ValueError(f"missing field {field.name!r}")
            value = values[field.name]
            base_shape = (*shape, *field.component_shape)
            if tuple(value.shape[: len(base_shape)]) != base_shape:
                raise ValueError(
                    f"field {field.name!r} has incompatible shape {tuple(value.shape)}; "
                    f"expected a prefix of {base_shape}"
                )
            trailing = tuple(value.shape[len(base_shape) :])
            if trailing_shape is None:
                trailing_shape = trailing
                reference = value
            elif trailing != trailing_shape:
                raise ValueError("all fields must have matching trailing dimensions")
            assert reference is not None
            if value.dtype != reference.dtype or value.device != reference.device:
                raise ValueError("all fields must have matching dtypes and devices")
            n_spatial = len(shape)
            n_component = len(field.component_shape)
            n_trailing = len(trailing)
            order = (
                *range(n_spatial, n_spatial + n_component),
                *range(n_spatial),
                *range(n_spatial + n_component, n_spatial + n_component + n_trailing),
            )
            component_major = value.permute(order) if order else value
            flat = component_major.reshape(prod(shape) * field.n_components, *trailing)
            if normalize:
                flat = flat / self._normalization_factors[field.name]
            packed.append(flat)
        return pt.cat(packed, dim=0)

    def split(
        self,
        state: pt.Tensor,
        *,
        spatial_shape: Optional[Sequence[int]] = None,
        denormalize: bool = True,
    ) -> dict[str, pt.Tensor]:
        """Restore physical field and component dimensions from a flat state."""
        shape = self.spatial_shape if spatial_shape is None else tuple(spatial_shape)
        trailing = tuple(state.shape[1:])
        expected = prod(shape) * sum(field.n_components for field in self.fields)
        if state.ndim < 1 or state.shape[0] != expected:
            raise ValueError(
                f"state has {state.shape[0] if state.ndim else 0} rows; expected {expected}"
            )
        result = {}
        offset = 0
        for field in self.fields:
            rows = prod(shape) * field.n_components
            value = state[offset : offset + rows]
            offset += rows
            value = value.reshape(*field.component_shape, *shape, *trailing)
            n_component = len(field.component_shape)
            n_spatial = len(shape)
            n_trailing = len(trailing)
            order = (
                *range(n_component, n_component + n_spatial),
                *range(n_component),
                *range(n_component + n_spatial, n_component + n_spatial + n_trailing),
            )
            value = value.permute(order) if order else value
            if denormalize:
                value = value * self._normalization_factors[field.name]
            result[field.name] = value
        return result


class StateVectorSource(ABC):
    """Random-access, spatially chunked source of flat state vectors."""

    @property
    @abstractmethod
    def n_snapshots(self) -> int: ...

    @property
    @abstractmethod
    def layout(self) -> StateVectorLayout: ...

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return self.layout.spatial_shape

    @abstractmethod
    def read(self, spatial_slice: slice, snapshot_slice: slice) -> pt.Tensor: ...

    def read_weight(self, spatial_slice: slice) -> Optional[pt.Tensor]:
        return None

    def fit_normalization(
        self,
        spatial_slice: slice,
        spatial_batch_size: int,
        snapshot_batch_size: int,
        execution: Any = None,
    ) -> None:
        """Fit deferred normalization, if any."""

    def restore_normalization(self, factors: Mapping[str, float]) -> None:
        """Restore checkpoint normalization or reject incompatible factors."""
        if self.layout.normalization_factors != dict(factors):
            raise ValueError("source normalization is incompatible")


WeightSource = Union[pt.Tensor, Callable[[slice], pt.Tensor]]


class DataloaderStateVectorSource(StateVectorSource):
    """Build state-vector chunks lazily from a flowTorch dataloader.

    Each physical field has one normalization factor, regardless of how many
    scalar components it contains.  For example, all three velocity components
    use ``u_ref`` while pressure uses ``p_ref``::

        fields = (
            FieldSpec("U", component_shape=(3,), component_names=("u", "v", "w")),
            FieldSpec("p"),
        )
        source = DataloaderStateVectorSource(
            loader,
            fields,
            times=loader.write_times,
            normalization={"U": u_ref, "p": p_ref},
            spatial_weight=loader.weights,
        )

    Passing ``normalization="rms"`` fits one weighted fluctuation RMS per
    field during the first SVD.  Snapshot and spatial slices are requested from
    the dataloader only when a decomposition or result chunk needs them.
    """

    def __init__(
        self,
        loader: Dataloader,
        fields: Sequence[FieldSpec],
        times: Optional[Sequence[str]] = None,
        normalization: Optional[Union[str, Mapping[str, float]]] = None,
        spatial_weight: Optional[WeightSource] = None,
        device: Union[str, pt.device] = "cpu",
    ) -> None:
        self._loader = loader
        self._fields = tuple(fields)
        if not self._fields:
            raise ValueError("at least one field is required")
        self._times = list(loader.write_times if times is None else times)
        if not self._times:
            raise ValueError("at least one snapshot time is required")
        spatial_shapes = []
        for field in self._fields:
            shape = tuple(loader.snapshot_shape(field.name, self._times[0]))
            n_component = len(field.component_shape)
            if n_component:
                if shape[-n_component:] != field.component_shape:
                    raise ValueError(
                        f"field {field.name!r} shape {shape} does not end with "
                        f"component shape {field.component_shape}"
                    )
                shape = shape[:-n_component]
            spatial_shapes.append(shape)
        if any(shape != spatial_shapes[0] for shape in spatial_shapes[1:]):
            raise ValueError("all fields must share the same spatial shape")
        if normalization is None:
            if len(self._fields) > 1:
                raise ValueError("multiple fields require normalization")
            factors = {self._fields[0].name: 1.0}
            self._fit_rms = False
        elif isinstance(normalization, str):
            if normalization != "rms":
                raise ValueError("normalization must be a mapping or 'rms'")
            factors = {field.name: 1.0 for field in self._fields}
            self._fit_rms = True
        else:
            factors = dict(normalization)
            self._fit_rms = False
        self._layout = StateVectorLayout(self._fields, spatial_shapes[0], factors)
        self._spatial_weight = spatial_weight
        self._device = pt.device(device)

    @property
    def n_snapshots(self) -> int:
        return len(self._times)

    @property
    def times(self) -> tuple[str, ...]:
        return tuple(self._times)

    @property
    def layout(self) -> StateVectorLayout:
        return self._layout

    @property
    def normalization_fitted(self) -> bool:
        return not self._fit_rms

    def _read_raw_fields(
        self, spatial_slice: slice, snapshot_slice: slice
    ) -> dict[str, pt.Tensor]:
        start, stop, step = snapshot_slice.indices(self.n_snapshots)
        if step != 1:
            raise ValueError("snapshot_slice must have unit stride")
        times = self._times[start:stop]
        if not times:
            raise ValueError("snapshot slice must not be empty")
        names = [field.name for field in self._fields]
        loaded = self._loader.load_snapshot_slice(names, times, spatial_slice)
        assert isinstance(loaded, list)
        fields = {}
        for field, value in zip(self._fields, loaded):
            expected_snapshot_ndim = len(self.spatial_shape) + len(
                field.component_shape
            )
            if value.ndim == expected_snapshot_ndim:
                if len(times) != 1:
                    raise ValueError(
                        f"field {field.name!r} is missing its snapshot dimension"
                    )
                value = value.unsqueeze(-1)
            if value.ndim != expected_snapshot_ndim + 1 or value.shape[-1] != len(
                times
            ):
                raise ValueError(
                    f"field {field.name!r} returned incompatible batched shape "
                    f"{tuple(value.shape)}"
                )
            fields[field.name] = value.to(self._device)
        return fields

    def read(self, spatial_slice: slice, snapshot_slice: slice) -> pt.Tensor:
        if self._fit_rms:
            raise RuntimeError("RMS normalization must be fitted before reading")
        fields = self._read_raw_fields(spatial_slice, snapshot_slice)
        start, stop, step = spatial_slice.indices(self.spatial_shape[0])
        if step != 1:
            raise ValueError("spatial_slice must have unit stride")
        chunk_shape = (stop - start, *self.spatial_shape[1:])
        return self.layout.pack(fields, spatial_shape=chunk_shape)

    def read_weight(self, spatial_slice: slice) -> Optional[pt.Tensor]:
        if self._spatial_weight is None:
            return None
        weight = (
            self._spatial_weight(spatial_slice)
            if callable(self._spatial_weight)
            else self._spatial_weight[spatial_slice]
        )
        return weight.to(self._device)

    def fit_normalization(
        self,
        spatial_slice: slice,
        spatial_batch_size: int,
        snapshot_batch_size: int,
        execution: Any = None,
    ) -> None:
        if not self._fit_rms:
            return
        local_start, local_stop, _ = spatial_slice.indices(self.spatial_shape[0])
        numerators = pt.zeros(len(self._fields), dtype=pt.float64, device=self._device)
        denominators = pt.zeros_like(numerators)
        for spatial_start in range(local_start, local_stop, spatial_batch_size):
            spatial_stop = min(spatial_start + spatial_batch_size, local_stop)
            current_slice = slice(spatial_start, spatial_stop)
            sums = [None] * len(self._fields)
            sums_squared = [None] * len(self._fields)
            for snapshot_start in range(0, self.n_snapshots, snapshot_batch_size):
                snapshot_stop = min(
                    snapshot_start + snapshot_batch_size, self.n_snapshots
                )
                fields = self._read_raw_fields(
                    current_slice, slice(snapshot_start, snapshot_stop)
                )
                for index, field in enumerate(self._fields):
                    value = fields[field.name]
                    accumulator_dtype = (
                        pt.complex128 if pt.is_complex(value) else pt.float64
                    )
                    value = value.to(accumulator_dtype)
                    batch_sum = value.sum(dim=-1)
                    batch_squared = value.abs().square().sum(dim=-1)
                    sums[index] = (
                        batch_sum if sums[index] is None else sums[index] + batch_sum
                    )
                    sums_squared[index] = (
                        batch_squared
                        if sums_squared[index] is None
                        else sums_squared[index] + batch_squared
                    )
            weight = self.read_weight(current_slice)
            for index, field in enumerate(self._fields):
                field_sum = sums[index]
                field_sum_squared = sums_squared[index]
                assert field_sum is not None and field_sum_squared is not None
                variation = (
                    field_sum_squared - field_sum.abs().square() / self.n_snapshots
                )
                variation = variation.clamp_min(0.0)
                if weight is None:
                    numerators[index] += variation.sum()
                    denominators[index] += self.n_snapshots * prod(
                        variation.shape[: len(self.spatial_shape)]
                    )
                else:
                    expanded = weight.to(pt.float64)
                    for _ in field.component_shape:
                        expanded = expanded.unsqueeze(-1)
                    numerators[index] += (variation * expanded).sum()
                    denominators[index] += (
                        self.n_snapshots * weight.to(pt.float64).sum()
                    )
        if execution is not None:
            group = execution.process_group
            dist.all_reduce(numerators, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(denominators, op=dist.ReduceOp.SUM, group=group)
        factors = (numerators / denominators).sqrt()
        if not bool(pt.isfinite(factors).all()) or bool((factors <= 0.0).any()):
            raise ValueError("cannot fit a positive finite RMS factor for every field")
        self._layout = self.layout.with_normalization(
            {
                field.name: float(factors[index].item())
                for index, field in enumerate(self._fields)
            }
        )
        self._fit_rms = False

    def restore_normalization(self, factors: Mapping[str, float]) -> None:
        """Restore frozen factors without rescanning snapshots."""
        if self._fit_rms:
            self._layout = self.layout.with_normalization(factors)
            self._fit_rms = False
        else:
            super().restore_normalization(factors)

    def with_times(self, times: Sequence[str]) -> "DataloaderStateVectorSource":
        """Create a compatible source for newly appended times."""
        if self._fit_rms:
            raise RuntimeError("normalization must be fitted first")
        return DataloaderStateVectorSource(
            self._loader,
            self._fields,
            times,
            normalization=self.layout.normalization_factors,
            spatial_weight=self._spatial_weight,
            device=self._device,
        )


class CompositeStateVectorSource(StateVectorSource):
    """Append-compatible view over consecutive state-vector sources."""

    def __init__(self, sources: Sequence[StateVectorSource]) -> None:
        self._sources = tuple(sources)
        if not self._sources:
            raise ValueError("at least one source is required")
        signature = self._sources[0].layout.signature
        if any(source.layout.signature != signature for source in self._sources[1:]):
            raise ValueError("state-vector sources have incompatible layouts")
        self._layout = self._sources[0].layout

    @property
    def n_snapshots(self) -> int:
        return sum(source.n_snapshots for source in self._sources)

    @property
    def layout(self) -> StateVectorLayout:
        return self._layout

    def read(self, spatial_slice: slice, snapshot_slice: slice) -> pt.Tensor:
        start, stop, step = snapshot_slice.indices(self.n_snapshots)
        if step != 1 or start >= stop:
            raise ValueError("snapshot_slice must be a non-empty unit-stride slice")
        pieces = []
        offset = 0
        for source in self._sources:
            source_stop = offset + source.n_snapshots
            overlap_start = max(start, offset)
            overlap_stop = min(stop, source_stop)
            if overlap_start < overlap_stop:
                pieces.append(
                    source.read(
                        spatial_slice,
                        slice(overlap_start - offset, overlap_stop - offset),
                    )
                )
            offset = source_stop
        return pt.cat(pieces, dim=1)

    def read_weight(self, spatial_slice: slice) -> Optional[pt.Tensor]:
        return self._sources[0].read_weight(spatial_slice)


class StateVectorResult:
    """Lazy local spatial result with optional field reconstruction.

    Flat chunks remain normalized.  Set ``split=True`` to obtain denormalized
    physical fields with their original component dimensions::

        for spatial_slice, fields in svd.U.iter_chunks(split=True):
            velocity_modes = fields["U"]
            pressure_modes = fields["p"]

    Distributed results remain local unless :meth:`gather` is called
    collectively on every rank.
    """

    def __init__(
        self,
        layout: StateVectorLayout,
        local_spatial_slice: slice,
        trailing_shape: Sequence[int],
        producer: Callable[[slice], pt.Tensor],
        spatial_batch_size: int,
        execution: Any = None,
    ) -> None:
        self.layout = layout
        self.local_spatial_slice = local_spatial_slice
        self.trailing_shape = tuple(trailing_shape)
        self._producer = producer
        self.spatial_batch_size = spatial_batch_size
        self.execution = execution

    @property
    def global_shape(self) -> tuple[int, ...]:
        return (self.layout.state_size, *self.trailing_shape)

    def read_chunk(
        self, spatial_slice: slice, split: bool = False
    ) -> Union[pt.Tensor, dict[str, pt.Tensor]]:
        """Evaluate one spatial chunk within the rank-local partition."""
        local_start, local_stop, _ = self.local_spatial_slice.indices(
            self.layout.spatial_shape[0]
        )
        start, stop, step = spatial_slice.indices(self.layout.spatial_shape[0])
        if step != 1 or start < local_start or stop > local_stop or start >= stop:
            raise ValueError("spatial slice must be a non-empty rank-local interval")
        value = self._producer(slice(start, stop))
        if not split:
            return value
        chunk_shape = (stop - start, *self.layout.spatial_shape[1:])
        return self.layout.split(value, spatial_shape=chunk_shape)

    def iter_chunks(
        self, split: bool = False
    ) -> Iterator[tuple[slice, Union[pt.Tensor, dict[str, pt.Tensor]]]]:
        start, stop, _ = self.local_spatial_slice.indices(self.layout.spatial_shape[0])
        for chunk_start in range(start, stop, self.spatial_batch_size):
            chunk_stop = min(chunk_start + self.spatial_batch_size, stop)
            current = slice(chunk_start, chunk_stop)
            yield current, self.read_chunk(current, split=split)

    def materialize_local(
        self, split: bool = False
    ) -> Union[pt.Tensor, dict[str, pt.Tensor]]:
        chunks = []
        for spatial_slice, value in self.iter_chunks(split=False):
            assert isinstance(value, pt.Tensor)
            start, stop, _ = spatial_slice.indices(self.layout.spatial_shape[0])
            chunks.append(
                self.layout.split(
                    value,
                    spatial_shape=(stop - start, *self.layout.spatial_shape[1:]),
                    denormalize=split,
                )
            )
        if not chunks:
            raise RuntimeError("this rank owns no spatial values")
        fields = {
            field.name: pt.cat([chunk[field.name] for chunk in chunks], dim=0)
            for field in self.layout.fields
        }
        if split:
            return fields
        start, stop, _ = self.local_spatial_slice.indices(self.layout.spatial_shape[0])
        local_shape = (stop - start, *self.layout.spatial_shape[1:])
        return self.layout.pack(fields, spatial_shape=local_shape, normalize=False)

    def gather(
        self, split: bool = False, root_rank: Optional[int] = None
    ) -> Optional[Union[pt.Tensor, dict[str, pt.Tensor]]]:
        """Materialize local chunks and gather complete CPU fields on one rank."""
        local = self.materialize_local(split=False)
        start, stop, _ = self.local_spatial_slice.indices(self.layout.spatial_shape[0])
        local_shape = (stop - start, *self.layout.spatial_shape[1:])
        local = self.layout.split(local, spatial_shape=local_shape, denormalize=False)
        assert isinstance(local, dict)
        local_cpu = {name: value.detach().cpu() for name, value in local.items()}
        if self.execution is None or not dist.is_initialized():
            if split:
                return local_cpu
            return self.layout.pack(local_cpu, normalize=False)
        group = self.execution.process_group
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        root = self.execution.root_rank if root_rank is None else root_rank
        if root < 0 or root >= world_size:
            raise ValueError("root_rank must identify a rank in the process group")
        global_ranks = [None] * world_size
        dist.all_gather_object(global_ranks, dist.get_rank(), group=group)
        gathered = [None] * world_size if rank == root else None
        dist.gather_object(
            local_cpu,
            gathered,
            dst=global_ranks[root],
            group=group,
        )
        if rank != root:
            return None
        assert gathered is not None
        if any(part is None for part in gathered):
            raise RuntimeError("distributed result gather is incomplete")
        complete_parts = cast(list[dict[str, pt.Tensor]], gathered)
        fields = {
            field.name: pt.cat([part[field.name] for part in complete_parts], dim=0)
            for field in self.layout.fields
        }
        if split:
            return {
                field.name: fields[field.name]
                * self.layout.normalization_factors[field.name]
                for field in self.layout.fields
            }
        return self.layout.pack(fields, normalize=False)


__all__ = [
    "CompositeStateVectorSource",
    "DataloaderStateVectorSource",
    "FieldSpec",
    "StateVectorLayout",
    "StateVectorResult",
    "StateVectorSource",
]
