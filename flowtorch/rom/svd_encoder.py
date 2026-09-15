"""Thin encoder and decoder views of :class:`flowtorch.analysis.SVD`."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch as pt

from flowtorch.analysis import DistributedExecution, SVD
from flowtorch.analysis.state_vector import (
    CompositeStateVectorSource,
    StateVectorSource,
)

from .base import Decoder, Encoder, StateData, StateResult


def _as_states(states: Union[StateData, Sequence[StateData]]) -> list[StateData]:
    if isinstance(states, (pt.Tensor, StateVectorSource)):
        return [states]
    result = list(states)
    if not result:
        raise ValueError("at least one state data set is required")
    return result


def fit_svd(
    states: Union[StateData, Sequence[StateData]],
    rank: Optional[int],
    *,
    subtract_mean: bool,
    mode: str = "auto",
    weight: Optional[pt.Tensor] = None,
    spatial_batch_size: Optional[int] = None,
    snapshot_batch_size: Optional[int] = None,
    execution: Optional[DistributedExecution] = None,
) -> tuple[SVD, tuple[int, ...]]:
    """Fit one SVD to tensor- or source-backed snapshot collections."""
    data = _as_states(states)
    tensor_backed = all(isinstance(value, pt.Tensor) for value in data)
    source_backed = all(isinstance(value, StateVectorSource) for value in data)
    if not tensor_backed and not source_backed:
        raise ValueError("state data sets must be all tensors or all sources")
    if tensor_backed:
        tensors = [value for value in data if isinstance(value, pt.Tensor)]
        if any(value.ndim != 2 for value in tensors):
            raise ValueError("state tensors must have shape (state, snapshots)")
        if any(value.shape[0] != tensors[0].shape[0] for value in tensors[1:]):
            raise ValueError("state tensors must have the same state dimension")
        if any(value.dtype != tensors[0].dtype for value in tensors[1:]):
            raise ValueError("state tensors must have the same dtype")
        if any(value.device != tensors[0].device for value in tensors[1:]):
            raise ValueError("state tensors must be on the same device")
        matrix: StateData = pt.cat(tensors, dim=1)
        lengths = tuple(value.shape[1] for value in tensors)
    else:
        sources = [value for value in data if isinstance(value, StateVectorSource)]
        matrix = (
            sources[0] if len(sources) == 1 else CompositeStateVectorSource(sources)
        )
        lengths = tuple(value.n_snapshots for value in sources)
    svd = SVD(
        matrix,
        rank=rank,
        mode=mode,
        weight=weight,
        subtract_mean=subtract_mean,
        spatial_batch_size=spatial_batch_size,
        snapshot_batch_size=snapshot_batch_size,
        execution=execution,
    )
    return svd, lengths


def training_coefficients(svd: SVD) -> pt.Tensor:
    """Return the retained coefficients of the SVD training snapshots."""
    return svd.s.unsqueeze(-1) * svd.V.conj().T


class SVDEncoder(Encoder):
    """Non-owning ROM encoder that delegates directly to a fitted SVD."""

    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "_svd", None)

    @property
    def svd(self) -> SVD:
        if self._svd is None:
            raise RuntimeError("the SVD encoder has not been fitted")
        return self._svd

    def set_svd(self, svd: SVD) -> None:
        object.__setattr__(self, "_svd", svd)

    def forward(self, state: StateData) -> pt.Tensor:
        return self.svd.encode(state)


class SVDDecoder(Decoder):
    """Non-owning ROM decoder that delegates directly to a fitted SVD."""

    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "_svd", None)

    @property
    def svd(self) -> SVD:
        if self._svd is None:
            raise RuntimeError("the SVD decoder has not been fitted")
        return self._svd

    def set_svd(self, svd: SVD) -> None:
        object.__setattr__(self, "_svd", svd)

    def forward(self, state: pt.Tensor) -> StateResult:
        return self.svd.decode(state)


__all__ = ["SVDDecoder", "SVDEncoder", "fit_svd", "training_coefficients"]
