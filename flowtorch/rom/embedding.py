"""Fixed latent-space embeddings for reduced-order models."""

from __future__ import annotations

from itertools import combinations_with_replacement
from math import comb
from typing import Optional

import torch as pt

from .base import LatentEmbedding


class TimeDelayEmbedding(LatentEmbedding):
    """Stack consecutive latent states from oldest to newest.

    A trajectory ``z`` with shape ``(rank, time)`` is converted to a block
    Hankel matrix whose columns are ``[z[k-d+1], ..., z[k]]``. The same
    operation accepts optional batch dimensions between the feature and time
    axes, which is useful for delayed control sequences.

    :param delays: number of consecutive samples in each embedded state
    """

    def __init__(self, delays: int) -> None:
        super().__init__()
        if not isinstance(delays, int) or isinstance(delays, bool) or delays < 1:
            raise ValueError("delays must be a positive integer")
        self.history_length = delays

    def transform_trajectory(self, state: pt.Tensor) -> pt.Tensor:
        """Embed a time-last trajectory, preserving optional batch axes."""
        if state.ndim < 2:
            raise ValueError("a trajectory must have feature and time dimensions")
        if state.shape[-1] < self.history_length:
            raise ValueError(
                f"trajectory has {state.shape[-1]} samples but "
                f"{self.history_length} delays were requested"
            )
        windows = state.unfold(-1, self.history_length, 1)
        # (features, ..., windows, delays) -> (delays, features, ..., windows)
        ordered = windows.movedim(-1, 0)
        return ordered.reshape(
            self.history_length * state.shape[0],
            *state.shape[1:-1],
            windows.shape[-2],
        )

    def initial(self, history: pt.Tensor) -> pt.Tensor:
        """Create one embedded state from ``(features, delays, *batch)``."""
        if history.ndim < 2 or history.shape[1] != self.history_length:
            raise ValueError(
                "initial history must have shape "
                f"(features, {self.history_length}, *batch)"
            )
        ordered = history.movedim(1, 0)
        return ordered.reshape(
            self.history_length * history.shape[0], *history.shape[2:]
        )

    def readout(self, embedded_state: pt.Tensor) -> pt.Tensor:
        """Return the newest latent state from an embedded prediction."""
        if embedded_state.ndim < 1 or embedded_state.shape[0] % self.history_length:
            raise ValueError("embedded state has an incompatible feature dimension")
        rank = embedded_state.shape[0] // self.history_length
        blocks = embedded_state.reshape(
            self.history_length, rank, *embedded_state.shape[1:]
        )
        return blocks[-1]


class MonomialEmbedding(LatentEmbedding):
    """Lift latent coordinates to all monomials up to a maximum degree.

    Terms are ordered first by total degree and then lexicographically by
    their variable indices. For two inputs and ``degree=2``, the result is
    ``[x_1, x_2, x_1^2, x_1 x_2, x_2^2]``. If ``include_constant=True``, a
    leading constant feature is added. Retaining the degree-one terms makes
    the original POD coordinates an exact slice of the lifted state and
    therefore provides a deterministic decoder readout.

    :param degree: largest total monomial degree to include
    :param include_constant: prepend the degree-zero constant monomial
    """

    def __init__(self, degree: int, *, include_constant: bool = False) -> None:
        super().__init__()
        if not isinstance(degree, int) or isinstance(degree, bool) or degree < 1:
            raise ValueError("degree must be a positive integer")
        self.degree = degree
        self.include_constant = bool(include_constant)
        self._input_features: Optional[int] = None
        self._index_cache: dict[tuple[int, pt.device], tuple[pt.Tensor, ...]] = {}

    def _indices(self, features: int, device: pt.device) -> tuple[pt.Tensor, ...]:
        key = (features, device)
        cached = self._index_cache.get(key)
        if cached is None:
            cached = tuple(
                pt.tensor(
                    list(combinations_with_replacement(range(features), degree)),
                    dtype=pt.long,
                    device=device,
                )
                for degree in range(1, self.degree + 1)
            )
            self._index_cache[key] = cached
        return cached

    def _lift(self, state: pt.Tensor, *, configure: bool) -> pt.Tensor:
        if state.ndim < 1 or state.shape[0] < 1:
            raise ValueError("state must contain at least one feature")
        features = state.shape[0]
        if configure:
            self._input_features = features
        elif self._input_features is None:
            raise RuntimeError("the monomial embedding has not been fitted")
        elif features != self._input_features:
            raise ValueError(
                f"expected {self._input_features} input features but got {features}"
            )
        terms = []
        if self.include_constant:
            terms.append(state.new_ones((1, *state.shape[1:])))
        for indices in self._indices(features, state.device):
            selected = state[indices]
            terms.append(selected.prod(dim=1))
        return pt.cat(terms, dim=0)

    def _apply(self, function):
        self._index_cache.clear()
        return super()._apply(function)

    def transform_trajectory(self, state: pt.Tensor) -> pt.Tensor:
        """Lift a time-last trajectory, preserving optional batch axes."""
        if state.ndim < 2:
            raise ValueError("a trajectory must have feature and time dimensions")
        return self._lift(state, configure=True)

    def initial(self, state: pt.Tensor) -> pt.Tensor:
        """Lift one state or a batch of states."""
        return self._lift(state, configure=False)

    def readout(self, embedded_state: pt.Tensor) -> pt.Tensor:
        """Extract the retained degree-one POD coordinates."""
        if self._input_features is None:
            raise RuntimeError("the monomial embedding has not been fitted")
        start = int(self.include_constant)
        expected = start + sum(
            comb(self._input_features + degree - 1, degree)
            for degree in range(1, self.degree + 1)
        )
        if embedded_state.ndim < 1 or embedded_state.shape[0] != expected:
            raise ValueError("embedded state has an incompatible feature dimension")
        return embedded_state[start : start + self._input_features]


__all__ = ["MonomialEmbedding", "TimeDelayEmbedding"]
