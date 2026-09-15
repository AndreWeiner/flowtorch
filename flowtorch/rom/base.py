"""Common interfaces and data containers for reduced-order models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch as pt

from flowtorch.analysis.state_vector import StateVectorResult, StateVectorSource

StateData = Union[pt.Tensor, StateVectorSource]
StateResult = Union[pt.Tensor, StateVectorResult]


@dataclass(frozen=True)
class ROMQuery:
    """Inputs at which a ROM prediction is requested."""

    time: Optional[pt.Tensor] = None
    parameters: Optional[pt.Tensor] = None
    forcing: Optional[pt.Tensor] = None
    forcing_history: Optional[pt.Tensor] = None


@dataclass(frozen=True)
class Prediction:
    """A deterministic or probabilistic ROM prediction."""

    mean: StateResult
    scale: Optional[StateResult] = None


@dataclass(frozen=True)
class InputSpec:
    """Declare the query values required by a latent regressor."""

    initial_state: bool = False
    time: bool = False
    parameters: bool = False
    forcing: bool = False

    def validate(self, initial_state: Any, query: ROMQuery) -> None:
        values = {
            "initial_state": initial_state,
            "time": query.time,
            "parameters": query.parameters,
            "forcing": query.forcing,
        }
        for name, required in (
            ("initial_state", self.initial_state),
            ("time", self.time),
            ("parameters", self.parameters),
            ("forcing", self.forcing),
        ):
            if required and values[name] is None:
                raise ValueError(f"{name} is required for this model")


@dataclass(frozen=True)
class Trajectory:
    """A state trajectory and its strictly increasing sample times."""

    states: StateData
    time: pt.Tensor


@dataclass(frozen=True)
class ControlledTrajectory(Trajectory):
    """A trajectory with one zero-order-held input per time interval."""

    forcing: pt.Tensor


@dataclass(frozen=True)
class ParametricSnapshots:
    """State snapshots paired columnwise with parameter coordinates."""

    states: StateData
    parameters: pt.Tensor


class Encoder(pt.nn.Module, ABC):
    """Map physical state vectors to latent coordinates."""

    @abstractmethod
    def forward(self, state: StateData) -> pt.Tensor: ...


class Decoder(pt.nn.Module, ABC):
    """Map latent coordinates to physical state vectors."""

    @abstractmethod
    def forward(self, state: pt.Tensor) -> StateResult: ...


class LatentEmbedding(pt.nn.Module):
    """Optional state-history or feature embedding in POD coordinates."""

    history_length = 1

    def transform_trajectory(self, state: pt.Tensor) -> pt.Tensor:
        return state

    def initial(self, history: pt.Tensor) -> pt.Tensor:
        return history

    def readout(self, embedded_state: pt.Tensor) -> pt.Tensor:
        return embedded_state


class Regressor(pt.nn.Module, ABC):
    """Predict latent quantities from declared query inputs."""

    input_spec = InputSpec()

    @abstractmethod
    def forward(
        self, initial_state: Optional[pt.Tensor], query: ROMQuery
    ) -> pt.Tensor: ...


class ROM(pt.nn.Module, ABC):
    """Base class for compositional reduced-order models."""

    input_spec = InputSpec()

    @abstractmethod
    def forward(
        self, initial_state: Optional[StateData], query: ROMQuery
    ) -> Prediction: ...

    def predict(
        self,
        initial_state: Optional[StateData] = None,
        *,
        time: Optional[pt.Tensor] = None,
        parameters: Optional[pt.Tensor] = None,
        forcing: Optional[pt.Tensor] = None,
        forcing_history: Optional[pt.Tensor] = None,
    ) -> Prediction:
        """Validate and evaluate a model using keyword query inputs."""
        query = ROMQuery(
            time=time,
            parameters=parameters,
            forcing=forcing,
            forcing_history=forcing_history,
        )
        self.input_spec.validate(initial_state, query)
        return self.forward(initial_state, query)


__all__ = [
    "ControlledTrajectory",
    "Decoder",
    "Encoder",
    "InputSpec",
    "LatentEmbedding",
    "ParametricSnapshots",
    "Prediction",
    "Regressor",
    "ROM",
    "ROMQuery",
    "StateData",
    "StateResult",
    "Trajectory",
]
