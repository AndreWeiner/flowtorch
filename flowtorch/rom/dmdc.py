"""Continuous-time dynamic mode decomposition with control."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch as pt

from .base import (
    ControlledTrajectory,
    InputSpec,
    LatentEmbedding,
    Prediction,
    ROMQuery,
    StateData,
    Trajectory,
)
from .dmd import DMD, _split_columns, _validate_times
from .dynamics import ContinuousOperator, _closest_proper_orthogonal
from .svd_encoder import fit_svd, training_coefficients


def _as_controlled(
    trajectories: Union[Trajectory, Sequence[Trajectory]],
) -> list[ControlledTrajectory]:
    if isinstance(trajectories, Trajectory):
        if isinstance(trajectories, ControlledTrajectory):
            return [trajectories]
        raise TypeError("all training data must be ControlledTrajectory instances")
    result: list[ControlledTrajectory] = []
    for value in trajectories:
        if not isinstance(value, ControlledTrajectory):
            raise TypeError("all training data must be ControlledTrajectory instances")
        result.append(value)
    if not result:
        raise ValueError("at least one controlled trajectory is required")
    return result


class DMDc(DMD):
    """POD plus continuous linear dynamics with zero-order-held control.

    Analytical initialization uses a proper-orthogonal latent discrete
    transition by default. Its eigenvalues have unit modulus and its initial
    continuous-time growth rates are exactly zero. With ``stationary=False``
    those rates are trainable; ``stationary=True`` fixes them at zero.
    Fine-tuning trains an unconstrained spectral basis, so the transition need
    not remain orthogonal or norm preserving. This is a neutral spectral
    constraint, not a statistical-stationarity or asymptotic-stability
    constraint. Set ``stationary_initialization=False`` to use an
    unconstrained joint least-squares initialization.
    """

    input_spec = InputSpec(initial_state=True, time=True, forcing=True)

    def __init__(
        self,
        *args,
        control_embedding: Optional[LatentEmbedding] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.control_embedding = (
            LatentEmbedding() if control_embedding is None else control_embedding
        )
        self.register_parameter("control", None)
        self._training_forcing: list[pt.Tensor] = []
        self.n_controls: Optional[int] = None
        self.n_control_features: Optional[int] = None

    def fit(
        self,
        trajectories: Union[Trajectory, Sequence[Trajectory]],
    ) -> "DMDc":
        data = _as_controlled(trajectories)
        controls, n_controls = self._validate_controls(data)
        self.svd, lengths = fit_svd(
            [value.states for value in data],
            self.requested_rank,
            subtract_mean=self.subtract_mean,
            **self.svd_options,
        )
        self.encoder.set_svd(self.svd)
        self.decoder.set_svd(self.svd)
        coefficients = _split_columns(training_coefficients(self.svd), lengths)
        return self._fit_control_regressor(data, coefficients, controls, n_controls)

    def fit_regressor(
        self,
        trajectories: Union[Trajectory, Sequence[Trajectory]],
    ) -> "DMDc":
        """Fit only latent dynamics and control using the existing SVD."""
        if self.svd is None:
            raise RuntimeError("fit the SVD before fitting the regressor")
        data = _as_controlled(trajectories)
        controls, n_controls = self._validate_controls(data)
        coefficients = [self.encoder(value.states) for value in data]
        return self._fit_control_regressor(data, coefficients, controls, n_controls)

    @staticmethod
    def _validate_controls(
        data: Sequence[ControlledTrajectory],
    ) -> tuple[list[pt.Tensor], int]:
        controls = []
        n_controls = None
        for trajectory in data:
            forcing = trajectory.forcing
            if forcing.ndim == 1:
                forcing = forcing.unsqueeze(0)
            if forcing.ndim != 2 or forcing.shape[1] != trajectory.time.numel() - 1:
                raise ValueError("forcing must contain one column per time interval")
            if n_controls is None:
                n_controls = forcing.shape[0]
            elif forcing.shape[0] != n_controls:
                raise ValueError("all trajectories must use the same control dimension")
            controls.append(forcing)
        assert n_controls is not None
        return controls, n_controls

    def _fit_control_regressor(
        self,
        data: Sequence[ControlledTrajectory],
        coefficients: Sequence[pt.Tensor],
        controls: Sequence[pt.Tensor],
        n_controls: int,
    ) -> "DMDc":
        self.dt = _validate_times(data)
        normalized_dt = self._resolve_time_scale(self.dt)
        self.n_controls = n_controls
        state_history = self.embedding.history_length
        control_history = self.control_embedding.history_length
        if not isinstance(state_history, int) or state_history < 1:
            raise ValueError("embedding history_length must be a positive integer")
        if not isinstance(control_history, int) or control_history < 1:
            raise ValueError(
                "control embedding history_length must be a positive integer"
            )
        start = max(state_history, control_history) - 1
        self._state_embedding_trim = start - (state_history - 1)
        self._pod_trajectories = list(coefficients)
        embedded_states = [
            self.embedding.transform_trajectory(value) for value in coefficients
        ]
        self._latent_trajectories = [
            value[:, self._state_embedding_trim :] for value in embedded_states
        ]
        if any(
            value.ndim != 2
            or value.shape[1] != trajectory.time.numel() - start
            or value.shape[1] < 2
            for value, trajectory in zip(self._latent_trajectories, data)
        ):
            raise ValueError(
                "aligned embedded trajectories must retain at least two states"
            )
        self._trajectory_times = [
            value.time[start:].to(
                device=coefficients[0].device, dtype=coefficients[0].real.dtype
            )
            for value in data
        ]
        embedded_controls = [
            self.control_embedding.transform_trajectory(value.to(coefficients[0]))
            for value in controls
        ]
        if any(value.ndim != 2 for value in embedded_controls):
            raise ValueError("embedded controls must be matrices")
        self.n_control_features = embedded_controls[0].shape[0]
        if any(
            value.shape[0] != self.n_control_features for value in embedded_controls
        ):
            raise ValueError("embedded control dimensions differ between trajectories")
        control_trim = start - (control_history - 1)
        self._training_forcing = [
            value[:, control_trim:] for value in embedded_controls
        ]
        if any(
            forcing.shape[1] != state.shape[1] - 1
            for state, forcing in zip(self._latent_trajectories, self._training_forcing)
        ):
            raise ValueError(
                "embedded state transitions and controls could not be aligned"
            )
        first = pt.cat([value[:, :-1] for value in self._latent_trajectories], dim=1)
        second = pt.cat([value[:, 1:] for value in self._latent_trajectories], dim=1)
        forcing = pt.cat(self._training_forcing, dim=1)
        size = first.shape[0]
        if self.stationary_initialization:
            forcing_inverse = pt.linalg.pinv(forcing)
            first_projected = first - (first @ forcing_inverse) @ forcing
            second_projected = second - (second @ forcing_inverse) @ forcing
            discrete_a = _closest_proper_orthogonal(
                second_projected @ first_projected.conj().T
            )
            discrete_b = (second - discrete_a @ first) @ forcing_inverse
        else:
            combined = second @ pt.linalg.pinv(pt.cat((first, forcing), dim=0))
            discrete_a = combined[:, :size]
            discrete_b = combined[:, size:]
        self.dynamics = ContinuousOperator(
            discrete_a,
            normalized_dt,
            self.stationary,
            zero_growth=self.stationary_initialization,
        )
        _, response = self.dynamics.transition_factors(normalized_dt)
        inverse_basis = pt.linalg.pinv(self.dynamics.basis)
        modal_discrete_control = inverse_basis @ discrete_b.to(
            self.dynamics.basis.dtype
        )
        modal_continuous_control = pt.linalg.pinv(response) @ modal_discrete_control
        continuous_control = self.dynamics.basis @ modal_continuous_control
        self.control = pt.nn.Parameter(
            continuous_control.to(self._latent_trajectories[0].dtype)
        )
        self._initialize_noise()
        return self

    def _require_control(self) -> pt.Tensor:
        self._require_fit()
        if self.control is None:
            raise RuntimeError("the DMDc has not been fitted")
        return self.control

    def _apply(self, function):
        super()._apply(function)
        self._training_forcing = [function(value) for value in self._training_forcing]
        return self

    def forward(
        self, initial_state: Optional[StateData], query: ROMQuery
    ) -> Prediction:
        self.input_spec.validate(initial_state, query)
        assert initial_state is not None
        assert query.time is not None and query.forcing is not None
        self._validate_prediction_time(query.time)
        forcing = self._prediction_forcing(
            query.forcing, query.forcing_history, query.time.numel() - 1
        )
        latent = self._encode_initial(initial_state)
        prediction = self._require_fit().controlled_rollout(
            latent,
            self._normalize_time(query.time.to(latent.real)),
            self._require_control(),
            forcing,
        )
        return Prediction(self.decoder(self.embedding.readout(prediction)))

    def _prediction_forcing(
        self,
        forcing: pt.Tensor,
        forcing_history: Optional[pt.Tensor],
        intervals: int,
    ) -> pt.Tensor:
        """Validate and embed future zero-order-held controls."""
        if forcing.ndim == 1:
            forcing = forcing.unsqueeze(0)
        if (
            forcing.ndim not in (2, 3)
            or forcing.shape[0] != self.n_controls
            or forcing.shape[-1] != intervals
        ):
            raise ValueError("forcing has an incompatible shape")
        if self.n_control_features is None:
            raise RuntimeError("the DMDc has not been fitted")
        history = self.control_embedding.history_length
        if intervals == 0:
            shape = (
                (self.n_control_features, 0)
                if forcing.ndim == 2
                else (self.n_control_features, forcing.shape[1], 0)
            )
            return forcing.new_empty(shape)
        if history == 1:
            if forcing_history is not None and forcing_history.numel():
                raise ValueError(
                    "forcing_history is only valid with a delayed control embedding"
                )
            combined = forcing
        else:
            if forcing_history is None:
                raise ValueError(
                    f"forcing_history with {history - 1} interval(s) is required"
                )
            if forcing.ndim == 2:
                if forcing_history.shape != (self.n_controls, history - 1):
                    raise ValueError("forcing_history has an incompatible shape")
            else:
                if forcing_history.ndim == 2:
                    if forcing_history.shape != (self.n_controls, history - 1):
                        raise ValueError("forcing_history has an incompatible shape")
                    forcing_history = forcing_history.unsqueeze(1).expand(
                        -1, forcing.shape[1], -1
                    )
                elif forcing_history.shape != (
                    self.n_controls,
                    forcing.shape[1],
                    history - 1,
                ):
                    raise ValueError("forcing_history has an incompatible shape")
            combined = pt.cat((forcing_history.to(forcing), forcing), dim=-1)
        embedded = self.control_embedding.transform_trajectory(combined)
        if (
            embedded.ndim != forcing.ndim
            or embedded.shape[0] != self.n_control_features
            or embedded.shape[-1] != intervals
        ):
            raise ValueError("control embedding returned an incompatible shape")
        return embedded

    def _rollout(
        self,
        initial: pt.Tensor,
        time: pt.Tensor,
        forcing: Optional[pt.Tensor] = None,
    ) -> pt.Tensor:
        if forcing is None:
            raise ValueError("controlled rollouts require forcing")
        return self._require_fit().controlled_rollout(
            initial,
            self._normalize_time(time),
            self._require_control(),
            forcing,
            uniform_time=True,
        )

    def _window_forcing(
        self, trajectory: int, start: int, horizon: int, backward: bool
    ) -> pt.Tensor:
        forcing = self._training_forcing[trajectory][:, start : start + horizon]
        return forcing.flip(1) if backward else forcing

    def _trajectory_forcing(self, trajectory: int) -> pt.Tensor:
        return self._training_forcing[trajectory]

    @property
    def B(self) -> pt.Tensor:
        return self._require_control() / self.time_scale


__all__ = ["DMDc"]
