"""Continuous-time dynamic mode decomposition as a compositional ROM."""

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import Any, Optional, Sequence, Union

import torch as pt

from flowtorch.analysis import SVD
from flowtorch.analysis.state_vector import StateVectorSource

from .base import (
    InputSpec,
    LatentEmbedding,
    Prediction,
    ROM,
    ROMQuery,
    StateData,
    Trajectory,
)
from .dynamics import ContinuousOperator, initialize_discrete_operator
from .svd_encoder import SVDDecoder, SVDEncoder, fit_svd, training_coefficients
from .training import (
    NoiseConfig,
    TrajectoryFineTuner,
    estimate_noise_std_variogram,
    initialize_noise,
)

TimeScale = Union[str, float, None]


def _validate_time_scale(value: TimeScale) -> TimeScale:
    if value is None or value == "sampling_interval":
        return value
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(
            "time_scale must be 'sampling_interval', None, or a positive number"
        )
    return float(value)


def _as_trajectories(
    trajectories: Union[Trajectory, Sequence[Trajectory]],
) -> list[Trajectory]:
    if isinstance(trajectories, Trajectory):
        return [trajectories]
    result = list(trajectories)
    if not result:
        raise ValueError("at least one trajectory is required")
    if not all(isinstance(value, Trajectory) for value in result):
        raise TypeError("all training data must be Trajectory instances")
    return result


def _snapshot_count(states: StateData) -> int:
    if isinstance(states, pt.Tensor):
        if states.ndim != 2:
            raise ValueError("trajectory states must have shape (state, time)")
        return states.shape[1]
    if isinstance(states, StateVectorSource):
        return states.n_snapshots
    raise TypeError("unsupported trajectory state type")


def _validate_times(trajectories: Sequence[Trajectory]) -> float:
    common_dt: Optional[pt.Tensor] = None
    for trajectory in trajectories:
        time = trajectory.time
        if time.ndim != 1 or time.numel() != _snapshot_count(trajectory.states):
            raise ValueError("trajectory time and state counts do not match")
        if time.numel() < 2 or not bool((time.diff() > 0.0).all()):
            raise ValueError("trajectory time must be strictly increasing")
        delta = time.diff()
        if not pt.allclose(delta, delta[0].expand_as(delta)):
            raise ValueError("DMD training trajectories must be uniformly sampled")
        if common_dt is None:
            common_dt = delta[0]
        elif not pt.allclose(delta[0], common_dt.to(delta)):
            raise ValueError("all training trajectories must use the same time step")
    assert common_dt is not None
    return float(common_dt)


def _split_columns(matrix: pt.Tensor, lengths: Sequence[int]) -> list[pt.Tensor]:
    return list(matrix.split(tuple(lengths), dim=1))


class DMD(ROM, TrajectoryFineTuner):
    """POD plus differentiable continuous-time linear latent dynamics.

    By default, the analytical fit starts from a proper-orthogonal discrete
    operator and exactly zero continuous growth. With ``stationary=False`` the
    growth rates remain trainable; ``stationary=True`` fixes them at zero.
    ``stationary_initialization=False`` selects an unconstrained least-squares
    initialization when growth or decay should be present from epoch zero.
    Physical time is normalized internally by the training sampling interval
    unless ``time_scale`` is a positive characteristic time or ``None``.
    """

    input_spec = InputSpec(initial_state=True, time=True)

    def __init__(
        self,
        rank: Optional[int] = None,
        *,
        stationary: bool = False,
        stationary_initialization: bool = True,
        subtract_mean: bool = True,
        embedding: Optional[LatentEmbedding] = None,
        noise: Optional[NoiseConfig] = None,
        time_scale: TimeScale = "sampling_interval",
        **svd_options,
    ) -> None:
        super().__init__()
        self.requested_rank = rank
        self.stationary = bool(stationary)
        self.stationary_initialization = bool(stationary_initialization or stationary)
        self._time_scale_option = _validate_time_scale(time_scale)
        self._time_scale_factor: Optional[float] = None
        self.embedding = LatentEmbedding() if embedding is None else embedding
        self.subtract_mean = bool(subtract_mean)
        self.svd_options = dict(svd_options)
        self.svd: Optional[SVD] = None
        self.encoder = SVDEncoder()
        self.decoder = SVDDecoder()
        self.dynamics: Optional[ContinuousOperator] = None
        self.noise_config = noise
        self.register_buffer("_estimated_noise_std", None, persistent=False)
        self.noise_parameters = pt.nn.ParameterList()
        self._pod_trajectories: list[pt.Tensor] = []
        self._latent_trajectories: list[pt.Tensor] = []
        self._trajectory_times: list[pt.Tensor] = []
        self._state_embedding_trim = 0
        self.training_log: dict[str, Any] = {}
        self.dt: Optional[float] = None

    def fit(self, trajectories: Union[Trajectory, Sequence[Trajectory]]) -> "DMD":
        """Fit POD and analytically initialize the continuous operator."""
        data = _as_trajectories(trajectories)
        self.svd, lengths = fit_svd(
            [value.states for value in data],
            self.requested_rank,
            subtract_mean=self.subtract_mean,
            **self.svd_options,
        )
        self.encoder.set_svd(self.svd)
        self.decoder.set_svd(self.svd)
        coefficients = _split_columns(training_coefficients(self.svd), lengths)
        return self._fit_regressor(data, coefficients)

    def fit_regressor(
        self, trajectories: Union[Trajectory, Sequence[Trajectory]]
    ) -> "DMD":
        """Fit only the latent dynamics using the existing fixed SVD."""
        if self.svd is None:
            raise RuntimeError("fit the SVD before fitting the regressor")
        data = _as_trajectories(trajectories)
        coefficients = [self.encoder(value.states) for value in data]
        return self._fit_regressor(data, coefficients)

    def _fit_regressor(
        self, data: Sequence[Trajectory], coefficients: Sequence[pt.Tensor]
    ) -> "DMD":
        self.dt = _validate_times(data)
        normalized_dt = self._resolve_time_scale(self.dt)

        history = self.embedding.history_length
        if not isinstance(history, int) or history < 1:
            raise ValueError("embedding history_length must be a positive integer")
        self._pod_trajectories = list(coefficients)
        self._state_embedding_trim = 0
        self._latent_trajectories = [
            self.embedding.transform_trajectory(value) for value in coefficients
        ]
        if any(
            value.ndim != 2
            or value.shape[1] != trajectory.time.numel() - history + 1
            or value.shape[1] < 2
            for value, trajectory in zip(self._latent_trajectories, data)
        ):
            raise ValueError(
                "embedded trajectories must remove history_length - 1 samples "
                "and retain at least two states"
            )
        self._trajectory_times = [
            value.time[history - 1 :].to(
                device=coefficients[0].device, dtype=coefficients[0].real.dtype
            )
            for value in data
        ]
        first = pt.cat([value[:, :-1] for value in self._latent_trajectories], dim=1)
        second = pt.cat([value[:, 1:] for value in self._latent_trajectories], dim=1)
        discrete = initialize_discrete_operator(
            first, second, self.stationary_initialization
        )
        self.dynamics = ContinuousOperator(
            discrete,
            normalized_dt,
            self.stationary,
            zero_growth=self.stationary_initialization,
        )
        self._initialize_noise()
        return self

    def _initialize_noise(self) -> None:
        """Create POD-coordinate noise before applying the state embedding."""
        config = self.noise_config
        estimate_noise_scale = config is not None and (
            config.penalty == "estimated_amplitude"
            or config.initialization == "gaussian"
        )
        if estimate_noise_scale:
            assert config is not None
            self._estimated_noise_std = estimate_noise_std_variogram(
                self._pod_trajectories,
                max_lag=config.variogram_max_lag,
                iterations=config.variogram_iterations,
                huber_delta=config.variogram_huber_delta,
            )
        else:
            self._estimated_noise_std = None
        self.noise_parameters = initialize_noise(
            self._pod_trajectories,
            config,
            self.encoder,
            self._estimated_noise_std,
        )

    def _clean(self, trajectory: int) -> pt.Tensor:
        """Embed a consistently denoised POD trajectory for fine-tuning."""
        if not self.noise_parameters:
            return self._latent_trajectories[trajectory]
        pod = self._pod_trajectories[trajectory] - self.noise_parameters[trajectory]
        embedded = self.embedding.transform_trajectory(pod)
        return embedded[:, self._state_embedding_trim :]

    def _resolve_time_scale(self, dt: float) -> float:
        self._time_scale_factor = (
            dt
            if self._time_scale_option == "sampling_interval"
            else (
                1.0
                if self._time_scale_option is None
                else float(self._time_scale_option)
            )
        )
        return dt / self.time_scale

    def _require_fit(self) -> ContinuousOperator:
        if self.dynamics is None:
            raise RuntimeError("the DMD has not been fitted")
        return self.dynamics

    def _apply(self, function):
        if self.svd is not None:
            self.svd._apply_tensor_transform(function)
        super()._apply(function)
        self._latent_trajectories = [
            function(value) for value in self._latent_trajectories
        ]
        self._pod_trajectories = [function(value) for value in self._pod_trajectories]
        self._trajectory_times = [function(value) for value in self._trajectory_times]
        return self

    @staticmethod
    def _validate_prediction_time(time: pt.Tensor) -> None:
        if time.ndim != 1 or time.numel() < 1:
            raise ValueError("prediction time must be a non-empty vector")
        if time.numel() > 1 and not bool((time.diff() > 0.0).all()):
            raise ValueError("prediction time must be strictly increasing")

    def forward(
        self, initial_state: Optional[StateData], query: ROMQuery
    ) -> Prediction:
        self.input_spec.validate(initial_state, query)
        assert initial_state is not None and query.time is not None
        self._validate_prediction_time(query.time)
        embedded = self._encode_initial(initial_state)
        offsets = self._normalize_time(query.time.to(embedded.real))
        prediction = self._require_fit().propagate(embedded, offsets)
        return Prediction(self.decoder(self.embedding.readout(prediction)))

    def _encode_initial(self, initial_state: StateData) -> pt.Tensor:
        """Encode one state or a possibly batched physical-state history."""
        history = self.embedding.history_length
        if history == 1:
            return self.embedding.initial(self.encoder(initial_state))
        if isinstance(initial_state, pt.Tensor) and initial_state.ndim == 3:
            if initial_state.shape[1] != history:
                raise ValueError(
                    f"initial history must contain {history} physical states"
                )
            flattened = initial_state.reshape(initial_state.shape[0], -1)
            encoded = self.encoder(flattened).reshape(
                self.rank, history, initial_state.shape[2]
            )
        else:
            encoded = self.encoder(initial_state)
        return self.embedding.initial(encoded)

    def _rollout(
        self,
        initial: pt.Tensor,
        time: pt.Tensor,
        forcing: Optional[pt.Tensor] = None,
    ) -> pt.Tensor:
        offsets = self._normalize_time(time)
        return self._require_fit().propagate(initial, offsets)

    @property
    def operator(self) -> pt.Tensor:
        return self._require_fit().operator / self.time_scale

    @property
    def frequency(self) -> pt.Tensor:
        return self._require_fit().frequencies / self.time_scale

    @property
    def growth_rate(self) -> pt.Tensor:
        return self._require_fit().growth_rate / self.time_scale

    @property
    def time_scale(self) -> float:
        if self._time_scale_factor is None:
            raise RuntimeError("the DMD has not been fitted")
        return self._time_scale_factor

    def _normalize_time(self, time: pt.Tensor) -> pt.Tensor:
        return (time - time[0]) / self.time_scale

    @property
    def rank(self) -> int:
        if self.svd is None:
            raise RuntimeError("the DMD has not been fitted")
        return self.svd.rank

    @property
    def embedded_rank(self) -> int:
        """Dimension of the state seen by the continuous dynamics."""
        return self._require_fit().latent_size


__all__ = ["DMD"]
