"""POD with differentiable interpolation in parameter space."""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Type, Union

import torch as pt

from flowtorch.analysis import SVD

from .base import InputSpec, ParametricSnapshots, Prediction, ROM, ROMQuery, StateData
from .svd_encoder import SVDDecoder, SVDEncoder, fit_svd, training_coefficients


class RBFInterpolator(pt.nn.Module):
    """Trainable Gaussian radial-basis interpolation in normalized coordinates."""

    def __init__(self, smoothing: Optional[float] = None) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.register_buffer("centers", None)
        self.register_buffer("coordinate_mean", None)
        self.register_buffer("coordinate_scale", None)
        self.register_parameter("log_length_scale", None)
        self.register_parameter("weights", None)

    def fit(self, coordinates: pt.Tensor, values: pt.Tensor) -> "RBFInterpolator":
        if coordinates.ndim != 2 or values.ndim != 2:
            raise ValueError("coordinates and values must be two-dimensional")
        if not coordinates.is_floating_point() or pt.is_complex(coordinates):
            raise ValueError("parameter coordinates must have a real floating dtype")
        if coordinates.shape[1] != values.shape[1] or coordinates.shape[1] < 2:
            raise ValueError("at least two coordinate/value pairs are required")
        if not bool(pt.isfinite(coordinates).all()) or not bool(
            pt.isfinite(values).all()
        ):
            raise ValueError("interpolation data must be finite")
        mean = coordinates.mean(dim=1, keepdim=True)
        scale = coordinates.std(dim=1, keepdim=True, unbiased=False)
        scale = scale.clamp_min(pt.finfo(coordinates.dtype).eps)
        normalized = ((coordinates - mean) / scale).T.contiguous()
        distances = pt.pdist(normalized)
        if bool((distances <= pt.finfo(coordinates.dtype).eps).any()):
            raise ValueError("parameter coordinates must be unique")
        length_scale = distances.median()
        kernel = pt.exp(-((pt.cdist(normalized, normalized) / length_scale) ** 2))
        smoothing = (
            10.0 * pt.finfo(coordinates.dtype).eps
            if self.smoothing is None
            else self.smoothing
        )
        if smoothing < 0.0:
            raise ValueError("smoothing must be non-negative")
        kernel = kernel + smoothing * pt.eye(
            kernel.shape[0], dtype=kernel.dtype, device=kernel.device
        )
        self.centers = normalized
        self.coordinate_mean = mean
        self.coordinate_scale = scale
        self.log_length_scale = pt.nn.Parameter(length_scale.log())
        self.weights = pt.nn.Parameter(pt.linalg.solve(kernel, values.T))
        return self

    def forward(self, coordinates: pt.Tensor) -> pt.Tensor:
        if (
            self.centers is None
            or self.weights is None
            or self.log_length_scale is None
        ):
            raise RuntimeError("the RBF interpolator has not been fitted")
        if coordinates.ndim == 1:
            coordinates = coordinates.unsqueeze(-1)
        if (
            coordinates.ndim != 2
            or coordinates.shape[0] != self.coordinate_mean.shape[0]
        ):
            raise ValueError("parameter coordinates have an incompatible shape")
        normalized = ((coordinates - self.coordinate_mean) / self.coordinate_scale).T
        kernel = pt.exp(
            -(pt.cdist(normalized, self.centers) / self.log_length_scale.exp()).square()
        )
        return (kernel @ self.weights).T


def _as_samples(
    samples: Union[ParametricSnapshots, Sequence[ParametricSnapshots]],
) -> list[ParametricSnapshots]:
    if isinstance(samples, ParametricSnapshots):
        return [samples]
    result = list(samples)
    if not result:
        raise ValueError("at least one parametric snapshot set is required")
    if not all(isinstance(value, ParametricSnapshots) for value in result):
        raise TypeError("all data must be ParametricSnapshots instances")
    return result


class PODI(ROM):
    """POD whose latent coefficients are interpolated over parameters."""

    input_spec = InputSpec(parameters=True)

    def __init__(
        self,
        rank: Optional[int] = None,
        *,
        interpolator: Optional[pt.nn.Module] = None,
        subtract_mean: bool = True,
        **svd_options,
    ) -> None:
        super().__init__()
        self.requested_rank = rank
        self.subtract_mean = bool(subtract_mean)
        self.svd_options = dict(svd_options)
        self.svd: Optional[SVD] = None
        self.encoder = SVDEncoder()
        self.decoder = SVDDecoder()
        self.interpolator = RBFInterpolator() if interpolator is None else interpolator
        self._coordinates: Optional[pt.Tensor] = None
        self._targets: Optional[pt.Tensor] = None
        self.training_log: dict[str, list[float]] = {}

    def _apply(self, function):
        if self.svd is not None:
            self.svd._apply_tensor_transform(function)
        super()._apply(function)
        if self._coordinates is not None:
            self._coordinates = function(self._coordinates)
        if self._targets is not None:
            self._targets = function(self._targets)
        return self

    def fit(
        self,
        samples: Union[ParametricSnapshots, Sequence[ParametricSnapshots]],
    ) -> "PODI":
        data = _as_samples(samples)
        parameter_size = data[0].parameters.shape[0]
        for value in data:
            if (
                value.parameters.ndim != 2
                or value.parameters.shape[0] != parameter_size
            ):
                raise ValueError(
                    "parameter arrays must have shape (parameters, samples)"
                )
            count = (
                value.states.shape[1]
                if isinstance(value.states, pt.Tensor)
                else value.states.n_snapshots
            )
            if value.parameters.shape[1] != count:
                raise ValueError("parameter and state sample counts do not match")
        self.svd, _ = fit_svd(
            [value.states for value in data],
            self.requested_rank,
            subtract_mean=self.subtract_mean,
            **self.svd_options,
        )
        self.encoder.set_svd(self.svd)
        self.decoder.set_svd(self.svd)
        coefficients = training_coefficients(self.svd)
        coordinates = pt.cat([value.parameters for value in data], dim=1).to(
            coefficients.real
        )
        targets = coefficients
        if not hasattr(self.interpolator, "fit"):
            raise TypeError("interpolator must define a fit method")
        self.interpolator.fit(coordinates, targets)
        self._coordinates = coordinates
        self._targets = targets
        return self

    def forward(
        self, initial_state: Optional[StateData], query: ROMQuery
    ) -> Prediction:
        self.input_spec.validate(initial_state, query)
        assert query.parameters is not None
        latent = self.interpolator(query.parameters)
        return Prediction(self.decoder(latent))

    def fine_tune(
        self,
        epochs: int = 100,
        *,
        optimizer: Type[pt.optim.Optimizer] = pt.optim.AdamW,
        optimizer_options: Optional[dict] = None,
        loss_function: Optional[Callable[[pt.Tensor, pt.Tensor], pt.Tensor]] = None,
    ) -> dict[str, list[float]]:
        if self._coordinates is None or self._targets is None:
            raise RuntimeError("PODI must be fitted before fine-tuning")
        if epochs < 1:
            raise ValueError("epochs must be positive")
        options = (
            {"lr": 1.0e-3} if optimizer_options is None else dict(optimizer_options)
        )
        optim = optimizer(self.interpolator.parameters(), **options)
        loss_function = (
            (
                lambda prediction, target: (prediction - target).norm()
                / prediction.numel() ** 0.5
            )
            if loss_function is None
            else loss_function
        )
        log = {"train_loss": []}

        def closure() -> pt.Tensor:
            optim.zero_grad()
            loss = loss_function(self.interpolator(self._coordinates), self._targets)
            loss.backward()
            return loss

        for _ in range(epochs):
            optim.step(closure)
            with pt.no_grad():
                log["train_loss"].append(
                    float(
                        loss_function(
                            self.interpolator(self._coordinates), self._targets
                        )
                    )
                )
        self.training_log = log
        return log


__all__ = ["PODI", "RBFInterpolator"]
