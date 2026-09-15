"""Trajectory windowing, learnable noise, and optimizer-based fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Callable, Literal, Optional, Sequence, Type, Union

import torch as pt
import torch.nn.functional as functional

NoiseInitialization = Union[str, Sequence[pt.Tensor]]
SchedulerFactory = Callable[[pt.optim.Optimizer], object]
ModelRegularizer = Callable[[object], pt.Tensor]


def _default_plateau_scheduler(
    optimizer: pt.optim.Optimizer,
) -> pt.optim.lr_scheduler.ReduceLROnPlateau:
    """Construct the default loss-plateau learning-rate scheduler."""
    return pt.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=20,
        min_lr=1.0e-6,
    )


@dataclass(frozen=True)
class NoiseConfig:
    """Configure optional per-snapshot latent measurement noise.

    ``initialization="gaussian"`` initializes the noise from the residual of a
    Gaussian temporal smoother whose width is ``sigma``; it does not draw a
    random Gaussian sample. The variogram-scaled Gaussian residual is the
    default because it provides a data-dependent starting amplitude while
    remaining deterministic.
    """

    initialization: NoiseInitialization = "gaussian"
    sigma: float = 1.0
    weight: float = 0.001
    penalty: Literal["magnitude", "estimated_amplitude"] = "magnitude"
    variogram_max_lag: int = 8
    variogram_iterations: int = 5
    variogram_huber_delta: float = 1.345

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError("noise weight must be non-negative")
        if self.penalty not in ("magnitude", "estimated_amplitude"):
            raise ValueError(
                "noise penalty must be 'magnitude' or 'estimated_amplitude'"
            )
        if self.variogram_max_lag < 2:
            raise ValueError("variogram maximum lag must be at least two")
        if self.variogram_iterations < 1:
            raise ValueError("variogram iterations must be positive")
        if self.variogram_huber_delta <= 0.0:
            raise ValueError("variogram Huber delta must be positive")
        if isinstance(self.initialization, str):
            if self.initialization not in ("zeros", "gaussian"):
                raise ValueError("noise initialization must be 'zeros' or 'gaussian'")
            if self.initialization == "gaussian" and self.sigma <= 0.0:
                raise ValueError("Gaussian noise initialization requires sigma > 0")


@dataclass(frozen=True)
class FrequencySeparation:
    """Smoothly keep the unique modal frequencies ordered and separated.

    ``min_spacing`` and ``temperature`` use the physical frequency units of
    the trajectory time coordinate. By default, the spacing is one Rayleigh
    bin based on the longest fitted trajectory, ``1 / (n_samples * dt)``. For
    real models, the regularizer operates on one non-negative frequency per
    conjugate mode pair. For complex models, it operates on the ordered signed
    frequencies. Optional boundary terms keep the frequencies inside the
    sampling Nyquist interval.
    """

    min_spacing: Optional[float] = None
    weight: float = 0.1
    temperature: Optional[float] = None
    enforce_nyquist: bool = True

    def __post_init__(self) -> None:
        if self.min_spacing is not None and self.min_spacing <= 0.0:
            raise ValueError("minimum frequency spacing must be positive")
        if self.weight < 0.0:
            raise ValueError("frequency-separation weight must be non-negative")
        if self.temperature is not None and self.temperature <= 0.0:
            raise ValueError("frequency-separation temperature must be positive")

    def __call__(self, model: object) -> pt.Tensor:
        dynamics = getattr(model, "dynamics", None)
        if dynamics is None or not hasattr(dynamics, "frequency"):
            raise TypeError("frequency separation requires fitted spectral dynamics")
        time_scale = float(getattr(model, "time_scale"))
        frequencies = dynamics.frequency / time_scale
        zero = frequencies.sum() * 0.0
        if self.weight == 0.0 or frequencies.numel() == 0:
            return zero
        if self.min_spacing is None:
            trajectories = getattr(model, "_trajectory_times", ())
            dt = getattr(model, "dt", None)
            if dt is None or not trajectories:
                raise RuntimeError(
                    "automatic frequency spacing requires fitted trajectories"
                )
            duration = max(value.numel() for value in trajectories) * float(dt)
            min_spacing = 1.0 / duration
        else:
            min_spacing = self.min_spacing
        temperature = (
            0.1 * min_spacing if self.temperature is None else self.temperature
        )

        def violation(value: pt.Tensor) -> pt.Tensor:
            smooth = temperature * functional.softplus(value / temperature)
            return (smooth / min_spacing).square()

        terms = []
        if frequencies.numel() > 1:
            terms.append(violation(min_spacing - frequencies.diff()))
        if self.enforce_nyquist:
            dt = getattr(model, "dt", None)
            if dt is None:
                raise RuntimeError(
                    "frequency bounds require a fitted sampling interval"
                )
            nyquist = 0.5 / float(dt)
            lower = -nyquist if bool(getattr(dynamics, "_complex", False)) else 0.0
            terms.append(violation(frequencies[:1].new_tensor(lower) - frequencies[:1]))
            terms.append(violation(frequencies[-1:] - frequencies.new_tensor(nyquist)))
        return zero if not terms else self.weight * pt.cat(terms).mean()


@dataclass(frozen=True)
class OptimizationStage:
    """Configure one stage of ROM fine-tuning."""

    optimizer: Type[pt.optim.Optimizer]
    epochs: int
    options: Optional[dict[str, Any]] = None
    scheduler: Optional[SchedulerFactory] = _default_plateau_scheduler
    patience: int = 40
    parameters: Literal["all", "dynamics"] = "all"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.patience < 1:
            raise ValueError("stage epochs and patience must be positive")
        if self.parameters not in ("all", "dynamics"):
            raise ValueError("stage parameters must be 'all' or 'dynamics'")


def _gaussian_residual(sequence: pt.Tensor, sigma: float) -> pt.Tensor:
    radius = min(int(ceil(4.0 * sigma)), sequence.shape[1] - 1)
    if radius < 1:
        return pt.zeros_like(sequence)
    coordinate = pt.arange(
        -radius, radius + 1, device=sequence.device, dtype=sequence.real.dtype
    )
    kernel = pt.exp(-0.5 * (coordinate / sigma).square())
    kernel = (kernel / kernel.sum()).to(sequence.dtype)
    data = sequence.unsqueeze(0)
    padded = functional.pad(data, (radius, radius), mode="replicate")
    weight = kernel.reshape(1, 1, -1).repeat(sequence.shape[0], 1, 1)
    smooth = functional.conv1d(padded, weight, groups=sequence.shape[0]).squeeze(0)
    return sequence - smooth


def estimate_noise_std_variogram(
    trajectories: Sequence[pt.Tensor],
    *,
    max_lag: int = 8,
    iterations: int = 5,
    huber_delta: float = 1.345,
) -> pt.Tensor:
    """Estimate channelwise white-noise scales from short-lag variograms.

    A robust MAD scale is computed for each channel and lag. Batched Huber
    iteratively reweighted least squares extrapolates ``a + b * lag**2`` to
    zero lag; the non-negative intercept ``a`` estimates the noise variance.
    All operations remain on the input tensor's device.
    """

    values = list(trajectories)
    if not values:
        raise ValueError("at least one trajectory is required")
    reference = values[0]
    if reference.ndim != 2 or pt.is_complex(reference):
        raise ValueError("variogram estimation requires real matrix trajectories")
    channels = reference.shape[0]
    if any(
        value.ndim != 2
        or value.shape[0] != channels
        or value.device != reference.device
        or value.dtype != reference.dtype
        or pt.is_complex(value)
        for value in values
    ):
        raise ValueError("variogram trajectories must share shape, dtype, and device")
    available = min(value.shape[1] - 1 for value in values)
    lag_count = min(int(max_lag), available)
    if lag_count < 2:
        raise ValueError("variogram estimation requires at least three samples")
    if iterations < 1 or huber_delta <= 0.0:
        raise ValueError("variogram fit options must be positive")

    eps = pt.finfo(reference.dtype).eps
    variograms = []
    for lag in range(1, lag_count + 1):
        differences = pt.cat(
            [value[:, lag:] - value[:, :-lag] for value in values], dim=1
        )
        center = differences.median(dim=1, keepdim=True).values
        mad = (differences - center).abs().median(dim=1).values
        scale = mad / reference.new_tensor(0.6744897501960817)
        variograms.append(0.5 * scale.square())
    target = pt.stack(variograms, dim=1)
    normalized_lag = (
        pt.arange(1, lag_count + 1, device=reference.device, dtype=reference.dtype)
        / lag_count
    )
    design = pt.stack((pt.ones_like(normalized_lag), normalized_lag.square()), dim=1)
    coefficients = pt.linalg.lstsq(design, target.T).solution.T.clamp_min(0.0)
    identity = pt.eye(2, device=reference.device, dtype=reference.dtype)
    for _ in range(iterations):
        residual = target - coefficients @ design.T
        center = residual.median(dim=1, keepdim=True).values
        residual_mad = (residual - center).abs().median(dim=1, keepdim=True).values
        residual_scale = (residual_mad / 0.6744897501960817).clamp_min(eps)
        normalized = residual.abs() / (huber_delta * residual_scale)
        weights = pt.where(
            normalized <= 1.0,
            pt.ones_like(normalized),
            normalized.reciprocal(),
        )
        gram = pt.einsum("cl,li,lj->cij", weights, design, design)
        rhs = pt.einsum("cl,li,cl->ci", weights, design, target)
        coefficients = pt.linalg.solve(gram + eps * identity, rhs).clamp_min(0.0)

    observed_rms = pt.cat(values, dim=1).square().mean(dim=1).sqrt()
    minimum = observed_rms * eps**0.5 + eps
    return coefficients[:, 0].sqrt().clamp_min(minimum)


def initialize_noise(
    latent_trajectories: Sequence[pt.Tensor],
    config: Optional[NoiseConfig],
    project: Callable[[pt.Tensor], pt.Tensor],
    estimated_std: Optional[pt.Tensor] = None,
) -> pt.nn.ParameterList:
    """Create trajectory-local latent noise parameters."""
    if config is None:
        return pt.nn.ParameterList()
    initialization = config.initialization
    values: list[pt.Tensor]
    if isinstance(initialization, str):
        if initialization == "zeros":
            values = [pt.zeros_like(value) for value in latent_trajectories]
        else:
            values = [
                _gaussian_residual(value, config.sigma) for value in latent_trajectories
            ]
            if estimated_std is not None:
                combined = pt.cat(values, dim=1)
                rms = combined.square().mean(dim=1).sqrt()
                epsilon = pt.finfo(combined.dtype).eps
                factor = estimated_std.to(combined) / rms.clamp_min(epsilon)
                values = [value * factor[:, None] for value in values]
    else:
        supplied = list(initialization)
        if len(supplied) != len(latent_trajectories):
            raise ValueError("provide exactly one noise tensor per trajectory")
        values = []
        for noise, latent in zip(supplied, latent_trajectories):
            value = noise if noise.shape == latent.shape else project(noise)
            if value.shape != latent.shape:
                raise ValueError("user noise has an incompatible shape")
            values.append(value.to(device=latent.device, dtype=latent.dtype))
    return pt.nn.ParameterList([pt.nn.Parameter(value.clone()) for value in values])


def _default_loss(prediction: pt.Tensor, target: pt.Tensor) -> pt.Tensor:
    """Mean-squared rollout error with a vanishing exact-fit gradient."""
    return (prediction - target).abs().square().mean()


class TrajectoryFineTuner:
    """Mixin implementing trajectory-aware multi-step fine-tuning."""

    _latent_trajectories: list[pt.Tensor]
    _trajectory_times: list[pt.Tensor]
    noise_parameters: pt.nn.ParameterList
    noise_config: Optional[NoiseConfig]
    _estimated_noise_std: Optional[pt.Tensor]

    def _rollout(
        self,
        initial: pt.Tensor,
        time: pt.Tensor,
        forcing: Optional[pt.Tensor] = None,
    ) -> pt.Tensor:
        raise NotImplementedError

    def _window_forcing(
        self, trajectory: int, start: int, horizon: int, backward: bool
    ) -> Optional[pt.Tensor]:
        return None

    def _trajectory_forcing(self, trajectory: int) -> Optional[pt.Tensor]:
        """Return all interval controls for vectorized window extraction."""
        return None

    @property
    def learned_noise(self) -> tuple[pt.Tensor, ...]:
        return tuple(value.detach() for value in self.noise_parameters)

    @property
    def denoised_latent_trajectories(self) -> tuple[pt.Tensor, ...]:
        return tuple(
            self._clean(index).detach()
            for index in range(len(self._latent_trajectories))
        )

    def _clean(self, trajectory: int) -> pt.Tensor:
        value = self._latent_trajectories[trajectory]
        return (
            value
            if not self.noise_parameters
            else value - self.noise_parameters[trajectory]
        )

    @property
    def estimated_noise_std(self) -> Optional[pt.Tensor]:
        """Robust variogram noise-scale estimate, when requested."""
        value = self._estimated_noise_std
        return None if value is None else value.detach()

    def _noise_regularization(self) -> pt.Tensor:
        noise = pt.cat(list(self.noise_parameters), dim=1)
        assert self.noise_config is not None
        if self.noise_config.penalty == "magnitude":
            return self.noise_config.weight * noise.norm() / noise.numel() ** 0.5
        if self._estimated_noise_std is None:
            raise RuntimeError("estimated-amplitude noise scales are unavailable")
        scale = self._estimated_noise_std.to(noise.real)
        epsilon = pt.finfo(noise.real.dtype).eps
        rms = (noise.abs().square().mean(dim=1) + (epsilon * scale).square()).sqrt()
        return self.noise_config.weight * ((rms / scale) - 1.0).square().mean()

    def _make_windows(
        self,
        horizon: int,
        n_shift: int,
        validation_fraction: float,
        forward_backward: bool,
    ) -> tuple[list[tuple[int, int, bool]], list[tuple[int, int, bool]]]:
        if horizon < 1 or n_shift < 1:
            raise ValueError("horizon and n_shift must be positive")
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie in [0, 1)")
        train: list[tuple[int, int, bool]] = []
        validation: list[tuple[int, int, bool]] = []
        for index, trajectory in enumerate(self._latent_trajectories):
            count = trajectory.shape[1]
            if validation_fraction == 0.0:
                split = count
            else:
                validation_count = max(
                    int(ceil(count * validation_fraction)), horizon + 1
                )
                split = count - validation_count
                if split < horizon + 1:
                    split = count
            if split < horizon + 1:
                raise ValueError("training segment is too short for the horizon")
            starts = range(0, split - horizon, n_shift)
            train.extend((index, start, False) for start in starts)
            if forward_backward:
                train.extend((index, start, True) for start in starts)
            if validation_fraction > 0.0 and split < count:
                starts = range(split, count - horizon, n_shift)
                validation.extend((index, start, False) for start in starts)
                if forward_backward:
                    validation.extend((index, start, True) for start in starts)
        if not train:
            raise ValueError("no training windows can be created")
        return train, validation

    def _window_loss(
        self,
        windows: Sequence[tuple[int, int, bool]],
        horizon: int,
        loss_function: Callable[[pt.Tensor, pt.Tensor], pt.Tensor],
    ) -> pt.Tensor:
        grouped = {
            False: [value for value in windows if not value[2]],
            True: [value for value in windows if value[2]],
        }
        losses = []
        weights = []
        for backward, selected in grouped.items():
            if not selected:
                continue
            targets: list[pt.Tensor] = []
            controls: list[pt.Tensor] = []
            for trajectory in sorted({value[0] for value in selected}):
                state = self._clean(trajectory)
                starts = pt.tensor(
                    [
                        start
                        for current_trajectory, start, _ in selected
                        if current_trajectory == trajectory
                    ],
                    device=state.device,
                )
                state_indices = starts[:, None] + pt.arange(
                    horizon + 1, device=state.device
                )
                target = state[:, state_indices]
                if backward:
                    target = target.flip(2)
                targets.append(target)
                forcing = self._trajectory_forcing(trajectory)
                if forcing is not None:
                    forcing_indices = starts[:, None] + pt.arange(
                        horizon, device=forcing.device
                    )
                    control = forcing[:, forcing_indices]
                    controls.append(control.flip(2) if backward else control)
            target_batch = pt.cat(targets, dim=1)
            first_trajectory, first_start, _ = selected[0]
            time = self._trajectory_times[first_trajectory][
                first_start : first_start + horizon + 1
            ]
            if backward:
                time = time.flip(0)
            forcing_batch = None if not controls else pt.cat(controls, dim=1)
            prediction = self._rollout(target_batch[:, :, 0], time, forcing_batch)
            losses.append(loss_function(prediction[:, :, 1:], target_batch[:, :, 1:]))
            weights.append(len(selected))
        total = sum(weights)
        rollout = sum(loss * (weight / total) for loss, weight in zip(losses, weights))
        if self.noise_parameters and self.noise_config is not None:
            rollout = rollout + self._noise_regularization()
        return rollout

    def _optimization_parameters(
        self, selection: Literal["all", "dynamics"]
    ) -> list[pt.nn.Parameter]:
        if selection == "all":
            return list(self.parameters())
        dynamics = getattr(self, "dynamics", None)
        if not isinstance(dynamics, pt.nn.Module):
            raise RuntimeError("dynamics-only optimization requires fitted dynamics")
        parameters = list(dynamics.parameters())
        control = getattr(self, "control", None)
        if isinstance(control, pt.nn.Parameter):
            parameters.append(control)
        return parameters

    @staticmethod
    def _default_optimization_stages(
        epochs: int,
        optimizer_options: Optional[dict[str, Any]],
        scheduler: Optional[SchedulerFactory],
        patience: int,
        lbfgs_patience: int = 10,
    ) -> list[OptimizationStage]:
        if epochs == 1:
            return [
                OptimizationStage(
                    pt.optim.AdamW,
                    1,
                    optimizer_options,
                    scheduler,
                    patience,
                )
            ]
        lbfgs_epochs = max(1, epochs // 3)
        return [
            OptimizationStage(
                pt.optim.AdamW,
                epochs - lbfgs_epochs,
                optimizer_options,
                scheduler,
                patience,
            ),
            OptimizationStage(
                pt.optim.LBFGS,
                lbfgs_epochs,
                {"line_search_fn": "strong_wolfe"},
                None,
                lbfgs_patience,
                "all",
            ),
        ]

    def fine_tune(
        self,
        epochs: int = 150,
        *,
        horizon: int = 8,
        n_shift: int = 1,
        forward_backward: bool = False,
        validation_fraction: float = 0.0,
        batch_size: Optional[int] = None,
        optimizer: Optional[Type[pt.optim.Optimizer]] = None,
        optimizer_options: Optional[dict[str, Any]] = None,
        stages: Optional[Sequence[OptimizationStage]] = None,
        loss_function: Optional[Callable[[pt.Tensor, pt.Tensor], pt.Tensor]] = None,
        regularizers: Optional[Sequence[ModelRegularizer]] = None,
        scheduler: Optional[SchedulerFactory] = _default_plateau_scheduler,
        patience: int = 40,
        lbfgs_patience: int = 10,
        noise_update_tolerance: float = 1.0e-4,
        validation_loss_weight: float = 0.5,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Fine-tune trainable ROM components on sliding trajectory windows.

        The fitted analytical model is evaluated as epoch zero and retained if
        no optimizer update improves the selection loss. ``train_loss`` and
        ``val_loss`` contain post-update values for epochs one onward; the
        corresponding epoch-zero values and restored-checkpoint diagnostics are
        stored separately in the returned log. When validation windows exist,
        the scheduler and early stopping monitor

        ``(1 - validation_loss_weight) * train_loss
        + validation_loss_weight * val_loss``.

        Without validation windows they monitor training loss and ignore the
        weight. When noise learning is active, early stopping additionally
        requires its relative parameter update to remain below
        ``noise_update_tolerance`` for the optimizer stage's loss patience.
        Model regularizers are added once to this mixed data loss.
        By default, :class:`FrequencySeparation` discourages modal-frequency
        collisions using one Rayleigh bin as the minimum spacing. Pass an
        empty sequence as ``regularizers`` to disable regularization.
        DMD and DMDc use forward rollout loss only by default. Set
        ``forward_backward=True`` to add the corresponding backward windows.

        Unless ``optimizer`` or ``stages`` is supplied, the epoch budget is
        split approximately two-to-one between AdamW and full-batch LBFGS; the
        default of 150 epochs therefore runs 100 AdamW and 50 LBFGS stage
        epochs. Both stages optimize all trainable model parameters. LBFGS uses
        strong-Wolfe line search and otherwise retains the PyTorch defaults.
        Supplying ``optimizer`` retains the single-stage interface. Explicit
        ``stages`` override the epoch budget and all single-stage optimizer
        settings.
        """
        if epochs < 1 or patience < 1 or lbfgs_patience < 1:
            raise ValueError("epochs and patience values must be positive")
        if noise_update_tolerance <= 0.0:
            raise ValueError("noise update tolerance must be positive")
        if not 0.0 <= validation_loss_weight <= 1.0:
            raise ValueError("validation_loss_weight must lie in [0, 1]")
        if stages is not None and optimizer is not None:
            raise ValueError("provide either stages or optimizer, not both")
        train, validation = self._make_windows(
            horizon, n_shift, validation_fraction, forward_backward
        )
        loss_function = _default_loss if loss_function is None else loss_function
        regularizers = (
            (FrequencySeparation(),) if regularizers is None else tuple(regularizers)
        )
        if stages is not None:
            optimization_stages = list(stages)
            if not optimization_stages:
                raise ValueError("at least one optimization stage is required")
        elif optimizer is None:
            optimization_stages = self._default_optimization_stages(
                epochs,
                optimizer_options,
                scheduler,
                patience,
                lbfgs_patience,
            )
        else:
            optimization_stages = [
                OptimizationStage(
                    optimizer,
                    epochs,
                    optimizer_options,
                    scheduler,
                    patience,
                )
            ]
        generator = pt.Generator().manual_seed(seed)

        def evaluate(windows: Sequence[tuple[int, int, bool]]) -> pt.Tensor:
            return self._window_loss(windows, horizon, loss_function)

        def regularization_loss() -> pt.Tensor:
            reference = next(self.parameters())
            total = reference.real.sum() * 0.0
            for regularizer in regularizers:
                value = regularizer(self)
                if value.ndim != 0 or pt.is_complex(value):
                    raise ValueError("regularizers must return real scalar tensors")
                total = total + value
            return total

        def objective(windows: Sequence[tuple[int, int, bool]]) -> pt.Tensor:
            return evaluate(windows) + regularization_loss()

        def mixed_data_loss(train_loss: float, validation_loss: float) -> float:
            if not validation:
                return train_loss
            return (
                1.0 - validation_loss_weight
            ) * train_loss + validation_loss_weight * validation_loss

        with pt.no_grad():
            initial_train_loss = float(evaluate(train))
            initial_validation_loss = (
                float(evaluate(validation)) if validation else initial_train_loss
            )
            initial_regularization_loss = float(regularization_loss())
            initial_selection_loss = (
                mixed_data_loss(initial_train_loss, initial_validation_loss)
                + initial_regularization_loss
            )
        log: dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "regularization_loss": [],
            "selection_loss": [],
            "learning_rate": [],
            "noise_update": [],
            "stage": [],
            "stage_epoch": [],
            "initial_train_loss": initial_train_loss,
            "initial_validation_loss": initial_validation_loss,
            "initial_regularization_loss": initial_regularization_loss,
            "initial_selection_loss": initial_selection_loss,
            "best_epoch": 0,
            "best_train_loss": initial_train_loss,
            "best_validation_loss": initial_validation_loss,
            "best_regularization_loss": initial_regularization_loss,
            "best_selection_loss": initial_selection_loss,
        }
        best = initial_selection_loss
        best_state = {
            key: value.detach().clone() for key, value in self.state_dict().items()
        }
        epoch = 0

        for stage_index, stage in enumerate(optimization_stages):
            if stage_index:
                self.load_state_dict(best_state)
            parameters = self._optimization_parameters(stage.parameters)
            # ``torch.optim.LBFGS`` flattens every gradient with ``view(-1)``.
            # Parameters originating from spectral decompositions can retain a
            # non-contiguous layout, which in turn produces non-contiguous
            # gradients and makes that internal ``view`` fail.  Preserve the
            # parameter values while giving LBFGS a contiguous leaf layout.
            if issubclass(stage.optimizer, pt.optim.LBFGS):
                for parameter in parameters:
                    if not parameter.is_contiguous():
                        parameter.data = parameter.data.contiguous()
            options = {} if stage.options is None else dict(stage.options)
            if issubclass(stage.optimizer, pt.optim.AdamW):
                options.setdefault("lr", 1.0e-3)
                options.setdefault("weight_decay", 0.0)
            elif issubclass(stage.optimizer, pt.optim.LBFGS):
                options.setdefault("line_search_fn", "strong_wolfe")
            else:
                options.setdefault("lr", 1.0e-3)
            optim = stage.optimizer(parameters, **options)
            is_lbfgs = isinstance(optim, pt.optim.LBFGS)
            if is_lbfgs and batch_size is not None and len(optimization_stages) == 1:
                raise ValueError("LBFGS requires deterministic full-batch training")
            schedule = None if stage.scheduler is None else stage.scheduler(optim)
            stale = 0
            noise_stale = 0
            previous_noise = tuple(
                value.detach().clone() for value in self.noise_parameters
            )

            for stage_epoch in range(1, stage.epochs + 1):
                epoch += 1
                if is_lbfgs:

                    def closure() -> pt.Tensor:
                        self.zero_grad(set_to_none=True)
                        loss = objective(train)
                        loss.backward()
                        return loss

                    optim.step(closure)
                else:
                    order = pt.randperm(len(train), generator=generator).tolist()
                    selected = [train[index] for index in order]
                    size = len(selected) if batch_size is None else batch_size
                    for start in range(0, len(selected), size):
                        batch = selected[start : start + size]

                        def closure() -> pt.Tensor:
                            self.zero_grad(set_to_none=True)
                            loss = objective(batch)
                            loss.backward()
                            return loss

                        optim.step(closure)
                with pt.no_grad():
                    train_loss = float(evaluate(train))
                    val_loss = float(evaluate(validation)) if validation else train_loss
                    regularization = float(regularization_loss())
                    selected_loss = (
                        mixed_data_loss(train_loss, val_loss) + regularization
                    )
                    if previous_noise:
                        update_squared = sum(
                            (current - previous).norm().square()
                            for current, previous in zip(
                                self.noise_parameters, previous_noise
                            )
                        )
                        current_squared = sum(
                            current.norm().square() for current in self.noise_parameters
                        )
                        previous_squared = sum(
                            previous.norm().square() for previous in previous_noise
                        )
                        scale = pt.maximum(
                            current_squared.sqrt(), previous_squared.sqrt()
                        ).clamp_min(pt.finfo(current_squared.dtype).eps)
                        noise_update = float(update_squared.sqrt() / scale)
                        previous_noise = tuple(
                            value.detach().clone() for value in self.noise_parameters
                        )
                    else:
                        noise_update = None
                log["train_loss"].append(train_loss)
                log["val_loss"].append(val_loss)
                log["regularization_loss"].append(regularization)
                log["selection_loss"].append(selected_loss)
                log["stage"].append(stage.optimizer.__name__)
                log["stage_epoch"].append(stage_epoch)
                if schedule is not None:
                    try:
                        schedule.step(selected_loss)
                    except TypeError:
                        schedule.step()
                log["learning_rate"].append(optim.param_groups[0]["lr"])
                if noise_update is not None:
                    log["noise_update"].append(noise_update)
                    noise_stale = (
                        noise_stale + 1 if noise_update < noise_update_tolerance else 0
                    )
                if selected_loss < best:
                    best = selected_loss
                    best_state = {
                        key: value.detach().clone()
                        for key, value in self.state_dict().items()
                    }
                    log["best_epoch"] = epoch
                    log["best_train_loss"] = train_loss
                    log["best_validation_loss"] = val_loss
                    log["best_regularization_loss"] = regularization
                    log["best_selection_loss"] = selected_loss
                    stale = 0
                else:
                    stale += 1
                loss_converged = stale >= stage.patience
                noise_converged = not previous_noise or noise_stale >= stage.patience
                if loss_converged and noise_converged:
                    break
        self.load_state_dict(best_state)
        self.training_log = log
        return log


__all__ = [
    "FrequencySeparation",
    "NoiseConfig",
    "estimate_noise_std_variogram",
    "OptimizationStage",
    "TrajectoryFineTuner",
    "initialize_noise",
]
