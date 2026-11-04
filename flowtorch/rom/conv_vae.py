"""Conditional image compression and autoregressive latent dynamics.

Image networks use batch-first tensors, unlike the state-vector POD models.
No spatial resampling, normalization, or device transfer is implicit.
"""

from __future__ import annotations

from bisect import bisect_right
from math import isfinite, prod
from typing import Iterable, Optional, Sequence

import torch as pt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from flowtorch.provenance import tracked
from .base import InputSpec, Prediction, ROM, ROMQuery
from .autoregressive import AutoregressiveModel, LatentDynamics, _positive, _condition


class ImageSequenceDataset(Dataset):
    """Views of windows that never cross trajectory boundaries.

    :param trajectories: tensors with shape ``(time, channels, height, width)``
    :param parameters: optional fixed vector per trajectory, e.g. scaled Mach
        and angle of attack; time-dependent parameters are rejected
    :param history: number of observed frames, ordered oldest to newest
    :param horizon: number of subsequent target frames

    Each item is a dictionary containing ``history``, ``future`` and, for a
    conditional dataset, ``parameters``. Storage is retained by reference;
    trajectories are not concatenated or copied. Inputs must already be finite
    and normalized. Different resolutions require separate DataLoader batches.
    This is an in-memory dataset, not an out-of-core loader.
    """

    def __init__(
        self,
        trajectories: Sequence[pt.Tensor],
        *,
        parameters: Optional[Sequence[pt.Tensor]] = None,
        history: int = 1,
        horizon: int = 1,
    ) -> None:
        self.history = _positive(history, "history")
        self.horizon = _positive(horizon, "horizon")
        self.trajectories = tuple(trajectories)
        self.parameters = None if parameters is None else tuple(parameters)
        if not self.trajectories:
            raise ValueError("at least one trajectory is required")
        if self.parameters is not None:
            if len(self.parameters) != len(self.trajectories):
                raise ValueError("provide one parameter vector per trajectory")
            size = self.parameters[0].numel()
            if size < 1 or any(p.shape != (size,) for p in self.parameters):
                raise ValueError("parameters must be fixed vectors of the same size")
            if any(not bool(pt.isfinite(p).all()) for p in self.parameters):
                raise ValueError("parameters must be finite")
        self._ends = [0]
        for states in self.trajectories:
            if states.ndim != 4 or min(states.shape) < 1:
                raise ValueError("trajectories must have shape (time, channels, H, W)")
            count = states.shape[0] - history - horizon + 1
            if count < 1:
                raise ValueError(
                    "each trajectory must contain history + horizon frames"
                )
            self._ends.append(self._ends[-1] + count)

    def __len__(self) -> int:
        return self._ends[-1]

    def __getitem__(self, index: int) -> dict[str, pt.Tensor]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        trajectory = bisect_right(self._ends, index) - 1
        start = index - self._ends[trajectory]
        stop = start + self.history
        states = self.trajectories[trajectory]
        item = {
            "history": states[start:stop],
            "future": states[stop : stop + self.horizon],
        }
        if self.parameters is not None:
            item["parameters"] = self.parameters[trajectory]
        return item


class ConvVAE(nn.Module):
    """Conditional convolutional VAE for the newest frame of an image history.

    :param image_shape: ``(channels, height, width)``, including odd dimensions
    :param latent_dim: number of stochastic latent coordinates
    :param channels: feature widths of the strided encoder layers
    :param history: observed frame count; 2D mode requires one frame
    :param conv_dim: 2 for single images or 3 for time-height-width convolutions
    :param parameter_dim: fixed operating-condition vector size (zero disables it)

    ``encode`` consumes ``(batch, history, C, H, W)``. In 3D mode every encoder
    layer uses a temporal kernel of three with temporal stride one. The decoder
    is spatial (2D): it reconstructs only the newest frame, not the entire
    history. Explicit interpolation sizes recover the configured resolution
    exactly. Parameters are concatenated at the latent bottleneck in both
    directions. The prior is a standard normal, independent of parameters.
    """

    def __init__(
        self,
        image_shape: Sequence[int],
        latent_dim: int,
        *,
        channels: Sequence[int] = (16, 32),
        history: int = 1,
        conv_dim: int = 2,
        parameter_dim: int = 0,
    ) -> None:
        super().__init__()
        self.image_shape = tuple(image_shape)
        if len(self.image_shape) != 3:
            raise ValueError("image_shape must be (channels, height, width)")
        for size in self.image_shape:
            _positive(size, "image_shape entries")
        self.latent_dim = _positive(latent_dim, "latent_dim")
        self.history = _positive(history, "history")
        self.channels = tuple(channels)
        if not self.channels:
            raise ValueError("channels must not be empty")
        for size in self.channels:
            _positive(size, "channels entries")
        if conv_dim not in (2, 3) or (conv_dim == 2 and history != 1):
            raise ValueError("conv_dim must be 2 or 3; history > 1 requires 3")
        if type(parameter_dim) is not int or parameter_dim < 0:
            raise ValueError("parameter_dim must be a non-negative integer")
        self.parameter_dim = parameter_dim
        self.conv_dim = conv_dim
        widths = (self.image_shape[0], *self.channels)
        layers = []
        self._sizes = [self.image_shape[1:]]
        for before, after in zip(widths[:-1], widths[1:]):
            if conv_dim == 3:
                layer = nn.Conv3d(before, after, 3, stride=(1, 2, 2), padding=1)
            else:
                layer = nn.Conv2d(before, after, 3, stride=2, padding=1)
            layers.extend((layer, nn.SiLU()))
            self._sizes.append(tuple((size + 1) // 2 for size in self._sizes[-1]))
        self.encoder = nn.Sequential(*layers)
        encoded_size = self.channels[-1] * prod(self._sizes[-1])
        encoder_size = encoded_size * (history if conv_dim == 3 else 1)
        self.posterior = nn.Linear(encoder_size + parameter_dim, 2 * latent_dim)
        self.decoder_input = nn.Linear(latent_dim + parameter_dim, encoded_size)
        self.decoder = nn.ModuleList(
            nn.Conv2d(widths[i + 1], widths[i], 3, padding=1)
            for i in reversed(range(len(self.channels)))
        )

    def encode(
        self, images: pt.Tensor, parameters: Optional[pt.Tensor] = None
    ) -> tuple[pt.Tensor, pt.Tensor]:
        """Return Gaussian mean and bounded log variance (no random sampling)."""
        if images.ndim != 5 or tuple(images.shape[1:]) != (
            self.history,
            *self.image_shape,
        ):
            raise ValueError("images must have shape (batch, history, *image_shape)")
        _condition(parameters, images.shape[0], self.parameter_dim)
        inputs = images.transpose(1, 2) if self.conv_dim == 3 else images[:, 0]
        features = self.encoder(inputs).flatten(1)
        if parameters is not None:
            features = pt.cat((features, parameters), dim=1)
        mean, logvar = self.posterior(features).chunk(2, dim=1)
        return mean, logvar.clamp(-20.0, 10.0)

    def decode(
        self, latent: pt.Tensor, parameters: Optional[pt.Tensor] = None
    ) -> pt.Tensor:
        """Decode ``(batch, latent_dim)`` to ``(batch, *image_shape)``."""
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("latent must have shape (batch, latent_dim)")
        _condition(parameters, latent.shape[0], self.parameter_dim)
        features = latent if parameters is None else pt.cat((latent, parameters), dim=1)
        result = self.decoder_input(features).reshape(
            latent.shape[0], self.channels[-1], *self._sizes[-1]
        )
        for i, layer in enumerate(self.decoder):
            result = F.interpolate(result, size=self._sizes[-2 - i], mode="nearest")
            result = layer(result)
            if i + 1 < len(self.decoder):
                result = F.silu(result)
        return result

    def forward(
        self,
        images: pt.Tensor,
        parameters: Optional[pt.Tensor] = None,
        *,
        sample: Optional[bool] = None,
    ) -> tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        """Reconstruct, sampling during training and using the mean in eval mode."""
        mean, logvar = self.encode(images, parameters)
        stochastic = self.training if sample is None else sample
        latent = (
            mean + pt.randn_like(mean) * (0.5 * logvar).exp() if stochastic else mean
        )
        return self.decode(latent, parameters), mean, logvar


class ConvVAEROM(ROM):
    """Conditional VAE with free-running discrete latent evolution.

    :param autoencoder: configured image/history compressor
    :param dt: positive, fixed sample interval shared by all trajectories
    :param dynamics: optional :class:`AutoregressiveModel`, or legacy custom
        single-state module with signature ``(latent, parameters)``; defaults
        to :class:`LatentDynamics`

    Rollouts evolve only the latent state, without re-encoding predicted images.
    They use posterior means by default. Sampling the initial posterior is
    optional and does not provide calibrated predictive uncertainty. The image
    history must have ``history_length`` frames: encoder history plus predictor
    history minus one. Overlapping observed histories initialize the complete
    latent context; future predictions never use observed future frames.
    """

    @tracked
    def __init__(
        self,
        autoencoder: ConvVAE,
        *,
        dt: float = 1.0,
        dynamics: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if not isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        self.autoencoder = autoencoder
        self.dt = float(dt)
        self.dynamics = (
            dynamics
            if dynamics is not None
            else LatentDynamics(autoencoder.latent_dim, autoencoder.parameter_dim)
        )
        if isinstance(self.dynamics, AutoregressiveModel):
            if (
                self.dynamics.state_dim != autoencoder.latent_dim
                or self.dynamics.parameter_dim != autoencoder.parameter_dim
            ):
                raise ValueError(
                    "dynamics state/parameter dimensions must match autoencoder"
                )
        self.input_spec = InputSpec(
            initial_state=True, time=True, parameters=bool(autoencoder.parameter_dim)
        )

    def _effective_settings(self) -> dict:
        return {
            "dt": self.dt,
            "image_shape": self.autoencoder.image_shape,
            "history": self.autoencoder.history,
            "conv_dim": self.autoencoder.conv_dim,
            "latent_dim": self.autoencoder.latent_dim,
            "parameter_dim": self.autoencoder.parameter_dim,
            "channels": self.autoencoder.channels,
            "dynamics": type(self.dynamics).__qualname__,
            "latent_history_length": self._latent_history_length,
        }

    @property
    def history_length(self) -> int:
        """Required observed image count, combining encoder and predictor histories."""
        return self.autoencoder.history + self._latent_history_length - 1

    @property
    def _latent_history_length(self) -> int:
        return (
            self.dynamics.history_length
            if isinstance(self.dynamics, AutoregressiveModel)
            else 1
        )

    def _encode_context(
        self, images: pt.Tensor, parameters: Optional[pt.Tensor]
    ) -> tuple[pt.Tensor, pt.Tensor]:
        if images.ndim != 5 or tuple(images.shape[1:]) != (
            self.history_length,
            *self.autoencoder.image_shape,
        ):
            raise ValueError(
                f"history must have shape (batch, {self.history_length}, *image_shape)"
            )
        batch = images.shape[0]
        length = self._latent_history_length
        _condition(parameters, batch, self.autoencoder.parameter_dim)
        # Unfold produces B,K,C,H,W,encoder_history. Encode all overlapping
        # contexts as one batch, with gradients retained for joint training.
        windows = images.unfold(1, self.autoencoder.history, 1).movedim(-1, 2)
        windows = windows.reshape(
            batch * length, self.autoencoder.history, *self.autoencoder.image_shape
        )
        conditions = (
            None
            if parameters is None
            else parameters[:, None].expand(-1, length, -1).reshape(batch * length, -1)
        )
        mean, logvar = self.autoencoder.encode(windows, conditions)
        return mean.reshape(batch, length, -1), logvar.reshape(batch, length, -1)

    def _latent_rollout(
        self, context: pt.Tensor, steps: int, parameters: Optional[pt.Tensor]
    ) -> pt.Tensor:
        if isinstance(self.dynamics, AutoregressiveModel):
            return self.dynamics.rollout(context, steps, parameters)
        # Compatibility with existing custom single-state time-steppers.
        latent = context[:, -1]
        predictions = []
        for _ in range(steps):
            latent = self.dynamics(latent, parameters)
            predictions.append(latent)
        return pt.stack(predictions, dim=1)

    def _decode_sequence(
        self, latent: pt.Tensor, parameters: Optional[pt.Tensor]
    ) -> pt.Tensor:
        batch, steps, width = latent.shape
        conditions = (
            None
            if parameters is None
            else parameters[:, None].expand(-1, steps, -1).reshape(batch * steps, -1)
        )
        frames = self.autoencoder.decode(
            latent.reshape(batch * steps, width), conditions
        )
        return frames.reshape(batch, steps, *self.autoencoder.image_shape)

    def rollout(
        self,
        history: pt.Tensor,
        steps: int,
        parameters: Optional[pt.Tensor] = None,
        *,
        sample: bool = False,
    ) -> pt.Tensor:
        """Return future frames only, shaped ``(batch, steps, C, H, W)``.

        ``history`` needs ``self.history_length`` observed frames. With
        ``sample=True`` each initial latent context state is sampled once.
        This method retains gradients. Use ``torch.no_grad()`` for inference.
        """
        _positive(steps, "steps")
        latent, logvar = self._encode_context(history, parameters)
        if sample:
            latent = latent + pt.randn_like(latent) * (0.5 * logvar).exp()
        predicted = self._latent_rollout(latent, steps, parameters)
        return self._decode_sequence(predicted, parameters)

    def forward(self, initial_state: pt.Tensor, query: ROMQuery) -> Prediction:
        """Predict at relative times ``[0, dt, ..., n*dt]``.

        Time zero is the newest observed frame and is returned reconstructed.
        Output is batch-first, ``(batch, query_times, C, H, W)``. Forcing is
        unsupported; operating conditions belong in ``query.parameters``.
        """
        self.input_spec.validate(initial_state, query)
        if query.forcing is not None or query.forcing_history is not None:
            raise ValueError("forcing is unsupported; use fixed parameters")
        time = query.time
        if time is None or time.ndim != 1 or time.numel() < 1:
            raise ValueError("time must be a nonempty vector starting at zero")
        expected = (
            pt.arange(time.numel(), device=time.device, dtype=time.dtype) * self.dt
        )
        if (
            not time.is_floating_point()
            or not bool(pt.isfinite(time).all())
            or not pt.allclose(time, expected)
        ):
            raise ValueError("time must be relative [0, dt, ..., n*dt]")
        mean, _ = self._encode_context(initial_state, query.parameters)
        initial = self.autoencoder.decode(mean[:, -1], query.parameters).unsqueeze(1)
        if time.numel() == 1:
            return Prediction(initial)
        latent = self._latent_rollout(mean, time.numel() - 1, query.parameters)
        return Prediction(
            pt.cat((initial, self._decode_sequence(latent, query.parameters)), dim=1)
        )

    def loss(
        self,
        history: pt.Tensor,
        future: Optional[pt.Tensor] = None,
        parameters: Optional[pt.Tensor] = None,
        *,
        stage: str = "joint",
        beta: float = 1.0e-3,
        latent_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        mask: Optional[pt.Tensor] = None,
    ) -> dict[str, pt.Tensor]:
        """Compute reconstruction, KL, free-rollout and latent matching losses.

        Stages are ``autoencoder``, ``dynamics`` and ``joint``. Dynamics-only
        loss detaches encoder outputs but retains decoder derivatives with
        respect to latent inputs. ``fine_tune`` also freezes its parameters.
        KL is summed over latent coordinates and averaged over the batch;
        pixel and latent errors are means. ``mask`` is a finite nonnegative
        ``(C, H, W)`` weight tensor with positive sum. Masked pixels must still
        contain finite values; fill invalid camera pixels before encoding.
        """
        if stage not in ("autoencoder", "dynamics", "joint"):
            raise ValueError("stage must be autoencoder, dynamics or joint")
        for weight in (beta, latent_weight, reconstruction_weight):
            if not isfinite(weight) or weight < 0:
                raise ValueError("loss weights must be finite and non-negative")
        if mask is not None:
            if (
                tuple(mask.shape) != self.autoencoder.image_shape
                or not bool(pt.isfinite(mask).all())
                or bool((mask < 0).any())
                or not bool(mask.sum() > 0)
            ):
                raise ValueError(
                    "mask must match image_shape, be finite and nonnegative with positive sum"
                )

        def error(predicted: pt.Tensor, target: pt.Tensor) -> pt.Tensor:
            squared = (predicted - target).square()
            if mask is None:
                return squared.mean()
            return ((squared * mask).sum(dim=(-3, -2, -1)) / mask.sum()).mean()

        mean, logvar = self._encode_context(history, parameters)
        zero = mean.new_zeros(())
        reconstruction, kl, rollout, latent_loss = zero, zero, zero, zero
        if stage != "dynamics":
            latent = (
                mean + pt.randn_like(mean) * (0.5 * logvar).exp()
                if self.training
                else mean
            )
            reconstruction = error(
                self.autoencoder.decode(latent[:, -1], parameters), history[:, -1]
            )
            kl = 0.5 * (mean.square() + logvar.exp() - 1 - logvar).sum(dim=-1).mean()
        if stage != "autoencoder":
            if (
                future is None
                or future.ndim != 5
                or future.shape[0] != history.shape[0]
                or future.shape[1] < 1
                or tuple(future.shape[2:]) != self.autoencoder.image_shape
            ):
                raise ValueError("future must have shape (batch, steps, *image_shape)")
            latent = mean.detach() if stage == "dynamics" else mean
            observed = pt.cat((history, future), dim=1)
            predicted = self._latent_rollout(latent, future.shape[1], parameters)
            rollout = error(self._decode_sequence(predicted, parameters), future)
            if latent_weight:
                # Future observations are target-only: they never initialize
                # or update the free-running prediction context.
                with pt.no_grad():
                    windows = observed.unfold(1, self.autoencoder.history, 1).movedim(
                        -1, 2
                    )
                    windows = windows[:, self._latent_history_length :]
                    batch, steps = windows.shape[:2]
                    conditions = (
                        None
                        if parameters is None
                        else parameters[:, None]
                        .expand(-1, steps, -1)
                        .reshape(batch * steps, -1)
                    )
                    target, _ = self.autoencoder.encode(
                        windows.reshape(
                            batch * steps,
                            self.autoencoder.history,
                            *self.autoencoder.image_shape,
                        ),
                        conditions,
                    )
                    target = target.reshape(batch, steps, -1)
                latent_loss = F.mse_loss(predicted, target)

        total = (
            reconstruction_weight * reconstruction
            + beta * kl
            + rollout
            + latent_weight * latent_loss
        )
        return {
            "loss": total,
            "reconstruction": reconstruction,
            "kl": kl,
            "rollout": rollout,
            "latent": latent_loss,
        }

    @tracked
    def fine_tune(
        self,
        batches: Iterable[dict[str, pt.Tensor]],
        *,
        stage: str = "joint",
        epochs: int = 10,
        horizons: Sequence[int] = (1,),
        lr: float = 1.0e-3,
        beta: float = 1.0e-3,
        latent_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        mask: Optional[pt.Tensor] = None,
        gradient_clip: Optional[float] = 1.0,
        validation_batches: Optional[Iterable[dict[str, pt.Tensor]]] = None,
        patience: Optional[int] = None,
        min_delta: float = 0.0,
    ) -> list[dict[str, float]]:
        """Train with Adam and an explicit increasing rollout curriculum.

        ``epochs`` is the count *per horizon*. Batches must be re-iterable
        (e.g. a DataLoader) and contain enough future frames for every horizon.
        Call separately for compression, frozen-compressor dynamics training,
        and joint refinement. Optimizer state is retained across horizons in
        one call, reset between calls. Existing requires-grad flags and module
        modes are restored even on failure. Optional ``patience`` advances to
        the next horizon after that many epochs without a validation-objective
        improvement greater than ``min_delta``. It requires independent,
        re-iterable ``validation_batches``. The final iterate is retained; no
        best-checkpoint restoration or convergence guarantee is implied.
        """
        _positive(epochs, "epochs")
        horizons = tuple(horizons)
        if not horizons:
            raise ValueError("horizons must not be empty")
        for horizon in horizons:
            _positive(horizon, "horizon")
        if any(b < a for a, b in zip(horizons, horizons[1:])):
            raise ValueError("horizons must be nondecreasing")
        if stage not in ("autoencoder", "dynamics", "joint"):
            raise ValueError("stage must be autoencoder, dynamics or joint")
        if not isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive")
        if gradient_clip is not None and (
            not isfinite(gradient_clip) or gradient_clip <= 0
        ):
            raise ValueError("gradient_clip must be finite and positive")
        if iter(batches) is batches:
            raise ValueError("batches must be re-iterable, not a one-shot iterator")
        if (
            validation_batches is not None
            and iter(validation_batches) is validation_batches
        ):
            raise ValueError("validation_batches must be re-iterable")
        if patience is not None:
            _positive(patience, "patience")
            if validation_batches is None:
                raise ValueError("patience requires validation_batches")
        if not isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be finite and non-negative")
        flags = [(p, p.requires_grad) for p in self.parameters()]
        modes = [(module, module.training) for module in self.modules()]
        log = []
        try:
            self.train()
            if stage == "dynamics":
                self.autoencoder.requires_grad_(False)
                self.autoencoder.eval()
            elif stage == "autoencoder":
                self.dynamics.requires_grad_(False)
                self.dynamics.eval()
            trainable = [p for p in self.parameters() if p.requires_grad]
            optimizer = pt.optim.Adam(trainable, lr=lr)
            for horizon in horizons:
                best, stale = float("inf"), 0
                for epoch in range(epochs):
                    sums: dict[str, float] = {}
                    count = 0
                    for batch in batches:
                        future = batch.get("future")
                        if stage != "autoencoder":
                            if future is None or future.shape[1] < horizon:
                                raise ValueError("batch future is shorter than horizon")
                            future = future[:, :horizon]
                        optimizer.zero_grad(set_to_none=True)
                        losses = self.loss(
                            batch["history"],
                            future,
                            batch.get("parameters"),
                            stage=stage,
                            beta=beta,
                            latent_weight=latent_weight,
                            reconstruction_weight=reconstruction_weight,
                            mask=mask,
                        )
                        if not bool(pt.isfinite(losses["loss"])):
                            raise ValueError(
                                "nonfinite training loss; check data and normalization"
                            )
                        losses["loss"].backward()
                        if gradient_clip is not None:
                            nn.utils.clip_grad_norm_(
                                trainable, gradient_clip, error_if_nonfinite=True
                            )
                        optimizer.step()
                        size = batch["history"].shape[0]
                        count += size
                        for key, value in losses.items():
                            sums[key] = (
                                sums.get(key, 0.0) + float(value.detach()) * size
                            )
                    if count == 0:
                        raise ValueError("batches must not be empty")
                    record = {
                        "horizon": float(horizon),
                        "epoch": float(epoch),
                        **{key: value / count for key, value in sums.items()},
                    }
                    if validation_batches is not None:
                        active_modes = [(m, m.training) for m in self.modules()]
                        validation_sum, validation_count = 0.0, 0
                        self.eval()
                        try:
                            with pt.no_grad():
                                for batch in validation_batches:
                                    future = batch.get("future")
                                    if stage != "autoencoder":
                                        if future is None or future.shape[1] < horizon:
                                            raise ValueError(
                                                "validation future is shorter than horizon"
                                            )
                                        future = future[:, :horizon]
                                    objective = self.loss(
                                        batch["history"],
                                        future,
                                        batch.get("parameters"),
                                        stage=stage,
                                        beta=beta,
                                        latent_weight=latent_weight,
                                        reconstruction_weight=reconstruction_weight,
                                        mask=mask,
                                    )["loss"]
                                    if not bool(pt.isfinite(objective)):
                                        raise ValueError("nonfinite validation loss")
                                    size = batch["history"].shape[0]
                                    validation_sum += float(objective) * size
                                    validation_count += size
                        finally:
                            for module, mode in active_modes:
                                module.training = mode
                        if validation_count == 0:
                            raise ValueError("validation_batches must not be empty")
                        record["validation_loss"] = validation_sum / validation_count
                        if record["validation_loss"] < best - min_delta:
                            best, stale = record["validation_loss"], 0
                        else:
                            stale += 1
                    log.append(record)
                    if patience is not None and stale >= patience:
                        break
        finally:
            for parameter, flag in flags:
                parameter.requires_grad_(flag)
            for module, mode in modes:
                module.training = mode
        return log
