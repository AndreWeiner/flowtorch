"""Reusable, fixed-context predictors for uniformly sampled vector time series.

These modules do not depend on an encoder, decoder, or spatial discretization.
All histories are batch-first and ordered oldest to newest. Operating conditions
are fixed per batch entry and are never advanced as part of the predicted state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import isfinite
from numbers import Real
from typing import Optional, Sequence

import torch as pt
from torch import nn
from torch.nn import functional as F


def _positive(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _condition(parameters: Optional[pt.Tensor], batch: int, size: int) -> None:
    if size == 0:
        if parameters is not None:
            raise ValueError("this model is unconditional; omit parameters")
    elif parameters is None or parameters.shape != (batch, size):
        raise ValueError(f"parameters must have shape ({batch}, {size})")
    elif not bool(pt.isfinite(parameters).all()):
        raise ValueError("parameters must be finite")


class AutoregressiveModel(nn.Module, ABC):
    """History-to-next-state interface, independent of a particular ROM.

    :param state_dim: number of predicted variables
    :param history_length: fixed number of observed/predicted states per window
    :param parameter_dim: fixed condition-vector size; zero means unconditional
    :param residual: predict an increment added to the latest input state

    ``forward`` maps ``(batch, history_length, state_dim)`` to
    ``(batch, state_dim)``. ``rollout`` shifts this window and appends each
    prediction, retaining all temporal gradients. There is no implicit mutable
    recurrent state or persistent cache between calls or between trajectories.
    """

    def __init__(
        self,
        state_dim: int,
        *,
        history_length: int = 1,
        parameter_dim: int = 0,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.state_dim = _positive(state_dim, "state_dim")
        self.history_length = _positive(history_length, "history_length")
        if type(parameter_dim) is not int or parameter_dim < 0:
            raise ValueError("parameter_dim must be a non-negative integer")
        self.parameter_dim = parameter_dim
        self.residual = bool(residual)

    def _inputs(
        self, history: pt.Tensor, parameters: Optional[pt.Tensor]
    ) -> tuple[pt.Tensor, pt.Tensor]:
        # Retain the original single-state LatentDynamics call convention.
        if history.ndim == 2 and self.history_length == 1:
            history = history.unsqueeze(1)
        if (
            history.ndim != 3
            or history.shape[0] < 1
            or tuple(history.shape[1:]) != (self.history_length, self.state_dim)
        ):
            raise ValueError(
                f"history must have shape (batch, {self.history_length}, {self.state_dim})"
            )
        _condition(parameters, history.shape[0], self.parameter_dim)
        inputs = history
        if parameters is not None:
            conditions = parameters.unsqueeze(1).expand(-1, self.history_length, -1)
            inputs = pt.cat((history, conditions), dim=-1)
        return history, inputs

    def _readout(self, prediction: pt.Tensor, history: pt.Tensor) -> pt.Tensor:
        return history[:, -1] + prediction if self.residual else prediction

    @abstractmethod
    def forward(
        self, history: pt.Tensor, parameters: Optional[pt.Tensor] = None
    ) -> pt.Tensor:
        """Predict exactly one next state from the full context window."""

    def rollout(
        self,
        history: pt.Tensor,
        steps: int,
        parameters: Optional[pt.Tensor] = None,
    ) -> pt.Tensor:
        """Return future-only predictions ``(batch, steps, state_dim)``.

        This is free-running fixed-window forecasting, without teacher forcing.
        Use eval mode and ``torch.no_grad()`` for deterministic inference without
        autograd storage. Training graphs grow with the requested horizon.
        """
        _positive(steps, "steps")
        window, _ = self._inputs(history, parameters)
        predictions = []
        for _ in range(steps):
            predicted = self(window, parameters)
            predictions.append(predicted)
            window = pt.cat((window[:, 1:], predicted.unsqueeze(1)), dim=1)
        return pt.stack(predictions, dim=1)

    def rollout_loss(
        self,
        history: pt.Tensor,
        future: pt.Tensor,
        parameters: Optional[pt.Tensor] = None,
    ) -> pt.Tensor:
        """Mean squared free-running error over batch, time and variables.

        ``future`` must have shape ``(batch, steps, state_dim)``. This method
        neither constructs an optimizer nor changes training mode.
        """
        if (
            future.ndim != 3
            or future.shape[0] != history.shape[0]
            or future.shape[1] < 1
            or future.shape[2] != self.state_dim
        ):
            raise ValueError("future must have shape (batch, steps, state_dim)")
        return F.mse_loss(self.rollout(history, future.shape[1], parameters), future)


class MLPAutoregressor(AutoregressiveModel):
    """Residual MLP baseline, by default with one 64-unit SiLU hidden layer.

    For longer contexts all state/condition tokens are flattened into one input.
    """

    def __init__(
        self,
        state_dim: int,
        parameter_dim: int = 0,
        hidden_dim: int = 64,
        *,
        history_length: int = 1,
        residual: bool = True,
    ) -> None:
        super().__init__(
            state_dim,
            history_length=history_length,
            parameter_dim=parameter_dim,
            residual=residual,
        )
        _positive(hidden_dim, "hidden_dim")
        self.network = nn.Sequential(
            nn.Linear(history_length * (state_dim + parameter_dim), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(
        self, history: pt.Tensor, parameters: Optional[pt.Tensor] = None
    ) -> pt.Tensor:
        """Predict from a fixed context or a single ``(batch, state_dim)`` state."""
        history, inputs = self._inputs(history, parameters)
        return self._readout(self.network(inputs.flatten(1)), history)


class LatentDynamics(MLPAutoregressor):
    """Compatibility name for the original single-state residual MLP.

    Constructor arguments and ``network`` state-dict keys remain unchanged.
    """

    def __init__(
        self, latent_dim: int, parameter_dim: int = 0, hidden_dim: int = 64
    ) -> None:
        super().__init__(latent_dim, parameter_dim, hidden_dim)


class _RecurrentAutoregressor(AutoregressiveModel):
    cell_type = nn.LSTM

    def __init__(
        self,
        state_dim: int,
        *,
        history_length: int = 8,
        parameter_dim: int = 0,
        hidden_dim: int = 64,
        num_layers: int = 1,
        residual: bool = True,
    ) -> None:
        super().__init__(
            state_dim,
            history_length=history_length,
            parameter_dim=parameter_dim,
            residual=residual,
        )
        _positive(hidden_dim, "hidden_dim")
        _positive(num_layers, "num_layers")
        self.recurrent = self.cell_type(
            state_dim + parameter_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, state_dim)

    def forward(
        self, history: pt.Tensor, parameters: Optional[pt.Tensor] = None
    ) -> pt.Tensor:
        """Start from zero hidden state and read the last token of the window."""
        history, inputs = self._inputs(history, parameters)
        outputs, _ = self.recurrent(inputs)
        return self._readout(self.output(outputs[:, -1]), history)


class LSTMAutoregressor(_RecurrentAutoregressor):
    """Unidirectional LSTM over a fixed context, reset for every forecast window.

    Hidden and cell states are local to each forward call. Every rollout step
    recomputes the shifted context from zero state, matching windowed training.
    This deliberately bounded-memory interface does not offer stateful streaming.
    """

    cell_type = nn.LSTM


class GRUAutoregressor(_RecurrentAutoregressor):
    """Unidirectional GRU with the same fixed-window semantics as the LSTM."""

    cell_type = nn.GRU


class _EasyAttention(nn.Module):
    """Dense multi-head easy attention: concatenate ``alpha_h X Wv_h``.

    Sanchis-Agudo et al., arXiv:2308.12874, equations (15) and (20).
    No query/key projections, softmax, or output projection are used.
    """

    def __init__(self, history_length: int, model_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.value = nn.Linear(model_dim, model_dim, bias=False)
        self.alpha = nn.Parameter(
            pt.eye(history_length).repeat(num_heads, 1, 1)
            + 0.02 * pt.randn(num_heads, history_length, history_length)
        )

    def forward(self, inputs: pt.Tensor) -> pt.Tensor:
        batch, length, width = inputs.shape
        values = self.value(inputs).reshape(batch, length, self.num_heads, -1)
        mixed = pt.einsum("hij,bjhd->bihd", self.alpha, values)
        return mixed.reshape(batch, length, width)


class _EasyAttentionBlock(nn.Module):
    def __init__(
        self, history_length: int, model_dim: int, num_heads: int, feedforward_dim: int
    ) -> None:
        super().__init__()
        self.attention = _EasyAttention(history_length, model_dim, num_heads)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, model_dim),
        )

    def forward(self, inputs: pt.Tensor) -> pt.Tensor:
        inputs = inputs + self.attention(self.norm1(inputs))
        return inputs + self.feedforward(self.norm2(inputs))


class EasyAttentionAutoregressor(AutoregressiveModel):
    """Transformer-style predictor using dense, input-independent easy attention.

    :param history_length: fixed context; attention weights depend on its size
    :param model_dim: token embedding size, divisible by ``num_heads``
    :param num_heads: independent learned temporal mixing matrices
    :param num_layers: number of pre-normalized attention/feedforward blocks
    :param feedforward_dim: hidden feedforward width (default twice model_dim)

    The attention mechanism follows Sanchis-Agudo et al.,
    https://arxiv.org/abs/2308.12874. The enclosing small predictor (pre-norm,
    last-token readout, residual state prediction) is a FlowTorch design,
    not a reproduction of the paper's complete experimental architecture.
    All context tokens are past observations, so dense mixing needs no causal
    mask. Only the next state is supervised; intermediate token outputs are
    not predictions at their corresponding times. Learned temporal matrices
    encode ordering, so no positional embedding is added.
    """

    def __init__(
        self,
        state_dim: int,
        *,
        history_length: int = 8,
        parameter_dim: int = 0,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        feedforward_dim: Optional[int] = None,
        residual: bool = True,
    ) -> None:
        super().__init__(
            state_dim,
            history_length=history_length,
            parameter_dim=parameter_dim,
            residual=residual,
        )
        _positive(model_dim, "model_dim")
        _positive(num_heads, "num_heads")
        _positive(num_layers, "num_layers")
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        feedforward_dim = 2 * model_dim if feedforward_dim is None else feedforward_dim
        _positive(feedforward_dim, "feedforward_dim")
        self.input = nn.Linear(state_dim + parameter_dim, model_dim)
        self.blocks = nn.Sequential(
            *[
                _EasyAttentionBlock(
                    history_length, model_dim, num_heads, feedforward_dim
                )
                for _ in range(num_layers)
            ]
        )
        self.output = nn.Linear(model_dim, state_dim)

    def forward(
        self, history: pt.Tensor, parameters: Optional[pt.Tensor] = None
    ) -> pt.Tensor:
        """Predict the next state from the last mixed token."""
        history, inputs = self._inputs(history, parameters)
        features = self.blocks(self.input(inputs))
        return self._readout(self.output(features[:, -1]), history)


def _temporal_activation(name: str) -> nn.Module:
    """Construct a fresh pointwise activation, preserving temporal causality."""
    activations = {
        "silu": nn.SiLU,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
        "identity": nn.Identity,
    }
    if not isinstance(name, str) or name not in activations:
        raise ValueError("activation must be one of " + ", ".join(activations))
    return activations[name]()


def _temporal_convolution(
    input_dim: int,
    output_dim: int,
    kernel_size: int,
    dilation: int,
    weight_normalization: bool,
) -> nn.Module:
    convolution = nn.Conv1d(input_dim, output_dim, kernel_size, dilation=dilation)
    if weight_normalization:
        try:
            from torch.nn.utils.parametrizations import weight_norm
        except ImportError:
            # Keep optional weight normalization available on supported older
            # torch releases. Prefer parametrizations whenever available.
            from torch.nn.utils import weight_norm
        convolution = weight_norm(convolution, dim=0)
    return convolution


class _CausalConvBlock(nn.Module):
    """Configurable left-padded convolutions with a pointwise residual shortcut."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        dilation: int,
        *,
        convs_per_block: int = 1,
        activation: str = "silu",
        dropout: float = 0.0,
        weight_normalization: bool = False,
        activate_residual: bool = False,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        # Retain the first convolution's original state-dict name.
        self.convolution = _temporal_convolution(
            input_dim, output_dim, kernel_size, dilation, weight_normalization
        )
        self.additional_convolutions = nn.ModuleList(
            [
                _temporal_convolution(
                    output_dim, output_dim, kernel_size, dilation, weight_normalization
                )
                for _ in range(convs_per_block - 1)
            ]
        )
        self.shortcut = (
            nn.Identity()
            if input_dim == output_dim
            else nn.Conv1d(input_dim, output_dim, 1)
        )
        self.activations = nn.ModuleList(
            [_temporal_activation(activation) for _ in range(convs_per_block)]
        )
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(convs_per_block)]
        )
        self.output_activation = (
            _temporal_activation(activation) if activate_residual else nn.Identity()
        )

    def forward(self, inputs: pt.Tensor) -> pt.Tensor:
        result = self.dropouts[0](
            self.activations[0](self.convolution(F.pad(inputs, (self.left_padding, 0))))
        )
        for index, convolution in enumerate(self.additional_convolutions, start=1):
            result = self.dropouts[index](
                self.activations[index](
                    convolution(F.pad(result, (self.left_padding, 0)))
                )
            )
        return self.output_activation(result + self.shortcut(inputs))


class CausalTemporalConv(nn.Module):
    """Reusable causal feature extractor for batch-first vector sequences.

    :param input_dim: number of input variables per time sample
    :param channels: output widths of successive residual convolution blocks
    :param kernel_size: temporal kernel size, shared across blocks
    :param dilations: one positive dilation per block; defaults to 1, 2, 4, ...

    Maps ``(batch, time, input_dim)`` to ``(batch, time, channels[-1])``.
    By default each block has one dilated convolution, SiLU activation, and an
    identity or pointwise-convolution shortcut. ``convs_per_block`` repeats the
    convolution/activation/dropout sequence at the same dilation. Optional
    weight normalization applies per output filter to temporal convolutions,
    not residual projections. Padding is zero-valued and left-only;
    no normalization or pooling mixes time samples. Output t depends only on
    input samples at or before t. The receptive field is
    ``1 + convs_per_block * (kernel_size - 1) * sum(dilations)``.
    ``activation`` accepts silu, relu, gelu, tanh, elu, leaky_relu or identity.
    ``dropout`` is elementwise after each activation and disabled in eval mode.
    ``activate_residual`` optionally applies activation after the residual sum.
    Weight normalization uses PyTorch parametrizations when available, with the
    legacy hook implementation on older torch releases. This component extracts features;
    it does not itself predict the next state or maintain a streaming cache.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        channels: Sequence[int] = (16, 16, 16),
        kernel_size: int = 3,
        dilations: Optional[Sequence[int]] = None,
        convs_per_block: int = 1,
        activation: str = "silu",
        dropout: float = 0.0,
        weight_normalization: bool = False,
        activate_residual: bool = False,
    ) -> None:
        super().__init__()
        self.convs_per_block = _positive(convs_per_block, "convs_per_block")
        _temporal_activation(activation)
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, Real)
            or not isfinite(dropout)
            or not 0.0 <= dropout <= 1.0
        ):
            raise ValueError("dropout must be finite and between zero and one")
        if (
            type(weight_normalization) is not bool
            or type(activate_residual) is not bool
        ):
            raise ValueError(
                "weight_normalization and activate_residual must be booleans"
            )
        self.activation = activation
        self.dropout = float(dropout)
        self.weight_normalization = weight_normalization
        self.activate_residual = activate_residual
        self.input_dim = _positive(input_dim, "input_dim")
        self.channels = tuple(channels)
        if not self.channels:
            raise ValueError("channels must not be empty")
        for width in self.channels:
            _positive(width, "channels entries")
        self.kernel_size = _positive(kernel_size, "kernel_size")
        self.dilations = (
            tuple(2**i for i in range(len(self.channels)))
            if dilations is None
            else tuple(dilations)
        )
        if len(self.dilations) != len(self.channels):
            raise ValueError("dilations must have one entry per channel block")
        for dilation in self.dilations:
            _positive(dilation, "dilations entries")
        self.receptive_field = 1 + convs_per_block * (kernel_size - 1) * sum(
            self.dilations
        )
        widths = (input_dim, *self.channels)
        self.blocks = nn.Sequential(
            *[
                _CausalConvBlock(
                    widths[i],
                    widths[i + 1],
                    kernel_size,
                    dilation,
                    convs_per_block=convs_per_block,
                    activation=activation,
                    dropout=float(dropout),
                    weight_normalization=weight_normalization,
                    activate_residual=activate_residual,
                )
                for i, dilation in enumerate(self.dilations)
            ]
        )

    def forward(self, history: pt.Tensor) -> pt.Tensor:
        """Extract a feature vector at each observed time, preserving sequence length."""
        if (
            history.ndim != 3
            or min(history.shape[:2]) < 1
            or history.shape[2] != self.input_dim
        ):
            raise ValueError(f"history must have shape (batch, time, {self.input_dim})")
        return self.blocks(history.transpose(1, 2)).transpose(1, 2)


class TCNMLPAutoregressor(AutoregressiveModel):
    """Causal temporal convolution features followed by an MLP next-state readout.

    :param history_length: number of consecutive past states in each forecast
    :param conv_channels: one feature width per causal convolution block
    :param hidden_dim: MLP hidden width; ``None`` selects a single linear readout
    :param kernel_size: temporal kernel size in each convolution block
    :param dilations: optional explicit dilations, otherwise successive powers of two

    Convolution depth, activation, dropout, weight normalization and optional
    post-residual activation are forwarded to :class:`CausalTemporalConv`.
    ``head_activation`` overrides the convolution activation for an MLP head.
    The last temporal feature feeds the configurable readout. Conditions are
    concatenated to each observed state before feature extraction. The default
    four convolution blocks have receptive field 31, covering the default
    16-state context. Contexts larger than the receptive field are rejected
    rather than silently ignoring their oldest samples. Left zero padding is
    recomputed for each window, consistently in training and free rollout.
    """

    def __init__(
        self,
        state_dim: int,
        *,
        history_length: int = 16,
        parameter_dim: int = 0,
        conv_channels: Sequence[int] = (16, 16, 16, 16),
        hidden_dim: Optional[int] = 64,
        kernel_size: int = 3,
        dilations: Optional[Sequence[int]] = None,
        convs_per_block: int = 1,
        activation: str = "silu",
        dropout: float = 0.0,
        weight_normalization: bool = False,
        activate_residual: bool = False,
        head_activation: Optional[str] = None,
        residual: bool = True,
    ) -> None:
        super().__init__(
            state_dim,
            history_length=history_length,
            parameter_dim=parameter_dim,
            residual=residual,
        )
        if hidden_dim is not None:
            _positive(hidden_dim, "hidden_dim")
        self.temporal = CausalTemporalConv(
            state_dim + parameter_dim,
            channels=conv_channels,
            kernel_size=kernel_size,
            dilations=dilations,
            convs_per_block=convs_per_block,
            activation=activation,
            dropout=dropout,
            weight_normalization=weight_normalization,
            activate_residual=activate_residual,
        )
        if history_length > self.temporal.receptive_field:
            raise ValueError(
                "history_length exceeds the temporal receptive field; increase kernel size, dilations or block count"
            )
        activation_name = activation if head_activation is None else head_activation
        _temporal_activation(activation_name)
        self.network = (
            nn.Sequential(nn.Linear(self.temporal.channels[-1], state_dim))
            if hidden_dim is None
            else nn.Sequential(
                nn.Linear(self.temporal.channels[-1], hidden_dim),
                _temporal_activation(activation_name),
                nn.Linear(hidden_dim, state_dim),
            )
        )

    def forward(
        self, history: pt.Tensor, parameters: Optional[pt.Tensor] = None
    ) -> pt.Tensor:
        """Predict one state from the latest causally extracted history feature."""
        history, inputs = self._inputs(history, parameters)
        features = self.temporal(inputs)[:, -1]
        return self._readout(self.network(features), history)
