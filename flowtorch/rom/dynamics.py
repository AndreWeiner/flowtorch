"""Differentiable continuous linear dynamics in spectral coordinates."""

from __future__ import annotations

from math import pi
from typing import Optional

import torch as pt

_FFT_CONVOLUTION_THRESHOLD = 64


def _closest_proper_orthogonal(cross_covariance: pt.Tensor) -> pt.Tensor:
    """Return the closest orientation-preserving orthogonal matrix."""
    left, _, right_h = pt.linalg.svd(cross_covariance, full_matrices=False)
    if pt.is_complex(cross_covariance):
        return left @ right_h
    correction = pt.ones(left.shape[1], dtype=left.dtype, device=left.device)
    if pt.linalg.det(left @ right_h).real < 0:
        correction[-1] = -1
    return (left * correction) @ right_h


class ContinuousOperator(pt.nn.Module):
    """Diagonalizable continuous operator with an exactly-real real-data path."""

    def __init__(
        self,
        discrete_operator: pt.Tensor,
        dt: float,
        stationary: bool,
        *,
        zero_growth: bool = False,
    ):
        super().__init__()
        if (
            discrete_operator.ndim != 2
            or discrete_operator.shape[0] != discrete_operator.shape[1]
        ):
            raise ValueError("discrete_operator must be square")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.stationary = bool(stationary)
        self.zero_growth = bool(zero_growth or stationary)
        self.dt = float(dt)
        self._complex = pt.is_complex(discrete_operator)
        if self._complex:
            self._initialize_complex(discrete_operator)
        else:
            self._initialize_real(discrete_operator)

    def _initialize_complex(self, operator: pt.Tensor) -> None:
        values, vectors = pt.linalg.eig(operator)
        order = pt.argsort(pt.angle(values))
        values = values[order]
        vectors = vectors[:, order]
        floor = pt.finfo(operator.real.dtype).eps * operator.norm().clamp_min(1.0)
        magnitude = values.abs().clamp_min(floor)
        growth = magnitude.log() / self.dt
        if self.zero_growth:
            growth = pt.zeros_like(growth)
        frequency = pt.angle(values) / (2.0 * pi * self.dt)
        self.basis = pt.nn.Parameter(vectors)
        self.frequency = pt.nn.Parameter(frequency.real)
        if self.stationary:
            self.register_buffer("growth", pt.zeros_like(growth.real))
        else:
            self.growth = pt.nn.Parameter(growth.real)
        self.n_pairs = 0
        self.n_real = values.numel()

    def _initialize_real(self, operator: pt.Tensor) -> None:
        complex_dtype = pt.complex128 if operator.dtype == pt.float64 else pt.complex64
        values, vectors = pt.linalg.eig(operator.to(complex_dtype))
        tolerance = (
            100.0 * pt.finfo(operator.dtype).eps * values.abs().max().clamp_min(1.0)
        )
        positive = [
            index
            for index, value in enumerate(values)
            if float(value.imag) > float(tolerance)
        ]
        positive.sort(key=lambda index: float(pt.angle(values[index])))
        real_indices = [
            index
            for index, value in enumerate(values)
            if abs(float(value.imag)) <= float(tolerance)
        ]
        negative = [
            index
            for index in real_indices
            if float(values[index].real) < -float(tolerance)
        ]
        real = [index for index in real_indices if index not in negative]
        columns = []
        pair_growth = []
        pair_frequency = []
        floor = pt.finfo(operator.dtype).eps * operator.norm().clamp_min(1.0)
        for index in positive:
            value = values[index]
            vector = vectors[:, index]
            columns.extend((vector.real, -vector.imag))
            pair_growth.append(value.abs().clamp_min(floor).log() / self.dt)
            pair_frequency.append(pt.angle(value) / (2.0 * pi * self.dt))
        while len(negative) >= 2:
            first = negative.pop(0)
            second = negative.pop(0)
            columns.extend((vectors[:, first].real, vectors[:, second].real))
            magnitude = 0.5 * (values[first].abs() + values[second].abs())
            pair_growth.append(magnitude.clamp_min(floor).log() / self.dt)
            pair_frequency.append(operator.new_tensor(0.5 / self.dt))
        real.extend(negative)
        for index in real:
            columns.append(vectors[:, index].real)

        size = operator.shape[0]
        if len(columns) != size:
            raise RuntimeError("could not construct a complete real spectral basis")
        basis = pt.stack(columns, dim=1).to(operator.dtype)
        if int(pt.linalg.matrix_rank(basis).item()) != size:
            # A defective eigensystem has no usable spectral representation.
            # Starting from an identity basis and the real operator remains a
            # valid, differentiable fallback; its Schur-like upper coupling is
            # deliberately not trainable in this first spectral model.
            raise ValueError("the initialized operator is not diagonalizable")
        self.basis = pt.nn.Parameter(basis)
        pair_growth_tensor = (
            pt.stack(pair_growth).to(operator.dtype)
            if pair_growth
            else operator.new_empty(0)
        )
        pair_frequency_tensor = (
            pt.stack(pair_frequency).to(operator.dtype)
            if pair_frequency
            else operator.new_empty(0)
        )
        real_values = values[real].real.to(operator.dtype)
        real_growth = real_values.abs().clamp_min(floor).log() / self.dt
        if self.zero_growth:
            pair_growth_tensor = pt.zeros_like(pair_growth_tensor)
            real_growth = pt.zeros_like(real_growth)
        self.frequency = pt.nn.Parameter(pair_frequency_tensor)
        self.n_pairs = len(pair_growth)
        self.n_real = len(real)
        if self.stationary:
            self.register_buffer("pair_growth", pt.zeros_like(pair_growth_tensor))
            self.register_buffer("real_growth", pt.zeros_like(real_growth))
        else:
            self.pair_growth = pt.nn.Parameter(pair_growth_tensor)
            self.real_growth = pt.nn.Parameter(real_growth)

    @property
    def latent_size(self) -> int:
        return self.basis.shape[0]

    @property
    def block_operator(self) -> pt.Tensor:
        if self._complex:
            eigenvalues = self.growth.to(self.basis.dtype) + (
                2.0j * pi * self.frequency.to(self.basis.dtype)
            )
            return pt.diag(eigenvalues)
        blocks = []
        for index in range(self.n_pairs):
            growth = self.pair_growth[index]
            omega = 2.0 * pi * self.frequency[index]
            blocks.append(
                pt.stack((pt.stack((growth, -omega)), pt.stack((omega, growth))))
            )
        for index in range(self.n_real):
            blocks.append(self.real_growth[index].reshape(1, 1))
        if not blocks:
            return self.basis.new_zeros((0, 0))
        return pt.block_diag(*blocks)

    @property
    def operator(self) -> pt.Tensor:
        """Continuous generator in POD/embedded coordinates."""
        right = pt.linalg.solve(self.basis.mT, self.block_operator.mT).mT
        return self.basis @ right

    @property
    def growth_rate(self) -> pt.Tensor:
        if self._complex:
            return self.growth
        values = []
        for growth in self.pair_growth:
            values.extend((growth, growth))
        values.extend(self.real_growth)
        return pt.stack(values) if values else self.basis.new_empty(0)

    @property
    def frequencies(self) -> pt.Tensor:
        if self._complex:
            return self.frequency
        values = []
        for frequency in self.frequency:
            values.extend((frequency, -frequency))
        values.extend(self.frequency.new_zeros(self.n_real))
        return pt.stack(values) if values else self.frequency.new_empty(0)

    @staticmethod
    def _exponential_integral(value: pt.Tensor, time: pt.Tensor) -> pt.Tensor:
        """Evaluate ``expm1(value*time)/value`` with its zero limit."""
        tolerance = 10.0 * pt.finfo(value.real.dtype).eps
        small = value.abs() <= tolerance
        safe = pt.where(small, pt.ones_like(value), value)
        result = pt.expm1(value * time) / safe
        return pt.where(small, time.to(result.dtype), result)

    def transition_factors(self, dt: pt.Tensor | float) -> tuple[pt.Tensor, pt.Tensor]:
        """Return vectorized spectral transitions and hold responses.

        A scalar time returns two ``(rank, rank)`` matrices. A tensor of times
        returns matrices with the time shape as leading dimensions.
        """
        time = pt.as_tensor(dt, dtype=self.basis.real.dtype, device=self.basis.device)
        shape = (*time.shape, self.latent_size, self.latent_size)
        transition = self.basis.new_zeros(shape)
        response = self.basis.new_zeros(shape)
        if self._complex:
            values = self.growth.to(self.basis.dtype) + (
                2.0j * pi * self.frequency.to(self.basis.dtype)
            )
            diagonal = pt.exp(time.unsqueeze(-1) * values)
            integral = self._exponential_integral(
                values, time.unsqueeze(-1).to(values.dtype)
            )
            indices = pt.arange(self.latent_size, device=self.basis.device)
            transition[..., indices, indices] = diagonal
            response[..., indices, indices] = integral
            return transition, response

        cursor = 0
        for index in range(self.n_pairs):
            growth = self.pair_growth[index]
            omega = 2.0 * pi * self.frequency[index]
            exponential = pt.exp(growth * time)
            cosine = pt.cos(omega * time)
            sine = pt.sin(omega * time)
            transition[..., cursor, cursor] = exponential * cosine
            transition[..., cursor, cursor + 1] = -exponential * sine
            transition[..., cursor + 1, cursor] = exponential * sine
            transition[..., cursor + 1, cursor + 1] = exponential * cosine
            value = pt.complex(growth, omega)
            integral = self._exponential_integral(value, time.to(value.dtype))
            response[..., cursor, cursor] = integral.real
            response[..., cursor, cursor + 1] = -integral.imag
            response[..., cursor + 1, cursor] = integral.imag
            response[..., cursor + 1, cursor + 1] = integral.real
            cursor += 2
        for index in range(self.n_real):
            growth = self.real_growth[index]
            transition[..., cursor, cursor] = pt.exp(growth * time)
            response[..., cursor, cursor] = self._exponential_integral(growth, time)
            cursor += 1
        return transition, response

    def propagate(self, initial_state: pt.Tensor, offsets: pt.Tensor) -> pt.Tensor:
        """Propagate one or a batch of latent states to every time offset."""
        if (
            initial_state.ndim not in (1, 2)
            or initial_state.shape[0] != self.latent_size
        ):
            raise ValueError("initial latent state has an incompatible shape")
        if offsets.ndim != 1:
            raise ValueError("time offsets must be one-dimensional")
        single = initial_state.ndim == 1
        initial = initial_state.unsqueeze(-1) if single else initial_state
        coordinates = pt.linalg.solve(self.basis, initial.to(self.basis.dtype))
        if self._complex:
            values = self.growth.to(self.basis.dtype) + (
                2.0j * pi * self.frequency.to(self.basis.dtype)
            )
            factors = pt.exp(values.unsqueeze(-1) * offsets.to(values.dtype))
            modal = coordinates.unsqueeze(-1) * factors.unsqueeze(1)
        else:
            pieces = []
            if self.n_pairs:
                pair = coordinates[: 2 * self.n_pairs].reshape(
                    self.n_pairs, 2, coordinates.shape[1]
                )
                time = offsets.reshape(1, 1, 1, -1)
                growth = self.pair_growth.reshape(-1, 1, 1, 1)
                omega = (2.0 * pi * self.frequency).reshape(-1, 1, 1, 1)
                exponential = pt.exp(growth * time)
                cosine = pt.cos(omega * time)
                sine = pt.sin(omega * time)
                first = exponential * (
                    cosine * pair[:, :1].unsqueeze(-1)
                    - sine * pair[:, 1:].unsqueeze(-1)
                )
                second = exponential * (
                    sine * pair[:, :1].unsqueeze(-1)
                    + cosine * pair[:, 1:].unsqueeze(-1)
                )
                pieces.append(pt.cat((first, second), dim=1).flatten(0, 1))
            if self.n_real:
                real = coordinates[2 * self.n_pairs :]
                factors = pt.exp(self.real_growth.unsqueeze(-1) * offsets.unsqueeze(0))
                pieces.append(real.unsqueeze(-1) * factors.unsqueeze(1))
            modal = pt.cat(pieces, dim=0)
        flattened = modal.flatten(1, 2)
        result = (self.basis @ flattened).reshape(
            self.latent_size, initial.shape[1], offsets.numel()
        )
        if single:
            result = result[:, 0]
        return result if self._complex else result.to(initial_state.dtype)

    @staticmethod
    def _controlled_modes(
        initial: pt.Tensor,
        forcing: pt.Tensor,
        values: pt.Tensor,
        time: pt.Tensor,
        uniform_time: bool,
    ) -> pt.Tensor:
        """Evolve independent scalar modes under zero-order-held forcing."""
        offsets = time - time[0]
        factors = pt.exp(values.unsqueeze(-1) * offsets.to(values.dtype))
        homogeneous = initial.unsqueeze(-1) * factors.unsqueeze(1)
        if time.numel() == 1:
            return homogeneous
        delta = time.diff().to(values.dtype)
        if uniform_time and delta.numel() >= _FFT_CONVOLUTION_THRESHOLD:
            response = ContinuousOperator._exponential_integral(values, delta[0])
            intervals = delta.numel()
            powers = pt.arange(intervals, dtype=time.real.dtype, device=time.device)
            kernel = pt.exp(values.unsqueeze(-1) * delta[0] * powers)
            signal = response[:, None, None] * forcing
            transform_size = 1 << (2 * intervals - 1).bit_length()
            if pt.is_complex(signal):
                convolution = pt.fft.ifft(
                    pt.fft.fft(signal, n=transform_size)
                    * pt.fft.fft(kernel.unsqueeze(1), n=transform_size),
                    n=transform_size,
                )[..., :intervals]
            else:
                convolution = pt.fft.irfft(
                    pt.fft.rfft(signal, n=transform_size)
                    * pt.fft.rfft(kernel.unsqueeze(1), n=transform_size),
                    n=transform_size,
                )[..., :intervals]
            forced = pt.cat(
                (initial.new_zeros((*initial.shape, 1)), convolution), dim=-1
            )
            return homogeneous + forced
        response = ContinuousOperator._exponential_integral(
            values.unsqueeze(-1), delta.unsqueeze(0)
        )
        injection = response.unsqueeze(1) * forcing
        mask = pt.tril(
            pt.ones(
                (time.numel(), time.numel() - 1),
                dtype=pt.bool,
                device=time.device,
            ),
            diagonal=-1,
        )
        lag = time.unsqueeze(1) - time[1:].unsqueeze(0)
        lag = pt.where(mask, lag, pt.zeros_like(lag)).to(values.dtype)
        propagation = pt.exp(values[:, None, None] * lag.unsqueeze(0))
        forced = pt.einsum("ptk,pbk->pbt", propagation * mask.unsqueeze(0), injection)
        return homogeneous + forced

    def controlled_rollout(
        self,
        initial_state: pt.Tensor,
        time: pt.Tensor,
        control: pt.Tensor,
        forcing: pt.Tensor,
        *,
        uniform_time: Optional[bool] = None,
    ) -> pt.Tensor:
        """Vectorized exact modal zero-order-hold rollout.

        Initial states may be ``(rank,)`` or ``(rank, batch)``. Controls may
        be ``(controls, intervals)`` or ``(controls, batch, intervals)``.
        """
        if time.ndim != 1 or time.numel() < 1:
            raise ValueError("time must be a non-empty one-dimensional tensor")
        if (
            initial_state.ndim not in (1, 2)
            or initial_state.shape[0] != self.latent_size
        ):
            raise ValueError("initial latent state has an incompatible shape")
        if forcing.ndim not in (2, 3) or forcing.shape[-1] != time.numel() - 1:
            raise ValueError("forcing must contain one column per time interval")
        if control.shape != (self.latent_size, forcing.shape[0]):
            raise ValueError("control matrix and forcing dimensions are incompatible")
        if uniform_time is None:
            delta = time.diff()
            uniform_time = delta.numel() <= 1 or pt.allclose(
                delta, delta[0].expand_as(delta)
            )
        single = initial_state.ndim == 1 and forcing.ndim == 2
        initial = initial_state.unsqueeze(-1) if single else initial_state
        if initial.ndim == 1:
            initial = initial.unsqueeze(-1)
        values = forcing.unsqueeze(1) if forcing.ndim == 2 else forcing
        if initial.shape[1] == 1 and values.shape[1] > 1:
            initial = initial.expand(-1, values.shape[1])
        if values.shape[1] == 1 and initial.shape[1] > 1:
            values = values.expand(-1, initial.shape[1], -1)
        if values.shape[1] != initial.shape[1]:
            raise ValueError("forcing and initial-state batch sizes do not match")
        transformed = pt.linalg.solve(
            self.basis, pt.cat((initial, control.to(self.basis.dtype)), dim=1)
        )
        coordinate = transformed[:, : initial.shape[1]]
        modal_control = transformed[:, initial.shape[1] :]
        modal_forcing = pt.einsum(
            "ij,jbk->ibk", modal_control, values.to(self.basis.dtype)
        )
        if self._complex:
            spectral_values = self.growth.to(self.basis.dtype) + (
                2.0j * pi * self.frequency.to(self.basis.dtype)
            )
            modal = self._controlled_modes(
                coordinate, modal_forcing, spectral_values, time, uniform_time
            )
        else:
            pieces = []
            if self.n_pairs:
                pair_initial = coordinate[: 2 * self.n_pairs].reshape(
                    self.n_pairs, 2, coordinate.shape[1]
                )
                pair_forcing = modal_forcing[: 2 * self.n_pairs].reshape(
                    self.n_pairs,
                    2,
                    modal_forcing.shape[1],
                    modal_forcing.shape[2],
                )
                complex_initial = pt.complex(pair_initial[:, 0], pair_initial[:, 1])
                complex_forcing = pt.complex(pair_forcing[:, 0], pair_forcing[:, 1])
                spectral_values = pt.complex(
                    self.pair_growth, 2.0 * pi * self.frequency
                )
                pair = self._controlled_modes(
                    complex_initial,
                    complex_forcing,
                    spectral_values,
                    time,
                    uniform_time,
                )
                pieces.append(pt.stack((pair.real, pair.imag), dim=1).flatten(0, 1))
            if self.n_real:
                pieces.append(
                    self._controlled_modes(
                        coordinate[2 * self.n_pairs :],
                        modal_forcing[2 * self.n_pairs :],
                        self.real_growth,
                        time,
                        uniform_time,
                    )
                )
            modal = pt.cat(pieces, dim=0)
        flattened = modal.flatten(1, 2)
        result = (self.basis @ flattened).reshape(
            self.latent_size, initial.shape[1], time.numel()
        )
        if single:
            result = result[:, 0]
        return result if self._complex else result.to(initial_state.dtype)


def initialize_discrete_operator(
    first: pt.Tensor, second: pt.Tensor, stationary: bool
) -> pt.Tensor:
    """Fit an unconstrained or proper-orthogonal one-step operator."""
    if stationary:
        return _closest_proper_orthogonal(second @ first.conj().T)
    return second @ pt.linalg.pinv(first)


__all__ = ["ContinuousOperator", "initialize_discrete_operator"]
