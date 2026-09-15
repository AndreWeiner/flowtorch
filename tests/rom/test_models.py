"""Tests for the compositional ROM implementations."""

from inspect import signature

import pytest
import torch as pt

from flowtorch.analysis import SVD
from flowtorch.analysis.state_vector import (
    FieldSpec,
    StateVectorLayout,
    StateVectorSource,
)
from flowtorch.rom import (
    ControlledTrajectory,
    DMD,
    DMDc,
    FrequencySeparation,
    MonomialEmbedding,
    NoiseConfig,
    OptimizationStage,
    ParametricSnapshots,
    PODI,
    TimeDelayEmbedding,
    Trajectory,
    estimate_noise_std_variogram,
)
from flowtorch.rom.dynamics import ContinuousOperator, _closest_proper_orthogonal
from flowtorch.rom.training import _default_loss


class MatrixSource(StateVectorSource):
    def __init__(self, data):
        self.data = data
        self._layout = StateVectorLayout((FieldSpec("q"),), (data.shape[0],))

    @property
    def n_snapshots(self):
        return self.data.shape[1]

    @property
    def layout(self):
        return self._layout

    def read(self, spatial_slice, snapshot_slice):
        return self.data[spatial_slice, snapshot_slice]


def oscillator(n_times=50, dt=0.1, omega=2.0):
    time = pt.arange(n_times, dtype=pt.float64) * dt
    state = pt.stack((pt.cos(omega * time), pt.sin(omega * time)))
    return time, state


def second_order_sequence(n_times=60, dt=0.1):
    """Scalar recurrence with two positive discrete characteristic roots."""
    time = pt.arange(n_times, dtype=pt.float64) * dt
    first = pt.exp(pt.tensor(-0.2 * dt, dtype=pt.float64))
    second = pt.exp(pt.tensor(-0.7 * dt, dtype=pt.float64))
    index = pt.arange(n_times, dtype=pt.float64)
    state = (0.8 * first**index - 0.3 * second**index).unsqueeze(0)
    return time, state, float(first + second), float(-first * second)


def test_time_delay_embedding_order_readout_and_batches():
    embedding = TimeDelayEmbedding(3)
    trajectory = pt.arange(8, dtype=pt.float64).reshape(2, 4)

    embedded = embedding.transform_trajectory(trajectory)

    assert pt.equal(
        embedded,
        pt.tensor(
            [[0.0, 1.0], [4.0, 5.0], [1.0, 2.0], [5.0, 6.0], [2.0, 3.0], [6.0, 7.0]],
            dtype=trajectory.dtype,
        ),
    )
    assert pt.equal(embedding.initial(trajectory[:, :3]), embedded[:, 0])
    assert pt.equal(embedding.readout(embedded), trajectory[:, 2:])

    batched = pt.stack((trajectory, trajectory + 10.0), dim=1)
    batched_embedded = embedding.transform_trajectory(batched)
    assert batched_embedded.shape == (6, 2, 2)
    assert pt.equal(embedding.readout(batched_embedded), batched[..., 2:])


def test_monomial_embedding_order_readout_batches_and_gradient():
    embedding = MonomialEmbedding(2, include_constant=True)
    trajectory = pt.tensor(
        [[2.0, 3.0], [5.0, 7.0]], dtype=pt.float64, requires_grad=True
    )

    embedded = embedding.transform_trajectory(trajectory)

    expected = pt.stack(
        (
            pt.ones(2, dtype=trajectory.dtype),
            trajectory[0],
            trajectory[1],
            trajectory[0].square(),
            trajectory[0] * trajectory[1],
            trajectory[1].square(),
        )
    )
    pt.testing.assert_close(embedded, expected)
    pt.testing.assert_close(embedding.initial(trajectory[:, 0]), expected[:, 0])
    assert pt.equal(embedding.readout(embedded), trajectory)

    batched = pt.stack((trajectory, trajectory + 1.0), dim=1)
    batched_embedded = embedding.transform_trajectory(batched)
    assert batched_embedded.shape == (6, 2, 2)
    assert pt.equal(embedding.readout(batched_embedded), batched)

    embedded.sum().backward()
    assert trajectory.grad is not None
    assert bool(pt.isfinite(trajectory.grad).all())


def test_monomial_dmd_lifts_exponential_observables():
    time = pt.arange(40, dtype=pt.float64) * 0.05
    state = pt.exp(-0.4 * time).unsqueeze(0)
    model = DMD(
        rank=1,
        subtract_mean=False,
        stationary_initialization=False,
        embedding=MonomialEmbedding(2),
    ).fit(Trajectory(state, time))

    prediction = model.predict(state[:, 0], time=time).mean

    assert model.rank == 1
    assert model.embedded_rank == 2
    assert not pt.is_complex(model.operator)
    pt.testing.assert_close(prediction, state, rtol=2.0e-6, atol=1.0e-8)


def test_higher_order_dmd_predicts_second_order_recurrence_and_batches():
    time, state, _, _ = second_order_sequence()
    model = DMD(
        rank=1,
        subtract_mean=False,
        stationary_initialization=False,
        embedding=TimeDelayEmbedding(2),
    ).fit(Trajectory(state, time))

    prediction = model.predict(state[:, :2], time=time[1:]).mean

    assert model.rank == 1
    assert model.embedded_rank == 2
    assert not pt.is_complex(model.operator)
    pt.testing.assert_close(prediction, state[:, 1:], rtol=2.0e-6, atol=1.0e-8)

    histories = pt.stack((state[:, :2], state[:, 5:7]), dim=2)
    batched = model.predict(histories, time=time[:5]).mean
    expected = pt.stack((state[:, 1:6], state[:, 6:11]), dim=1)
    assert batched.shape == (1, 2, 5)
    pt.testing.assert_close(batched, expected, rtol=2.0e-6, atol=1.0e-8)


def test_delayed_noise_is_learned_before_hankel_embedding():
    time, state, _, _ = second_order_sequence(n_times=20)
    model = DMD(
        rank=1,
        subtract_mean=False,
        stationary_initialization=False,
        embedding=TimeDelayEmbedding(3),
        noise=NoiseConfig(initialization="zeros"),
    ).fit(Trajectory(state, time))

    assert model.learned_noise[0].shape == (1, 20)
    assert model.denoised_latent_trajectories[0].shape == (3, 18)


def test_pod_basis_centers_and_supports_lazy_sources():
    time, state = oscillator()
    shifted = state + pt.tensor([[2.0], [-1.0]], dtype=state.dtype)
    svd = SVD(shifted, rank=2, subtract_mean=True)
    pt.testing.assert_close(svd.decode(svd.encode(shifted)), shifted)

    source_svd = SVD(
        MatrixSource(shifted),
        rank=2,
        subtract_mean=True,
        spatial_batch_size=1,
        snapshot_batch_size=7,
    )
    coefficients = source_svd.encode(MatrixSource(shifted))
    result = source_svd.decode(coefficients)
    reconstructed = result.materialize_local()
    pt.testing.assert_close(reconstructed, shifted)


def test_stationary_dmd_is_exactly_real_and_predicts_continuous_time():
    time, state = oscillator()
    model = DMD(rank=2, stationary=True, subtract_mean=False).fit(
        Trajectory(state, time)
    )
    assert not pt.is_complex(model.operator)
    assert model.operator.dtype == state.dtype
    pt.testing.assert_close(model.growth_rate, pt.zeros(2, dtype=state.dtype))
    expected_frequency = pt.tensor(2.0 / (2.0 * pt.pi), dtype=state.dtype)
    pt.testing.assert_close(model.frequency.abs(), expected_frequency.expand(2))
    prediction = model.predict(state[:, 0], time=time[:12]).mean
    pt.testing.assert_close(prediction, state[:, :12], rtol=1.0e-6, atol=1.0e-7)

    loss = prediction.square().mean()
    loss.backward()
    assert model.dynamics.basis.grad is not None
    assert not pt.is_complex(model.dynamics.basis.grad)

    batched = model.predict(state[:, :3], time=time[:12]).mean
    assert batched.shape == (2, 3, 12)
    for index in range(3):
        expected = model.predict(state[:, index], time=time[:12]).mean
        pt.testing.assert_close(batched[:, index], expected)


def test_default_time_scaling_preserves_physical_dmd_units():
    time, state = oscillator()
    seconds = DMD(rank=2, stationary=True, subtract_mean=False).fit(
        Trajectory(state, time)
    )
    milliseconds = DMD(rank=2, stationary=True, subtract_mean=False).fit(
        Trajectory(state, 1000.0 * time)
    )

    assert seconds.time_scale == pytest.approx(float(time[1] - time[0]))
    assert milliseconds.time_scale == pytest.approx(1000.0 * seconds.time_scale)
    pt.testing.assert_close(
        milliseconds.dynamics.frequencies,
        seconds.dynamics.frequencies,
    )
    pt.testing.assert_close(milliseconds.frequency, seconds.frequency / 1000.0)
    pt.testing.assert_close(milliseconds.operator, seconds.operator / 1000.0)
    prediction = milliseconds.predict(state[:, 0], time=1000.0 * time[:12]).mean
    pt.testing.assert_close(prediction, state[:, :12], rtol=1.0e-6, atol=1.0e-7)


def test_time_scaling_can_be_disabled_or_set_explicitly():
    time, state = oscillator()
    unscaled = DMD(
        rank=2,
        stationary=True,
        subtract_mean=False,
        time_scale=None,
    ).fit(Trajectory(state, time))
    explicit = DMD(
        rank=2,
        stationary=True,
        subtract_mean=False,
        time_scale=2.5,
    ).fit(Trajectory(state, time))

    assert unscaled.time_scale == 1.0
    assert unscaled.dynamics.dt == pytest.approx(unscaled.dt)
    assert explicit.time_scale == 2.5
    assert explicit.dynamics.dt == pytest.approx(explicit.dt / 2.5)
    pt.testing.assert_close(
        unscaled.predict(state[:, 0], time=time[:12]).mean,
        explicit.predict(state[:, 0], time=time[:12]).mean,
    )


@pytest.mark.parametrize("time_scale", [0.0, -1.0, float("inf"), "invalid"])
def test_time_scale_is_validated(time_scale):
    with pytest.raises(ValueError, match="time_scale"):
        DMD(time_scale=time_scale)


def test_dmd_regressor_can_be_refitted_with_a_fixed_svd():
    time, state = oscillator()
    model = DMD(rank=2, subtract_mean=False).fit(Trajectory(state, time))
    modes = model.svd.U.clone()
    shifted_time = time + 3.0
    model.fit_regressor(Trajectory(0.8 * state, shifted_time))
    pt.testing.assert_close(model.svd.U, modes)
    assert not list(model.encoder.parameters())
    assert not list(model.decoder.parameters())
    assert all("svd" not in name for name, _ in model.named_parameters())

    model.to(dtype=pt.float32)
    assert model.svd.U.dtype == pt.float32
    prediction = model.predict(state[:, 0].float(), time=shifted_time[:4].float()).mean
    assert prediction.dtype == pt.float32


def test_real_nyquist_pair_remains_real():
    discrete = -pt.eye(2, dtype=pt.float64)
    operator = ContinuousOperator(discrete, dt=0.1, stationary=True)
    transition, _ = operator.transition_factors(0.1)
    reconstructed = operator.basis @ transition @ pt.linalg.pinv(operator.basis)
    pt.testing.assert_close(reconstructed, discrete)
    assert not pt.is_complex(operator.operator)


def test_multiple_trajectory_windows_noise_and_optimizers():
    assert NoiseConfig().weight == pytest.approx(0.001)
    assert NoiseConfig().initialization == "gaussian"
    time, state = oscillator(n_times=40)
    trajectories = [Trajectory(state, time), Trajectory(0.7 * state, time)]
    model = DMD(
        rank=2,
        subtract_mean=False,
        noise=NoiseConfig("gaussian", sigma=1.0, weight=0.1),
    ).fit(trajectories)
    assert len(model.learned_noise) == 2
    assert all(noise.norm() > 0.0 for noise in model.learned_noise)
    assert model.estimated_noise_std is not None
    initialized_rms = pt.cat(model.learned_noise, dim=1).square().mean(dim=1).sqrt()
    pt.testing.assert_close(initialized_rms, model.estimated_noise_std)
    train, validation = model._make_windows(4, 2, 0.25, True)
    assert train and validation
    assert {trajectory for trajectory, _, _ in train} == {0, 1}
    model.fine_tune(
        epochs=2,
        horizon=4,
        validation_fraction=0.25,
        optimizer=pt.optim.SGD,
        optimizer_options={"lr": 1.0e-3},
    )
    assert len(model.training_log["train_loss"]) == 2

    lbfgs = DMD(rank=2, subtract_mean=False).fit(trajectories[0])
    lbfgs.fine_tune(
        epochs=1,
        horizon=4,
        validation_fraction=0.0,
        optimizer=pt.optim.LBFGS,
        optimizer_options={"lr": 0.1, "max_iter": 2},
    )
    with pytest.raises(ValueError, match="full-batch"):
        lbfgs.fine_tune(
            epochs=1,
            horizon=4,
            validation_fraction=0.0,
            batch_size=2,
            optimizer=pt.optim.LBFGS,
        )

    extended = pt.vstack((state, 0.3 * state[:1]))
    full_noise = 0.01 * pt.ones_like(extended)
    supplied = DMD(
        rank=2,
        subtract_mean=False,
        noise=NoiseConfig((full_noise,)),
    ).fit(Trajectory(extended, time))
    assert supplied.learned_noise[0].shape == (2, time.numel())


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if pt.cuda.is_available() else [])
)
def test_robust_variogram_estimates_noise_scale_on_device(device):
    generator = pt.Generator().manual_seed(4)
    count = 4000
    time = pt.arange(count, dtype=pt.float64) * 0.002
    clean = pt.stack(
        (
            pt.sin(2.0 * pt.pi * 0.5 * time),
            0.7 * pt.cos(2.0 * pt.pi * time),
        )
    )
    expected = pt.tensor([0.05, 0.12], dtype=clean.dtype)
    observed = clean + expected[:, None] * pt.randn(
        clean.shape, generator=generator, dtype=clean.dtype
    )
    observed = observed.to(device)
    estimated = estimate_noise_std_variogram([observed], max_lag=8)
    assert estimated.device.type == device
    pt.testing.assert_close(estimated.cpu(), expected, rtol=0.15, atol=0.005)


def test_estimated_amplitude_loss_matches_variogram_scale():
    time, state = oscillator(n_times=100)
    model = DMD(
        rank=2,
        subtract_mean=False,
        noise=NoiseConfig(initialization="zeros", penalty="estimated_amplitude"),
    ).fit(Trajectory(state, time))
    assert model.estimated_noise_std is not None
    zero_loss = model._noise_regularization()
    with pt.no_grad():
        model.noise_parameters[0].copy_(
            model.estimated_noise_std[:, None].expand_as(model.noise_parameters[0])
        )
    matched_loss = model._noise_regularization()
    assert float(zero_loss.detach()) == pytest.approx(model.noise_config.weight)
    assert float(matched_loss.detach()) < float(zero_loss.detach()) * 1.0e-6


def test_gaussian_residual_initialization_matches_estimated_amplitude_loss():
    time, state = oscillator(n_times=100)
    model = DMD(
        rank=2,
        subtract_mean=False,
        noise=NoiseConfig(
            initialization="gaussian",
            penalty="estimated_amplitude",
        ),
    ).fit(Trajectory(state, time))
    assert model.estimated_noise_std is not None
    initialized_rms = pt.cat(model.learned_noise, dim=1).square().mean(dim=1).sqrt()
    pt.testing.assert_close(initialized_rms, model.estimated_noise_std)
    assert float(model._noise_regularization().detach()) < 1.0e-12


def test_default_dynamics_use_trainable_zero_growth_initialization():
    assert not DMD().stationary
    assert not DMDc().stationary
    assert DMD().stationary_initialization
    assert DMDc().stationary_initialization

    time = pt.arange(30, dtype=pt.float64) * 0.1
    state = pt.stack((pt.exp(0.2 * time), pt.exp(-0.1 * time)))
    model = DMD(rank=2, subtract_mean=False).fit(Trajectory(state, time))
    assert pt.equal(model.growth_rate, pt.zeros_like(model.growth_rate))
    assert model.growth_rate.requires_grad
    transition, _ = model.dynamics.transition_factors(model.dt / model.time_scale)
    discrete = model.dynamics.basis @ transition @ pt.linalg.pinv(model.dynamics.basis)
    pt.testing.assert_close(discrete.T @ discrete, pt.eye(2, dtype=state.dtype))

    control_time, controlled_state, forcing = _controlled_data()
    controlled = DMDc(rank=2, subtract_mean=False).fit(
        ControlledTrajectory(controlled_state, control_time, forcing)
    )
    assert pt.equal(controlled.growth_rate, pt.zeros_like(controlled.growth_rate))
    assert controlled.growth_rate.requires_grad
    transition, _ = controlled.dynamics.transition_factors(
        controlled.dt / controlled.time_scale
    )
    discrete = (
        controlled.dynamics.basis
        @ transition
        @ pt.linalg.pinv(controlled.dynamics.basis)
    )
    pt.testing.assert_close(
        discrete.T @ discrete,
        pt.eye(2, dtype=controlled_state.dtype),
    )

    unconstrained = DMD(
        rank=2,
        stationary_initialization=False,
        subtract_mean=False,
    ).fit(Trajectory(state, time))
    assert not pt.allclose(
        unconstrained.growth_rate,
        pt.zeros_like(unconstrained.growth_rate),
    )


def test_default_loss_is_mse():
    prediction = pt.tensor([1.0, 3.0], dtype=pt.float64, requires_grad=True)
    target = pt.tensor([0.0, 1.0], dtype=pt.float64)
    loss = _default_loss(prediction, target)
    pt.testing.assert_close(loss, pt.tensor(2.5, dtype=pt.float64))
    loss.backward()
    pt.testing.assert_close(prediction.grad, pt.tensor([1.0, 2.0], dtype=pt.float64))


@pytest.mark.parametrize("controlled", [False, True])
def test_fine_tuning_retains_better_epoch_zero_for_dmd_and_dmdc(controlled):
    if controlled:
        time, state, forcing = _controlled_data()
        model = DMDc(rank=2, subtract_mean=False).fit(
            ControlledTrajectory(state, time, forcing)
        )
    else:
        time, state = oscillator(n_times=60)
        model = DMD(rank=2, subtract_mean=False).fit(Trajectory(state, time))
    initial_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    log = model.fine_tune(
        epochs=2,
        horizon=4,
        validation_fraction=0.2,
        optimizer=pt.optim.AdamW,
        optimizer_options={"lr": 0.1, "weight_decay": 1.0},
        patience=1,
    )
    assert log["best_epoch"] == 0
    assert log["best_train_loss"] == log["initial_train_loss"]
    assert log["best_validation_loss"] == log["initial_validation_loss"]
    assert log["train_loss"][0] > log["initial_train_loss"]
    assert log["val_loss"][0] > log["initial_validation_loss"]
    for name, value in model.state_dict().items():
        assert pt.equal(value, initial_state[name])


@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("stationary", [False, True])
def test_default_fine_tuning_does_not_degrade_exact_dmd_or_dmdc(controlled, stationary):
    if controlled:
        time, state, forcing = _controlled_data()
        model = DMDc(rank=2, stationary=stationary, subtract_mean=False).fit(
            ControlledTrajectory(state, time, forcing)
        )
        prediction_options = {"forcing": forcing}
    else:
        time, state = oscillator(n_times=60)
        model = DMD(rank=2, stationary=stationary, subtract_mean=False).fit(
            Trajectory(state, time)
        )
        prediction_options = {}
    before = model.predict(state[:, 0], time=time, **prediction_options).mean
    before_error = (before - state).norm()
    log = model.fine_tune(epochs=5, horizon=4, validation_fraction=0.2)
    after = model.predict(state[:, 0], time=time, **prediction_options).mean
    after_error = (after - state).norm()
    assert log["best_epoch"] == 0
    pt.testing.assert_close(after_error, before_error, rtol=0.0, atol=0.0)


def test_adamw_defaults_to_zero_weight_decay():
    class RecordingAdamW(pt.optim.AdamW):
        options = None

        def __init__(self, parameters, **options):
            type(self).options = options.copy()
            super().__init__(parameters, **options)

    time, state = oscillator(n_times=30)
    model = DMD(rank=2, subtract_mean=False).fit(Trajectory(state, time))
    model.fine_tune(
        epochs=1,
        horizon=4,
        validation_fraction=0.2,
        optimizer=RecordingAdamW,
        optimizer_options={"lr": 1.0e-3},
    )
    assert RecordingAdamW.options["weight_decay"] == 0.0


def test_validation_split_reserves_a_complete_horizon_or_falls_back():
    assert signature(DMD.fine_tune).parameters["validation_fraction"].default == 0.0
    time, state = oscillator(n_times=257)
    model = DMD(rank=2, subtract_mean=False).fit(Trajectory(state, time))
    train, validation = model._make_windows(64, 1, 0.2, False)
    assert train
    assert validation == [(0, 192, False)]

    short_time, short_state = oscillator(n_times=100)
    short = DMD(rank=2, subtract_mean=False).fit(Trajectory(short_state, short_time))
    train, validation = short._make_windows(64, 1, 0.2, False)
    assert train
    assert not validation


def test_dmd_and_dmdc_disable_backward_loss_by_default():
    assert signature(DMD.fine_tune).parameters["forward_backward"].default is False
    assert signature(DMDc.fine_tune).parameters["forward_backward"].default is False


@pytest.mark.parametrize("validation_loss_weight", [0.0, 0.5, 1.0])
def test_selection_loss_mixes_training_and_validation(
    validation_loss_weight,
):
    time, state = oscillator(n_times=40)
    model = DMD(rank=2, subtract_mean=False).fit(Trajectory(state, time))
    log = model.fine_tune(
        epochs=1,
        horizon=4,
        validation_fraction=0.2,
        validation_loss_weight=validation_loss_weight,
        optimizer=pt.optim.SGD,
        optimizer_options={"lr": 0.0},
        regularizers=[],
        scheduler=None,
    )
    expected_initial = (1.0 - validation_loss_weight) * log[
        "initial_train_loss"
    ] + validation_loss_weight * log["initial_validation_loss"]
    expected_epoch = (1.0 - validation_loss_weight) * log["train_loss"][
        0
    ] + validation_loss_weight * log["val_loss"][0]
    assert log["initial_selection_loss"] == pytest.approx(expected_initial)
    assert log["selection_loss"][0] == pytest.approx(expected_epoch)


def test_selection_loss_uses_training_when_validation_is_unavailable():
    time, state = oscillator(n_times=100)
    model = DMD(rank=2, subtract_mean=False).fit(Trajectory(state, time))
    log = model.fine_tune(
        epochs=1,
        horizon=64,
        validation_fraction=0.2,
        validation_loss_weight=1.0,
        optimizer=pt.optim.SGD,
        optimizer_options={"lr": 0.0},
        regularizers=[],
        scheduler=None,
    )
    assert log["initial_validation_loss"] == log["initial_train_loss"]
    assert log["initial_selection_loss"] == log["initial_train_loss"]
    assert log["selection_loss"] == log["train_loss"]


@pytest.mark.parametrize("validation_loss_weight", [-0.1, 1.1])
def test_selection_loss_weight_is_validated(validation_loss_weight):
    time, state = oscillator(n_times=20)
    model = DMD(rank=2, subtract_mean=False).fit(Trajectory(state, time))
    with pytest.raises(ValueError, match="validation_loss_weight"):
        model.fine_tune(
            epochs=1,
            horizon=2,
            validation_loss_weight=validation_loss_weight,
        )


def test_default_scheduler_reduces_learning_rate_on_a_plateau():
    assert signature(DMD.fine_tune).parameters["validation_loss_weight"].default == 0.5
    time, state = oscillator(n_times=30)
    model = DMD(rank=2, subtract_mean=False).fit(Trajectory(state, time))

    def constant_loss(prediction, target):
        return prediction.sum() * 0.0 + 1.0

    log = model.fine_tune(
        epochs=22,
        horizon=4,
        validation_fraction=0.2,
        optimizer=pt.optim.SGD,
        optimizer_options={"lr": 0.1},
        loss_function=constant_loss,
        patience=30,
    )
    assert log["learning_rate"][0] == pytest.approx(0.1)
    assert log["learning_rate"][-1] == pytest.approx(0.05)


def test_frequency_separation_is_smooth_physical_and_order_aware():
    assert FrequencySeparation().weight == 0.1
    time = pt.arange(80, dtype=pt.float64) * 0.05
    state = pt.stack(
        (
            pt.cos(time),
            pt.sin(time),
            pt.cos(2.5 * time),
            pt.sin(2.5 * time),
        )
    )
    model = DMD(rank=4, stationary=True, subtract_mean=False).fit(
        Trajectory(state, time)
    )
    assert bool((model.dynamics.frequency.diff() >= 0.0).all())
    regularizer = FrequencySeparation(
        min_spacing=0.1,
        weight=1.0,
        temperature=0.01,
        enforce_nyquist=False,
    )
    with pt.no_grad():
        model.dynamics.frequency.copy_(
            model.time_scale * pt.tensor([0.2, 0.21], dtype=time.dtype)
        )
    collapsed = regularizer(model)
    collapsed.backward()
    assert model.dynamics.frequency.grad[0] > 0.0
    assert model.dynamics.frequency.grad[1] < 0.0

    model.zero_grad(set_to_none=True)
    with pt.no_grad():
        model.dynamics.frequency.copy_(
            model.time_scale * pt.tensor([0.2, 0.5], dtype=time.dtype)
        )
    separated = regularizer(model)
    assert separated < collapsed
    resolution = 1.0 / (time.numel() * float(time[1] - time[0]))
    automatic = FrequencySeparation(
        weight=1.0,
        temperature=0.01,
        enforce_nyquist=False,
    )
    explicit = FrequencySeparation(
        min_spacing=resolution,
        weight=1.0,
        temperature=0.01,
        enforce_nyquist=False,
    )
    pt.testing.assert_close(automatic(model), explicit(model))


def test_regularization_is_included_once_in_selection_loss():
    time = pt.arange(80, dtype=pt.float64) * 0.05
    state = pt.stack(
        (
            pt.cos(time),
            pt.sin(time),
            pt.cos(2.5 * time),
            pt.sin(2.5 * time),
        )
    )
    model = DMD(rank=4, stationary=True, subtract_mean=False).fit(
        Trajectory(state, time)
    )
    with pt.no_grad():
        model.dynamics.frequency.copy_(
            model.time_scale * pt.tensor([0.2, 0.21], dtype=time.dtype)
        )
    log = model.fine_tune(
        epochs=1,
        horizon=4,
        validation_fraction=0.2,
        optimizer=pt.optim.SGD,
        optimizer_options={"lr": 0.0},
        regularizers=[FrequencySeparation(0.1, enforce_nyquist=False)],
        scheduler=None,
    )
    mixed = 0.5 * (log["initial_train_loss"] + log["initial_validation_loss"])
    assert log["initial_regularization_loss"] > 0.0
    assert log["initial_selection_loss"] == pytest.approx(
        mixed + log["initial_regularization_loss"]
    )
    assert len(log["regularization_loss"]) == 1


def test_frequency_separation_is_enabled_by_default_and_can_be_disabled():
    time = pt.arange(80, dtype=pt.float64) * 0.05
    state = pt.stack(
        (
            pt.cos(time),
            pt.sin(time),
            pt.cos(2.5 * time),
            pt.sin(2.5 * time),
        )
    )
    model = DMD(rank=4, stationary=True, subtract_mean=False).fit(
        Trajectory(state, time)
    )
    with pt.no_grad():
        model.dynamics.frequency.copy_(
            model.time_scale * pt.tensor([0.2, 0.21], dtype=time.dtype)
        )
    common = dict(
        epochs=1,
        horizon=4,
        optimizer=pt.optim.SGD,
        optimizer_options={"lr": 0.0},
        scheduler=None,
    )
    default_log = model.fine_tune(**common)
    disabled_log = model.fine_tune(**common, regularizers=[])
    assert default_log["initial_regularization_loss"] > 0.0
    assert disabled_log["initial_regularization_loss"] == 0.0


def test_default_fine_tuning_uses_adamw_then_lbfgs():
    assert signature(DMD.fine_tune).parameters["epochs"].default == 150
    time, state = oscillator(n_times=40)
    model = DMD(rank=2, subtract_mean=False).fit(Trajectory(state, time))
    log = model.fine_tune(
        epochs=5,
        horizon=4,
        validation_fraction=0.0,
        batch_size=8,
    )
    assert log["stage"] == ["AdamW"] * 4 + ["LBFGS"]
    assert log["stage_epoch"] == [1, 2, 3, 4, 1]


def test_noise_update_is_tracked_only_when_noise_learning_is_active():
    assert (
        signature(DMD.fine_tune).parameters["noise_update_tolerance"].default == 1.0e-4
    )
    time, state = oscillator(n_times=40)
    plain = DMD(rank=2, subtract_mean=False).fit(Trajectory(state, time))
    plain_log = plain.fine_tune(
        epochs=2,
        horizon=4,
        optimizer=pt.optim.SGD,
        optimizer_options={"lr": 0.0},
        patience=1,
        regularizers=[],
        scheduler=None,
    )
    assert plain_log["noise_update"] == []

    noisy = DMD(
        rank=2,
        subtract_mean=False,
        noise=NoiseConfig(),
    ).fit(Trajectory(state, time))
    noisy_log = noisy.fine_tune(
        epochs=2,
        horizon=4,
        optimizer=pt.optim.SGD,
        optimizer_options={"lr": 0.0},
        patience=1,
        regularizers=[],
        scheduler=None,
    )
    assert noisy_log["noise_update"] == [0.0]


def test_default_lbfgs_stage_accepts_noncontiguous_spectral_parameters():
    time, state = oscillator(n_times=40)
    model = DMD(rank=2, subtract_mean=False).fit(Trajectory(state, time))
    basis = model.dynamics.basis.detach()
    model.dynamics.basis.data = basis.mT.contiguous().mT
    assert not model.dynamics.basis.is_contiguous()

    log = model.fine_tune(
        epochs=2,
        horizon=4,
        validation_fraction=0.0,
        scheduler=None,
    )

    assert log["stage"] == ["AdamW", "LBFGS"]
    assert model.dynamics.basis.is_contiguous()
    stages = model._default_optimization_stages(150, None, None, 40)
    assert [stage.epochs for stage in stages] == [100, 50]
    assert [stage.patience for stage in stages] == [40, 10]
    assert all(stage.parameters == "all" for stage in stages)
    assert stages[1].options == {"line_search_fn": "strong_wolfe"}


def test_explicit_optimization_stages_override_epoch_budget():
    time, state = oscillator(n_times=30)
    model = DMD(rank=2, subtract_mean=False).fit(Trajectory(state, time))
    stages = [
        OptimizationStage(
            pt.optim.SGD,
            epochs=2,
            options={"lr": 0.0},
            scheduler=None,
        )
    ]
    log = model.fine_tune(
        epochs=100,
        horizon=4,
        validation_fraction=0.0,
        stages=stages,
    )
    assert log["stage"] == ["SGD", "SGD"]


def _controlled_data(n_times=60, dt=0.05, growth=0.0):
    generator = pt.Generator().manual_seed(4)
    time = pt.arange(n_times, dtype=pt.float64) * dt
    forcing = pt.randn((1, n_times - 1), generator=generator, dtype=pt.float64)
    operator = pt.tensor([[growth, -1.5], [1.5, growth]], dtype=pt.float64)
    control = pt.tensor([[0.8], [-0.2]], dtype=pt.float64)
    state = pt.zeros((2, n_times), dtype=pt.float64)
    for index in range(n_times - 1):
        augmented = pt.zeros((3, 3), dtype=pt.float64)
        augmented[:2, :2] = operator
        augmented[:2, 2:] = control
        transition = pt.matrix_exp(augmented * dt)
        state[:, index + 1] = (
            transition[:2, :2] @ state[:, index]
            + transition[:2, 2:] @ forcing[:, index]
        )
    return time, state, forcing


def _delayed_control_data(n_times=80, dt=0.1):
    generator = pt.Generator().manual_seed(12)
    time = pt.arange(n_times, dtype=pt.float64) * dt
    forcing = pt.randn((1, n_times - 1), generator=generator, dtype=pt.float64)
    first_root = 0.96
    second_root = 0.87
    first_order = first_root + second_root
    second_order = -(first_root * second_root)
    state = pt.zeros((1, n_times), dtype=pt.float64)
    state[:, 0] = 0.2
    state[:, 1] = -0.1
    for index in range(1, n_times - 1):
        state[:, index + 1] = (
            first_order * state[:, index]
            + second_order * state[:, index - 1]
            + 0.4 * forcing[:, index]
            - 0.15 * forcing[:, index - 1]
        )
    return time, state, forcing


def test_monomial_dmdc_lifts_control_features():
    n_times = 80
    dt = 0.1
    time = pt.arange(n_times, dtype=pt.float64) * dt
    generator = pt.Generator().manual_seed(4)
    forcing = pt.randn((1, n_times - 1), generator=generator, dtype=pt.float64)
    rate = -0.3
    linear_control = 0.7
    quadratic_control = -0.2
    transition = pt.exp(pt.tensor(rate * dt, dtype=pt.float64))
    response = pt.expm1(pt.tensor(rate * dt, dtype=pt.float64)) / rate
    state = pt.empty((1, n_times), dtype=pt.float64)
    state[:, 0] = 0.25
    for index in range(n_times - 1):
        lifted_control = (
            linear_control * forcing[:, index]
            + quadratic_control * forcing[:, index].square()
        )
        state[:, index + 1] = transition * state[:, index] + response * lifted_control

    model = DMDc(
        rank=1,
        subtract_mean=False,
        stationary_initialization=False,
        control_embedding=MonomialEmbedding(2),
    ).fit(ControlledTrajectory(state, time, forcing))
    prediction = model.predict(state[:, 0], time=time, forcing=forcing).mean

    assert model.n_controls == 1
    assert model.n_control_features == 2
    pt.testing.assert_close(
        model.B,
        pt.tensor([[linear_control, quadratic_control]], dtype=state.dtype),
    )
    pt.testing.assert_close(prediction, state, rtol=2.0e-6, atol=1.0e-8)


def test_higher_order_dmdc_delays_state_and_forcing():
    time, state, forcing = _delayed_control_data()
    model = DMDc(
        rank=1,
        subtract_mean=False,
        stationary_initialization=False,
        embedding=TimeDelayEmbedding(2),
        control_embedding=TimeDelayEmbedding(2),
    ).fit(ControlledTrajectory(state, time, forcing))

    with pytest.raises(ValueError, match="forcing_history"):
        model.predict(
            state[:, :2],
            time=time[1:],
            forcing=forcing[:, 1:],
        )

    prediction = model.predict(
        state[:, :2],
        time=time[1:],
        forcing=forcing[:, 1:],
        forcing_history=forcing[:, :1],
    ).mean

    assert model.n_controls == 1
    assert model.n_control_features == 2
    assert model.B.shape == (2, 2)
    assert not pt.is_complex(model.operator)
    assert not pt.is_complex(model.B)
    pt.testing.assert_close(prediction, state[:, 1:], rtol=2.0e-5, atol=2.0e-7)

    horizon = 10
    histories = pt.stack((state[:, :2], state[:, 10:12]), dim=2)
    future_forcing = pt.stack(
        (forcing[:, 1 : 1 + horizon], forcing[:, 11 : 11 + horizon]), dim=1
    )
    forcing_history = pt.stack((forcing[:, :1], forcing[:, 10:11]), dim=1)
    batched = model.predict(
        histories,
        time=time[: horizon + 1],
        forcing=future_forcing,
        forcing_history=forcing_history,
    ).mean
    expected = pt.stack((state[:, 1 : 2 + horizon], state[:, 11 : 12 + horizon]), dim=1)
    assert batched.shape == (1, 2, horizon + 1)
    pt.testing.assert_close(batched, expected, rtol=2.0e-5, atol=2.0e-7)


def test_dmdc_modal_prediction_matches_augmented_exponential():
    time, state, forcing = _controlled_data()
    model = DMDc(rank=2, stationary=True, subtract_mean=False).fit(
        ControlledTrajectory(state, time, forcing)
    )
    count = 15
    prediction = model.predict(
        state[:, 0], time=time[:count], forcing=forcing[:, : count - 1]
    ).mean
    latent = model.encoder(state[:, 0])
    expected = [latent]
    for index, delta in enumerate(time[:count].diff()):
        augmented = pt.zeros((3, 3), dtype=latent.dtype)
        augmented[:2, :2] = model.operator
        augmented[:2, 2:] = model.B
        transition = pt.matrix_exp(augmented * delta)
        expected.append(
            transition[:2, :2] @ expected[-1] + transition[:2, 2:] @ forcing[:, index]
        )
    expected_full = model.decoder(pt.stack(expected, dim=1))
    pt.testing.assert_close(prediction, expected_full, rtol=1.0e-6, atol=1.0e-7)
    assert not pt.is_complex(model.operator)
    assert not pt.is_complex(model.B)

    latent = model.encoder(state)
    first, second = latent[:, :-1], latent[:, 1:]
    forcing_inverse = pt.linalg.pinv(forcing)
    first_projected = first - (first @ forcing_inverse) @ forcing
    second_projected = second - (second @ forcing_inverse) @ forcing
    expected_a = _closest_proper_orthogonal(second_projected @ first_projected.T)
    expected_b = (second - expected_a @ first) @ forcing_inverse
    transition, response = model.dynamics.transition_factors(
        model.dt / model.time_scale
    )
    basis_inverse = pt.linalg.pinv(model.dynamics.basis)
    actual_a = model.dynamics.basis @ transition @ basis_inverse
    actual_b = (
        model.dynamics.basis @ response @ basis_inverse @ model._require_control()
    )
    pt.testing.assert_close(actual_a, expected_a)
    pt.testing.assert_close(actual_b, expected_b)

    batched = model.predict(
        state[:, :3],
        time=time[:count],
        forcing=forcing[:, : count - 1],
    ).mean
    assert batched.shape == (2, 3, count)
    for index in range(3):
        expected = model.predict(
            state[:, index],
            time=time[:count],
            forcing=forcing[:, : count - 1],
        ).mean
        pt.testing.assert_close(batched[:, index], expected)

    forcing_batch = forcing[:, : count - 1].unsqueeze(1).repeat(1, 3, 1)
    independently_forced = model.predict(
        state[:, 0], time=time[:count], forcing=forcing_batch
    ).mean
    assert independently_forced.shape == (2, 3, count)
    pt.testing.assert_close(independently_forced[:, 0], prediction)

    query_time = pt.tensor([0.0, 0.02, 0.09, 0.17, 0.31], dtype=time.dtype)
    query_forcing = forcing[:, : query_time.numel() - 1]
    nonuniform = model.predict(state[:, 0], time=query_time, forcing=query_forcing).mean
    expected = [model.encoder(state[:, 0])]
    for index, delta in enumerate(query_time.diff()):
        augmented = pt.zeros((3, 3), dtype=state.dtype)
        augmented[:2, :2] = model.operator
        augmented[:2, 2:] = model.B
        step = pt.matrix_exp(augmented * delta)
        expected.append(
            step[:2, :2] @ expected[-1] + step[:2, 2:] @ query_forcing[:, index]
        )
    pt.testing.assert_close(nonuniform, model.decoder(pt.stack(expected, dim=1)))

    modes = model.svd.U.clone()
    model.fit_regressor(ControlledTrajectory(state, time, forcing))
    pt.testing.assert_close(model.svd.U, modes)
    model.fine_tune(
        epochs=1,
        horizon=3,
        forward_backward=True,
        validation_fraction=0.0,
        optimizer=pt.optim.SGD,
        optimizer_options={"lr": 1.0e-4},
    )
    assert model.control.grad is not None


def test_dmd_and_dmdc_support_lbfgs_fine_tuning():
    time, state, forcing = _controlled_data(growth=0.18)
    cases = (
        (DMD(rank=2, subtract_mean=False), Trajectory(state, time)),
        (
            DMDc(rank=2, subtract_mean=False),
            ControlledTrajectory(state, time, forcing),
        ),
    )
    for model, data in cases:
        model.fit(data)
        log = model.fine_tune(
            epochs=3,
            horizon=6,
            validation_fraction=0.2,
            optimizer=pt.optim.LBFGS,
            optimizer_options={
                "lr": 0.5,
                "max_iter": 8,
                "history_size": 10,
                "line_search_fn": "strong_wolfe",
            },
        )
        assert log["best_selection_loss"] < log["initial_selection_loss"]
        assert any(parameter.grad is not None for parameter in model.parameters())
        if isinstance(model, DMDc):
            assert model.control.grad is not None


def test_default_time_scaling_preserves_physical_dmdc_units():
    time, state, forcing = _controlled_data()
    seconds = DMDc(rank=2, stationary=True, subtract_mean=False).fit(
        ControlledTrajectory(state, time, forcing)
    )
    milliseconds = DMDc(rank=2, stationary=True, subtract_mean=False).fit(
        ControlledTrajectory(state, 1000.0 * time, forcing)
    )

    pt.testing.assert_close(milliseconds.operator, seconds.operator / 1000.0)
    pt.testing.assert_close(milliseconds.B, seconds.B / 1000.0)
    prediction = milliseconds.predict(
        state[:, 0],
        time=1000.0 * time[:12],
        forcing=forcing[:, :11],
    ).mean
    expected = seconds.predict(
        state[:, 0], time=time[:12], forcing=forcing[:, :11]
    ).mean
    pt.testing.assert_close(prediction, expected)


def test_podi_rbf_reproduces_training_samples_and_is_differentiable():
    parameters = pt.linspace(-1.0, 1.0, 9, dtype=pt.float64).unsqueeze(0)
    state = pt.vstack((parameters.squeeze().square(), pt.sin(parameters.squeeze())))
    model = PODI(rank=2, subtract_mean=False).fit(
        ParametricSnapshots(state, parameters)
    )
    query = parameters.clone().requires_grad_(True)
    prediction = model.predict(parameters=query).mean
    pt.testing.assert_close(prediction, state, rtol=2.0e-5, atol=2.0e-5)
    prediction.square().mean().backward()
    assert query.grad is not None
    assert model.interpolator.log_length_scale.grad is not None


def test_podi_supports_multiple_parameter_dimensions():
    angle = pt.tensor([-4.0, 0.0, 4.0], dtype=pt.float64)
    mach = pt.tensor([0.3, 0.6, 0.8], dtype=pt.float64)
    alpha_grid, mach_grid = pt.meshgrid(angle, mach, indexing="ij")
    parameters = pt.stack((alpha_grid.flatten(), mach_grid.flatten()))
    states = pt.stack(
        (
            parameters[0] + 2.0 * parameters[1],
            parameters[0].square() + parameters[1],
            pt.sin(parameters[0]) * parameters[1],
        )
    )
    model = PODI(rank=3, subtract_mean=False).fit(
        ParametricSnapshots(states, parameters)
    )
    prediction = model.predict(parameters=parameters[:, :2]).mean
    pt.testing.assert_close(prediction, states[:, :2], rtol=2.0e-5, atol=2.0e-5)


def test_training_time_and_forcing_validation():
    time, state = oscillator(n_times=12)
    bad_time = time.clone()
    bad_time[4] += 0.01
    with pytest.raises(ValueError, match="uniformly"):
        DMD(rank=2).fit(Trajectory(state, bad_time))
    with pytest.raises(ValueError, match="one column"):
        DMDc(rank=2).fit(ControlledTrajectory(state, time, pt.zeros((1, time.numel()))))
