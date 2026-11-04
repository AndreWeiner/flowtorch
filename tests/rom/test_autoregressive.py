"""Numerical contracts for reusable finite-context autoregressive predictors."""

from copy import deepcopy

import pytest
import torch as pt
from torch import nn
from torch.utils.data import DataLoader

from flowtorch.rom import (
    AutoregressiveModel,
    CausalTemporalConv,
    TCNMLPAutoregressor,
    ConvVAE,
    ConvVAEROM,
    EasyAttentionAutoregressor,
    GRUAutoregressor,
    ImageSequenceDataset,
    LatentDynamics,
    LSTMAutoregressor,
    MLPAutoregressor,
)
from flowtorch.rom.autoregressive import _EasyAttention


@pytest.fixture(autouse=True)
def deterministic_small_networks():
    previous = pt.get_num_threads()
    pt.set_num_threads(1)
    pt.manual_seed(41)
    yield
    pt.set_num_threads(previous)


MODELS = [
    MLPAutoregressor,
    LSTMAutoregressor,
    GRUAutoregressor,
    EasyAttentionAutoregressor,
    TCNMLPAutoregressor,
]


@pytest.mark.parametrize("model_type", MODELS)
def test_standalone_forecast_gradients_conditioning_and_reload(model_type):
    model = model_type(3, history_length=4, parameter_dim=2).double()
    history = pt.randn(2, 4, 3, dtype=pt.float64, requires_grad=True)
    parameters = pt.randn(2, 2, dtype=pt.float64, requires_grad=True)
    forecast = model.rollout(history, 3, parameters)
    assert forecast.shape == (2, 3, 3)
    assert forecast.dtype == pt.float64
    forecast[:, -1].square().mean().backward()
    assert history.grad[:, 0].abs().sum() > 0
    assert parameters.grad.abs().sum() > 0
    assert all(
        p.grad is not None and pt.isfinite(p.grad).all() for p in model.parameters()
    )
    # No hidden state leaks between calls, batches, or conditions.
    model.eval()
    model.rollout(pt.randn_like(history), 2, pt.randn_like(parameters))
    pt.testing.assert_close(model.rollout(history, 3, parameters), forecast)
    separate = pt.cat(
        [model.rollout(history[i : i + 1], 3, parameters[i : i + 1]) for i in range(2)]
    )
    pt.testing.assert_close(separate, forecast)
    restored = deepcopy(model)
    restored.load_state_dict(model.state_dict())
    pt.testing.assert_close(restored.rollout(history, 3, parameters), forecast)
    future = pt.randn_like(forecast)
    pt.testing.assert_close(
        model.rollout_loss(history, future, parameters),
        (forecast - future).square().mean(),
    )
    with pytest.raises(ValueError, match="history"):
        model(history[:, 1:], parameters)
    with pytest.raises(ValueError, match="parameters"):
        model(history, parameters[:, None])
    with pytest.raises(ValueError, match="future"):
        model.rollout_loss(history, future[:1], parameters)


@pytest.mark.parametrize("model_type", [LSTMAutoregressor, GRUAutoregressor])
def test_recurrent_next_state_matches_native_sequence_computation(model_type):
    model = model_type(
        2, history_length=3, parameter_dim=1, hidden_dim=5, num_layers=2
    ).double()
    history = pt.randn(2, 3, 2, dtype=pt.float64)
    condition = pt.randn(2, 1, dtype=pt.float64)
    tokens = pt.cat((history, condition[:, None].expand(-1, 3, -1)), dim=-1)
    # Explicitly supply a zero hidden/cell state as an independent API path.
    hidden = pt.zeros(2, 2, 5, dtype=pt.float64)
    state = (hidden, hidden.clone()) if model_type is LSTMAutoregressor else hidden
    output, _ = model.recurrent(tokens, state)
    expected = (
        history[:, -1] + output[:, -1] @ model.output.weight.T + model.output.bias
    )
    pt.testing.assert_close(model(history, condition), expected)
    manual = []
    for _ in range(3):
        predicted = model(history, condition)
        manual.append(predicted)
        history = pt.cat((history[:, 1:], predicted[:, None]), dim=1)
    # Each window must restart, rather than secretly carrying previous states.
    original = tokens[:, :, :2]
    pt.testing.assert_close(
        model.rollout(original, 3, condition), pt.stack(manual, dim=1)
    )


def test_easy_attention_matches_paper_matrix_equation_and_gradients():
    attention = _EasyAttention(3, 4, 2).double()
    x = pt.arange(24, dtype=pt.float64).reshape(2, 3, 4) / 10
    with pt.no_grad():
        attention.alpha.copy_(
            pt.tensor(
                [
                    [[1.0, 2.0, -1.0], [0.0, 3.0, 1.0], [2.0, 0.0, 4.0]],
                    [[-2.0, 1.0, 0.0], [3.0, 0.0, 2.0], [1.0, -1.0, 3.0]],
                ]
            )
        )
    values = x @ attention.value.weight.T
    expected = pt.cat(
        [attention.alpha[h] @ values[:, :, 2 * h : 2 * h + 2] for h in range(2)], dim=-1
    )
    actual = attention(x)
    pt.testing.assert_close(actual, expected)
    # Unnormalized, signed weights; no hidden softmax or input-dependent scores.
    pt.testing.assert_close(attention(2 * x), 2 * actual)
    actual.sum().backward()
    expected_gradient = values.detach().reshape(2, 3, 2, 2).sum(dim=(0, 3)).T
    pt.testing.assert_close(
        attention.alpha.grad, expected_gradient[:, None].expand(-1, 3, -1)
    )


class OldestPlusNewest(AutoregressiveModel):
    def __init__(self):
        super().__init__(1, history_length=3)

    def forward(self, history, parameters=None):
        return history[:, 0] + history[:, -1]


def test_free_running_window_shift_and_analytic_temporal_gradient():
    model = OldestPlusNewest()
    history = pt.tensor([[[1.0], [2.0], [3.0]]], requires_grad=True)
    predicted = model.rollout(history, 3)
    pt.testing.assert_close(predicted, pt.tensor([[[4.0], [6.0], [9.0]]]))
    predicted[:, -1].sum().backward()
    pt.testing.assert_close(history.grad, pt.tensor([[[1.0], [1.0], [2.0]]]))


def test_original_latent_mlp_state_dict_and_call_convention_are_preserved():
    old_network = nn.Sequential(nn.Linear(5, 7), nn.SiLU(), nn.Linear(7, 3))
    model = LatentDynamics(3, 2, 7)
    model.load_state_dict(
        {"network." + key: value for key, value in old_network.state_dict().items()}
    )
    latent, parameters = pt.randn(2, 3), pt.randn(2, 2)
    expected = latent + old_network(pt.cat((latent, parameters), dim=1))
    pt.testing.assert_close(model(latent, parameters), expected)
    pt.testing.assert_close(model(latent[:, None], parameters), expected)


@pytest.mark.parametrize("model_type", MODELS)
def test_predictors_work_inside_conditional_vae_training(model_type):
    autoencoder = ConvVAE(
        (1, 5, 7), 3, history=2, conv_dim=3, channels=(2,), parameter_dim=1
    )
    dynamics = model_type(3, history_length=3, parameter_dim=1)
    model = ConvVAEROM(autoencoder, dynamics=dynamics)
    assert model.history_length == 4
    dataset = ImageSequenceDataset(
        [pt.randn(8, 1, 5, 7)],
        history=model.history_length,
        horizon=2,
        parameters=[pt.tensor([0.3])],
    )
    batches = DataLoader(dataset, batch_size=2)
    batch = next(iter(batches))
    # Batched window encoding must match individually encoded, ordered histories.
    mean, variance = model._encode_context(batch["history"], batch["parameters"])
    for i in range(3):
        expected, logvar = autoencoder.encode(
            batch["history"][:, i : i + 2], batch["parameters"]
        )
        pt.testing.assert_close(mean[:, i], expected)
        pt.testing.assert_close(variance[:, i], logvar)
    ae_before = deepcopy(autoencoder.state_dict())
    model.fine_tune(batches, stage="dynamics", epochs=1, horizons=(1, 2))
    for key, value in ae_before.items():
        pt.testing.assert_close(autoencoder.state_dict()[key], value)
    model.fine_tune(batches, stage="joint", epochs=1, horizons=(2,))
    assert any(
        not pt.equal(autoencoder.state_dict()[key], value)
        for key, value in ae_before.items()
    )
    model.eval()
    prediction = model.predict(
        batch["history"], time=pt.arange(3.0).float(), parameters=batch["parameters"]
    )
    pt.testing.assert_close(
        prediction.mean[:, 1:], model.rollout(batch["history"], 2, batch["parameters"])
    )
    assert prediction.mean.shape == (2, 3, 1, 5, 7)


def test_vae_target_histories_are_aligned_without_teacher_forcing():
    class PixelEncoder(ConvVAE):
        def __init__(self):
            super().__init__((1, 1, 1), 1, history=2, conv_dim=3, channels=(1,))

        def encode(self, images, parameters=None):
            mean = images[:, -1].reshape(-1, 1)
            return mean, pt.zeros_like(mean)

        def decode(self, latent, parameters=None):
            return latent.reshape(-1, 1, 1, 1)

    model = ConvVAEROM(PixelEncoder(), dynamics=OldestPlusNewest())
    history = pt.arange(4.0).reshape(1, 4, 1, 1, 1)
    future = pt.tensor([4.0, 5.0, 6.0]).reshape(1, 3, 1, 1, 1)
    losses = model.loss(history, future, stage="dynamics")
    # Encoded context is [1,2,3], forecasts [4,6,9], targets [4,5,6].
    pt.testing.assert_close(losses["latent"], pt.tensor(10.0 / 3))
    pt.testing.assert_close(losses["rollout"], pt.tensor(10.0 / 3))


@pytest.mark.parametrize("model_type", MODELS)
def test_small_standalone_training_reduces_manufactured_forecast_error(model_type):
    model = model_type(2, history_length=3)
    history = pt.randn(8, 3, 2) * 0.2
    future = pt.stack((history[:, -1] + 0.1, history[:, -1] + 0.2), dim=1)
    initial = model.rollout_loss(history, future).detach()
    optimizer = pt.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(35):
        optimizer.zero_grad()
        loss = model.rollout_loss(history, future)
        loss.backward()
        optimizer.step()
    assert model.rollout_loss(history, future).detach() < 0.15 * initial


def test_invalid_predictor_configuration_and_dimension_mismatch():
    with pytest.raises(ValueError, match="divisible"):
        EasyAttentionAutoregressor(2, model_dim=7, num_heads=2)
    with pytest.raises(ValueError, match="history_length"):
        LSTMAutoregressor(2, history_length=0)
    with pytest.raises(ValueError, match="dimensions"):
        ConvVAEROM(ConvVAE((1, 3, 5), 2), dynamics=GRUAutoregressor(3))


@pytest.mark.parametrize("model_type", MODELS)
@pytest.mark.parametrize("residual", [False, True])
def test_zero_predictor_distinguishes_absolute_and_increment_readout(
    model_type, residual
):
    model = model_type(2, history_length=3, residual=residual)
    with pt.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    history = pt.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    expected = history[:, -1] if residual else pt.zeros(1, 2)
    pt.testing.assert_close(model(history), expected)


def test_causal_temporal_features_cannot_see_future_samples():
    temporal = CausalTemporalConv(2, channels=(3, 4), dilations=(1, 3)).double()
    inputs = pt.randn(2, 13, 2, dtype=pt.float64)
    altered = inputs.clone()
    altered[:, 7:] += 100.0
    features = temporal(inputs)
    assert features.shape == (2, 13, 4)
    pt.testing.assert_close(features[:, :7], temporal(altered)[:, :7], rtol=0, atol=0)
    pt.testing.assert_close(features[:, :7], temporal(inputs[:, :7]))
    assert not pt.equal(features[:, 7:], temporal(altered)[:, 7:])


def test_temporal_receptive_field_matches_analytic_gradient_support():
    temporal = CausalTemporalConv(1, channels=(1, 1, 1), kernel_size=3)
    assert temporal.receptive_field == 15
    with pt.no_grad():
        for parameter in temporal.parameters():
            parameter.fill_(0.1)
    history = pt.ones(1, 21, 1, requires_grad=True)
    temporal(history)[:, -1].sum().backward()
    pt.testing.assert_close(history.grad[:, :6], pt.zeros(1, 6, 1), rtol=0, atol=0)
    assert bool((history.grad[:, 6:] > 0).all())


@pytest.mark.parametrize(
    "options",
    [
        dict(channels=()),
        dict(dilations=(1,)),
        dict(kernel_size=0),
        dict(channels=(0,)),
        dict(dilations=(1, 2, 0)),
    ],
)
def test_invalid_temporal_feature_configuration(options):
    with pytest.raises(ValueError):
        CausalTemporalConv(2, **options)


def test_tcn_predictor_rejects_history_beyond_receptive_field():
    with pytest.raises(ValueError, match="receptive field"):
        TCNMLPAutoregressor(3, history_length=64)
    model = TCNMLPAutoregressor(3, history_length=64, conv_channels=(4,) * 6)
    assert model.temporal.receptive_field == 127
    assert model(pt.randn(2, 64, 3)).shape == (2, 3)


@pytest.mark.parametrize(
    "activation", ["relu", "silu", "gelu", "tanh", "elu", "leaky_relu", "identity"]
)
@pytest.mark.parametrize("convs", [1, 2])
def test_configurable_temporal_blocks_remain_causal(activation, convs):
    temporal = (
        CausalTemporalConv(
            2,
            channels=(3, 3),
            convs_per_block=convs,
            activation=activation,
            dropout=0.2,
            weight_normalization=True,
            activate_residual=True,
        )
        .double()
        .eval()
    )
    history = pt.randn(2, 11, 2, dtype=pt.float64)
    changed = history.clone()
    changed[:, 6:] += 20.0
    expected = temporal(history)
    pt.testing.assert_close(expected[:, :6], temporal(changed)[:, :6], rtol=0, atol=0)
    pt.testing.assert_close(expected[:, :6], temporal(history[:, :6]))
    assert temporal.receptive_field == 1 + convs * 2 * (1 + 2)


def test_multiple_convolutions_extend_exact_gradient_support():
    temporal = CausalTemporalConv(
        1, channels=(1, 1), convs_per_block=2, activation="identity"
    )
    assert temporal.receptive_field == 13
    with pt.no_grad():
        for parameter in temporal.parameters():
            parameter.fill_(0.1)
    x = pt.ones(1, 18, 1, requires_grad=True)
    temporal(x)[:, -1].sum().backward()
    pt.testing.assert_close(x.grad[:, :5], pt.zeros(1, 5, 1), rtol=0, atol=0)
    assert (x.grad[:, 5:] > 0).all()


def test_weight_normalization_matches_filter_formula_and_survives_checkpoint():
    model = TCNMLPAutoregressor(
        2,
        history_length=4,
        conv_channels=(3, 3),
        hidden_dim=None,
        convs_per_block=2,
        activation="relu",
        weight_normalization=True,
        dropout=0.3,
        activate_residual=True,
    ).double()
    convolution = model.temporal.blocks[0].convolution
    assert hasattr(convolution, "parametrizations")
    parametrization = convolution.parametrizations.weight
    with pt.no_grad():
        parametrization.original0.fill_(2.0)
        parametrization.original1.copy_(
            pt.arange(1.0, parametrization.original1.numel() + 1).reshape_as(
                parametrization.original1
            )
        )
    v = parametrization.original1
    expected = (
        parametrization.original0 * v / pt.sqrt(v.square().sum((1, 2), keepdim=True))
    )
    pt.testing.assert_close(convolution.weight, expected)
    history = pt.randn(2, 4, 2, dtype=pt.float64)
    future = pt.randn(2, 2, 2, dtype=pt.float64)
    optimizer = pt.optim.Adam(model.parameters(), lr=0.001)
    loss = model.rollout_loss(history, future)
    loss.backward()
    assert all(
        p.grad is not None and pt.isfinite(p.grad).all() for p in model.parameters()
    )
    optimizer.step()
    model.eval()
    restored = deepcopy(model)
    restored.load_state_dict(model.state_dict())
    pt.testing.assert_close(model.rollout(history, 2), restored.rollout(history, 2))
    assert len(model.network) == 1  # Linear readout, as in authors' adding task.


def test_dropout_switches_off_in_eval_mode():
    feature = CausalTemporalConv(2, channels=(6, 6), convs_per_block=2, dropout=0.5)
    history = pt.randn(4, 12, 2)
    feature.train()
    assert not pt.equal(feature(history), feature(history))
    feature.eval()
    pt.testing.assert_close(feature(history), feature(history), rtol=0, atol=0)


def test_paper_style_block_matches_independent_symmetric_padding_and_crop():
    # Independent padding path equivalent to the authors' conv/chomp/ReLU block.
    module = CausalTemporalConv(
        2,
        channels=(3,),
        kernel_size=3,
        dilations=(2,),
        convs_per_block=2,
        activation="relu",
        dropout=0.0,
        activate_residual=True,
    ).double()
    x = pt.randn(2, 7, 2, dtype=pt.float64)
    block = module.blocks[0]
    result = x.transpose(1, 2)
    for convolution in [block.convolution, *block.additional_convolutions]:
        result = pt.nn.functional.conv1d(
            result, convolution.weight, convolution.bias, padding=4, dilation=2
        )[..., :-4]
        result = pt.relu(result)
    expected = pt.relu(result + block.shortcut(x.transpose(1, 2))).transpose(1, 2)
    pt.testing.assert_close(module(x), expected)


@pytest.mark.parametrize(
    "options",
    [
        dict(convs_per_block=0),
        dict(activation="unknown"),
        dict(dropout=-0.1),
        dict(dropout=1.1),
        dict(dropout=float("nan")),
        dict(weight_normalization="yes"),
        dict(activate_residual=1),
    ],
)
def test_new_temporal_options_are_validated(options):
    with pytest.raises(ValueError):
        CausalTemporalConv(2, **options)
