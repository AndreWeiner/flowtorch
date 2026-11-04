"""Contracts for conditional image compression and free-running evolution."""

from copy import deepcopy

import pytest
import torch as pt
from torch import nn
from torch.utils.data import DataLoader

from flowtorch.rom import ConvVAE, ConvVAEROM, ImageSequenceDataset


@pytest.fixture(autouse=True)
def small_network_threads():
    previous = pt.get_num_threads()
    pt.set_num_threads(1)
    pt.manual_seed(8)
    yield
    pt.set_num_threads(previous)


@pytest.mark.parametrize("shape", [(1, 9, 13), (2, 8, 11), (1, 1, 3)])
@pytest.mark.parametrize("conv_dim,history", [(2, 1), (3, 3)])
def test_odd_shapes_precision_gradients_and_sampling(shape, conv_dim, history):
    vae = ConvVAE(
        shape, 3, channels=(3, 5), history=history, conv_dim=conv_dim, parameter_dim=2
    ).double()
    images = pt.randn(2, history, *shape, dtype=pt.float64, requires_grad=True)
    parameters = pt.randn(2, 2, dtype=pt.float64, requires_grad=True)
    reconstruction, mean, logvar = vae(images, parameters)
    assert reconstruction.shape == (2, *shape)
    assert reconstruction.dtype == pt.float64
    assert mean.shape == logvar.shape == (2, 3)
    reconstruction.square().mean().backward()
    assert images.grad.abs().sum() > 0
    assert parameters.grad.abs().sum() > 0
    # In 3D mode the oldest frame also contributes to the newest-frame state.
    assert images.grad[:, 0].abs().sum() > 0
    vae.eval()
    pt.testing.assert_close(vae(images, parameters)[0], vae(images, parameters)[0])
    assert not pt.equal(
        vae(images, parameters, sample=True)[0], vae(images, parameters, sample=True)[0]
    )


def test_conditions_reach_all_components():
    vae = ConvVAE((1, 5, 7), 2, channels=(3,), parameter_dim=2)
    model = ConvVAEROM(vae)
    images = pt.randn(2, 1, 1, 5, 7)
    parameters = pt.tensor([[0.0, 0.0], [1.0, 2.0]])
    repeated = images[:1].expand_as(images)
    mean, _ = vae.encode(repeated, parameters)
    assert not pt.equal(mean[0], mean[1])
    z = pt.ones(2, 2)
    assert not pt.equal(vae.decode(z, parameters)[0], vae.decode(z, parameters)[1])
    assert not pt.equal(
        model.dynamics(z, parameters)[0], model.dynamics(z, parameters)[1]
    )
    with pytest.raises(ValueError, match="parameters"):
        model.rollout(images, 1)
    with pytest.raises(ValueError, match="parameters"):
        model.rollout(images, 1, parameters[:, None])
    with pytest.raises(ValueError, match="unconditional"):
        ConvVAE((1, 5, 7), 2).encode(images, parameters)


def test_windows_preserve_boundaries_and_fixed_parameters():
    trajectories = [
        pt.arange(8.0).reshape(8, 1, 1, 1),
        pt.arange(100.0, 107.0).reshape(7, 1, 1, 1),
    ]
    parameters = [pt.tensor([0.7, 2.0]), pt.tensor([0.8, 4.0])]
    dataset = ImageSequenceDataset(
        trajectories, parameters=parameters, history=3, horizon=2
    )
    assert len(dataset) == 7
    pt.testing.assert_close(dataset[3]["history"].flatten(), pt.tensor([3.0, 4.0, 5.0]))
    pt.testing.assert_close(dataset[3]["future"].flatten(), pt.tensor([6.0, 7.0]))
    pt.testing.assert_close(
        dataset[4]["history"].flatten(), pt.tensor([100.0, 101.0, 102.0])
    )
    pt.testing.assert_close(dataset[-1]["parameters"], parameters[1])
    assert dataset[0]["history"].data_ptr() == trajectories[0].data_ptr()
    with pytest.raises(IndexError):
        dataset[len(dataset)]
    with pytest.raises(ValueError, match="fixed vectors"):
        ImageSequenceDataset(trajectories, parameters=[pt.ones(8, 2), pt.ones(7, 2)])
    with pytest.raises(ValueError, match=r"history \+ horizon"):
        ImageSequenceDataset(trajectories, history=8, horizon=1)


class Increment(nn.Module):
    def forward(self, latent, parameters=None):
        return latent + 1 if parameters is None else latent + parameters[:, :1]


def test_rollout_is_latent_autoregression_and_predict_uses_relative_time():
    vae = ConvVAE((1, 3, 5), 2, channels=(2,), parameter_dim=1)
    model = ConvVAEROM(vae, dt=0.25, dynamics=Increment())
    images = pt.randn(2, 1, 1, 3, 5)
    parameters = pt.tensor([[0.2], [0.4]])
    z, _ = vae.encode(images, parameters)
    expected = pt.stack(
        [vae.decode(z + k * parameters, parameters) for k in range(1, 4)], dim=1
    )
    pt.testing.assert_close(model.rollout(images, 3, parameters), expected)
    result = model.predict(images, time=pt.arange(4) * 0.25, parameters=parameters)
    pt.testing.assert_close(result.mean[:, 1:], expected)
    pt.testing.assert_close(result.mean[:, 0], vae.decode(z, parameters))
    assert result.scale is None
    for time in (
        pt.tensor([1.0, 1.25]),
        pt.tensor([0.0, 0.5]),
        pt.tensor([0.0, float("nan")]),
    ):
        with pytest.raises(ValueError, match="relative"):
            model.predict(images, time=time, parameters=parameters)
    with pytest.raises(ValueError, match="forcing"):
        model.predict(
            images, time=pt.zeros(1), parameters=parameters, forcing=pt.ones(1)
        )


def test_kl_and_mask_reductions_against_closed_form():
    vae = ConvVAE((1, 3, 5), 2, channels=(2,))
    model = ConvVAEROM(vae).eval()
    with pt.no_grad():
        for p in model.parameters():
            p.zero_()
        vae.posterior.bias[:2].fill_(2.0)
        vae.posterior.bias[2:].fill_(pt.log(pt.tensor(4.0)))
    image = pt.arange(15.0).reshape(1, 1, 1, 3, 5)
    mask = pt.zeros(1, 3, 5)
    mask[0, 0, :2] = pt.tensor([1.0, 3.0])
    losses = model.loss(image, stage="autoencoder", beta=0.3, mask=mask)
    expected_kl = 4.0 + 4.0 - 1.0 - pt.log(pt.tensor(4.0))
    pt.testing.assert_close(losses["kl"], expected_kl)
    pt.testing.assert_close(losses["reconstruction"], pt.tensor(0.75))
    pt.testing.assert_close(losses["loss"], 0.75 + 0.3 * expected_kl)
    with pytest.raises(ValueError, match="mask"):
        model.loss(image, stage="autoencoder", mask=pt.zeros(1, 3, 5))


def make_training():
    data = ImageSequenceDataset([pt.randn(8, 1, 5, 7)], history=2, horizon=2)
    model = ConvVAEROM(ConvVAE((1, 5, 7), 3, channels=(2,), history=2, conv_dim=3))
    return model, DataLoader(data, batch_size=3)


@pytest.mark.parametrize("stage", ["autoencoder", "dynamics", "joint"])
def test_training_freezes_requested_components_and_restores_modes(stage):
    model, batches = make_training()
    before = deepcopy(model.state_dict())
    model.eval()
    model.autoencoder.encoder.train()  # Preserve mixed submodule modes too.
    modes = [m.training for m in model.modules()]
    log = model.fine_tune(batches, stage=stage, epochs=2, horizons=(1, 2))
    assert [entry["horizon"] for entry in log] == [1.0, 1.0, 2.0, 2.0]
    assert all(pt.isfinite(pt.tensor(entry["loss"])) for entry in log)
    assert [m.training for m in model.modules()] == modes
    assert all(p.requires_grad for p in model.parameters())
    after = model.state_dict()
    ae_changes = any(
        not pt.equal(before[k], after[k])
        for k in before
        if k.startswith("autoencoder.")
    )
    dyn_changes = any(
        not pt.equal(before[k], after[k]) for k in before if k.startswith("dynamics.")
    )
    assert ae_changes == (stage != "dynamics")
    assert dyn_changes == (stage != "autoencoder")
    # State dict alone reproduces deterministic inference with the same config.
    restored, _ = make_training()
    restored.load_state_dict(after)
    batch = next(iter(batches))
    pt.testing.assert_close(
        model.rollout(batch["history"], 2), restored.rollout(batch["history"], 2)
    )


def test_training_failure_restores_flags_and_modes():
    model, batches = make_training()
    model.eval()
    with pytest.raises(ValueError, match="shorter"):
        model.fine_tune(batches, stage="dynamics", horizons=(3,))
    assert not model.training
    assert all(p.requires_grad for p in model.parameters())
    with pytest.raises(ValueError, match="re-iterable"):
        model.fine_tune(iter(batches))
    with pytest.raises(ValueError, match="nondecreasing"):
        model.fine_tune(batches, horizons=(2, 1))


def test_free_running_loss_has_multistep_gradients():
    model, batches = make_training()
    batch = next(iter(batches))
    # Last-step-only target perturbation must influence the dynamics gradient.
    model.eval()
    losses = model.loss(**batch, latent_weight=0.0)
    grad = pt.autograd.grad(losses["rollout"], tuple(model.dynamics.parameters()))
    future = batch["future"].clone()
    future[:, -1] += 1.0
    altered = model.loss(batch["history"], future, latent_weight=0.0)
    changed = pt.autograd.grad(altered["rollout"], tuple(model.dynamics.parameters()))
    assert any(not pt.allclose(a, b) for a, b in zip(grad, changed))


def test_small_reconstruction_problem_can_be_learned():
    # Numerical learning check on a simple manufactured field, not PSP validation.
    data = ImageSequenceDataset([pt.ones(5, 1, 5, 7)])
    batches = DataLoader(data, batch_size=4)
    model = ConvVAEROM(ConvVAE((1, 5, 7), 2, channels=(3,)))
    model.eval()
    batch = next(iter(batches))
    initial = model.loss(**batch, stage="autoencoder", beta=0.0)[
        "reconstruction"
    ].item()
    model.fine_tune(batches, stage="autoencoder", epochs=60, lr=0.02, beta=0.0)
    final = model.loss(**batch, stage="autoencoder", beta=0.0)["reconstruction"].item()
    assert final < 0.1 * initial


@pytest.mark.parametrize(
    "kwargs",
    [
        {"conv_dim": 2, "history": 2},
        {"channels": ()},
        {"latent_dim": 0},
        {"parameter_dim": -1},
    ],
)
def test_invalid_architecture(kwargs):
    options = {"latent_dim": 2, **kwargs}
    with pytest.raises(ValueError):
        ConvVAE((1, 5, 7), **options)


def test_two_step_gradient_matches_analytic_chain_rule():
    class Scale(nn.Module):
        def __init__(self):
            super().__init__()
            self.factor = nn.Parameter(pt.tensor(0.8, dtype=pt.float64))

        def forward(self, latent, parameters=None):
            return self.factor * latent

    # A one-layer decoder is affine, allowing a closed-form derivative.
    vae = ConvVAE((1, 3, 5), 2, channels=(2,)).double()
    dynamics = Scale()
    model = ConvVAEROM(vae, dynamics=dynamics)
    history = pt.randn(1, 1, 1, 3, 5, dtype=pt.float64)
    z, _ = vae.encode(history)
    slope = (vae.decode(z) - vae.decode(pt.zeros_like(z))).sum().detach()
    last_frame = model.rollout(history, 2)[:, -1].sum()
    (derivative,) = pt.autograd.grad(last_frame, dynamics.factor)
    pt.testing.assert_close(derivative, 2 * dynamics.factor.detach() * slope)


def test_validation_plateau_advances_horizon_and_keeps_modes():
    model, batches = make_training()
    model.eval()
    # An intentionally huge minimum improvement forces a reproducible plateau.
    log = model.fine_tune(
        batches,
        stage="dynamics",
        epochs=5,
        horizons=(1, 2),
        validation_batches=batches,
        patience=1,
        min_delta=1.0e9,
    )
    assert [entry["horizon"] for entry in log] == [1.0, 1.0, 2.0, 2.0]
    assert all("validation_loss" in entry for entry in log)
    assert not model.training
    with pytest.raises(ValueError, match="requires validation_batches"):
        model.fine_tune(batches, patience=2)


def test_surface_quadrature_weights_apply_to_reconstruction_and_forecast():
    from flowtorch.data import grid_element_areas, element_areas_to_node_weights

    # Two rectangular surface cells have areas 2 and 6, with nodal Cp samples.
    x, y = pt.meshgrid(pt.tensor([0.0, 1.0, 4.0]), pt.tensor([0.0, 2.0]), indexing="ij")
    z = pt.zeros_like(x)
    areas = grid_element_areas(x, y, z)
    weights = element_areas_to_node_weights(areas, normalize=False, square_root=False)
    pt.testing.assert_close(areas, pt.tensor([[2.0], [6.0]]))
    pt.testing.assert_close(weights.sum(), pt.tensor(8.0))
    model = ConvVAEROM(ConvVAE((1, 3, 2), 2, channels=(2,))).eval()
    with pt.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    images = pt.tensor([2.0, 2.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 1, 1, 3, 2)
    mask = weights.unsqueeze(0)
    losses = model.loss(images, images, mask=mask, latent_weight=0.0)
    pt.testing.assert_close(losses["reconstruction"], pt.tensor(0.5))
    pt.testing.assert_close(losses["rollout"], pt.tensor(0.5))
    scaled = model.loss(images, images, mask=17 * mask, latent_weight=0.0)
    pt.testing.assert_close(losses["loss"], scaled["loss"])
    unweighted = model.loss(images, images, latent_weight=0.0)
    pt.testing.assert_close(unweighted["reconstruction"], pt.tensor(4.0 / 3))
