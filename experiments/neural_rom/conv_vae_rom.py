"""Dataset-free conditional ROM example; the random fields have no flow physics."""

import torch
from torch.utils.data import DataLoader

from flowtorch.rom import ConvVAE, ConvVAEROM, ImageSequenceDataset


def main():
    torch.manual_seed(7)
    # In a PSP application replace these with separately loaded Cp trajectories.
    trajectories = [torch.randn(18, 1, 15, 21), torch.randn(16, 1, 15, 21)]
    # Already scaled conditions for the example: [Mach, angle of attack].
    conditions = [torch.tensor([-0.5, -0.8]), torch.tensor([0.5, 0.8])]
    dataset = ImageSequenceDataset(
        trajectories, parameters=conditions, history=3, horizon=4
    )
    batches = DataLoader(dataset, batch_size=4, shuffle=True)
    autoencoder = ConvVAE(
        (1, 15, 21),
        latent_dim=6,
        channels=(4, 8),
        conv_dim=3,
        history=3,
        parameter_dim=2,
    )
    model = ConvVAEROM(autoencoder, dt=0.001)
    # Short demonstration budgets; select real budgets on held-out data.
    model.fine_tune(batches, stage="autoencoder", epochs=1)
    model.fine_tune(batches, stage="dynamics", epochs=1)
    log = model.fine_tune(batches, stage="joint", epochs=1, horizons=(1, 2, 4))
    model.eval()
    with torch.no_grad():
        future = model.rollout(
            trajectories[0][:3].unsqueeze(0), 8, conditions[0].unsqueeze(0)
        )
    print("Future shape:", tuple(future.shape))
    print("Joint training horizons:", [entry["horizon"] for entry in log])


if __name__ == "__main__":
    main()
