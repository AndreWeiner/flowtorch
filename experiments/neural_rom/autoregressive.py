"""Standalone conditional vector forecasting; no VAE or external data required."""

import torch

from flowtorch.rom import LSTMAutoregressor


def main():
    torch.manual_seed(11)
    predictor = LSTMAutoregressor(
        state_dim=3,
        history_length=5,
        parameter_dim=2,
        hidden_dim=16,
    )
    # Replace these with normalized training windows, e.g. POD coefficients.
    history = torch.randn(8, 5, 3)
    future = torch.randn(8, 3, 3)
    conditions = torch.randn(8, 2)  # Fixed per trajectory, e.g. scaled Mach/alpha.
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1.0e-3)
    for _ in range(3):
        optimizer.zero_grad()
        loss = predictor.rollout_loss(history, future, conditions)
        loss.backward()
        optimizer.step()
    predictor.eval()
    with torch.no_grad():
        prediction = predictor.rollout(history, steps=10, parameters=conditions)
    print("Standalone forecast shape:", tuple(prediction.shape))


if __name__ == "__main__":
    main()
