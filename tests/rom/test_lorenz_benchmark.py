"""Independent small checks of the executable documentation benchmark."""

import importlib.util
from pathlib import Path

import numpy as np
import torch

BENCHMARK = (
    Path(__file__).resolve().parents[2] / "experiments/neural_rom/lorenz_benchmark.py"
)
spec = importlib.util.spec_from_file_location("lorenz_benchmark", BENCHMARK)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_reference_equilibria_and_rhs():
    np.testing.assert_allclose(benchmark.rhs(0, [1.0, 2.0, 3.0]), [10.0, 23.0, -6.0])
    equilibrium = np.array([np.sqrt(72.0), np.sqrt(72.0), 27.0])
    np.testing.assert_allclose(
        benchmark.integrate(equilibrium, 101),
        np.tile(equilibrium, (101, 1)),
        rtol=1.0e-9,
        atol=1.0e-9,
    )


def test_sample_windows_are_aligned_and_do_not_cross_series():
    data = torch.arange(60.0).reshape(3, 20, 1)
    history, future = benchmark.windows(
        data, 4, 3, 50, torch.Generator().manual_seed(7)
    )
    sequence = torch.cat((history, future), dim=1)
    torch.testing.assert_close(sequence.diff(dim=1), torch.ones(50, 6, 1))
    assert torch.equal(sequence[:, 0, 0] // 20, sequence[:, -1, 0] // 20)


def test_metric_units_and_right_censoring():
    class Constant:
        def __init__(self, value):
            self.value = value

        def eval(self):
            pass

        def rollout(self, history, steps):
            return torch.full((history.shape[0], steps, 3), self.value)

    config = {"history": 2, "test_steps": 3, "dt": 0.01}
    data = {"test": torch.zeros(2, 5, 3)}
    perfect, _ = benchmark.evaluate(Constant(0.0), data, {}, config, long=False)
    assert perfect["horizon_mean"] == 0.03
    assert perfect["horizon_right_censored"] == 2
    assert perfect["short_rmse_first_100"] == 0.0
    failed, _ = benchmark.evaluate(Constant(1.0), data, {}, config, long=False)
    assert failed["horizon_mean"] == 0.01
    assert failed["horizon_right_censored"] == 0
    assert failed["short_rmse_first_100"] == 1.0
