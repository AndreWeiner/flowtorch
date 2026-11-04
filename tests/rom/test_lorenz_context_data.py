"""Controlled contexts, nested data budgets and shared held-out forecast origins."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

EXAMPLES = Path(__file__).resolve().parents[2] / "experiments/neural_rom"
sys.path.insert(0, str(EXAMPLES))
try:
    spec = importlib.util.spec_from_file_location(
        "lorenz_context_data", EXAMPLES / "lorenz_context_data.py"
    )
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
finally:
    sys.path.pop(0)


def test_context_changes_preserve_training_forecast_origin_and_future():
    data = torch.arange(240.0).reshape(3, 80, 1).expand(-1, -1, 3)
    h1, y1 = study.sample_windows(data, 1, 4, 30, torch.Generator().manual_seed(5), 16)
    h8, y8 = study.sample_windows(data, 8, 4, 30, torch.Generator().manual_seed(5), 16)
    torch.testing.assert_close(h1, h8[:, -1:])
    torch.testing.assert_close(y1, y8)
    torch.testing.assert_close(y8[:, 0] - h8[:, -1], torch.ones(30, 3))
    assert torch.equal((h8[:, 0, 0] // 80), (y8[:, -1, 0] // 80))


def test_nested_budgets_and_fixed_state_counts(tmp_path):
    np.savez(
        tmp_path / "training_pool_0.npz",
        **{
            f"source_{i}": np.arange(300.0).reshape(100, 3) + i * 1000 for i in range(4)
        },
    )
    config = {"mean": [0.0, 0.0, 27.0], "scale": [10.0, 10.0, 10.0]}
    small = {"budget": 80, "count": 4, "guard": 4}
    large = {"budget": 160, "count": 4, "guard": 4}
    a = study.training_data(tmp_path, small, 0, config)
    b = study.training_data(tmp_path, large, 0, config)
    assert a.shape == (4, 20, 3)
    torch.testing.assert_close(a, b[:, :20])
    assert study.eligible_windows(small) == 4 * (20 - 4 - 4 + 1)
    # Changing trajectory allocation preserves the total retained state count.
    few = study.training_data(tmp_path, {**small, "count": 1}, 0, config)
    assert few.numel() == a.numel()


def test_validation_contexts_share_targets(tmp_path):
    config = {**study.CONFIG, "contexts": [1, 8], "validation_steps": 10}
    array = np.arange(2 * 60 * 3).reshape(2, 60, 3).astype(float)
    np.savez(
        tmp_path / "held_out.npz",
        validation=array,
        test=array,
        validation_series=np.array([0, 1]),
        validation_origins=np.array([12, 25]),
    )
    h1, y1, t1, ty1 = study.held_out(tmp_path, 1, config)
    h8, y8, t8, ty8 = study.held_out(tmp_path, 8, config)
    torch.testing.assert_close(h1, h8[:, -1:])
    torch.testing.assert_close(y1, y8)
    torch.testing.assert_close(t1, t8[:, -1:])
    torch.testing.assert_close(ty1, ty8)
    expected = study.normalize(array[1, 25:35], config)
    torch.testing.assert_close(y8[1], expected)


def test_sweep_size_and_labels():
    cells = study.cells(study.CONFIG)
    assert len(cells) == 13
    assert len({study.cell_id(cell) for cell in cells}) == 13
    assert len(cells) * 4 * 3 * 3 == 468
    assert all(study.eligible_windows(cell) > 0 for cell in cells)
