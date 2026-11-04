"""Check dataset-level aggregation used by the sensitivity figures."""

import importlib.util
from pathlib import Path
import sys

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[2] / "experiments/neural_rom"


def test_aggregation_preserves_data_replication_level():
    sys.path.insert(0, str(EXAMPLES))
    try:
        spec = importlib.util.spec_from_file_location(
            "plot_lorenz_context_data", EXAMPLES / "plot_lorenz_context_data.py"
        )
        plots = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plots)
    finally:
        sys.path.pop(0)
    rows = [
        dict(replicate=0, score=1.0),
        dict(replicate=0, score=3.0),
        dict(replicate=1, score=10.0),
    ]
    # Equal weight per independent data realization, not per number of rows.
    np.testing.assert_equal(plots.seed_groups(rows, "score"), np.array([2.0, 10.0]))
