"""Helpers shared by the flowTorch test suite."""

import pytest

from flowtorch import DATASETS


def requires_datasets(*dataset_names):
    """Skip a test module when required external datasets are unavailable."""
    missing = (
        [name for name in dataset_names if name not in DATASETS]
        if dataset_names else ([] if DATASETS else ["<any dataset>"])
    )
    reason = (
        "requires FLOWTORCH_DATASETS with datasets: "
        + ", ".join(missing or dataset_names)
    )
    marker = pytest.mark.skipif(bool(missing), reason=reason)
    if missing:
        pytest.skip(reason, allow_module_level=True)
    return marker
