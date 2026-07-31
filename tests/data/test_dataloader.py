"""Tests for shared dataloader utilities."""

import pytest
import torch as pt

from flowtorch.data.dataloader import _preallocate_time_series


def test_preallocate_time_series_preserves_values_and_time_last():
    snapshots = {
        "0": pt.arange(6, dtype=pt.float32).reshape(2, 3),
        "1": pt.arange(6, 12, dtype=pt.float32).reshape(2, 3),
    }

    series = _preallocate_time_series(snapshots.__getitem__, ["0", "1"])

    assert series.shape == (2, 3, 2)
    assert series.is_contiguous()
    assert pt.equal(series[..., 0], snapshots["0"])
    assert pt.equal(series[..., 1], snapshots["1"])


def test_preallocate_time_series_rejects_empty_times():
    with pytest.raises(ValueError, match="At least one snapshot"):
        _preallocate_time_series(lambda time: pt.tensor([time]), [])


def test_preallocate_time_series_rejects_shape_changes():
    snapshots = {"0": pt.zeros(2), "1": pt.zeros(3)}

    with pytest.raises(ValueError, match="same shape"):
        _preallocate_time_series(snapshots.__getitem__, ["0", "1"])
