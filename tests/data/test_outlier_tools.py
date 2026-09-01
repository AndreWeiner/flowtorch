# third party packages
import pytest
import torch as pt

# flowtorch packages
from flowtorch.data import (
    iqr_outlier_replacement,
    replace_spatial_outliers,
    replace_temporal_outliers,
)


def test_irq_outlier_replacement():
    data = pt.tensor([[3.0, 2.0, 4.0, 8.0, 1.0, 0.0], [3.0, 2.0, 4.0, 5.0, 1.0, 0.0]])
    with pytest.warns(DeprecationWarning, match="replace_temporal_outliers"):
        clean_data = iqr_outlier_replacement(data)
    # the shape of both datasets should be equal
    assert clean_data.shape == data.shape
    # check if outlier is detected and replaced;
    # the number of elements in the second direction
    # is even -> PyTorch returns the lower median
    assert clean_data[0][3].item() == 2.0
    # decrease sensitivity
    with pytest.warns(DeprecationWarning):
        data_clean = iqr_outlier_replacement(data, k=2.0)
    assert data_clean[0][3] == 8.0
    # use only the two nearest neighbors
    with pytest.warns(DeprecationWarning):
        data_clean = iqr_outlier_replacement(data, nb=1)
    assert data_clean[0][3] == 4.0
    # test with 1D tensor
    data = pt.tensor([3.0, 2.0, 4.0, 8.0, 1.0, 0.0])
    with pytest.warns(DeprecationWarning):
        data_clean = iqr_outlier_replacement(data)
    assert len(data_clean.shape) == 1
    assert data_clean[3] == 2.0


def test_replace_temporal_outliers():
    data = pt.ones((2, 9))
    data[0, 4] = 100.0
    data[1, 5] = -100.0

    clean = replace_temporal_outliers(data)

    assert pt.equal(clean, pt.ones_like(data))
    assert clean.shape == data.shape
    assert clean.data_ptr() != data.data_ptr()


def test_replace_temporal_outliers_handles_boundaries_and_trends():
    data = pt.stack(
        (
            pt.tensor([100.0, 1.0, 1.0, 1.0, 1.0]),
            pt.arange(5.0),
        )
    )

    clean = replace_temporal_outliers(data)

    assert clean[0, 0] == 1.0
    pt.testing.assert_close(clean[1], data[1])


def test_replace_temporal_outliers_processes_leading_dimensions_independently():
    data = pt.ones((2, 3, 7))
    data[0, 1, 3] = 100.0
    data[1, 2] *= 5.0

    clean = replace_temporal_outliers(data)

    assert clean[0, 1, 3] == 1.0
    pt.testing.assert_close(clean[1, 2], data[1, 2])


def test_replace_temporal_outliers_preserves_nonfinite_values():
    data = pt.tensor([1.0, 1.0, float("nan"), 1.0, float("inf"), 1.0])

    clean = replace_temporal_outliers(data)

    assert pt.isnan(clean[2])
    assert pt.isinf(clean[4])
    assert pt.equal(pt.isfinite(clean), pt.isfinite(data))


def test_replace_temporal_outliers_window_one_returns_clone():
    data = pt.arange(5.0)

    clean = replace_temporal_outliers(data, window_size=1)

    pt.testing.assert_close(clean, data)
    assert clean.data_ptr() != data.data_ptr()


@pytest.mark.parametrize(
    ("data", "kwargs", "message"),
    [
        (pt.tensor([1, 2, 3]), {}, "floating-point"),
        (pt.tensor([]), {}, "at least one value"),
        (pt.ones(3), {"threshold": 0.0}, "threshold"),
        (pt.ones(3), {"threshold": float("inf")}, "threshold"),
        (pt.ones(3), {"window_size": 0}, "window_size"),
        (pt.ones(3), {"window_size": 2}, "window_size"),
        (pt.ones(3), {"window_size": 3.0}, "window_size"),
    ],
)
def test_replace_temporal_outliers_validates_inputs(data, kwargs, message):
    with pytest.raises(ValueError, match=message):
        replace_temporal_outliers(data, **kwargs)


def test_replace_spatial_outliers():
    data = pt.ones((5, 5, 2))
    data[2, 2, 0] = 100.0
    data[1, 1, 1] = float("nan")

    clean = replace_spatial_outliers(data)

    assert clean[2, 2, 0] == 1.0
    assert pt.isnan(clean[1, 1, 1])
    assert pt.equal(clean[..., 1].isnan(), data[..., 1].isnan())


def test_replace_spatial_outliers_preserves_infinite_values():
    data = pt.ones((5, 5))
    data[2, 2] = float("inf")
    data[1, 1] = 100.0

    clean = replace_spatial_outliers(data)

    assert pt.isinf(clean[2, 2])
    assert clean[1, 1] == 1.0
