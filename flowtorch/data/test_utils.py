# third party packages
import pytest
from torch import rand, cat, zeros
# flowtorch packages
from flowtorch.data.utils import (format_byte_size, check_and_standardize_path,
                                  check_list_or_str)
from flowtorch.data.utils import CellVolumeEstimator


def test_byte_formatting():
    size, unit = format_byte_size(int(1e3))
    assert size == 1e3 and unit == "b"
    size, unit = format_byte_size(int(1e4))
    assert size == 1e4/1024 and unit == "Kb"
    size, unit = format_byte_size(int(1e7))
    assert size == 1e7/1024**2 and unit == "Mb"


def test_check_and_standardize_path():
    with pytest.raises(ValueError):
        _ = check_and_standardize_path("does/not/exist/")
    path = check_and_standardize_path("./")
    assert path == "."
    path = check_and_standardize_path("./flowtorch/__init__.py", folder=False)
    assert path == "./flowtorch/__init__.py"


def test_check_list_or_str():
    with pytest.raises(ValueError):
        check_list_or_str([], "test")
    with pytest.raises(ValueError):
        check_list_or_str(0, "test")
    with pytest.raises(ValueError):
        check_list_or_str(["", 0], "test")


@pytest.mark.parametrize("dims, n_points", [(2, 100), (3, 200)])
def test_cell_volume_estimator(dims: int, n_points: int):
    # make up some coordinates
    coords = rand(n_points, dims)

    # add a flat dimension to simulate 2D simulation when passing 3D points
    if dims == 2:
        coords = cat([coords, zeros(n_points, 1)], dim=1)

    estimator = CellVolumeEstimator(coords, k=4, normalize=True)

    # ensure dimensionality detection works
    if dims == 2:
        assert len(estimator._dims) == 2
    else:
        assert len(estimator._dims) == 3

    # execute estimation
    estimator.estimate_cell_volume()

    # check weights shape matches number of points
    assert estimator.weights.shape[0] == n_points

    # volumes should be positive
    assert estimator._volume_original > 0

    # since we normalize the weights they should be in [0, 1]
    assert all(estimator.weights >= 0)
    assert all(estimator.weights <= 1)

