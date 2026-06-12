# third party packages
import pytest
import torch as pt
# flowtorch packages
from flowtorch.data import iqr_outlier_replacement


def test_irq_outlier_replacement():
    data = pt.tensor([
        [3.0, 2.0, 4.0, 8.0, 1.0, 0.0],
        [3.0, 2.0, 4.0, 5.0, 1.0, 0.0]
    ])
    clean_data = iqr_outlier_replacement(data)
    # the shape of both datasets should be equal
    assert clean_data.shape == data.shape
    # check if outlier is detected and replaced;
    # the number of elements in the second direction
    # is even -> PyTorch returns the lower median
    assert clean_data[0][3].item() == 2.0
    # decrease sensitivity
    data_clean = iqr_outlier_replacement(data, k=2.0)
    data_clean[0][3] == 8.0
    # use only the two nearest neighbors
    data_clean = iqr_outlier_replacement(data, nb=1)
    assert data_clean[0][3] == 4.0
    # test with 1D tensor
    data = pt.tensor([3.0, 2.0, 4.0, 8.0, 1.0, 0.0])
    data_clean = iqr_outlier_replacement(data)
    assert len(data_clean.shape) == 1
    assert data_clean[3] == 2.0
