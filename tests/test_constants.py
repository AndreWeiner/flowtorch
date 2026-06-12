# standard library packages
import os
import sys
# third party packages
import pytest
import torch as pt
# flowtorch packages
from flowtorch.constants import find_datasets
from tests.helpers import requires_datasets


requires_dataset = requires_datasets()
pytestmark = [requires_dataset, pytest.mark.integration]

def test_find_datasets():
    # assuming that the datasets are available and FLOWTORCH_DATASETS
    # is properly set
    path = os.environ["FLOWTORCH_DATASETS"]
    assert find_datasets(path)
    assert not find_datasets("")
    assert not find_datasets(123)
