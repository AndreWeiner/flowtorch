"""
 class for testing the s_cube_dataloader
"""
import pytest

from tests.helpers import requires_datasets


requires_dataset = requires_datasets("s_cube")
pytestmark = [requires_dataset, pytest.mark.integration]

from flowtorch.constants import DATASETS
from flowtorch.data import SCUBEDataloader

FILENAME = r"s_cube_test_dataset.h5"


def test_dataloader():
    # path to the dataset
    load_path = DATASETS["s_cube"]

    # the file should contain 209 cells, 247 nodes, pressure field, t = 0.4, metric, levels (can be checked implicitly
    # via weights) and the grid
    n_cells = 209
    n_nodes = 247
    n_dimensions = 2
    write_times = ["0.4"]
    field_names = {'0.4': ['p']}

    # instantiate dataloader
    dataloader = SCUBEDataloader(load_path, "s_cube_test_dataset.h5")

    # check if everything is present and can be loaded correctly
    assert len(dataloader.write_times) == 1
    assert dataloader.write_times == write_times
    assert dataloader.field_names == field_names
    assert dataloader.vertices.shape == (n_cells, n_dimensions)
    assert dataloader.weights.shape == dataloader.levels.shape
    assert dataloader.faces.shape == (n_cells, pow(2, n_dimensions))
    assert dataloader.nodes.shape == (n_nodes, n_dimensions)
    assert dataloader.load_snapshot("p", "0.4").shape == (n_cells, 1)


if __name__ == "__main__":
    pass
