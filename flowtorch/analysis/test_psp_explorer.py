# standard library packages
import pytest
# third party packages
import plotly.graph_objects as go
# flowtorch packages
from flowtorch.conftest import requires_datasets


requires_dataset = requires_datasets("ipsp_fake.hdf5")
pytestmark = [requires_dataset, pytest.mark.integration]

from flowtorch import DATASETS
from .psp_explorer import PSPExplorer


def test_psp_explorer():
    path = DATASETS["ipsp_fake.hdf5"]
    explorer = PSPExplorer(path)
    times = explorer.loader.write_times
    zones = explorer.loader.zone_names
    fig = explorer.interact(zones[0], "Cp", times[:10])
    assert isinstance(fig, go.Figure)
    fig = explorer.mean(zones[0], "Cp", times[:10])
    assert isinstance(fig, go.Figure)
    fig = explorer.std(zones[0], "Cp", times[:10])
    assert isinstance(fig, go.Figure)
