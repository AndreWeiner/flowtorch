# standard library packages
import pytest
# third party packages
import torch as pt
# flowtorch packages
from flowtorch.conftest import requires_datasets


requires_dataset = requires_datasets("ipsp_fake.hdf5")
pytestmark = [requires_dataset, pytest.mark.integration]

from flowtorch import DATASETS
from flowtorch.data import PSPDataloader


class TestPSPDataloader:
    def test_zone(self):
        loader = PSPDataloader(DATASETS["ipsp_fake.hdf5"])
        names = loader.zone_names
        assert "Zone0000" in names and "Zone0001" in names
        assert loader.zone == "Zone0000"
        loader.zone = names[-1]
        assert loader.zone == names[-1]
        loader.zone = ""
        assert loader.zone == names[-1]

    def test_mask(self):
        loader = PSPDataloader(DATASETS["ipsp_fake.hdf5"])
        masks = loader.mask_names
        assert masks == ["Mask"]
        loader.mask = "Mask"
        assert loader.mask == "Mask"
        loader.mask = ""
        assert loader.mask == "Mask"

    def test_info(self):
        loader = PSPDataloader(DATASETS["ipsp_fake.hdf5"])
        info = loader.info
        assert "Mach" in info.keys()
        assert len(info["Mach"]) == 2
        assert isinstance(info["Mach"][0], float)
        assert isinstance(info["Mach"][1], str)

    def test_zone_info(self):
        loader = PSPDataloader(DATASETS["ipsp_fake.hdf5"])
        info = loader.zone_info
        assert "SamplingFrequency" in info
        assert isinstance(info["SamplingFrequency"][0], float)
        assert info["ZoneName"][0] == "UpperSide"
        loader.zone = loader.zone_names[-1]
        info = loader.zone_info
        assert info["ZoneName"][0] == "LowerSide"

    def test_time_to_index(self):
        loader = PSPDataloader(DATASETS["ipsp_fake.hdf5"])
        times = loader.write_times
        indices = loader._time_to_index(times)
        assert indices == list(range(30))

    def test_common_properties(self):
        loader = PSPDataloader(DATASETS["ipsp_fake.hdf5"])
        n_snapshots = 30
        freq = 10
        times = loader.write_times
        assert len(times) == n_snapshots
        assert times[-1] == str(round((n_snapshots - 1) / freq, 8))
        assert times[0] == str(round(0.0, 8))
        fields = loader.field_names
        assert len(fields.keys()) == 1
        assert fields[times[0]][0] == "Cp"
        vertices = loader.vertices
        assert vertices.shape == (50, 20, 3)
        weights = loader.weights
        assert weights.shape == (50, 20)
        loader.zone = loader.zone_names[-1]
        vertices = loader.vertices
        assert vertices.shape == (50, 20, 3)
        weights = loader.weights
        assert weights.shape == (50, 20)
        # load single field, single snapshot
        cp = loader.load_snapshot("Cp", times[-1])
        assert cp.shape == (50, 20)
        # load single field, multiple snapshots
        cp_s = loader.load_snapshot("Cp", times[-10:])
        assert cp_s.shape == (50, 20, 10)
        assert pt.allclose(cp, cp_s[:, :, -1])
        # load multiple fields, single snapshot
        cp, = loader.load_snapshot(["Cp"], times[0])
        assert cp.shape == (50, 20)
        # load multiple fields, multiple snapshots
        cp_s, = loader.load_snapshot(["Cp"], times[:10])
        assert pt.allclose(cp, cp_s[:, :, 0])
