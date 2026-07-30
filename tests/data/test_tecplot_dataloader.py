# standard library packages
from os.path import join

# third party packages
import pytest
import torch as pt

# flowtorch packages
from tests.helpers import requires_datasets

requires_dataset = requires_datasets("plt_naca2409_surface")
pytestmark = [requires_dataset, pytest.mark.integration]

from flowtorch import DATASETS
from flowtorch.data import TecplotDataloader


class TestTecplotDataloader:
    path = DATASETS["plt_naca2409_surface"]
    loader = TecplotDataloader.from_tau(path, "alfa16.surface.pval.unsteady_")

    def test_from_tau(self):
        file_names = [
            "alfa16.surface.pval.unsteady_i=1600_t=5.6805000e-01.plt",
            "alfa16.surface.pval.unsteady_i=1610_t=5.6955000e-01.plt",
            "alfa16.surface.pval.unsteady_i=1620_t=5.7105000e-01.plt",
        ]
        times = ["5.6805000e-01", "5.6955000e-01", "5.7105000e-01"]
        assert list(self.loader._file_names.keys()) == times
        assert list(self.loader._file_names.values()) == file_names

    def test_assemble_file_path(self):
        first_file_path = self.loader._assemble_file_path("5.6805000e-01")
        true_path = join(
            self.path, "alfa16.surface.pval.unsteady_i=1600_t=5.6805000e-01.plt"
        )
        assert first_file_path == true_path

    def test_create_tecplot_reader(self):
        reader = self.loader._create_tecplot_reader("5.6805000e-01")
        assert type(reader).__name__ == "VisItTecplotBinaryReader"

    def test_field_names(self):
        field_names = self.loader.field_names
        assert "5.6805000e-01" in field_names.keys()
        expected_fields = [
            "X",
            "Y",
            "Z",
            "density",
            "pressure",
            "cp",
            "cf",
            "cfx",
            "cfy",
            "cfz",
            "yplus",
            "eddy_viscosity",
        ]
        assert field_names["5.6805000e-01"] == expected_fields

    def test_parse_block_name(self):
        meta_data = """vtkInformation (0x1a1812a0)
  Debug: Off
  Modified Time: 358329
  Reference Count: 2
  Registered Events: (none)
  NAME: ls
        """
        block_name = self.loader._parse_block_name(meta_data)
        assert block_name == "ls"

    def test_zone_names(self):
        zone_names = self.loader.zone_names
        assert zone_names == ["ls", "te", "us"]

    def test_zone(self):
        assert self.loader.zone == "ls"
        self.loader.zone = "us"
        assert self.loader.zone == "us"
        self.loader.zone = "none"
        assert self.loader.zone == "us"
        self.loader.zone = "ls"

    def test_vertices(self):
        vertices = self.loader.vertices
        assert vertices.shape == (300, 3)
        self.loader.zone = "te"
        vertices = self.loader.vertices
        assert vertices.shape == (30, 3)
        self.loader.zone = "ls"

    def test_weights(self):
        weights = self.loader.weights
        assert weights.shape == (300,)

    def test_load_single_snapshot(self):
        density = self.loader._load_single_snapshot(
            "density", self.loader.write_times[0]
        )
        assert density.shape == (300,)
        with pytest.raises(ValueError):
            _ = self.loader._load_single_snapshot("none", self.loader.write_times[0])

    def test_load_multiple_snapshots(self):
        density = self.loader._load_multiple_snapshots(
            "density", self.loader.write_times[:2]
        )
        assert density.shape == (300, 2)

    def test_load_snapshot(self):
        times = self.loader.write_times
        # single field, single time
        cf = self.loader.load_snapshot("cf", times[0])
        assert cf.shape == (300,)
        # single field, multiple times
        self.loader.zone = "te"
        cf = self.loader.load_snapshot("cf", times[:2])
        assert cf.shape == (30, 2)
        # multiple fields, single time
        cf, cp = self.loader.load_snapshot(["cf", "cp"], times[0])
        assert cf.shape == (30,)
        assert cp.shape == (30,)
        # multiple fields, multiple times
        cf, cp = self.loader.load_snapshot(["cf", "cp"], times[:2])
        assert cf.shape == (30, 2)
        assert cp.shape == (30, 2)
        self.loader.zone = "ls"
