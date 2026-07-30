# standard library packages
import pytest

# third party packages
import torch as pt

# flowtorch packages
from tests.helpers import requires_datasets

requires_dataset = requires_datasets(
    "csv_naca0012_alpha4_surface",
    "csv_surface_mounted_cube_xy",
    "csv_aoa8_beta0_xc100_stereopiv",
    "csv_davis_multiple_fields",
)
pytestmark = [requires_dataset, pytest.mark.integration]

from flowtorch.constants import FLOAT_TOLERANCE
from flowtorch import DATASETS
from flowtorch.data import CSVDataloader
from flowtorch.data.csv_dataloader import (
    _parse_davis_header,
    _parse_foam_surface_header,
)


def test_parse_foam_surface():
    header = "# x y z  U_x U_y U_z  area_x area_y area_z"
    columns = ["x", "y", "z", "U_x", "U_y", "U_z", "area_x", "area_y", "area_z"]
    assert columns == _parse_foam_surface_header(header)


def test_from_foam_surface():
    path = DATASETS["csv_naca0012_alpha4_surface"]
    loader = CSVDataloader.from_foam_surface(path, "total(p)_coeff_airfoil.raw")
    times = loader.write_times
    assert len(times) == 250
    assert times[0] == "0.001"
    assert times[-1] == "0.25"
    fields = loader.field_names
    assert fields[times[0]][0] == "total(p)_coeff"
    n_points = 28892
    vertices = loader.vertices
    assert vertices.shape == (n_points, 3)
    weights = loader.weights
    # should be a tensor filled with ones
    assert pt.sum(weights).item() == n_points
    # single snapshot, single field
    snapshot = loader.load_snapshot("total(p)_coeff", times[0])
    assert snapshot.shape == (n_points,)
    # multiple snapshots, single field
    snapshots = loader.load_snapshot("total(p)_coeff", times[:10])
    assert snapshots.shape == (n_points, 10)
    assert pt.allclose(snapshot, snapshots[:, 0])
    # single snapshot, multiple fields
    snapshot = loader.load_snapshot(["total(p)_coeff"], times[0])
    assert len(snapshot) == 1
    assert snapshot[0].shape == (n_points,)
    # multiple snapshots, multiple fields
    snapshots = loader.load_snapshot(["total(p)_coeff"], times[:10])
    assert len(snapshots) == 1
    assert snapshots[0].shape == (n_points, 10)
    assert pt.allclose(snapshots[0][:, 0], snapshot[0])

    # test case with multiple fields and face area weights
    path = DATASETS["csv_surface_mounted_cube_xy"]
    loader = CSVDataloader.from_foam_surface(path, "U_plane_xy.raw")
    fields, times = loader.field_names, loader.write_times
    assert fields[times[0]] == ["U_x", "U_y", "U_z"]
    # face area weights
    weights = loader.weights
    assert weights.shape == (35154,)
    assert (
        weights[0] - pt.tensor([5.31517e-18, 2.85608e-18, 0.00985272]).norm()
    ).item() < FLOAT_TOLERANCE


def test_parse_davis_header():
    header = 'VARIABLES = "x", "y", "Vx", "Vy", "Vz", "swirl strength", "vector length", "vorticity", "isValid"'
    columns = _parse_davis_header(header)
    expected_columns = [
        "x",
        "y",
        "Vx",
        "Vy",
        "Vz",
        "swirl strength",
        "vector length",
        "vorticity",
        "isValid",
    ]
    assert columns == expected_columns


def test_from_davis():
    path = DATASETS["csv_aoa8_beta0_xc100_stereopiv"]
    loader = CSVDataloader.from_davis(path, "B")
    times = loader.write_times
    assert len(times) == 5000
    assert times[0] == "00001"
    assert times[-1] == "05000"
    fields = loader.field_names
    fields = fields[times[0]]
    n_points = 3741
    vertices = loader.vertices
    # DaVis files have only x and y components
    assert vertices.shape == (n_points, 2)
    weights = loader.weights
    assert weights.shape == (n_points,)
    # single snapshot, single field
    snapshot = loader.load_snapshot("Vx", times[0])
    assert snapshot.shape == (n_points,)
    # multiple snapshots, single field
    snapshots = loader.load_snapshot("Vx", times[:10])
    assert snapshots.shape == (n_points, 10)
    assert pt.allclose(snapshot, snapshots[:, 0])
    # single snapshot, multiple fields
    snapshot = loader.load_snapshot(fields[:3], times[0])
    assert len(snapshot) == 3
    assert snapshot[0].shape == (n_points,)
    assert snapshot[1].shape == (n_points,)
    assert snapshot[2].shape == (n_points,)
    # multiple snapshots, multiple fields
    snapshots = loader.load_snapshot(fields[:3], times[:10])
    assert len(snapshots) == 3
    assert snapshots[0].shape == (n_points, 10)
    assert snapshots[1].shape == (n_points, 10)
    assert snapshots[2].shape == (n_points, 10)
    assert pt.allclose(snapshots[0][:, 0], snapshot[0])
    # test parsing multiple fields (header containing more than velocity)
    path = DATASETS["csv_davis_multiple_fields"]
    loader = CSVDataloader.from_davis(path, "B")
    times = loader.write_times
    fields = loader.field_names[times[0]]
    assert fields == ["Vx", "Vy", "Vz", "swirl strength", "vector length", "vorticity"]
    snapshot = loader.load_snapshot("swirl strength", times[0])
    assert snapshot.shape == (n_points,)
