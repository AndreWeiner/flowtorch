# standard library packages
import pytest

# third party packages
import torch as pt

# flowtorch packages
from tests.helpers import requires_datasets

requires_dataset = requires_datasets(
    "vtk_cylinder_re200_flexi",
    "vtk_su2_airfoil_2D",
)
pytestmark = [requires_dataset, pytest.mark.integration]

from flowtorch import DATASETS, FLOAT_TOLERANCE
from flowtorch.data import VTKDataloader


def test_vtk_flexi_field():
    path = DATASETS["vtk_cylinder_re200_flexi"]
    loader = VTKDataloader.from_flexi(path, "Cylinder_Re200_Solution_")
    write_times = loader.write_times
    assert len(write_times) == 3
    assert write_times == ["0000000", "0000005", "0000300"]
    field_names = loader.field_names
    assert len(field_names[write_times[0]]) == 4
    assert field_names[write_times[0]][0] == "Density"
    vertices = loader.vertices
    assert vertices.shape == (729000, 3)
    # check bounds for x and z value
    min_values, _ = vertices.min(dim=0)
    max_values, _ = vertices.max(dim=0)
    assert abs(min_values[0] + 100) < FLOAT_TOLERANCE
    assert abs(min_values[2] + 2) < FLOAT_TOLERANCE
    assert abs(max_values[0] - 100) < FLOAT_TOLERANCE
    assert abs(max_values[2] - 2) < FLOAT_TOLERANCE
    with pytest.raises(NotImplementedError):
        weights = loader.weights
    # single field, single snapshot
    fields = field_names[write_times[0]]
    snapshot = loader.load_snapshot(fields[0], write_times[-1])
    n_points = vertices.shape[0]
    assert snapshot.shape == (n_points,)
    # single field, multiple snapshots
    snapshots = loader.load_snapshot(fields[0], write_times)
    assert snapshots.shape == (n_points, len(write_times))
    assert pt.allclose(snapshot, snapshots[:, -1])
    # multiple fields, single snapshot
    density, momX = loader.load_snapshot(fields[:2], write_times[-1])
    assert density.shape == (n_points,)
    assert momX.shape == (n_points,)
    # multiple fields, multiple snapshots
    density_series, momX_series = loader.load_snapshot(fields[:2], write_times)
    assert density_series.shape == (n_points, len(write_times))
    assert momX_series.shape == (n_points, len(write_times))
    assert pt.allclose(density, density_series[:, -1])
    assert pt.allclose(momX, momX_series[:, -1])


def test_vtk_flexi_surface():
    path = DATASETS["vtk_cylinder_re200_flexi"]
    loader = VTKDataloader.from_flexi(path, "Cylinder_Re200_Surf_")
    write_times = loader.write_times
    assert len(write_times) == 3
    assert write_times == ["0000000", "0000005", "0000300"]
    field_names = loader.field_names
    assert len(field_names[write_times[0]]) == 4
    assert field_names[write_times[0]][0] == "Density"
    vertices = loader.vertices
    assert vertices.shape == (1620, 3)
    # check bounds for x and z value
    min_values, _ = vertices.min(dim=0)
    max_values, _ = vertices.max(dim=0)
    assert abs(min_values[0] + 0.5) < FLOAT_TOLERANCE
    assert abs(min_values[2] + 2) < FLOAT_TOLERANCE
    assert abs(max_values[0] - 0.5) < FLOAT_TOLERANCE
    assert abs(max_values[2] - 2) < FLOAT_TOLERANCE
    with pytest.raises(NotImplementedError):
        weights = loader.weights
    # single field, single snapshot
    fields = field_names[write_times[0]]
    snapshot = loader.load_snapshot(fields[0], write_times[-1])
    n_points = vertices.shape[0]
    assert snapshot.shape == (n_points,)
    # single field, multiple snapshots
    snapshots = loader.load_snapshot(fields[0], write_times)
    assert snapshots.shape == (n_points, len(write_times))
    assert pt.allclose(snapshot, snapshots[:, -1])
    # multiple fields, single snapshot
    density, momX = loader.load_snapshot(fields[:2], write_times[-1])
    assert density.shape == (n_points,)
    assert momX.shape == (n_points,)
    # multiple fields, multiple snapshots
    density_series, momX_series = loader.load_snapshot(fields[:2], write_times)
    assert density_series.shape == (n_points, len(write_times))
    assert momX_series.shape == (n_points, len(write_times))
    assert pt.allclose(density, density_series[:, -1])
    assert pt.allclose(momX, momX_series[:, -1])


def test_vtk_su2():
    path = DATASETS["vtk_su2_airfoil_2D"]
    loader = VTKDataloader.from_su2(path, "flow_")
    write_times = loader.write_times
    assert len(write_times) == 4
    assert write_times == ["0038{:1d}".format(i) for i in range(4)]
    field_names = loader.field_names
    assert len(field_names[write_times[0]]) == 6
    vertices = loader.vertices
    assert vertices.shape == (214403, 3)
    # single field, single snapshot
    fields = field_names[write_times[0]]
    snapshot = loader.load_snapshot("Pressure", write_times[0])
    n_points = vertices.shape[0]
    assert snapshot.shape == (n_points,)
    # single field, multiple snapshots
    snapshots = loader.load_snapshot("Pressure", write_times[:2])
    assert snapshots.shape == (n_points, 2)
    assert pt.allclose(snapshot, snapshots[:, 0])
    # multiple fields, single snapshot
    p, U = loader.load_snapshot(["Pressure", "Velocity"], write_times[0])
    assert p.shape == (n_points,)
    assert U.shape == (n_points, 3)
    # multiple fields, multiple snapshots
    p_series, U_series = loader.load_snapshot(["Pressure", "Velocity"], write_times[:2])
    assert p_series.shape == (n_points, 2)
    assert U_series.shape == (n_points, 3, 2)
    assert pt.allclose(p, p_series[:, 0])
    assert pt.allclose(U, U_series[:, :, 0])
