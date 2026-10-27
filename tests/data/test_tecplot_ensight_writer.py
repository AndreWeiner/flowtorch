"""Round trips from Tecplot geometry through the EnSight writer."""

import numpy as np
import pytest
import torch as pt

pytest.importorskip("vtkmodules.vtkIOEnSight")
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid, vtkTriangle, vtkQuad
from vtkmodules.util.numpy_support import vtk_to_numpy

from flowtorch.data import TecplotDataloader, EnSightDataloader, EnSightWriter
from flowtorch.data._tecplot_pvpython import _export_mesh
import flowtorch.data.tecplot_dataloader as tecplot_module


def source_mesh():
    points = vtkPoints()
    for point in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0), (2, 1, 0)]:
        points.InsertNextPoint(*point)
    mesh = vtkUnstructuredGrid()
    mesh.SetPoints(points)
    for cls, nodes in [
        (vtkTriangle, [0, 1, 2]),
        (vtkQuad, [1, 4, 5, 2]),
        (vtkTriangle, [0, 2, 3]),
    ]:
        cell = cls()
        for j, node in enumerate(nodes):
            cell.GetPointIds().SetId(j, node)
        mesh.InsertNextCell(cell.GetCellType(), cell.GetPointIds())
    return mesh


@pytest.fixture
def loader_factory(tmp_path, monkeypatch):
    class Client:
        def __init__(self, executable):
            pass

        def request(self, operation, **kwargs):
            if operation == "zone_names":
                return [
                    {"name": "surface", "path": [], "full_name": "surface"},
                    {"name": "other", "path": [1], "full_name": "other"},
                ]
            if operation == "mesh":
                return _export_mesh(source_mesh(), np, kwargs.get("cell_type"))
            raise AssertionError(operation)

        def close(self):
            pass

    monkeypatch.setattr(tecplot_module, "_PvpythonClient", Client)
    monkeypatch.setattr(tecplot_module, "_resolve_pvpython", lambda value: "pvpython")
    source = tmp_path / "input.plt"
    source.write_bytes(b"source must be preserved")

    def create(association="point"):
        return TecplotDataloader(
            str(tmp_path), {"0": source.name}, association=association
        )

    return create


@pytest.mark.parametrize("association", ["point", "cell"])
@pytest.mark.parametrize("format", ["ascii", "binary"])
def test_mesh_static_and_series_roundtrip(
    loader_factory, tmp_path, association, format
):
    with loader_factory(association) as loader:
        loader.zone = "other"
        mesh = loader.mesh
        count = (
            mesh.GetNumberOfPoints()
            if association == "point"
            else mesh.GetNumberOfCells()
        )
        scalar = pt.arange(count, dtype=pt.float64) + 0.5
        vector = scalar[:, None].repeat(1, 3)
        with EnSightWriter.from_loader(
            loader, tmp_path / "output", format=format
        ) as writer:
            loader.zone = "surface"  # Writer retains its initial zone.
            writer.write_static("mean", scalar)
            writer.write_static("mode", vector.to(pt.complex128) * (1 + 2j))
            writer.write_series(
                {
                    "value": pt.stack([scalar, scalar + 1], -1),
                    "vector": pt.stack([vector, vector + 1], -1),
                },
                ["0.1", "0.3"],
            )
        case_file = writer.case_file
    result = EnSightDataloader(case_file, association=association, dtype=pt.float64)
    assert result.part == "other"
    assert case_file.name == "results.case"
    assert result.write_times == ["0.1", "0.3"]
    np.testing.assert_allclose(
        vtk_to_numpy(result.mesh.GetPoints().GetData()),
        vtk_to_numpy(mesh.GetPoints().GetData()),
    )
    # EnSight groups element types in order of first occurrence.
    rows = np.array([0, 2, 1])
    np.testing.assert_array_equal(EnSightDataloader._cell_types(result.mesh), [5, 5, 9])
    np.testing.assert_array_equal(
        vtk_to_numpy(result.mesh.GetCells().GetConnectivityArray()),
        [0, 1, 2, 0, 2, 3, 1, 4, 5, 2],
    )
    expected = scalar if association == "point" else scalar[rows]
    for time, delta in [("0.1", 0), ("0.3", 1)]:
        pt.testing.assert_close(result.load_snapshot("mean", time), expected)
        pt.testing.assert_close(result.load_snapshot("value", time), expected + delta)
        pt.testing.assert_close(
            result.load_snapshot("vector", time),
            (expected + delta)[:, None].repeat(1, 3),
        )
        pt.testing.assert_close(
            result.load_snapshot("mode_imag", time), expected[:, None].repeat(1, 3) * 2
        )
    assert (tmp_path / "input.plt").read_bytes() == b"source must be preserved"


def test_mesh_copy_is_independent(loader_factory):
    with loader_factory() as loader:
        mesh = loader.mesh
        mesh.GetPoints().SetPoint(0, 99, 99, 99)
        np.testing.assert_allclose(loader.mesh.GetPoint(0), (0, 0, 0))
        assert mesh.GetPointData().GetNumberOfArrays() == 0
        assert mesh.GetCellData().GetNumberOfArrays() == 0


def test_source_directory_is_protected(loader_factory, tmp_path):
    with loader_factory() as loader:
        with pytest.raises(ValueError, match="source EnSight or Tecplot"):
            EnSightWriter.from_loader(loader, tmp_path)


@pytest.mark.integration
@pytest.mark.parametrize("association", ["point", "cell"])
def test_real_plt_roundtrip(tmp_path, association):
    import shutil
    from flowtorch import DATASETS

    if "plt_naca2409_surface" not in DATASETS or shutil.which("pvpython") is None:
        pytest.skip("Requires NACA PLT dataset and pvpython")
    with TecplotDataloader.from_tau(
        DATASETS["plt_naca2409_surface"],
        "alfa16.surface.pval.unsteady_",
        association=association,
        cell_type="quad",
    ) as loader:
        loader.zone = "us"
        mesh = loader.mesh
        count = (
            mesh.GetNumberOfPoints()
            if association == "point"
            else mesh.GetNumberOfCells()
        )
        values = (
            loader.load_snapshot("cp", loader.write_times[:2])
            if association == "point"
            else pt.arange(count, dtype=pt.float64)[:, None].repeat(1, 2)
        )
        times = loader.write_times[:2]
        with EnSightWriter.from_loader(loader, tmp_path / "output") as writer:
            writer.write_static("mean", values.mean(-1))
            writer.write_series({"reconstructed": values}, times)
    result = EnSightDataloader(
        writer.case_file, association=association, dtype=values.dtype
    )
    assert result.part == "us"
    np.testing.assert_allclose(
        vtk_to_numpy(result.mesh.GetPoints().GetData()),
        vtk_to_numpy(mesh.GetPoints().GetData()),
    )
    np.testing.assert_array_equal(
        vtk_to_numpy(result.mesh.GetCells().GetConnectivityArray()),
        vtk_to_numpy(mesh.GetCells().GetConnectivityArray()),
    )
    np.testing.assert_array_equal(
        EnSightDataloader._cell_types(result.mesh), EnSightDataloader._cell_types(mesh)
    )
    pt.testing.assert_close(result.load_snapshot("reconstructed", times), values)
    pt.testing.assert_close(result.load_snapshot("mean", times[0]), values.mean(-1))


def test_export_does_not_mutate_or_copy_source_fields():
    from vtkmodules.util.numpy_support import numpy_to_vtk

    mesh = source_mesh()
    values = numpy_to_vtk(np.arange(mesh.GetNumberOfPoints(), dtype=float), deep=True)
    values.SetName("original")
    mesh.GetPointData().AddArray(values)
    payload = _export_mesh(mesh, np)
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

    reader = vtkXMLUnstructuredGridReader()
    reader.ReadFromInputStringOn()
    reader.SetInputString(payload.tobytes().decode("utf-8"))
    reader.Update()
    assert reader.GetOutput().GetPointData().GetNumberOfArrays() == 0
    np.testing.assert_array_equal(
        vtk_to_numpy(mesh.GetPointData().GetArray("original")),
        np.arange(mesh.GetNumberOfPoints()),
    )
