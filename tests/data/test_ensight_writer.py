"""Geometry and field round trips through independent VTK EnSight readers."""

from hashlib import sha256
from pathlib import Path
import shutil
from zipfile import ZipFile

import numpy as np
import pytest
import torch as pt

pytest.importorskip("vtkmodules.vtkIOEnSight")
from vtkmodules.util.numpy_support import vtk_to_numpy
from flowtorch.data import EnSightDataloader, EnSightWriter


@pytest.fixture(scope="module")
def exports(tmp_path_factory):
    folder = tmp_path_factory.mktemp("ensight-writer-source")
    with ZipFile(Path(__file__).parent / "fixtures/ensight_cavity.zip") as archive:
        archive.extractall(folder)
    return folder


def source_case(exports, variant="", source="volume_binary"):
    stem = variant or "ensight_cavity"
    return exports / variant / "postProcessing" / source / f"{stem}.case"


def assert_mesh_equal(a, b, ascii_polygons=False):
    np.testing.assert_allclose(
        vtk_to_numpy(a.GetPoints().GetData()),
        vtk_to_numpy(b.GetPoints().GetData()),
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        EnSightDataloader._cell_types(a), EnSightDataloader._cell_types(b)
    )
    expected_offsets = vtk_to_numpy(a.GetCells().GetOffsetsArray())
    expected_nodes = vtk_to_numpy(a.GetCells().GetConnectivityArray()).copy()
    if ascii_polygons:
        # vtkEnSightGoldReader reverses each ASCII nsided row on input; the
        # writer keeps the VTK source connectivity unchanged in the file.
        for row in np.flatnonzero(EnSightDataloader._cell_types(a) == 7):
            start, stop = expected_offsets[row : row + 2]
            expected_nodes[start:stop] = expected_nodes[start:stop][::-1]
    np.testing.assert_array_equal(
        expected_offsets, vtk_to_numpy(b.GetCells().GetOffsetsArray())
    )
    np.testing.assert_array_equal(
        expected_nodes, vtk_to_numpy(b.GetCells().GetConnectivityArray())
    )
    if 42 in EnSightDataloader._cell_types(a) and hasattr(a, "GetPolyhedronFaces"):
        for accessor in ("GetPolyhedronFaces", "GetPolyhedronFaceLocations"):
            for array in ("GetOffsetsArray", "GetConnectivityArray"):
                np.testing.assert_array_equal(
                    vtk_to_numpy(getattr(getattr(a, accessor)(), array)()),
                    vtk_to_numpy(getattr(getattr(b, accessor)(), array)()),
                )


def unique_fields(count, dtype=pt.float64):
    index = pt.arange(count, dtype=dtype)
    fields = {"index": index}
    for name, components in [("vector", 3), ("symm", 6), ("tensor", 9)]:
        fields[name] = index[:, None] * 10 + pt.arange(components, dtype=dtype)[None, :]
    return fields


@pytest.mark.parametrize("variant", ["", "dual", "parallel"])
@pytest.mark.parametrize("format", ["ascii", "binary"])
@pytest.mark.parametrize("part", ["internalMesh", "movingWall", "fixedWalls"])
def test_volume_patch_round_trip(exports, tmp_path, variant, format, part):
    loader = EnSightDataloader(
        source_case(exports, variant), part=part, dtype=pt.float64
    )
    times = loader.write_times
    fields = dict(
        zip(["p", "U", "S", "T"], loader.load_snapshot(["p", "U", "S", "T"], times))
    )
    for name, value in unique_fields(len(loader.weights)).items():
        fields[name] = pt.stack([value, value + 1], dim=-1)
    with EnSightWriter.from_loader(
        loader, tmp_path / "export", case_name="reconstruction", format=format
    ) as writer:
        writer.write_series(fields, times)
        # A successfully published snapshot is readable even before close.
        assert EnSightDataloader(writer.case_file).write_times == times
    result = EnSightDataloader(writer.case_file, dtype=pt.float64)
    assert result.parts == [part]
    assert result.part == part
    assert writer.case_file.name == "reconstruction.case"
    assert result.write_times == times
    assert_mesh_equal(loader.mesh, result.mesh, ascii_polygons=format == "ascii")
    pt.testing.assert_close(loader.vertices, result.vertices, rtol=2e-6, atol=1e-7)
    pt.testing.assert_close(loader.weights, result.weights, rtol=2e-6, atol=1e-10)
    for name, expected in fields.items():
        pt.testing.assert_close(
            result.load_snapshot(name, times), expected, rtol=2e-6, atol=1e-7
        )


@pytest.mark.parametrize("variant", ["", "dual", "parallel"])
@pytest.mark.parametrize("format", ["ascii", "binary"])
@pytest.mark.parametrize("surface", ["midplane", "oblique", "lid"])
def test_surface_round_trip(exports, tmp_path, variant, format, surface):
    path = exports / variant / "postProcessing/cuts_ascii" / surface / f"{surface}.case"
    loader = EnSightDataloader(path, dtype=pt.float64)
    fields = unique_fields(len(loader.weights))
    with EnSightWriter.from_case(path, tmp_path / "export", format=format) as writer:
        assert writer.case_file.name == f"{surface}.case"
        writer.write_snapshot(fields, "0.05")
        writer.write_snapshot(
            {name: values + 2 for name, values in fields.items()}, "0.125"
        )
    result = EnSightDataloader(writer.case_file, dtype=pt.float64)
    assert result.write_times == ["0.05", "0.125"]
    assert_mesh_equal(loader.mesh, result.mesh, ascii_polygons=format == "ascii")
    for name, expected in fields.items():
        pt.testing.assert_close(result.load_snapshot(name, "0.05"), expected)
        pt.testing.assert_close(result.load_snapshot(name, "0.125"), expected + 2)


@pytest.mark.parametrize("variant", ["", "dual", "parallel"])
@pytest.mark.parametrize("format", ["ascii", "binary"])
def test_point_round_trip(exports, tmp_path, variant, format):
    loader = EnSightDataloader(
        source_case(exports, variant, "nodes_binary"),
        part="internalMesh",
        association="point",
        dtype=pt.float64,
    )
    values = unique_fields(len(loader.weights))
    with EnSightWriter.from_loader(
        loader, tmp_path / "export", format=format
    ) as writer:
        writer.write_snapshot(values, 0.25)
    result = EnSightDataloader(writer.case_file, association="point", dtype=pt.float64)
    assert_mesh_equal(loader.mesh, result.mesh, ascii_polygons=format == "ascii")
    pt.testing.assert_close(result.weights, loader.weights)
    for name, expected in values.items():
        pt.testing.assert_close(
            result.load_snapshot(name, result.write_times[0]), expected
        )


@pytest.mark.parametrize(
    "association,source", [("cell", "volume_binary"), ("point", "nodes_binary")]
)
@pytest.mark.parametrize("format", ["ascii", "binary"])
def test_modes_and_complex_reconstruction(
    exports, tmp_path, association, source, format
):
    loader = EnSightDataloader(
        source_case(exports, source=source),
        part="internalMesh",
        association=association,
        dtype=pt.float64,
    )
    values = unique_fields(len(loader.vertices))
    complex_vector = values["vector"].to(pt.complex128) * (1 + 2j)
    with EnSightWriter.from_loader(
        loader, tmp_path / "export", format=format
    ) as writer:
        for name, data in values.items():
            writer.write_static("mode_" + name, data)
        writer.write_static("complex_mode", complex_vector)
        static = EnSightDataloader(
            writer.case_file, association=association, dtype=pt.float64
        )
        assert static.write_times == ["0"]
        pt.testing.assert_close(
            static.load_snapshot("complex_mode_real", "0"), complex_vector.real
        )
        pt.testing.assert_close(
            static.load_snapshot("complex_mode_imag", "0"), complex_vector.imag
        )
        writer.write_series(
            {"complex_rec": pt.stack([complex_vector, complex_vector * 3], dim=-1)},
            ["0.05", "0.1"],
        )
        writer.write_static("later_mode", values["index"])
    result = EnSightDataloader(
        writer.case_file, association=association, dtype=pt.float64
    )
    for time, factor in zip(result.write_times, [1, 3]):
        for name, data in values.items():
            pt.testing.assert_close(result.load_snapshot("mode_" + name, time), data)
        pt.testing.assert_close(
            result.load_snapshot("complex_rec_real", time), complex_vector.real * factor
        )
        pt.testing.assert_close(
            result.load_snapshot("complex_rec_imag", time), complex_vector.imag * factor
        )
        pt.testing.assert_close(
            result.load_snapshot("later_mode", time), values["index"]
        )


def test_source_untouched_and_output_portable(exports, tmp_path):
    original = source_case(exports)
    source = tmp_path / "source"
    shutil.copytree(original.parent, source)
    before = {
        path.relative_to(source): sha256(path.read_bytes()).digest()
        for path in source.rglob("*")
        if path.is_file()
    }
    loader = EnSightDataloader(source / original.name, part="internalMesh")
    data = loader.load_snapshot("U", loader.write_times[0])
    with EnSightWriter.from_loader(loader, tmp_path / "output") as writer:
        writer.write_static("mode", data)
    after = {
        path.relative_to(source): sha256(path.read_bytes()).digest()
        for path in source.rglob("*")
        if path.is_file()
    }
    assert before == after
    shutil.rmtree(source)
    destination = tmp_path / "moved"
    shutil.move(tmp_path / "output", destination)
    result = EnSightDataloader(destination / original.name)
    pt.testing.assert_close(result.load_snapshot("mode", "0"), data)


def test_selection_and_output_protection(exports, tmp_path):
    path = source_case(exports)
    with pytest.raises(ValueError, match="Select a part"):
        EnSightWriter.from_case(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    loader = EnSightDataloader(path, part="movingWall")
    for output in [path.parent, path.parent.parent, exports]:
        with pytest.raises(ValueError, match="source EnSight or Tecplot files"):
            EnSightWriter.from_loader(loader, output)
    directory = tmp_path / "occupied"
    directory.mkdir()
    existing = directory / "keep.txt"
    existing.write_text("keep")
    with pytest.raises(FileExistsError, match="new or empty"):
        EnSightWriter.from_loader(loader, directory)
    assert existing.read_text() == "keep"
    with pytest.raises(ValueError, match="case_name"):
        EnSightWriter.from_loader(loader, tmp_path / "new", case_name="../input")
    with pytest.raises(ValueError, match="format"):
        EnSightWriter.from_loader(loader, tmp_path / "new", format="vtk")
    # New subdirectories inside the source folder are safe: no source file is replaced.
    with EnSightWriter.from_loader(loader, path.parent / "new-export") as writer:
        writer.write_static("mode", pt.ones(len(loader.vertices)))
    assert writer.case_file.name == path.name
    with pytest.raises(FileExistsError):
        EnSightWriter.from_loader(loader, writer.case_file.parent)


def test_mesh_copy_and_noncontiguous_tensors(exports, tmp_path):
    loader = EnSightDataloader(source_case(exports), part="internalMesh")
    mesh = loader.mesh
    mesh.GetPoints().SetPoint(0, 10, 20, 30)
    assert loader.mesh.GetPoint(0) != (10, 20, 30)
    data = (
        pt.arange(len(loader.vertices) * 3, dtype=pt.float64)
        .reshape(3, -1)
        .T.requires_grad_()
    )
    assert not data.is_contiguous()
    before = data.detach().clone()
    with EnSightWriter.from_loader(loader, tmp_path / "output") as writer:
        writer.write_static("mode", data)
    result = EnSightDataloader(writer.case_file, dtype=pt.float64)
    pt.testing.assert_close(result.load_snapshot("mode", "0"), before)
    pt.testing.assert_close(data.detach(), before)


@pytest.fixture
def writer(exports, tmp_path):
    loader = EnSightDataloader(source_case(exports), part="internalMesh")
    return EnSightWriter.from_loader(loader, tmp_path / "output")


@pytest.mark.parametrize(
    "fields,time,match",
    [
        ({}, "0", "nonempty"),
        ({"bad": pt.ones(399)}, "0", "shape"),
        ({"bad": pt.ones(400, 2)}, "0", "shape"),
        ({"bad": pt.ones(400, dtype=pt.int64)}, "0", "floating-point"),
        ({"../bad": pt.ones(400)}, "0", "field name"),
        ({"bad": pt.full((400,), float("nan"))}, "0", "nonfinite"),
        ({"bad": pt.full((400,), 1e100, dtype=pt.float64)}, "0", "float32 range"),
        ({"p": pt.ones(400)}, float("inf"), "finite"),
        ({"p": pt.ones(400)}, "0\n1", "finite"),
    ],
)
def test_invalid_snapshot_does_not_publish(writer, fields, time, match):
    with pytest.raises(ValueError, match=match):
        writer.write_snapshot(fields, time)
    assert writer.write_times == []
    assert not writer.case_file.exists()
    writer.write_snapshot({"p": pt.ones(400)}, "0")
    writer.close()


def test_schema_and_closed_writer(writer):
    writer.write_static("mode", pt.ones(400))
    with pytest.raises(ValueError, match="already been written"):
        writer.write_static("mode", pt.ones(400))
    with pytest.raises(ValueError, match="static field names"):
        writer.write_snapshot({"mode": pt.ones(400)}, "0")
    writer.write_snapshot({"p": pt.ones(400)}, "0.1")
    for time in ["0.1", "0", "-1"]:
        with pytest.raises(ValueError, match="unique and increasing"):
            writer.write_snapshot({"p": pt.ones(400)}, time)
    for fields in [
        {"U": pt.ones(400, 3)},
        {"p": pt.ones(400, 3)},
        {"p": pt.ones(400), "q": pt.ones(400)},
    ]:
        with pytest.raises(ValueError, match="same field schema"):
            writer.write_snapshot(fields, "0.2")
    before = writer.case_file.read_bytes()
    with pytest.raises(ValueError, match="is complex"):
        writer.write_static(
            "complex_mode", pt.ones(400, dtype=pt.complex64), complex_mode="error"
        )
    with pytest.raises(ValueError, match="Duplicate output field name"):
        writer.write_snapshot(
            {"p": pt.ones(400, dtype=pt.complex64), "p_real": pt.ones(400)}, "0.2"
        )
    assert writer.case_file.read_bytes() == before
    writer.close()
    writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        writer.write_static("new", pt.ones(400))
    with pytest.raises(RuntimeError, match="closed"):
        writer.write_series({"p": pt.ones(400, 2)}, ["0.2", "0.3"])


def test_series_validation(writer):
    for fields, times in [
        ({}, [0]),
        ({"p": pt.ones(400, 2)}, []),
        ({"p": pt.ones(400)}, [0, 1]),
        ({"p": pt.ones(400, 3)}, [0, 1]),
    ]:
        with pytest.raises(ValueError):
            writer.write_series(fields, times)
    assert not writer.case_file.exists()


def test_failed_write_preserves_published_case(writer, monkeypatch):
    writer.write_snapshot({"p": pt.ones(400), "U": pt.ones(400, 3)}, "0.1")
    before = writer.case_file.read_bytes()
    original = writer._write_variable

    def fail_second(path, name, values):
        if name == "U":
            raise OSError("disk write failed")
        return original(path, name, values)

    with monkeypatch.context() as patch:
        patch.setattr(writer, "_write_variable", fail_second)
        with pytest.raises(OSError, match="disk write failed"):
            writer.write_snapshot(
                {"p": pt.ones(400) * 2, "U": pt.ones(400, 3) * 2}, "0.2"
            )
    assert writer.case_file.read_bytes() == before
    assert writer.write_times == ["0.1"]
    assert not (writer.case_file.parent / "data/00000001").exists()
    assert not list(writer.case_file.parent.glob(".ensight-*"))
    result = EnSightDataloader(writer.case_file)
    pt.testing.assert_close(result.load_snapshot("p", "0.1"), pt.ones(400))
    writer.write_snapshot({"p": pt.ones(400) * 2, "U": pt.ones(400, 3) * 2}, "0.2")


def test_context_exception_preserves_results(exports, tmp_path):
    loader = EnSightDataloader(source_case(exports), part="internalMesh")
    with pytest.raises(RuntimeError, match="analysis failed"):
        with EnSightWriter.from_loader(loader, tmp_path / "output") as writer:
            writer.write_static("mode", pt.ones(400))
            raise RuntimeError("analysis failed")
    result = EnSightDataloader(writer.case_file)
    pt.testing.assert_close(result.load_snapshot("mode", "0"), pt.ones(400))


def test_empty_writer(exports, tmp_path):
    with EnSightWriter.from_case(
        source_case(exports), tmp_path / "output", part="internalMesh"
    ) as writer:
        pass
    result = EnSightDataloader(writer.case_file)
    assert result.field_names == {"0": []}
    assert len(result.vertices) == 400


def test_pod_mode_and_reconstruction(exports, tmp_path):
    from flowtorch.analysis import SVD

    loader = EnSightDataloader(
        source_case(exports), part="internalMesh", dtype=pt.float64
    )
    velocity = loader.load_snapshot("U", loader.write_times)
    decomposition = SVD(
        velocity.reshape(-1, len(loader.write_times)), rank=1, mode="svd"
    )
    mode = decomposition.U[:, 0].reshape(velocity.shape[:-1])
    reconstruction = decomposition.reconstruct().reshape_as(velocity)
    with EnSightWriter.from_loader(loader, tmp_path / "pod") as writer:
        writer.write_static("U_mode_001", mode)
        writer.write_series({"U_reconstructed": reconstruction}, loader.write_times)
    result = EnSightDataloader(writer.case_file, dtype=pt.float64)
    pt.testing.assert_close(
        result.load_snapshot("U_mode_001", result.write_times[0]),
        mode,
        rtol=2e-6,
        atol=1e-8,
    )
    pt.testing.assert_close(
        result.load_snapshot("U_reconstructed", result.write_times),
        reconstruction,
        rtol=2e-6,
        atol=1e-8,
    )


@pytest.mark.skipif(not pt.cuda.is_available(), reason="CUDA device unavailable")
def test_gpu_tensor_transfer(exports, tmp_path):
    loader = EnSightDataloader(source_case(exports), part="internalMesh")
    data = pt.arange(400, dtype=pt.float32, device="cuda")
    with EnSightWriter.from_loader(loader, tmp_path / "gpu") as writer:
        writer.write_static("mode", data)
    result = EnSightDataloader(writer.case_file)
    pt.testing.assert_close(result.load_snapshot("mode", "0"), data.cpu())


@pytest.mark.parametrize(
    "element,points,volume",
    [
        ("tetra4", [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], 1 / 6),
        (
            "penta6",
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1)],
            1 / 2,
        ),
        (
            "pyramid5",
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 0.5, 1)],
            1 / 3,
        ),
    ],
)
@pytest.mark.parametrize("format", ["ascii", "binary"])
def test_other_linear_cells(tmp_path, element, points, volume, format):
    source = tmp_path / "source"
    source.mkdir()
    coordinates = "\n".join(str(v) for v in np.array(points).T.reshape(-1))
    (source / "geometry").write_text(
        f"geometry\nFlowTorch test\nnode id assign\nelement id assign\npart\n9\nvolume\ncoordinates\n{len(points)}\n{coordinates}\n{element}\n1\n"
        + " ".join(str(i + 1) for i in range(len(points)))
        + "\n"
    )
    (source / "input.case").write_text(
        "FORMAT\ntype: ensight gold\nGEOMETRY\nmodel: geometry\n"
    )
    loader = EnSightDataloader(source / "input.case")
    with EnSightWriter.from_loader(
        loader, tmp_path / "output", format=format
    ) as writer:
        writer.write_static("mode", pt.tensor([7.0]))
    result = EnSightDataloader(writer.case_file)
    assert_mesh_equal(loader.mesh, result.mesh)
    assert result.weights.item() == pytest.approx(volume, rel=1e-6)
    pt.testing.assert_close(result.load_snapshot("mode", "0"), pt.tensor([7.0]))


def test_higher_order_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    points = np.array(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0.5, 0, 0), (0.5, 0.5, 0), (0, 0.5, 0)]
    )
    coordinates = "\n".join(str(v) for v in points.T.reshape(-1))
    (source / "geometry").write_text(
        f"geometry\nFlowTorch test\nnode id assign\nelement id assign\npart\n1\nsurface\ncoordinates\n6\n{coordinates}\ntria6\n1\n1 2 3 4 5 6\n"
    )
    (source / "input.case").write_text(
        "FORMAT\ntype: ensight gold\nGEOMETRY\nmodel: geometry\n"
    )
    with pytest.raises(ValueError, match="only linear cells"):
        EnSightWriter.from_case(source / "input.case", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_numeric_case_stem(exports, tmp_path):
    source = tmp_path / "source"
    shutil.copytree(source_case(exports).parent, source)
    (source / "ensight_cavity.case").rename(source / "2026.case")
    with EnSightWriter.from_case(
        source / "2026.case", tmp_path / "output", part="internalMesh"
    ) as writer:
        writer.write_static("mode", pt.ones(400))
    assert writer.case_file.name == "2026.case"
    pt.testing.assert_close(
        EnSightDataloader(writer.case_file).load_snapshot("mode", "0"), pt.ones(400)
    )
