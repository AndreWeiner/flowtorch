"""Self-contained EnSight tests; no OpenFOAM, MPI or external datasets needed."""

from pathlib import Path
import shutil
from zipfile import ZipFile

import numpy as np
import pytest
import torch as pt

pytest.importorskip("vtkmodules.vtkIOEnSight")
from flowtorch.data import EnSightDataloader


@pytest.fixture(scope="module")
def exports(tmp_path_factory):
    folder = tmp_path_factory.mktemp("ensight")
    with ZipFile(Path(__file__).parent / "fixtures/ensight_cavity.zip") as archive:
        archive.extractall(folder)
    return folder


def case_path(exports, source="volume", encoding="binary", variant=""):
    stem = variant or "ensight_cavity"
    return (
        exports / variant / "postProcessing" / f"{source}_{encoding}" / f"{stem}.case"
    )


@pytest.mark.parametrize("variant", ["", "dual", "parallel"])
@pytest.mark.parametrize("encoding", ["ascii", "binary"])
@pytest.mark.parametrize(
    "part,measure",
    [("internalMesh", 0.0001), ("movingWall", 0.001), ("fixedWalls", 0.003)],
)
def test_volume_parts(exports, variant, encoding, part, measure):
    loader = EnSightDataloader(
        case_path(exports, encoding=encoding, variant=variant),
        part=part,
        dtype=pt.float64,
    )
    assert loader.parts == ["internalMesh", "movingWall", "fixedWalls"]
    assert [float(t) for t in loader.write_times] == [0.05, 0.1]
    assert set(loader.field_names) == set(loader.write_times)
    assert all(
        set(fields) == {"p", "U", "S", "T"} for fields in loader.field_names.values()
    )
    assert loader.weights.sum().item() == pytest.approx(measure, rel=1e-6)
    assert pt.all(loader.weights > 0)
    assert loader.vertices.shape == (len(loader.weights), 3)
    assert loader.vertices.dtype == pt.float64
    first, last = loader.write_times
    p = loader.load_snapshot("p", first)
    assert p.shape == (len(loader.weights),)
    series = loader.load_snapshot("p", [last, first, last])
    assert series.shape == (len(p), 3)
    pt.testing.assert_close(series[:, 1], p)
    pt.testing.assert_close(series[:, 0], series[:, 2])
    velocity, symm, tensor = loader.load_snapshot(["U", "S", "T"], first)
    assert velocity.shape == (len(p), 3)
    assert symm.shape == (len(p), 6)
    assert tensor.shape == (len(p), 9)
    pt.testing.assert_close(symm, pt.arange(1, 7, dtype=pt.float64).expand(len(p), -1))
    pt.testing.assert_close(
        tensor, pt.arange(1, 10, dtype=pt.float64).expand(len(p), -1)
    )
    fields = loader.load_snapshot(["p", "U", "S", "T"], loader.write_times)
    for field, snapshot in zip(fields, [p, velocity, symm, tensor]):
        pt.testing.assert_close(field[..., 0], snapshot)
    assert loader.snapshot_shape("S", first) == (len(p), 6)
    assert loader.load_snapshot_slice("p", first, slice(2, 5)).shape == (3,)
    # Caller-owned snapshots survive subsequent reader updates.
    original = p.clone()
    loader.load_snapshot("p", last)
    pt.testing.assert_close(p, original)


@pytest.mark.parametrize("variant", ["", "dual", "parallel"])
@pytest.mark.parametrize("encoding", ["ascii", "binary"])
def test_patches_and_planes(exports, variant, encoding):
    patch = EnSightDataloader(
        case_path(exports, "patch", encoding, variant), part="movingWall"
    )
    assert patch.weights.sum().item() == pytest.approx(0.001, rel=1e-6)
    root = exports / variant / "postProcessing" / f"cuts_{encoding}"
    for plane, area in [("midplane", 0.01), ("oblique", 0.00757458847), ("lid", 0.001)]:
        loader = EnSightDataloader(root / plane / f"{plane}.case")
        assert loader.parts == ["geometry"]
        assert loader.weights.sum().item() == pytest.approx(area, rel=1e-6)
        if plane == "oblique":
            # nsided, quad4, tria3 occur within the same field array.
            expected = 382 if variant == "dual" else 319
            assert len(loader.weights) == expected
        symm, tensor = loader.load_snapshot(["S", "T"], loader.write_times)
        assert symm.shape == (len(loader.weights), 6, 2)
        pt.testing.assert_close(
            symm, pt.arange(1, 7, dtype=symm.dtype)[None, :, None].expand_as(symm)
        )
        pt.testing.assert_close(
            tensor,
            pt.arange(1, 10, dtype=tensor.dtype)[None, :, None].expand_as(tensor),
        )


@pytest.mark.parametrize("variant", ["", "dual", "parallel"])
@pytest.mark.parametrize(
    "part,measure", [("internalMesh", 0.0001), ("movingWall", 0.001)]
)
def test_point_weights(exports, variant, part, measure):
    loader = EnSightDataloader(
        case_path(exports, "nodes", variant=variant), part=part, association="point"
    )
    assert loader.weights.sum().item() == pytest.approx(measure, rel=1e-6)
    assert pt.all(loader.weights > 0)
    assert (
        loader.load_snapshot("U", loader.write_times[0]).shape == loader.vertices.shape
    )
    symm = loader.load_snapshot("S", loader.write_times[0])
    pt.testing.assert_close(symm, pt.arange(1, 7, dtype=symm.dtype).expand_as(symm))
    assert loader.vertices is loader.vertices
    assert loader.weights is loader.weights


@pytest.mark.parametrize("variant", ["", "dual", "parallel"])
def test_encodings_agree(exports, variant):
    for source in ("volume", "patch"):
        for part in (
            ["internalMesh", "movingWall", "fixedWalls"]
            if source == "volume"
            else ["movingWall", "fixedWalls"]
        ):
            a = EnSightDataloader(
                case_path(exports, source, "ascii", variant), part=part
            )
            b = EnSightDataloader(
                case_path(exports, source, "binary", variant), part=part
            )
            pt.testing.assert_close(a.vertices, b.vertices)
            pt.testing.assert_close(a.weights, b.weights)
            for field in ("p", "U", "S", "T"):
                pt.testing.assert_close(
                    a.load_snapshot(field, a.write_times),
                    b.load_snapshot(field, b.write_times),
                    rtol=2e-5,
                    atol=1e-7,
                )
    for plane in ("midplane", "oblique", "lid"):
        loaders = [
            EnSightDataloader(
                exports
                / variant
                / "postProcessing"
                / f"cuts_{enc}"
                / plane
                / f"{plane}.case"
            )
            for enc in ("ascii", "binary")
        ]
        a, b = loaders
        pt.testing.assert_close(a.vertices, b.vertices)
        pt.testing.assert_close(a.weights, b.weights)
        for field in ("p", "U", "S", "T"):
            pt.testing.assert_close(
                a.load_snapshot(field, a.write_times),
                b.load_snapshot(field, b.write_times),
                rtol=2e-5,
                atol=1e-7,
            )


def test_parallel_export_matches_serial(exports):
    loaders = [
        EnSightDataloader(case_path(exports, variant=v), part="internalMesh")
        for v in ("", "parallel")
    ]
    a, b = loaders
    order = [
        np.lexsort(np.round(loader.vertices.numpy(), 7).T[::-1]) for loader in loaders
    ]
    pt.testing.assert_close(a.vertices[order[0]], b.vertices[order[1]])
    for field in ("p", "U", "S", "T"):
        pt.testing.assert_close(
            a.load_snapshot(field, a.write_times)[order[0]],
            b.load_snapshot(field, b.write_times)[order[1]],
            rtol=1e-4,
            atol=2e-6,
        )


def test_errors_and_argument_validation(exports, tmp_path):
    path = case_path(exports)
    with pytest.raises(ValueError, match="Select a part"):
        EnSightDataloader(path)
    with pytest.raises(ValueError, match="Unknown part"):
        EnSightDataloader(path, part="missing")
    with pytest.raises(ValueError, match="part name"):
        EnSightDataloader(path, part=0)
    with pytest.raises(ValueError, match="association"):
        EnSightDataloader(path, part="internalMesh", association="face")
    with pytest.raises(ValueError, match="floating-point"):
        EnSightDataloader(path, part="internalMesh", dtype=pt.int64)
    with pytest.raises(FileNotFoundError):
        EnSightDataloader(tmp_path / "absent.case")
    loader = EnSightDataloader(path, part="internalMesh")
    for fields, times, match in [
        ("p", "0.05", "Unknown time"),
        ("missing", loader.write_times[0], "Unknown cell field"),
        ([], loader.write_times, "must not be empty"),
        ("p", [], "must not be empty"),
        ("p", [0.05], "list of strings"),
    ]:
        with pytest.raises(ValueError, match=match):
            loader.load_snapshot(fields, times)
    nodes = EnSightDataloader(case_path(exports, "nodes"), part="internalMesh")
    assert nodes.field_names == {t: [] for t in nodes.write_times}
    with pytest.raises(ValueError, match="Unknown cell field"):
        nodes.load_snapshot("p", nodes.write_times[0])
    # Metadata copies cannot corrupt subsequent calls.
    loader.write_times.clear()
    loader.parts.clear()
    loader.field_names[loader.write_times[0]].clear()
    assert len(loader.write_times) == 2
    assert loader.field_names[loader.write_times[0]]


@pytest.fixture
def mutable_case(exports, tmp_path):
    source = case_path(exports, encoding="ascii").parent
    folder = tmp_path / "case"
    shutil.copytree(source, folder)
    return folder / "ensight_cavity.case"


def test_nonconsecutive_numbers_and_time_labels(mutable_case):
    path = mutable_case
    data = path.parent / "data"
    (data / "00000001").rename(data / "00000013")
    (data / "00000000").rename(data / "00000007")
    text = path.read_text().replace("data/00000000/geometry", "data/00000007/geometry")
    text = text.replace(
        "filename start number:  0\nfilename increment:     1",
        "filename numbers:\n7 13",
    )
    text = text.replace("5.00000e-02", "0.050").replace("1.00000e-01", "0.1000")
    path.write_text(text)
    loader = EnSightDataloader(path, part="internalMesh")
    assert loader.write_times == ["0.050", "0.1000"]
    assert loader.load_snapshot("p", loader.write_times).shape == (400, 2)


@pytest.mark.parametrize(
    "old,new,match",
    [
        (
            "model:          data/00000000/geometry",
            "model: 1 data/********/geometry",
            "Time-dependent geometry",
        ),
        (
            "5.00000e-02  1.00000e-01",
            "5.00000e-02  5.00000e-02",
            "unique and increasing",
        ),
        (
            "tensor symm per element",
            "complex scalar per element",
            "Unsupported EnSight variable",
        ),
    ],
)
def test_unsupported_descriptors(mutable_case, old, new, match):
    mutable_case.write_text(mutable_case.read_text().replace(old, new))
    with pytest.raises(ValueError, match=match):
        EnSightDataloader(mutable_case, part="internalMesh")


def test_missing_snapshot_file(mutable_case):
    loader = EnSightDataloader(mutable_case, part="internalMesh")
    (mutable_case.parent / "data/00000001/p").unlink()
    with pytest.raises(FileNotFoundError, match="Missing EnSight field file"):
        loader.load_snapshot("p", loader.write_times[-1])


def test_reader_errors_are_raised(mutable_case):
    (mutable_case.parent / "data/00000000/geometry").write_text("broken geometry\n")
    with pytest.raises(ValueError, match="Invalid EnSight Gold geometry header"):
        EnSightDataloader(mutable_case, part="internalMesh")


def test_static_polygon_point_data(tmp_path):
    geometry = """Static polygon
five vertices
node id assign
element id assign
part
7
surface
coordinates
5
0
2
2
1
0
0
0
1
2
1
0
0
0
0
0
nsided
1
5
1 2 3 4 5
"""
    (tmp_path / "geometry").write_text(geometry)
    (tmp_path / "p").write_text("p\npart\n7\ncoordinates\n0\n1\n2\n3\n4\n")
    path = tmp_path / "static.case"
    path.write_text(
        "FORMAT\ntype: ensight gold\n\nGEOMETRY\nmodel: geometry\n\nVARIABLE\nscalar per node: p p\n"
    )
    loader = EnSightDataloader(path, association="point", dtype=pt.float64)
    assert loader.write_times == ["0"]
    pt.testing.assert_close(
        loader.load_snapshot("p", "0"), pt.arange(5, dtype=pt.float64)
    )
    pt.testing.assert_close(loader.weights, pt.full((5,), 0.6, dtype=pt.float64))
    cells = EnSightDataloader(path)
    pt.testing.assert_close(cells.vertices, pt.tensor([[1.0, 0.8, 0.0]]))
    assert cells.weights.item() == pytest.approx(3.0)
    assert cells.field_names == {"0": []}
    # Equal numbers of cells/points do not make mixed-dimensional parts valid.
    (tmp_path / "geometry").write_text(geometry + "tetra4\n1\n1 2 3 4\n")
    with pytest.raises(ValueError, match="only 2D or only 3D"):
        EnSightDataloader(path)


def test_duplicate_part_names(mutable_case):
    geometry = mutable_case.parent / "data/00000000/geometry"
    geometry.write_text(geometry.read_text().replace("fixedWalls", "movingWall"))
    with pytest.raises(ValueError, match="unique, nonempty names"):
        EnSightDataloader(mutable_case, part="movingWall")


def test_file_sets_rejected(mutable_case):
    mutable_case.write_text(
        mutable_case.read_text() + "\nFILE\nfile set: 1\nnumber of steps: 2\n"
    )
    with pytest.raises(ValueError, match="file sets are not supported"):
        EnSightDataloader(mutable_case, part="internalMesh")


def test_vtk_read_errors_are_raised(mutable_case):
    (mutable_case.parent / "data/00000000/p").write_text(
        "p\npart\n1\ninvalid_element_type\n"
    )
    with pytest.raises(ValueError, match="VTK could not read EnSight data"):
        EnSightDataloader(mutable_case, part="internalMesh")
