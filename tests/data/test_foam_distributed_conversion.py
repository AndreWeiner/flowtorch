"""Static processor geometry, field streaming and append without reconstruction."""

import json
import shutil
import weakref
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest
import torch

from flowtorch.data import FOAM2HDF5, FOAMDataloader, HDF5Dataloader
from flowtorch.data._foam_stream import field_chunks
from tests.data.test_foam_stream import write_field, foam_header


def make_chain(root, count, offset=0, *, binary=False, compressed=False):
    """Unit cubes sharing internal faces, with known centres/volumes and fields."""
    mesh = root / "constant/polyMesh"
    mesh.mkdir(parents=True)

    def listing(name, kind, values, extra=""):
        (mesh / name).write_text(
            f"FoamFile {{format ascii; class {kind}; object {name}; {extra}}}\n"
            + f"{len(values)}\n(\n"
            + "\n".join(values)
            + "\n)\n"
        )

    points = [
        (offset + x, y, z)
        for x in range(count + 1)
        for y, z in [(0, 0), (1, 0), (1, 1), (0, 1)]
    ]
    listing("points", "vectorField", [f"({x} {y} {z})" for x, y, z in points])
    faces, owners, neighbours = [], [], {}
    lookup = {}
    for cell in range(count):
        a, b = 4 * cell, 4 * (cell + 1)
        v = [a, b, b + 1, a + 1, a + 3, b + 3, b + 2, a + 2]
        for ids in [
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ]:
            face = [v[i] for i in ids]
            key = tuple(sorted(face))
            if key in lookup:
                neighbours[lookup[key]] = cell
            else:
                lookup[key] = len(faces)
                faces.append(face)
                owners.append(cell)
    order = sorted(range(len(faces)), key=lambda i: i not in neighbours)
    listing(
        "faces", "faceList", ["4(" + " ".join(map(str, faces[i])) + ")" for i in order]
    )
    listing(
        "owner",
        "labelList",
        [str(owners[i]) for i in order],
        f'note "nPoints:{len(points)} nCells:{count} nFaces:{len(faces)} nInternalFaces:{len(neighbours)}";',
    )
    listing(
        "neighbour", "labelList", [str(neighbours[i]) for i in order if i in neighbours]
    )
    listing(
        "boundary",
        "polyBoundaryMesh",
        [
            f"walls {{type wall; nFaces {len(faces)-len(neighbours)}; startFace {len(neighbours)};}}"
        ],
    )
    for time in ["0", "1", "2"]:
        folder = root / time
        folder.mkdir()
        p = np.arange(offset, offset + count, dtype=float) + 10 * int(time) + 0.25
        write_field(folder / "p", p, binary=binary, compressed=compressed)
        write_field(
            folder / "U",
            np.stack([p, p + 1, -p], axis=1),
            dim=3,
            binary=binary,
            compressed=compressed,
        )
        (folder / "uniform").write_bytes(foam_header() + b"internalField uniform 2.5;")
        (folder / "phi").write_text(
            "FoamFile {format ascii; class surfaceScalarField;}"
        )
    return root


def decomposed(root, **options):
    make_chain(root / "processor0", 2, **options)
    make_chain(root / "processor1", 3, offset=2, **options)
    return root


@pytest.mark.parametrize(
    "binary,compressed", [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("codec", ["gzip", "zstd"])
def test_conversion_and_append(tmp_path, monkeypatch, binary, compressed, dtype, codec):
    if codec == "zstd":
        pytest.importorskip("hdf5plugin")
    source = decomposed(tmp_path / "source", binary=binary, compressed=compressed)
    converter = FOAM2HDF5(str(source), dtype=dtype)
    # Conversion must not invoke the full-field or concatenated mesh paths.
    monkeypatch.setattr(
        FOAMDataloader, "load_snapshot", lambda *a: pytest.fail("full field")
    )
    monkeypatch.setattr(
        type(converter._loader._mesh),
        "_load_mesh",
        lambda *a: pytest.fail("cached full mesh"),
    )
    field = tmp_path / "out/fields.h5"
    converter.convert(
        str(field),
        times=["0"],
        spatial_chunk=1,
        compression=codec,
        shuffle=True,
        geometry_compression=codec,
    )
    geometry = field.with_suffix(".geometry.h5")
    before_geometry = geometry.read_bytes()
    with h5py.File(field) as f:
        old_chunk = f["frames/t000000/z000000/f000000"].id.read_direct_chunk((0, 0))
    converter.convert(str(field), times=["2", "1"], append=True, spatial_chunk=2)
    assert geometry.read_bytes() == before_geometry
    assert converter._mesh_points is None and converter._topology is None
    with h5py.File(field) as f:
        assert (
            f["frames/t000000/z000000/f000000"].id.read_direct_chunk((0, 0))
            == old_chunk
        )
        assert len(json.loads(f.attrs["committed_frames"])) == 3
    with HDF5Dataloader(field, dtype=dtype) as reader:
        assert reader.zone_names == ["processor0", "processor1"]
        assert reader.write_times == ["0", "1", "2"]
        expected = torch.arange(5, dtype=dtype) + 20.25
        torch.testing.assert_close(reader.load_snapshot("p", "2"), expected)
        torch.testing.assert_close(
            reader.load_snapshot_slice("p", "2", slice(1, 4)), expected[1:4]
        )
        torch.testing.assert_close(
            reader.load_snapshot("uniform", "1"), torch.full((5,), 2.5, dtype=dtype)
        )
        torch.testing.assert_close(reader.weights, torch.ones(5, dtype=dtype))
        torch.testing.assert_close(
            reader.vertices[:, 0], torch.arange(5, dtype=torch.float64) + 0.5
        )
        assert "phi" not in reader.field_names["0"]
    with HDF5Dataloader(field, dtype=dtype, zones=["processor1"]) as reader:
        torch.testing.assert_close(
            reader.load_snapshot("p", "1"), torch.arange(2, 5, dtype=dtype) + 10.25
        )
    frames = ET.parse(field.with_suffix(".xdmf")).findall("./Domain/Grid/Grid")
    assert [float(f.find("Time").get("Value")) for f in frames] == [0, 1, 2]
    for frame in frames:
        assert [g.get("Name") for g in frame.findall("Grid")] == [
            "processor0",
            "processor1",
        ]
        assert [
            int(g.find("Topology").get("NumberOfElements"))
            for g in frame.findall("Grid")
        ] == [2, 3]
    with h5py.File(geometry) as f:
        for group in f["zones"].values():
            assert group["points"].dtype == np.float64
            assert group["weights"].dtype == np.float64


def test_matches_reconstructed_reference(tmp_path):
    distributed = decomposed(tmp_path / "distributed")
    reconstructed = make_chain(tmp_path / "reconstructed", 5)
    for root in [distributed, reconstructed]:
        FOAM2HDF5(str(root), dtype=torch.float64).convert(
            "fields.h5",
            geometry_file=str(root / "geometry.h5"),
            fields=["p", "U"],
            times=["1"],
        )
    with (
        HDF5Dataloader(distributed / "fields.h5", dtype=torch.float64) as a,
        HDF5Dataloader(reconstructed / "fields.h5", dtype=torch.float64) as b,
    ):
        torch.testing.assert_close(a.weights, b.weights)
        torch.testing.assert_close(a.vertices, b.vertices)
        for name in ["p", "U"]:
            torch.testing.assert_close(
                a.load_snapshot(name, "1"), b.load_snapshot(name, "1")
            )


@pytest.mark.parametrize(
    "problem",
    [
        "missing_time",
        "missing_field",
        "class",
        "count",
        "moving",
        "duplicate",
        "geometry",
        "decomposition",
        "precision",
        "compression",
    ],
)
def test_preflight_preserves_existing_output(tmp_path, problem):
    source = decomposed(tmp_path / "source")
    converter = FOAM2HDF5(str(source))
    output = tmp_path / "fields.h5"
    converter.convert(str(output), fields=["p"], times=["0"], compression="gzip")
    paths = [output, output.with_suffix(".geometry.h5"), output.with_suffix(".xdmf")]
    before = [p.read_bytes() for p in paths]
    options = dict(times=["1"], append=True)
    proc = source / "processor1"
    if problem == "missing_time":
        shutil.rmtree(proc / "1")
    elif problem == "missing_field":
        (proc / "1/p").unlink()
    elif problem == "class":
        write_field(proc / "1/p", np.zeros((3, 3)), dim=3)
    elif problem == "count":
        write_field(proc / "1/p", np.zeros(2))
    elif problem == "moving":
        (proc / "0.5/polyMesh").mkdir(parents=True)
    elif problem == "duplicate":
        options["times"] = ["0"]
    elif problem == "geometry":
        points = proc / "constant/polyMesh/points"
        points.write_text(points.read_text().replace("(2 0 0)", "(2.1 0 0)"))
    elif problem == "decomposition":
        make_chain(source / "processor2", 1, offset=5)
    elif problem == "precision":
        converter = FOAM2HDF5(str(source), dtype=torch.float64)
    elif problem == "compression":
        options["compression"] = "zstd"
    with pytest.raises((ValueError, FileNotFoundError)):
        converter.convert(str(output), **options)
    assert [p.read_bytes() for p in paths] == before


def test_geometry_reuse_and_release(tmp_path, monkeypatch):
    source = decomposed(tmp_path / "source")
    from flowtorch.data import unstructured

    original = unstructured.UnstructuredGeometry
    references = []

    def tracked(*args, **kwargs):
        assert all(ref() is None for ref in references), "previous processor retained"
        geom = original(*args, **kwargs)
        references.append(weakref.ref(geom))
        return geom

    monkeypatch.setattr(unstructured, "UnstructuredGeometry", tracked)
    geometry = tmp_path / "shared.h5"
    converter = FOAM2HDF5(str(source))
    converter.convert(
        str(tmp_path / "a.h5"), fields=["p"], times=["0"], geometry_file=str(geometry)
    )
    before = geometry.read_bytes()
    converter.convert(
        str(tmp_path / "b.h5"), fields=["U"], times=["1"], geometry_file=str(geometry)
    )
    assert geometry.read_bytes() == before
    assert all(ref() is None for ref in references)


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("dim", [1, 3, 6, 9])
def test_sequential_field_chunks(tmp_path, binary, compressed, dim):
    values = np.arange(7 * dim, dtype=float).reshape(7, dim) + 0.25
    if dim == 1:
        values = values[:, 0]
    write_field(tmp_path / "p", values, dim=dim, binary=binary, compressed=compressed)
    # Spy on seeks: binary chunks may seek relative zero but never rewind or
    # skip prior chunks; ASCII keeps its parser alive across chunk boundaries.
    import flowtorch.data._foam_stream as native

    original = native.foam_open
    opened = []
    seeks = []
    # Count opens through a narrow context wrapper, preserving the native stream.
    from contextlib import contextmanager

    @contextmanager
    def tracked(path):
        opened.append(path)
        with original(path) as stream:
            seek = stream.seek

            def tracked_seek(offset, whence=0):
                seeks.append((offset, whence))
                return seek(offset, whence)

            stream.seek = tracked_seek
            yield stream

    from unittest.mock import patch

    with patch.object(native, "foam_open", tracked):
        chunks = list(field_chunks(str(tmp_path / "p"), 7, 2))
    assert len(opened) == 1
    assert all(offset == 0 and whence == 1 for offset, whence in seeks)
    assert [part.stop - part.start for part, _ in chunks] == [2, 2, 2, 1]
    np.testing.assert_array_equal(np.concatenate([v for _, v in chunks]), values)


def test_append_abort_and_retry(tmp_path, monkeypatch):
    import flowtorch.data._foam_stream as native

    root = decomposed(tmp_path / "source")
    output = tmp_path / "fields.h5"
    converter = FOAM2HDF5(str(root))
    converter.convert(str(output), fields=["p"], times=["0"])
    descriptor = output.with_suffix(".xdmf").read_bytes()
    geometry = output.with_suffix(".geometry.h5").read_bytes()
    original = native.field_chunks

    def interrupted(path, count, chunk):
        if "processor1" in path:
            raise RuntimeError("interrupted")
        yield from original(path, count, chunk)

    with monkeypatch.context() as patch:
        patch.setattr(native, "field_chunks", interrupted)
        with pytest.raises(RuntimeError, match="interrupted"):
            converter.convert(str(output), times=["1"], append=True)
    with HDF5Dataloader(output) as reader:
        assert reader.write_times == ["0"]
    assert output.with_suffix(".xdmf").read_bytes() == descriptor
    assert output.with_suffix(".geometry.h5").read_bytes() == geometry
    converter.convert(str(output), times=["1"], append=True)
    with HDF5Dataloader(output) as reader:
        assert reader.write_times == ["0", "1"]
        torch.testing.assert_close(
            reader.load_snapshot("p", "1"), torch.arange(5).float() + 10.25
        )


def test_numeric_processor_order(tmp_path):
    for i in range(12):
        make_chain(tmp_path / f"processor{i}", 1, offset=i)
    output = FOAM2HDF5(str(tmp_path)).convert("fields.h5", fields=["p"], times=["0"])
    with HDF5Dataloader(output) as reader:
        assert reader.zone_names == [f"processor{i}" for i in range(12)]
        torch.testing.assert_close(
            reader.load_snapshot("p", "0"), torch.arange(12).float() + 0.25
        )


@pytest.mark.parametrize("damage", ["short", "extra", "closing", "semicolon"])
@pytest.mark.parametrize("binary", [False, True])
def test_stream_rejects_invalid_payload(tmp_path, damage, binary):
    path = tmp_path / "p"
    write_field(path, np.arange(5, dtype=float), binary=binary)
    data = path.read_bytes()
    if damage == "short":
        data = data.replace(b"5\n(", b"6\n(")
        count = 6
    elif damage == "extra":
        data = data.replace(b"5\n(", b"4\n(")
        count = 4
    elif damage == "closing":
        data = data.replace(b");\nboundaryField", b";\nboundaryField")
        count = 5
    else:
        data = data.replace(b");\nboundaryField", b")\nboundaryField")
        count = 5
    path.write_bytes(data)
    with pytest.raises(ValueError):
        list(field_chunks(str(path), count, 2))


@pytest.mark.integration
@pytest.mark.parametrize(
    "dataset", ["of_cavity_ascii_parallel", "of_cavity_binary_parallel"]
)
def test_external_decomposed_cavity(tmp_path, dataset):
    from flowtorch import DATASETS
    from pathlib import Path

    if dataset not in DATASETS or not Path(DATASETS[dataset]).is_dir():
        pytest.skip(f"External dataset unavailable: {dataset}")
    source = Path(DATASETS[dataset])
    target = tmp_path / "source"
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("*.h5", "*.hdf5", "*.xdmf", "*.xmf"),
    )
    original = FOAMDataloader(str(source), dtype=torch.float64)
    labels = original.write_times
    converter = FOAM2HDF5(str(target), dtype=torch.float64)
    output = tmp_path / "fields.h5"
    converter.convert(
        str(output),
        fields=["p", "U"],
        times=labels[:-1],
        compression="gzip",
        shuffle=True,
        spatial_chunk=31,
    )
    converter.convert(str(output), times=labels[-1:], append=True, spatial_chunk=31)
    with HDF5Dataloader(output, dtype=torch.float64) as reader:
        torch.testing.assert_close(reader.vertices, original.vertices)
        torch.testing.assert_close(reader.weights, original.weights)
        assert reader.write_times == labels
        for time in labels:
            for name in ["p", "U"]:
                torch.testing.assert_close(
                    reader.load_snapshot(name, time),
                    original.load_snapshot(name, time),
                    rtol=0,
                    atol=0,
                )
