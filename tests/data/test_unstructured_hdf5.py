"""Manufactured PLT, geometry reuse, chunked analysis and result export."""

import struct
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest
import torch

from flowtorch.data import (
    PLT2HDF5,
    HDF5Dataloader,
    HDF5Writer,
    TecplotDataloader,
    export_fields,
)


def make_plt(folder, count=8, volume=False, moving=False):
    folder.mkdir()

    def string(value):
        return struct.pack("<" + "i" * (len(value) + 1), *map(ord, value), 0)

    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    if volume:
        points = np.concatenate([points, points + [0, 0, 1]])
    for k in range(count):
        with (folder / f"flow_t={k*.125}.plt").open("wb") as f:
            f.write(b"#!TDV71 " + struct.pack("<i", 1) + string("fixture"))
            f.write(struct.pack("<i", 4))
            for name in ["X", "Y", "Z", "cp"]:
                f.write(string(name))
            for name in ["a", "b"]:
                f.write(
                    struct.pack("<f", 299)
                    + string(name)
                    + struct.pack("<5i", 2, -1, len(points), 1, 3 if volume else 1)
                )
            f.write(struct.pack("<f", 357))
            for i in range(2):
                f.write(struct.pack("<fi4i", 299, 0, 2, 2, 2, 2))
                coordinates = points + [i * 3 + 1.0e-9 * (k if moving else 1), 0, 0]
                cp = np.arange(len(points), dtype=float) + i + np.sin(k)
                for values in [*coordinates.T, cp]:
                    f.write(np.asarray(values, dtype="<f8").tobytes())
                f.write(struct.pack("<i", 0))
                f.write(np.arange(1, len(points) + 1, dtype="<i4").tobytes())
    return folder


@pytest.mark.parametrize("volume", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_conversion(tmp_path, volume, dtype, monkeypatch):
    folder = make_plt(tmp_path / "plt", volume=volume)
    out = tmp_path / "out"
    descriptor = PLT2HDF5(folder).convert(
        out, ["cp"], zones=["b", "a"], field_dtype=dtype, spatial_chunk=3
    )
    assert ET.parse(descriptor).getroot().tag == "Xdmf"
    with h5py.File(out / "geometry.h5", "r") as f:
        assert f["zones/z000000/points"].dtype == np.float64
        assert f["zones/z000000/topology"].dtype.kind == "i"
    stamp = (out / "geometry.h5").stat().st_mtime_ns
    with HDF5Dataloader(out / "fields.h5", dtype=dtype) as loader:
        assert loader.zone_names == ["b", "a"]
        n = 8 if volume else 4
        torch.testing.assert_close(loader.weights.sum(), torch.tensor(2.0, dtype=dtype))
        expected = torch.tensor(np.r_[np.arange(n) + 1, np.arange(n)], dtype=dtype)
        torch.testing.assert_close(loader.load_snapshot("cp", "0.0"), expected)
        monkeypatch.setattr(
            loader, "load_snapshot", lambda *a: pytest.fail("full snapshot read")
        )
        actual = loader.load_snapshot_slice("cp", ["0.0", "0.125"], slice(n - 1, n + 2))
        torch.testing.assert_close(actual[:, 0], expected[n - 1 : n + 2])
        assert actual.shape == (3, 2)
        geom_id = loader.geometry_id
    # Standalone result writer needs no PLT source.
    from flowtorch.analysis import FieldSpec, StateVectorLayout, StateVectorResult

    layout = StateVectorLayout([FieldSpec("cp")], (2 * n,), {"cp": 2.0})
    value = torch.complex(expected.double(), expected.double() + 1)
    result = StateVectorResult(
        layout, slice(0, 2 * n), (), lambda part: value[part] / 2, 3
    )
    export_fields(
        result,
        tmp_path / "modes.h5",
        out / "geometry.h5",
        geometry_id=geom_id,
        field_dtype=torch.float64,
    )
    with HDF5Dataloader(tmp_path / "modes.h5", dtype=torch.float64) as loader:
        torch.testing.assert_close(loader.load_snapshot("cp_real", "0"), value.real)
        torch.testing.assert_close(loader.load_snapshot("cp_imag", "0"), value.imag)
    assert (out / "geometry.h5").stat().st_mtime_ns == stamp
    with HDF5Dataloader(out / "fields.h5", zones=["a"]) as loader:
        assert loader.snapshot_shape("cp", "0.0") == (n,)
    with pytest.raises(FileExistsError):
        PLT2HDF5(folder).convert(out, ["cp"])


def test_invalid_and_incomplete(tmp_path):
    folder = make_plt(tmp_path / "plt", count=2)
    converter = PLT2HDF5(folder)
    for options in (
        {"zones": ["missing"]},
        {"zones": ["a", "a"]},
        {"fields": ["missing"]},
    ):
        with pytest.raises(ValueError):
            converter.convert(tmp_path / "bad", **options)
    converter.convert(tmp_path / "ok", ["cp"])
    geometry = tmp_path / "ok/geometry.h5"
    with HDF5Writer(str(tmp_path / "incomplete.h5"), geometry_file=geometry) as writer:
        writer.allocate_field("p", (), "0", "a")
        with pytest.raises(ValueError):
            writer.write_xdmf()
    with pytest.raises(ValueError, match="incomplete"):
        HDF5Dataloader(tmp_path / "incomplete.h5")
    with h5py.File(geometry, "r+") as f:
        f["zones/z000000/points"][0, 0] += 0.1
    with pytest.raises(ValueError, match="identity"):
        HDF5Dataloader(tmp_path / "ok/fields.h5")
    moving = make_plt(tmp_path / "moving", count=2, moving=True)
    with pytest.raises(ValueError, match="Moving"):
        PLT2HDF5(moving).convert(tmp_path / "changed", ["cp"])


def test_native_slices(tmp_path, monkeypatch):
    folder = make_plt(tmp_path / "plt", count=2)
    from flowtorch.data.plt_converter import discover_plt

    with TecplotDataloader(
        str(folder),
        discover_plt(folder),
        backend="native",
        geometry_dtype=torch.float64,
    ) as loader:
        assert loader.supports_direct_slicing
        monkeypatch.setattr(
            loader, "load_snapshot", lambda *a: pytest.fail("full read")
        )
        assert loader.snapshot_shape("cp", "0.0") == (4,)
        assert loader.load_snapshot_slice("cp", "0.0", slice(1, 3)).shape == (2,)
        assert loader.vertices.dtype == torch.float64
        assert loader.geometry().topology.dtype.kind == "i"


def test_legacy_hdf(tmp_path):
    path = tmp_path / "legacy.h5"
    with HDF5Writer(str(path)) as writer:
        writer.write("centers", (2, 3), torch.zeros(2, 3))
        writer.write("volumes", (2,), torch.ones(2))
        writer.write("connectivity", (2,), torch.tensor([0, 1]), dtype=torch.int64)
        writer.write("p", (2,), torch.tensor([3.0, 4.0]), "10")
        writer.write("p", (2,), torch.tensor([1.0, 2.0]), "2")
    with HDF5Dataloader(path) as loader:
        assert loader.write_times == ["2", "10"]
        assert loader.connectivity.dtype == torch.int64
        torch.testing.assert_close(
            loader.load_snapshot_slice("p", "2", slice(1, 2)), torch.tensor([2.0])
        )


def test_chunked_svd_spod(tmp_path, monkeypatch):
    from flowtorch.analysis import DataloaderStateVectorSource, FieldSpec, SVD, PAMSPOD
    from flowtorch.data import export_spod

    folder = make_plt(tmp_path / "plt", count=16)
    PLT2HDF5(folder).convert(tmp_path / "converted", ["cp"], spatial_chunk=3)
    with HDF5Dataloader(
        tmp_path / "converted/fields.h5", dtype=torch.float64
    ) as loader:
        direct = loader.load_snapshot("cp", loader.write_times)
        weight = loader.weights
        monkeypatch.setattr(
            loader, "load_snapshot", lambda *a: pytest.fail("full snapshot read")
        )
        source = DataloaderStateVectorSource(
            loader, [FieldSpec("cp")], normalization={"cp": 2.0}, spatial_weight=weight
        )
        svd = SVD(source, rank=2, spatial_batch_size=3, subtract_mean=False)
        reference = torch.linalg.svdvals(weight.sqrt()[:, None] * direct / 2)[:2]
        torch.testing.assert_close(svd.s, reference, rtol=1e-10, atol=1e-12)
        basis = svd.U.materialize_local()
        torch.testing.assert_close(
            basis.T @ (weight[:, None] * basis),
            torch.eye(2, dtype=torch.float64),
            rtol=1e-9,
            atol=1e-10,
        )
        first = DataloaderStateVectorSource(
            loader,
            [FieldSpec("cp")],
            times=loader.write_times[:8],
            normalization={"cp": 2.0},
            spatial_weight=weight,
        )
        second = DataloaderStateVectorSource(
            loader,
            [FieldSpec("cp")],
            times=loader.write_times[8:],
            normalization={"cp": 2.0},
            spatial_weight=weight,
        )
        batched = SVD(first, rank=2, spatial_batch_size=3)
        batched.update(second)
        torch.testing.assert_close(batched.s, reference, rtol=1e-10, atol=1e-12)
        spod = PAMSPOD.from_svd(
            svd, dt=0.125, nfft=8, adaptive=False, max_tapers=2, keep_n_modes=1
        )
        paths = export_spod(
            spod,
            tmp_path / "spod",
            tmp_path / "converted/geometry.h5",
            frequency_indices=[1],
            animation_phases=[0, np.pi / 2, np.pi],
            amplitude=2.0,
            geometry_id=loader.geometry_id,
            field_dtype=torch.float64,
            compression="gzip",
            shuffle=True,
        )
        for descriptor in paths:
            with h5py.File(descriptor.with_suffix(".h5")) as handle:
                dataset = next(
                    iter(next(iter(handle["frames/t000000"].values())).values())
                )
                assert dataset.compression == "gzip" and dataset.shuffle
        expected = spod.get_mode(1, 0).materialize_local(split=True)["cp"]
        with HDF5Dataloader(paths[0].with_suffix(".h5"), dtype=torch.float64) as mode:
            torch.testing.assert_close(
                mode.load_snapshot("cp_real", "0"), expected.real
            )
        with HDF5Dataloader(
            paths[1].with_suffix(".h5"), dtype=torch.float64
        ) as animation:
            torch.testing.assert_close(
                animation.load_snapshot("cp", "0"), 2 * expected.real
            )
            torch.testing.assert_close(
                animation.load_snapshot("cp", "1"),
                -2 * expected.imag,
                atol=1e-12,
                rtol=1e-10,
            )


def make_foam(folder):
    """One unit cube, all boundary faces, explicit cell centre and volume."""
    mesh = folder / "constant/polyMesh"
    mesh.mkdir(parents=True)

    def header(kind, name, extra=""):
        return (
            f"FoamFile\n{{\nformat ascii;\nclass {kind};\nobject {name};\n{extra}\n}}\n"
        )

    def listing(path, kind, items, extra=""):
        path.write_text(
            header(kind, path.name, extra)
            + f"\n{len(items)}\n(\n"
            + "\n".join(items)
            + "\n)\n"
        )

    listing(
        mesh / "points",
        "vectorField",
        [
            "(0 0 0)",
            "(1 0 0)",
            "(1 1 0)",
            "(0 1 0)",
            "(0 0 1)",
            "(1 0 1)",
            "(1 1 1)",
            "(0 1 1)",
        ],
    )
    listing(
        mesh / "faces",
        "faceList",
        [
            "4(0 3 2 1)",
            "4(4 5 6 7)",
            "4(0 1 5 4)",
            "4(1 2 6 5)",
            "4(2 3 7 6)",
            "4(3 0 4 7)",
        ],
    )
    listing(
        mesh / "owner",
        "labelList",
        ["0"] * 6,
        'note "nPoints:8 nCells:1 nFaces:6 nInternalFaces:0";',
    )
    listing(mesh / "neighbour", "labelList", [])
    listing(mesh / "boundary", "polyBoundaryMesh", [])
    for name, kind, values in [
        ("C", "vectorField", ["(0.5 0.5 0.5)"]),
        ("V", "scalarField", ["1"]),
    ]:
        listing(folder / "constant" / name, kind, values)
    for time in ["0", "1"]:
        (folder / time).mkdir()
        for name, kind, components, value in [
            ("p", "volScalarField", "scalar", "1.123456789012"),
            ("U", "volVectorField", "vector", "(1 2 3)"),
        ]:
            (folder / time / name).write_text(
                header(kind, name)
                + f"\ninternalField nonuniform List<{components}>\n1\n(\n{value}\n)\n;\nboundaryField {{}}\n"
            )
    return folder


@pytest.mark.parametrize("split", [False, True])
def test_foam_conversion(tmp_path, split):
    from flowtorch.data import FOAM2HDF5, XDMFWriter

    case = make_foam(tmp_path / "foam")
    converter = FOAM2HDF5(str(case), dtype=torch.float64)
    geometry = str(tmp_path / "geometry.h5") if split else None
    result = converter.convert(
        "converted.h5", ["p", "U"], ["1"], geometry_file=geometry, spatial_chunk=1
    )
    with HDF5Dataloader(result, dtype=torch.float64) as loader:
        torch.testing.assert_close(
            loader.load_snapshot("p", "1").reshape(-1),
            torch.tensor([1.123456789012], dtype=torch.float64),
        )
        torch.testing.assert_close(loader.weights, torch.ones(1, dtype=torch.float64))
        assert loader.connectivity.dtype == torch.int64
    with h5py.File(result) as file:
        ds = file["frames/t000000/z000000/f000000"] if split else file["variable/1/p"]
        assert ds.dtype == np.float64
        if not split:
            assert file["constant/vertices"].dtype == np.float64
    with XDMFWriter.from_filepath(result) as writer:
        writer.create_xdmf("regenerated.xdmf")
    assert ET.parse(case / "regenerated.xdmf").getroot().tag == "Xdmf"
    with pytest.raises(FileExistsError):
        converter.convert("converted.h5", ["p"], ["1"], geometry_file=geometry)
    with pytest.raises(ValueError):
        converter.convert("bad.h5", ["missing"], ["1"])


def _distributed_export_worker(rank, folder):
    import torch.distributed as dist
    from flowtorch.analysis import (
        DistributedExecution,
        FieldSpec,
        StateVectorLayout,
        StateVectorResult,
    )
    from pathlib import Path

    folder = Path(folder)
    dist.init_process_group(
        "gloo", init_method=f"file://{folder}/rendezvous", rank=rank, world_size=2
    )
    try:
        layout = StateVectorLayout(
            [FieldSpec("U", (3,)), FieldSpec("p")], (8,), {"U": 2.0, "p": 3.0}
        )

        def produce(part):
            p = torch.arange(part.start, part.stop, dtype=torch.float64)
            return layout.pack(
                {"U": torch.stack([p, p + 1, p + 2], dim=1), "p": p * 4},
                spatial_shape=(len(p),),
            )

        result = StateVectorResult(
            layout,
            slice(rank * 4, (rank + 1) * 4),
            (),
            produce,
            3,
            DistributedExecution(),
        )
        export_fields(
            result,
            folder / "distributed.h5",
            folder / "converted/geometry.h5",
            staging_directory=folder,
        )
    finally:
        dist.destroy_process_group()


def test_distributed_export(tmp_path):
    import torch.multiprocessing as mp

    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_gloo_available()
    ):
        pytest.skip("Gloo unavailable")
    # Skip only when local socket creation is forbidden by the execution host.
    import socket

    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
    except PermissionError:
        pytest.skip("Local sockets are restricted")
    folder = make_plt(tmp_path / "plt", count=1)
    PLT2HDF5(folder).convert(tmp_path / "converted", ["cp"])
    mp.spawn(_distributed_export_worker, args=(str(tmp_path),), nprocs=2, join=True)
    with HDF5Dataloader(tmp_path / "distributed.h5", dtype=torch.float64) as loader:
        p = torch.arange(8, dtype=torch.float64)
        torch.testing.assert_close(loader.load_snapshot("p", "0"), p * 4)
        torch.testing.assert_close(
            loader.load_snapshot("U", "0"), torch.stack([p, p + 1, p + 2], dim=1)
        )
    assert not list(tmp_path.glob("flowtorch-export-*"))


@pytest.mark.parametrize(
    "element,points,measure",
    [
        (0, [[0, 0, 0], [2, 0, 0], [0, 1, 0]], 1.0),
        (2, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], 1 / 6),
    ],
)
@pytest.mark.parametrize("endian", ["<", ">"])
def test_native_triangle_tetra(tmp_path, element, points, measure, endian):
    def string(value):
        return struct.pack(endian + "i" * (len(value) + 1), *map(ord, value), 0)

    path = tmp_path / "one.plt"
    with path.open("wb") as f:
        f.write(
            b"#!TDV71 "
            + struct.pack(endian + "i", 1)
            + string("fixture")
            + struct.pack(endian + "i", 4)
        )
        for name in ["X", "Y", "Z", "p"]:
            f.write(string(name))
        f.write(
            struct.pack(endian + "f", 299)
            + string("zone")
            + struct.pack(endian + "5i", 2, -1, len(points), 1, element)
        )
        f.write(struct.pack(endian + "ffi4i", 357, 299, 0, 2, 2, 2, 2))
        for values in [*np.asarray(points).T, np.arange(len(points))]:
            f.write(np.asarray(values, dtype=endian + "f8").tobytes())
        f.write(
            struct.pack(endian + "i", 0)
            + np.arange(1, len(points) + 1, dtype=endian + "i4").tobytes()
        )
    with TecplotDataloader(str(tmp_path), {"0": path.name}, backend="native") as native:
        torch.testing.assert_close(
            native.load_snapshot("p", "0"),
            torch.arange(len(points), dtype=torch.float32),
        )
    PLT2HDF5(path).convert(tmp_path / "out", ["p"])
    with HDF5Dataloader(tmp_path / "out/fields.xdmf", dtype=torch.float64) as loader:
        torch.testing.assert_close(
            loader.weights,
            torch.full((len(points),), measure / len(points), dtype=torch.float64),
        )


def test_finalized_writer_and_singleton_legacy(tmp_path):
    folder = make_plt(tmp_path / "plt", count=1)
    PLT2HDF5(folder).convert(tmp_path / "converted", ["cp"])
    with HDF5Writer(
        str(tmp_path / "final.h5"), geometry_file=tmp_path / "converted/geometry.h5"
    ) as writer:
        for zone in ["a", "b"]:
            writer.allocate_field("p", (), "0", zone)
            writer.write_slice("p", "0", zone, slice(0, 4), torch.ones(4))
        writer.finish()
        with pytest.raises(ValueError, match="finalized"):
            writer.allocate_field("p", (), "1", "a")
    with h5py.File(tmp_path / "singleton.h5", "w") as file:
        file.create_dataset("variable/0/p", data=np.array([[7.0]]))
    with HDF5Dataloader(tmp_path / "singleton.h5") as loader:
        assert loader.snapshot_shape("p", "0") == (1,)
        assert loader.load_snapshot("p", "0").shape == (1,)
