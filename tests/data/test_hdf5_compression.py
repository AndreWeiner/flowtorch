"""Lossless filters through public writers, converters and result exports."""

import json
import sys

import h5py
import numpy as np
import pytest
import torch

from flowtorch.data import (
    HDF5Dataloader,
    HDF5Writer,
    PLT2HDF5,
    FOAM2HDF5,
    export_fields,
)
from flowtorch.data import _hdf5_compression as filters
from tests.data.test_unstructured_hdf5 import make_plt, make_foam
from tests.data.test_geometry_streaming import geometry

CONFIGS = [
    (None, False),
    (None, True),
    ("gzip", False),
    ("gzip", True),
    ("zstd", False),
    ("zstd", True),
]


def require_codec(codec):
    if codec == "zstd":
        pytest.importorskip("hdf5plugin")


def assert_filter(ds, codec, shuffle):
    if codec == "zstd":
        pipeline = ds.id.get_create_plist()
        assert any(
            pipeline.get_filter(i)[0] == 32015 for i in range(pipeline.get_nfilters())
        )
    else:
        assert ds.compression == codec
    assert ds.shuffle == shuffle
    assert ds.chunks[0] <= 3
    if codec == "gzip":
        assert ds.compression_opts == 1
    if codec == "zstd":
        assert next(
            pipeline.get_filter(i)[2]
            for i in range(pipeline.get_nfilters())
            if pipeline.get_filter(i)[0] == 32015
        ) == (3,)


@pytest.mark.parametrize("codec,shuffle", CONFIGS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_plt_compression_and_reuse(tmp_path, codec, shuffle, dtype):
    require_codec(codec)
    level = 1 if codec == "gzip" else None
    source = make_plt(tmp_path / "plt", count=2)
    output = tmp_path / "out"
    options = dict(
        compression=codec,
        compression_level=level,
        shuffle=shuffle,
        geometry_compression=codec,
        geometry_compression_level=level,
        geometry_shuffle=shuffle,
        spatial_chunk=3,
        field_dtype=dtype,
    )
    descriptor = PLT2HDF5(source).convert(output, fields=["cp"], **options)
    with h5py.File(output / "fields.h5") as f:
        ds = f["frames/t000000/z000000/f000000"]
        assert_filter(ds, codec, shuffle)
        assert ds.dtype == (np.float32 if dtype == torch.float32 else np.float64)
        meta = json.loads(f.attrs["compression_settings"])["fields"]
        assert meta["codec"] == codec and meta["shuffle"] == shuffle
    with h5py.File(output / "geometry.h5") as f:
        for ds in f["zones/z000000"].values():
            if codec is not None or shuffle:
                assert_filter(ds, codec, shuffle)
            else:
                assert ds.chunks is None  # Preserve legacy contiguous geometry default.
            assert ds.dtype == (
                np.int64 if ds.name.endswith("topology") else np.float64
            )
    with HDF5Dataloader(descriptor, dtype=dtype) as reader:
        expected = torch.tensor([0, 1, 2, 3, 1, 2, 3, 4], dtype=dtype)
        assert torch.equal(reader.load_snapshot("cp", "0.0"), expected)
        assert torch.equal(
            reader.load_snapshot_slice("cp", "0.0", slice(2, 6)), expected[2:6]
        )
        np.testing.assert_array_equal(reader.geometry("a").topology, [[0, 1, 2, 3]])
        assert reader.geometry("a").points.dtype == np.float64
    before = (output / "geometry.h5").read_bytes()
    # New field file can use a different codec without modifying reused geometry.
    PLT2HDF5(source).convert(
        tmp_path / "reuse",
        fields=["cp"],
        geometry_file=output / "geometry.h5",
        compression="gzip",
        shuffle=True,
    )
    assert (output / "geometry.h5").read_bytes() == before


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("codec", ["gzip", "zstd"])
def test_foam_compression(tmp_path, split, codec):
    require_codec(codec)
    case = make_foam(tmp_path / "foam")
    path = FOAM2HDF5(str(case), dtype=torch.float64).convert(
        "compressed.h5",
        ["p", "U"],
        ["1"],
        geometry_file=str(tmp_path / "geometry.h5") if split else None,
        compression=codec,
        shuffle=True,
        geometry_compression=codec,
        geometry_shuffle=True,
        spatial_chunk=3,
    )
    with HDF5Dataloader(path, dtype=torch.float64) as reader:
        assert torch.equal(
            reader.load_snapshot("p", "1").reshape(-1),
            torch.tensor([1.123456789012], dtype=torch.float64),
        )
        assert torch.equal(reader.weights, torch.ones(1, dtype=torch.float64))
        assert reader.edge_vertices.dtype == torch.float64
    with h5py.File(path) as f:
        ds = f["frames/t000000/z000000/f000000"] if split else f["variable/1/p"]
        pipeline = ds.id.get_create_plist()
        assert any(
            pipeline.get_filter(i)[0] == (32015 if codec == "zstd" else 1)
            for i in range(pipeline.get_nfilters())
        )
        assert ds.shuffle
        if not split:
            assert f["constant/vertices"].compression == ds.compression


@pytest.mark.parametrize("codec", ["gzip", "zstd"])
def test_result_export(tmp_path, codec):
    require_codec(codec)
    from flowtorch.analysis import FieldSpec, StateVectorLayout, StateVectorResult

    geom = tmp_path / "geometry.h5"
    HDF5Writer.write_geometry(geom, [("a", geometry())])
    before = geom.read_bytes()
    values = torch.tensor([1 + 2j, 3 - 4j, 5 + 6j], dtype=torch.complex128)
    layout = StateVectorLayout([FieldSpec("p")], (3,))
    result = StateVectorResult(layout, slice(0, 3), (), lambda part: values[part], 2)
    output = tmp_path / "mode.h5"
    export_fields(
        result, output, geom, compression=codec, shuffle=True, field_dtype=torch.float64
    )
    with HDF5Dataloader(output, dtype=torch.float64) as reader:
        assert torch.equal(reader.load_snapshot("p_real", "0"), values.real)
        assert torch.equal(reader.load_snapshot("p_imag", "0"), values.imag)
    assert geom.read_bytes() == before


@pytest.mark.parametrize(
    "options",
    [
        {"compression": "lzf"},
        {"compression": "gzip", "compression_level": 10},
        {"compression": "zstd", "compression_level": 23},
        {"compression_level": 1},
        {"compression": "gzip", "compression_level": True},
        {"compression": "gzip", "compression_level": 1.5},
        {"shuffle": "yes"},
    ],
)
def test_invalid_options_before_output(tmp_path, options):
    output = tmp_path / "bad.h5"
    with pytest.raises(ValueError):
        HDF5Writer(output, **options)
    assert not output.exists()
    with pytest.raises(ValueError):
        PLT2HDF5(make_plt(tmp_path / "plt", count=1)).convert(
            tmp_path / "out", **options
        )
    assert not (tmp_path / "out").exists()


def test_missing_plugin_before_output(tmp_path, monkeypatch):
    original = filters.h5py.h5z.filter_avail
    monkeypatch.setattr(
        filters.h5py.h5z,
        "filter_avail",
        lambda fid: False if fid == 32015 else original(fid),
    )

    def missing(name):
        raise ImportError("not installed")

    monkeypatch.setattr(filters.importlib, "import_module", missing)
    with pytest.raises(ImportError, match="compression"):
        HDF5Writer(tmp_path / "bad.h5", compression="zstd")
    assert not (tmp_path / "bad.h5").exists()


def test_zstd_negative_level(tmp_path):
    require_codec("zstd")
    with HDF5Writer(
        tmp_path / "fast.h5", compression="zstd", compression_level=-5
    ) as writer:
        writer.write(
            "p",
            (100,),
            torch.arange(100, dtype=torch.float64),
            "0",
            dtype=torch.float64,
        )
    with HDF5Dataloader(tmp_path / "fast.h5", dtype=torch.float64) as reader:
        assert torch.equal(
            reader.load_snapshot("p", "0"), torch.arange(100, dtype=torch.float64)
        )


def test_cli_compression(tmp_path, monkeypatch):
    from flowtorch.data.plt_converter import main

    source = make_plt(tmp_path / "plt", count=1)
    output = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flowtorch-plt-to-hdf5",
            str(source),
            str(output),
            "--compression",
            "gzip",
            "--compression-level",
            "1",
            "--shuffle",
            "--geometry-compression",
            "gzip",
            "--geometry-shuffle",
        ],
    )
    main()
    with h5py.File(output / "fields.h5") as f:
        assert f["frames/t000000/z000000/f000000"].shuffle
    with h5py.File(output / "geometry.h5") as f:
        assert f["zones/z000000/points"].shuffle


def test_invalid_compressed_shape_preserves_existing_dataset(tmp_path):
    output = tmp_path / "existing.h5"
    with HDF5Writer(output, compression="gzip") as writer:
        writer.write("p", (3,), torch.arange(3.0), "0")
        with pytest.raises(ValueError, match="positive"):
            writer.write("p", (0,), time="0")
        np.testing.assert_array_equal(writer._file["variable/0/p"][:], np.arange(3.0))
