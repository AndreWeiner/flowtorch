"""Appending snapshots preserves committed data, filters and shared geometry."""

import json
import sys
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest
import torch

from flowtorch.data import PLT2HDF5, HDF5Dataloader, HDF5Writer, FOAM2HDF5, XDMFWriter
from tests.data.test_unstructured_hdf5 import make_plt, make_foam


def times(descriptor):
    return [
        float(node.attrib["Value"])
        for node in ET.parse(descriptor).findall("./Domain/Grid/Grid/Time")
    ]


def write_frame(writer, time, count=4):
    for zone in ["a", "b"]:
        writer.allocate_field("cp", (), time, zone)
        writer.write_slice(
            "cp", time, zone, slice(0, count), torch.arange(count, dtype=torch.float32)
        )


@pytest.mark.parametrize("codec", [None, "gzip", "zstd"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_append_plt_inherits_filters_and_orders_times(tmp_path, codec, dtype):
    if codec == "zstd":
        pytest.importorskip("hdf5plugin")
    source = make_plt(tmp_path / "plt", count=3)
    converter = PLT2HDF5(source)
    labels = list(converter.files)
    output = tmp_path / "out"
    descriptor = converter.convert(
        output,
        fields=["cp"],
        times=[labels[0], labels[2]],
        zones=["b", "a"],
        field_dtype=dtype,
        compression=codec,
        shuffle=True,
        geometry_compression=codec,
        geometry_shuffle=True,
        spatial_chunk=3,
    )
    geometry_before = (output / "geometry.h5").read_bytes()
    with h5py.File(output / "fields.h5") as f:
        old_chunk = f["frames/t000000/z000000/f000000"].id.read_direct_chunk((0,))
        old_address = h5py.h5o.get_info(f["frames/t000000"].id).addr
    # Omitted zone/field selections and compression settings inherit the old data.
    converter.convert(
        output, times=[labels[1]], field_dtype=dtype, append=True, spatial_chunk=3
    )
    assert times(descriptor) == [0.0, 0.125, 0.25]
    with HDF5Dataloader(output / "fields.h5", dtype=dtype) as reader:
        assert reader.zone_names == ["b", "a"]
        assert list(map(float, reader.write_times)) == [0.0, 0.125, 0.25]
        expected = torch.tensor(
            np.r_[np.arange(4) + 1, np.arange(4)] + np.sin(1), dtype=dtype
        )
        assert torch.equal(reader.load_snapshot("cp", labels[1]), expected)
        assert torch.equal(
            reader.load_snapshot_slice("cp", labels[1], slice(3, 6)), expected[3:6]
        )
    with h5py.File(output / "fields.h5") as f:
        assert (
            f["frames/t000000/z000000/f000000"].id.read_direct_chunk((0,)) == old_chunk
        )
        assert h5py.h5o.get_info(f["frames/t000000"].id).addr == old_address
        assert not any(k.startswith("_pending_append_") for k in f)
        assert len(json.loads(f.attrs["committed_frames"])) == 3
        ds = f["frames/t000002/z000000/f000000"]
        assert ds.shuffle
        ids = [
            ds.id.get_create_plist().get_filter(i)[0]
            for i in range(ds.id.get_create_plist().get_nfilters())
        ]
        assert (32015 in ids) == (codec == "zstd")
        assert (1 in ids) == (codec == "gzip")
    assert (output / "geometry.h5").read_bytes() == geometry_before
    with XDMFWriter.from_filepath(str(output / "fields.h5")) as writer:
        writer.create_xdmf("regenerated.xdmf")
    assert times(output / "regenerated.xdmf") == [0.0, 0.125, 0.25]


@pytest.mark.parametrize(
    "options",
    [
        {"times": ["0.0"]},
        {"fields": ["X"]},
        {"zones": ["b", "a"]},
        {"field_dtype": torch.float64},
        {"compression_level": 1},
        {"shuffle": False},
        {"overwrite": True},
    ],
)
def test_append_preflight_preserves_existing_files(tmp_path, options):
    converter = PLT2HDF5(make_plt(tmp_path / "plt", count=2))
    output = tmp_path / "out"
    converter.convert(
        output, fields=["cp"], times=["0.0"], compression="gzip", shuffle=True
    )
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    arguments = dict(times=["0.125"], append=True)
    arguments.update(options)
    with pytest.raises(ValueError):
        converter.convert(output, **arguments)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


def test_direct_writer_abort_and_retry(tmp_path):
    converter = PLT2HDF5(make_plt(tmp_path / "plt", count=2))
    output = tmp_path / "out"
    converter.convert(output, fields=["cp"], times=["0.0"])
    descriptor = (output / "fields.xdmf").read_bytes()
    with pytest.raises(RuntimeError, match="interrupted"):
        with HDF5Writer(output / "fields.h5", append=True) as writer:
            writer.allocate_field("cp", (), "0.125", "a")
            writer.write_slice("cp", "0.125", "a", slice(0, 2), torch.ones(2))
            raise RuntimeError("interrupted")
    with HDF5Dataloader(output) as reader:
        assert reader.write_times == ["0.0"]
    assert (output / "fields.xdmf").read_bytes() == descriptor
    with HDF5Writer(output / "fields.h5", append=True) as writer:
        write_frame(writer, "0.125")
        writer.finish()  # Idempotent, including the later write_xdmf call.
        writer.finish()
        writer.write_xdmf(output / "fields.xdmf")
    assert times(output / "fields.xdmf") == [0.0, 0.125]


def test_mid_commit_failure_is_hidden_and_next_append_recovers(tmp_path, monkeypatch):
    output = tmp_path / "out"
    converter = PLT2HDF5(make_plt(tmp_path / "plt", count=2))
    converter.convert(output, fields=["cp"], times=["0.0"])
    with pytest.raises(RuntimeError, match="commit interrupted"):
        with HDF5Writer(output / "fields.h5", append=True) as writer:
            write_frame(writer, "0.125")
            original = writer._file.move

            def interrupted(*args, **kwargs):
                original(*args, **kwargs)
                raise RuntimeError("commit interrupted")

            monkeypatch.setattr(writer._file, "move", interrupted)
            writer.finish()
    with HDF5Dataloader(output) as reader:
        assert reader.write_times == ["0.0"]
    with XDMFWriter.from_filepath(str(output / "fields.h5")) as writer:
        writer.create_xdmf("recovered.xdmf")
    assert times(output / "recovered.xdmf") == [0.0]
    converter.convert(output, times=["0.125"], append=True)
    with h5py.File(output / "fields.h5") as f:
        assert len(f["frames"]) == 2
        assert not any(k.startswith("_pending_append_") for k in f)
    assert times(output / "fields.xdmf") == [0.0, 0.125]


def test_incomplete_staging_is_ignored_and_removed(tmp_path):
    output = tmp_path / "out"
    converter = PLT2HDF5(make_plt(tmp_path / "plt", count=2))
    converter.convert(output, fields=["cp"], times=["0.0"])
    with h5py.File(output / "fields.h5", "r+") as f:
        pending = f.create_group("_pending_append_abandoned")
        pending.create_group("invalid-frame")
    with HDF5Dataloader(output) as reader:
        assert reader.write_times == ["0.0"]
    converter.convert(output, times=["0.125"], append=True)
    with h5py.File(output / "fields.h5") as f:
        assert "_pending_append_abandoned" not in f


def test_descriptor_failure_preserves_old_descriptor(tmp_path, monkeypatch):
    output = tmp_path / "out"
    converter = PLT2HDF5(make_plt(tmp_path / "plt", count=2))
    converter.convert(output, fields=["cp"], times=["0.0"])
    old = (output / "fields.xdmf").read_bytes()
    from pathlib import Path

    original = Path.replace

    def fail(path, target):
        if str(target).endswith("fields.xdmf"):
            raise OSError("descriptor unavailable")
        return original(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail)
        with pytest.raises(OSError, match="descriptor unavailable"):
            converter.convert(output, times=["0.125"], append=True)
    assert (output / "fields.xdmf").read_bytes() == old
    with HDF5Dataloader(output) as reader:
        assert len(reader.write_times) == 2
    with XDMFWriter.from_filepath(str(output / "fields.h5")) as writer:
        writer.create_xdmf("fields.xdmf")
    assert times(output / "fields.xdmf") == [0.0, 0.125]


def test_foam_append_and_legacy_rejection(tmp_path):
    case = make_foam(tmp_path / "foam")
    converter = FOAM2HDF5(str(case), dtype=torch.float64)
    converter.convert(
        "fields.h5",
        ["p", "U"],
        ["0"],
        geometry_file=str(tmp_path / "geometry.h5"),
        compression="gzip",
        shuffle=True,
    )
    converter.convert("fields.h5", times=["1"], append=True)
    with HDF5Dataloader(case / "fields.h5", dtype=torch.float64) as reader:
        assert reader.write_times == ["0", "1"]
        assert torch.equal(
            reader.load_snapshot("p", "1"), reader.load_snapshot("p", "0")
        )
    assert times(case / "fields.xdmf") == [0.0, 1.0]
    converter.convert("legacy.h5", ["p"], ["0"])
    with pytest.raises(ValueError):
        converter.convert("legacy.h5", times=["1"], append=True)


def test_cli_append_and_conflicting_modes(tmp_path, monkeypatch):
    from flowtorch.data.plt_converter import main

    source = make_plt(tmp_path / "plt", count=2)
    output = tmp_path / "out"
    base = ["flowtorch-plt-to-hdf5", str(source), str(output)]
    monkeypatch.setattr(
        sys, "argv", base + ["--times", "0.0", "--compression", "gzip", "--shuffle"]
    )
    main()
    monkeypatch.setattr(sys, "argv", base + ["--times", "0.125", "--append"])
    main()
    assert times(output / "fields.xdmf") == [0.0, 0.125]
    monkeypatch.setattr(sys, "argv", base + ["--append", "--overwrite"])
    with pytest.raises(SystemExit):
        main()


def test_append_requires_existing_complete_split_file(tmp_path):
    converter = PLT2HDF5(make_plt(tmp_path / "plt", count=1))
    with pytest.raises(FileNotFoundError):
        converter.convert(tmp_path / "missing", append=True)
    assert not (tmp_path / "missing").exists()
    output = tmp_path / "out"
    converter.convert(output)
    with h5py.File(output / "fields.h5", "r+") as f:
        f.attrs["complete"] = False
    with pytest.raises(ValueError, match="incomplete"):
        converter.convert(output, append=True)


def test_export_fields_append(tmp_path):
    from flowtorch.analysis import FieldSpec, StateVectorLayout, StateVectorResult
    from flowtorch.data import export_fields
    from tests.data.test_geometry_streaming import geometry

    geom = tmp_path / "geometry.h5"
    HDF5Writer.write_geometry(geom, [("a", geometry())])
    layout = StateVectorLayout([FieldSpec("p")], (3,))
    values = torch.tensor([1 + 2j, 3 - 4j, 5 + 6j], dtype=torch.complex128)
    result = StateVectorResult(layout, slice(0, 3), (), lambda part: values[part], 2)
    output = tmp_path / "results.h5"
    export_fields(
        result,
        output,
        geom,
        labels=["0"],
        field_dtype=torch.float64,
        compression="gzip",
        shuffle=True,
        metadata={"batch": 0},
    )
    export_fields(
        result,
        output,
        geom,
        labels=["1"],
        field_dtype=torch.float64,
        append=True,
        metadata={"batch": 1},
    )
    with HDF5Dataloader(output, dtype=torch.float64) as reader:
        assert reader.write_times == ["0", "1"]
        assert torch.equal(reader.load_snapshot("p_real", "1"), values.real)
        assert torch.equal(reader.load_snapshot("p_imag", "0"), values.imag)
    with h5py.File(output) as f:
        assert json.loads(f.attrs["result_metadata"]) == {"batch": 0}
        assert json.loads(f["frames/t000001"].attrs["result_metadata"]) == {"batch": 1}
    with pytest.raises(ValueError, match="explicit"):
        export_fields(result, output, geom, field_dtype=torch.float64, append=True)


def test_append_rejects_changed_geometry(tmp_path):
    converter = PLT2HDF5(make_plt(tmp_path / "plt", count=2, moving=True))
    output = tmp_path / "out"
    converter.convert(output, fields=["cp"], times=["0.0"])
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ValueError, match="differs"):
        converter.convert(output, times=["0.125"], append=True)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


def test_append_requires_every_old_field_and_zone(tmp_path):
    output = tmp_path / "out"
    PLT2HDF5(make_plt(tmp_path / "plt", count=2)).convert(
        output, fields=["cp"], times=["0.0"]
    )
    with pytest.raises(ValueError, match="Every snapshot"):
        with HDF5Writer(output / "fields.h5", append=True) as writer:
            writer.allocate_field("cp", (), "0.125", "a")
            writer.write_slice("cp", "0.125", "a", slice(0, 4), torch.ones(4))
            writer.finish()
    with HDF5Dataloader(output) as reader:
        assert reader.write_times == ["0.0"]


def test_append_with_relocated_geometry(tmp_path):
    output = tmp_path / "out"
    converter = PLT2HDF5(make_plt(tmp_path / "plt", count=2))
    converter.convert(output, fields=["cp"], times=["0.0"])
    relocated = tmp_path / "relocated.h5"
    (output / "geometry.h5").rename(relocated)
    before = relocated.read_bytes()
    converter.convert(output, times=["0.125"], geometry_file=relocated, append=True)
    with HDF5Dataloader(output) as reader:
        assert reader.write_times == ["0.0", "0.125"]
    assert relocated.read_bytes() == before
