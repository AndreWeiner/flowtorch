"""Manufactured TDV112 records, independent binary fixtures and loader contracts."""

import struct

import numpy as np
import pytest
import torch

from flowtorch.data import TecplotDataloader, PLT2HDF5, HDF5Dataloader
from flowtorch.data._tecplot112 import index_by_zone

POINTS = {
    1: [[0, 0, 0], [2, 0, 0]],
    2: [[0, 0, 0], [2, 0, 0], [0, 1, 0]],
    3: [[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]],
    4: [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
    5: [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
}


def write112(path, kind=4, endian="<", shared=False, neighbour=False, pressure_code=1):
    """Two zones; nodal/cell data, passive field and auxiliary records."""
    points = np.asarray(POINTS[kind], dtype=float) + [1.0e-9, 0, 0]
    names = ["X", "Y", "Z", "p", "cell", "passive"]
    pressure_dtype = {1: "f4", 2: "f8", 3: "i4", 4: "i2", 5: "u1"}[pressure_code]

    def pack(code, *values):
        return struct.pack(endian + code, *values)

    def string(text):
        return pack("i" * (len(text) + 1), *map(ord, text), 0)

    with path.open("wb") as f:
        f.write(
            b"#!TDV112" + pack("ii", 1, 0) + string("fixture") + pack("i", len(names))
        )
        for name in names:
            f.write(string(name))
        f.write(pack("f", 799) + string("test.aux") + pack("i", 0) + string("value"))
        f.write(
            pack("fi", 899, 3)
            + string("test.variable")
            + pack("i", 0)
            + string("pressure")
        )
        for name in ["a", "b"]:
            f.write(
                pack("f", 299)
                + string(name)
                + pack("ii", -1, -1)
                + pack("d", 0.25)
                + pack("ii", -1, kind)
            )
            f.write(pack("i", 1) + pack("6i", 0, 0, 0, 0, 1, 0))
            f.write(pack("2i", 0, int(neighbour)))
            f.write(pack("5i", len(points), 1, 0, 0, 0))
            f.write(
                pack("i", 1)
                + string("zone.aux")
                + pack("i", 0)
                + string("data")
                + pack("i", 0)
            )
        f.write(pack("f", 357))
        for zone in range(2):
            refs = [0] * 4 + [-1, -1] if shared and zone == 1 else [-1] * 6
            values = [
                *points.T,
                (np.arange(len(points), dtype=float) + 0.25).astype(pressure_dtype),
                np.array([7.0 + zone]),
                None,
            ]
            f.write(pack("f", 299) + pack("6i", 2, 2, 2, pressure_code, 2, 1))
            f.write(pack("i", 1) + pack("6i", 0, 0, 0, 0, 0, 1))
            f.write(
                pack("i", 1)
                + pack("6i", *refs)
                + pack("i", 0 if shared and zone == 1 else -1)
            )
            for i, array in enumerate(values):
                if array is not None and refs[i] == -1:
                    f.write(pack("dd", array.min(), array.max()))
            for i, array in enumerate(values):
                if array is not None and refs[i] == -1:
                    f.write(
                        np.asarray(
                            array, dtype=endian + (pressure_dtype if i == 3 else "f8")
                        ).tobytes()
                    )
            if not (shared and zone == 1):
                f.write(np.arange(len(points), dtype=endian + "i4").tobytes())
    return points


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("kind", [1, 2, 3, 4, 5])
def test_native_fields_geometry(tmp_path, kind, endian, monkeypatch):
    path = tmp_path / "one.plt"
    points = write112(path, kind, endian, shared=True)
    with TecplotDataloader(
        str(tmp_path), {"0": path.name}, backend="auto", dtype=torch.float64
    ) as loader:
        assert loader.backend == "native" and loader.supports_direct_slicing
        assert loader.zone_names == ["a", "b"]
        loader.zone = "b"
        assert "cell" not in loader.field_names["0"]
        assert loader.snapshot_shape("p", "0") == (len(points),)
        expected = torch.arange(len(points), dtype=torch.float64) + 0.25
        torch.testing.assert_close(loader.load_snapshot("p", "0"), expected)
        monkeypatch.setattr(
            loader, "load_snapshot", lambda *a: pytest.fail("full field")
        )
        torch.testing.assert_close(
            loader.load_snapshot_slice("p", "0", slice(1, 3)), expected[1:3]
        )
        torch.testing.assert_close(
            loader.load_snapshot_slice("passive", "0", slice(0, 2)),
            torch.zeros(2, dtype=torch.float64),
        )
        torch.testing.assert_close(
            loader.load_connectivity(), torch.arange(len(points)).reshape(1, -1)
        )
        geometry = loader.geometry()
        np.testing.assert_array_equal(geometry.points, points)
        assert geometry.weights.sum() == pytest.approx(
            {1: 2.0, 2: 1.0, 3: 2.0, 4: 1 / 6, 5: 1.0}[kind]
        )
    with TecplotDataloader(
        str(tmp_path),
        {"0": path.name},
        backend="native",
        association="cell",
        dtype=torch.float64,
    ) as loader:
        loader.zone = "b"
        assert loader.field_names["0"] == ["cell"]
        assert loader.snapshot_shape("cell", "0") == (1,)
        torch.testing.assert_close(
            loader.load_snapshot_slice("cell", "0", slice(None)),
            torch.tensor([8.0], dtype=torch.float64),
        )
        assert loader.geometry().locations.shape == (1, 3)


@pytest.mark.parametrize(
    "association,fields", [("point", ["p", "passive"]), ("cell", ["cell"])]
)
def test_conversion112(tmp_path, association, fields):
    source = tmp_path / "source"
    source.mkdir()
    write112(source / "one.plt", shared=True)
    descriptor = PLT2HDF5(source, association=association).convert(
        tmp_path / "out", fields=fields, spatial_chunk=2
    )
    with HDF5Dataloader(descriptor, dtype=torch.float64) as loader:
        assert loader.association == association
        assert loader.zone_names == ["a", "b"]
        expected = (
            np.r_[np.arange(4) + 0.25, np.arange(4) + 0.25]
            if association == "point"
            else [7.0, 8.0]
        )
        torch.testing.assert_close(
            loader.load_snapshot(fields[0], "0"),
            torch.tensor(expected, dtype=torch.float64),
        )


def test_reject_unsupported_and_truncated(tmp_path):
    path = tmp_path / "one.plt"
    write112(path, neighbour=True)
    with pytest.raises(ValueError, match="neighbour"):
        index_by_zone(path)
    write112(path)
    original = path.read_bytes()
    path.write_bytes(original[:-1])
    with pytest.raises(ValueError, match="Truncated"):
        index_by_zone(path)
    path.write_bytes(original + b"extra")
    with pytest.raises(ValueError, match="trailing"):
        index_by_zone(path)
    path.write_bytes(original[:8] + b"xxxx" + original[12:])
    with pytest.raises(ValueError, match="byte-order"):
        index_by_zone(path)


@pytest.mark.parametrize(
    "change,message",
    [
        ("zone", "zone type"),
        ("type", "variable type"),
        ("sharing", "earlier zone"),
    ],
)
def test_invalid_layouts(tmp_path, change, message):
    path = tmp_path / "one.plt"
    write112(path)
    data = bytearray(path.read_bytes())
    if change == "zone":
        marker = struct.pack("<f", 299) + struct.pack("<2i", ord("a"), 0)
        offset = data.index(marker) + 32
        data[offset : offset + 4] = struct.pack(
            "<i", 0
        )  # Ordered is outside this scope.
    else:
        start = data.index(struct.pack("<f", 357)) + 4
        offset = start + 4 if change == "type" else start + 60
        data[offset : offset + 4] = struct.pack("<i", 6 if change == "type" else 0)
    path.write_bytes(data)
    with pytest.raises(ValueError, match=message):
        index_by_zone(path)


def test_invalid_connectivity_is_validated_on_read(tmp_path):
    path = tmp_path / "one.plt"
    write112(path)
    data = path.read_bytes()
    path.write_bytes(data[:-4] + struct.pack("<i", 1000))
    with TecplotDataloader(str(tmp_path), {"0": path.name}, backend="native") as loader:
        loader.zone = "b"
        with pytest.raises(ValueError, match="connectivity"):
            loader.load_connectivity()
        with pytest.raises(ValueError, match="connectivity"):
            loader.geometry()


@pytest.mark.parametrize("code", [1, 2, 3, 4, 5])
def test_source_numeric_types(tmp_path, code):
    path = tmp_path / "one.plt"
    write112(path, pressure_code=code)
    with TecplotDataloader(
        str(tmp_path), {"0": path.name}, backend="native", dtype=torch.float64
    ) as loader:
        expected = torch.arange(4, dtype=torch.float64) + (0.25 if code <= 2 else 0.0)
        torch.testing.assert_close(
            loader.load_snapshot_slice("p", "0", slice(None)), expected
        )
