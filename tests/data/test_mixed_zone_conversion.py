"""Conversion preserves separate surface/volume meshes and their raw measures."""

import struct
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest
import torch

from flowtorch.data import PLT2HDF5, HDF5Dataloader


def mixed_plt(path, version):
    def pack(code, *values):
        return struct.pack("<" + code, *values)

    def string(text):
        return pack("i" * (len(text) + 1), *map(ord, text), 0)

    points = [
        np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]], dtype=float),
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
    ]
    with path.open("wb") as f:
        f.write(b"#!TDV71 " if version == 71 else b"#!TDV112")
        f.write(pack("i", 1))
        if version == 112:
            f.write(pack("i", 0))
        f.write(string("mixed") + pack("i", 4))
        for name in ["X", "Y", "Z", "p"]:
            f.write(string(name))
        for i, name in enumerate(["surface", "volume"]):
            f.write(pack("f", 299) + string(name))
            if version == 71:
                f.write(pack("5i", 2, -1, 4, 1, 1 if i == 0 else 2))
            else:
                f.write(
                    pack("ii", -1, -1)
                    + pack("d", 0)
                    + pack("ii", -1, 3 if i == 0 else 4)
                )
                f.write(pack("8i", 0, 0, 0, 4, 1, 0, 0, 0) + pack("i", 0))
        f.write(pack("f", 357))
        for i, coordinates in enumerate(points):
            f.write(pack("f", 299))
            if version == 71:
                f.write(pack("i", 0))
            f.write(pack("4i", 2, 2, 2, 2))
            if version == 112:
                f.write(pack("3i", 0, 0, -1))
                for a in [*coordinates.T, np.arange(4) + i * 10]:
                    f.write(pack("dd", a.min(), a.max()))
            for a in [*coordinates.T, np.arange(4) + i * 10]:
                f.write(np.asarray(a, dtype="<f8").tobytes())
            if version == 71:
                f.write(pack("i", 0))
            f.write(
                np.arange(
                    1 if version == 71 else 0, 5 if version == 71 else 4, dtype="<i4"
                ).tobytes()
            )


@pytest.mark.parametrize("version", [71, 112])
def test_mixed_conversion(tmp_path, version):
    source = tmp_path / "mixed.plt"
    mixed_plt(source, version)
    descriptor = PLT2HDF5(source).convert(
        tmp_path / "converted", fields=["p"], spatial_chunk=3
    )
    with HDF5Dataloader(descriptor, dtype=torch.float64) as loader:
        assert loader.zone_names == ["surface", "volume"]
        assert loader.zone_dimensions == {"surface": 2, "volume": 3}
        torch.testing.assert_close(
            loader.load_snapshot("p", "0"),
            torch.tensor(
                [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0], dtype=torch.float64
            ),
        )
        torch.testing.assert_close(
            loader.load_snapshot_slice("p", "0", slice(3, 6)),
            torch.tensor([3.0, 10.0, 11.0], dtype=torch.float64),
        )
        assert loader.weights_by_zone["surface"].sum() == pytest.approx(2.0)
        assert loader.weights_by_zone["volume"].sum() == pytest.approx(1 / 6)
        with pytest.raises(ValueError, match="mix geometric dimensions"):
            _ = loader.weights
    with HDF5Dataloader(descriptor, zones=["volume"], dtype=torch.float64) as volume:
        assert volume.weights.sum() == pytest.approx(1 / 6)
    with h5py.File(tmp_path / "converted/geometry.h5") as geometry:
        assert [g.attrs["dimension"] for g in geometry["zones"].values()] == [2, 3]
        assert all(g["weights"].dtype == np.float64 for g in geometry["zones"].values())
    grids = ET.parse(descriptor).findall("./Domain/Grid/Grid/Grid")
    assert [g.find("Topology").attrib["TopologyType"] for g in grids] == [
        "Quadrilateral",
        "Tetrahedron",
    ]
    stamp = (tmp_path / "converted/geometry.h5").stat().st_mtime_ns
    PLT2HDF5(source).convert(
        tmp_path / "reuse",
        fields=["p"],
        geometry_file=tmp_path / "converted/geometry.h5",
    )
    assert (tmp_path / "converted/geometry.h5").stat().st_mtime_ns == stamp
