"""Geometry lifetime, incremental persistence and failure/reuse semantics."""

import weakref

import h5py
import numpy as np
import pytest

from flowtorch.data import HDF5Writer, PLT2HDF5, TecplotDataloader, UnstructuredGeometry
from flowtorch.data._hdf5_schema import validate_geometry_file
from tests.data.test_unstructured_hdf5 import make_plt


def geometry():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return UnstructuredGeometry(
        points,
        np.array([[0, 1, 2]], dtype=np.int64),
        "Triangle",
        1,
        weights=np.full(3, 1 / 6),
        dimension=2,
    )


@pytest.mark.parametrize("compression", [None, "gzip", "zstd"])
def test_writer_releases_before_producing_next(tmp_path, compression):
    if compression == "zstd":
        pytest.importorskip("hdf5plugin")
    path = tmp_path / "geometry.h5"
    refs = []

    def zones():
        for i in range(3):
            assert all(ref() is None for ref in refs)
            if i:
                with h5py.File(path, "r") as file:
                    assert len(file["zones"]) == i
                    assert not file.attrs["complete"]
            geom = geometry()
            refs.extend(
                [
                    weakref.ref(geom),
                    weakref.ref(geom.points),
                    weakref.ref(geom.topology),
                ]
            )
            try:
                yield str(i), geom
            finally:
                del geom

    identity = HDF5Writer.write_geometry(
        path, zones(), compression=compression, shuffle=compression is not None
    )
    assert all(ref() is None for ref in refs)
    with h5py.File(path, "r") as file:
        validate_geometry_file(file)
        assert file.attrs["geometry_id"] == identity
    before = path.read_bytes()
    assert (
        HDF5Writer.write_geometry(path, ((str(i), geometry()) for i in range(3)))
        == identity
    )
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="differs"):
        HDF5Writer.write_geometry(path, [("wrong", geometry())])
    assert path.read_bytes() == before


def test_failure_keeps_incomplete_file_and_closes_generator(tmp_path):
    path = tmp_path / "geometry.h5"
    closed = []

    def zones():
        try:
            yield "a", geometry()
            raise RuntimeError("source interrupted")
        finally:
            closed.append(True)

    with pytest.raises(RuntimeError, match="interrupted"):
        HDF5Writer.write_geometry(path, zones())
    assert closed == [True]
    with h5py.File(path, "r") as file:
        assert list(file["zones"]) == ["z000000"]
        with pytest.raises(ValueError, match="incomplete"):
            validate_geometry_file(file)
    closed.clear()

    def duplicate():
        try:
            yield "a", geometry()
            yield "a", geometry()
        finally:
            closed.append(True)

    with pytest.raises(ValueError, match="unique"):
        HDF5Writer.write_geometry(tmp_path / "duplicate.h5", duplicate())
    assert closed == [True]


def test_converter_has_one_live_geometry(tmp_path, monkeypatch):
    source = make_plt(tmp_path / "plt", count=3)
    original = TecplotDataloader.geometry
    refs = []
    calls = []

    def checked(self, time=None):
        assert all(ref() is None for ref in refs), "Previous geometry still retained"
        geom = original(self, time)
        refs.extend(
            [weakref.ref(geom), weakref.ref(geom.points), weakref.ref(geom.topology)]
        )
        calls.append((self.zone, time))
        return geom

    monkeypatch.setattr(TecplotDataloader, "geometry", checked)
    PLT2HDF5(source).convert(tmp_path / "out", fields=["cp"])
    assert len(calls) == 6  # Two zones, all three snapshots validated.
    assert all(ref() is None for ref in refs)
