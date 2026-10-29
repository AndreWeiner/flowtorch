"""Tests for the isolated Tecplot pvpython backend."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch as pt

import flowtorch.data.tecplot_dataloader as tecplot_module
from flowtorch.data.tecplot_dataloader import (
    ParaViewProcessError,
    TecplotDataloader,
    _PvpythonClient,
)


class _FakeClient:
    executable = None

    def __init__(self, executable):
        type(self).executable = executable
        self.closed = False

    def request(self, operation, **parameters):
        if operation == "zone_names":
            return [
                {"name": "lower", "full_name": "root/lower", "path": [0, 0]},
                {"name": "upper", "full_name": "root/upper", "path": [0, 1]},
            ]
        if operation == "field_names":
            if parameters["association"] == "cell":
                return ["cell_volume"]
            if parameters["block_path"] == [0, 1]:
                return ["density", "temperature"]
            return ["density", "pressure"]
        if operation == "weights":
            return np.arange(1, 5, dtype=np.float64)
        if operation == "locations":
            return np.ones((4, 3))
        if operation == "vertices":
            zone = parameters["block_path"][-1]
            return np.arange(12, dtype=np.float64).reshape(4, 3) + zone
        if operation == "field":
            offset = float(parameters["file_path"].split("t=")[-1].split(".plt")[0])
            zone = parameters["block_path"][-1]
            return np.arange(4, dtype=np.float64) + offset + zone
        raise AssertionError(f"unexpected operation: {operation}")

    def close(self):
        self.closed = True


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Tecplot subprocess backend requires POSIX file-descriptor passing",
)
def test_binary_subprocess_protocol():
    worker = Path(__file__).with_name("_fake_tecplot_worker.py")
    client = _PvpythonClient(sys.executable, worker_path=worker)
    try:
        assert client.request("json", value={"zones": ["lower", "upper"]}) == {
            "zones": ["lower", "upper"]
        }
        array = client.request("array")
        assert np.array_equal(array, np.arange(12).reshape(3, 4))
        with pytest.raises(ParaViewProcessError, match="fake worker error"):
            client.request("error")
    finally:
        client.close()


def test_binary_subprocess_backend_rejects_windows(monkeypatch):
    monkeypatch.setattr(tecplot_module.sys, "platform", "win32")
    with pytest.raises(NotImplementedError, match="not supported on Windows"):
        _PvpythonClient(sys.executable)


def test_from_tau_uses_configured_pvpython_without_paraview(tmp_path, monkeypatch):
    for time in ("1.0", "2.0"):
        (tmp_path / f"surface_i=1_t={time}.plt").touch()

    monkeypatch.setattr(tecplot_module, "_resolve_pvpython", lambda value: str(value))
    monkeypatch.setattr(tecplot_module, "_PvpythonClient", _FakeClient)

    with TecplotDataloader.from_tau(
        str(tmp_path),
        "surface_",
        pvpython="/opt/ParaView/bin/pvpython",
    ) as loader:
        assert _FakeClient.executable == "/opt/ParaView/bin/pvpython"
        assert loader.write_times == ["1.0", "2.0"]
        assert loader.zone_names == ["lower", "upper"]
        assert loader.field_names == {"1.0": ["density", "pressure"]}
        assert loader.vertices.shape == (4, 3)
        assert loader.weights.dtype == pt.float32
        assert loader.load_snapshot("density", loader.write_times).shape == (4, 2)

        loader.zone = "upper"
        assert pt.equal(loader.vertices, pt.from_numpy(np.arange(12).reshape(4, 3) + 1))
        assert loader.field_names == {"1.0": ["density", "temperature"]}
        assert loader.field_names_for("cell") == {"1.0": ["cell_volume"]}
        with pytest.raises(ValueError, match="constructor association"):
            loader.load_snapshot("cell_volume", "1.0", association="cell")


def test_from_tau_filters_extensionless_files_with_same_timestamp(
    tmp_path, monkeypatch
):
    extensionless = tmp_path / "surface_i=1_t=1.0"
    requested = tmp_path / "surface_i=1_t=1.0.plt"
    extensionless.touch()
    requested.touch()
    monkeypatch.setattr(tecplot_module, "_resolve_pvpython", lambda value: str(value))
    monkeypatch.setattr(tecplot_module, "_PvpythonClient", _FakeClient)

    with TecplotDataloader.from_tau(
        str(tmp_path), "surface_", suffix=".plt", pvpython="pvpython"
    ) as loader:
        assert loader._file_names == {"1.0": requested.name}


def test_unknown_field_is_rejected_before_worker_request(tmp_path, monkeypatch):
    (tmp_path / "surface_i=1_t=1.0.plt").touch()
    monkeypatch.setattr(tecplot_module, "_resolve_pvpython", lambda value: str(value))
    monkeypatch.setattr(tecplot_module, "_PvpythonClient", _FakeClient)

    with TecplotDataloader.from_tau(
        str(tmp_path), "surface_", pvpython="pvpython"
    ) as loader:
        with pytest.raises(ValueError, match="Unknown field"):
            loader.load_snapshot("temperature", "1.0")


def test_invalid_field_association_is_rejected(tmp_path, monkeypatch):
    (tmp_path / "surface_i=1_t=1.0.plt").touch()
    monkeypatch.setattr(tecplot_module, "_resolve_pvpython", lambda value: str(value))
    monkeypatch.setattr(tecplot_module, "_PvpythonClient", _FakeClient)

    with TecplotDataloader.from_tau(
        str(tmp_path), "surface_", pvpython="pvpython"
    ) as loader:
        with pytest.raises(ValueError, match="expected 'point' or 'cell'"):
            loader.load_snapshot("density", "1.0", association="field")


def test_constructor_cell_association_and_cached_weights(tmp_path, monkeypatch):
    (tmp_path / "surface_i=1_t=1.0.plt").touch()
    monkeypatch.setattr(tecplot_module, "_resolve_pvpython", lambda value: str(value))
    monkeypatch.setattr(tecplot_module, "_PvpythonClient", _FakeClient)
    with TecplotDataloader.from_tau(
        str(tmp_path), "surface_", association="cell", dtype=pt.float64
    ) as loader:
        assert loader.association == "cell"
        assert loader.field_names == {"1.0": ["cell_volume"]}
        assert loader.load_snapshot("cell_volume", "1.0").shape == (4,)
        assert loader.locations.shape == (4, 3)
        assert pt.equal(loader.weights, pt.arange(1, 5, dtype=pt.float64))
        first = loader.weights
        assert loader.weights is first
        loader.zone = "upper"
        assert loader.weights is not first
        loader.zone = "lower"
        assert loader.weights is first


def test_invalid_constructor_association_is_rejected_before_start(monkeypatch):
    def unexpected(value):
        raise AssertionError("Worker must not start")

    monkeypatch.setattr(tecplot_module, "_PvpythonClient", unexpected)
    with pytest.raises(ValueError, match="expected 'point' or 'cell'"):
        TecplotDataloader(".", {"0": "mesh.plt"}, association="field")


@pytest.mark.parametrize("dtype", [pt.float32, pt.float64])
@pytest.mark.parametrize("source_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("association", ["point", "cell"])
def test_requested_dtype_for_fields_and_geometry(
    tmp_path, monkeypatch, dtype, source_dtype, association
):
    precise = 1.0000000001234568

    class Client(_FakeClient):
        def request(self, operation, **parameters):
            value = super().request(operation, **parameters)
            if isinstance(value, np.ndarray):
                return np.full(value.shape, precise, dtype=source_dtype)
            return value

    (tmp_path / "surface_i=1_t=1.0.plt").touch()
    monkeypatch.setattr(tecplot_module, "_resolve_pvpython", lambda value: str(value))
    monkeypatch.setattr(tecplot_module, "_PvpythonClient", Client)
    with TecplotDataloader.from_tau(
        str(tmp_path), "surface_", dtype=dtype, association=association
    ) as loader:
        field = "density" if association == "point" else "cell_volume"
        expected = pt.tensor(np.array(precise, dtype=source_dtype), dtype=dtype)
        outputs = [
            loader.load_snapshot(field, "1.0"),
            loader.load_snapshot(field, ["1.0"]),
            *loader.load_snapshot([field], ["1.0"]),
            loader.vertices,
            loader.locations,
            loader.weights,
        ]
        for output in outputs:
            assert output.dtype == dtype
            pt.testing.assert_close(output, expected.expand_as(output), rtol=0, atol=0)
