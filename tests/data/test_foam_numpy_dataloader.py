"""Tests for foamToNumpy function-object output loading."""

from pathlib import Path

import numpy as np
import pytest
import torch as pt

from flowtorch.data import FOAMNumpyDataloader


def _write_segment(root: Path, name: str) -> Path:
    segment = root / name
    segment.mkdir(parents=True)
    (segment / "segmentInfo").write_text(
        "startTime 0;\nfirstOutput 0.1;\ndataType float64;\n"
        "order fortran;\nbatchSize 2;\n"
    )
    return segment


def _vector_values(scalar: np.ndarray) -> np.ndarray:
    return np.asfortranarray(np.stack((scalar, scalar + 100.0, scalar + 200.0), axis=1))


def _write_batch(
    segment: Path,
    index: int,
    times: list[float],
    pressure: list[np.ndarray],
    *,
    count: int | None = None,
    mesh_revision: int = 0,
    c_order: bool = False,
) -> Path:
    batch = segment / f"batch_{index:06d}"
    batch.mkdir()
    committed = len(times) if count is None else count
    state = (
        f"batch {index};\n"
        f"meshRevision {mesh_revision};\n"
        f"count {committed};\n"
        "sealed true;\n"
        "fields (p U);\n"
        "fieldClasses\n{\n"
        "    p volScalarField;\n"
        "    U volVectorField;\n"
        "}\n"
    )
    (batch / "state").write_text(state)
    np.save(batch / "times.npy", np.asfortranarray(np.asarray(times, dtype=float)))
    for processor, scalar in enumerate(pressure):
        scalar = np.asarray(scalar, dtype=np.float64)
        field = np.array(scalar, order="C") if c_order else np.asfortranarray(scalar)
        np.save(batch / f"p_proc_{processor}.npy", field)
        np.save(batch / f"U_proc_{processor}.npy", _vector_values(scalar))
    return batch


def _write_geometry(segment: Path, centres: list[np.ndarray]) -> None:
    geometry = segment / "geometry_000000"
    geometry.mkdir()
    for processor, values in enumerate(centres):
        values = np.asarray(values, dtype=np.float64)
        np.save(
            geometry / f"cellCentres_proc_{processor}.npy",
            np.asfortranarray(values[..., None]),
        )
        volumes = np.arange(1, values.shape[0] + 1, dtype=np.float64)[..., None]
        np.save(
            geometry / f"cellVolumes_proc_{processor}.npy",
            np.asfortranarray(volumes),
        )


def _save_fortran(path: Path, values: np.ndarray) -> None:
    """Write an array with an explicitly Fortran-order NumPy header."""
    values = np.asfortranarray(values)
    with path.open("wb") as stream:
        np.lib.format.write_array_header_1_0(
            stream,
            {
                "descr": np.lib.format.dtype_to_descr(values.dtype),
                "fortran_order": True,
                "shape": values.shape,
            },
        )
        stream.write(values.tobytes(order="F"))


def _write_zone_segment(
    root: Path,
    name: str,
    times: list[float],
    pressure: list[np.ndarray],
    zones: dict[str, list[np.ndarray]],
    *,
    mesh_revision: int = 0,
) -> Path:
    """Write a compact fixture following numpyToFoam zone-layout version 1."""
    segment = root / name
    segment.mkdir(parents=True)
    n_processors = len(pressure)
    (segment / "segmentInfo").write_text(
        "zoneLayoutVersion 1;\nregion region0;\n"
        f"nProcs {n_processors};\nstartTime 0;\nfirstOutput {times[0]};\n"
        "dataType float64;\norder fortran;\nbatchSize 100;\n"
    )

    geometry = segment / f"geometry_{mesh_revision:06d}"
    geometry.mkdir()
    zone_names = " ".join(zones)
    for processor, values in enumerate(pressure):
        n_cells = values.shape[0]
        (geometry / f"mesh_proc_{processor}").write_text(
            f"nCells {n_cells};\nmeshRevision {mesh_revision};\n"
        )
        centres = np.column_stack(
            (
                np.arange(n_cells, dtype=np.float64) + 10.0 * processor,
                np.zeros(n_cells),
                np.zeros(n_cells),
            )
        )
        volumes = np.arange(1, n_cells + 1, dtype=np.float64)
        for zone, mappings in zones.items():
            zone_geometry = geometry / "cellZones" / zone
            zone_geometry.mkdir(parents=True, exist_ok=True)
            cell_ids = np.asarray(mappings[processor], dtype=np.int64)
            _save_fortran(zone_geometry / f"cellIds_proc_{processor}.npy", cell_ids)
            _save_fortran(
                zone_geometry / f"cellCentres_proc_{processor}.npy",
                centres[cell_ids, ..., None],
            )
            _save_fortran(
                zone_geometry / f"cellVolumes_proc_{processor}.npy",
                volumes[cell_ids, None],
            )

    batch = segment / "batch_000000"
    batch.mkdir()
    (batch / "state").write_text(
        f"batch 0;\nmeshRevision {mesh_revision};\ncount {len(times)};\n"
        f"sealed true;\nfields (p U);\nzoneLayoutVersion 1;\n"
        f"cellZones ({zone_names});\nfieldClasses\n{{\n"
        "    p volScalarField;\n    U volVectorField;\n}\n"
    )
    _save_fortran(batch / "times.npy", np.asarray(times, dtype=np.float64))
    for zone, mappings in zones.items():
        zone_batch = batch / "cellZones" / zone
        zone_batch.mkdir(parents=True)
        for processor, values in enumerate(pressure):
            cell_ids = np.asarray(mappings[processor], dtype=np.int64)
            selected = np.asarray(values, dtype=np.float64)[cell_ids]
            _save_fortran(zone_batch / f"p_proc_{processor}.npy", selected)
            _save_fortran(
                zone_batch / f"U_proc_{processor}.npy", _vector_values(selected)
            )
    return segment


@pytest.fixture()
def foam_numpy_output(tmp_path: Path) -> Path:
    root = tmp_path / "numpyExport"
    first = _write_segment(root, "0")
    _write_batch(
        first,
        0,
        [0.1, 0.2],
        [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0]])],
    )
    _write_batch(
        first,
        1,
        [0.3],
        [np.array([[7.0], [8.0]]), np.array([[9.0]])],
    )
    centres = [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([[2.0, 0.0, 0.0]]),
    ]
    _write_geometry(first, centres)

    second = _write_segment(root, "1")
    _write_batch(
        second,
        0,
        [0.2, 0.4],
        [np.array([[20.0, 40.0], [21.0, 41.0]]), np.array([[22.0, 42.0]])],
    )
    _write_geometry(second, centres)
    return root


def test_later_segment_replaces_duplicate_only(foam_numpy_output: Path):
    loader = FOAMNumpyDataloader(str(foam_numpy_output))

    assert loader.write_times == ["0.1", "0.2", "0.3", "0.4"]
    assert loader.field_names == {time: ["p", "U"] for time in loader.write_times}
    assert pt.equal(loader.load_snapshot("p", "0.2"), pt.tensor([20.0, 21.0, 22.0]))
    # Pointwise replacement retains 0.3 from the older segment.
    assert pt.equal(loader.load_snapshot("p", "0.3"), pt.tensor([7.0, 8.0, 9.0]))


def test_ignore_segments(foam_numpy_output: Path):
    loader = FOAMNumpyDataloader(str(foam_numpy_output), ignore_segments="1")
    assert pt.equal(loader.load_snapshot("p", "0.2"), pt.tensor([2.0, 4.0, 6.0]))


def test_load_multiple_times_fields_and_dtype(foam_numpy_output: Path):
    loader = FOAMNumpyDataloader(str(foam_numpy_output), dtype=pt.float64)

    pressure = loader.load_snapshot("p", ["0.4", "0.1"])
    assert pressure.dtype == pt.float64
    assert pressure.shape == (3, 2)
    assert pressure.is_contiguous()
    assert pt.equal(
        pressure,
        pt.tensor([[40.0, 1.0], [41.0, 3.0], [42.0, 5.0]], dtype=pt.float64),
    )

    pressure, velocity = loader.load_snapshot(["p", "U"], "0.1")
    assert pressure.shape == (3,)
    assert velocity.shape == (3, 3)
    assert pt.equal(velocity[:, 0], pressure)
    assert pt.equal(velocity[:, 2], pressure + 200.0)


def test_load_snapshot_slice_reads_global_processor_interval(
    foam_numpy_output: Path,
):
    loader = FOAMNumpyDataloader(str(foam_numpy_output), dtype=pt.float64)

    pressure, velocity = loader.load_snapshot_slice(
        ["p", "U"], ["0.4", "0.1"], slice(1, 3)
    )

    assert pressure.shape == (2, 2)
    assert velocity.shape == (2, 3, 2)
    pt.testing.assert_close(
        pressure,
        loader.load_snapshot("p", ["0.4", "0.1"])[1:3],
    )
    pt.testing.assert_close(
        velocity,
        loader.load_snapshot("U", ["0.4", "0.1"])[1:3],
    )
    assert loader.snapshot_shape("U", "0.1") == (3, 3)


def test_load_snapshot_slice_preserves_explicit_single_time_axis(
    foam_numpy_output: Path,
):
    loader = FOAMNumpyDataloader(str(foam_numpy_output), dtype=pt.float64)

    pressure = loader.load_snapshot_slice("p", ["0.1"], slice(1, 3))

    assert pressure.shape == (2, 1)
    pt.testing.assert_close(pressure[..., 0], loader.load_snapshot("p", "0.1")[1:3])


def test_load_geometry_and_direct_segment_path(foam_numpy_output: Path):
    loader = FOAMNumpyDataloader(str(foam_numpy_output / "0"))

    assert pt.equal(
        loader.vertices,
        pt.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )
    assert pt.equal(loader.weights, pt.tensor([1.0, 2.0, 1.0]))


def test_later_batch_replaces_duplicate(tmp_path: Path):
    root = tmp_path / "output"
    segment = _write_segment(root, "0")
    _write_batch(segment, 0, [0.1, 0.2], [np.array([[1.0, 2.0]])])
    _write_batch(segment, 1, [0.2, 0.3], [np.array([[20.0, 30.0]])])

    loader = FOAMNumpyDataloader(str(root))

    assert loader.write_times == ["0.1", "0.2", "0.3"]
    assert pt.equal(loader.load_snapshot("p", "0.2"), pt.tensor([20.0]))


def test_segments_are_sorted_numerically(tmp_path: Path):
    root = tmp_path / "output"
    later = _write_segment(root, "10")
    _write_batch(later, 0, [0.1], [np.array([[10.0]])])
    earlier = _write_segment(root, "2")
    _write_batch(earlier, 0, [0.1], [np.array([[2.0]])])

    loader = FOAMNumpyDataloader(str(root))

    assert pt.equal(loader.load_snapshot("p", "0.1"), pt.tensor([10.0]))


def test_rejects_unknown_ignored_segment(foam_numpy_output: Path):
    with pytest.raises(ValueError, match="unknown NumPy segments"):
        FOAMNumpyDataloader(str(foam_numpy_output), ignore_segments=["does-not-exist"])


def test_ignores_uncommitted_trailing_values(tmp_path: Path):
    root = tmp_path / "output"
    segment = _write_segment(root, "0")
    _write_batch(
        segment,
        0,
        [0.1, 0.2, 0.3],
        [np.array([[1.0, 2.0, 999.0]])],
        count=2,
    )

    loader = FOAMNumpyDataloader(str(root))

    assert loader.write_times == ["0.1", "0.2"]
    assert pt.equal(
        loader.load_snapshot("p", loader.write_times), pt.tensor([[1.0, 2.0]])
    )


def test_missing_geometry_is_reported(tmp_path: Path):
    root = tmp_path / "output"
    segment = _write_segment(root, "0")
    _write_batch(segment, 0, [0.1], [np.array([[1.0]])])
    loader = FOAMNumpyDataloader(str(root))

    with pytest.raises(NotImplementedError, match="writeCellCentres"):
        _ = loader.vertices
    with pytest.raises(NotImplementedError, match="writeCellVolumes"):
        _ = loader.weights


def test_rejects_multiple_mesh_revisions(tmp_path: Path):
    root = tmp_path / "output"
    segment = _write_segment(root, "0")
    _write_batch(segment, 0, [0.1], [np.array([[1.0]])])
    _write_batch(segment, 1, [0.2], [np.array([[2.0]])], mesh_revision=1)

    with pytest.raises(ValueError, match="multiple mesh revisions"):
        FOAMNumpyDataloader(str(root))


def test_rejects_c_order_field_arrays(tmp_path: Path):
    root = tmp_path / "output"
    segment = _write_segment(root, "0")
    _write_batch(
        segment,
        0,
        [0.1, 0.2],
        [np.array([[1.0, 2.0], [3.0, 4.0]])],
        c_order=True,
    )

    with pytest.raises(ValueError, match="not Fortran-contiguous"):
        FOAMNumpyDataloader(str(root))


def test_supported_volume_field_classes(tmp_path: Path):
    root = tmp_path / "output"
    segment = _write_segment(root, "0")
    batch = segment / "batch_000000"
    batch.mkdir()
    classes = {
        "scalar": "volScalarField",
        "vector": "volVectorField",
        "spherical": "volSphericalTensorField",
        "symmetric": "volSymmTensorField",
        "tensor": "volTensorField",
    }
    fields = " ".join(classes)
    class_entries = "\n".join(
        f"    {field} {field_class};" for field, field_class in classes.items()
    )
    (batch / "state").write_text(
        "batch 0;\nmeshRevision 0;\ncount 2;\nsealed true;\n"
        f"fields ({fields});\nfieldClasses\n{{\n{class_entries}\n}}\n"
    )
    np.save(batch / "times.npy", np.array([0.1, 0.2]))
    components = {
        "scalar": 1,
        "vector": 3,
        "spherical": 1,
        "symmetric": 6,
        "tensor": 9,
    }
    for field, n_components in components.items():
        shape = (2, 2) if n_components == 1 else (2, n_components, 2)
        values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape, order="F")
        np.save(batch / f"{field}_proc_0.npy", values)

    loader = FOAMNumpyDataloader(str(root))

    for field, n_components in components.items():
        expected_shape = (2, 2) if n_components == 1 else (2, n_components, 2)
        assert loader.load_snapshot(field, loader.write_times).shape == expected_shape


@pytest.fixture()
def foam_numpy_zone_output(tmp_path: Path) -> Path:
    root = tmp_path / "zoneExport"
    pressure = [
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
    ]
    zones = {
        "left": [np.array([0, 2]), np.array([1])],
        "right": [np.array([1]), np.array([0, 2])],
        "overlap": [np.array([0]), np.array([], dtype=int)],
        "empty": [np.array([], dtype=int), np.array([], dtype=int)],
    }
    _write_zone_segment(root, "0", [0.1, 0.2], pressure, zones)
    return root


def test_cell_zone_selection_fields_and_geometry(foam_numpy_zone_output: Path):
    loader = FOAMNumpyDataloader(str(foam_numpy_zone_output))

    assert loader.zone_names == ["left", "right", "overlap", "empty"]
    assert loader.zone == "left"
    assert loader.snapshot_shape("U", "0.1") == (3, 3)
    pt.testing.assert_close(
        loader.load_snapshot("p", loader.write_times),
        pt.tensor([[1.0, 2.0], [5.0, 6.0], [9.0, 10.0]]),
    )
    pt.testing.assert_close(
        loader.vertices,
        pt.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [11.0, 0.0, 0.0]]),
    )
    pt.testing.assert_close(loader.weights, pt.tensor([1.0, 3.0, 2.0]))

    loader.zone = "right"
    expected = pt.tensor([[3.0, 4.0], [7.0, 8.0], [11.0, 12.0]])
    pt.testing.assert_close(loader.load_snapshot("p", loader.write_times), expected)
    pt.testing.assert_close(
        loader.load_snapshot_slice("p", loader.write_times, slice(1, 3)),
        expected[1:3],
    )
    pt.testing.assert_close(
        loader.vertices,
        pt.tensor([[1.0, 0.0, 0.0], [10.0, 0.0, 0.0], [12.0, 0.0, 0.0]]),
    )


def test_empty_cell_zone_and_empty_processor_portion(
    foam_numpy_zone_output: Path,
):
    loader = FOAMNumpyDataloader(str(foam_numpy_zone_output), zone="overlap")

    assert loader.load_snapshot("p", loader.write_times).shape == (1, 2)
    loader.zone = "empty"
    assert loader.load_snapshot("p", loader.write_times).shape == (0, 2)
    assert loader.load_snapshot("U", "0.1").shape == (0, 3)
    assert loader.vertices.shape == (0, 3)
    assert loader.weights.shape == (0,)


def test_zone_selection_errors_and_legacy_zone_state(
    foam_numpy_output: Path, foam_numpy_zone_output: Path
):
    with pytest.raises(ValueError, match="not found"):
        FOAMNumpyDataloader(str(foam_numpy_zone_output), zone="missing")

    loader = FOAMNumpyDataloader(str(foam_numpy_zone_output))
    with pytest.raises(ValueError, match="not found"):
        loader.zone = "missing"

    legacy = FOAMNumpyDataloader(str(foam_numpy_output))
    assert legacy.zone_names == []
    assert legacy.zone is None
    with pytest.raises(ValueError, match="no selectable cell zones"):
        legacy.zone = "left"
    with pytest.raises(ValueError, match="whole-mesh"):
        FOAMNumpyDataloader(str(foam_numpy_output), zone="left")


def test_cell_zone_restart_precedence(tmp_path: Path):
    root = tmp_path / "zoneExport"
    zones = {"core": [np.array([0]), np.array([0])]}
    _write_zone_segment(
        root,
        "0",
        [0.1, 0.2],
        [np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]])],
        zones,
    )
    _write_zone_segment(
        root,
        "1",
        [0.2, 0.3],
        [np.array([[20.0, 30.0]]), np.array([[40.0, 50.0]])],
        zones,
    )

    loader = FOAMNumpyDataloader(str(root))

    assert loader.write_times == ["0.1", "0.2", "0.3"]
    pt.testing.assert_close(
        loader.load_snapshot("p", loader.write_times),
        pt.tensor([[1.0, 20.0, 30.0], [3.0, 40.0, 50.0]]),
    )


def test_rejects_changed_cell_zone_mapping(tmp_path: Path):
    root = tmp_path / "zoneExport"
    pressure = [np.array([[1.0], [2.0]])]
    _write_zone_segment(root, "0", [0.1], pressure, {"core": [np.array([0])]})
    _write_zone_segment(root, "1", [0.2], pressure, {"core": [np.array([1])]})

    with pytest.raises(ValueError, match="mapping.*changes"):
        FOAMNumpyDataloader(str(root))


def test_rejects_invalid_cell_zone_ids(tmp_path: Path):
    root = tmp_path / "zoneExport"
    segment = _write_zone_segment(
        root,
        "0",
        [0.1],
        [np.array([[1.0], [2.0]])],
        {"core": [np.array([0, 1])]},
    )
    ids_path = segment / "geometry_000000" / "cellZones" / "core" / "cellIds_proc_0.npy"
    _save_fortran(ids_path, np.array([1, 0], dtype=np.int64))

    with pytest.raises(ValueError, match="strictly increasing"):
        FOAMNumpyDataloader(str(root))


def test_rejects_cell_ids_without_fortran_header(tmp_path: Path):
    root = tmp_path / "zoneExport"
    segment = _write_zone_segment(
        root,
        "0",
        [0.1],
        [np.array([[1.0], [2.0]])],
        {"core": [np.array([0, 1])]},
    )
    ids_path = segment / "geometry_000000" / "cellZones" / "core" / "cellIds_proc_0.npy"
    np.save(ids_path, np.array([0, 1], dtype=np.int64))

    with pytest.raises(ValueError, match="not stored in Fortran order"):
        FOAMNumpyDataloader(str(root))
