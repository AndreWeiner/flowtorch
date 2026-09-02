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
