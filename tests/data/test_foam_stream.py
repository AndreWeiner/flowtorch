"""Native field slicing and gzip regression tests without external datasets."""

import gzip
import io
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from flowtorch.data import FOAMDataloader, FOAMCase, FOAMMesh
from flowtorch.data._foam_stream import field_info, read_values


def foam_header(cls="volScalarField", binary=False, arch="LSB;label=32;scalar=64"):
    return (
        "/* comment with class and format */\n"
        + "\n" * 45
        + f'FoamFile {{ version 2.0; format {"binary" if binary else "ascii"}; '
        f'class {cls}; arch "{arch}"; object p; }}\n'
    ).encode()


def case(tmp_path, processors=1):
    for proc in range(processors):
        root = tmp_path if processors == 1 else tmp_path / f"processor{proc}"
        mesh = root / "constant/polyMesh"
        mesh.mkdir(parents=True)
        for name in ("owner", "neighbour", "points", "faces", "boundary"):
            (mesh / name).write_bytes(b'FoamFile { note "nCells:4"; }\n')
        (root / "0").mkdir()
    return tmp_path


def write_field(
    path, values, dim=1, binary=False, compressed=False, endian="<", width=8
):
    cls = {
        1: "volScalarField",
        3: "volVectorField",
        6: "volSymmTensorField",
        9: "volTensorField",
    }[dim]
    data = foam_header(
        cls, binary, f'{"LSB" if endian == "<" else "MSB"};label=64;scalar={width*8}'
    )
    data += f"dimensions [0 0 0 0 0 0 0]; internalField nonuniform List<scalar> {len(values)}\n(".encode()
    if binary:
        data += np.asarray(values, dtype=f"{endian}f{width}").tobytes()
    else:
        for row in np.asarray(values).reshape(-1, dim):
            data += (
                " ".join(map(str, row))
                if dim == 1
                else "(" + " ".join(map(str, row)) + ")"
            ).encode() + b"\n"
    data += b");\nboundaryField {}\n"
    if compressed:
        path = Path(str(path) + ".gz")
        data = gzip.compress(data)
    path.write_bytes(data)


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("dim", [1, 3, 6, 9])
def test_fields_slices_and_series(tmp_path, compressed, binary, dim):
    root = case(tmp_path, 2)
    expected = torch.arange(8 * dim, dtype=torch.float64).reshape(8, dim)
    if dim == 1:
        expected = expected[:, 0]
    for proc in range(2):
        write_field(
            root / f"processor{proc}/0/p",
            expected[proc * 4 : (proc + 1) * 4],
            dim,
            binary,
            compressed,
        )
    loader = FOAMDataloader(str(root), dtype=torch.float64)
    assert loader.field_names == {"0": ["p"]}
    assert loader.snapshot_shape("p", "0") == expected.shape
    torch.testing.assert_close(loader.load_snapshot("p", "0"), expected)
    for selection in (slice(2, 6), slice(-2, None), slice(6, 2), slice(50, 80)):
        torch.testing.assert_close(
            loader.load_snapshot_slice("p", "0", selection), expected[selection]
        )
    series = loader.load_snapshot_slice(["p"], ["0", "0"], slice(2, 6))[0]
    torch.testing.assert_close(series, torch.stack([expected[2:6]] * 2, dim=-1))
    with pytest.raises(ValueError, match="stride"):
        loader.load_snapshot_slice("p", "0", slice(None, None, 2))


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("width", [4, 8])
def test_binary_architecture(tmp_path, endian, width):
    case(tmp_path)
    expected = torch.tensor([1.25, -4.5, 2.75, 0.0], dtype=torch.float64)
    write_field(tmp_path / "0/p", expected, binary=True, endian=endian, width=width)
    torch.testing.assert_close(
        FOAMDataloader(str(tmp_path), dtype=torch.float64).load_snapshot("p", "0"),
        expected,
    )


@pytest.mark.parametrize("dim", [1, 3])
def test_uniform(tmp_path, dim):
    case(tmp_path)
    value = "2.5" if dim == 1 else "(1 2 3)"
    (tmp_path / "0/p").write_bytes(
        foam_header("volScalarField" if dim == 1 else "volVectorField")
        + f"internalField uniform {value};".encode()
    )
    actual = FOAMDataloader(str(tmp_path)).load_snapshot_slice("p", "0", slice(1, 3))
    expected = (
        torch.full((2,), 2.5)
        if dim == 1
        else torch.tensor([[1.0, 2.0, 3.0]]).repeat(2, 1)
    )
    torch.testing.assert_close(actual, expected)


def test_no_owner_note(tmp_path):
    mesh = tmp_path
    (mesh / "owner").write_bytes(foam_header("labelList") + b"4(0 1 1 3)")
    assert FOAMMesh._get_n_cells(str(mesh)) == 4
    (mesh / "owner").write_bytes(b"FoamFile {}")
    with pytest.raises(ValueError):
        FOAMMesh._get_n_cells(str(mesh))


def test_missing_mesh_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        FOAMCase(str(tmp_path))


def test_binary_slice_reads_only_requested_payload():
    class CountingStream(io.BytesIO):
        def read(self, count=-1):
            assert count >= 0
            result = super().read(count)
            self.bytes_read += len(result)
            return result

    payload = np.arange(100000, dtype="<f8").tobytes()
    stream = CountingStream(
        foam_header(binary=True)
        + b"internalField nonuniform List<scalar> 100000\n("
        + payload
        + b");"
    )
    stream.bytes_read = 0
    metadata, dim, count, _ = field_info(stream)
    stream.bytes_read = 0
    values = read_values(stream, metadata, dim, count, 90000, 90010)
    np.testing.assert_array_equal(values, np.arange(90000, 90010))
    assert stream.bytes_read == 80


@pytest.mark.parametrize("binary", [False, True])
def test_truncated_fields_fail(tmp_path, binary):
    case(tmp_path)
    write_field(tmp_path / "0/p", [1.0, 2.0, 3.0, 4.0], binary=binary)
    data = (tmp_path / "0/p").read_bytes()
    (tmp_path / "0/p").write_bytes(
        data[: data.index(b"List<scalar>") + len(b"List<scalar>")] + b" 4\n("
    )
    with pytest.raises(ValueError):
        FOAMDataloader(str(tmp_path)).load_snapshot("p", "0")


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("binary", [False, True])
def test_cube_mesh(tmp_path, compressed, binary):
    root = case(tmp_path)
    mesh = root / "constant/polyMesh"
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ]
    )

    def save(name, cls, values, dim=1, kind="scalar"):
        data = foam_header(cls, binary, "MSB;label=64;scalar=32")

        def block(values, dim, kind):
            values = np.asarray(values)
            payload = f"{len(values)}\n(".encode()
            if binary:
                payload += values.astype(">i8" if kind == "label" else ">f4").tobytes()
            else:
                for row in values.reshape(-1, dim):
                    text = " ".join(map(str, row))
                    payload += (text if dim == 1 else "(" + text + ")").encode() + b"\n"
            return payload + b")\n"

        if name == "faces":
            if binary:
                data += block(np.arange(0, 25, 4), 1, "label")
                data += block(faces.ravel(), 1, "label")
            else:
                data += (
                    b"6\n(\n"
                    + b"".join(
                        b"4(" + " ".join(map(str, row)).encode() + b")\n"
                        for row in faces
                    )
                    + b")"
                )
        else:
            data += block(values, dim, kind)
        path = mesh / name
        if compressed:
            path.unlink()
            path = Path(str(path) + ".gz")
            data = gzip.compress(data)
        path.write_bytes(data)

    save("points", "vectorField", points, 3)
    save("faces", "faceCompactList" if binary else "faceList", faces)
    save("owner", "labelList", [0] * 6, kind="label")
    save("neighbour", "labelList", [], kind="label")
    loader = FOAMDataloader(str(root), dtype=torch.float64)
    torch.testing.assert_close(
        loader.vertices, torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64)
    )
    torch.testing.assert_close(loader.weights, torch.ones(1, dtype=torch.float64))


@pytest.mark.integration
@pytest.mark.parametrize(
    "name",
    [
        "of_cavity_ascii",
        "of_cavity_binary",
        "of_cavity_ascii_parallel",
        "of_cavity_binary_parallel",
    ],
)
def test_compressed_cavity(tmp_path, name):
    from flowtorch import DATASETS

    if name not in DATASETS:
        pytest.skip("OpenFOAM cavity dataset unavailable")
    source = Path(DATASETS[name])
    target = tmp_path / "case"
    shutil.copytree(source, target)
    for path in target.rglob("*"):
        if path.is_file():
            Path(str(path) + ".gz").write_bytes(gzip.compress(path.read_bytes()))
            path.unlink()
    original = FOAMDataloader(str(source), dtype=torch.float64)
    compressed = FOAMDataloader(str(target), dtype=torch.float64)
    torch.testing.assert_close(original.vertices, compressed.vertices)
    torch.testing.assert_close(original.weights, compressed.weights)
    for field in ("p", "U"):
        for time in ("0", "0.5"):
            torch.testing.assert_close(
                original.load_snapshot(field, time),
                compressed.load_snapshot(field, time),
            )
            torch.testing.assert_close(
                original.load_snapshot_slice(field, time, slice(90, 110)),
                compressed.load_snapshot_slice(field, time, slice(90, 110)),
            )
    if not original._case.distributed:
        from flowtorch.data import FOAM2HDF5

        FOAM2HDF5(str(target)).convert(
            "compressed.h5", fields=["p", "U"], times=["0.5"]
        )
        # Re-open with the intended reader and verify field values.
        from flowtorch.data import HDF5Dataloader

        result = HDF5Dataloader(str(target / "compressed.h5"), dtype=torch.float64)
        torch.testing.assert_close(
            result.load_snapshot("p", "0.5"), original.load_snapshot("p", "0.5")
        )


def test_spherical_uniform_and_ascii(tmp_path):
    case(tmp_path)
    path = tmp_path / "0/p"
    path.write_bytes(
        foam_header("volSphericalTensorField") + b"internalField uniform (2);"
    )
    loader = FOAMDataloader(str(tmp_path))
    torch.testing.assert_close(loader.load_snapshot("p", "0"), torch.full((4,), 2.0))
    path.write_bytes(
        foam_header("volSphericalTensorField")
        + b"internalField nonuniform List<sphericalTensor> 4 ((1)(2)(3)(4));"
    )
    torch.testing.assert_close(loader.load_snapshot("p", "0"), torch.arange(1.0, 5.0))


@pytest.mark.parametrize("payload", [b"3 (1 2 3", b"2 (1 2 3);", b"3 (1 2);"])
def test_malformed_ascii_list(tmp_path, payload):
    case(tmp_path)
    (tmp_path / "0/p").write_bytes(
        foam_header() + b"internalField nonuniform List<scalar> " + payload
    )
    with pytest.raises(ValueError):
        FOAMDataloader(str(tmp_path)).load_snapshot("p", "0")
