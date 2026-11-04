"""Bounded field reads from native OpenFOAM streams (including gzip)."""

import gzip
import re
from itertools import islice
from os.path import isfile
from typing import BinaryIO, Iterator, Optional, Tuple, List, cast

import numpy as np


def resolve(path: str) -> str:
    """Prefer the plain file when both storage variants exist."""
    return path if isfile(path) or path.endswith(".gz") else path + ".gz"


def foam_open(path: str) -> BinaryIO:
    path = resolve(path)
    return cast(
        BinaryIO, gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")
    )


def token(stream: BinaryIO) -> bytes:
    """Read a dictionary token without reading ahead into binary payloads."""
    pending = getattr(stream, "_foam_token", None)
    if pending is not None:
        delattr(stream, "_foam_token")
        return pending
    while True:
        char = stream.read(1)
        if not char:
            raise ValueError("Unexpected end of OpenFOAM file")
        if char.isspace():
            continue
        if char == b"/":
            following = stream.read(1)
            if following == b"/":
                stream.readline()
                continue
            if following == b"*":
                previous = b""
                while True:
                    current = stream.read(1)
                    if not current:
                        raise ValueError("Unterminated OpenFOAM comment")
                    if previous == b"*" and current == b"/":
                        break
                    previous = current
                continue
            raise ValueError("Unexpected slash in OpenFOAM dictionary")
        break
    if char in b"{}();":
        return char
    if char == b'"':
        value = bytearray(char)
        while True:
            char = stream.read(1)
            if not char:
                raise ValueError("Unterminated OpenFOAM string")
            value.extend(char)
            if char == b'"':
                return bytes(value)
    value = bytearray(char)
    while True:
        char = stream.read(1)
        if not char or char.isspace():
            return bytes(value)
        if char in b"{}();":
            setattr(stream, "_foam_token", char)
            return bytes(value)
        value.extend(char)


def header(stream: BinaryIO) -> dict:
    while token(stream) != b"FoamFile":
        pass
    if token(stream) != b"{":
        raise ValueError("Missing FoamFile dictionary")
    result: dict[bytes, bytes] = {}
    while True:
        key = token(stream)
        if key == b"}":
            return result
        values = []
        while (value := token(stream)) != b";":
            values.append(value)
        result[key] = b" ".join(values)


def binary_dtype(metadata: dict, kind: str = "scalar") -> np.dtype:
    arch = metadata.get(b"arch", b'"LSB;label=32;scalar=64"').strip(b'"')
    endian = (
        "<" if arch.startswith(b"LSB") else ">" if arch.startswith(b"MSB") else None
    )
    match = re.search(kind.encode() + rb"=(32|64)(?:;|$)", arch)
    if endian is None or match is None:
        raise ValueError(f"Unsupported OpenFOAM architecture: {arch!r}")
    return np.dtype(
        endian + ("f" if kind == "scalar" else "i") + str(int(match[1]) // 8)
    )


def field_info(
    stream: BinaryIO,
) -> Tuple[dict, int, Optional[int], Optional[List[float]]]:
    metadata = header(stream)
    dimensions = {
        b"volScalarField": 1,
        b"volVectorField": 3,
        b"volSphericalTensorField": 1,
        b"volSymmTensorField": 6,
        b"volTensorField": 9,
    }
    field_class = metadata.get(b"class")
    if field_class not in dimensions:
        raise ValueError(f"Field type {field_class!r} not supported")
    while token(stream) != b"internalField":
        pass
    mode = token(stream)
    dim = dimensions[field_class]
    if mode == b"uniform":
        bracketed = field_class != b"volScalarField"
        if bracketed and token(stream) != b"(":
            raise ValueError("Missing uniform vector opening delimiter")
        values = [float(token(stream)) for _ in range(dim)]
        if bracketed and token(stream) != b")":
            raise ValueError("Missing uniform vector closing delimiter")
        if token(stream) != b";":
            raise ValueError("Missing uniform field terminator")
        return metadata, dim, None, values
    if mode != b"nonuniform":
        raise ValueError(f"Unsupported internalField mode {mode!r}")
    token(stream)  # List<scalar>, List<vector>, ...
    count = int(token(stream))
    if count < 0 or token(stream) != b"(":
        raise ValueError("Invalid nonuniform field list")
    return metadata, dim, count, None


def read_values(
    stream: BinaryIO, metadata: dict, dim: int, count: int, start: int, stop: int
) -> np.ndarray:
    """Read only selected values; ASCII scanning has bounded working storage."""
    size = max(0, stop - start) * dim
    if metadata.get(b"format") == b"binary":
        dtype = binary_dtype(metadata)
        stream.seek(start * dim * dtype.itemsize, 1)
        raw = stream.read(size * dtype.itemsize)
        if len(raw) != size * dtype.itemsize:
            raise ValueError("Truncated binary internalField")
        if stop == count and (token(stream) != b")" or token(stream) != b";"):
            raise ValueError("Missing binary internalField terminator")
        return np.frombuffer(raw, dtype=dtype).astype(
            dtype.newbyteorder("="), copy=True
        )
    if metadata.get(b"format") != b"ascii":
        raise ValueError("Unsupported OpenFOAM file format")

    source = _ascii_values(stream)
    iterator = islice(source, start * dim, stop * dim)
    try:
        values = np.fromiter(iterator, dtype=np.float64, count=size)
        if stop == count:
            if next(source, None) is not None:
                raise ValueError("More values than declared in internalField")
        return values
    except ValueError as error:
        raise ValueError("Truncated or invalid ASCII internalField") from error


def _ascii_values(stream: BinaryIO) -> Iterator[float]:
    # readline bounds memory by the longest input line, rather than snapshot size.
    in_comment = False
    depth = 1
    for line in stream:
        line = re.sub(rb"/\*.*?\*/", b" ", line)
        if in_comment:
            if b"*/" not in line:
                continue
            line = line.split(b"*/", 1)[1]
            in_comment = False
        if b"/*" in line:
            line = line.split(b"/*", 1)[0]
            in_comment = True
        line = line.split(b"//", 1)[0]
        for match in re.finditer(rb"[^\s();]+|[();]", line):
            value = match[0]
            if value == b"(":
                depth += 1
            elif value == b")":
                depth -= 1
                if depth == 0:
                    tail = line[match.end() :].lstrip()
                    if not tail:
                        tail = token(stream)
                    if not tail.startswith(b";"):
                        raise ValueError("Missing ASCII internalField terminator")
                    return
            elif value == b";":
                raise ValueError("Unexpected internalField terminator")
            else:
                yield float(value)
    raise ValueError("Missing ASCII internalField closing delimiter")


def field_chunks(
    path: str, n_cells: int, chunk_size: int
) -> Iterator[Tuple[slice, np.ndarray]]:
    """Read a complete cell field once in bounded chunks.

    ASCII scanning also retains the current input line. Unlike repeated spatial
    slices, ASCII/gzip input is consumed sequentially.
    The caller must exhaust the iterator to validate the list terminator.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    with foam_open(path) as stream:
        metadata, dim, count, uniform = field_info(stream)
        if count is not None and count != n_cells:
            raise ValueError(f"Field cell count does not match mesh: {path}")
        source = (
            _ascii_values(stream)
            if uniform is None and metadata.get(b"format") == b"ascii"
            else None
        )
        for start in range(0, n_cells, chunk_size):
            stop = min(start + chunk_size, n_cells)
            if uniform is not None:
                values = np.broadcast_to(uniform, (stop - start, dim)).copy()
            elif source is not None:
                try:
                    values = np.fromiter(
                        islice(source, (stop - start) * dim),
                        dtype=np.float64,
                        count=(stop - start) * dim,
                    )
                except ValueError as error:
                    raise ValueError(
                        "Truncated or invalid ASCII internalField"
                    ) from error
            else:
                # Stream already points at this chunk; only the final call
                # consumes the closing delimiters (including for gzip binary).
                values = read_values(
                    stream, metadata, dim, n_cells - start, 0, stop - start
                )
            shape = (stop - start,) if dim == 1 else (stop - start, dim)
            yield slice(start, stop), values.reshape(shape)
        if source is not None and next(source, None) is not None:
            raise ValueError("More values than declared in internalField")


def read_list(
    stream: BinaryIO, metadata: dict, dim: int = 1, kind: str = "scalar"
) -> np.ndarray:
    """Read a complete primitive mesh list, validating its delimiters."""
    count = int(token(stream))
    if count < 0 or token(stream) != b"(":
        raise ValueError("Invalid OpenFOAM list")
    if metadata.get(b"format") == b"binary":
        dtype = binary_dtype(metadata, kind)
        raw = stream.read(count * dim * dtype.itemsize)
        if len(raw) != count * dim * dtype.itemsize:
            raise ValueError("Truncated OpenFOAM list")
        values = np.frombuffer(raw, dtype=dtype).astype(
            dtype.newbyteorder("="), copy=True
        )
    elif metadata.get(b"format") == b"ascii":

        def entries():
            for _ in range(count):
                if dim > 1 and token(stream) != b"(":
                    raise ValueError("Invalid vector list")
                for _ in range(dim):
                    yield (
                        float(token(stream)) if kind == "scalar" else int(token(stream))
                    )
                if dim > 1 and token(stream) != b")":
                    raise ValueError("Invalid vector list")

        values = np.fromiter(
            entries(),
            dtype=np.float64 if kind == "scalar" else np.int64,
            count=count * dim,
        )
    else:
        raise ValueError("Unsupported OpenFOAM format")
    if token(stream) != b")":
        raise ValueError("Missing OpenFOAM list closing delimiter")
    return values.reshape(count, dim) if dim > 1 else values
