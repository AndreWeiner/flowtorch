"""Read-only offsets into legacy TDV71 FE-block data, without reader downcasts.

ParaView/VisIt's legacy reader returns float32 even for double source arrays.
This deliberately narrow adapter preserves source arrays for the supplied TAU
files. Topology remains integer-valued; geometry and numerical operations use VTK/FlowTorch.
Unsupported layouts fail explicitly. Layout reference:
https://github.com/visit-dav/visit/blob/develop/src/databases/Tecplot/TecplotFile.C
"""

from pathlib import Path
import struct
import numpy as np


class ArrayReference:
    """A binary array index without an open file or resident field buffer."""

    def __init__(self, path, dtype, offset, shape):
        self.path, self.dtype, self.offset, self.shape = (
            Path(path),
            np.dtype(dtype),
            offset,
            tuple(shape),
        )

    def __len__(self):
        return self.shape[0]

    def map(self):
        return np.memmap(
            self.path, dtype=self.dtype, mode="r", offset=self.offset, shape=self.shape
        )

    def __getitem__(self, index):
        return self.map()[index]

    def __array__(self, dtype=None, copy=None):
        value = np.asarray(self.map(), dtype=dtype)
        return value.copy() if copy else value


def index_by_zone(path):
    path = Path(path)
    length = path.stat().st_size
    with path.open("rb") as stream:
        if stream.read(8) != b"#!TDV71 ":
            raise ValueError("Native PLT input supports TDV71 FE-block files only")
        byte_order = stream.read(4)
        if byte_order not in (b"\x01\x00\x00\x00", b"\x00\x00\x00\x01"):
            raise ValueError("Invalid TDV71 byte-order marker")
        endian = "<" if byte_order == b"\x01\x00\x00\x00" else ">"

        def read(code):
            size = struct.calcsize(code)
            data = stream.read(size)
            if len(data) != size:
                raise ValueError("Truncated TDV71 file")
            return struct.unpack(endian + code, data)[0]

        def string():
            values = []
            for _ in range(65536):
                code = read("i")
                if code == 0:
                    return "".join(values)
                values.append(chr(code))
            raise ValueError("Unterminated TDV71 string")

        string()
        nvars = read("i")
        if not 0 < nvars < 10000:
            raise ValueError("Invalid TDV71 variable count")
        names = [string() for _ in range(nvars)]
        if len(set(names)) != len(names) or not all(
            key in names for key in ("X", "Y", "Z")
        ):
            raise ValueError("Need unique variables including X, Y and Z coordinates")
        zones = []
        while True:
            marker = read("f")
            if marker == 357:
                break
            if marker != 299:
                raise ValueError("Unsupported TDV71 header record")
            name = string()
            packing = read("i")
            read("i")  # strand/unused
            if packing != 2:
                raise ValueError("Native input requires TDV71 FE-block zones")
            points, cells, element = read("i"), read("i"), read("i")
            nodes = {0: 3, 1: 4, 2: 4, 3: 8}.get(element)
            if nodes is None or points <= 0 or cells <= 0:
                raise ValueError("Unsupported TDV71 element or zone size")
            zones.append((name, points, cells, nodes, element))
        result = {}
        for name, points, cells, nodes, element in zones:
            if name in result:
                raise ValueError("Duplicate TDV71 zone names")
            if read("f") != 299 or read("i") != 0:
                raise ValueError("Unsupported TDV71 data record or variable sharing")
            formats = [read("i") for _ in names]
            arrays = {}
            for variable, code in zip(names, formats):
                kind = {1: "f4", 2: "f8", 3: "i4", 4: "i2", 5: "u1"}.get(code)
                if kind is None:
                    raise ValueError("Unsupported TDV71 variable type")
                dtype = np.dtype(endian + kind)
                offset = stream.tell()
                size = points * dtype.itemsize
                if offset + size > length:
                    raise ValueError("Truncated TDV71 variable")
                arrays[variable] = ArrayReference(path, dtype, offset, (points,))
                stream.seek(size, 1)
            if read("i") != 0:
                raise ValueError("Shared TDV71 connectivity is unsupported")
            connectivity = ArrayReference(
                path, endian + "i4", stream.tell(), (cells, nodes)
            )
            stream.seek(cells * nodes * 4, 1)
            if stream.tell() > length:
                raise ValueError("Truncated TDV71 connectivity")
            result[name] = {
                "fields": arrays,
                "connectivity": connectivity,
                "cell_type": element,
            }
        if stream.tell() != length:
            raise ValueError("Unexpected trailing TDV71 records")
        return result
