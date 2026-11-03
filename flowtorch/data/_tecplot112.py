"""Read-only TDV112 FE-block indices, following Tecplot's binary format guide.

https://tecplot.azureedge.net/products/360/current/360-data-format.html
Only linear line/triangle/quad/tetra/brick zones are supported. Header and array
indices are resident; field and connectivity payloads remain on disk. Geometry
calculations are handled by the existing VTK path, not by this parser.
"""

from pathlib import Path
import struct
from typing import Any
from typing import BinaryIO

import numpy as np

from ._tecplot71 import ArrayReference


class _Reader:
    def __init__(self, stream: BinaryIO, length: int, endian: str):
        self.stream, self.length, self.endian = stream, length, endian

    def read(self, code: str):
        size = struct.calcsize(code)
        data = self.stream.read(size)
        if len(data) != size:
            raise ValueError("Truncated TDV112 record")
        return struct.unpack(self.endian + code, data)[0]

    def string(self) -> str:
        values: list[str] = []
        for _ in range(65536):
            value = self.read("i")
            if value == 0:
                return "".join(values)
            if not 0 < value <= 0x10FFFF:
                raise ValueError("Invalid TDV112 string")
            values.append(chr(value))
        raise ValueError("Unterminated TDV112 string")

    def flag(self) -> int:
        value = self.read("i")
        if value not in (0, 1):
            raise ValueError("Invalid TDV112 flag")
        return value

    def skip(self, size: int) -> None:
        if size < 0 or self.stream.tell() + size > self.length:
            raise ValueError("Truncated TDV112 array")
        self.stream.seek(size, 1)

    def auxiliary(self) -> None:
        self.string()
        if self.read("i") != 0:
            raise ValueError("Unsupported TDV112 auxiliary value type")
        self.string()


class _PassiveArray:
    """A passive Tecplot variable is zero, allocated only for the requested slice."""

    def __init__(self, count: int, dtype: np.dtype):
        self.shape, self.dtype = (count,), dtype

    def __getitem__(self, part):
        if not isinstance(part, slice):
            raise ValueError("Passive arrays require a spatial slice")
        return np.zeros((len(range(*part.indices(self.shape[0]))),), dtype=self.dtype)


def index_by_zone(path: str | Path) -> dict:
    """Index supported TDV112 records without reading field/topology payloads.

    Shared variables/connectivity may refer only to earlier compatible zones.
    Ordered, face-based, bit-packed and face-neighbour layouts are rejected.
    Source min/max records are metadata, not a full validation of field values.
    """
    path = Path(path)
    with path.open("rb") as stream:
        if stream.read(8) != b"#!TDV112":
            raise ValueError("Expected TDV112 magic")
        order = stream.read(4)
        if order not in (b"\1\0\0\0", b"\0\0\0\1"):
            raise ValueError("Invalid TDV112 byte-order marker")
        reader = _Reader(stream, path.stat().st_size, "<" if order[0] == 1 else ">")
        if reader.read("i") != 0:
            raise ValueError("Native TDV112 requires a full geometry-and-field file")
        reader.string()  # title
        nvars = reader.read("i")
        if not 0 < nvars <= 10000:
            raise ValueError("Invalid TDV112 variable count")
        names = [reader.string() for _ in range(nvars)]
        if len(set(names)) != nvars or not all(n in names for n in ("X", "Y", "Z")):
            raise ValueError("Need unique variables including X, Y and Z")
        headers = []
        zone_names = set()
        types = {1: (2, 3), 2: (3, 5), 3: (4, 9), 4: (4, 10), 5: (8, 12)}
        while True:
            marker = reader.read("f")
            if marker == 357:
                break
            if marker in (799, 899):
                if marker == 899:
                    variable = reader.read("i")
                    if not 0 <= variable < nvars:
                        raise ValueError("Invalid auxiliary variable index")
                reader.auxiliary()
                continue
            if marker != 299:
                raise ValueError(f"Unsupported TDV112 header marker {marker}")
            name = reader.string()
            if not name or name in zone_names:
                raise ValueError("Need unique nonempty TDV112 zone names")
            zone_names.add(name)
            reader.read("i")  # unused parent zone
            reader.read("i")  # strand
            solution_time = reader.read("d")
            reader.read("i")  # colour
            kind = reader.read("i")
            if kind not in types:
                raise ValueError(
                    f"Unsupported TDV112 zone type {kind}; expected linear FE zones"
                )
            locations = [reader.flag() for _ in names] if reader.flag() else [0] * nvars
            if any(locations[names.index(n)] for n in ("X", "Y", "Z")):
                raise ValueError("Coordinates must be nodal")
            raw, neighbours = reader.flag(), reader.read("i")
            if raw or neighbours:
                raise ValueError("TDV112 face-neighbour records are unsupported")
            points, cells = reader.read("i"), reader.read("i")
            if not 0 < points <= reader.length or not 0 < cells <= reader.length:
                raise ValueError("Invalid TDV112 mesh counts")
            for _ in range(3):
                reader.read("i")  # reserved cell dimensions
            while reader.flag():
                reader.auxiliary()
            headers.append(
                dict(
                    name=name,
                    kind=kind,
                    points=points,
                    cells=cells,
                    locations=locations,
                    solution_time=solution_time,
                )
            )
        if not headers:
            raise ValueError("TDV112 file has no zones")
        result = {}
        previous: list[dict[str, Any]] = []
        formats = {1: "f4", 2: "f8", 3: "i4", 4: "i2", 5: "u1"}
        for index, zone in enumerate(headers):
            if reader.read("f") != 299:
                raise ValueError("Expected TDV112 zone data marker")
            codes = [reader.read("i") for _ in names]
            if any(code not in formats for code in codes):
                raise ValueError(
                    "Unsupported TDV112 variable type (including bit-packed data)"
                )
            passive = [reader.flag() for _ in names] if reader.flag() else [0] * nvars
            shared = (
                [reader.read("i") for _ in names] if reader.flag() else [-1] * nvars
            )
            shared_cells = reader.read("i")
            if any(ref < -1 or ref >= index for ref in [*shared, shared_cells]):
                raise ValueError("TDV112 sharing must reference an earlier zone")
            owned = [i for i in range(nvars) if not passive[i] and shared[i] == -1]
            extrema = {names[i]: (reader.read("d"), reader.read("d")) for i in owned}
            arrays: dict[str, ArrayReference | _PassiveArray] = {}
            for i, name in enumerate(names):
                count = zone["cells"] if zone["locations"][i] else zone["points"]
                dtype = np.dtype(reader.endian + formats[codes[i]])
                if passive[i]:
                    if shared[i] != -1 or name in ("X", "Y", "Z"):
                        raise ValueError(
                            "Invalid passive coordinate or shared/passive variable"
                        )
                    arrays[name] = _PassiveArray(count, dtype)
                elif shared[i] != -1:
                    source = previous[shared[i]]
                    array = source["fields"][name]
                    if (
                        array.shape != (count,)
                        or array.dtype != dtype
                        or source["field_associations"][name]
                        != ("cell" if zone["locations"][i] else "point")
                    ):
                        raise ValueError("Incompatible shared TDV112 variable")
                    arrays[name] = array
                else:
                    arrays[name] = ArrayReference(path, dtype, stream.tell(), (count,))
                    reader.skip(count * dtype.itemsize)
            nodes, vtk_type = types[zone["kind"]]
            if shared_cells != -1:
                source = previous[shared_cells]
                if (
                    source["vtk_cell_type"] != vtk_type
                    or source["connectivity"].shape != (zone["cells"], nodes)
                    or headers[shared_cells]["points"] != zone["points"]
                ):
                    raise ValueError("Incompatible shared TDV112 connectivity")
                connectivity = source["connectivity"]
            else:
                connectivity = ArrayReference(
                    path, reader.endian + "i4", stream.tell(), (zone["cells"], nodes)
                )
                reader.skip(zone["cells"] * nodes * 4)
            entry = dict(
                fields=arrays,
                connectivity=connectivity,
                vtk_cell_type=vtk_type,
                index_base=0,
                format="TDV112",
                solution_time=zone["solution_time"],
                extrema=extrema,
                field_associations={
                    n: "cell" if zone["locations"][i] else "point"
                    for i, n in enumerate(names)
                },
            )
            result[zone["name"]] = entry
            previous.append(entry)
        if stream.tell() != reader.length:
            raise ValueError("Unexpected trailing TDV112 records")
        return result
