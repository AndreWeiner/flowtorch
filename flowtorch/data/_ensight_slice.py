"""Direct dense C-binary field slices; VTK owns geometry and mesh operations."""

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, NoReturn

import numpy as np

# EnSight name -> VTK cell type, fixed connectivity width.
_ELEMENTS = {
    "tria3": (5, 3),
    "quad4": (9, 4),
    "tetra4": (10, 4),
    "hexa8": (12, 8),
    "penta6": (13, 6),
    "pyramid5": (14, 5),
    "nsided": (7, None),
    "nfaced": (42, None),
}
COMPONENTS = {"scalar": 1, "vector": 3, "tensor symm": 6, "tensor asym": 9}


@dataclass
class _Part:
    name: str
    points: int
    blocks: list[tuple[str, int]]


class _Binary:
    def __init__(self, stream: BinaryIO, path: Path, endian: str):
        self.stream = stream
        self.path = path
        self.endian = endian
        self.size = path.stat().st_size

    def fail(self, message: str) -> NoReturn:
        raise ValueError(
            f"Native EnSight slice reader: {message}: {self.path}; "
            'use slice_backend="vtk" for unsupported formats'
        )

    def read(self, size: int) -> bytes:
        data = self.stream.read(size)
        if len(data) != size:
            self.fail("truncated binary record")
        return data

    def record(self) -> str:
        try:
            return self.read(80).rstrip(b"\x00 ").decode("ascii").strip()
        except UnicodeDecodeError:
            self.fail("invalid or unsupported binary record")

    def integer(self) -> int:
        value = int(np.frombuffer(self.read(4), dtype=self.endian + "i4")[0])
        if value < 0:
            self.fail("negative count or part ID")
        return value

    def skip(self, size: int):
        if size < 0 or self.stream.tell() + size > self.size:
            self.fail("payload extends beyond file")
        self.stream.seek(size, 1)

    def sum_counts(self, count: int) -> int:
        # Ragged connectivity counts are scanned in bounded buffers, never
        # materialized with coordinates or field payloads.
        total = 0
        while count:
            length = min(count, 16384)
            values = np.frombuffer(self.read(4 * length), dtype=self.endian + "i4")
            if np.any(values < 0):
                self.fail("negative connectivity count")
            total += int(values.sum(dtype=np.int64))
            count -= length
        return total


class EnSightSliceReader:
    """Index metadata omitted by VTK's public Python API.

    Only static unstructured C-binary geometry and dense fields are supported.
    Geometry payloads are skipped; the caller checks counts/order against VTK.
    Field indexes are cached by file identity in a bounded LRU cache.
    """

    def __init__(self, geometry: Path, byte_order: str):
        self.endian = "<" if byte_order == "LittleEndian" else ">"
        self.parts: dict[int, _Part] = {}
        self._cache: OrderedDict = OrderedDict()
        with geometry.open("rb") as stream:
            reader = _Binary(stream, geometry, self.endian)
            if reader.record().lower() != "c binary":
                reader.fail("only C-binary geometry is supported")
            reader.record()
            reader.record()
            node_ids = reader.record()
            element_ids = reader.record()
            if node_ids not in {
                f"node id {s}" for s in ("off", "assign", "given", "ignore")
            }:
                reader.fail("unsupported node IDs")
            if element_ids not in {
                f"element id {s}" for s in ("off", "assign", "given", "ignore")
            }:
                reader.fail("unsupported element IDs")
            listed_nodes = node_ids.split()[-1] in ("given", "ignore")
            listed_elements = element_ids.split()[-1] in ("given", "ignore")
            part = None
            while stream.tell() < reader.size:
                record = reader.record()
                if record == "extents" and part is None:
                    reader.skip(24)
                elif record == "part":
                    part_id = reader.integer()
                    if part_id in self.parts:
                        reader.fail("duplicate part ID")
                    name = reader.record()
                    if reader.record() != "coordinates":
                        reader.fail("only unstructured coordinates are supported")
                    points = reader.integer()
                    reader.skip(4 * points * (3 + int(listed_nodes)))
                    part = _Part(name, points, [])
                    self.parts[part_id] = part
                elif part is not None and record in _ELEMENTS:
                    count = reader.integer()
                    if listed_elements:
                        reader.skip(4 * count)
                    width = _ELEMENTS[record][1]
                    if record == "nfaced":
                        faces = reader.sum_counts(count)
                        nodes = reader.sum_counts(faces)
                    elif record == "nsided":
                        nodes = reader.sum_counts(count)
                    else:
                        assert width is not None
                        nodes = count * width
                    reader.skip(4 * nodes)
                    if any(name == record for name, _ in part.blocks):
                        reader.fail("repeated element blocks are unsupported")
                    part.blocks.append((record, count))
                else:
                    reader.fail(f"unsupported geometry section {record!r}")

    def validate(self, grids: dict, cell_types) -> None:
        if set(grids) != {part.name for part in self.parts.values()}:
            raise ValueError("Native EnSight part metadata does not match VTK")
        for part in self.parts.values():
            grid = grids[part.name]
            types = cell_types(grid)
            start = 0
            for element, count in part.blocks:
                if not np.all(types[start : start + count] == _ELEMENTS[element][0]):
                    raise ValueError(
                        "Native EnSight element ordering does not match VTK"
                    )
                start += count
            if start != len(types) or part.points != grid.GetNumberOfPoints():
                raise ValueError("Native EnSight geometry counts do not match VTK")

    def index(self, path: Path, components: int, association: str):
        stat = path.stat()
        key = (path, stat.st_size, stat.st_mtime_ns, components, association)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        result = {}
        with path.open("rb") as stream:
            reader = _Binary(stream, path, self.endian)
            reader.record()  # Description, not a C Binary signature.
            while stream.tell() < reader.size:
                if reader.record() != "part":
                    reader.fail("expected part record")
                part_id = reader.integer()
                if part_id not in self.parts or part_id in result:
                    reader.fail("unknown or duplicate field part ID")
                part = self.parts[part_id]
                expected = (
                    [("coordinates", part.points)]
                    if association == "point"
                    else part.blocks
                )
                blocks = []
                # Field blocks may be reordered relative to geometry. Store
                # their destination offsets, rather than concatenating blindly.
                offsets = {}
                start = 0
                for element, count in expected:
                    offsets[element] = (start, count)
                    start += count
                seen = set()
                for _ in expected:
                    element = reader.record()
                    if element not in offsets or element in seen:
                        reader.fail(f"unsupported field section {element!r}")
                    seen.add(element)
                    start, count = offsets[element]
                    blocks.append((start, count, stream.tell()))
                    reader.skip(4 * count * components)
                result[part.name] = blocks
        self._cache[key] = result
        if len(self._cache) > 32:
            self._cache.popitem(last=False)
        return result

    def read(
        self, path: Path, part: str, kind: str, association: str, start: int, stop: int
    ):
        components = COMPONENTS[kind]
        index = self.index(path, components, association)
        if part not in index:
            raise ValueError(f"Field is absent from part {part!r}: {path}")
        values = np.empty((stop - start, components), dtype=np.float32)
        with path.open("rb") as stream:
            reader = _Binary(stream, path, self.endian)
            for base, count, offset in index[part]:
                first, last = max(start, base), min(stop, base + count)
                if first >= last:
                    continue
                for component in range(components):
                    stream.seek(offset + 4 * (component * count + first - base))
                    data = reader.read(4 * (last - first))
                    values[first - start : last - start, component] = np.frombuffer(
                        data, dtype=self.endian + "f4"
                    )
        if kind == "tensor symm":
            values = values[:, [0, 3, 4, 1, 5, 2]]
        return values[:, 0] if components == 1 else values
