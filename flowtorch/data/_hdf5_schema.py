"""Versioned multi-zone storage used by the public HDF5 reader and writer."""

from pathlib import Path
import hashlib
from typing import Any
import json
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import torch

from .unstructured import UnstructuredGeometry
from ._hdf5_compression import (
    compression_settings,
    record_compression,
    prepare_read,
    prepare_dataset_read,
    dataset_compression,
)

SCHEMA = "flowtorch-unstructured"
VERSION = 1


def select_names(available, requested, what):
    """Validate an exact, nonempty, duplicate-free ordered selection."""
    names = (
        list(available)
        if requested is None
        else ([requested] if isinstance(requested, str) else list(requested))
    )
    if (
        not names
        or len(set(names)) != len(names)
        or any(n not in available for n in names)
    ):
        raise ValueError(
            f"Invalid {what} selection {names}; available: {list(available)}"
        )
    return names


def geometry_identity(groups):
    digest = hashlib.sha256()
    for name, signature in groups:
        digest.update(json.dumps([name, signature], ensure_ascii=True).encode())
    return digest.hexdigest()


def read_geometry(group):
    for name in ("points", "topology", "locations", "weights"):
        prepare_dataset_read(group[name])
    return UnstructuredGeometry(
        group["points"][:],
        group["topology"][:],
        group.attrs["topology_type"],
        int(group.attrs["n_cells"]),
        group.attrs["association"],
        group["locations"][:],
        group["weights"][:],
        int(group.attrs["dimension"]),
    )


def validate_geometry_file(file, *, contents=True):
    prepare_read(file)
    if (
        file.attrs.get("schema") != SCHEMA
        or file.attrs.get("schema_version") != VERSION
        or not file.attrs.get("complete", False)
    ):
        raise ValueError("Unsupported or incomplete geometry file")
    if not contents:
        return
    signatures = []
    for group in file["zones"].values():
        signature = read_geometry(group).signature
        if signature != group.attrs["signature"]:
            raise ValueError("Geometry content does not match its recorded identity")
        signatures.append((group.attrs["name"], signature))
    if geometry_identity(signatures) != file.attrs["geometry_id"]:
        raise ValueError("Geometry identity mismatch")


def write_geometry(
    path,
    geometries,
    *,
    compression=None,
    compression_level=None,
    shuffle=False,
    spatial_chunk=65536,
):
    """Consume ordered geometry zones one at a time; retain only names/hashes.

    An iterator can release a zone before producing the next. Caller-owned
    lists still own their arrays. Existing geometry is validated before the
    iterator is consumed and is never modified. New files remain incomplete
    if generation, validation or writing fails.
    """
    settings = compression_settings(compression, compression_level, shuffle)
    if spatial_chunk < 1:
        raise ValueError("spatial_chunk must be positive")
    path = Path(path)
    reuse = path.exists()
    if not reuse:
        path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "r" if reuse else "x", track_order=True) as file:
        if reuse:
            validate_geometry_file(file)
            zones = file["zones"]
            expected_keys = list(zones)
        else:
            file.attrs.update(schema=SCHEMA, schema_version=VERSION, complete=False)
            record_compression(file, geometry=settings)
            zones = file.create_group("zones", track_order=True)
        signatures: list[tuple[str, str]] = []
        names = set()
        association = None
        iterator = iter(geometries)
        try:
            while True:
                try:
                    name, geom = next(iterator)
                except StopIteration:
                    break
                try:
                    if not name or name in names:
                        raise ValueError("Need unique nonempty geometry zones")
                    names.add(name)
                    if association is None:
                        association = geom.association
                    elif geom.association != association:
                        raise ValueError("Selected zones must share field association")
                    signature = geom.signature
                    index = len(signatures)
                    if reuse:
                        if (
                            index >= len(expected_keys)
                            or zones[expected_keys[index]].attrs["name"] != name
                            or zones[expected_keys[index]].attrs["signature"]
                            != signature
                        ):
                            raise ValueError(
                                "Existing geometry differs in coordinates, topology, weights or zone order"
                            )
                    else:
                        group = zones.create_group(f"z{index:06d}")
                        group.attrs.update(
                            name=name,
                            signature=signature,
                            topology_type=geom.topology_type,
                            n_cells=geom.n_cells,
                            association=geom.association,
                            dimension=geom.dimension,
                        )
                        for key in ("points", "topology", "locations", "weights"):
                            dtype = np.int64 if key == "topology" else np.float64
                            values = getattr(geom, key)
                            options = settings.dataset_options
                            if settings.enabled:
                                options["chunks"] = (
                                    min(len(values), spatial_chunk),
                                    *values.shape[1:],
                                )
                            ds = group.create_dataset(
                                key, shape=values.shape, dtype=dtype, **options
                            )
                            for start in range(0, len(values), spatial_chunk):
                                ds[start : start + spatial_chunk] = values[
                                    start : start + spatial_chunk
                                ]
                            del values
                    signatures.append((name, signature))
                finally:
                    # A for-loop would retain geom while next(iterator) builds
                    # the following zone. Drop this reference before advancing.
                    del geom
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        if not signatures:
            raise ValueError("Need unique nonempty geometry zones")
        identity = geometry_identity(signatures)
        if reuse:
            if identity != file.attrs["geometry_id"]:
                raise ValueError(
                    "Existing geometry differs in coordinates, topology, weights or zone order"
                )
        else:
            file.attrs["geometry_id"] = identity
            file.attrs["complete"] = True
        return identity


def committed_frames(file):
    """Return only published frames, ignoring interrupted append transactions."""
    frames = file["frames"]
    keys = (
        json.loads(file.attrs["committed_frames"])
        if "committed_frames" in file.attrs
        else list(frames)
    )
    if (
        not isinstance(keys, list)
        or len(set(keys)) != len(keys)
        or any(key not in frames for key in keys)
    ):
        raise ValueError("Invalid committed frame index")
    return {key: frames[key] for key in keys}


def inspect_append(
    path,
    dtype,
    geometry_file=None,
    *,
    compression=None,
    compression_level=None,
    shuffle=None,
):
    """Read-only metadata preflight; geometry content is checked by the writer."""
    reader = SeriesReader(
        path, dtype, geometry_file=geometry_file, _validate_geometry=False
    )
    try:
        expected_dtype = np.dtype("float32" if dtype == torch.float32 else "float64")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Expected float32 or float64 storage precision")
        settings = None
        schema = {}
        for time in reader.frames:
            for name, key in zip(reader.names, reader.keys):
                datasets = reader.datasets[time, key]
                for ds in datasets.values():
                    if ds.dtype != expected_dtype:
                        raise ValueError(
                            "Append storage precision must match existing fields"
                        )
                    current = dataset_compression(ds)
                    if settings is not None and current != settings:
                        raise ValueError(
                            "Existing field compression differs between datasets"
                        )
                    settings = current
                schema[name] = {
                    field: tuple(ds.shape) for field, ds in datasets.items()
                }
        assert settings is not None
        assert reader.geometry is not None
        requested = compression_settings(
            settings.codec if compression is None else compression,
            settings.level if compression_level is None else compression_level,
            settings.shuffle if shuffle is None else shuffle,
        )
        if requested != settings:
            raise ValueError("Append compression must match existing field compression")
        return dict(
            geometry_path=Path(reader.geometry.filename).resolve(),
            zone_names=list(reader.names),
            schema=schema,
            compression=settings,
            times={float(time) for time in reader.frames},
            frame_keys=list(committed_frames(reader.file)),
        )
    finally:
        reader.close()


def validate_append_selection(info, times, schemas):
    """Reject duplicate times and incompatible zone/field selections before writes."""
    if any(float(time) in info["times"] for time in times):
        raise ValueError("Append contains an already existing numeric timestamp")
    if list(schemas) != info["zone_names"] or schemas != info["schema"]:
        raise ValueError(
            "Append zone order, fields, components or shapes differ from existing data"
        )


class FieldFile:
    """Owns one field file and a read-only shared geometry handle."""

    def __init__(
        self,
        path,
        geometry_file,
        dtype=torch.float32,
        chunk_size=65536,
        compression=None,
        overwrite=False,
        *,
        compression_level=None,
        shuffle=None,
        append=False,
    ):
        if append and overwrite:
            raise ValueError("append and overwrite are mutually exclusive")
        if dtype not in (torch.float32, torch.float64) or chunk_size < 1:
            raise ValueError("Expected floating field dtype and positive chunks")
        self.path = Path(path).resolve()
        self.append = append
        self._finished = False
        self._pending_name = None
        self._moved_keys = []
        self.reference: dict[str, Any] = (
            inspect_append(
                self.path,
                dtype,
                geometry_file,
                compression=compression,
                compression_level=compression_level,
                shuffle=shuffle,
            )
            if append
            else {}
        )
        settings = (
            self.reference["compression"]
            if append
            else compression_settings(
                compression, compression_level, False if shuffle is None else shuffle
            )
        )
        self.geometry_path = (
            self.reference["geometry_path"] if append else Path(geometry_file).resolve()
        )
        if self.path == self.geometry_path:
            raise ValueError("Field output cannot replace shared geometry")
        self.geometry = h5py.File(self.geometry_path, "r")
        try:
            validate_geometry_file(self.geometry)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = h5py.File(
                self.path,
                "r+" if append else ("w" if overwrite else "x"),
                track_order=True,
            )
            if append:
                # A previous process may have stopped before publishing its batch.
                active = committed_frames(self.file)
                if list(active) != self.reference["frame_keys"]:
                    raise ValueError(
                        "Existing frames changed during append preflight; retry"
                    )
                for key in list(self.file["frames"]):
                    if key not in active:
                        del self.file["frames"][key]
                for key in list(self.file):
                    if key.startswith("_pending_append_"):
                        del self.file[key]
                import uuid

                self._pending_name = "_pending_append_" + uuid.uuid4().hex
                self.frames = self.file.create_group(
                    self._pending_name, track_order=True
                )
            else:
                import os

                self.file.attrs.update(
                    schema=SCHEMA,
                    schema_version=VERSION,
                    complete=False,
                    geometry_file=os.path.relpath(self.geometry_path, self.path.parent),
                    geometry_id=self.geometry.attrs["geometry_id"],
                )
                record_compression(self.file, fields=settings)
                self.frames = self.file.create_group("frames", track_order=True)
        except Exception:
            if hasattr(self, "file"):
                self.file.close()
            self.geometry.close()
            raise
        self.dtype = np.dtype("float32" if dtype == torch.float32 else "float64")
        self.chunk_size = chunk_size
        self.compression = settings
        self.coverage = {}

    @property
    def zones(self):
        return {g.attrs["name"]: key for key, g in self.geometry["zones"].items()}

    def allocate(self, name, component_shape, time, zone, metadata=None):
        if self._finished:
            raise ValueError("Cannot modify finalized output")
        if zone not in self.zones or not name:
            raise ValueError("Unknown zone or empty field name")
        time = str(time)
        if not np.isfinite(float(time)):
            raise ValueError("Time must be finite")
        if self.append:
            if float(time) in self.reference["times"]:
                raise ValueError(
                    "Append contains an already existing numeric timestamp"
                )
            expected = self.reference["schema"][zone]
            if name not in expected or tuple(component_shape) != expected[name][1:]:
                raise ValueError(
                    "Append field names or components differ from existing data"
                )
        matches = [
            key for key, group in self.frames.items() if group.attrs["label"] == time
        ]
        if matches:
            frame = self.frames[matches[0]]
        else:
            if any(
                float(g.attrs["label"]) == float(time) for g in self.frames.values()
            ):
                raise ValueError("Duplicate numeric time")
            frame = self.frames.create_group(
                f"t{len(self.frames):06d}", track_order=True
            )
            frame.attrs["label"] = time
        key = self.zones[zone]
        fields = frame.require_group(key)
        if any(ds.attrs["name"] == name for ds in fields.values()):
            raise ValueError("Field already allocated")
        count = len(self.geometry["zones"][key]["weights"])
        shape = (count, *tuple(component_shape))
        if any(int(n) != n or n < 1 for n in shape):
            raise ValueError("Field dimensions must be positive integers")
        ds = fields.create_dataset(
            f"f{len(fields):06d}",
            shape=shape,
            dtype=self.dtype,
            chunks=(min(count, self.chunk_size), *shape[1:]),
            **self.compression.dataset_options,
        )
        ds.attrs["name"] = name
        if metadata is not None:
            ds.attrs["metadata"] = json.dumps(metadata)
        self.coverage[ds.name] = 0
        return ds

    def dataset(self, name, time, zone):
        for frame in self.frames.values():
            if frame.attrs["label"] == str(time):
                for ds in frame[self.zones[zone]].values():
                    if ds.attrs["name"] == name:
                        return ds
        raise KeyError((name, time, zone))

    def write_slice(self, name, time, zone, part, values):
        if self._finished:
            raise ValueError("Cannot modify finalized output")
        ds = self.dataset(name, time, zone)
        start, stop, step = part.indices(ds.shape[0])
        if step != 1 or start != self.coverage[ds.name] or stop <= start:
            raise ValueError(
                "Write contiguous, nonoverlapping chunks in increasing order"
            )
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        values = np.asarray(values)
        if (
            values.shape != (stop - start, *ds.shape[1:])
            or np.iscomplexobj(values)
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "Field chunk has invalid shape or nonfinite/complex values"
            )
        with np.errstate(over="ignore"):
            values = values.astype(ds.dtype, copy=False)
        if not np.isfinite(values).all():
            raise ValueError("Field values overflow requested storage precision")
        ds[start:stop] = values
        self.coverage[ds.name] = stop

    def finish(self):
        if self._finished:
            return
        if not self.coverage or any(
            self.file[name].shape[0] != count for name, count in self.coverage.items()
        ):
            raise ValueError("Cannot complete output with unwritten fields")
        expected = (
            {
                self.zones[name]: fields
                for name, fields in self.reference["schema"].items()
            }
            if self.append
            else None
        )
        for frame in self.frames.values():
            if set(frame) != set(self.geometry["zones"]):
                raise ValueError("Every snapshot must contain all geometry zones")
            schema = {
                zone: {ds.attrs["name"]: ds.shape for ds in group.values()}
                for zone, group in frame.items()
            }
            components = [
                {name: shape[1:] for name, shape in zone.items()}
                for zone in schema.values()
            ]
            if any(value != components[0] for value in components[1:]):
                raise ValueError("Field components differ across geometry zones")
            if any(not group for group in schema.values()):
                raise ValueError("Empty field zone")
            if expected is not None and schema != expected:
                raise ValueError("Field schema changes between snapshots")
            expected = schema
        if self.append:
            old_keys = self.reference["frame_keys"]
            # Publish an explicit old index before moving staged, fully valid frames.
            self.file.attrs["committed_frames"] = json.dumps(old_keys)
            self.file.flush()
            for key in list(self.frames):
                index = max((int(k[1:]) for k in self.file["frames"]), default=-1) + 1
                new_key = f"t{index:06d}"
                self.file.move(self.frames[key].name, "/frames/" + new_key)
                self._moved_keys.append(new_key)
            self.file.flush()
            # An explicit override may relocate identical read-only geometry.
            import os

            self.file.attrs["geometry_file"] = os.path.relpath(
                self.geometry_path, self.path.parent
            )
            self.file.attrs["committed_frames"] = json.dumps(
                old_keys + self._moved_keys
            )
            self.file.flush()
            self._finished = True
            del self.file[self._pending_name]
            self.frames = committed_frames(self.file)
        else:
            self.file.attrs["complete"] = True
            self.file.flush()
            self._finished = True

    def write_xdmf(self, path):
        import os

        if self.file.mode == "r":
            if not self.file.attrs.get("complete", False):
                raise ValueError("Cannot publish incomplete output")
        else:
            self.finish()
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        root = ET.Element("Xdmf", Version="3.0")
        temporal = ET.SubElement(
            ET.SubElement(root, "Domain"),
            "Grid",
            Name="flowTorch",
            GridType="Collection",
            CollectionType="Temporal",
        )

        def item(parent, ds, source):
            node = ET.SubElement(
                parent,
                "DataItem",
                Dimensions=" ".join(map(str, ds.shape)),
                NumberType="Int" if ds.dtype.kind in "iu" else "Float",
                Precision=str(ds.dtype.itemsize),
                Format="HDF",
            )
            node.text = f"{os.path.relpath(source, path.parent)}:{ds.name}"

        for frame in sorted(
            self.frames.values(), key=lambda g: float(g.attrs["label"])
        ):
            spatial = ET.SubElement(
                temporal, "Grid", GridType="Collection", CollectionType="Spatial"
            )
            ET.SubElement(spatial, "Time", Value=frame.attrs["label"])
            for key, geom in self.geometry["zones"].items():
                grid = ET.SubElement(
                    spatial, "Grid", Name=geom.attrs["name"], GridType="Uniform"
                )
                top = ET.SubElement(
                    grid,
                    "Topology",
                    TopologyType=geom.attrs["topology_type"],
                    NumberOfElements=str(geom.attrs["n_cells"]),
                )
                if geom.attrs["topology_type"] == "Polyline":
                    top.set("NodesPerElement", "2")
                item(top, geom["topology"], self.geometry_path)
                item(
                    ET.SubElement(grid, "Geometry", GeometryType="XYZ"),
                    geom["points"],
                    self.geometry_path,
                )
                for ds in frame[key].values():
                    components = int(np.prod(ds.shape[1:]))
                    kind = {
                        1: "Scalar",
                        2: "Vector",
                        3: "Vector",
                        6: "Tensor6",
                        9: "Tensor",
                    }.get(components)
                    if kind is None:
                        raise ValueError(
                            f"Unsupported XDMF component count: {components}"
                        )
                    attr = ET.SubElement(
                        grid,
                        "Attribute",
                        Name=ds.attrs["name"],
                        AttributeType=kind,
                        Center=(
                            "Node" if geom.attrs["association"] == "point" else "Cell"
                        ),
                    )
                    item(attr, ds, self.path)
        temporary = path.with_name(path.name + ".tmp")
        ET.ElementTree(root).write(temporary, encoding="utf-8", xml_declaration=True)
        temporary.replace(path)

    def close(self):
        try:
            if self.append and not self._finished and self.file.id.valid:
                active = committed_frames(self.file)
                for key in self._moved_keys:
                    if key not in active and key in self.file["frames"]:
                        del self.file["frames"][key]
                if self._pending_name in self.file:
                    del self.file[self._pending_name]
                self.file.flush()
        finally:
            self.file.close()
            self.geometry.close()


class SeriesReader:
    """Reads only requested field hyperslabs; geometry remains a separate file."""

    def __init__(
        self, path, dtype, zones=None, geometry_file=None, *, _validate_geometry=True
    ):
        self.path = Path(path).resolve()
        self.dtype = dtype
        self.file = h5py.File(self.path, "r")
        self.geometry: Any = None
        try:
            prepare_read(self.file)
            if (
                self.file.attrs.get("schema") != SCHEMA
                or self.file.attrs.get("schema_version") != VERSION
                or not self.file.attrs.get("complete", False)
            ):
                raise ValueError("Unsupported or incomplete field file")
            self.geometry = h5py.File(
                geometry_file or self.path.parent / self.file.attrs["geometry_file"],
                "r",
            )
            validate_geometry_file(self.geometry, contents=_validate_geometry)
            if self.file.attrs["geometry_id"] != self.geometry.attrs["geometry_id"]:
                raise ValueError("Fields reference a different geometry")
            available = {
                g.attrs["name"]: key for key, g in self.geometry["zones"].items()
            }
            self.names = select_names(available, zones, "zone")
            self.keys = [available[name] for name in self.names]
            self.association = self.geometry["zones"][self.keys[0]].attrs["association"]
            self.geometry_id = self.geometry.attrs["geometry_id"]
            self.offsets = np.cumsum(
                [0] + [len(self.geometry["zones"][key]["weights"]) for key in self.keys]
            )
            self.frames = {
                g.attrs["label"]: g
                for g in sorted(
                    committed_frames(self.file).values(),
                    key=lambda g: float(g.attrs["label"]),
                )
            }
            if not self.frames:
                raise ValueError("No field snapshots")
            self.datasets = {}
            reference = None
            for time, frame in self.frames.items():
                schema: dict[str, tuple[int, ...]] = {}
                for key in self.keys:
                    datasets = {ds.attrs["name"]: ds for ds in frame[key].values()}
                    for ds in datasets.values():
                        prepare_dataset_read(ds)
                    self.datasets[time, key] = datasets
                    current = {name: ds.shape[1:] for name, ds in datasets.items()}
                    if schema and current != schema:
                        raise ValueError("Field components differ between zones")
                    schema = current
                    count = len(self.geometry["zones"][key]["weights"])
                    if any(ds.shape[0] != count for ds in datasets.values()):
                        raise ValueError("Field association/geometry mismatch")
                if not schema or reference is not None and schema != reference:
                    raise ValueError("Field schema changes across snapshots")
                reference = schema
        except Exception:
            self.close()
            raise

    def shape(self, name, time):
        ds = self.datasets[time, self.keys[0]][name]
        return (int(self.offsets[-1]), *ds.shape[1:])

    def read(self, name, time, part):
        shape = self.shape(name, time)
        start, stop, step = part.indices(shape[0])
        if step != 1:
            raise ValueError("Spatial slices require unit stride")
        result = torch.empty((max(0, stop - start), *shape[1:]), dtype=self.dtype)
        for i, key in enumerate(self.keys):
            lo, hi = max(start, self.offsets[i]), min(stop, self.offsets[i + 1])
            if hi > lo:
                ds = self.datasets[time, key][name]
                result[lo - start : hi - start] = torch.as_tensor(
                    ds[lo - self.offsets[i] : hi - self.offsets[i]], dtype=self.dtype
                )
        return result

    def geometry_values(self, name):
        values = [self.geometry["zones"][key][name][:] for key in self.keys]
        return np.concatenate(values, axis=0)

    def close(self):
        self.file.close()
        if self.geometry is not None:
            self.geometry.close()
