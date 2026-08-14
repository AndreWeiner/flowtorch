"""ParaView-side worker for :mod:`flowtorch.data.tecplot_dataloader`.

This file is executed by pvpython, not imported by the normal flowTorch
interpreter. Requests arrive as newline-delimited JSON on stdin. Responses use
a dedicated binary file descriptor so ParaView console output cannot corrupt
the protocol.
"""

from __future__ import annotations

import json
import os
import struct
import sys

_HEADER_SIZE = 8


def _write_header(stream, header):
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    stream.write(struct.pack("!Q", len(encoded)))
    stream.write(encoded)


def _send_json(stream, value):
    _write_header(stream, {"status": "ok", "kind": "json", "value": value})
    stream.flush()


def _send_array(stream, value, numpy):
    array = numpy.ascontiguousarray(value)
    _write_header(
        stream,
        {
            "status": "ok",
            "kind": "array",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "nbytes": array.nbytes,
        },
    )
    stream.write(memoryview(array).cast("B"))
    stream.flush()


def _send_error(stream, error):
    _write_header(
        stream,
        {
            "status": "error",
            "error_type": type(error).__name__,
            "message": str(error),
        },
    )
    stream.flush()


def _create_reader(request, reader_type):
    return reader_type(
        registrationName=request["registration_name"],
        FileName=[request["file_path"]],
    )


def _parse_block_name(meta_data):
    """Parse a composite name from printable metadata as a fallback."""
    for line in meta_data.split("\n"):
        if "NAME" in line:
            return line.split(":")[-1].strip()
    return None


def _child_api(data):
    """Return the index/count methods for a supported composite dataset."""
    candidates = (
        ("GetNumberOfBlocks", "GetBlock"),
        ("GetNumberOfPartitionedDataSets", "GetPartitionedDataSet"),
        ("GetNumberOfPartitions", "GetPartition"),
    )
    for count_name, child_name in candidates:
        if hasattr(data, count_name) and hasattr(data, child_name):
            return getattr(data, count_name), getattr(data, child_name)
    if (
        hasattr(data, "IsA")
        and data.IsA("vtkMultiPieceDataSet")
        and hasattr(data, "GetNumberOfPieces")
        and hasattr(data, "GetPiece")
    ):
        return data.GetNumberOfPieces, data.GetPiece
    return None


def _metadata_name(parent, index, name_key=None):
    """Read a child name from VTK metadata, with text parsing as fallback."""
    if not hasattr(parent, "GetMetaData"):
        return None
    try:
        metadata = parent.GetMetaData(index)
    except Exception:
        return None
    if metadata is None:
        return None
    if name_key is not None:
        try:
            if metadata.Has(name_key):
                value = metadata.Get(name_key)
                if value:
                    return str(value)
        except Exception:
            pass
    return _parse_block_name(str(metadata))


def _leaf_records(root, name_key=None):
    """Return all non-composite descendants with stable names and paths."""
    leaves = []

    def visit(data, index_path, name_path):
        api = _child_api(data)
        if api is None:
            full_name = "/".join(name_path) if name_path else "root"
            leaves.append(
                {
                    "leaf_name": name_path[-1] if name_path else "root",
                    "full_name": full_name,
                    "path": list(index_path),
                }
            )
            return
        count, child_at = api
        for index in range(count()):
            child = child_at(index)
            if child is None:
                continue
            name = _metadata_name(data, index, name_key)
            if not name:
                name = "root" if not index_path and index == 0 else f"block_{index}"
            visit(child, index_path + [index], name_path + [name])

    visit(root, [], [])
    leaf_counts = {}
    for leaf in leaves:
        name = leaf["leaf_name"]
        leaf_counts[name] = leaf_counts.get(name, 0) + 1
    used_names = set()
    records = []
    for leaf in leaves:
        public_name = (
            leaf["leaf_name"]
            if leaf_counts[leaf["leaf_name"]] == 1
            else leaf["full_name"]
        )
        if public_name in used_names:
            indices = "/".join(str(index) for index in leaf["path"])
            public_name = f"{public_name} [{indices}]"
        used_names.add(public_name)
        records.append(
            {
                "name": public_name,
                "full_name": leaf["full_name"],
                "path": leaf["path"],
            }
        )
    return records


def _resolve_leaf(root, block_path):
    """Resolve and validate a leaf from a sequence of composite indices."""
    if not isinstance(block_path, list) or any(
        not isinstance(index, int) or index < 0 for index in block_path
    ):
        raise ValueError(f"Invalid block path: {block_path!r}")
    current = root
    traversed = []
    for index in block_path:
        api = _child_api(current)
        if api is None:
            raise ValueError(
                f"Block path {block_path!r} continues past leaf {traversed!r}"
            )
        count, child_at = api
        if index >= count():
            raise ValueError(
                f"Block path {block_path!r} has no child {index} at {traversed!r}"
            )
        current = child_at(index)
        traversed.append(index)
        if current is None:
            raise ValueError(f"Block path {block_path!r} selects an empty block")
    if _child_api(current) is not None:
        raise ValueError(f"Block path {block_path!r} selects a non-leaf dataset")
    return current


def _reader_array_names(reader, property_name):
    """Read names advertised by a ParaView array-selection property."""
    try:
        property_value = reader.GetProperty(property_name)
        if property_value is None:
            return []
        values = list(property_value)
    except Exception:
        return []
    return [str(name) for name in values[::2] if name]


def _update_pipeline(reader, enable_arrays=False):
    """Populate reader metadata and output, optionally enabling known arrays."""
    reader.UpdatePipelineInformation()
    if enable_arrays:
        for info_name, status_name in (
            ("PointArrayInfo", "PointArrayStatus"),
            ("CellArrayInfo", "CellArrayStatus"),
        ):
            names = _reader_array_names(reader, info_name)
            if names:
                setattr(reader, status_name, names)
    reader.UpdatePipeline()


def _fetch_leaf(request, reader_type, servermanager, dataset_adapter):
    reader = _create_reader(request, reader_type)
    _update_pipeline(reader, enable_arrays=True)
    fetched = servermanager.Fetch(reader)
    leaf = _resolve_leaf(fetched, request["block_path"])
    return reader, leaf, dataset_adapter.WrapDataObject(leaf)


def _association_data(wrapper, association):
    if association == "point":
        return wrapper.PointData
    if association == "cell":
        return wrapper.CellData
    raise ValueError(
        f"Unknown field association {association!r}; expected 'point' or 'cell'"
    )


def _array_names(attributes):
    """Return stable array names from a wrapped VTK attribute collection."""
    return list(attributes.keys())


def _field_association(wrapper, field_name, requested_association=None):
    """Resolve an array association, detecting legacy-request ambiguity."""
    point_names = _array_names(wrapper.PointData)
    cell_names = _array_names(wrapper.CellData)
    if requested_association is None:
        if field_name in point_names and field_name in cell_names:
            raise ValueError(
                f"Field {field_name!r} exists in both point and cell data; "
                "supply an association"
            )
        requested_association = "point"
    attributes = _association_data(wrapper, requested_association)
    names = point_names if requested_association == "point" else cell_names
    if field_name not in names:
        raise ValueError(
            f"Field {field_name!r} is missing from {requested_association} data; "
            f"available fields are {names}"
        )
    return requested_association, attributes


def _copy_field(leaf, wrapper, field_name, association, numpy):
    """Copy one named field and validate its tuple count."""
    association, attributes = _field_association(
        wrapper, field_name, requested_association=association
    )
    value = numpy.array(attributes[field_name], copy=True, order="C")
    expected = (
        leaf.GetNumberOfPoints() if association == "point" else leaf.GetNumberOfCells()
    )
    actual = value.shape[0] if value.ndim else 0
    if actual != expected:
        raise ValueError(
            f"Field {field_name!r} has {actual} values, but the selected leaf has "
            f"{expected} {association}s"
        )
    return value


def _handle(
    request,
    reader_type,
    servermanager,
    dataset_adapter,
    delete,
    numpy,
    name_key=None,
):
    operation = request["operation"]
    if operation == "field_names":
        reader, _, wrapper = _fetch_leaf(
            request, reader_type, servermanager, dataset_adapter
        )
        try:
            association = request.get("association", "point")
            return "json", _array_names(_association_data(wrapper, association))
        finally:
            delete(reader)
    if operation == "zone_names":
        reader = _create_reader(request, reader_type)
        try:
            _update_pipeline(reader)
            fetched = servermanager.Fetch(reader)
            return "json", _leaf_records(fetched, name_key)
        finally:
            delete(reader)
    if operation == "field":
        reader, leaf, wrapper = _fetch_leaf(
            request, reader_type, servermanager, dataset_adapter
        )
        try:
            return "array", _copy_field(
                leaf,
                wrapper,
                request["field_name"],
                request.get("association"),
                numpy,
            )
        finally:
            delete(reader)
    if operation == "vertices":
        reader, leaf, wrapper = _fetch_leaf(
            request, reader_type, servermanager, dataset_adapter
        )
        try:
            number_of_points = leaf.GetNumberOfPoints()
            if number_of_points < 1 or wrapper.Points is None:
                raise ValueError(
                    f"Selected leaf at {request['block_path']!r} has empty geometry"
                )
            points = numpy.array(wrapper.Points, copy=True, order="C")
            if points.ndim != 2 or points.shape != (number_of_points, 3):
                raise ValueError(
                    f"Selected leaf reports {number_of_points} points, but its "
                    f"geometry array has shape {points.shape}"
                )
            return "array", points
        finally:
            delete(reader)
    raise ValueError(f"Unknown operation: {operation!r}")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: pvpython _tecplot_pvpython.py DATA_FD")
    # pvpython replaces sys.stdin with a VTK capture helper before this script
    # starts. File descriptor 0 still refers to the subprocess command pipe.
    command_stream = os.fdopen(os.dup(0), "rb", buffering=0)
    stream = os.fdopen(int(sys.argv[1]), "wb", buffering=0)
    try:
        try:
            import numpy
            from paraview import servermanager
            from paraview.simple import Delete, VisItTecplotBinaryReader
            from paraview.vtk.numpy_interface import dataset_adapter
            from vtkmodules.vtkCommonDataModel import vtkCompositeDataSet
        except Exception as error:
            _send_error(stream, error)
            return 1

        _send_json(stream, {"protocol": 1})
        for line in command_stream:
            try:
                request = json.loads(line)
                if request.get("operation") == "close":
                    _send_json(stream, None)
                    return 0
                kind, value = _handle(
                    request,
                    VisItTecplotBinaryReader,
                    servermanager,
                    dataset_adapter,
                    Delete,
                    numpy,
                    vtkCompositeDataSet.NAME(),
                )
                if kind == "array":
                    _send_array(stream, value, numpy)
                else:
                    _send_json(stream, value)
            except Exception as error:
                _send_error(stream, error)
        return 0
    finally:
        command_stream.close()
        stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
