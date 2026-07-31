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


def _fetch_zone(request, reader_type, servermanager, dataset_adapter):
    reader = _create_reader(request, reader_type)
    if "field_names" in request:
        reader.PointArrayStatus = request["field_names"]
    fetched = servermanager.Fetch(reader)
    block = fetched.GetBlock(0).GetBlock(request["zone_index"])
    return reader, dataset_adapter.WrapDataObject(block)


def _parse_block_name(meta_data):
    for line in meta_data.split("\n"):
        if "NAME" in line:
            return line.split(":")[-1].strip()
    raise ValueError("Could not find a block name in the Tecplot metadata")


def _handle(request, reader_type, servermanager, dataset_adapter, delete, numpy):
    operation = request["operation"]
    if operation == "field_names":
        reader = _create_reader(request, reader_type)
        try:
            return "json", list(reader.GetProperty("PointArrayInfo")[::2])
        finally:
            delete(reader)
    if operation == "zone_names":
        reader = _create_reader(request, reader_type)
        try:
            fetched = servermanager.Fetch(reader)
            root = fetched.GetBlock(0)
            names = [
                _parse_block_name(str(root.GetMetaData(index)))
                for index in range(root.GetNumberOfBlocks())
            ]
            return "json", names
        finally:
            delete(reader)
    if operation == "field":
        reader, wrapper = _fetch_zone(
            request, reader_type, servermanager, dataset_adapter
        )
        try:
            field_names = request["field_names"]
            value = wrapper.PointData[field_names.index(request["field_name"])]
            return "array", numpy.array(value, copy=True, order="C")
        finally:
            delete(reader)
    if operation == "vertices":
        reader, wrapper = _fetch_zone(
            request, reader_type, servermanager, dataset_adapter
        )
        try:
            return "array", numpy.array(wrapper.Points, copy=True, order="C")
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
