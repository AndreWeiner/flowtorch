"""Run with pvpython: check_xdmf.py descriptor.xdmf report.json.

Application-only readback: checks arrays against the referenced HDF5 datasets,
geometry precision, association, cell counts and positive cell measures.
Requires h5py in pvpython; large arrays are checked one zone at a time.
"""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from paraview import servermanager
from paraview.simple import OpenDataFile
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter


def leaves(data):
    if data.IsA("vtkUnstructuredGrid"):
        return [data]
    iterator = data.NewIterator()
    iterator.InitTraversal()
    result = []
    while not iterator.IsDoneWithTraversal():
        item = iterator.GetCurrentDataObject()
        if item is not None and item.IsA("vtkUnstructuredGrid"):
            result.append(item)
        iterator.GoToNextItem()
    return result


def check(descriptor: Path, report_path: Path) -> None:
    """Validate each time/zone against descriptor references and compiled measures."""
    descriptor = descriptor.resolve()
    frames = ET.parse(descriptor).findall("./Domain/Grid/Grid")
    reader = OpenDataFile(str(descriptor))
    reader.UpdatePipelineInformation()
    for key in ["GridStatus", "PointArrayStatus", "CellArrayStatus"]:
        if key in reader.ListProperties():
            available = getattr(getattr(reader, key), "Available", [])
            if available:
                setattr(reader, key, list(available))
    report = {"descriptor": str(descriptor), "steps": [], "passed": False}

    def array(item):
        file, key = item.text.strip().rsplit(":", 1)
        with h5py.File(descriptor.parent / file, "r") as handle:
            return handle[key][:]

    for frame in frames:
        time = float(frame.find("Time").attrib["Value"])
        reader.UpdatePipeline(time=time)
        actual = leaves(servermanager.Fetch(reader))
        expected = frame.findall("Grid")
        assert len(actual) == len(expected), "Zone count mismatch"
        zones = []
        for mesh, grid in zip(actual, expected):
            points = vtk_to_numpy(mesh.GetPoints().GetData())
            assert points.dtype == np.float64
            np.testing.assert_array_equal(
                points, array(grid.find("./Geometry/DataItem"))
            )
            topology = grid.find("Topology")
            assert mesh.GetNumberOfCells() == int(topology.attrib["NumberOfElements"])
            if topology.attrib["TopologyType"] != "Mixed":
                np.testing.assert_array_equal(
                    vtk_to_numpy(mesh.GetCells().GetConnectivityArray()),
                    array(topology.find("DataItem")).reshape(-1),
                )
            fields = {}
            for attribute in grid.findall("Attribute"):
                data = (
                    mesh.GetPointData()
                    if attribute.attrib["Center"] == "Node"
                    else mesh.GetCellData()
                )
                values = vtk_to_numpy(data.GetArray(attribute.attrib["Name"]))
                reference = array(attribute.find("DataItem"))
                assert values.dtype == reference.dtype
                np.testing.assert_array_equal(
                    values.reshape(reference.shape), reference
                )
                fields[attribute.attrib["Name"]] = str(values.dtype)
            sizes = vtkCellSizeFilter()
            sizes.SetInputData(mesh)
            sizes.Update()
            dimension = mesh.GetCell(0).GetCellDimension()
            measure = vtk_to_numpy(
                sizes.GetOutput()
                .GetCellData()
                .GetArray({1: "Length", 2: "Area", 3: "Volume"}[dimension])
            )
            assert np.all(np.isfinite(measure) & (measure > 0)), "Invalid cell measures"
            zones.append(
                {
                    "name": grid.attrib["Name"],
                    "points": len(points),
                    "cells": mesh.GetNumberOfCells(),
                    "minimum_measure": float(measure.min()),
                    "fields": fields,
                }
            )
        report["steps"].append({"time": time, "zones": zones})
        print(f"t={time}: {len(zones)} zones passed", flush=True)
    report["passed"] = bool(report["steps"])
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("descriptor", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    check(args.descriptor, args.report)
