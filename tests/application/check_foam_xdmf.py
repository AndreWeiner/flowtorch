"""ParaView readback of the manufactured two-processor unit-cube chain.

Generate input with tests.data.test_foam_distributed_conversion.decomposed,
convert p/U at times 0,1,2, then run with pvpython --force-offscreen-rendering:
    check_foam_xdmf.py fields.xdmf report.json
No h5py is loaded into ParaView's bundled HDF5 runtime.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from paraview import servermanager
from paraview.simple import OpenDataFile
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("descriptor", type=Path)
parser.add_argument("report", type=Path)
args = parser.parse_args()
reader = OpenDataFile(str(args.descriptor.resolve()))
reader.UpdatePipelineInformation()
for key in ["GridStatus", "CellArrayStatus"]:
    if key in reader.ListProperties():
        prop = getattr(reader, key)
        if prop.Available:
            setattr(reader, key, list(prop.Available))
report = []
for t in [0.0, 1.0, 2.0]:
    reader.UpdatePipeline(time=t)
    data = servermanager.Fetch(reader)
    iterator = data.NewIterator()
    iterator.InitTraversal()
    meshes = []
    while not iterator.IsDoneWithTraversal():
        grid = iterator.GetCurrentDataObject()
        if grid is not None and grid.IsA("vtkUnstructuredGrid"):
            meshes.append(grid)
        iterator.GoToNextItem()
    assert len(meshes) == 2, len(meshes)
    offset = 0
    for grid, count in zip(meshes, [2, 3]):
        assert grid.GetNumberOfCells() == count
        points = vtk_to_numpy(grid.GetPoints().GetData())
        assert points.dtype == np.float64
        expected = np.array(
            [
                (offset + x, y, z)
                for x in range(count + 1)
                for y, z in [(0, 0), (1, 0), (1, 1), (0, 1)]
            ]
        )
        np.testing.assert_array_equal(points, expected)
        pressure = np.arange(offset, offset + count) + 10 * t + 0.25
        for name, values in [
            ("p", pressure),
            ("U", np.stack([pressure, pressure + 1, -pressure], axis=1)),
        ]:
            actual = vtk_to_numpy(grid.GetCellData().GetArray(name))
            assert actual.dtype == np.float32
            np.testing.assert_array_equal(actual, values)
        size = vtkCellSizeFilter()
        size.SetInputData(grid)
        size.Update()
        volumes = vtk_to_numpy(size.GetOutput().GetCellData().GetArray("Volume"))
        np.testing.assert_allclose(volumes, np.ones(count), atol=1e-12)
        offset += count
    report.append(
        {"time": t, "zones": 2, "cells": offset, "volumes": "unit", "fields": "exact"}
    )
print(json.dumps(report, indent=2))
args.report.write_text(json.dumps({"passed": True, "steps": report}, indent=2) + "\n")
