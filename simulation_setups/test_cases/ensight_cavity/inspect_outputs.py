"""Inspect generated EnSight files with VTK (also runs with pvpython)."""

import json
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

root = Path(__file__).resolve().parent
report = {}
for case in sorted(root.rglob("*.case")):
    reader = vtk.vtkGenericEnSightReader()
    reader.SetCaseFileName(str(case))
    reader.UpdateInformation()
    times = vtk_to_numpy(reader.GetTimeSets().GetItem(0)).tolist()
    assert np.allclose(times, [0.05, 0.1]), (case, times)
    records = []
    for time in times:
        reader.SetTimeValue(time)
        reader.Update()
        output = reader.GetOutput()
        parts = []
        for i in range(output.GetNumberOfBlocks()):
            grid = output.GetBlock(i)
            name = output.GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME())
            sizes = vtk.vtkCellSizeFilter()
            sizes.SetInputData(grid)
            sizes.Update()
            sized = sizes.GetOutput()
            area = vtk_to_numpy(sized.GetCellData().GetArray("Area")).sum()
            volume = vtk_to_numpy(sized.GetCellData().GetArray("Volume")).sum()
            association = "point" if "nodes_binary" in str(case) else "cell"
            attrs = (
                grid.GetPointData() if association == "point" else grid.GetCellData()
            )
            n = (
                grid.GetNumberOfPoints()
                if association == "point"
                else grid.GetNumberOfCells()
            )
            fields = {}
            for field in ("p", "U", "T", "S"):
                values = vtk_to_numpy(attrs.GetArray(field))
                assert values.shape == (
                    (n,) if field == "p" else (n, {"U": 3, "T": 9, "S": 6}[field])
                )
                assert np.isfinite(values).all()
                if field in ("T", "S"):
                    expected = (
                        np.arange(1, 10)
                        if field == "T"
                        else np.array([1, 4, 6, 2, 5, 3])
                    )
                    np.testing.assert_allclose(
                        values, np.broadcast_to(expected, values.shape), atol=1e-6
                    )
                fields[field] = list(values.shape)
            assert n > 0
            parts.append(
                dict(
                    name=name,
                    points=grid.GetNumberOfPoints(),
                    cells=grid.GetNumberOfCells(),
                    cell_types=sorted(
                        {grid.GetCellType(j) for j in range(grid.GetNumberOfCells())}
                    ),
                    association=association,
                    fields=fields,
                    area=float(area),
                    volume=float(volume),
                )
            )
        records.append(dict(time=time, parts=parts))
    report[str(case.relative_to(root))] = records
# Compare all paired element exports at both physical times.
for prefix in ("", "dual/", "parallel/"):
    for source in ("volume", "patch", "cuts"):
        for ascii_case in sorted(
            (root / prefix / "postProcessing" / (source + "_ascii")).rglob("*.case")
        ):
            binary_case = Path(
                str(ascii_case).replace(source + "_ascii", source + "_binary")
            )
            for time in (0.05, 0.1):
                readers = []
                for case in (ascii_case, binary_case):
                    reader = vtk.vtkGenericEnSightReader()
                    reader.SetCaseFileName(str(case))
                    reader.UpdateInformation()
                    reader.SetTimeValue(time)
                    reader.Update()
                    readers.append(reader)
                grids = [reader.GetOutput() for reader in readers]
                assert grids[0].GetNumberOfBlocks() == grids[1].GetNumberOfBlocks()
                for i in range(grids[0].GetNumberOfBlocks()):
                    for field in ("p", "U", "T", "S"):
                        np.testing.assert_allclose(
                            vtk_to_numpy(
                                grids[0].GetBlock(i).GetCellData().GetArray(field)
                            ),
                            vtk_to_numpy(
                                grids[1].GetBlock(i).GetCellData().GetArray(field)
                            ),
                            rtol=2e-5,
                            atol=1e-7,
                        )
for prefix in ("", "dual/"):
    volume_case = (
        prefix
        + "postProcessing/volume_ascii/"
        + ("dual" if prefix else "ensight_cavity")
        + ".case"
    )
    for snapshot in report[volume_case]:
        parts = {part["name"]: part for part in snapshot["parts"]}
        assert np.isclose(parts["internalMesh"]["volume"], 0.0001, rtol=1e-6)
        assert np.isclose(parts["movingWall"]["area"], 0.001, rtol=1e-6)
        assert np.isclose(parts["fixedWalls"]["area"], 0.003, rtol=1e-6)
    for snapshot in report[prefix + "postProcessing/cuts_ascii/midplane/midplane.case"]:
        assert np.isclose(snapshot["parts"][0]["area"], 0.01, rtol=1e-6)
    for snapshot in report[prefix + "postProcessing/cuts_ascii/oblique/oblique.case"]:
        assert 7 in snapshot["parts"][0]["cell_types"]  # VTK_POLYGON / nsided
assert (
    42
    in report["dual/postProcessing/volume_ascii/dual.case"][0]["parts"][0]["cell_types"]
)

# MPI exports are assembled into global parts, with processor-dependent row order.
for time in (0.05, 0.1):
    outputs = []
    for path in (
        "postProcessing/volume_binary/ensight_cavity.case",
        "parallel/postProcessing/volume_binary/parallel.case",
    ):
        reader = vtk.vtkGenericEnSightReader()
        reader.SetCaseFileName(str(root / path))
        reader.UpdateInformation()
        reader.SetTimeValue(time)
        reader.Update()
        outputs.append(reader)
    for i in range(outputs[0].GetOutput().GetNumberOfBlocks()):
        ordered = []
        for reader in outputs:
            grid = reader.GetOutput().GetBlock(i)
            centers = vtk.vtkCellCenters()
            centers.SetInputData(grid)
            centers.Update()
            xyz = vtk_to_numpy(centers.GetOutput().GetPoints().GetData())
            rounded = np.round(xyz, decimals=7)
            order = np.lexsort(rounded.T[::-1])
            ordered.append((grid, xyz, order))
        np.testing.assert_allclose(
            ordered[0][1][ordered[0][2]], ordered[1][1][ordered[1][2]], atol=1e-7
        )
        for field in ("p", "U", "T", "S"):
            a, b = [
                vtk_to_numpy(grid.GetCellData().GetArray(field))[order]
                for grid, xyz, order in ordered
            ]
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=2e-6)

(root / "inspection.json").write_text(json.dumps(report, indent=2) + "\n")
print(f'Validated {len(report)} EnSight case files; report: {root / "inspection.json"}')
