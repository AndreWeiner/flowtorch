"""Native TDV71/TDV112 backend for :class:`TecplotDataloader`. No ParaView required."""

from pathlib import Path
import numpy as np
from ._tecplot71 import index_by_zone as index71
from ._tecplot112 import index_by_zone as index112


class NativeTecplotClient:
    """Offset index cache; field requests read only the requested array interval."""

    def __init__(self):
        self.indices = {}

    def index(self, path):
        path = str(Path(path))
        if path not in self.indices:
            with open(path, "rb") as stream:
                magic = stream.read(8)
            if magic not in (b"#!TDV71 ", b"#!TDV112"):
                raise ValueError("Native Tecplot supports TDV71 and TDV112 only")
            self.indices[path] = (index71 if magic == b"#!TDV71 " else index112)(path)
        return self.indices[path]

    def arrays(self, path, zone):
        return self.index(path)[zone]

    def field_arrays(self, path, zone, association="point"):
        item = self.arrays(path, zone)
        locations = item.get("field_associations", {})
        return {
            name: array
            for name, array in item["fields"].items()
            if locations.get(name, "point") == association
        }

    def vtk_cell_type(self, path, zone):
        item = self.arrays(path, zone)
        if "vtk_cell_type" in item:
            return item["vtk_cell_type"]
        return {0: 5, 1: 9, 2: 10, 3: 12}[item["cell_type"]]

    def connectivity(self, path, zone, part=slice(None)):
        item = self.arrays(path, zone)
        return np.asarray(item["connectivity"][part], dtype=np.int64) - item.get(
            "index_base", 1
        )

    def mesh(self, path, zone, cell_type=None):
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid, vtkCellArray
        from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

        item = self.arrays(path, zone)
        points = np.column_stack(
            [item["fields"][key][:] for key in ("X", "Y", "Z")]
        ).astype(np.float64)
        ids = self.connectivity(path, zone)
        if not np.isfinite(points).all() or ids.min() < 0 or ids.max() >= len(points):
            raise ValueError("Invalid PLT coordinates or connectivity")
        kind = self.vtk_cell_type(path, zone)
        if cell_type is not None:
            kind, nodes = {
                "triangle": (5, 3),
                "quad": (9, 4),
                "tetra": (10, 4),
                "hexahedron": (12, 8),
                "line": (3, 2),
            }[cell_type]
            if ids.shape[1] != nodes:
                raise ValueError("Cell type override does not match connectivity")
        vtk_points = vtkPoints()
        vtk_points.SetData(numpy_to_vtk(points, deep=True))
        vtk_cells = vtkCellArray()
        vtk_cells.SetData(
            numpy_to_vtkIdTypeArray(
                np.arange(len(ids) + 1, dtype=np.int64) * ids.shape[1], deep=True
            ),
            numpy_to_vtkIdTypeArray(ids.reshape(-1), deep=True),
        )
        mesh = vtkUnstructuredGrid()
        mesh.SetPoints(vtk_points)
        mesh.SetCells(kind, vtk_cells)
        return mesh

    def request(self, operation, file_path, **parameters):
        index = self.index(file_path)
        if operation == "zone_names":
            return [{"name": name, "path": [i]} for i, name in enumerate(index)]
        zone = list(index)[parameters["block_path"][0]]
        item = index[zone]
        association = parameters.get("association", "point")
        if operation == "field_names":
            return list(self.field_arrays(file_path, zone, association))
        if operation == "field":
            if association != "point":
                raise ValueError(
                    "Native TDV71/TDV112 backend supports nodal fields only"
                )
            array = item["fields"][parameters["field_name"]]
            return np.array(array[:], dtype=array.dtype.newbyteorder("="), copy=True)
        if operation == "vertices":
            return np.column_stack(
                [item["fields"][key][:] for key in ("X", "Y", "Z")]
            ).astype(np.float64)
        from ._tecplot_pvpython import _geometric_weights, _cell_locations

        mesh = self.mesh(file_path, zone, parameters.get("cell_type"))
        if operation == "weights":
            return _geometric_weights(mesh, association, np)
        if operation == "locations":
            return _cell_locations(mesh, np)
        raise ValueError(f"Unsupported native operation {operation}")

    def close(self):
        self.indices.clear()
