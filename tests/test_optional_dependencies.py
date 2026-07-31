"""Tests for optional dependency isolation."""

import subprocess
import sys


def _imported_modules(code: str):
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()


def test_analysis_does_not_import_data_dependencies():
    modules = _imported_modules("""
import sys
import flowtorch.analysis
for name in ("flowtorch.data", "h5py", "netCDF4", "pandas", "plotly", "vtk"):
    if name in sys.modules:
        print(name)
""")
    assert modules == []


def test_lightweight_data_export_does_not_import_loader_dependencies():
    modules = _imported_modules("""
import sys
from flowtorch.data import mask_box
for name in ("h5py", "netCDF4", "pandas", "paraview", "vtk"):
    if name in sys.modules:
        print(name)
""")
    assert modules == []


def test_tecplot_loader_does_not_import_paraview():
    modules = _imported_modules("""
import sys
from flowtorch.data import TecplotDataloader
for name in sys.modules:
    if name == "paraview" or name.startswith("paraview."):
        print(name)
""")
    assert modules == []
