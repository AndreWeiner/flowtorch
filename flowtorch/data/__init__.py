"""Data loading and preprocessing tools.

Public objects are imported lazily so using one loader does not require the
optional dependencies of every other loader.
"""

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "FOAMDataloader": ("foam_dataloader", "FOAMDataloader"),
    "FOAMCase": ("foam_dataloader", "FOAMCase"),
    "FOAMMesh": ("foam_dataloader", "FOAMMesh"),
    "HDF5Dataloader": ("hdf5_file", "HDF5Dataloader"),
    "HDF5Writer": ("hdf5_file", "HDF5Writer"),
    "FOAM2HDF5": ("hdf5_file", "FOAM2HDF5"),
    "XDMFWriter": ("hdf5_file", "XDMFWriter"),
    "copy_hdf5_mesh": ("hdf5_file", "copy_hdf5_mesh"),
    "CSVDataloader": ("csv_dataloader", "CSVDataloader"),
    "VTKDataloader": ("vtk_dataloader", "VTKDataloader"),
    "PSPDataloader": ("psp_dataloader", "PSPDataloader"),
    "TAUDataloader": ("tau_dataloader", "TAUDataloader"),
    "TAUSurfaceDataloader": ("tau_dataloader", "TAUSurfaceDataloader"),
    "TAUConfig": ("tau_dataloader", "TAUConfig"),
    "TecplotDataloader": ("tecplot_dataloader", "TecplotDataloader"),
    "SequenceTensorDataset": ("sequence_dataset", "SequenceTensorDataset"),
    "SCUBEDataloader": ("s_cube_dataloader", "SCUBEDataloader"),
    "ImageDataloader": ("image_dataloader", "ImageDataloader"),
    "curvilinear_gradient": ("differential_tools", "curvilinear_gradient"),
    "element_areas_to_node_weights": (
        "geometry_tools",
        "element_areas_to_node_weights",
    ),
    "grid_element_areas": ("geometry_tools", "grid_element_areas"),
    "map_points_to_grid_2d": ("interpolation_tools", "map_points_to_grid_2d"),
    "replace_masked_values": ("interpolation_tools", "replace_masked_values"),
    "mask_box": ("selection_tools", "mask_box"),
    "mask_image_interactive": ("selection_tools", "mask_image_interactive"),
    "mask_polygon": ("selection_tools", "mask_polygon"),
    "mask_psp_interactive": ("selection_tools", "mask_psp_interactive"),
    "mask_sphere": ("selection_tools", "mask_sphere"),
    "iqr_outlier_replacement": ("outlier_tools", "iqr_outlier_replacement"),
    "replace_spatial_outliers": ("outlier_tools", "replace_spatial_outliers"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Load a public data object when it is first accessed."""
    try:
        module_name, object_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(f".{module_name}", __name__), object_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
