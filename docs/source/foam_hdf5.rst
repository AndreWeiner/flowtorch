OpenFOAM to HDF5 conversion
===========================

``FOAM2HDF5`` keeps its public constructor and ``convert`` entry point. Existing
single-file output remains the default. For separate reusable Float64 geometry::

    import torch
    from flowtorch.data import FOAM2HDF5

    FOAM2HDF5("/path/to/case", dtype=torch.float64).convert(
        "fields.h5", fields=["p", "U"], times=["0.1", "0.2"],
        geometry_file="/path/to/geometry.h5", spatial_chunk=65536,
        compression="gzip", compression_level=4, shuffle=True,
        geometry_compression="gzip", geometry_shuffle=True,
    )

The converter retains polyhedral connectivity, cell centres and raw volumes.
With ``fields=None``, it exports supported cell-associated volume fields;
face-associated fields such as ``phi`` are excluded. Explicitly requesting an
unsupported field raises an error before creating output.
Field precision is honored, including double fields. Geometry uses Float64 for
both layouts. Unknown selections fail and replacing fields requires explicit
``overwrite=True``. Decomposed cases and time-dependent mesh directories are
rejected; reconstruct the case or map it to a fixed mesh first. Boundary-patch
export is outside this interface. The converter still reads complete fields;
chunked output does not make its input path out of core. Native gzip inputs are
supported. For partial field reads and their storage-dependent limits, see
:doc:`foam_reader`.

The same compression options are available for single-file output. Field settings
apply to time-dependent arrays; geometry settings apply to constant arrays.
Existing shared geometry is reused read-only, preserving its compression.
Zstd is also supported with the optional ``compression`` installation extra.

For the shared storage schema and geometry-based result export, see
:doc:`hdf5_workflow`.

Appending additional times
--------------------------

Use split output for append. The existing geometry path and field compression are
inherited, while the converter's dtype must match the original field precision::

    converter = FOAM2HDF5("/path/to/case", dtype=torch.float64)
    converter.convert(
        "fields.h5", fields=["p", "U"], times=["0.1", "0.2"],
        geometry_file="/path/to/geometry.h5", compression="gzip", shuffle=True,
    )
    converter.convert("fields.h5", times=["0.3", "0.4"], append=True)

Duplicate times and changed geometry/field schemas are rejected. ``append=True``
and ``overwrite=True`` are mutually exclusive. Existing legacy single-file output
cannot use this append mode. See :doc:`hdf5_workflow` for the complete compression,
publication and recovery options.
