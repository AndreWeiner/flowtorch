OpenFOAM to HDF5 conversion
===========================

``FOAM2HDF5`` keeps its public constructor and ``convert`` entry point. Existing
single-file output remains the default for reconstructed cases. For separate reusable
Float64 geometry::

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
``overwrite=True``. Time-dependent mesh directories are rejected, including
updates at unselected times which later snapshots might inherit. Map moving
meshes to a fixed mesh before conversion. Boundary-patch export, collated
``processors*`` output and multi-region meshes are outside this interface.
Native gzip inputs are supported; see :doc:`foam_reader`.

Decomposed cases
----------------

Ordinary ``processor0``, ``processor1``, ... cases can be converted directly,
without reconstructing them in OpenFOAM. Each processor mesh must contain at
least one cell. Conversion runs in one process; this
is support for decomposed input, not MPI-parallel HDF5 writing::

    converter = FOAM2HDF5("/path/to/decomposed_case", dtype=torch.float32)
    converter.convert(
        "/path/to/output/fields.h5", fields=["p", "U"], times=["0.1", "0.2"],
        geometry_file="/path/to/output/geometry.h5",
        compression="gzip", shuffle=True, spatial_chunk=65536,
    )
    converter.convert(
        "/path/to/output/fields.h5", times=["0.3"], append=True,
    )

Open ``fields.xdmf`` in ParaView. Each time step contains one grid per processor,
with local polyhedral connectivity. Processor zones are ordered numerically
(``processor2`` precedes ``processor10``); cells within each zone retain their
local OpenFOAM ordering. Interface coordinates are present in both neighbouring
zones, while volume cells and their integration weights are not duplicated.
No global point merging is performed. ParaView filters requiring connectivity
across zones may require merging the grids first.

Decomposed input always produces split output. If ``geometry_file`` is omitted,
``fields.h5`` produces ``fields.geometry.h5`` beside it. For append, the existing
geometry reference is inherited. Supplying ``geometry_file`` explicitly allows
multiple field/result files to reuse the same validated geometry.

The converter checks selected times, fields, classes and cell counts on every
processor before writing output. Append requires the same processor zones,
local geometry and cell ordering, field schema, dtype and compression. A changed
mesh or decomposition requires a new dataset. The shared geometry is checked
read-only and is not rewritten on append. Duplicate times and interrupted append
follow the same validation and staging rules as :doc:`hdf5_workflow`.

``HDF5Dataloader`` concatenates the zones in their saved order, consistently for
fields, cell centres and raw volume weights. This differs from reconstructed
OpenFOAM cell numbering. Select a subset with, for example,
``HDF5Dataloader("fields.h5", zones=["processor1"])``. Spatial slices can cross
processor boundaries. Modes/results can be exported against the saved geometry
using the existing :doc:`hdf5_workflow` interfaces.

Memory and input reading
------------------------

Both reconstructed and decomposed split output stream cell fields sequentially
in ``spatial_chunk`` batches, opening each input field once. This avoids repeated
prefix scans/decompression of ASCII/gzip fields. The ASCII parser also retains
the current input line, so an unusually long line can exceed the chunk size.
Geometry remains eager within each processor: its points, faces and temporary
polyhedral topology arrays must fit in memory. Processor geometry is released
before the next is assembled; the converter does not concatenate all processor
meshes or fields. ``spatial_chunk`` therefore does not impose a total peak-memory
limit. Legacy single-file conversion still materializes complete fields and mesh.

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
