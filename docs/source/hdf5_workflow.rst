PLT conversion and reusable HDF5 geometry
=========================================

Convert a fixed-mesh PLT time series independently of any analysis::

    flowtorch-plt-to-hdf5 /path/to/plt converted --fields u v w cp

    # Exact zone order, double fields and optional compression:
    flowtorch-plt-to-hdf5 /path/to/plt converted-double \
        --zones wing-upper wing-lower --fields cp --precision double \
        --geometry converted/geometry.h5 --compression gzip

The equivalent Python interface is::

    import torch
    from flowtorch.data import PLT2HDF5, HDF5Dataloader

    descriptor = PLT2HDF5("/path/to/plt").convert(
        "converted", fields=["cp"], field_dtype=torch.float32,
        spatial_chunk=65536,
    )
    with HDF5Dataloader(descriptor, dtype=torch.float64) as loader:
        values = loader.load_snapshot_slice("cp", loader.write_times[:8], slice(0, 10000))

File discovery uses numeric ``t=...`` filename labels. For different naming
schemes, supply ``file_names={"0.0": "snapshot-a.plt", ...}`` to ``PLT2HDF5``.
One isolated file without a time label receives time zero. Duplicate numeric
times fail, including different field variants of the same snapshot; select
those files explicitly. Selection errors, changing meshes, invalid measures,
and unsupported records raise exceptions. Existing field outputs require
``append=True``/``--append`` to add snapshots, or ``overwrite=True``/``--overwrite``
to replace fields. These modes are mutually exclusive. Existing shared geometry is never
silently replaced, even with overwrite enabled.

For a complete conversion, analysis and export example, see
:doc:`plt_spod_example`.

Backends and memory
-------------------

``TecplotDataloader(..., backend="native")`` reads supported TDV71 and TDV112
finite-element block layouts directly. TDV71 supports nodal triangle, quad,
tetrahedron and brick data. TDV112 additionally supports line zones, nodal and
cell-centred variables, passive variables, and compatible variable/connectivity
sharing from earlier zones. Source array precision is preserved. Direct field
slices access file offsets without loading whole fields. Unsupported layouts
fail explicitly. Native parsing needs NumPy; geometry
measures additionally need VTK. HDF5 output needs h5py. These dependencies are
available through the ``data`` installation extra.

``backend="auto"`` selects native for TDV71 and TDV112 signatures and otherwise
uses ParaView. A native parse error is not hidden by a fallback. The loader's legacy
default remains ``backend="paraview"``. ParaView needs a separate pvpython
installation; its field slicing may materialize complete snapshots and some
Tecplot readers round source coordinates internally. Writing Float64 afterward
does not recover lost digits. ``supports_direct_slicing`` reports this distinction.
The native TDV112 path currently requires full geometry-and-field files with
linear FE-block zones. Ordered/structured zones, polygon/polyhedron face maps,
bit-packed variables and face-neighbour records are unsupported. Header
auxiliary records are accepted, but geometry/text drawing records are rejected.
Prism and pyramid zones encoded as degenerate eight-node bricks retain that
source connectivity when passed to VTK.

A file may contain both surface and volume zones. Conversion preserves all
selected zones separately, including their own dimension and raw weights.
You can convert the complete mixed collection or select specific zones::

    converter = PLT2HDF5("/path/to/volume-snapshot.plt", backend="native")
    converter.convert("converted-all", fields=["mean-p"])
    # Or select a subset:
    converter.convert("converted-volume", fields=["mean-p"], zones=["Extracted pyra"])

Selecting a single file also avoids treating multiple field variants with the
same timestamp as different snapshots.

For integer topology access without constructing a VTK mesh, use
``loader.load_connectivity(slice(first_cell, last_cell), time=...)``. Returned
indices are zero-based Int64 for both native formats. As with field slicing,
this validates the requested interval, not every unrequested connectivity entry.
``snapshot_shape`` and field/zone discovery inspect metadata only. Mesh construction
and geometric weights still use VTK with complete selected-zone geometry.

``loader.geometry(time=None)`` returns an ``UnstructuredGeometry`` with Float64
points, integer zero-based topology, explicit association, analysis locations
and raw weights. ``geometry_dtype`` independently controls the ``vertices``
tensor property; its default preserves the older loader dtype behavior.
Legacy four-node surface records can require ``cell_type="quad"``.

Conversion computes, validates and writes one zone's geometry at a time, then
releases its arrays and VTK mesh before constructing the next zone. Later
snapshots are checked against the reference zone's compact signature without
retaining both geometry objects. Only zone names, field schemas and signatures
remain across zones. Existing geometry is validated and reused read-only; new
geometry stays incomplete until every selected zone and snapshot is validated.
The largest individual zone and its VTK working arrays must still fit in memory.
Field buffers are one spatial chunk at one time for native input;
ParaView may hold a complete snapshot. HDF5 chunk size does not bound geometry,
VTK, HDF5 caches or analysis work arrays. Compression is lossless and optional,
with no compression or shuffle by default. There is no automatic chunk tuning.

Lossless compression
--------------------

The HDF5 writer, PLT converter and OpenFOAM converter support ``compression=None``,
``"gzip"`` or ``"zstd"``, with ``compression_level`` and optional ``shuffle=True``.
Gzip defaults to level 4 (allowed: 0--9); Zstd defaults to level 3 (allowed:
-131072--22). Negative Zstd levels favor speed. Higher levels can cost more CPU
and memory. Byte shuffle is reversible; compression does not change Float64
geometry, integer connectivity or the selected field precision.

Configure geometry separately using ``geometry_compression``,
``geometry_compression_level`` and ``geometry_shuffle``::

    descriptor = PLT2HDF5("/path/to/plt").convert(
        "converted-zstd", fields=["cp"], spatial_chunk=65536,
        compression="zstd", compression_level=3, shuffle=True,
        geometry_compression="zstd", geometry_compression_level=3,
        geometry_shuffle=True,
    )

The command-line equivalent is::

    flowtorch-plt-to-hdf5 /path/to/plt converted-zstd --fields cp \
        --compression zstd --compression-level 3 --shuffle \
        --geometry-compression zstd --geometry-compression-level 3 --geometry-shuffle

Available compression settings are:

.. list-table:: Lossless compression options
   :header-rows: 1
   :widths: 24 24 52

   * - Python option
     - PLT command-line option
     - Behavior
   * - ``compression=None``
     - Omit ``--compression``
     - No codec for new output; inherit the existing field codec on append.
   * - ``compression="gzip"``
     - ``--compression gzip``
     - HDF5 DEFLATE; no optional compression plugin required in typical readers.
   * - ``compression="zstd"``
     - ``--compression zstd``
     - Plain HDF5 Zstandard filter; requires the optional compression extra.
   * - ``compression_level``
     - ``--compression-level N``
     - Gzip 0--9 (default 4), Zstd -131072--22 (default 3). Omit to inherit on append.
   * - ``shuffle=None``
     - Omit both shuffle flags
     - Off for new output; inherit the existing field setting on append.
   * - ``shuffle=True`` / ``False``
     - ``--shuffle`` / ``--no-shuffle``
     - Enable/disable reversible byte shuffle. On append, an explicit value must match.
   * - ``geometry_compression``
     - ``--geometry-compression gzip|zstd``
     - Independent codec for new geometry; default uncompressed.
   * - ``geometry_compression_level``
     - ``--geometry-compression-level N``
     - Independent geometry level, with the same codec-specific defaults/ranges.
   * - ``geometry_shuffle=True``
     - ``--geometry-shuffle``
     - Shuffle new geometry; default off. Existing geometry is never rewritten.
   * - ``spatial_chunk=65536``
     - ``--spatial-chunk N``
     - Rows per storage chunk. May differ for appended snapshots; old chunks remain unchanged.

No compression, shuffle alone, gzip with/without shuffle, and Zstd with/without
shuffle are supported. LZF, Blosc and lossy filters are not exposed by this writer.

For portable exports, use gzip with shuffle instead. For new files, both geometry
and field settings default independently to uncompressed. Passing ``shuffle=True`` without
a codec enables shuffle alone. Geometry settings cover points, topology,
locations and raw weights. Compression uses the same spatial chunk size as
field output; uncompressed geometry retains its contiguous layout. Fields remain
chunked even without compression. Each container records resolved compression
settings; actual filter pipelines and chunk shapes are stored by HDF5 on each
dataset. Compression is not part of geometry identity.

For direct geometry writing, use
``HDF5Writer.write_geometry(path, zones, compression="gzip", compression_level=4,
shuffle=True, spatial_chunk=65536)``. A writer created with ``geometry_file=...``
opens that geometry read-only: pass only field compression options to its
constructor. Converter geometry options apply only when creating a new geometry
file. Reusing geometry never recompresses or otherwise modifies it, including
when ``overwrite=True``. To change its storage settings, create a new geometry
file. The legacy single-file writer also accepts independent geometry settings;
these apply to constant datasets while field settings apply to time-dependent
datasets.

Zstd requires the optional native HDF5 filter. Install it with::

    pip install 'flowtorch-fluid[data,compression]'

From a development checkout, install the updated implementation and extras with::

    pip install -e '.[data,compression]'

FlowTorch registers Zstd automatically when writing or reading compressed files.
Gzip does not require this extra. Invalid options and unavailable encoders are
checked before writers or converters create output files. This implementation
uses the plain HDF5 Zstd filter, not Blosc-Zstd. Both support chunk-based reads,
but every touched compressed chunk must be decompressed. Compression reduces
storage and I/O traffic, not the size of materialized analysis arrays.

ParaView needs the filter in its own HDF5 environment. Before starting the GUI,
``pvpython`` or ``pvserver``, configure::

    export HDF5_PLUGIN_PATH="$(python -c 'import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)')"

Use a Python environment containing ``hdf5plugin``. For client-server sessions,
configure the server environment on every reading process. Plugin binaries must
be compatible with the reader's HDF5 build. Gzip plus shuffle requires no extra
plugin in the tested ParaView installation. Zstd plus shuffle also passed a
local ParaView 6.0.1 readback with the plugin path configured; this does not
establish compatibility for every ParaView or cluster build.

Appending snapshots
-------------------

Append adds snapshots to an **existing, complete split-geometry dataset** without
rewriting old fields or geometry. It is available through ``HDF5Writer``,
``PLT2HDF5.convert``, split-file ``FOAM2HDF5.convert`` and ``export_fields``.
It is not supported by the converters for legacy single-file output. The explicit
``HDF5Writer(..., append=True)`` mode also requires the split schema. Legacy
low-level writer calls retain their historical HDF5 append-by-path behavior;
they do not acquire the validation and publication guarantees described here.

For example, convert an initial batch and then append another batch::

    converter = PLT2HDF5("/path/to/plt", backend="native")
    times = list(converter.files)  # Numeric time order; assume at least 20 snapshots.
    converter.convert(
        "converted", fields=["cp"], times=times[:10], zones=["wing-upper"],
        compression="zstd", compression_level=3, shuffle=True,
        geometry_compression="zstd", geometry_shuffle=True,
    )
    converter.convert("converted", times=times[10:20], append=True)

The second call inherits field names, zone order, the geometry-file reference,
and field compression. Storage precision must match the existing file: pass
``field_dtype=torch.float64`` again when appending to double-precision PLT output.
Explicit zone/field selections must match exactly. New time labels may precede
existing labels; readers and the regenerated XDMF descriptor sort times numerically.

The command-line interface provides the same behavior::

    flowtorch-plt-to-hdf5 /path/to/plt converted --fields cp --zone wing-upper \
        --times 0.0 0.1 --compression gzip --shuffle \
        --geometry-compression gzip --geometry-shuffle
    flowtorch-plt-to-hdf5 /path/to/plt converted --times 0.2 0.3 --append

Discovery happens when ``PLT2HDF5`` is constructed. Recreate the converter after
new PLT files arrive, or provide an explicit ``file_names`` mapping. The old PLT
files are not needed when appending files from a separate new-input directory.

Select only new snapshots. Discovering a directory containing old and new files
without ``times``/``--times`` selects both, and duplicate timestamps are rejected.
Numeric equality counts as duplication (for example, ``0.1`` and ``0.10``).
There is no automatic skip or replacement of duplicate times, and append does
not resume a partially written snapshot.

.. list-table:: Split-file output modes
   :header-rows: 1
   :widths: 25 25 50

   * - Python
     - PLT CLI
     - Existing field output
   * - Default
     - Neither mode flag
     - Refuse to replace or extend it.
   * - ``append=True``
     - ``--append``
     - Require a complete split dataset, validate compatibility, add new snapshots.
   * - ``overwrite=True``
     - ``--overwrite``
     - Replace fields with the selected snapshots; do not replace shared geometry.

``append`` and ``overwrite`` cannot be combined. Append checks geometry identity
and content, zone order, field association, field names/components, array sizes,
and storage precision. Existing field filters must be uniform. Omitted field
compression options inherit the actual stored filters. Explicit codec, level or
shuffle conflicts are rejected; append never recompresses old data. New snapshots
may use a different spatial chunk size. If geometry was relocated, pass its new
path through ``geometry_file``/``--geometry``; after identity validation the field
container records that reference on commit. Geometry options do not modify reused
geometry, regardless of append or overwrite.

A direct writer can discover the existing geometry automatically::

    with HDF5Writer("converted/fields.h5", append=True) as writer:
        writer.allocate_field("cp", (), "0.2", "wing-upper")
        # Write all spatial chunks of every field in every geometry zone:
        # writer.write_slice("cp", "0.2", "wing-upper", part, values)
        # Publish and refresh the complete time series:
        # writer.write_xdmf("converted/fields.xdmf")

Use the same ``field_dtype`` as the existing file. ``finish()`` commits a batch;
``write_xdmf()`` commits it if necessary and replaces the descriptor using a
temporary file. Simply closing an unfinished append discards its staged data.
For physical result exports, pass explicit, unique new ``labels``::

    export_fields(
        reconstruction, "reconstructed.h5", "converted/geometry.h5",
        labels=new_times, append=True,
    )

Exported frames retain their batch's result metadata and trailing shape. Existing
file-level result metadata continues to describe the initial export batch.
``export_spod`` writes separate mode/animation files; use ``export_fields`` to
extend an existing reconstructed-field series.

Publication and recovery
~~~~~~~~~~~~~~~~~~~~~~~~

Appends stage new data in a reserved hidden HDF5 group. Only after all new frames
are complete and compatible does the writer publish their keys in the
``committed_frames`` index. FlowTorch readers and XDMF regeneration ignore hidden
staging groups and uncommitted frames. Ordinary exceptions leave previously
committed snapshots readable. The next append discards abandoned staging and
uncommitted frame groups, then writes the requested batch afresh. Cleanup may
leave reusable free space inside HDF5; it does not promise to shrink the file.

HDF5 publication and XDMF replacement are separate operations. If descriptor
writing fails after committing fields, the old descriptor is retained and the
HDF5 reader sees the newly committed snapshots. Regenerate the descriptor rather
than repeating the append, which would encounter duplicate times::

    from flowtorch.data import XDMFWriter
    with XDMFWriter.from_filepath("converted/fields.h5") as writer:
        writer.create_xdmf("fields.xdmf")

Close readers/ParaView before appending and reopen or reload afterward. Concurrent
writers, live reader refresh/SWMR, and crash/power-loss durability are not provided.
The publication index protects logical visibility when the HDF5 file itself
remains healthy; it cannot repair filesystem or HDF5 metadata corruption.

Geometry and schema
-------------------

The versioned format consists of ``geometry.h5``, ``fields.h5`` and
``fields.xdmf`` (``.xmf`` is also accepted when reading). Geometry stores Float64
points, association-compatible locations and raw geometric measures; topology
is integer. Fields default to Float32, independently of loader computation dtype.
Reading Float32 fields as Float64 does not restore discarded source precision.

Both containers identify schema ``flowtorch-unstructured``, version 1. Stable
internal zone, time and field identifiers retain original names as attributes.
Geometry has ``/zones/zNNNNNN/{points,topology,locations,weights}``; fields have
``/frames/tNNNNNN/zNNNNNN/fNNNNNN``. Frames retain numeric labels and are read in
numeric order. Each container records a geometry identity and completion flag.
Fields record the relative geometry-file path. XDMF references both containers
relative to the descriptor. Move the files together, retaining relative paths,
or supply ``geometry_file=...`` when loading relocated fields.

Geometry reuse verifies content hashes, zone order, association and integration
semantics. Geometry opens read-only. Each selected zone retains its local points;
interface copies are not merged. Surface and volume zones can coexist: each zone
retains its own dimension, topology and weight array. Selected zones must still
share field association, and the converter requires the same selected field
components in every zone.

``HDF5Dataloader.zone_dimensions`` reports the dimension of each selected zone;
``weights_by_zone`` returns the raw measures separately. Field loading and
visualization work across mixed dimensions. The combined ``loader.weights``
convenience property rejects mixed dimensions, because areas and volumes do not
automatically define a common analysis norm. For modal analysis, select compatible
zones (for example, ``zones=["Extracted pyra"]``) or deliberately construct
analysis weights from ``weights_by_zone`` with an explicit scaling convention.
This restriction applies to automatic analysis weighting, not conversion.

Readers reject incomplete containers. Writers require contiguous, nonoverlapping
spatial writes covering every allocated field, with all zones and consistent
fields at each time. Descriptor publication follows successful data writing.
Interrupted initial conversions can be inspected but cannot be loaded as completed
series; initial-conversion resume is not implemented. Failed append batches are
handled separately as described above. This does not promise transactional recovery across
multiple files or filesystems.

Existing unversioned FlowTorch HDF5 files remain readable, and legacy writer
calls remain available. They do not gain completion metadata automatically.
``copy_hdf5_mesh`` retains its legacy behavior for unversioned inputs; for
versioned inputs it creates/reuses a standalone geometry file without an XDMF
field descriptor.

Chunked SVD and SPOD
--------------------

Spatial slices can cross selected-zone boundaries. ``snapshot_shape`` reads
metadata without loading fields. Both the PLT native backend and revised HDF5
loader implement the existing slice interface::

    from flowtorch.analysis import DataloaderStateVectorSource, FieldSpec, SVD, PAMSPOD

    loader = HDF5Dataloader("converted/fields.h5", dtype=torch.float64)
    source = DataloaderStateVectorSource(
        loader, [FieldSpec("cp")], spatial_weight=loader.weights,
        normalization={"cp": 1.0},
    )
    svd = SVD(source, rank=20, subtract_mean=True, spatial_batch_size=65536)
    spod = PAMSPOD.from_svd(svd, dt=0.001, keep_n_modes=3)

Temporal batching uses the existing source/time-selection and ``SVD.update``
interfaces. Each batch must use the same zone order, geometry, field layout and
normalization. Temporal batching alone does not bound spatial memory. Source
requests and HDF5 storage layout support both spatial and temporal subsets.

Spatial modes and reconstructed fields
--------------------------------------

A writer can be created from geometry alone, without original PLT files::

    from flowtorch.data import HDF5Writer

    with HDF5Writer("new-fields.h5", geometry_file="converted/geometry.h5") as writer:
        writer.allocate_field("cp", (), "0", "wing-upper")
        # Supply all chunks and all geometry zones:
        # writer.write_slice("cp", "0", "wing-upper", spatial_slice, values)
        # writer.write_xdmf("new-fields.xdmf")

Use the physical-field adapter for chunk-readable analysis results::

    from flowtorch.data import export_fields, export_spod

    export_fields(svd.U, "pod-modes.h5", "converted/geometry.h5",
                  geometry_id=loader.geometry_id, zones=loader.zone_names,
                  compression="gzip", compression_level=4, shuffle=True)
    export_spod(spod, "spod", "converted/geometry.h5",
                frequency_indices=[2, 4], mode_indices=[0],
                animation_phases=[0.0, 1.5707963267948966, 3.141592653589793],
                amplitude=1.0, geometry_id=loader.geometry_id,
                compression="gzip", compression_level=4, shuffle=True)
    loader.close()

The adapter uses ``StateVectorResult.iter_chunks(split=True)`` to recover
physical field shapes and undo normalization. It writes directly in spatial
chunks, preserving geometry order. For a subset/reordering, create matching
geometry first. Supply the recorded geometry identity whenever available;
matching sizes alone cannot prove physical correspondence.

Complex fields become ``*_real`` and ``*_imag`` pairs. Optional
``representation="magnitude_phase"`` writes magnitude and phase in radians.
Trailing result dimensions become numeric frame labels, with the original
trailing shape retained as metadata; these labels are not automatically physical
times. Explicit ``labels`` can supply reconstructed-flow times.

SPOD exports record frequency, mode index, eigenvalue and normalization metadata.
Phase animations evaluate ``amplitude * real(mode * exp(1j * phase))`` and label
frames by sample index. They illustrate a mode, not reconstructed flow.
For actual reconstructions, pass ``spod.partial_reconstruction(...)`` to
``export_fields``; its ``add_mean`` option controls mean restoration.
The helper requires PAMSPOD results backed by a state-vector SVD.

Distributed export is collective on the result's process group. Supply an
explicit shared ``staging_directory`` on every rank. Ranks write disjoint
physical-field chunk files; only the root writes HDF5. Staging needs disk space
for the entire requested result in its original precision, including complex
values, in addition to final outputs. It does not gather full spatial arrays
in RAM. Normal completion and caught failures clean the unique staging folder;
process crashes can leave temporary files. Parallel HDF5 is not required.
