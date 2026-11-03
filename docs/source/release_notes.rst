Release notes
=============

1.11.0
------

This release adds native Tecplot conversion, reusable HDF5 geometry, lossless
compression and validated snapshot append, together with expanded native
OpenFOAM reading and a complete PLT-to-SPOD-to-ParaView workflow.

Data formats and conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Add native TDV71 and TDV112 finite-element block readers, including direct
  spatial field and integer connectivity slices. TDV112 supports compatible
  variable/connectivity sharing, passive variables and point/cell association.
  Unsupported layouts fail explicitly; the legacy ParaView backend remains
  available and remains the default for direct ``TecplotDataloader`` calls.
* Add ``PLT2HDF5`` and ``flowtorch-plt-to-hdf5``. Converted datasets retain
  separate zones, including mixed surface/volume collections. Geometry and raw
  integration weights use Float64; field storage independently supports
  Float32 (default) or Float64.
* Introduce separate reusable geometry and field containers with XDMF output,
  geometry identity checks, completion metadata and direct spatial field reads.
  Geometry is computed and written one zone at a time; the largest zone and its
  geometry work arrays must still fit in memory.
* Support lossless gzip and optional Zstandard compression, configurable levels,
  and reversible byte shuffle, with independent geometry/field settings.
  Install the ``compression`` extra for Zstd. ParaView needs compatible native
  filters configured through ``HDF5_PLUGIN_PATH`` when reading Zstd data.
* Add validated append to existing complete split-file datasets. New snapshots
  are staged before publication; duplicate numeric times, incompatible field
  schemas/precision and explicit compression conflicts are rejected. Existing
  geometry and field chunks are preserved. Append inherits omitted field/zone
  selections and compression settings. ``append`` and ``overwrite`` are exclusive.
* Extend the existing OpenFOAM converter with split geometry, compression and
  append. Default conversion exports supported volume fields and skips face
  fields such as ``phi``; explicit unsupported requests fail before output.

OpenFOAM reader
~~~~~~~~~~~~~~~

* Read native ASCII and binary internal fields, uniform fields, and gzip files.
  Binary scalar/label sizes and byte order follow the ``arch`` header.
* Add metadata-based shape discovery and processor-consistent spatial slices.
  Binary uncompressed fields support direct seeks; ASCII and gzip slices scan
  or decompress preceding data and are not constant-time random access.
* Improve native mesh parsing and bulk geometry operations. Cell association
  and raw geometric integration measures remain consistent across fields,
  locations and weights.

Analysis export and documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Export chunk-readable physical fields, complex mode components and SPOD phase
  animations against existing read-only geometry. ``export_fields`` can append
  explicitly labelled reconstructed frames. Normalization is undone on export.
* Document conversion, chunked SVD/SPOD, compression, append/recovery and ParaView
  workflows. Expand the glossary and separate format guides, algorithm examples
  and worked applications in the documentation navigation.
* Handle native Windows path separators when detecting OpenFOAM processor
  directories and regenerating XDMF descriptors.
* Fix environment capture with older PyTorch precision proxies, and resolve
  static typing findings across loaders, exports, provenance and numerical APIs.

Compatibility and limits
~~~~~~~~~~~~~~~~~~~~~~~~

Existing unversioned HDF5 files remain readable. New output defaults to no
compression; existing shared geometry is never silently replaced. Explicit
validated append requires split-file output; legacy low-level writer behavior
is unchanged. Readers and ParaView must be closed before appending. Concurrent
writers, SWMR/live refresh, partial-snapshot resume and power-loss durability
are not provided. The XDMF descriptor can be regenerated if its replacement
fails after a successful data commit.

Selected zones with different dimensions can be converted together, but their
area and volume weights do not automatically define one modal-analysis norm.
Choose compatible zones or supply deliberate combined weights. Ordered Tecplot
zones, polyhedral face maps, bit-packed variables and face-neighbour records
remain unsupported by the native backend. OpenFOAM boundary fields, collated
output and moving meshes are outside the revised reader/converter scope.

Validation
~~~~~~~~~~

The complete available test suite passed on Python 3.10, 3.12 and 3.14:
1,045 passed and five CUDA-dependent skips in each environment. Formatting,
lint, mypy, Sphinx and distribution checks passed. Separate application checks
cover the supplied surface and volume PLT datasets, lossless HDF5 compression,
append, and representative ParaView readback; full-volume rendering was not
part of the compression check. See the workflow guides and validation records
for the exact scope of each check.

1.10.1
------

This patch release fixes floating-point dtype handling in data loaders and
makes EnSight precision limitations explicit.

Fixes
~~~~~

* ``TecplotDataloader`` now applies its requested ``dtype`` consistently to
  snapshots, time series, vertices, cell locations, and integration weights.
  Previously, only weights honored this setting. Backend array transport
  preserves the reader's dtype until conversion at the tensor boundary.
* ``ImageDataloader`` converts pixels to the requested dtype before
  normalization, avoiding an intermediate float32 division when float64 is
  requested.

Precision warnings
~~~~~~~~~~~~~~~~~~

* ``EnSightDataloader(dtype=torch.float64)`` emits a ``UserWarning`` once per
  construction. EnSight Gold binary files store float32 fields and coordinates;
  the current VTK ASCII reader also reads these values as float32. Returned
  float64 tensors cannot recover the additional source digits. The native
  binary chunk backend shares the file format's precision limit.
* ``EnSightWriter`` emits a ``UserWarning`` once per writer when converting
  float64 fields (including complex128 components) to binary float32 output.
  Binary remains the default. ASCII preserves double-precision field values
  in the file, but the current VTK EnSight reader reads them as float32.

Compatibility
~~~~~~~~~~~~~

Tecplot snapshots and coordinates now follow the default ``torch.float32``
dtype, even when the reader provides float64 arrays. Pass
``dtype=torch.float64`` explicitly to retain the reader's available precision
in returned tensors. This setting cannot restore precision lost in a source
file or its reader. The VTK mesh export retains its reader-provided precision.
SCube behavior is unchanged.
