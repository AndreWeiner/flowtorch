Native OpenFOAM reader
======================

``FOAMDataloader`` reads cell-associated internal fields in native ASCII or
binary files, including gzip files named ``p.gz``, ``U.gz``, and compressed
``constant/polyMesh`` files. Compression is detected per file, so mixed cases
work without a constructor option or extra package. Plain files take precedence
if both variants exist; field names omit ``.gz``. Binary scalar and label widths
and byte order follow the ``arch`` header (32/64 bits, LSB/MSB); absent headers
retain the historical little-endian label32/scalar64 assumption.

Scalar, vector, spherical tensor, symmetric tensor and tensor volume fields
are supported. Uniform fields expand using the processor-local mesh cell count.
Decomposed cases concatenate cells in numeric processor order, consistently with
centres and volumes; this differs from reconstructed OpenFOAM cell ordering.
Processor directories must be contiguous starting at ``processor0``. Collated
output, boundary fields, multiple regions and moving meshes remain outside this
reader's supported layout.

Partial field loading
---------------------

Use the existing loader slice interface::

    from flowtorch.data import FOAMDataloader

    loader = FOAMDataloader("/path/to/case")
    shape = loader.snapshot_shape("U", "0.5")
    velocity = loader.load_snapshot_slice("U", "0.5", slice(100000, 200000))
    series = loader.load_snapshot_slice("p", ["0.1", "0.2"], slice(0, 65536))

Slices follow Python bounds and negative-index rules with unit stride. Field
lists and time lists use the same return conventions as ``load_snapshot``;
time is the last dimension. Non-unit strides are rejected. Shape discovery
reads nonuniform field metadata without loading payloads. Uniform shape
discovery may scan owner addressing if its optional cell-count note is absent.

* Uncompressed binary: seek to the selected range and read only its payload.
* ASCII: scan preceding values, retaining the requested range. Text-parser
  workspace also depends on the longest input line.
* Gzip ASCII/binary: decompress preceding bytes to reach the range. This bounds
  field materialization but does not provide inexpensive random access.

Repeated slice requests reopen the files. Therefore, repeated chunks of ASCII
or gzip data can repeatedly scan/decompress the same prefix. For repeated
analysis, conversion to a chunk-addressable format such as HDF5 is preferable.
Output memory scales with the selected cells, components, fields and times;
requesting all times still allocates that entire selected time series.

Geometry is eager: accessing ``vertices`` or ``weights`` loads or computes the
whole mesh. Binary mesh faces use ``faceCompactList`` connectivity. Geometry
computation uses tensor reductions across faces and rejects non-finite or
non-positive volumes instead of returning invalid integration weights.
``FOAM2HDF5`` accepts gzip inputs but currently loads complete fields and mesh
geometry; output chunk settings alone do not bound conversion memory.

See :doc:`foam_hdf5` for conversion options.
