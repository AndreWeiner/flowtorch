EnSight modes and reconstructions
=================================

EnSightDataloader and EnSightWriter support static OpenFOAM EnSight Gold volume
meshes, patches and sampled surfaces. Both ASCII and C-binary files are supported,
including arbitrary polygons (nsided) and polyhedra (nfaced). VTK is an optional
runtime dependency, installed with ``pip install 'flowtorch-fluid[data]'``.

Loading field chunks
--------------------

Dense C-binary field slices use the native reader by default::

    loader = EnSightDataloader(
        "input.case", part="internalMesh", slice_backend="native"
    )
    velocity_chunk = loader.load_snapshot_slice(
        "U", loader.write_times[0], slice(1000, 2000)
    )

``slice_backend="native"`` is the default. Select ``"vtk"`` to read full
snapshots before slicing. For C-binary cases, construction loads geometry through
VTK without loading
field arrays. Slice requests read only the intersecting field component ranges;
``snapshot_shape`` and ``field_names`` do not load field payloads. Multiple fields
and times retain the usual return shapes. Slices must have unit stride.

Full ``load_snapshot`` requests always use VTK, even with native slices enabled.
Geometry, vertices, weights and mesh export also retain the existing VTK path.
Geometry stays resident; field memory scales with the requested slice, component
count, number of requested times and output dtype, with temporary chunk buffers.

Native slices support dense nodal and cell scalar, vector and tensor fields on
static unstructured C-binary geometry, including mixed linear elements, polygons
and polyhedra. The original part IDs and block counts are indexed without reading
coordinates or field payloads and checked against VTK topology. Field indexes are
cached in a bounded cache and refreshed when file size or modification time changes.
Files must remain unchanged during a read.

ASCII, Fortran binary, ghost elements, higher-order elements, repeated element
blocks, and partial/undefined fields are unsupported by native slices. Unsupported
sections raise an exception when native slices are requested; full snapshots
continue to use VTK. Explicitly select ``slice_backend="vtk"`` to use the
existing reader. There is no automatic fallback to full-field materialization.

Selecting and copying a mesh
----------------------------

Select a named part explicitly for a multipart source. A single-part cutting
plane can be selected automatically. Parts describe physical regions or surfaces,
not MPI ranks; assembled parallel OpenFOAM exports work like serial exports::

    from flowtorch.data import EnSightDataloader, EnSightWriter

    loader = EnSightDataloader("input.case", part="internalMesh")
    writer = EnSightWriter.from_loader(
        loader, "results/reconstruction", case_name="reconstructed"
    )

The equivalent constructor from an existing descriptor is::

    writer = EnSightWriter.from_case(
        "input.case",
        output_dir="results/reconstruction",
        case_name="reconstructed",
        part="internalMesh",
        association="cell",
        format="binary",
    )

These are alternative calls, not consecutive calls to the same output directory.
``case_name`` defaults to the source filename stem. The examples produce
``results/reconstruction/reconstructed.case``. The selected part retains its name
and receives output part ID 1. For nodal data, use ``association="point"`` in the
loader or from_case constructor; from_loader inherits that association.

The output directory must be new or empty. The writer refuses locations containing
source descriptors, meshes or variable files and refuses existing output datasets.
It serializes a copy of the selected part's actual mesh once and writes independent
variable files. The output is portable and remains readable if the original data
are removed. ``loader.mesh`` provides an independent VTK mesh copy;
``loader.vertices`` gives field locations and is not sufficient to rebuild a mesh.

Writing a mode and reconstructed signal
---------------------------------------

This dense SVD example explicitly restores the spatial and component dimensions::

    from flowtorch.analysis import SVD
    from flowtorch.data import EnSightDataloader, EnSightWriter

    loader = EnSightDataloader("input.case", part="internalMesh")
    times = loader.write_times
    velocity = loader.load_snapshot("U", times)  # (N, 3, T)
    decomposition = SVD(velocity.reshape(-1, len(times)), rank=1, mode="svd")
    velocity_mode = decomposition.U[:, 0].reshape(velocity.shape[:-1])
    reconstructed = decomposition.reconstruct().reshape_as(velocity)

    with EnSightWriter.from_loader(
        loader, "results/pod", case_name="pod"
    ) as writer:
        writer.write_static("U_mode_001", velocity_mode)
        writer.write_series({"U_reconstructed": reconstructed}, times)
        output_case = writer.case_file

Open the resulting ``pod.case`` in ParaView. The mode is a static field available
at every reconstruction time; ``U_reconstructed`` follows the physical time values.
Modes can also be written alone, or several named modes can be added with repeated
write_static calls. A modal index is not treated as a physical time.

``write_static`` accepts any time-independent field, such as a mean, a spatial
mask or an individual mode. It performs no modal processing and adds no timestep::

    with EnSightWriter.from_loader(loader, "results/mean") as writer:
        writer.write_static("mean_pressure", mean_pressure)

Viewing static fields and time series in ParaView
-------------------------------------------------

A writer produces one ``.case`` descriptor containing both kinds of fields.
The mesh is in ``geometry``, static variable files are under ``data/static/``,
and transient variable files are under numbered directories such as
``data/00000000/``. Open the ``.case`` descriptor, rather than the individual
variable files, in ParaView and click **Apply**. Enable the desired arrays in
Properties if needed, then select a field in the coloring selector.

A static field remains unchanged as you advance through time. A transient field
follows the selected time; use **Play** to animate it. Two views can display
these fields side by side. Complex input produces separate ``_real`` and
``_imag`` arrays. A case containing only static fields has no transient time set.

Streaming snapshots
-------------------

For large or lazily evaluated reconstructions, write one snapshot at a time::

    with EnSightWriter.from_loader(loader, "results/streamed") as writer:
        for time, snapshot in reconstructed_snapshots:
            writer.write_snapshot({"U_reconstructed": snapshot}, time)

Each successful operation publishes the descriptor only after its referenced
files are complete. Successfully written snapshots remain readable if a later
operation fails. An interrupted write can leave unreferenced files, which are
not silently overwritten on retry. A writer is closed after its context exits.

Field conventions
-----------------

Snapshot shapes match the loader: scalar ``(N,)``, vector ``(N,3)``, symmetric
tensor ``(N,6)``, and full tensor ``(N,9)``. Series append time last. Symmetric
tensors use OpenFOAM order ``xx,xy,xz,yy,yz,zz`` and full tensors use row-major
order. The writer converts components to the EnSight file convention.

Transient snapshots must have the same field schema and unique increasing numeric
times. String time labels are preserved. Static fields may coexist under different
names. Tensors are detached and transferred to CPU one snapshot at a time; caller
values are not modified. Binary values and coordinates are single precision;
ASCII uses 17 significant digits, but readers can still use single precision.

Complex modes and reconstructions are split into ``NAME_real`` and ``NAME_imag``
fields by default. ``complex_mode="error"`` rejects complex input explicitly.
The imaginary component is never discarded silently. Native complex EnSight
variables are not emitted by this API.

The writer expects physical field shapes and values. When using normalized,
weighted, combined or distributed state vectors, use the analysis result's layout
and physical-field helpers to recover the intended arrays before writing. The
writer does not infer an analysis layout, undo normalization, add a removed mean,
or gather distributed results automatically.

Scope and reader behavior
-------------------------

The initial writer supports one static part and one field association, with serial
output. Linear triangles, quads, polygons, tetrahedra, hexahedra, wedges, pyramids
and polyhedra are supported. Higher-order elements, moving geometry, multipart
output and distributed writing are rejected or outside the interface.

Values follow the source VTK mesh row order. Output geometry and variables use
matching element-type blocks. Cell labels are not OpenFOAM's original numbering,
and different decompositions need not share the same row order.

The serializer preserves polygon connectivity in the files. VTK's ASCII EnSight
reader reverses nsided winding on readback, which can reverse polygon surface
normals; its binary reader preserves winding. Binary is the recommended format
when exact winding through ParaView matters.

Validation
----------

Self-contained unit fixtures cover ordinary and polyhedral meshes, assembled MPI
exports, patches, cutting planes, scalar/vector/tensor fields, nodal data, complex
modes, POD reconstruction, source protection, portable copies and failed writes.
OpenFOAM and ParaView are not required for unit tests. ParaView's reader was also
checked separately with generated ASCII and binary modal/reconstructed datasets.
