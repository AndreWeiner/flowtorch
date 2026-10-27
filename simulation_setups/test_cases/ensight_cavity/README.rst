OpenFOAM v2606 EnSight loader investigation
==========================================

Run ``./Allrun`` after sourcing OpenFOAM. The source is the installed v2606
``incompressible/icoFoam/cavity/cavity`` tutorial, retaining its GPL headers.
This 20 x 20 x 1 lid-driven cavity runs to 0.1 s and exports p, U and constant asymmetric/symmetric tensor fixtures T and S at 0.05
and 0.1 s through function objects. ``Allrun`` also creates a dual-mesh variant
using ``polyDualMesh -overwrite 60`` before solving it independently.
Solver tolerances are tightened to 1e-10 with zero relative tolerance so serial
and parallel comparisons are not dominated by early linear-solver termination.
The loader is implemented as ``flowtorch.data.EnSightDataloader``. The plan below
records the investigation; see the implemented API documentation for current scope.
Regenerate committed test fixtures with ``python save_fixtures.py`` after Allrun.

Terminology: EnSight Gold is the file format; ``nsided`` is its arbitrary
polygon element type, and ``nfaced`` its arbitrary polyhedron element type.
A loader should support the format rather than only the nsided block type.

Generated outputs
-----------------

All paths below are under ``postProcessing/`` (also under ``dual/`` and ``parallel/``).
Each export has a .case descriptor and static geometry plus transient fields.

* ``volume_ascii``: element values, internalMesh and two wall parts.
* ``volume_binary``: equivalent C-binary export.
* ``nodes_binary``: point values on the volume and wall parts.
* ``patch_ascii`` and ``patch_binary``: movingWall and fixedWalls only, element values.
* ``cuts_ascii/midplane`` and ``cuts_binary/midplane``: non-interpolated polygonal plane at z=0.0049.
* ``cuts_ascii/oblique`` and ``cuts_binary/oblique``: non-interpolated oblique plane containing tria3,
  quad4 and nsided blocks in the same part.
* ``cuts_ascii/lid`` and ``cuts_binary/lid``: sampled movingWall surface, element values.

The ordinary mesh has 400 hexahedral cells and 20/60 wall faces. Its oblique
plane has 319 elements including 21 pentagons. The dual mesh has 882 cells,
including nfaced cells, 80/240 wall faces and 382 oblique-plane elements.
The frontAndBack empty patches are deliberately excluded from mesh exports.
``interpolate false`` means surface variables are per element; cellPoint is
configured for possible later interpolated fixtures.

Verification
------------

``inspect_outputs.py`` uses vtkGenericEnSightReader, and can run with either
Python plus VTK or the installed ParaView pvpython. It checks both snapshots
for all 33 .case files, field shapes and finite values, ASCII/binary agreement,
nsided/nfaced presence, and analytical total volume/areas. Detailed metadata
is in ``inspection.json``. Generated solver outputs and logs are ignored by Git
but remain locally available; regenerate them with Allrun.

Verified locally with OpenFOAM v2606 and ParaView's VTK 9.6.2. The project test environment also validates these outputs using standard VTK. The cavity volume is
0.0001 m^3, lid area 0.001 m^2, fixed-wall area 0.003 m^2 and horizontal cut
area 0.01 m^2. A cut at exactly z=0.005 overcounted the dual-mesh surface;
moving it to z=0.0049 resolved the coincident-face ambiguity.

Original implementation plan
----------------------------

This plan records the initial investigation. The implemented behavior below and
``docs/source/ensight_io.rst`` describe the current API; numeric part IDs and
an exact-centroid option were not implemented.

1. Add ``EnSightDataloader(Dataloader)`` in ``flowtorch/data/ensight_dataloader.py``
   with ``case_file``, ``part`` (name or ID), ``association`` (cell or point,
   default cell), and ``dtype=DEFAULT_DTYPE``. Expose a ``parts`` inventory.
   Require an explicit part for multipart files; allow omission for a single
   part. Do not combine volume and wall parts, whose integration measures differ.
   Use one instance per selected part so coordinates and weights describe every
   returned field consistently. A .case path avoids ambiguous directory discovery.

2. Start with a VTK backend, using the existing optional ``data`` dependency and
   lazy export in ``flowtorch/data/__init__.py``. vtkGenericEnSightReader was
   experimentally verified for ASCII, binary, nsided and nfaced here. Keep this
   backend independent of the existing VTKDataloader: that loader assumes one
   file per snapshot, a single unstructured grid and point fields. Avoid a custom
   binary/ASCII parser initially. Verify ordinary pip VTK as well as ParaView
   before shipping; this investigation only exercised the latter.

3. Read physical times and variable metadata from the .case descriptor, preserving
   their text labels and mapping each label to the reader's numeric time. Never
   infer physical times from data/00000000 numbering. Return ordered write_times
   and field_names for every time, restricted to the selected association/part.
   Identify parts using stable EnSight IDs/names rather than transient block index.
   Check for duplicate names and ambiguous numeric time labels. Fail clearly for
   missing fields, unknown times/parts, missing files and unsupported descriptors.

4. Implement load_snapshot with FlowTorch's list/string combinations. Scalar
   snapshots have shape (N,), vectors (N,3), and time is the last dimension:
   (N,T) or (N,3,T). Read a snapshot once for all requested fields, copy VTK arrays
   into tensors of the requested dtype, and preallocate time series. Preserve
   reader element order across mixed tria3/quad4/nsided or hexa8/nfaced blocks;
   that order is not OpenFOAM's original cell numbering. Do not expose VTK views
   whose storage can be overwritten by the next reader update. Implement cheap
   snapshot_shape metadata; inherited load_snapshot_slice is sufficient initially.

5. For element fields, vertices are representative cell/face centers, not mesh nodes;
   weights are volumes for 3D elements or areas for 2D elements. vtkCellSizeFilter
   gives correct total measures for the fixtures. Verify individual values against
   FOAMDataloader for the base and dual meshes, with a spatial correspondence
   rather than row-number comparison. Use vtkCellCenters as the fast default and
   document that these are parametric centers, which may differ from OpenFOAM's
   finite-volume centroids. Offer an explicit exact-centroid mode if needed for
   coordinate-sensitive applications; do not force Python per-cell geometry work
   on large meshes. Cache coordinates/weights once for static geometry. For point
   fields, vertices are mesh points and weights distribute each element's measure
   equally over its incident nodes within the selected part. Check conservation
   and positive measures.
   Exclude empty-patch geometry and reject mixed-dimensional parts initially.

6. Define the initial scope as static geometry and stable topology. Cache selected
   geometry; reject moving geometry/topology with a clear explanation because the
   base interface exposes time-independent vertices and weights. Later add explicit
   geometry_at(time) and a policy for time-dependent integration. Detect changed
   connectivity as well as changed counts. Changing only coordinates also matters.
   Support second-order tensors initially: asymmetric tensors (N,9) in row-major
   xx,xy,xz,yx,yy,yz,zx,zy,zz order, symmetric tensors (N,6) normalized to OpenFOAM
   xx,xy,xz,yy,yz,zz order. VTK returns symmetric components as xx,yy,zz,xy,yz,xz;
   reorder with indices [0,3,5,1,4,2]. Append time last as for vectors.
   Defer measured/cloud data and unsupported time-set layouts until fixtures exist.

7. Add small, committed ASCII/binary fixtures from both runs under the project's
   dataset conventions, with provenance and a regeneration script. Tests should
   cover all four list/string calls, two physical times, multipart selection,
   mixed element ordering, nsided/nfaced, dtype, field association, analytical
   volume/area totals, point-weight conservation, ASCII/binary agreement and
   expected errors. Add a small interpolate=true surface fixture to cover per-node
   sampling separately from ensightWrite nodeValues. Add geometry-change and
   nonconsecutive filename-number fixtures for rejection/time mapping tests.
   Export/document the class and provide volume, patch and plane examples.

References
----------

* https://doc.openfoam.com/2606/tools/post-processing/function-objects/utilities/ensightWrite/
* https://doc.openfoam.com/2606/tools/post-processing/function-objects/sampling/surfaces/
* Installed v2606 ensightWrite.H, sampledPlane implementation and annotated sampleDict.

Follow-up: tensors, centers, binary surfaces and MPI
--------------------------------------------------

T has the OpenFOAM components (1 2 3 4 5 6 7 8 9); S has (1 2 3 4 5 6).
The readFields object loads these fixtures before exporters execute. VTK reads
T as [1,2,3,4,5,6,7,8,9] and S as [1,4,6,2,5,3]. These distinctive constants
verify component order in volume, patch and slice exports, both encodings and
both associations. They test serialization, not evolution of a tensor solution.

Binary patch, horizontal/oblique slice and sampled lid exports now exist for
ordinary, dual and parallel runs. All ASCII/binary pairs are compared at both
times. Validation also checks analytic tensor components.

Allrun additionally decomposes the ordinary mesh into two x-direction partitions
and runs icoFoam with MPI. Both ensightWrite and sampled surfaces produce global
.case datasets at the case root; processor interfaces are not extra wall parts.
VTK sees the same global part counts. The inspector matches cell centers to
compare parallel/serial p, U, T and S, allowing small solver roundoff differences.
No reconstructPar step is needed for these exports. Element ordering can depend
on decomposition; never assume serial and parallel row IDs match. Across a single
static export series the loader preserves the export ordering. Raw processor*
OpenFOAM files and independent per-rank EnSight exports are different inputs;
a generic .case loader does not reconstruct either implicitly.

MPI requires local sockets; this session's sandbox required an approved execution
outside that restriction. Normal local execution of Allrun needs a working MPI
runtime. Ordinary Python still needs the optional VTK dependency, or use pvpython.

Implemented loader
------------------

EnSightDataloader selects parts by their unique names, not numeric EnSight IDs
(which VTK does not expose). It supports static unstructured geometry and a single
time set, preserves textual time labels and handles both sequential and explicit
filename numbering. Scalar, vector, symmetric and full tensor fields are supported
at elements or nodes; list-valued time requests place time on the last axis.

Coordinates and integration weights are computed once and cached. Standard cells
use vtkCellCenters. Arbitrary polygons use a vectorized mean of their vertices:
testing found vtkPolygon parametric centers could shift substantially between ASCII
and binary output after tiny coordinate-rounding changes. Neither convention
promises exact OpenFOAM finite-volume centroids; an exact-centroid mode is deferred.
Areas/volumes come from vtkCellSizeFilter. Point weights conservatively distribute
each element's measure equally among its incident nodes, without Python cell loops.

The 33 exports are compressed into tests/data/fixtures/ensight_cavity.zip. Unit tests
need only PyTorch, NumPy, SciPy and VTK, not OpenFOAM or MPI. They exercise serial,
parallel, polyhedral, polygonal, patch and plane data, all field associations and
types, conservation, component order, physical time/file numbering and errors.

Writing modal results
---------------------

EnSightWriter.from_loader(loader, output_dir, case_name=None, format="binary")
copies the selected mesh to a separate dataset. EnSightWriter.from_case also
accepts a .case path with explicit part/association selection. The default output
.case name is the source filename stem; the chosen part retains its name and is
assigned output ID 1. New or empty output directories are required and source
files are protected against replacement.

write_static writes static scalar/vector/tensor fields; write_snapshot accepts a
complete field dictionary and physical time; write_series accepts time-last
arrays and streams their snapshots. Complex values are split into named real and
imaginary fields. Both ASCII and binary serialization support nsided/nfaced.
VTK reverses ASCII nsided winding during readback; binary preserves it exactly.
This affects surface normals but not field association, polygon membership or area.

The writer is covered by self-contained unit tests, including distinct per-element
and per-component values, conservation, geometry connectivity, a real POD workflow,
portable output after source deletion, and recovery from an interrupted staged
write. Separate ParaView reader checks cover volume, patch and plane modes and
reconstructions in both formats. See docs/source/ensight_io.rst for working examples.
