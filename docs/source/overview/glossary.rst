Glossary
========

These terms describe the data formats, numerical methods and workflows used in
flowTorch. Start with :term:`snapshot`, :term:`field association` and
:term:`state vector` when preparing data for modal analysis.

.. glossary::
   :sorted:

   API
      Application programming interface: the public classes, functions and
      arguments through which a program uses flowTorch.

   AMSPOD
      Adaptive multitaper spectral proper orthogonal decomposition. The
      ``AMSPOD`` class estimates frequency-dependent modes directly from a data
      matrix; it also supports a fixed number of tapers.

   chunk
      A subset of an array processed or stored as a block. A requested
      :term:`spatial chunk` and an HDF5 storage chunk need not have the same size.

   CPU
      Central processing unit. CPU computations use host memory and may run
      across several threads.

   DFT
      Discrete Fourier transform: a representation of sampled data in discrete
      frequency bins. flowTorch's ``DFT`` also exposes spatial Fourier modes
      and weighted spectra.

   DMD
      Dynamic mode decomposition: a data-driven approximation of evolution
      between snapshots using a linear operator. Its modes have associated
      oscillation frequencies and growth or decay rates; interpreting those
      rates physically requires appropriate data and modelling assumptions.

   dtype
      Numeric data type, such as Float32 or Float64. Storage precision and
      analysis precision can differ. Casting Float32 values to Float64 does
      not recover digits discarded earlier.

   EnSight Gold
      A visualization format containing a case descriptor, geometry and field
      files. flowTorch supports reading and writing supported ASCII and
      C-binary layouts. See :doc:`../ensight_io`.

   field association
      The mesh entities at which field values are defined: points (nodes) or
      cells. Coordinates used for analysis, integration weights and field
      shapes must match the chosen association. XDMF calls these locations
      ``Node`` and ``Cell``.

   frequency bin
      A discrete frequency represented by a spectral estimator. Its index is
      distinct from its physical frequency. Frequencies have inverse units of
      the snapshot time labels, so time in seconds gives frequency in Hz.

   geometry reuse
      Referencing one geometry container from several field or result files.
      The versioned HDF5 workflow validates geometry identity and zone order
      before reuse; matching array sizes alone is insufficient.

   GPU
      Graphics processing unit, used for supported tensor computations through
      PyTorch. Device transfers and host-memory buffers still contribute to
      the cost of a workflow.

   HDF5
      Hierarchical Data Format, version 5: a binary container for arrays and
      metadata. The versioned flowTorch format can separate reusable geometry
      from time-dependent fields and supports direct spatial field slices.
      See :doc:`../hdf5_workflow`.

   HODMD
      Higher-order dynamic mode decomposition. It extends the observed state
      with time-delay embeddings before fitting the dynamics. See the
      :doc:`algorithm example <../notebooks/hodmd>`.

   integration weights
      Raw geometric measures defining a discrete spatial inner product, such
      as cell areas, cell volumes or their conservative nodal lumping. For a
      diagonal weight matrix W, the inner product is x* W y, where * denotes
      conjugate transpose. The SVD applies square roots internally; supply raw,
      finite, positive measures rather than their square roots.

   mode
      A spatial pattern produced by a decomposition. Its normalization and
      interpretation depend on the method. POD modes describe variance in a
      chosen inner product; SPOD modes are frequency-dependent and generally
      complex. A mode by itself does not establish a physical mechanism.

   MPI
      Message Passing Interface: a standard for communication between
      processes. It is distinct from a particular PyTorch distributed backend;
      available execution paths depend on the installation and process group.

   MSSA
      Multivariate singular spectrum analysis: an analysis of time-delay
      embedded multivariate data, used here for tasks such as noise reduction.
      ``PMSSA`` performs the analysis after POD compression.

   normalization
      Scaling physical fields before assembling a state vector. flowTorch's
      state-vector interface uses one factor per field, shared by its vector
      or tensor components. Physical-field result export undoes this scaling.
      Field normalization is distinct from geometric integration weighting.

   out-of-core
      Processing data or intermediate factors without keeping the entire
      dataset in RAM. The claim depends on the complete read and computation
      path: a slicing interface may still load full snapshots before slicing.
      Geometry, caches and work arrays can remain resident.

   PAMSPOD
      The POD-compressed implementation of AMSPOD. It computes SPOD in the
      subspace of a retained SVD/POD basis, then projects selected results back
      to physical space. ``PAMSPOD.from_svd`` reuses an existing SVD.

   ParaView
      A scientific visualization application used to inspect converted fields,
      spatial modes and animations. Open the XDMF descriptor for flowTorch's
      HDF5 exports, keeping the referenced HDF5 files accessible.

   PCA
      Principal component analysis. For appropriately centred data, its
      principal directions are closely related to POD modes computed by SVD;
      scaling and the selected inner product determine the comparison.

   PDFT
      POD-compressed discrete Fourier analysis. The ``PDFT`` class applies
      Fourier analysis to POD temporal factors and projects results back into
      the retained spatial subspace. See :doc:`../dft_pdft`.

   phase animation
      A visualization of a complex mode's oscillation, commonly evaluated as
      A times the real part of mode times exp(i phase). A is a chosen amplitude.
      It is distinct from reconstructing the original flow using expansion
      coefficients and any removed mean. Phase frames may also be constructed
      directly in ParaView from real and imaginary mode fields.

   PIV
      Particle image velocimetry: an experimental method that estimates flow
      velocities from images of tracer particles. See the
      :doc:`worked PIV example <../notebooks/piv_analysis>`.

   PLT
      Tecplot's binary data-file extension. The extension alone does not specify
      the binary layout. The current native flowTorch backend supports selected
      TDV71 and TDV112 FE-block layouts; other layouts need a different reader.

   POD
      Proper orthogonal decomposition: an orthogonal representation that orders
      modes by their contribution to the squared data norm in a chosen spatial
      inner product. It can be computed by SVD, with centering and geometric
      weights chosen for the intended interpretation.

   provenance
      A record of input descriptions, settings, dependencies and software
      environment associated with an operation. It helps reproduce a result;
      metadata capture is not a substitute for validation or data archiving.
      See :doc:`../provenance`.

   PSP
      Pressure-sensitive paint: an optical technique for measuring surface
      pressure. The :doc:`interactive PSP example <../notebooks/ipsp_explorer>`
      demonstrates working with zoned measurement data.

   PyTorch
      The tensor-computation library used by flowTorch for numerical arrays,
      linear algebra and supported CPU/GPU computations.

   rank truncation
      Retaining a subset of decomposition components to reduce dimension or
      filter noise. The retained SVD rank affects subsequent POD-compressed
      analyses; a small rank or a short record needs convergence checks for
      scientific use.

   reconstruction
      Combining spatial modes and expansion coefficients to approximate fields.
      A reconstruction can be partial, for example using selected frequencies
      or modes. Restoring a removed mean is an explicit choice.

   ROM
      Reduced-order model: a representation that predicts or approximates a
      system using fewer degrees of freedom than the original description.
      flowTorch provides encoders, decoders and regressors for such workflows.

   snapshot
      Field data at one sampling time, potentially containing several scalar,
      vector or tensor fields. A sequence used for modal analysis must preserve
      its spatial layout and field association.

   spatial chunk
      A contiguous interval along the first spatial array dimension. Native
      Tecplot and HDF5 field readers can load that interval directly. Requesting a
      small chunk does not by itself bound total process memory.

   SPOD
      Spectral proper orthogonal decomposition: a frequency-dependent modal
      decomposition of the cross-spectral density for statistically stationary
      data. It identifies modes ordered by spectral energy in the chosen inner
      product. flowTorch implements multitaper estimation through AMSPOD and
      its POD-compressed variant PAMSPOD.

   state vector
      A consistent arrangement of selected physical field components into one
      vector. ``StateVectorLayout`` records field order, component shapes and
      normalization; spatial results can be split back into physical fields.

   SVD
      Singular value decomposition: a factorization A = U Sigma V* into
      orthonormal singular vectors and nonnegative singular values. flowTorch
      also supports geometrically weighted, chunked and distributed paths.
      With weights, physical spatial modes are orthonormal in the weighted
      inner product.

   TDV71
      A legacy Tecplot binary layout identified by ``#!TDV71`` followed by a space.
      flowTorch's
      native reader supports selected finite-element block layouts for
      triangles, quadrilaterals, tetrahedra and bricks, preserving source array
      precision and allowing direct field slices.

   TDV112
      A newer Tecplot binary layout identified by ``#!TDV112``. It has different
      header and data records from TDV71. The native backend supports selected
      linear FE-block layouts, including shared and passive variables and
      point/cell association; ``backend="auto"`` selects it for this signature.

   temporal batch
      A subset of snapshot times processed together. Source-backed SVD can
      append batches through ``SVD.update``. Every batch must preserve the
      geometry, field layout and normalization; temporal batching alone does
      not limit the size of a full spatial snapshot.

   XDMF
      eXtensible Data Model and Format: an XML descriptor that describes mesh
      topology, geometry, field association and time steps while referencing
      array data, commonly stored in HDF5. Open ``.xdmf`` or ``.xmf`` descriptors
      in ParaView to view the corresponding data.

   zone
      A named mesh or measurement region. Selected zones retain a deterministic
      order when combined for analysis. Zone-local interface points are not
      automatically merged, which matters when interpreting integration norms.
