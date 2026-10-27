Reproducibility metadata
========================

Selected analysis and ROM entry points automatically capture their arguments,
including defaults, effective settings, input descriptions, and the numerical
software environment. Data loaders record construction settings; state-vector
sources describe field/time selection and normalization. Metadata collection
never reads array contents, hashes files, or materializes lazy snapshots.

.. code-block:: python

   from flowtorch.analysis import SVD

   svd = SVD(data, rank=20, mode="auto", subtract_mean=True)
   svd.dump_settings("settings.json")
   svd.dump_versions("versions.json")
   svd.dump_provenance("provenance.json")

``settings`` contains initial call arguments. ``versions`` contains versions at
that call, rather than at export time. ``provenance`` includes chronological
records and current effective settings. Returned dictionaries are independent
copies. Requested and resolved ranks, methods, devices and chunk sizes are kept
separate. Each operation snapshots its dependencies, so later mutations do not
rewrite completed history.

SVD ``update`` records new input batches and effective settings. Existing SVD
checkpoints preserve captured records, including source-backed directory
checkpoints. Loading older checkpoints returns empty historical metadata; the
current factor settings remain available. Loading does not replace the original
computation environment. ROM ``fit``, ``fit_regressor`` and ``fine_tune`` record
separate stages. Fine-tuning includes resolved optimization stages and stopping
outcomes. ROM metadata is exported separately from PyTorch ``state_dict``;
existing tensor-only state dictionaries and optimizer handling are unchanged.

Visualization recording is opt-in and preserves existing return types:

.. code-block:: python

   from flowtorch.provenance import ProvenanceSession, dump_figure_provenance
   from flowtorch.visualization import plot_scalar_fields

   with ProvenanceSession() as session:
       figure, axes = plot_scalar_fields([field], ["pressure"], 1, 1)
   session.dump_provenance("plot_calls.json")
   # Capture supported final axes settings after any manual adjustments.
   dump_figure_provenance(figure, "figure.json", session)

The session records public visualization functions, including LIC configuration.
Final figure descriptions cover size, DPI, axes, and image colour limits; they
are not complete Matplotlib artist serialization. Standalone decorated functions
outside a session perform no metadata collection.

Extension points
----------------

.. code-block:: python

   from flowtorch.provenance import ProvenanceMixin, tracked

   class MyAnalysis(ProvenanceMixin):
       @tracked
       def __init__(self, data, tolerance=1.e-5):
           self.tolerance = tolerance
           # Perform computation here.

       def _effective_settings(self):
           return {"tolerance": self.tolerance}

Use ``@tracked`` on selected computational methods, and underneath
``@classmethod`` for factories that return a tracked object. Nested constructor
calls on the same instance are suppressed. Failed operations are marked failed;
they are not reported as completed results. Loader subclasses inherit constructor
capture through the existing ``Dataloader`` boundary. Effective hooks must read
already available state, not trigger loading or collectives.

Limits
------

Tensor descriptions contain shape, dtype and device, not data or checksums.
Dataset paths are identifiers, not proof of unchanged contents. Custom callable
identifiers and unsupported objects explicitly indicate that their configuration
was not recorded. Full random-generator states are not stored. Seeds supplied
to tracked methods are recorded, but seeds alone do not guarantee replay.

Optional distribution versions are looked up without importing their libraries.
External ParaView process versions are not currently queried. CUDA hardware and
cuDNN versions are captured only if CUDA is already initialized. Git identity is
best-effort and includes checkout dirty status, not a patch of local changes.
Records are diagnostic provenance, not an executable workflow or a guarantee of
bitwise equivalence across library versions or platforms.

JSON export is local and introduces no distributed collective. Select one rank
to write a shared JSON path, or use distinct per-rank paths. Distributed SVD
checkpoints retain their existing collective and shard behavior; their shared
metadata describes the writing rank rather than every host's environment.
