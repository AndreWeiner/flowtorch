# Guidance for agents working on flowTorch

Read `CONTRIBUTING.md` for workflow and coding conventions. Use `tox.ini`,
`pyproject.toml`, and `pytest.ini` for current checks and test configuration.
Verify older notes against the current implementation before assuming an API exists.

## Design and scope

- Prefer small, explicit designs with optional behavior and clear overrides.
  Extend existing loaders, writers, and analysis APIs rather than adding parallel paths.
- Preserve unrelated working-tree changes. When an isolated commit is requested,
  include only the requested changes and report its identifier.
- Use vectorized tensor operations or compiled geometry operations for large meshes;
  avoid Python loops over individual points or cells where bulk operations suffice.

## Numerical and loader contracts

- Keep point/cell association consistent across fields, coordinates, weights, and
  exported topology. Select association once at the loader boundary when supported.
- Geometric integration weights must be raw, finite, positive measures. SVD applies
  square roots internally; do not pre-transform loader weights.
- Validate geometry rather than concealing invalid measures. Legacy four-node
  Tecplot surface connectivity may require an explicit quad cell-type override.
- Test weighted orthogonality, reconstruction, normalization, and equivalence of
  supported computation paths as relevant. Compare subspaces or reconstructions
  when singular-vector signs or degenerate bases are arbitrary.

## Large-data and distributed performance

- Trace loader reads and materialization before claiming out-of-core behavior.
  Spatial slicing may load full snapshots; request batch size does not bound peak memory.
- Measure loading, materialization, casts, and synchronized device transfers alongside
  computation. Separate cache effects and tuning overhead from steady-state timing.
- Respect explicit chunk and memory settings. Bound automatic trials and account for
  temporary allocations, the tightest rank's capacity, and ranks sharing host memory.
- Coordinate distributed tuning and broadcast decisions. Validate numerical results
  alongside speed and memory; record workload, hardware, dtype, device, and rank count.

## Reproducibility and validation

- For notebook or release audits, verify `flowtorch.__file__`, version/commit, and local
  changes. Notebooks may import a different checkout.
- Trace the execution path actually exercised. Synthetic agreement cannot establish
  equivalence on unavailable original datasets or explain real-data differences alone.
- Distinguish implemented behavior, proposed designs, and unverified results.
- Run relevant tests and required checks. External datasets use `FLOWTORCH_DATASETS`;
  report unavailable data or dependencies and any integration checks not run.
- Distinguish MPI, ParaView, or local-network restrictions from algorithm failures.
- Verify exported files with an appropriate reader or renderer when topology or visual
  layout matters. Report validation performed and remaining limitations concisely.
