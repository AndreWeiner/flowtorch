Out-of-core and distributed SVD/SPOD
====================================

flowTorch can construct a flat physical state vector lazily and compute its
SVD without materializing the complete data matrix.  The same public API covers
in-memory tensors, shared-memory out-of-core execution, and spatially
distributed execution.  PAMSPOD can either construct that SVD or reuse an
existing one.

State-vector layout and normalization
-------------------------------------

Every snapshot must have the same spatial shape.  Component dimensions follow
the spatial dimensions in each field.  The state vector stores complete scalar
components contiguously, in field order.  One normalization scalar is required
per physical field, and that scalar is shared by all components of vector or
tensor fields::

    from flowtorch.analysis import (
        DataloaderStateVectorSource,
        FieldSpec,
        PAMSPOD,
        SVD,
    )

    fields = (
        FieldSpec("U", component_shape=(3,), component_names=("u", "v", "w")),
        FieldSpec("p"),
    )
    source = DataloaderStateVectorSource(
        loader,
        fields,
        times=loader.write_times,
        normalization={"U": 20.0, "p": 100_000.0},
        spatial_weight=loader.weights,
    )

The factors above produce ``[U_x/20, U_y/20, U_z/20, p/100000]`` in the flat
state.  Use ``normalization="rms"`` instead to fit one weighted fluctuation RMS
per field during the first decomposition.  Fitted values are frozen and reused
by :meth:`~flowtorch.analysis.state_vector.DataloaderStateVectorSource.with_times`,
so appended data do not silently change the definition of the state vector.

Shared-memory, out-of-core execution
------------------------------------

Omitting ``execution`` selects a single-process SVD.  PyTorch may still use its
CPU thread pool or one GPU, while flowTorch processes spatial chunks::

    svd = SVD(
        source,
        rank=40,
        subtract_mean=True,
        spatial_batch_size=100_000,

    )

``spatial_batch_size`` controls how many values of the first spatial dimension
are processed together. Every spatial request includes all snapshots in the
source supplied to the constructor or ``update()``. There is no snapshot batch
size parameter. The caller selects temporal windows by supplying sources for
those windows and appending them with ``update()``.

Each incoming temporal batch must be loadable using the selected dataloader.
For a loader that reads full snapshots before slicing, the complete batch must
fit its loading memory. A directly sliceable format only needs to load the
requested spatial portion. The TSQR working block contains
``spatial_batch_size * source.n_snapshots`` values, including field components
and remaining spatial dimensions. This assumption does not cause all snapshots
to be cached automatically.

CPU-side prefetching
~~~~~~~~~~~~~~~~~~~~

Source-backed SVD prepares chunks sequentially by default. Use
``SVD(source, prefetch_chunks=True)`` to prepare one CPU chunk ahead. The source's
``iter_chunks(spatial_slice, spatial_batch_size, snapshot_slice=None,
prefetch=False)`` also exposes this traversal to other consumers. Items contain
a spatial slice, a packed tensor, and raw spatial weights. Close an iterator
when stopping early, for example with ``contextlib.closing``.

PyTorch Dataset/DataLoader dispatch runs in one background thread with a
bounded one-item queue. It uses no worker processes, so it does not copy meshes,
resident bases, or open file handles into subprocesses. Reader calls are serial
within each iterator. Do not concurrently mutate or read the same loader from
another iterator. The thread shares PyTorch's configured CPU thread pool; it
does not change process-wide thread limits or promise a dedicated CPU core.

Upcoming data stays in ordinary CPU RAM. Device transfers occur only when the
SVD consumes it, with no pinned buffers, separate CUDA streams, or GPU-ahead
queue. Loader-backed sources configured for CUDA provide host-reading hooks.
Custom GPU-only sources remain synchronous unless they implement ``read_cpu``
and ``read_weight_cpu``. Wrappers must forward those hooks to retain the
underlying source's host-reading capability.

Update passes prepare disk-backed basis, mean/weight, and staged-batch chunks
using the same helper. GPU-resident bases are accessed directly by the consumer;
only incoming host data is prepared ahead. Output basis writes remain
synchronous. Each distributed rank prepares only its own spatial partition.
Loading exceptions propagate, and shutdown drains the outstanding read before
closing readers or deleting temporary basis files.

Memory estimates include extra CPU preparation buffers. GPU runs separately
check available host RAM, shared by hostname. Full-snapshot loader temporaries
are still not independently bounded; the temporal batch must remain loadable
while the queued chunk is retained. Prefetching overlaps work but does not
increase storage bandwidth or guarantee a speedup.

Memory-bounded spatial chunks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``spatial_batch_size`` is omitted, SVD targets 256 MiB of incoming
snapshot data per rank. The spatial count is the target divided by the number
of packed scalar components, incoming snapshots, and bytes per scalar. For
three-component fp32 data and 20 snapshots this is about 1.12 million cells.
The local partition and estimated memory capacity can reduce this count.
There are no timing trials, search parameters, or tuning results.

The automatic capacity estimate includes the maximum retained rank, temporary
processing arrays, and the extra host buffers when prefetching is enabled.
Memory-backed bases additionally reserve space for resident basis updates.
If rank is automatic, the initial snapshot count supplies the construction
estimate; update checks account for the actual growing basis and temporal
factors. Updates reuse the selected size and only shrink it for memory safety.

By default, the estimate uses 60% of available device memory, respecting
Linux cgroup v2 limits. CPU RAM and GPU host-prefetch buffers are divided
among ranks sharing a hostname. CUDA assumes a separate GPU per rank.
The tightest rank determines the common spatial size. A small metadata read
is needed to establish dtype and packed shape; deferred RMS normalization is
fitted first using bounded traversal.

An explicit ``spatial_batch_size`` overrides the automatic construction
selection. ``memory_budget_bytes`` supplies an explicit per-rank memory
budget, bypassing automatic memory probing, sharing, and the 60% safety factor.
For example, four ranks sharing 64 GiB may each be given 10 GiB::

    svd = SVD(source, rank=100, memory_budget_bytes=10 * 1024**3,
              execution=execution)

Or select the spatial size explicitly::

    svd = SVD(source, rank=100, spatial_batch_size=2_000_000,
              prefetch_chunks=True, execution=execution)

Memory bounds are estimates, not allocator limits. A loader may materialize
whole snapshots or allocate parsing buffers independently of the requested
chunk. Incoming temporal batches must remain loadable. Node rank/thread
placement is configured by the application and launcher, not by SVD.

The built-in HDF5 and FOAM NumPy dataloaders read spatial slices directly.
Other dataloaders inherit a compatible fallback that loads a snapshot before
slicing it; custom loaders should override ``load_snapshot_slice`` to obtain
true out-of-core I/O.

Left singular vectors and physical reconstructions remain lazy.  Flat results
are normalized; splitting restores the original component shapes and physical
units::

    for spatial_slice, result in svd.U.iter_chunks(split=True):
        velocity_modes = result["U"]  # (local points, 3, rank)
        pressure_modes = result["p"]  # (local points, rank)

    reconstruction = svd.reconstruct(rank=20)
    for spatial_slice, fields in reconstruction.iter_chunks(split=True):
        write_fields(spatial_slice, fields)

Call ``materialize_local`` only when the rank-local result fits in memory.

Spatially distributed execution
-------------------------------

Initialize ``torch.distributed`` before constructing
:class:`~flowtorch.analysis.statistics.DistributedExecution`.  Every process
opens the same logical source and retains the full time dimension; flowTorch
partitions only the first spatial dimension.  Rank-local QR factors are merely
``n_snapshots`` square, and are merged collectively before the small SVD::

    import os
    import torch
    import torch.distributed as dist
    from flowtorch.analysis import DistributedExecution, SVD

    dist.init_process_group(backend="nccl")  # use "gloo" on CPUs
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    execution = DistributedExecution(root_rank=0)
    source = DataloaderStateVectorSource(
        loader,
        fields,
        normalization={"U": 20.0, "p": 100_000.0},
        device=f"cuda:{local_rank}",
    )
    svd = SVD(
        source,
        rank=40,
        spatial_batch_size=100_000,

        execution=execution,
    )

    # This is collective. Only root_rank receives the complete CPU tensor.
    global_modes = svd.U.gather(root_rank=0)
    dist.destroy_process_group()

Use Gloo for portable CPU execution on Linux, macOS, and Windows.  NCCL is for
NVIDIA GPUs and MPI is available when the installed PyTorch build includes it.
Backend and launcher availability still depends on the local PyTorch build and
cluster configuration.

A minimal multi-node Slurm launch for one process per GPU is::

    #!/bin/bash
    #SBATCH --nodes=2
    #SBATCH --ntasks-per-node=4
    #SBATCH --gpus-per-node=4
    #SBATCH --cpus-per-task=8

    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_PORT=29500

    srun python distributed_spod.py

This is hybrid parallelism: distributed ranks partition space across nodes and
GPUs, while dataloader work and PyTorch kernels can use the CPUs assigned to
each process.  Configure the script's global rank, local rank, and world size
from Slurm's ``SLURM_PROCID``, ``SLURM_LOCALID``, and ``SLURM_NTASKS`` when the
site's ``srun`` integration does not populate the variables expected by
``init_process_group``.

Updating a decomposition
------------------------

Construct a compatible source for new times and append it through the same API
on every rank::

    new_source = source.with_times(new_times)
    svd.update(new_source)

Both centered and uncentered decompositions update using the retained local
POD basis. Previously processed snapshots are not read or retained. For a
centered decomposition, each rank stores its local mean; an update combines
means with snapshot-count weighting and includes the rank-one mean-shift
correction in the factorization. Spatial weights and normalization stay fixed.
The new batch is read once per spatial chunk and temporarily stored for the
projection, residual factorization, and basis rotation passes.

Updates reuse the initial spatial batch size without timing trials. Before
allocating update factors, all ranks report current available memory (or their
explicit per-process budgets). A conservative estimate reserves compact
factors, new temporal factors, basis/residual chunk workspace, and replacement
basis plus staged-batch storage when using memory mode. CPU memory is shared
by hostname as during initial selection. The tightest rank bounds the common size;
it is reduced only if needed and never increased by this check. Insufficient
capacity for one spatial value raises an error before reading new snapshots.
Explicit budgets are retained in each rank's checkpoint shard. Full-snapshot
loader buffers remain the caller's responsibility; this estimate is not a hard
memory limit.

For source-backed TSQR, an explicit ``rank`` is a maximum across updates.
``svd.rank`` reports the currently retained dimension and ``svd.maximum_rank``
reports the requested ceiling. For example, ``rank=100`` allows a basis from an
initial ten-snapshot batch to grow as further batches arrive, up to 100 modes.
Numerically null directions are omitted. With ``rank=None``, singular value
hard thresholding chooses the rank after initialization and each update.
Truncation after intermediate batches is approximate and may depend on batch
size and order. With all resolved directions retained, the result agrees with
a full centered or uncentered SVD up to numerical error. Tensor-input rank
semantics are unchanged.

Basis storage
~~~~~~~~~~~~~

Source-backed TSQR retains each rank's local physical left singular vectors.
Disk storage is the default::

    svd = SVD(
        source,
        rank=100,
        subtract_mean=True,
        basis_storage="disk",
        basis_directory="/scratch/my-job",  # existing directory, preferably local
        spatial_batch_size=100_000,
    )

``basis_storage="memory"`` retains local basis chunks in RAM (or on the source
device). ``basis_directory`` selects the parent for owned temporary directories;
when omitted, Python's system temporary directory is used. Disk arrays are
NumPy files, accessed through memory mapping and copied only for requested
spatial rows. Each update writes a replacement basis before releasing the old
one. Call ``svd.close()`` when finished to release temporary files; saved
checkpoints are independent. Temporary directories are also cleaned up when
their storage objects are collected.

The local basis contains roughly ``local_state_size * rank`` values, plus a
local mean and any weights. Memory storage also retains the local new batch
during an update, and old and replacement bases coexist until the update
succeeds. Disk updates similarly require space for both bases and the staged
new batch until completion. Disk storage keeps these arrays out of persistent tensor
memory, but filesystem cache still uses reclaimable host RAM. Chunk buffers,
QR workspace, compact update matrices, and the replicated temporal factor
require additional memory. Initial construction needs an additional snapshot
pass to form and store the basis after TSQR.

Saving and loading
------------------

Source-backed checkpoints are self-contained directories::

    svd.save("svd-checkpoint")
    restored = SVD.load(
        "svd-checkpoint",
        execution=execution,
        map_location=f"cuda:{local_rank}",
        basis_storage="disk",  # or "memory"
        basis_directory="/scratch/my-job",
    )

They contain singular values, temporal factors, retained physical basis chunks,
local means and weights, layout and normalization metadata, snapshot count,
and current and maximum rank. No original dataloader or snapshots are needed
for reconstruction, mode access, or subsequent updates. ``source`` may be
supplied to ``load`` only to validate layout and normalization.

Saving streams local chunk files rather than gathering the complete basis.
In distributed execution every rank calls ``save`` on the same shared path;
each writes its own shard and rank zero writes completion metadata last.
Loading currently requires the same rank count and spatial partition, but can
change storage mode, processing batch sizes, device, and ``prefetch_chunks``.
The checkpoint retains the prefetch setting when no override is supplied. An existing checkpoint
directory is not overwritten. This checkpoint format has no compatibility with
previous versions. Tensor-backed decompositions continue to use a single
checkpoint file containing their retained factors.

PAMSPOD and an existing SVD
---------------------------

PAMSPOD accepts the same lazy source directly::

    spod = PAMSPOD(
        source,
        dt=0.001,
        rank=40,
        spatial_batch_size=100_000,

        execution=execution,
    )

When an SVD already exists, reuse its temporal factors without reloading the
physical snapshots::

    spod = PAMSPOD.from_svd(
        svd,
        dt=0.001,
        adaptive=True,
        keep_n_modes=3,
    )

The reduced PAMSPOD calculation is replicated on the ranks; large spatial modes
and reconstructions stay distributed and lazy.  For source-backed results,
``spod.modes`` has flat state-first order ``(state, frequency, mode)``.
``get_mode``, ``mode_reconstruction``, and ``partial_reconstruction`` all return
lazy results that support ``iter_chunks(split=True)`` and restore velocity,
pressure, and other original fields.
