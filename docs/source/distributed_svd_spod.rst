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
CPU thread pool or one GPU, while flowTorch limits data loading to spatial and
snapshot batches::

    svd = SVD(
        source,
        rank=40,
        subtract_mean=True,
        spatial_batch_size=100_000,
        snapshot_batch_size=16,
    )

``spatial_batch_size`` controls how many values of the first spatial dimension
are processed together.  ``snapshot_batch_size`` controls each request to the
dataloader.  The TSQR accumulator still holds one spatial batch across all time
steps, because the time dimension is the skinny dimension.  Its dominant
working storage is therefore proportional to
``spatial_batch_size * n_snapshots`` rather than the global spatial size.

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
        snapshot_batch_size=16,
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

For an uncentered SVD, this performs an incremental distributed update and does
not reread the original snapshots.  It is exact when the original SVD retained
its complete rank and otherwise updates the retained approximation.  A centered
update recomputes the decomposition out of core over the composite source so
the changed global mean is exact.  Layout, spatial shape, normalization,
weights, rank order, and collective call order must agree across ranks.

Saving and loading
------------------

Checkpoints use an explicit, versioned dictionary rather than serializing the
Python SVD or dataloader objects::

    svd.save("svd.pt")

    restored = SVD.load(
        "svd.pt",
        source=source,
        execution=execution,
        map_location=f"cuda:{local_rank}",
    )

A tensor-backed checkpoint contains the complete decomposition and can be
loaded with ``SVD.load("svd.pt")``.  A source-backed checkpoint stores compact
singular and temporal factors, state layout, normalization, and batch metadata,
but deliberately omits the tall left modes and the dataloader.  Loading it
therefore requires a compatible source; no snapshots are read until a lazy
spatial result is evaluated.  A source recreated with ``normalization="rms"``
receives the saved, frozen factors during loading without rescanning snapshots.

In distributed execution, ``save`` is collective: the configured root writes
one checkpoint on a shared filesystem and all ranks synchronize.  Every rank
then calls ``load`` with its local execution configuration.  On GPUs,
``map_location`` should match the device used by that rank's source.

PAMSPOD and an existing SVD
---------------------------

PAMSPOD accepts the same lazy source directly::

    spod = PAMSPOD(
        source,
        dt=0.001,
        rank=40,
        spatial_batch_size=100_000,
        snapshot_batch_size=16,
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
