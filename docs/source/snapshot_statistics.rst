Snapshot statistics at scale
============================

The :func:`flowtorch.analysis.snapshot_statistics` function computes any
combination of moment fields, fraction dependency, linear trends, spatial
time series, and a spatiotemporal histogram. With fixed histogram edges or a
fixed range, all selected collectors share one traversal of the snapshot
source. An automatically ranged histogram needs a second traversal to count
values after determining the global range.

Serial use
----------

An indexed callable keeps only a batch of snapshots in memory. Its ``start``
and ``stop`` arguments refer to the complete sequence::

   import torch as pt

   from flowtorch.analysis import snapshot_statistics

   def load_snapshots(start: int, stop: int) -> pt.Tensor:
       # Return spatial_shape + (stop - start,) from storage.
       return dataset[..., start:stop]

   domain_mask = load_spatial_mask().to(dtype=pt.bool)
   result = snapshot_statistics(
       load_snapshots,
       n_snapshots=100_000,
       batch_size=16,
       fractions=(0.1, 0.25, 0.5, 0.75, 1.0),
       spatial_mask=domain_mask,
       histogram_range=(-5.0, 5.0),
   )

``batch_size`` is the maximum number of snapshots loaded by one process at a
time. It does not change a data fraction or the numerical definition of a
statistic. Smaller batches reduce peak device and host memory; larger batches
usually improve I/O throughput.

``spatial_mask`` is a boolean tensor matching or broadcasting to the spatial
shape. Only true locations contribute to reductions. Moment and trend fields
keep their original spatial shape, with excluded locations set to ``NaN``.
The mask does not include a snapshot dimension, so its size is independent of
both the number of snapshots and ``batch_size``. A dense ``torch.bool`` mask
uses one byte per spatial location.

Distributed and hybrid use
--------------------------

The public functions keep the same interface for serial and distributed
execution. Passing :class:`flowtorch.analysis.DistributedExecution` partitions
the global snapshot axis into balanced, disjoint, contiguous ranges. The
callback still receives global indices, and only the configured root rank
returns a result. Initialize the process group before the call::

   import os

   import torch as pt
   import torch.distributed as dist

   from flowtorch.analysis import DistributedExecution, snapshot_statistics

   rank = int(os.environ["SLURM_PROCID"])
   world_size = int(os.environ["SLURM_NTASKS"])
   local_rank = int(os.environ["SLURM_LOCALID"])
   os.environ["RANK"] = str(rank)
   os.environ["WORLD_SIZE"] = str(world_size)

   dist.init_process_group("gloo", init_method="env://")
   pt.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))

   try:
       result = snapshot_statistics(
           load_snapshots,
           n_snapshots=100_000,
           batch_size=16,
           histogram_range=(-5.0, 5.0),
           execution=DistributedExecution(root_rank=0),
       )
       if result is not None:
           pt.save(result, "statistics.pt")
   finally:
       dist.destroy_process_group()

This is hybrid parallelism: Slurm starts multiple distributed processes, and
each process uses PyTorch's intra-operation CPU thread pool. A corresponding
CPU job can be submitted as follows::

   #!/bin/bash
   #SBATCH --job-name=flowtorch-statistics
   #SBATCH --nodes=2
   #SBATCH --ntasks-per-node=2
   #SBATCH --cpus-per-task=16
   #SBATCH --time=01:00:00

   export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
   export MASTER_PORT=29500
   export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
   export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
   export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

   srun --cpu-bind=cores --distribution=block:block python -u statistics_job.py

For GPUs, request one task per GPU, select the NCCL backend, and set the CUDA
device before initializing the process group::

   #SBATCH --nodes=2
   #SBATCH --ntasks-per-node=4
   #SBATCH --gpus-per-task=1
   #SBATCH --cpus-per-task=8

   srun --cpu-bind=cores python -u statistics_job.py

In ``statistics_job.py``, use ``pt.cuda.set_device(0)`` when Slurm exposes one
GPU to each task, or ``pt.cuda.set_device(local_rank)`` when every node-local
GPU remains visible. Then initialize NCCL with ``init_method="env://"``.
Snapshot batches, ``spatial_weight``, and ``spatial_mask`` must be on the
selected CUDA device. The MPI backend can be selected in the same interface
when PyTorch was built from source with MPI support; launch and environment
details depend on the cluster's MPI and Slurm integration.

``batch_size`` remains per process in every backend. Therefore, a job with
four ranks and ``batch_size=16`` can process up to 64 snapshots concurrently,
while each rank holds no more than 16 snapshots from its assigned range.
