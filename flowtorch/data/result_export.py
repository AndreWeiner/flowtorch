"""Chunked physical-field and SPOD export against reusable HDF5 geometry."""

from pathlib import Path
import itertools
import json
from typing import Any, cast

import numpy as np
import torch

from .hdf5_file import HDF5Writer
from ._hdf5_schema import select_names


def export_fields(
    result,
    output_file,
    geometry_file,
    *,
    labels=None,
    zones=None,
    geometry_id=None,
    field_dtype=torch.float32,
    compression=None,
    compression_level=None,
    shuffle=None,
    representation="real_imag",
    metadata=None,
    overwrite=False,
    append=False,
    staging_directory=None,
):
    """Export a serial, chunk-readable StateVectorResult in physical units.

    Trailing result dimensions are flattened into explicitly labelled frames.
    ``result.iter_chunks(split=True)`` restores component shapes and scaling.
    Complex values become real/imaginary, or magnitude/phase field pairs.
    The caller must supply the original zone order and geometry identity when
    known; dimensions alone cannot establish physical correspondence.
    compression (None, "gzip", "zstd"), compression_level and shuffle configure
    lossless field storage. Shared geometry is never recompressed. Zstd needs
    the optional compression extra on the writing/root process.
    append=True extends a complete split dataset and requires explicit new labels.
    Omitted compression/level/shuffle inherit existing fields; conflicts and
    overwrite=True are rejected. Frame metadata describes each export batch;
    existing file-level result metadata remains unchanged.
    """
    import torch.distributed as dist
    from ._hdf5_compression import compression_settings

    # Root performs plugin validation for a collective export, broadcasting errors.
    if (
        result.execution is not None
        and dist.is_initialized()
        and dist.get_world_size(result.execution.process_group) > 1
    ):
        return _export_distributed(
            result,
            output_file,
            geometry_file,
            staging_directory,
            dict(
                labels=labels,
                zones=zones,
                geometry_id=geometry_id,
                field_dtype=field_dtype,
                compression=compression,
                compression_level=compression_level,
                shuffle=shuffle,
                representation=representation,
                metadata=metadata,
                overwrite=overwrite,
                append=append,
            ),
        )
    if append and overwrite:
        raise ValueError("append and overwrite are mutually exclusive")
    if not append:
        compression_settings(
            compression, compression_level, False if shuffle is None else shuffle
        )
    if append and labels is None:
        raise ValueError(
            "Appending exported results requires explicit new frame labels"
        )
    if representation not in ("real_imag", "magnitude_phase"):
        raise ValueError("Expected real_imag or magnitude_phase representation")
    if len(result.layout.spatial_shape) != 1:
        raise ValueError("Unstructured exports require one spatial dimension")
    count = result.layout.spatial_shape[0]
    if result.local_spatial_slice.indices(count) != (0, count, 1):
        raise ValueError(
            "Distributed results must be staged collectively before single-writer export"
        )
    n_frames = int(np.prod(result.trailing_shape))
    labels = (
        [str(i) for i in range(n_frames)] if labels is None else list(map(str, labels))
    )
    if len(labels) != n_frames or len(set(map(float, labels))) != n_frames:
        raise ValueError("Need one unique numeric label per result frame")
    output = Path(output_file).resolve()
    descriptor = output.with_suffix(".xdmf")
    if descriptor.exists() and not overwrite and not append:
        raise FileExistsError(descriptor)
    with HDF5Writer(
        str(output),
        geometry_file=geometry_file,
        field_dtype=field_dtype,
        compression=compression,
        compression_level=compression_level,
        shuffle=shuffle,
        overwrite=overwrite,
        append=append,
        spatial_chunk=result.spatial_batch_size,
    ) as writer:
        storage = writer._series
        names = select_names(storage.zones, zones, "zone")
        if names != list(storage.zones):
            raise ValueError(
                "Result zone order must match the supplied geometry file; create matching subset geometry first"
            )
        if (
            geometry_id is not None
            and geometry_id != storage.geometry.attrs["geometry_id"]
        ):
            raise ValueError("Result belongs to a different geometry")
        offsets = np.cumsum(
            [0]
            + [
                len(storage.geometry["zones"][storage.zones[n]]["weights"])
                for n in names
            ]
        )
        if offsets[-1] != count:
            raise ValueError("Result spatial size does not match geometry")
        if not append:
            writer._file.attrs["result_metadata"] = json.dumps(metadata or {})
            writer._file.attrs["trailing_shape"] = result.trailing_shape
        first = True
        expected = None
        for part, values in result.iter_chunks(split=True):
            current = {
                name: (tuple(value.shape[1:]), value.is_complex())
                for name, value in values.items()
            }
            if expected is not None and current != expected:
                raise ValueError("Result field schema changes across chunks")
            expected = current
            for name, value in values.items():
                component_shape = next(
                    f.component_shape for f in result.layout.fields if f.name == name
                )
                value = value.reshape(
                    part.stop - part.start, *component_shape, n_frames
                )
                if value.is_complex():
                    if representation == "real_imag":
                        fields = {
                            name + "_real": value.real,
                            name + "_imag": value.imag,
                        }
                    else:
                        fields = {
                            name + "_magnitude": value.abs(),
                            name + "_phase": value.angle(),
                        }
                else:
                    fields = {name: value}
                for output_name, physical in fields.items():
                    for frame, label in enumerate(labels):
                        for i, zone in enumerate(names):
                            if first:
                                writer.allocate_field(
                                    output_name, component_shape, label, zone
                                )
                            lo, hi = max(part.start, offsets[i]), min(
                                part.stop, offsets[i + 1]
                            )
                            if hi > lo:
                                writer.write_slice(
                                    output_name,
                                    label,
                                    zone,
                                    slice(lo - offsets[i], hi - offsets[i]),
                                    physical[
                                        lo - part.start : hi - part.start, ..., frame
                                    ],
                                )
            first = False
        for frame in storage.frames.values():
            frame.attrs["result_metadata"] = json.dumps(metadata or {})
            frame.attrs["trailing_shape"] = result.trailing_shape
        if descriptor.exists() and not append:
            descriptor.unlink()
        writer.write_xdmf(descriptor)
    return descriptor


def export_spod(
    spod,
    output_directory,
    geometry_file,
    *,
    frequency_indices,
    mode_indices=(0,),
    animation_phases=None,
    amplitude=1.0,
    geometry_id=None,
    field_dtype=torch.float32,
    compression=None,
    compression_level=None,
    shuffle=False,
    overwrite=False,
    staging_directory=None,
):
    """Export selected PAM-SPOD modes and optional real phase animations.

    Animations use ``amplitude * real(mode * exp(1j*phase))``. Their labels are
    phase-sample indices, not physical flow times. They are not reconstructions.
    Use export_fields on partial_reconstruction() for actual reconstructed data.
    compression, compression_level and shuffle apply to both mode and animation
    field files, leaving shared geometry unchanged.
    """
    from flowtorch.analysis.state_vector import StateVectorResult

    if not np.isfinite(amplitude):
        raise ValueError("Animation amplitude must be finite")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    descriptors = []
    for frequency, mode in itertools.product(frequency_indices, mode_indices):
        result = spod.get_mode(frequency, mode)
        if not isinstance(result, StateVectorResult):
            raise ValueError(
                "Export requires PAMSPOD.from_svd with a state-vector source"
            )
        metadata = dict(
            kind="spod_mode",
            frequency_index=int(frequency),
            mode_index=int(mode),
            frequency=float(spod.frequency[frequency]),
            eigenvalue=float(spod.eigvals[frequency, mode]),
            normalization=result.layout.normalization_factors,
        )
        descriptors.append(
            export_fields(
                result,
                output / f"frequency_{frequency}_mode_{mode}.h5",
                geometry_file,
                geometry_id=geometry_id,
                field_dtype=field_dtype,
                compression=compression,
                compression_level=compression_level,
                shuffle=shuffle,
                metadata=metadata,
                overwrite=overwrite,
                staging_directory=staging_directory,
            )
        )
        if animation_phases is not None:
            phases = np.asarray(animation_phases, dtype=float)
            if phases.ndim != 1 or not len(phases) or not np.isfinite(phases).all():
                raise ValueError("Need a nonempty finite phase sequence")

            def produce(part):
                value = cast(torch.Tensor, result.read_chunk(part))
                rotations = torch.exp(
                    1j
                    * torch.as_tensor(
                        phases, dtype=value.real.dtype, device=value.device
                    )
                )
                return amplitude * (value.reshape(-1, 1) * rotations).real

            animation = StateVectorResult(
                result.layout,
                result.local_spatial_slice,
                (len(phases),),
                produce,
                result.spatial_batch_size,
                result.execution,
            )
            descriptors.append(
                export_fields(
                    animation,
                    output / f"frequency_{frequency}_mode_{mode}_animation.h5",
                    geometry_file,
                    geometry_id=geometry_id,
                    field_dtype=field_dtype,
                    compression=compression,
                    compression_level=compression_level,
                    shuffle=shuffle,
                    metadata={
                        **metadata,
                        "kind": "mode_phase_animation",
                        "phases": phases.tolist(),
                        "amplitude": amplitude,
                    },
                    overwrite=overwrite,
                    staging_directory=staging_directory,
                )
            )
    return descriptors


def _export_distributed(result, output_file, geometry_file, staging_directory, options):
    """Collective rank-disjoint disk staging followed by a root-only writer."""
    import shutil
    import tempfile
    import torch.distributed as dist
    from flowtorch.analysis.state_vector import StateVectorResult

    if staging_directory is None:
        raise ValueError(
            "Distributed export requires an explicit shared staging_directory"
        )
    group = result.execution.process_group
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    root = result.execution.root_rank
    ranks = [None] * size
    dist.all_gather_object(ranks, dist.get_rank(), group=group)
    status: list[Any] = [None, None]
    if rank == root:
        try:
            from ._hdf5_compression import compression_settings

            if options.get("append"):
                from ._hdf5_schema import inspect_append

                info = inspect_append(
                    output_file,
                    options["field_dtype"],
                    geometry_file,
                    compression=options["compression"],
                    compression_level=options["compression_level"],
                    shuffle=options["shuffle"],
                )
                if options["overwrite"] or options["labels"] is None:
                    raise ValueError(
                        "Append requires explicit new labels and forbids overwrite"
                    )
                if any(float(t) in info["times"] for t in options["labels"]):
                    raise ValueError(
                        "Append contains an already existing numeric timestamp"
                    )
            else:
                compression_settings(
                    options["compression"],
                    options["compression_level"],
                    False if options["shuffle"] is None else options["shuffle"],
                )
            status[0] = tempfile.mkdtemp(
                prefix="flowtorch-export-", dir=staging_directory
            )
        except Exception as error:
            status[1] = str(error)
    dist.broadcast_object_list(status, src=ranks[root], group=group)
    if status[1] is not None:
        raise RuntimeError(status[1])
    folder = Path(status[0])
    record = {
        "chunks": [],
        "error": None,
        "signature": result.layout.signature,
        "trailing_shape": result.trailing_shape,
    }
    try:
        for index, (part, fields) in enumerate(result.iter_chunks(split=True)):
            paths = {}
            for field_index, (name, value) in enumerate(fields.items()):
                path = folder / f"rank{rank}_chunk{index}_field{field_index}.npy"
                np.save(path, value.detach().cpu().numpy(), allow_pickle=False)
                paths[name] = str(path)
            record["chunks"].append((part.start, part.stop, paths))
    except Exception as error:
        record["error"] = str(error)
    records: list[Any] = [None] * size
    dist.all_gather_object(records, record, group=group)
    staging_error = next((r["error"] for r in records if r["error"] is not None), None)
    status = [None, staging_error]
    if rank == root:
        try:
            if staging_error:
                raise RuntimeError(staging_error)
            if any(
                r["signature"] != record["signature"]
                or r["trailing_shape"] != record["trailing_shape"]
                for r in records
            ):
                raise ValueError("Distributed result layouts differ")
            chunks = sorted(chunk for r in records for chunk in r["chunks"])
            position = 0
            for start, stop, _ in chunks:
                if start != position or stop <= start:
                    raise ValueError("Distributed partitions overlap or have gaps")
                position = stop
            if position != result.layout.spatial_shape[0]:
                raise ValueError("Incomplete distributed spatial result")

            def produce(part):
                fields = {}
                for start, stop, paths in chunks:
                    lo, hi = max(start, part.start), min(stop, part.stop)
                    if hi <= lo:
                        continue
                    for name, path in paths.items():
                        mapped = np.load(path, mmap_mode="r", allow_pickle=False)
                        if name not in fields:
                            fields[name] = torch.empty(
                                (part.stop - part.start, *mapped.shape[1:]),
                                dtype=torch.from_numpy(
                                    np.empty((), dtype=mapped.dtype)
                                ).dtype,
                            )
                        fields[name][lo - part.start : hi - part.start] = (
                            torch.from_numpy(
                                np.array(mapped[lo - start : hi - start], copy=True)
                            )
                        )
                return result.layout.pack(
                    fields, spatial_shape=(part.stop - part.start,)
                )

            staged = StateVectorResult(
                result.layout,
                slice(0, position),
                result.trailing_shape,
                produce,
                result.spatial_batch_size,
            )
            status[0] = str(
                export_fields(staged, output_file, geometry_file, **options)
            )
        except Exception as failure:
            status[1] = str(failure)
        finally:
            shutil.rmtree(folder)
    dist.broadcast_object_list(status, src=ranks[root], group=group)
    if status[1] is not None:
        raise RuntimeError(f"Distributed export failed: {status[1]}")
    return Path(status[0]) if rank == root else None
