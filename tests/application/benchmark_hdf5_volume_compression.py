"""Bounded-memory sample or full lossless repack of a converted PLT dataset.

Samples estimate compressed chunk payload, not complete file size. Full runs
measure actual file sizes and verify every source/output array byte. All source
files are opened read-only. Run on an allocated compute node for large inputs.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import time

import h5py
import numpy as np

from flowtorch.data._hdf5_compression import compression_settings, record_compression

CHUNK = 65536
METHODS = {
    "gzip": ("gzip", False),
    "gzip_shuffle": ("gzip", True),
    "zstd": ("zstd", False),
    "zstd_shuffle": ("zstd", True),
}


def identity(path):
    info = path.stat()
    with h5py.File(path, "r") as f:
        return dict(
            path=str(path.resolve()),
            bytes=info.st_size,
            mtime_ns=info.st_mtime_ns,
            geometry_id=str(f.attrs.get("geometry_id", "")),
        )


def repack(source, output, method, samples, deadline):
    codec, shuffle = METHODS[method]
    settings = compression_settings(codec, 4 if codec == "gzip" else 3, shuffle)
    rows = []
    start = time.perf_counter()
    with h5py.File(source, "r") as src, h5py.File(output, "x", track_order=True) as dst:
        dst.attrs.update(dict(src.attrs))
        dst.attrs["complete"] = False
        record_compression(
            dst, **{"geometry" if source.name == "geometry.h5" else "fields": settings}
        )

        def visit(name, obj):
            if time.monotonic() > deadline:
                raise TimeoutError("Bounded compression test exceeded its time budget")
            if isinstance(obj, h5py.Group):
                dst.require_group(name).attrs.update(dict(obj.attrs))
                return
            count = len(obj)
            assert count > 0 and obj.dtype.kind in "fiu"
            chunks = (min(CHUNK, count), *obj.shape[1:])
            ds = dst.create_dataset(
                name,
                shape=obj.shape,
                dtype=obj.dtype,
                chunks=chunks,
                **settings.dataset_options,
            )
            ds.attrs.update(dict(obj.attrs))
            n_chunks = (count + CHUNK - 1) // CHUNK
            indices = (
                list(range(n_chunks))
                if samples is None
                else sorted(
                    set(
                        np.linspace(
                            0, n_chunks - 1, min(samples, n_chunks), dtype=int
                        ).tolist()
                    )
                )
            )
            for i in indices:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        "Bounded compression test exceeded its time budget"
                    )
                part = slice(i * CHUNK, min(count, (i + 1) * CHUNK))
                ds[part] = obj[part]
            dst.flush()
            sizes = [
                ds.id.get_chunk_info_by_coord((i * CHUNK, *([0] * (obj.ndim - 1)))).size
                for i in indices
            ]
            full_count, edge = divmod(count, CHUNK)
            full_sizes = [size for i, size in zip(indices, sizes) if i < full_count]
            estimated = float(np.mean(full_sizes)) * full_count if full_count else 0.0
            if edge:
                estimated += sizes[indices.index(n_chunks - 1)]
            rows.append(
                dict(
                    dataset=name,
                    shape=list(obj.shape),
                    dtype=str(obj.dtype),
                    raw_bytes=obj.nbytes,
                    sampled_chunk_indices=indices,
                    sampled_compressed_chunk_bytes=sizes,
                    estimated_compressed_payload_bytes=estimated,
                )
            )

        src.visititems(visit)
    with open(output, "rb") as f:
        os.fsync(f.fileno())
    write_seconds = time.perf_counter() - start
    check_start = time.perf_counter()
    source_hash, output_hash = hashlib.sha256(), hashlib.sha256()
    with h5py.File(source, "r") as src, h5py.File(output, "r") as dst:
        for row in rows:
            name = row["dataset"]
            a, b = src[name], dst[name]
            assert a.shape == b.shape and a.dtype == b.dtype
            source_hash.update(name.encode())
            output_hash.update(name.encode())
            for i in row["sampled_chunk_indices"]:
                part = slice(i * CHUNK, min(len(a), (i + 1) * CHUNK))
                original, restored = a[part].tobytes(), b[part].tobytes()
                assert original == restored, (method, name, i)
                source_hash.update(original)
                output_hash.update(restored)
    assert source_hash.hexdigest() == output_hash.hexdigest()
    if samples is None:
        with h5py.File(output, "r+") as dst:
            dst.attrs["complete"] = True
        with open(output, "rb") as f:
            os.fsync(f.fileno())
    return dict(
        source=identity(source),
        output=str(output),
        file_bytes=output.stat().st_size,
        write_seconds=write_seconds,
        validation_seconds=time.perf_counter() - check_start,
        exact_readback=True,
        logical_source_sha256=source_hash.hexdigest(),
        logical_output_sha256=output_hash.hexdigest(),
        estimated_compressed_payload_bytes=sum(
            r["estimated_compressed_payload_bytes"] for r in rows
        ),
        datasets=rows,
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int)
    parser.add_argument(
        "--methods", nargs="+", choices=list(METHODS), default=list(METHODS)
    )
    parser.add_argument("--budget-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.samples is not None and args.samples < 2:
        parser.error("Use at least two chunks per dataset")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    deadline = time.monotonic() + args.budget_seconds
    inputs = {
        kind: identity(args.source / (kind + ".h5")) for kind in ["geometry", "fields"]
    }
    report = dict(
        complete=False,
        mode="sample" if args.samples else "full",
        job_id=os.environ.get("SLURM_JOB_ID"),
        node=socket.gethostname(),
        samples_per_dataset=args.samples,
        chunk_rows=CHUNK,
        versions={
            p: importlib.metadata.version(p) for p in ["h5py", "hdf5plugin", "numpy"]
        },
        hdf5=h5py.version.hdf5_version,
        inputs=inputs,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(
            Path(
                __import__(
                    "flowtorch.data._hdf5_compression", fromlist=["__file__"]
                ).__file__
            ).read_bytes()
        ).hexdigest(),
        methods={},
        timing_scope="Source HDF5 reads, compression, chunk writes, attributes, close and fsync; validation timed separately. One trial; cache state uncontrolled. No PLT parsing or geometry computation.",
        estimate_limit="Sampled compressed chunk payload extrapolation excludes full-file metadata and is not a measured full file size.",
    )
    for method in args.methods:
        folder = args.output / method
        folder.mkdir()
        report["methods"][method] = {}
        for kind in ["geometry", "fields"]:
            result = repack(
                args.source / (kind + ".h5"),
                folder / (kind + ".h5"),
                method,
                args.samples,
                deadline,
            )
            assert (
                identity(args.source / (kind + ".h5")) == inputs[kind]
            ), "Input changed during test"
            report["methods"][method][kind] = result
            (args.output / "results.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(
                method,
                kind,
                "bytes",
                result["file_bytes"],
                "write_s",
                result["write_seconds"],
                "validated",
                flush=True,
            )
        if args.samples is None:
            shutil.copyfile(args.source / "fields.xdmf", folder / "fields.xdmf")
    report.update(
        complete=True,
        elapsed_seconds=time.perf_counter() - start,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print("All selected datasets passed exact readback", flush=True)


if __name__ == "__main__":
    main()
