"""Benchmark lossless HDF5 filters on an existing PLT conversion.

Run with a Python environment providing h5py, numpy and hdf5plugin.
Input loading and validation are outside timers; writes include close and fsync.
Reads follow writes (warm OS cache), reopening files for each measured pass.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time

import h5py
import hdf5plugin  # Registers native Zstd filter.
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path, pilot):
    arrays, attrs = {}, {}
    with h5py.File(path, "r") as f:
        attrs[""] = dict(f.attrs)

        def visit(name, obj):
            attrs[name] = dict(obj.attrs)
            if isinstance(obj, h5py.Dataset):
                arrays[name] = obj[: min(len(obj), 4096)] if pilot else obj[:]

        f.visititems(visit)
    return arrays, attrs


def write(path, arrays, attrs, options):
    start = time.perf_counter()
    with h5py.File(path, "w") as f:
        f.attrs.update(attrs[""])
        for name, data in arrays.items():
            parent = f.require_group(name.rsplit("/", 1)[0])
            ds = parent.create_dataset(
                name.rsplit("/", 1)[1],
                shape=data.shape,
                dtype=data.dtype,
                chunks=(min(len(data), 65536), *data.shape[1:]),
                **options,
            )
            ds.attrs.update(attrs[name])
            for i in range(0, len(data), 65536):
                ds[i : i + 65536] = data[i : i + 65536]
        for name, attributes in attrs.items():
            if name and name not in arrays:
                f.require_group(name).attrs.update(attributes)
    with open(path, "rb") as f:
        os.fsync(f.fileno())
    return time.perf_counter() - start


def read(path, arrays, sliced):
    start = time.perf_counter()
    with h5py.File(path, "r") as f:
        result = {}
        for name, data in arrays.items():
            lo = max(0, (len(data) - 8192) // 2) if sliced else 0
            hi = min(len(data), lo + 8192) if sliced else len(data)
            result[name] = f[name][lo:hi]
    elapsed = time.perf_counter() - start
    # Exact byte equality checks signed zero and NaN payloads as well as values.
    for name, actual in result.items():
        data = arrays[name]
        lo = max(0, (len(data) - 8192) // 2) if sliced else 0
        expected = data[lo : lo + len(actual)]
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes(), name
    return elapsed


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    methods = {
        "none": {},
        "gzip4": {"compression": "gzip", "compression_opts": 4},
        "gzip4_shuffle": {
            "compression": "gzip",
            "compression_opts": 4,
            "shuffle": True,
        },
        "zstd3": dict(hdf5plugin.Zstd(clevel=3)),
        "zstd3_shuffle": {**dict(hdf5plugin.Zstd(clevel=3)), "shuffle": True},
    }
    manifest = args.source / "conversion_manifest.json"
    record = {
        "complete": False,
        "pilot": args.pilot,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        ),
        "versions": {
            p: importlib.metadata.version(p) for p in ["h5py", "numpy", "hdf5plugin"]
        },
        "hdf5": h5py.version.hdf5_version,
        "harness_sha256": digest(Path(__file__)),
        "chunk_rows": 65536,
        "repeats": args.repeats,
        "methods": methods,
        "cache": "Warm OS cache: reads follow writes; no cold-cache claim. Reopened HDF5 handles.",
        "write_timing": "dataset creation, source-array slice writes, attributes, HDF5 close, fsync; input loading excluded",
        "read_timing": "file open, all requested dataset allocations/reads, close; verification excluded",
        "slice": "One central contiguous slice of up to 8192 rows per dataset",
        "validation": "Every dataset, shape, dtype and bytes checked for full and sliced reads in every trial",
        "source_manifest": (
            json.loads(manifest.read_text()) if manifest.exists() else None
        ),
        "inputs": {},
        "trials": [],
    }
    for category in ["geometry", "fields"]:
        source = args.source / (category + ".h5")
        arrays, attrs = load(source, args.pilot)
        record["inputs"][category] = {
            "path": str(source.resolve()),
            "sha256": digest(source),
            "datasets": len(arrays),
            "array_bytes": sum(x.nbytes for x in arrays.values()),
            "source_file_bytes": source.stat().st_size,
        }
        # Bounded untimed warm-up of each codec, using the real numerical arrays.
        small = {name: data[: min(len(data), 4096)] for name, data in arrays.items()}
        for method, options in methods.items():
            path = args.output / ("warmup-" + method + ".h5")
            write(path, small, attrs, options)
            read(path, small, False)
            path.unlink()
        del small
        for rep in range(args.repeats):
            order = list(methods)
            random.Random(813 + rep).shuffle(order)
            for method in order:
                path = args.output / f"{category}-{method}.h5"
                write_seconds = write(path, arrays, attrs, methods[method])
                full_seconds = read(path, arrays, False)
                slice_seconds = read(path, arrays, True)
                trial = dict(
                    category=category,
                    method=method,
                    repeat=rep,
                    file_bytes=path.stat().st_size,
                    write_seconds=write_seconds,
                    full_read_seconds=full_seconds,
                    slice_read_seconds=slice_seconds,
                )
                record["trials"].append(trial)
                (args.output / "results.json").write_text(
                    json.dumps(record, indent=2) + "\n"
                )
                print(json.dumps(trial), flush=True)
        del arrays
    record["summary"] = []
    for category in ["geometry", "fields"]:
        for method in methods:
            rows = [
                r
                for r in record["trials"]
                if r["category"] == category and r["method"] == method
            ]
            summary = dict(
                category=category, method=method, file_bytes=rows[0]["file_bytes"]
            )
            for key in ["write_seconds", "full_read_seconds", "slice_read_seconds"]:
                values = [r[key] for r in rows]
                summary[key] = dict(
                    median=statistics.median(values),
                    minimum=min(values),
                    maximum=max(values),
                )
            record["summary"].append(summary)
    record["complete"] = True
    (args.output / "results.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
