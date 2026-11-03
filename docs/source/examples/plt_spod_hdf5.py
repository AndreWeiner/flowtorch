"""Convert fixed-mesh PLT snapshots, compute SPOD and export ParaView fields."""

import argparse
from pathlib import Path

import numpy as np
import torch

from flowtorch.analysis import DataloaderStateVectorSource, FieldSpec, PAMSPOD, SVD
from flowtorch.data import HDF5Dataloader, PLT2HDF5, export_spod


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Directory containing PLT snapshots")
    parser.add_argument("output", type=Path, help="New directory for this workflow")
    parser.add_argument("--fields", nargs="+", default=["cp"])
    parser.add_argument("--max-snapshots", type=int)
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--frequency-index", type=int, default=1)
    args = parser.parse_args()

    # 1. Convert once: Float64 geometry and Float32 physical fields.
    converter = PLT2HDF5(args.input, backend="native")
    times = list(converter.files)  # Already ordered by numeric time.
    if args.max_snapshots is not None:
        if args.max_snapshots < 4:
            parser.error("Use at least four snapshots for this demonstration")
        times = times[: args.max_snapshots]
    if len(times) < 4 or not 2 <= args.rank < len(times):
        parser.error("Need at least four snapshots and 2 <= rank < snapshot count")
    numeric_times = np.asarray(times, dtype=float)
    increments = np.diff(numeric_times)
    dt = float(increments[0])
    if dt <= 0 or not np.allclose(increments, dt, rtol=1e-6, atol=1e-12):
        parser.error("SPOD requires uniformly spaced snapshot times")
    if not 0 <= args.frequency_index <= len(times) // 2:
        parser.error("Frequency index must lie between zero and snapshot_count // 2")

    converted = args.output / "converted"
    descriptor = converter.convert(
        converted,
        fields=args.fields,
        times=times,
        compression="gzip",
        shuffle=True,
        geometry_compression="gzip",
        geometry_shuffle=True,
    )
    geometry = converted / "geometry.h5"

    # 2. Read chunks from HDF5 and compute a centred, geometrically weighted SVD.
    with HDF5Dataloader(descriptor, dtype=torch.float64) as loader:
        fields = [
            FieldSpec(name, component_shape=loader.snapshot_shape(name, times[0])[1:])
            for name in args.fields
        ]
        source = DataloaderStateVectorSource(
            loader,
            fields,
            normalization={name: 1.0 for name in args.fields},
            spatial_weight=loader.weights,
        )
        svd = SVD(source, rank=args.rank, subtract_mean=True, spatial_batch_size=65536)

        # 3. Reuse the SVD for fixed-taper SPOD; do not reload the original PLT.
        spod = PAMSPOD.from_svd(
            svd, dt=dt, adaptive=False, max_tapers=2, keep_n_modes=1
        )
        results = args.output / "spod"
        results.mkdir(parents=True, exist_ok=True)
        np.savez(
            results / "spectrum.npz",
            frequency=spod.frequency.detach().cpu().numpy(),
            eigenvalues=spod.eigvals.detach().cpu().numpy(),
        )

        # 4. Write modes and phase frames; reuse geometry.h5 read-only.
        descriptors = export_spod(
            spod,
            results,
            geometry,
            frequency_indices=[args.frequency_index],
            mode_indices=[0],
            animation_phases=np.linspace(0, 2 * np.pi, 24, endpoint=False),
            amplitude=1.0,
            geometry_id=loader.geometry_id,
            field_dtype=torch.float32,
            compression="gzip",
            shuffle=True,
        )
        print(
            f"Exported mode frequency: {float(spod.frequency[args.frequency_index]):g}"
        )
        for path in descriptors:
            print(path.resolve())


if __name__ == "__main__":
    main()
