"""Fixed-mesh PLT conversion using the shared FlowTorch HDF5 interfaces."""

from pathlib import Path
from typing import Any
import re
import warnings

import numpy as np
import torch

from .tecplot_dataloader import TecplotDataloader
from .hdf5_file import HDF5Writer
from ._hdf5_schema import select_names


def discover_plt(path: str | Path, pattern: str = "*.plt") -> dict[str, str]:
    """Find unique numeric t= labels; accept an explicit mapping for other names."""
    path = Path(path).resolve()
    paths = sorted(path.glob(pattern)) if path.is_dir() else [path]
    entries = []
    for file in paths:
        match = re.search(
            r"t=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", file.name
        )
        if match is None:
            if len(paths) != 1:
                raise ValueError(
                    f"Cannot infer time from {file.name}; supply file_names"
                )
            label = "0"
        else:
            label = match[1]
        if not np.isfinite(float(label)):
            raise ValueError("Nonfinite time")
        entries.append((label, file))
    entries.sort(key=lambda pair: float(pair[0]))
    if not entries or len({float(t) for t, _ in entries}) != len(entries):
        raise ValueError("Need snapshots with unique finite times")
    return {t: str(p) for t, p in entries}


class PLT2HDF5:
    """Convert fixed-mesh PLT series, preserving original geometry precision.

    Native TDV71 input supports direct field slices. ParaView input can require
    full snapshots and may have already rounded source geometry internally.
    All fields are point-associated by default; no centering or scaling occurs.
    """

    def __init__(
        self,
        path: str | Path,
        pattern: str = "*.plt",
        *,
        file_names: dict[str, str] | None = None,
        backend: str = "auto",
        pvpython: str | Path | None = None,
        association: str = "point",
        cell_type: str | None = None,
    ):
        self.path = Path(path)
        self.files = (
            discover_plt(path, pattern) if file_names is None else dict(file_names)
        )
        if not self.files or len({float(t) for t in self.files}) != len(self.files):
            raise ValueError("Need unique numeric snapshot times")
        self.files = dict(sorted(self.files.items(), key=lambda pair: float(pair[0])))
        self.options: dict[str, Any] = dict(
            backend=backend,
            pvpython=pvpython,
            association=association,
            cell_type=cell_type,
        )

    def convert(
        self,
        output_directory: str | Path,
        fields: list[str] | None = None,
        times: list[str] | None = None,
        zones: list[str] | None = None,
        *,
        field_dtype: torch.dtype = torch.float32,
        geometry_file: str | Path | None = None,
        spatial_chunk: int = 65536,
        compression: str | None = None,
        compression_level: int | None = None,
        shuffle: bool | None = None,
        geometry_compression: str | None = None,
        geometry_compression_level: int | None = None,
        geometry_shuffle: bool = False,
        overwrite: bool = False,
        append: bool = False,
    ) -> Path:
        """Write geometry.h5, fields.h5 and fields.xdmf; return descriptor path.

        Geometry must match at every selected time. Existing geometry is reused
        read-only after content validation. Existing fields require overwrite.
        compression (None, "gzip", "zstd"), compression_level and shuffle
        configure fields. The ``geometry_`` prefixed equivalents configure only
        new geometry files. Zstd requires the optional compression extra.
        Compression is lossless and never changes numerical storage precision.
        append=True requires complete existing split output and adds unique new
        times. It inherits omitted zone/field selections and field compression;
        explicit conflicts or changed storage precision raise ValueError.
        append and overwrite are mutually exclusive. Geometry stays read-only.
        """
        from ._hdf5_compression import compression_settings

        if append and overwrite:
            raise ValueError("append and overwrite are mutually exclusive")
        if not append:
            compression_settings(
                compression, compression_level, False if shuffle is None else shuffle
            )
        compression_settings(
            geometry_compression, geometry_compression_level, geometry_shuffle
        )
        if field_dtype not in (torch.float32, torch.float64) or spatial_chunk < 1:
            raise ValueError(
                "Expected float32/float64 fields and positive spatial_chunk"
            )
        selected_times = select_names(self.files, times, "time")
        selected_times.sort(key=float)
        files = {t: self.files[t] for t in selected_times}
        root = self.path if self.path.is_dir() else self.path.parent
        output = Path(output_directory).resolve()
        field_path = output / "fields.h5"
        descriptor = output / "fields.xdmf"
        if (
            not overwrite
            and not append
            and (field_path.exists() or descriptor.exists())
        ):
            raise FileExistsError(field_path)
        geometry_path = (
            Path(geometry_file).resolve()
            if geometry_file is not None
            else output / "geometry.h5"
        )
        info = None
        if append:
            from ._hdf5_schema import inspect_append, validate_append_selection

            info = inspect_append(
                field_path,
                field_dtype,
                geometry_file,
                compression=compression,
                compression_level=compression_level,
                shuffle=shuffle,
            )
            geometry_path = info["geometry_path"]
            if zones is None:
                zones = info["zone_names"]
        # Reading in double avoids rounding double source fields before storage casts.
        with TecplotDataloader(
            str(root),
            files,
            dtype=torch.float64,
            geometry_dtype=torch.float64,
            **self.options,
        ) as loader:
            if not loader.supports_direct_slicing:
                warnings.warn(
                    "ParaView conversion may materialize complete snapshots and may round source coordinates",
                    RuntimeWarning,
                )
            names = select_names(loader.zone_names, zones, "zone")
            if info is not None and names != info["zone_names"]:
                raise ValueError("Append zone order differs from existing data")
            schemas = {}
            for zone in names:
                loader.zone = zone
                available = [
                    n
                    for n in loader.field_names[selected_times[0]]
                    if n not in ("X", "Y", "Z")
                ]
                chosen = select_names(
                    available,
                    (
                        list(info["schema"][zone])
                        if info is not None and fields is None
                        else fields
                    ),
                    "field",
                )
                shapes = {
                    n: tuple(loader.snapshot_shape(n, selected_times[0]))
                    for n in chosen
                }
                schemas[zone] = shapes
                for time in selected_times:
                    if any(
                        tuple(loader.snapshot_shape(n, time)) != shape
                        for n, shape in shapes.items()
                    ):
                        raise ValueError(f"Field schema changes in {zone} at {time}")
            common = [
                {n: shape[1:] for n, shape in schema.items()}
                for schema in schemas.values()
            ]
            if any(schema != common[0] for schema in common[1:]):
                raise ValueError("Selected zones must have the same field components")

            if info is not None:
                validate_append_selection(info, selected_times, schemas)

            def geometries():
                for zone in names:
                    loader.zone = zone
                    geom = loader.geometry()
                    try:
                        if any(
                            shape[0] != len(geom.weights)
                            for shape in schemas[zone].values()
                        ):
                            raise ValueError(
                                "Field association does not match geometry"
                            )
                        signature = geom.signature
                        yield zone, geom
                    finally:
                        del geom
                    # Validate later snapshots after writing/releasing the
                    # reference geometry. A failure leaves the file incomplete.
                    for time in selected_times[1:]:
                        current_signature = loader.geometry(time).signature
                        if current_signature != signature:
                            raise ValueError(
                                f"Moving/reordered mesh in {zone} at {time}"
                            )

            HDF5Writer.write_geometry(
                geometry_path,
                geometries(),
                compression=geometry_compression,
                compression_level=geometry_compression_level,
                shuffle=geometry_shuffle,
                spatial_chunk=spatial_chunk,
            )
            output.mkdir(parents=True, exist_ok=True)
            if overwrite and descriptor.exists():
                descriptor.unlink()
            with HDF5Writer(
                str(field_path),
                geometry_file=geometry_path,
                field_dtype=field_dtype,
                spatial_chunk=spatial_chunk,
                compression=compression,
                compression_level=compression_level,
                shuffle=shuffle,
                overwrite=overwrite,
                append=append,
            ) as writer:
                if not append:
                    writer._file.attrs["source_format"] = "Tecplot"
                    writer._file.attrs["source_backend"] = loader.backend
                for time in selected_times:
                    for zone in names:
                        loader.zone = zone
                        for name, shape in schemas[zone].items():
                            writer.allocate_field(name, shape[1:], time, zone)
                            for start in range(0, shape[0], spatial_chunk):
                                part = slice(
                                    start, min(start + spatial_chunk, shape[0])
                                )
                                writer.write_slice(
                                    name,
                                    time,
                                    zone,
                                    part,
                                    loader.load_snapshot_slice(name, time, part),
                                )
                writer.write_xdmf(descriptor)
        return descriptor


def main() -> None:
    """Command-line entry point for independent PLT conversion."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--pattern", default="*.plt")
    parser.add_argument("--fields", nargs="+")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--zone")
    selection.add_argument("--zones", nargs="+")
    parser.add_argument("--times", nargs="+")
    parser.add_argument("--precision", choices=["single", "double"], default="single")
    parser.add_argument(
        "--backend", choices=["auto", "native", "paraview"], default="auto"
    )
    parser.add_argument("--pvpython")
    parser.add_argument("--association", choices=["point", "cell"], default="point")
    parser.add_argument(
        "--cell-type", choices=["line", "triangle", "quad", "tetra", "hexahedron"]
    )
    parser.add_argument("--geometry")
    parser.add_argument("--spatial-chunk", type=int, default=65536)
    parser.add_argument("--compression", choices=["gzip", "zstd"])
    parser.add_argument("--compression-level", type=int)
    parser.add_argument(
        "--shuffle", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--geometry-compression", choices=["gzip", "zstd"])
    parser.add_argument("--geometry-compression-level", type=int)
    parser.add_argument("--geometry-shuffle", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--append", action="store_true")
    args = parser.parse_args()
    converter = PLT2HDF5(
        args.input,
        args.pattern,
        backend=args.backend,
        pvpython=args.pvpython,
        association=args.association,
        cell_type=args.cell_type,
    )
    print(
        converter.convert(
            args.output,
            args.fields,
            args.times,
            [args.zone] if args.zone else args.zones,
            field_dtype=torch.float64 if args.precision == "double" else torch.float32,
            geometry_file=args.geometry,
            spatial_chunk=args.spatial_chunk,
            compression=args.compression,
            compression_level=args.compression_level,
            shuffle=args.shuffle,
            geometry_compression=args.geometry_compression,
            geometry_compression_level=args.geometry_compression_level,
            geometry_shuffle=args.geometry_shuffle,
            overwrite=args.overwrite,
            append=args.append,
        )
    )


if __name__ == "__main__":
    main()
