"""Validated lossless HDF5 filters shared by writers and converters."""

from dataclasses import dataclass
import importlib
import json
from numbers import Integral

import h5py

ZSTD_FILTER = 32015


def enable_zstd():
    """Register the optional native filter, unless HDF5 already provides it."""
    if h5py.h5z.filter_avail(ZSTD_FILTER):
        return
    try:
        importlib.import_module("hdf5plugin")
    except ImportError as error:
        raise ImportError(
            "Zstd compression requires the optional HDF5 filter. "
            "Install flowtorch-fluid[compression] (or hdf5plugin). "
            "ParaView/pvserver additionally needs HDF5_PLUGIN_PATH configured."
        ) from error
    if not h5py.h5z.filter_avail(ZSTD_FILTER):
        raise RuntimeError("The installed HDF5 plugin did not register Zstd")


@dataclass(frozen=True)
class CompressionSettings:
    """Resolved codec, level and reversible byte shuffle (internal API)."""

    codec: str | None
    level: int | None
    shuffle: bool

    @property
    def enabled(self):
        return self.codec is not None or self.shuffle

    @property
    def dataset_options(self):
        options: dict[str, object] = {"shuffle": True} if self.shuffle else {}
        if self.codec == "gzip":
            options.update(compression="gzip", compression_opts=self.level)
        elif self.codec == "zstd":
            assert self.level is not None
            # HDF5 filter options are unsigned; Zstd interprets this as int32.
            options.update(
                compression=ZSTD_FILTER, compression_opts=(self.level & 0xFFFFFFFF,)
            )
        return options

    @property
    def metadata(self):
        return dict(codec=self.codec, level=self.level, shuffle=self.shuffle)


def compression_settings(codec=None, level=None, shuffle=False):
    """Validate and resolve defaults before any output is created."""
    if codec not in (None, "gzip", "zstd"):
        raise ValueError("Compression must be None, 'gzip' or 'zstd'")
    if not isinstance(shuffle, bool):
        raise ValueError("shuffle must be a bool")
    if level is not None and (
        isinstance(level, bool) or not isinstance(level, Integral)
    ):
        raise ValueError("compression_level must be an integer")
    if codec is None:
        if level is not None:
            raise ValueError("compression_level requires a compression codec")
    else:
        level = int(level) if level is not None else (4 if codec == "gzip" else 3)
        low, high = (0, 9) if codec == "gzip" else (-131072, 22)
        if not low <= level <= high:
            raise ValueError(
                f"{codec} compression_level must be between {low} and {high}"
            )
        if codec == "zstd":
            enable_zstd()
            flags = h5py.h5z.get_filter_info(ZSTD_FILTER)
            if not flags & h5py.h5z.FILTER_CONFIG_ENCODE_ENABLED:
                raise RuntimeError("The available Zstd filter cannot encode data")
    return CompressionSettings(codec, level, shuffle)


def record_compression(file, **settings):
    """Record resolved settings; actual chunk shapes remain on each dataset."""
    file.attrs["compression_settings"] = json.dumps(
        {name: value.metadata for name, value in settings.items()}, sort_keys=True
    )


def prepare_read(file):
    """Register Zstd automatically for files produced with our writer."""
    settings = json.loads(file.attrs.get("compression_settings", "{}"))
    if any(value.get("codec") == "zstd" for value in settings.values()):
        enable_zstd()


def prepare_dataset_read(dataset):
    """Also recognize externally produced Zstd datasets without our metadata."""
    properties = dataset.id.get_create_plist()
    if any(
        properties.get_filter(i)[0] == ZSTD_FILTER
        for i in range(properties.get_nfilters())
    ):
        enable_zstd()


def dataset_compression(dataset):
    """Recover reproducible settings from an existing field's actual filters."""
    codec, level, shuffle = None, None, False
    properties = dataset.id.get_create_plist()
    for index in range(properties.get_nfilters()):
        identifier, _, options, _ = properties.get_filter(index)
        if identifier == h5py.h5z.FILTER_SHUFFLE:
            shuffle = True
        elif identifier == h5py.h5z.FILTER_DEFLATE:
            codec, level = "gzip", int(options[0])
        elif identifier == ZSTD_FILTER:
            codec, level = "zstd", int(options[0])
            if level >= 2**31:
                level -= 2**32
        else:
            raise ValueError(
                "Append requires fields using only gzip/Zstd and optional shuffle"
            )
    return compression_settings(codec, level, shuffle)
