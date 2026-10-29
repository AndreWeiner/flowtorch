Release notes
=============

1.10.1
------

This patch release fixes floating-point dtype handling in data loaders and
makes EnSight precision limitations explicit.

Fixes
~~~~~

* ``TecplotDataloader`` now applies its requested ``dtype`` consistently to
  snapshots, time series, vertices, cell locations, and integration weights.
  Previously, only weights honored this setting. Backend array transport
  preserves the reader's dtype until conversion at the tensor boundary.
* ``ImageDataloader`` converts pixels to the requested dtype before
  normalization, avoiding an intermediate float32 division when float64 is
  requested.

Precision warnings
~~~~~~~~~~~~~~~~~~

* ``EnSightDataloader(dtype=torch.float64)`` emits a ``UserWarning`` once per
  construction. EnSight Gold binary files store float32 fields and coordinates;
  the current VTK ASCII reader also reads these values as float32. Returned
  float64 tensors cannot recover the additional source digits. The native
  binary chunk backend shares the file format's precision limit.
* ``EnSightWriter`` emits a ``UserWarning`` once per writer when converting
  float64 fields (including complex128 components) to binary float32 output.
  Binary remains the default. ASCII preserves double-precision field values
  in the file, but the current VTK EnSight reader reads them as float32.

Compatibility
~~~~~~~~~~~~~

Tecplot snapshots and coordinates now follow the default ``torch.float32``
dtype, even when the reader provides float64 arrays. Pass
``dtype=torch.float64`` explicitly to retain the reader's available precision
in returned tensors. This setting cannot restore precision lost in a source
file or its reader. The VTK mesh export retains its reader-provided precision.
SCube behavior is unchanged.
