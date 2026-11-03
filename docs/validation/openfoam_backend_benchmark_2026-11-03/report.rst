OpenFOAM backend benchmark, 2026-11-03
=====================================

Scope and reproduction
----------------------

Compare the current working-tree OpenFOAM backend against the saved baseline
from commit 56ec0ce406cb25622d02bd9f64232ff78f8cbef2. Original datasets are read-only;
all generated inputs are created in a temporary directory. Run from the
repository root (requires the local cavity dataset)::

    .tox/py310/bin/python docs/validation/openfoam_backend_benchmark_2026-11-03/benchmark.py --ascii

The benchmark stores raw samples in results.json. --resume reuses already
completed comparisons; omit it for a fresh full run. run.log includes a
pilot failure caused by a manufactured owner-note spelling accepted by the
new parser but rejected by the old one. That fixture was corrected before
grid timing; valid earlier field/cavity results were retained.

Method
------

CPU: Intel(R) Core(TM) Ultra 7 255H. PyTorch 2.14.0+cu130.
All tensors are CPU Float64. Fields use one CPU thread. Geometry is tested
with one thread and the machine default of 16. File pages are warmed by
one untimed call per method; method order alternates between repetitions.
There are five field trials and three geometry trials. Tiny cavity field
trials average 20 calls each. No cold-disk or native solver-write timings
are claimed. Gzip uses Python gzip at level 6 on identical native contents.
Compression creation, case initialization and test-data generation are
excluded from read timings. Shape discovery, parsing, allocation and dtype
conversion inside a read are included. No peak-memory speedup is measured.

The old slice path reads the entire snapshot and then slices in memory.
Requested chunks are 1% of cells, starting halfway through the field.
The large fields are manufactured smooth velocity vectors, not solver output.
Geometry grids are manufactured unit-cube hexahedral meshes (4096 and
32768 cells). Geometry end-to-end timings bypass precomputed C/V and
include native mesh reads, parsing, face geometry, and cell geometry.
Compute-only timings reuse the same parsed arrays for both implementations.
Old/new fields and geometry agree at rtol=1e-11 or tighter, atol=1e-12. Structured-grid centres and volumes also checked analytically, including an independent post-timing check of the 32^3 grid.

Raw timing summary
------------------

All entries below are median [minimum, maximum], in milliseconds.

fields / cavity
~~~~~~~~~~~~~~~

* old_binary_slice: 0.0639 [0.0608, 0.0736] ms
* new_binary_slice: 0.1420 [0.1352, 0.1620] ms
* new_gzip_slice: 0.4680 [0.4623, 0.4993] ms
* old_binary_full: 0.0685 [0.0664, 0.0736] ms
* new_binary_full: 0.1486 [0.1400, 0.1638] ms
* new_gzip_full: 0.4979 [0.4741, 0.5166] ms

* native_case_binary_bytes: 200651 bytes
* native_case_gzip_bytes: 117569 bytes
* U_binary_bytes: 10824 bytes
* U_gzip_bytes: 8253 bytes

fields / manufactured_250k
~~~~~~~~~~~~~~~~~~~~~~~~~~

* old_binary_slice: 41.4963 [39.7638, 43.6047] ms
* new_binary_slice: 0.2849 [0.1465, 0.3027] ms
* new_gzip_slice: 11.6992 [11.4577, 11.9311] ms
* old_binary_full: 41.2828 [39.5711, 42.4656] ms
* new_binary_full: 4.2720 [3.8637, 4.4760] ms
* new_gzip_full: 25.7886 [25.2790, 26.5840] ms

* native_case_binary_bytes: 6000256 bytes
* native_case_gzip_bytes: 4971849 bytes
* U_binary_bytes: 6000201 bytes
* U_gzip_bytes: 4971694 bytes

fields / manufactured_250k_ascii
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* old_ascii_slice: 344.7512 [339.5841, 419.4715] ms
* new_ascii_slice: 264.9334 [261.4614, 271.4977] ms
* new_gzip_slice: 313.8722 [310.0857, 339.6077] ms
* old_ascii_full: 350.7256 [333.5521, 392.2860] ms
* new_ascii_full: 544.4938 [537.3026, 561.8632] ms
* new_gzip_full: 632.6255 [629.4390, 637.2332] ms

* native_case_ascii_bytes: 14618777 bytes
* native_case_gzip_bytes: 5587801 bytes
* U_ascii_bytes: 14618722 bytes
* U_gzip_bytes: 5587646 bytes

geometry / cavity
~~~~~~~~~~~~~~~~~

* old_end_to_end: 55.4191 [54.8478, 55.7006] ms
* new_end_to_end: 1.3677 [1.1368, 1.5700] ms
* old_compute_only: 44.3755 [44.1166, 44.9205] ms
* new_compute_only: 0.6760 [0.6277, 0.7625] ms

geometry / grid16_one_thread
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* old_end_to_end: 619.0032 [575.7037, 633.3280] ms
* new_end_to_end: 5.1629 [4.9237, 5.3591] ms
* old_compute_only: 526.5057 [475.5809, 547.5600] ms
* new_compute_only: 3.6326 [3.5208, 5.8528] ms

geometry / grid32_one_thread
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* old_end_to_end: 4770.1749 [4459.6121, 4880.3603] ms
* new_end_to_end: 42.3650 [40.5528, 42.3821] ms
* old_compute_only: 3862.8102 [3859.4806, 3914.0330] ms
* new_compute_only: 33.5997 [33.4312, 33.6949] ms

geometry / grid32_default_threads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* old_end_to_end: 4552.0395 [4490.1407, 4963.9477] ms
* new_end_to_end: 25.6986 [22.1883, 27.8629] ms
* old_compute_only: 3833.0983 [3826.0236, 3924.2796] ms
* new_compute_only: 25.6730 [19.3948, 29.6563] ms

Interpretation
--------------

Binary partial reads are substantially faster on the larger field, but
tiny fields have higher metadata overhead in the new parser. Large ASCII
partial reads improve only modestly and full ASCII reads regress. Gzip
saves storage but adds decompression cost and cannot directly seek to a
chunk without processing the prefix. Geometry gains apply to these meshes
and CPU settings; they are not a universal scaling claim. Numerical
agreement was checked separately from timing. The largest native mesh
contained 32768 cells and 101376 faces. No production-size CFD benchmark
or native OpenFOAM writing benchmark was run.
