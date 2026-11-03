PLT to SPOD to ParaView: a complete workflow
============================================

The following complete, single-process example converts a fixed-mesh PLT series,
computes a chunked weighted SVD, reuses that SVD for SPOD, and writes the leading
complex mode and a phase animation. All selected zones remain in their original
order. The geometry is written once and reused read-only for both result files.

Download :download:`plt_spod_hdf5.py <examples/plt_spod_hdf5.py>` and run it with
FlowTorch installed with its ``data`` extra. This example explicitly selects the
native backend for supported TDV71 and TDV112 FE-block layouts::

    python plt_spod_hdf5.py /path/to/surface_plt output --fields cp --rank 20

For a quick application check using the first ten snapshots in numeric time
order, choose a small retained SVD rank::

    python plt_spod_hdf5.py /path/to/surface_plt output-first10 \
        --fields cp --max-snapshots 10 --rank 3 --frequency-index 1

Use a new output directory for each run. Ten snapshots are useful for checking
file handling and visualization, but are insufficient to establish convergence
of an SPOD spectrum. Choose the retained rank, taper settings, record length and
frequency bins for the scientific question when analysing the full series.

.. literalinclude:: examples/plt_spod_hdf5.py
   :language: python
   :start-at: import argparse

The resulting layout is::

    output/
        converted/
            geometry.h5
            fields.h5
            fields.xdmf
        spod/
            spectrum.npz
            frequency_1_mode_0.h5
            frequency_1_mode_0.xdmf
            frequency_1_mode_0_animation.h5
            frequency_1_mode_0_animation.xdmf

With the default frequency index, the result descriptors reference
``../converted/geometry.h5``. No second geometry file is created. Conversion
stores fields in Float32 and coordinates in Float64; the example uses Float64
analysis arithmetic and Float32 result storage. Geometry, fields, modes and
animation frames use lossless gzip plus shuffle; reused geometry is not rewritten.
For optional Zstd and plugin setup, see :doc:`hdf5_workflow`.
Analysis in double precision
cannot restore precision discarded during field conversion. Change
``field_dtype`` in the conversion and export calls to retain double fields.

For later analyses of the same converted dataset, skip the conversion step and
start from ``HDF5Dataloader("output/converted/fields.h5", dtype=torch.float64)``.
Pass ``output/converted/geometry.h5`` and that loader's ``geometry_id`` to the
export helper, choosing a new result directory. The original PLT files are no
longer needed for analysis or export. To extend the converted time series, use
``converter.convert(..., times=new_times, append=True)`` with only new times;
see :doc:`hdf5_workflow` for validation, inherited compression and recovery.
Reopen the loader afterward. Appending snapshots does not automatically update
an already computed SVD or SPOD.

The example derives ``dt`` from the saved time labels and checks uniform time
spacing before conversion. Frequencies are in inverse units of those labels
(Hz if the labels are seconds). It uses the complete available record for SPOD;
this implementation's ``nfft`` parameter controls FFT length/zero padding, not a
Welch segment length. Omitting it uses the record length.

For ``cp`` alone, normalization by 1 leaves the physical scale unchanged.
For multiple fields with different units or characteristic scales, supply
appropriate normalization factors or use ``normalization="rms"`` deliberately.
``export_spod`` restores field normalization when exporting physical fields;
do not multiply the modes by the normalization factors again.

Opening the output in ParaView
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Open ``converted/fields.xdmf`` to inspect the converted physical snapshots.
   To select zones before loading, choose **XDMF Reader**, which exposes grid
   selection; the automatically selected Xdmf3 reader may not expose it. Select
   zones and arrays in the reader properties before clicking **Apply**.
2. Open ``spod/frequency_1_mode_0.xdmf`` to inspect the selected complex mode.
   For ``cp``, its arrays are named ``cp_real`` and ``cp_imag``. They retain the
   original point association and zone topology.
3. Open ``spod/frequency_1_mode_0_animation.xdmf``, choose the ``cp`` array, and
   use the animation controls to step through the 24 phase frames. Their time
   labels are frame indices, not the original simulation times.

Keep the folder layout when moving these files so that relative HDF5 references
remain valid. The exported phase animation shows
``real(mode * exp(1j * phase))`` at the chosen amplitude. It is a visualization
of one mode, not a reconstruction of the measured or simulated flow. The
frequency, eigenvalue and normalization are recorded with the mode; the complete
spectrum is saved separately in ``spectrum.npz``. Select another bin with
``--frequency-index`` after examining that spectrum.

Alternatively, the exported real and imaginary mode fields can be used to
reconstruct and animate the modal oscillation directly in ParaView, instead of
precomputing and saving the phase-animation frames in FlowTorch.

The example reads spatial chunks and stores the SVD basis on disk by default.
It does not gather the full spatial modes before writing. RAM still includes
geometry, weights and SVD/SPOD work arrays, and the basis requires temporary disk
space. To batch snapshots as well, use the source ``with_times`` and
``SVD.update`` workflow in :doc:`distributed_svd_spod`, preserving the same
normalization and geometry for every batch.
