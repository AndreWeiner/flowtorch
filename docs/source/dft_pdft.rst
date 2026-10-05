Fourier modes and POD-compressed Fourier analysis
=================================================

``DFT`` analyzes a spatial-by-temporal tensor. ``PDFT`` first fits a POD basis
through ``SVD`` and analyzes its temporal factors. Both default to temporal
centering and a boxcar window (no taper). ``PDFT`` delegates centering to SVD;
it retains the original mean separately and does not center the coefficients
again. ``rank=None`` continues to use SVD's automatic rank selection.

Weighted modes and spectra
--------------------------

Supply positive raw spatial weights through ``weight=``. Physical modes have
unit weighted norm, and unresolved Fourier modes are zero. ``amplitude`` and
its explicit alias ``bin_energy`` return total spatially weighted bin energy,
with window-energy correction and one-sided doubling for real signals except
at DC and Nyquist. ``spectral_density`` divides this energy by the frequency
spacing. This matches AMSPOD/PAMSPOD's total-energy convention, although their
multi-taper estimator produces a different spectrum.

For an untapered centered signal, the sum of bin energies equals the temporal
mean of the total weighted squared fluctuation norm. Centered public spectra
omit DC. Tapering can produce DC even after centering; that internal bin is
retained for reconstruction but excluded from the public spectrum.
``subtract_mean=False`` exposes DC at index zero.

Compatibility changes
---------------------

* Previously, ``amplitude`` divided squared orthonormal FFT magnitudes by
  spatial size and mean window energy. It now divides by summed window energy
  and includes spatial weights. Without weights, multiply old amplitudes by
  ``n_spatial / n_snapshots`` to obtain the new values.
* PDFT now centers before fitting POD. Truncated bases and spectra can change;
  mean restoration uses the original mean rather than its truncated projection.
* ``top_modes()`` defaults to ten logarithmic frequency segments, matching
  PAMSPOD. It returns one strongest bin per nonempty segment, in segment order,
  rather than globally ranked peaks. ``n=1`` selects one strongest nonzero bin.
  ``spacing="linear"`` also permits DC selection. Negative frequencies are
  supported and the upper frequency bound is exclusive.
* Existing positional constructor arguments are retained; new preprocessing
  and source arguments are keyword-only. Complex PDFT coefficients now use
  the conjugate transpose of SVD's right factors.

Reuse and lazy inputs
---------------------

.. code-block:: python

   svd = SVD(source, rank=20, subtract_mean=True,
             spatial_batch_size=100_000, execution=execution)
   pdft = PDFT.from_svd(svd, dt=0.001)
   for spatial_slice, fields in pdft.get_mode(5).iter_chunks(split=True):
       mode = fields["q"]

``from_svd`` uses stored singular values and right factors without rereading
snapshots. It inherits the supplied SVD's centering and weights. Rebuild PDFT
after ``svd.update()``; an existing spectral analysis does not refresh itself.
``PDFT`` also accepts a source directly, forwarding spatial/snapshot batching
and distributed execution to SVD. Lazy physical modes and reconstructions
return ``StateVectorResult``; tensor inputs return tensors. ``get_mode`` projects
only the requested bin. Reduced spectral results are replicated across ranks.

Reconstruction and inspection
-----------------------------

``partial_reconstruction`` uses inverse FFT at the original sampling times.
For real inputs, negative-frequency conjugate partners are implicit.
``include_mean=True`` restores the retained mean; ``reconstruction`` includes
all bins and the mean. No projection-derived temporal coefficients are used.

Non-boxcar reconstruction divides by the taper. Small taper values amplify
errors, and samples where the taper vanishes cannot be recovered. Hamming's
nonzero endpoints make it more stable to invert than Hann or Blackman. Partial
frequency reconstructions can still be strongly distorted near weak endpoints.

``mode_reconstruction`` animates a normalized mode over one period, defaulting
to 25 samples; it is not the amplitude-scaled contribution of that bin to the
original signal. DC animation uses the original sampling interval.
``show_spectrum`` supports density, normalization, and Strouhal scaling.
``mode_similarity`` and ``show_mode_similarity`` compare weighted spatial
Fourier modes across frequencies; PDFT performs these comparisons in reduced
coordinates. Similarities involving unresolved modes are NaN, and the full
similarity matrix requires quadratic storage in the frequency-bin count.
