"""Windowed Fourier modes and POD-compressed Fourier analysis.

Spectra contain total spatially weighted bin energy, rather than spatially
averaged squared FFT magnitudes. Real spectra include one-sided energy factors.
"""

import logging
from math import isfinite
from typing import Any, Literal, Set

import torch as pt

from flowtorch.provenance import ProvenanceMixin, tracked

from ..constants import FLOAT_TOLERANCE
from .svd import SVD, _prepare_weight
from .state_vector import StateVectorResult, StateVectorSource
from .statistics import DistributedExecution
from .spod import AMSPOD, mode_similarity

logger = logging.getLogger(__name__)

WINDOWS = {
    "boxcar": pt.ones,
    "hann": pt.hann_window,
    "hamming": pt.hamming_window,
    "blackman": pt.blackman_window,
}


class DFT(ProvenanceMixin):
    """Compute Fourier modes of a spatial-by-temporal tensor.

    Modes have unit norm in the supplied spatial inner product. Null modes
    are zero. ``amplitude`` is bin energy, not the square root of energy.
    Centered analyses omit DC from the public spectrum, but preserve the
    window-induced DC internally for inverse-FFT reconstruction.
    """

    def _effective_settings(self):
        names = (
            "_dt",
            "_nfft",
            "_device",
            "_adaptive",
            "_max_tapers",
            "_tol",
            "_n_keep",
            "_subtract_mean",
            "_wname",
            "_nx",
            "_nt",
            "_svd",
        )
        return {
            name.lstrip("_"): vars(self)[name] for name in names if name in vars(self)
        }

    @tracked
    def __init__(
        self,
        data_matrix: pt.Tensor,
        dt: float,
        nfft: int | None = None,
        window: str = "boxcar",
        device: str = "cpu",
        *,
        weight: pt.Tensor | None = None,
        subtract_mean: bool = True,
    ):
        """Transform snapshots along their last axis.

        ``dt`` must be positive and finite. ``nfft`` defaults to the snapshot
        count; smaller values are increased to that count. Positive raw spatial
        weights may cover the full state or repeat across stacked variables.
        Computation uses ``device`` and results reside on CPU. When CUDA is
        unavailable, computation falls back to CPU, as in AMSPOD.
        """
        if data_matrix.ndim != 2 or min(data_matrix.shape) < 1:
            raise ValueError("data_matrix must be a nonempty two-dimensional tensor")
        if not (data_matrix.is_floating_point() or pt.is_complex(data_matrix)):
            raise ValueError("data_matrix must have a floating-point or complex dtype")
        if not isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if nfft is not None and (
            not isinstance(nfft, int) or isinstance(nfft, bool) or nfft < 1
        ):
            raise ValueError("nfft must be a positive integer")
        if window not in WINDOWS:
            raise ValueError(
                f"Unknown window option {window}. Available: {', '.join(WINDOWS)}"
            )
        self._real_dtype = data_matrix.real.dtype
        self._nx, self._nt = data_matrix.shape
        self._complex = pt.is_complex(data_matrix)
        self._subtract_mean = subtract_mean
        self._mean = data_matrix.mean(dim=1).cpu() if subtract_mean else None
        self._dt = dt
        self._nfft = max(nfft or self._nt, self._nt)
        self._wname = window
        self._window = WINDOWS[window](self._nt, dtype=data_matrix.real.dtype)
        if not bool(self._window.square().sum() > 0):
            raise ValueError("window has zero energy for this snapshot count")
        _, sqrt_weight = _prepare_weight(
            weight, self._nx, pt.device("cpu"), data_matrix.real.dtype
        )
        self._sqrt_weight = sqrt_weight
        self._device = str(device)
        if "cuda" in self._device and not pt.cuda.is_available():
            logger.warning("CUDA unavailable; falling back to CPU")
            self._device = "cpu"
        self._frequency = (
            pt.fft.fftfreq(self._nfft, dt, dtype=data_matrix.real.dtype)
            if self._complex
            else pt.fft.rfftfreq(self._nfft, dt, dtype=data_matrix.real.dtype)
        )
        self._start = 1 if subtract_mean else 0
        centered = data_matrix.to(self._device)
        if self._mean is not None:
            centered = centered - self._mean.to(self._device).unsqueeze(-1)
        if window != "boxcar":
            taper = self._window.to(self._device)
            if pt.is_grad_enabled() and centered.requires_grad:
                centered = centered * taper
            else:
                # Reuse an owned buffer, but never overwrite caller storage.
                if self._mean is None and centered is data_matrix:
                    centered = centered.clone()
                centered.mul_(taper)
        fft = pt.fft.fft if self._complex else pt.fft.rfft
        self._modes = fft(
            centered,
            n=self._nfft,
            dim=-1,
            norm="ortho",
        ).cpu()
        if self._sqrt_weight is None:
            self._amplitude = self._modes.norm(dim=0)
        else:
            # Bound the weighted temporary instead of duplicating the spectrum.
            batch_size = max(1, (1024 * 1024) // self._nx)
            self._amplitude = pt.cat(
                [
                    (chunk * self._sqrt_weight).norm(dim=0)
                    for chunk in self._modes.split(batch_size, dim=1)
                ]
            )
        # Used by the shared AMSPOD frequency-segment selector.
        self._adaptive = False

    @property
    def amplitude(self) -> pt.Tensor:
        """Total weighted bin energy, corrected for window energy.

        For boxcar, summing all bins gives the temporal mean of the weighted
        squared state norm. Centered analyses omit DC; non-boxcar tapering can
        create DC even after centering, so the exposed sum then excludes it.
        """
        energy = self._amplitude.square()
        if not self._complex:
            energy[1 : -1 if self._nfft % 2 == 0 else None] *= 2
        return energy[self._start :] / self._window.square().sum()

    @property
    def bin_energy(self) -> pt.Tensor:
        """Explicit alias for :attr:`amplitude`."""
        return self.amplitude

    @property
    def eigvals(self) -> pt.Tensor:
        """Single energy branch, shaped like the AMSPOD spectrum."""
        return self.amplitude.unsqueeze(-1)

    @property
    def frequency(self) -> pt.Tensor:
        """FFT frequencies; DC is exposed only when centering is disabled."""
        return self._frequency[self._start :]

    @property
    def modes(self) -> pt.Tensor | StateVectorResult:
        """Weighted unit-norm physical modes; unresolved bins are zero."""
        return self._normalized_modes(slice(self._start, None))

    def _normalized_modes(self, indices: int | slice) -> pt.Tensor:
        """Normalize only the requested internal bins."""
        norms = self._amplitude[indices]
        threshold = (
            pt.finfo(norms.dtype).eps * max(self._nx, self._nt) * self._amplitude.max()
        )
        inverse = pt.where(
            norms > threshold,
            norms.clamp_min(pt.finfo(norms.dtype).tiny).reciprocal(),
            0,
        )
        return self._modes[:, indices] * inverse

    @property
    def spectral_density(self) -> pt.Tensor:
        """Bin energy per Hz, using spacing ``1 / (nfft * dt)``."""
        return self.amplitude * (self._nfft * self._dt)

    @property
    def reconstruction(self) -> pt.Tensor | StateVectorResult:
        """Inverse FFT using all bins and the retained mean."""
        return self.partial_reconstruction(set(range(len(self.frequency))), True)

    def _validate_index(self, f_idx: int) -> None:
        if (
            not isinstance(f_idx, int)
            or isinstance(f_idx, bool)
            or not 0 <= f_idx < len(self.frequency)
        ):
            raise ValueError(
                f"frequency index must be in [0, {len(self.frequency) - 1}]"
            )

    def get_mode(self, f_idx: int) -> pt.Tensor | StateVectorResult:
        """Return a single normalized physical Fourier mode."""
        self._validate_index(f_idx)
        return self._normalized_modes(f_idx + self._start)

    def partial_reconstruction(
        self, mode_indices: Set[int] | int, include_mean: bool = False
    ) -> pt.Tensor | StateVectorResult:
        """Inverse FFT of selected public bins at the original sampling times.

        For centered data, internal DC is always retained to preserve taper DC.
        ``include_mean`` restores the mean only if centering was enabled.
        Window division amplifies errors where the taper is small; zero-window
        samples cannot be recovered. Hamming has nonzero endpoints and is more
        stable to invert. Real-input reconstruction includes conjugate partners.
        """
        if isinstance(mode_indices, int):
            mode_indices = {mode_indices}
        indices = list(mode_indices)
        for index in indices:
            self._validate_index(index)
        if self._wname in ("hann", "blackman"):
            logger.warning(
                "Recovery is unreliable near zero window values; consider Hamming"
            )
        mask = pt.zeros_like(self._amplitude)
        if self._start:
            mask[0] = 1
        mask[pt.tensor(indices, dtype=pt.int64) + self._start] = 1
        inverse = pt.fft.ifft if self._complex else pt.fft.irfft
        result = inverse(
            (self._modes * mask).to(self._device), n=self._nfft, dim=-1, norm="ortho"
        ).cpu()[:, : self._nt]
        result = result / self._window.clamp_min(FLOAT_TOLERANCE)
        if include_mean and self._mean is not None:
            result = result + self._mean.unsqueeze(-1)
        return result

    def top_modes(
        self,
        n: int = 10,
        f_min: float = -float("inf"),
        f_max: float = float("inf"),
        *,
        spacing: Literal["log", "linear"] = "log",
    ) -> pt.Tensor:
        """Select the strongest bin per frequency segment, as in PAMSPOD.

        ``n`` counts segments, not global peaks. Log spacing excludes DC and
        supports negative frequencies. The upper frequency bound is exclusive.
        """
        return AMSPOD.top_modes(
            self, n=n, eig_idx=0, f_min=f_min, f_max=f_max, spacing=spacing
        )

    def mode_similarity(self) -> pt.Tensor:
        """Absolute weighted similarities across Fourier frequency bins."""
        modes = DFT.modes.fget(self)
        if self._sqrt_weight is not None:
            modes = modes * self._sqrt_weight
        return mode_similarity(modes, modes)

    def show_mode_similarity(
        self, reference_timescale: float | None = None, ax: Any = None, **kwargs: Any
    ):
        """Plot the frequency-by-frequency weighted mode similarities."""
        from flowtorch.visualization import plot_mode_similarity

        return plot_mode_similarity(
            self.frequency,
            self.frequency,
            self.mode_similarity(),
            reference_timescale=reference_timescale,
            ax=ax,
            **kwargs,
        )

    def show_spectrum(
        self,
        n_show: int = 1,
        show_sum: bool = False,
        reference_timescale: float | None = None,
        energy_density: bool = False,
        normalize_energy: bool = False,
        ax: Any = None,
        **kwargs: Any,
    ):
        """Plot bin energy or density using the shared SPOD spectrum helper."""
        from flowtorch.visualization import plot_spod_spectrum

        return plot_spod_spectrum(
            self.frequency,
            self.eigvals,
            n_show=n_show,
            show_sum=show_sum,
            reference_timescale=reference_timescale,
            energy_density=energy_density,
            normalize_energy=normalize_energy,
            ax=ax,
            **kwargs,
        )

    def mode_reconstruction(
        self, f_idx: int, dt: float | None = None, N: int | None = None
    ) -> pt.Tensor | StateVectorResult:
        """Animate a unit-norm mode over one period (25 samples by default).

        This is harmonic animation, not the amplitude-scaled inverse-FFT
        contribution. DC uses the original time step, as in PAMSPOD.
        """
        self._validate_index(f_idx)
        steps = 25 if N is None else N
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError("N must be a positive integer")
        if dt is not None and (not isfinite(dt) or dt <= 0):
            raise ValueError("dt must be finite and positive")
        frequency = self.frequency[f_idx]
        resolution = (
            (
                1 / (frequency.abs() * (steps - 1))
                if frequency != 0 and steps > 1
                else self._dt
            )
            if dt is None
            else dt
        )
        time = pt.arange(steps, dtype=self._real_dtype) * resolution
        mode = self._normalized_modes(f_idx + self._start)
        result = pt.outer(mode, pt.exp(2j * pt.pi * frequency * time))
        return result if self._complex else result.real


class PDFT(DFT):
    """Fourier analysis in a retained, optionally weighted POD subspace.

    SVD centers before fitting the POD basis. Lazy sources use SVD's batched
    and optionally distributed computation; physical results remain lazy.
    """

    @tracked
    def __init__(
        self,
        data_matrix: pt.Tensor | StateVectorSource,
        dt: float,
        rank: int | None = None,
        nfft: int | None = None,
        window: str = "boxcar",
        device: str = "cpu",
        *,
        weight: pt.Tensor | None = None,
        subtract_mean: bool = True,
        prefetch_chunks: bool = False,
        spatial_batch_size: int | None = None,
        execution: DistributedExecution | None = None,
    ):
        """Fit SVD then transform its temporal factors.

        ``rank=None`` retains SVD's automatic rank selection. Centering and
        weighting are performed only by SVD; reduced coordinates are Euclidean.
        See :class:`DFT` for FFT arguments and :class:`SVD` for source batching.
        """
        svd = SVD(
            data_matrix,
            rank,
            weight=weight,
            subtract_mean=subtract_mean,
            prefetch_chunks=prefetch_chunks,
            spatial_batch_size=spatial_batch_size,
            execution=execution,
        )
        self._initialize_from_svd(svd, dt, nfft, window, device)

    @classmethod
    @tracked
    def from_svd(
        cls,
        svd: SVD,
        dt: float,
        nfft: int | None = None,
        window: str = "boxcar",
        device: str = "cpu",
    ) -> "PDFT":
        """Reuse SVD factors, centering and weights without reading snapshots.

        Rebuild PDFT after updating the supplied SVD: spectral results are
        computed at construction and do not automatically track factor changes.
        """
        if not isinstance(svd, SVD):
            raise ValueError("svd must be an SVD instance")
        instance = cls.__new__(cls)
        instance._initialize_from_svd(svd, dt, nfft, window, device)
        return instance

    def _initialize_from_svd(
        self, svd: SVD, dt: float, nfft: int | None, window: str, device: str
    ) -> None:
        self._svd = svd
        reduced = svd.s.unsqueeze(-1).type(svd.V.dtype) * svd.V.conj().T
        DFT.__init__(self, reduced, dt, nfft, window, device, subtract_mean=False)
        # Do not center coefficients again; merely inherit the public DC policy.
        self._start = 1 if svd.mean is not None else 0

    @property
    def svd(self) -> SVD:
        """The decomposition reused for POD projection."""
        return self._svd

    def _project(
        self, coefficients: pt.Tensor, include_mean: bool = False
    ) -> pt.Tensor | StateVectorResult:
        basis = self.svd.U
        mean = self.svd.mean if include_mean else None

        def project(local_basis: pt.Tensor, local_mean: pt.Tensor | None) -> pt.Tensor:
            result = (
                local_basis.to(device=coefficients.device, dtype=coefficients.dtype)
                @ coefficients
            )
            if local_mean is not None:
                result = result + local_mean.to(result.device).reshape(
                    -1, *((1,) * (result.ndim - 1))
                )
            return result

        if isinstance(basis, pt.Tensor):
            return project(basis, mean)

        def produce(spatial_slice: slice) -> pt.Tensor:
            local_mean = mean.read_chunk(spatial_slice) if mean is not None else None
            return project(basis.read_chunk(spatial_slice), local_mean)

        return StateVectorResult(
            basis.layout,
            basis.local_spatial_slice,
            tuple(coefficients.shape[1:]),
            produce,
            basis.spatial_batch_size,
            basis.execution,
        )

    @property
    def modes(self) -> pt.Tensor | StateVectorResult:
        """Weighted unit-norm Fourier modes in physical coordinates."""
        return self._project(DFT.modes.fget(self))

    def get_mode(self, f_idx: int) -> pt.Tensor | StateVectorResult:
        """Project only one Fourier mode to physical coordinates."""
        self._validate_index(f_idx)
        return self._project(self._normalized_modes(f_idx + self._start))

    def partial_reconstruction(
        self, mode_indices: Set[int] | int, include_mean: bool = False
    ) -> pt.Tensor | StateVectorResult:
        """Inverse-FFT reconstruction in physical space, optionally restoring mean."""
        return self._project(
            DFT.partial_reconstruction(self, mode_indices, False), include_mean
        )

    def mode_reconstruction(
        self, f_idx: int, dt: float | None = None, N: int | None = None
    ) -> pt.Tensor | StateVectorResult:
        """Project a unit-norm harmonic animation to physical space."""
        result = DFT.mode_reconstruction(self, f_idx, dt, N)
        return self._project(result)
