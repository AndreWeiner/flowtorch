"""PyTorch implementation of adaptive multi-taper SPOD.

The adaptive multi-taper spectral proper orthogonal decomposition (AMSPOD)
uses a frequency-variable number of sine-tapers per bin. The main references are:

- Yeung, B. C. Y., Schmidt, O. T.: Adaptive spectral proper orthogonal decomposition
    of broadband-tonal flows, Theor. Comput. Fluid Dyn. 38, 355-374, 2024,
    DOI 10.1007/s00162-024-00695-0
- Matlab implementation by the original authors available at:
    https://github.com/SpectralPOD/spod_matlab/blob/master/spod_adapt.m

Compared to the original implementation, multiple realizations of the DFT
are generated only by multiple tapers. There is no additional subdivision into
overlapping blocks. Moreover, the compressed version is implemented as a
lightweight derived class wrapping around the `AMSPOD`, termed `PAMSPOD`.
"""

# standard library packages
import logging
import gc
import warnings
from typing import Any, Literal, Union, Tuple
from math import sqrt
from collections import defaultdict

# third party packages
import torch as pt
from numpy import pi

# flowTorch packages
from .svd import SVD

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _prepare_weights(w: pt.Tensor, nx: int) -> pt.Tensor:
    """Prepare state vector weights.

    :param w: weights for the entire state vector or one variable within the state vector
    :type w: pt.Tensor
    :param nx: size of the state vector
    :type nx: int
    :raises ValueError: if the state vector size is no exact multiple of the weight vector size
    :return: weight vector ready for columnwise multiplication with the data matrix
    :rtype: pt.Tensor
    """
    w = w.sqrt().unsqueeze(-1)
    nxw = w.shape[0]
    if nxw != nx:
        if nx % nxw == 0:
            w = w.repeat((nx // nxw, 1))
        else:
            raise ValueError(
                f"The size of the weight vector ({nxw:d}) does not match the size of the state ({nx:d})."
            )
    return w


def mode_similarity(first_modes: pt.Tensor, second_modes: pt.Tensor) -> pt.Tensor:
    r"""Compute normalized similarities between two sets of modes.

    Modes are arranged columnwise, with tensors of shape ``(n_state, n_modes)``.
    The returned matrix contains the absolute normalized inner products

    .. math::

        \rho_{ij} = \frac{|\phi_i^*\psi_j|}
        {\|\phi_i\|_2\|\psi_j\|_2},

    and therefore has shape ``(n_first_modes, n_second_modes)``. The absolute
    value makes the result invariant to the sign or complex phase of either
    mode. To compute similarities in a weighted inner product, multiply both
    mode sets by the square root of the weights before calling this function.

    Similarities involving zero-norm or non-finite modes are returned as NAN.

    :param first_modes: first columnwise set of modes
    :type first_modes: pt.Tensor
    :param second_modes: second columnwise set of modes
    :type second_modes: pt.Tensor
    :raises ValueError: if either input is not two-dimensional, the state
        dimensions differ, or the tensors are on different devices
    :return: pairwise normalized mode similarities
    :rtype: pt.Tensor
    """
    if first_modes.ndim != 2 or second_modes.ndim != 2:
        raise ValueError("mode sets must be two-dimensional tensors")
    if not (first_modes.is_floating_point() or pt.is_complex(first_modes)):
        raise ValueError("mode sets must have a floating-point or complex dtype")
    if not (second_modes.is_floating_point() or pt.is_complex(second_modes)):
        raise ValueError("mode sets must have a floating-point or complex dtype")
    if first_modes.shape[0] != second_modes.shape[0]:
        raise ValueError("mode sets must have the same state dimension")
    if first_modes.device != second_modes.device:
        raise ValueError("mode sets must be on the same device")

    dtype = pt.promote_types(first_modes.dtype, second_modes.dtype)
    first = first_modes.to(dtype=dtype)
    second = second_modes.to(dtype=dtype)
    products = first.conj().T @ second
    denominator = pt.outer(
        pt.linalg.vector_norm(first, dim=0),
        pt.linalg.vector_norm(second, dim=0),
    )
    finite = pt.logical_and(pt.isfinite(products.abs()), pt.isfinite(denominator))
    valid = pt.logical_and(finite, denominator > 0.0)
    similarity = products.abs() / denominator
    similarity = similarity.clamp(0.0, 1.0)
    return pt.where(valid, similarity, pt.full_like(similarity, float("nan")))


def _calc_mode_similarity(first: pt.Tensor, second: pt.Tensor) -> float:
    """Compute normalized similarity between two 1D tensors.

    :param first: first 1D tensor
    :type first: pt.Tensor
    :param second: second 1D tensor
    :type second: pt.Tensor
    :return: absolute value of the dot-product between the two tensors normalized
        by the product of their norms
    :rtype: float
    """
    return mode_similarity(first.unsqueeze(-1), second.unsqueeze(-1))[0, 0].item()


def _free_memory(device: str):
    """Helper function to free memory.

    :param device: device to clear
    :type device: str
    """
    if "cuda" in device:
        pt.cuda.empty_cache()
    else:
        gc.collect()


class AMSPOD(object):
    """Class to manage AMSPOD computation without compression."""

    def __init__(
        self,
        data_matrix: pt.Tensor,
        dt: float,
        nfft: Union[int, None] = None,
        adaptive: bool = True,
        max_tapers: int = 50,
        tolerance: float = 1.0e-5,
        weight: Union[pt.Tensor, None] = None,
        subtract_mean: bool = True,
        keep_n_modes: int = 3,
        device: str = "cpu",
        verbose: bool = False,
    ):
        """Define and compute AMSPOD.

        :param data_matrix: time series of snapshots arranged as 2D tensor with the
            1st dimension (rows, size M) denoting space and the 2nd dimension denoting
            time (columns, size N)
        :type data_matrix: pt.Tensor
        :param dt: time increment between snapshots; assumed to be constant
        :type dt: float
        :param nfft: number of frequency bins used when computing the DFT; zero-padding is applied
            if nfft > N; if nfft is None or nfft <= N, then nfft=N is set; defaults to None
        :type nfft: Union[int, None], optional
        :param adaptive: choose the number of tapers adaptively for each frequency bin;
            adaptivity reduces the bias for quickly converging frequencies (see `tolerance`);
            defaults to True
        :type adaptive: bool, optional
        :param max_tapers: max. number of tapers to use when `adaptive=True` or constant number
            of tapers for all frequencies if `adaptive=False`; the minimum number of tapers is 2; defaults to 50
        :type max_tapers: int, optional
        :param tolerance: tolerance at which the leading mode of each frequency is considered
            to be converged; the difference is measured by comparing the leading mode
            with i and i+1 tapers for increasing values of i, defaults to 1.0e-5
        :type tolerance: float, optional
        :param weight: weight vector for reducing the mesh-induced bias when computing inner products;
            should be of length M; defaults to None
        :type weight: Union[pt.Tensor, None], optional
        :param subtract_mean: subtract mean before computing the DFT, defaults to True
        :type subtract_mean: bool, optional
        :param keep_n_modes: number of spatial modes to keep, defaults to 3
        :type keep_n_modes: int, optional
        :param device: device used for computing; defaults to "cpu"
        :type device: str, optional
        :param verbose: extended output of log messages; defaults to False
        :type verbose: str, optional
        """
        self._dm = data_matrix
        self._nx, self._nt = data_matrix.shape
        self._complex = pt.is_complex(data_matrix)
        self._dt = dt
        if nfft is None:
            self._nfft = self._nt
        else:
            self._nfft = max(self._nt, nfft)
        self._adaptive = adaptive
        self._max_tapers = max(2, min(max_tapers, self._nx, self._nt))
        self._tol = tolerance
        if weight is None:
            self._weight = pt.ones(self._nx).unsqueeze(-1)
        else:
            self._weight = _prepare_weights(weight, self._nx)
        self._weight = self._weight.type(self._dm.dtype)
        self._subtract_mean = subtract_mean
        self._n_keep = keep_n_modes
        self._device = device
        if "cuda" in device and not pt.cuda.is_available():
            logger.warning(
                f"selected device is '{device}' but cuda is not available; falling back to 'cpu'"
            )
            self._device = "cpu"
        self._verbose = verbose
        self._log: dict[str, Any] = {}
        self._taper_norm = sqrt(2.0 / (self._nfft + 1.0))
        self._modes, self._eigvals, self._frequency = self._spod()

    def _spod(self) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        """Main driver method for computing the SPOD.

        This method
        - prepares the weighted, mean-subtracted data matrix
        - computes the DFT of the untapered data
        - determines the number of tapers per frequency
        - computes the SPOD modes and eigenvalues for each frequency

        :return: SPOD eigenvectors (modes), SPOD eigenvalues, bin-frequencies
        :rtype: Tuple[pt.Tensor, pt.Tensor, pt.Tensor]
        """
        Q_var = self._dm.clone()
        if self._subtract_mean:
            logger.info("computing and subtracting temporal mean")
            Q_var -= Q_var.mean(dim=1).unsqueeze(-1)
        Q_var *= self._weight
        f = (
            pt.fft.fftfreq(self._nfft, self._dt)
            if pt.is_complex(self._dm)
            else pt.fft.rfftfreq(self._nfft, self._dt)
        )
        logger.info(f"computing untapered FFT on device '{self._device}'")
        Q_hat = pt.fft.fft(
            Q_var.to(self._device), n=2 * self._nfft, dim=1, norm="ortho"
        ) * sqrt(self._dt)
        del Q_var
        _free_memory(self._device)
        n_freq = f.shape[0]
        n_win = self._determine_n_tapers(Q_hat, n_freq)
        n_keep = min(self._max_tapers, self._n_keep)
        modes = pt.empty((n_freq, self._nx, n_keep), dtype=Q_hat.dtype)
        evals = pt.zeros((n_freq, self._max_tapers), dtype=Q_hat.dtype)
        logger.info(f"computing SPOD for {n_freq} frequency bins")
        for i in range(n_freq):
            if self._verbose:
                logger.info(
                    f"using {int(n_win[i])} tapers for bin {i:d} (f={f[i].item():1.4f}Hz)"
                )
            m, ev = self._spod_at_freq(Q_hat, i, int(n_win[i]))
            m = m / self._weight.type(m.dtype).to(self._device)
            modes[i] = m[:, : min(n_keep, int(n_win[i]))].cpu()
            evals[i, : int(n_win[i])] = (
                ev.cpu() * 2 * self._nfft / (self._dt * self._nt)
            )
        return modes, evals, f

    def _parabolic_weights(self, n_tapers: int) -> pt.Tensor:
        """Generate parabolic window weights for a given number of tapers/windows.

        The method implements formula (11) of the reference article. The same factorized
        formula as in the Matlab implementation is used.

        :param n_tapers: number of sin-tapers/windows
        :type n_tapers: int
        :return: parabolic window weights decreasing with window order
        :rtype: pt.Tensor
        """
        k = pt.arange(1, n_tapers + 1, device=self._device).type(self._dm.dtype)
        mu = (
            6.0
            / (n_tapers * (4.0 * n_tapers - 1.0) * (n_tapers + 1.0))
            * (n_tapers**2 - (k - 1.0) ** 2)
        )
        return mu

    def _sin_taper_dft(self, Q_hat: pt.Tensor, f_idx: int, n_tapers: int) -> pt.Tensor:
        """Compute the sin-tapered DFT based on the untapered DFT.

        The bin difference formula is used to compute the DFT for various tapers
        all at once rather than computing the DFT of various tapered data matrices.
        Refer to formula (13) of the reference article.

        :param Q_hat: DFT of the untapered data of size M x 2*nfft
        :type Q_hat: pt.Tensor
        :param f_idx: index of the frequency bin for which to compute tapered versions
        :type f_idx: int
        :param n_tapers: number of sin-tapers
        :type n_tapers: int
        :return: tapered DFT modes of the selected frequency bin for specified number of
            tapers (K) arranged as tensor of shape M x K
        :rtype: pt.Tensor
        """
        idx_c = 2 * f_idx
        shifts = pt.arange(1, n_tapers + 1, device=self._device)
        idx_l = (idx_c - shifts) % (2 * self._nfft)
        idx_u = (idx_c + shifts) % (2 * self._nfft)
        return self._taper_norm * (Q_hat[:, idx_l] - Q_hat[:, idx_u]) / (2j)

    def _spod_at_freq(
        self, Q_hat: pt.Tensor, f_idx: int, n_tapers: int
    ) -> Tuple[pt.Tensor, pt.Tensor]:
        """Compute the SPOD at a single frequency for given number of tapers.

        :param Q_hat: DFT of the untapered data of size M x 2*nfft
        :type Q_hat: pt.Tensor
        :param f_idx: index of the frequency bin for which to compute the SPOD
        :type f_idx: int
        :param n_tapers: number of sin-tapers
        :type n_tapers: int
        :return: eigenvectors (M x K) and eigenvalues (K) of cross spectral density matrix; both tensors are sorted
            according to the (real-valued) eigenvalues in descending order
        :rtype: Tuple[pt.Tensor, pt.Tensor]
        """
        Q_f = self._sin_taper_dft(Q_hat, f_idx, n_tapers) * self._parabolic_weights(
            n_tapers
        ).sqrt().type(Q_hat.dtype)
        if self._nx >= n_tapers:
            vals, vecs = pt.linalg.eigh(Q_f.conj().T @ Q_f)
            vecs = Q_f @ (vecs / vals.sqrt())
        else:
            vals, vecs = pt.linalg.eigh(Q_f @ Q_f.conj().T)
        return vecs.flip(dims=[-1]), vals.flip(dims=[-1])

    def _determine_n_tapers(self, Q_hat: pt.Tensor, n_freq: int) -> pt.Tensor:
        """Determine the number of tapers for each frequency bin.

        If the selection is adaptive, the optimal number of tapers is determined for each frequency
        individually by tracking the change in the leading mode between increasing numbers of tapers.
        A frequency-mode-pair is considered converged if:
            1) the change in the mode is below a user-defined tolerance
            2) the difference in the taper number between two bins exceeds 1
        If the selection is non-adaptive, the user-defined maximum number of tapers
        is used for each bin


        :param Q_hat: un-tapered DFT of weighted data matrix
        :type Q_hat: pt.Tensor
        :param n_freq: number of frequency bins
        :type n_freq: int
        :return: number of tapers per bin to use for the final SPOD
        :rtype: pt.Tensor
        """
        n_tapers = pt.ones(n_freq, dtype=pt.int32) * 2
        if self._adaptive:
            logger.info("performing adaptive taper selection")
            converged = pt.zeros(n_freq, dtype=pt.bool)
            prev_modes = pt.ones(
                (n_freq, self._nx), dtype=Q_hat.dtype, device=self._device
            )
            self._log["convergence"] = defaultdict(list)
            itr = 0
            while not bool(converged.all()):
                if self._verbose:
                    logger.info(
                        f"iteration {itr} - {converged.sum().item()}/{n_freq} frequency bins converged"
                    )
                for f_idx in range(n_freq):
                    if converged[f_idx]:
                        continue
                    K = int(n_tapers[f_idx])
                    modes, _ = self._spod_at_freq(Q_hat, f_idx, K)
                    similarity = _calc_mode_similarity(modes[:, 0], prev_modes[f_idx])
                    if itr > 0:
                        self._log["convergence"][f_idx].append(similarity)
                    prev_modes[f_idx] = modes[:, 0]
                    if similarity >= 1.0 - self._tol or K >= self._max_tapers:
                        converged[f_idx] = True
                    else:
                        if f_idx > 0:
                            if K - n_tapers[f_idx - 1] == 1:
                                n_tapers[f_idx] = K
                                converged[f_idx] = True
                        if f_idx < n_freq - 1:
                            if K - n_tapers[f_idx + 1] == 1:
                                n_tapers[f_idx] = K
                                converged[f_idx] = True
                        if not converged[f_idx]:
                            n_tapers[f_idx] = K + 1
                itr += 1
        else:
            n_tapers[:] = self._max_tapers
        self._log["n_tapers"] = n_tapers.clone()
        return n_tapers

    @property
    def frequency(self) -> pt.Tensor:
        """SPOD bin frequencies.

        For real-valued input data, only positive frequencies are returned.

        :return: SPOD bin frequencies
        :rtype: pt.Tensor
        """
        return self._frequency

    @property
    def half_bandwidth(self) -> Union[float, pt.Tensor]:
        """Resulting sin-taper half-bandwidth.

        The half-bandwidth quantifies the effective frequency resolution.
        A larger bandwidth leads to a greater smoothing of the spectrum.
        The effect of smoothing is lower variance (of the eigenvalues) but
        also poorer effective frequency resolution.

        :return: constant half-bandwidth if the number of windows is constant;
            otherwise, one half-bandwidth per frequency bin
        :rtype: Union[float, pt.Tensor]
        """
        n_tapers = self.log["n_tapers"] if self._adaptive else self._max_tapers
        return 0.5 * (n_tapers + 1) / (self._nfft + 1) / self._dt

    @property
    def eigvals(self) -> pt.Tensor:
        """Eigenvalues of the cross-spectral density realizations.

        - the absolute value of each eigenvalue is returned
        - for real input data, all eigenvalues are doubled, except for
          zero and Nyquist frequencies to account for the energy content
          of the negative part of the spectrum

        :return: absolute value of eigenvalues accounting for one-sided spectra
        :rtype: pt.Tensor
        """
        ev = self._eigvals.abs()
        if not self._complex:
            if self._nfft % 2 == 0:
                ev[1:-1] *= 2
            else:
                ev[1:] *= 2
        return ev

    def show_spectrum(
        self,
        n_show: int = 3,
        show_sum: bool = True,
        reference_timescale: Union[float, None] = None,
        energy_density: bool = False,
        normalize_energy: bool = False,
        ax: Any = None,
        **kwargs: Any,
    ):
        r"""Plot the SPOD eigenvalue spectrum and its half-bandwidth.

        This convenience method passes :attr:`frequency`, :attr:`eigvals`, and
        :attr:`half_bandwidth` to
        :func:`flowtorch.visualization.plot_spod_spectrum`. The returned figure
        and axes remain fully modifiable.

        :param n_show: maximum number of leading eigenvalues, defaults to 3
        :type n_show: int, optional
        :param show_sum: plot the sum over all eigenvalues, defaults to ``True``
        :type show_sum: bool, optional
        :param reference_timescale: reference time used to plot against
            Strouhal number
        :type reference_timescale: float, optional
        :param energy_density: convert bin energy to density in the displayed
            horizontal coordinate, defaults to ``False``
        :type energy_density: bool, optional
        :param normalize_energy: divide by the total resolved spectral energy,
            defaults to ``False``
        :type normalize_energy: bool, optional
        :param ax: existing Matplotlib axes
        :type ax: matplotlib.axes.Axes, optional
        :param kwargs: additional arguments for ``plot_spod_spectrum``
        :return: modifiable Matplotlib figure and axes
        :rtype: Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        """
        from flowtorch.visualization import plot_spod_spectrum

        return plot_spod_spectrum(
            self.frequency,
            self.eigvals,
            n_show=n_show,
            show_sum=show_sum,
            reference_timescale=reference_timescale,
            energy_density=energy_density,
            normalize_energy=normalize_energy,
            half_bandwidth=self.half_bandwidth,
            ax=ax,
            **kwargs,
        )

    def show_time_coefficients(
        self,
        n_modes: int = 1,
        reference_timescale: Union[float, None] = None,
        ax: Any = None,
        **kwargs: Any,
    ):
        r"""Plot the L2 amplitude of the SPOD temporal coefficients.

        Temporal coefficients are computed by oblique projection before they
        are passed with :attr:`frequency` and the sampling times to
        :func:`flowtorch.visualization.plot_spod_time_coefficients`. Computing
        the coefficients can be substantially more expensive than plotting
        the eigenvalue spectrum.

        :param n_modes: number of leading modes included per frequency,
            defaults to 1
        :type n_modes: int, optional
        :param reference_timescale: common scale for nondimensional time and
            frequency
        :type reference_timescale: float, optional
        :param ax: existing Matplotlib axes
        :type ax: matplotlib.axes.Axes, optional
        :param kwargs: additional arguments for ``plot_spod_time_coefficients``
        :return: modifiable Matplotlib figure and axes
        :rtype: Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        """
        from flowtorch.visualization import plot_spod_time_coefficients

        coefficients = self.temporal_coefficients(n_modes=n_modes)
        time = pt.arange(self._nt, dtype=self._dm.real.dtype, device=self._device)
        time *= self._dt
        return plot_spod_time_coefficients(
            time,
            self.frequency,
            coefficients,
            n_modes=coefficients.shape[1],
            reference_timescale=reference_timescale,
            ax=ax,
            **kwargs,
        )

    @property
    def modes(self) -> pt.Tensor:
        """Leading eigenvectors (modes) of the cross spectral density realizations.

        :return: user-defined number of eigenvectors; see `keep_n_modes`
        :rtype: pt.Tensor
        """
        return self._modes

    @property
    def log(self) -> dict:
        """Log data stored throughout the computation.

        :return: dictionary with various useful logs
        :rtype: dict
        """
        return self._log

    @property
    def residual(self) -> Union[pt.Tensor, None]:
        """Residual of the leading mode during adaptive refinement.

        :return: residual computed as ``abs(1 - mode_similarity)``; since the
            number of tapers varies per bin, a tensor of size n_freq x
            (n_tapers - 2) is filled with NANs, and then the residuals are
            overwritten if available; (n_tapers - 2) is a consequence of the
            minimum number of tapers (2) and the residual being computed from
            the mode similarity between two consecutive taper values, so in
            the first iteration, there is no sensible similarity
        :rtype: pt.Tensor
        """
        if self._adaptive:
            n_freq = self.frequency.shape[0]
            n_max = self._max_tapers - 2
            sim = self._log["convergence"]
            res = pt.full((n_freq, n_max), float("nan"), dtype=pt.float64)
            for key in sim.keys():
                tmp = (pt.tensor(sim[key]) - 1.0).abs()
                res[int(key), : tmp.shape[0]] = tmp
            return res
        else:
            logger.warning("residuals are only available for adaptive taper selection")
            return None

    def show_residual(
        self,
        reference_timescale: Union[float, None] = None,
        ax: Any = None,
        **kwargs: Any,
    ):
        r"""Plot the adaptive leading-mode convergence residual.

        For AMSPOD, ``r`` is the absolute difference between one and the
        leading-mode similarity at consecutive taper counts. Residuals are
        available only when adaptive taper selection was enabled.

        :param reference_timescale: reference time used to plot Strouhal number
        :type reference_timescale: float, optional
        :param ax: existing Matplotlib axes
        :type ax: matplotlib.axes.Axes, optional
        :param kwargs: additional arguments for ``plot_adaptive_residual``
        :raises RuntimeError: if adaptive taper selection was disabled
        :return: modifiable Matplotlib figure and axes
        :rtype: Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        """
        if not self._adaptive:
            raise RuntimeError("residuals require adaptive taper selection")
        from flowtorch.visualization import plot_adaptive_residual

        residual = self.residual
        if residual is None:
            raise RuntimeError("adaptive residuals are unavailable")
        return plot_adaptive_residual(
            self.frequency,
            residual,
            reference_timescale=reference_timescale,
            ax=ax,
            **kwargs,
        )

    def mode_similarity(
        self, first_mode_idx: int = 0, second_mode_idx: int = 0
    ) -> pt.Tensor:
        r"""Compare two SPOD mode branches across all frequency bins.

        Entry ``(i, j)`` of the returned tensor is the normalized, absolute
        weighted inner product between ``first_mode_idx`` at frequency bin
        ``i`` and ``second_mode_idx`` at frequency bin ``j``. Consequently,
        the result has shape ``(n_frequency, n_frequency)``. For equal mode
        indices it is symmetric, with a unit diagonal wherever the mode is
        available.

        AMSPOD modes are compared in the spatial inner product supplied at
        construction. PAMSPOD inherits this implementation and compares its
        reduced modes directly, which is equivalent to the weighted
        full-space product within the retained POD subspace.

        Similarities involving a mode unavailable at an adaptive frequency
        bin are returned as NAN.

        :param first_mode_idx: mode index used along the first matrix axis,
            defaults to 0
        :type first_mode_idx: int, optional
        :param second_mode_idx: mode index used along the second matrix axis,
            defaults to 0
        :type second_mode_idx: int, optional
        :raises ValueError: if either mode index is invalid or was not retained
        :return: frequency-by-frequency mode similarity matrix
        :rtype: pt.Tensor
        """
        n_modes = self._modes.shape[-1]
        for name, index in (
            ("first_mode_idx", first_mode_idx),
            ("second_mode_idx", second_mode_idx),
        ):
            if not isinstance(index, int) or isinstance(index, bool):
                raise ValueError(f"{name} must be an integer")
            if index < 0 or index >= n_modes:
                raise ValueError(f"{name} must be in the range [0, {n_modes - 1:d}]")

        first_modes = self._modes[:, :, first_mode_idx].T
        second_modes = self._modes[:, :, second_mode_idx].T
        weight = self._weight.to(device=first_modes.device, dtype=first_modes.dtype)
        similarities = mode_similarity(first_modes * weight, second_modes * weight)

        if self._adaptive:
            n_tapers = self._log["n_tapers"].to(similarities.device)
            first_available = n_tapers > first_mode_idx
            second_available = n_tapers > second_mode_idx
            available = pt.logical_and(
                first_available.unsqueeze(-1), second_available.unsqueeze(0)
            )
            similarities = similarities.masked_fill(~available, float("nan"))
        return similarities

    def show_mode_similarity(
        self,
        first_mode_idx: int = 0,
        second_mode_idx: int = 0,
        reference_timescale: Union[float, None] = None,
        ax: Any = None,
        **kwargs: Any,
    ):
        r"""Plot mode similarity across the SPOD frequency spectrum.

        Symmetric matrices are shown as their lower triangle including the
        diagonal by default. Plot behavior can be changed with the
        ``triangle`` argument accepted by
        :func:`flowtorch.visualization.plot_mode_similarity`.

        :param first_mode_idx: mode index used along the horizontal axis,
            defaults to 0
        :type first_mode_idx: int, optional
        :param second_mode_idx: mode index used along the vertical axis,
            defaults to 0
        :type second_mode_idx: int, optional
        :param reference_timescale: reference time used to plot Strouhal number
        :type reference_timescale: float, optional
        :param ax: existing Matplotlib axes
        :type ax: matplotlib.axes.Axes, optional
        :param kwargs: additional arguments for ``plot_mode_similarity``
        :return: modifiable Matplotlib figure and axes
        :rtype: Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        """
        from flowtorch.visualization import plot_mode_similarity

        similarities = self.mode_similarity(first_mode_idx, second_mode_idx)
        return plot_mode_similarity(
            self.frequency,
            self.frequency,
            similarities,
            reference_timescale=reference_timescale,
            ax=ax,
            **kwargs,
        )

    def top_modes(
        self,
        n: int = 10,
        eig_idx: Union[int, str] = 0,
        f_min: float = -float("inf"),
        f_max: float = float("inf"),
        spacing: Literal["log", "linear"] = "log",
    ) -> pt.Tensor:
        """Get the indices of dominant modes in frequency segments.

        Segments can be equally spaced on a logarithmic or linear frequency
        coordinate. Negative frequencies are supported for both choices. Log
        spacing uses a signed logarithmic coordinate and omits the
        zero-frequency bin because it has no logarithmic representation.
        Linear spacing includes the zero-frequency bin when it is within the
        requested frequency range.

        :param n: number of segments into which to divide the frequency range;
            defaults to 10
        :type n: int
        :param eig_idx: eigenvalue index according to which to rank the modes;
            if ``eig_idx="sum"``, the sum of all eigenvalues in a bin is used;
            for adaptive SPOD, the index is limited to the minimum number of
            tapers across all bins;
            defaults to 0
        :type eig_idx: Union[int, str], optional
        :param f_min: consider only modes with a frequency larger or equal
            to f_min; defaults to -inf
        :type f_min: float, optional
        :param f_max: consider only modes with a frequency smaller than f_max;
            defaults to inf
        :type f_max: float, optional
        :param spacing: distribute segment boundaries logarithmically or
            linearly in frequency; defaults to ``"log"``
        :type spacing: Literal["log", "linear"], optional
        :raises ValueError: if ``spacing`` is neither ``"log"`` nor
            ``"linear"``
        :return: frequency-bin index corresponding to the largest eigenvalue in
            each non-empty frequency segment
        :rtype: pt.Tensor
        """
        if n < 1:
            raise ValueError("n must be at least 1")
        if spacing not in ("log", "linear"):
            raise ValueError("spacing must be 'log' or 'linear'")
        if eig_idx != "sum":
            if not isinstance(eig_idx, int):
                raise ValueError("eig_idx must be an integer or 'sum'")
            if eig_idx < 0:
                raise ValueError("eig_idx must be non-negative")
            if self._adaptive:
                n_available = int(self._log["n_tapers"].min().item())
                if eig_idx >= n_available:
                    warnings.warn(
                        f"eig_idx={eig_idx:d} exceeds the minimum number of "
                        f"available modes per frequency bin ({n_available:d}); "
                        f"using eig_idx={n_available - 1:d}",
                        UserWarning,
                    )
                    eig_idx = n_available - 1
            elif eig_idx >= self.eigvals.shape[-1]:
                raise ValueError(
                    f"eig_idx must be in the range [0, {self.eigvals.shape[-1] - 1:d}]"
                )

        modes_in_range = pt.logical_and(self.frequency >= f_min, self.frequency < f_max)
        if spacing == "log":
            modes_in_range = pt.logical_and(modes_in_range, self.frequency != 0.0)
        mode_indices = pt.arange(modes_in_range.shape[0], dtype=pt.int64)[
            modes_in_range
        ]
        if mode_indices.numel() == 0:
            return mode_indices

        freq = self.frequency[mode_indices]
        if spacing == "linear":
            segment_coordinate = freq
        elif bool((freq > 0.0).all()):
            segment_coordinate = pt.log10(freq)
        elif bool((freq < 0.0).all()):
            segment_coordinate = -pt.log10(freq.abs())
        else:
            f_ref = freq.abs().min()
            segment_coordinate = freq.sign() * pt.log10(freq.abs() / f_ref + 1.0)
        edges = pt.linspace(
            segment_coordinate.min().item(),
            segment_coordinate.max().item(),
            n + 1,
            dtype=freq.dtype,
            device=freq.device,
        )
        top_modes = []
        eigvals = self.eigvals
        score = eigvals[:, eig_idx] if eig_idx != "sum" else eigvals.sum(dim=1)
        for i in range(n):
            if i == n - 1:
                in_segment = pt.logical_and(
                    segment_coordinate >= edges[i],
                    segment_coordinate <= edges[i + 1],
                )
            else:
                in_segment = pt.logical_and(
                    segment_coordinate >= edges[i],
                    segment_coordinate < edges[i + 1],
                )
            segment_indices = mode_indices[in_segment]
            if segment_indices.numel() > 0:
                top_modes.append(segment_indices[score[segment_indices].argmax()])
        return pt.stack(top_modes) if top_modes else mode_indices[:0]

    def _valid_n_coeff_modes(self, n_modes: int) -> int:
        """Limit coefficient mode count to modes available at every frequency."""
        if n_modes < 1:
            raise ValueError("n_modes must be at least 1")
        if n_modes > self._n_keep:
            warnings.warn(
                f"n_modes={n_modes:d} exceeds keep_n_modes={self._n_keep:d}; "
                f"using n_modes={self._n_keep:d}",
                UserWarning,
            )
            n_modes = self._n_keep
        n_available = self._modes.shape[-1]
        if self._adaptive:
            n_available = min(n_available, int(self._log["n_tapers"].min().item()))
        if n_modes > n_available:
            warnings.warn(
                f"n_modes={n_modes:d} exceeds the minimum number of available "
                f"modes per frequency bin ({n_available:d}); using n_modes={n_available:d}",
                UserWarning,
            )
            n_modes = n_available
        return n_modes

    def temporal_coefficients(self, n_modes: int = 1) -> pt.Tensor:
        """Compute SPOD time coefficients by oblique projection.

        The leading ``n_modes`` from all frequency bins are collected in a
        single non-orthogonal basis. The coefficients are obtained with the
        weighted oblique projection described in frequency-time SPOD analysis,
        i.e. by solving the normal equations for the weighted spatial modes.
        See Nekkanti, A. and Schmidt, O. T., "Frequency-time analysis,
        low-rank reconstruction and denoising of turbulent flows using SPOD",
        Journal of Fluid Mechanics, 926, A26, 2021,
        https://doi.org/10.1017/jfm.2021.681.

        :param n_modes: number of leading modes per frequency bin used in the
            projection; if larger than ``keep_n_modes`` or, for adaptive SPOD,
            larger than the minimum number of available modes per bin, the
            value is reduced with a warning; defaults to 1
        :type n_modes: int, optional
        :raises ValueError: if ``n_modes`` is smaller than one
        :return: temporal coefficients arranged as
            ``(n_frequency, n_modes, n_snapshots)``
        :rtype: pt.Tensor
        """
        n_modes = self._valid_n_coeff_modes(n_modes)
        modes = self._modes[:, :, :n_modes].permute(1, 0, 2)
        basis = modes.reshape(self._nx, -1)
        snapshots = self._dm.clone()
        if self._subtract_mean:
            snapshots -= snapshots.mean(dim=1).unsqueeze(-1)

        if self._complex:
            weight = self._weight.type(basis.dtype)
            basis_weighted = basis * weight
            snapshots_weighted = snapshots.type(basis.dtype) * weight
            gram = basis_weighted.conj().T @ basis_weighted
            rhs = basis_weighted.conj().T @ snapshots_weighted
            coefficients = pt.linalg.pinv(gram) @ rhs
        else:
            basis_real = pt.cat((basis.real, -basis.imag), dim=1)
            weight = self._weight.type(basis_real.dtype)
            basis_weighted = basis_real * weight
            snapshots_weighted = snapshots.type(basis_real.dtype) * weight
            gram = basis_weighted.T @ basis_weighted
            rhs = basis_weighted.T @ snapshots_weighted
            coefficients_real = pt.linalg.pinv(gram) @ rhs
            n_basis = basis.shape[1]
            coefficients = coefficients_real[:n_basis].type(
                basis.dtype
            ) + 1j * coefficients_real[n_basis:].type(basis.dtype)
        return coefficients.reshape(self._frequency.shape[0], n_modes, self._nt)

    def partial_reconstruction(
        self,
        f_min: Union[int, float],
        f_max: Union[int, float],
        n_modes: int = 1,
        start_idx: int = 0,
        n_snapshots: int = 100,
        add_mean: bool = False,
    ) -> pt.Tensor:
        """Reconstruct snapshots from a subset of frequency bins and modes.

        If both frequency limits are integers, they are interpreted as
        frequency-bin indices. Otherwise, they are interpreted as physical
        frequency values and all bins within the inclusive range are selected.
        The reconstruction is formed from the SPOD modes and their temporal
        coefficients. If fewer modes are available than requested, the number
        is reduced according to :meth:`temporal_coefficients`.

        :param f_min: first frequency-bin index or minimum frequency to include
        :type f_min: Union[int, float]
        :param f_max: last frequency-bin index or maximum frequency to include
        :type f_max: Union[int, float]
        :param n_modes: maximum number of leading modes to include per frequency
            bin; defaults to 1
        :type n_modes: int, optional
        :param start_idx: index of the first snapshot to reconstruct; defaults to 0
        :type start_idx: int, optional
        :param n_snapshots: maximum number of snapshots to reconstruct; defaults
            to 100
        :type n_snapshots: int, optional
        :param add_mean: add the temporal mean if it was subtracted during the
            SPOD computation; defaults to False
        :type add_mean: bool, optional
        :raises ValueError: for invalid frequency-bin or snapshot indices
        :raises ValueError: if ``n_modes`` or ``n_snapshots`` is smaller than one
        :return: partial reconstruction arranged as
            ``(n_spatial_points, n_reconstructed_snapshots)``
        :rtype: pt.Tensor
        """
        n_freq = self._frequency.shape[0]
        use_indices = isinstance(f_min, int) and isinstance(f_max, int)
        if use_indices:
            if f_min < 0 or f_min >= n_freq:
                raise ValueError(f"f_min must be in the range [0, {n_freq - 1:d}]")
            if f_max < f_min or f_max >= n_freq:
                raise ValueError(
                    f"f_max must be in the range [{f_min:d}, {n_freq - 1:d}]"
                )
            frequency_indices = pt.arange(f_min, f_max + 1, dtype=pt.int64)
        else:
            if f_max < f_min:
                raise ValueError("f_max must be greater than or equal to f_min")
            in_range = pt.logical_and(
                self._frequency >= f_min, self._frequency <= f_max
            )
            frequency_indices = pt.arange(n_freq, dtype=pt.int64)[in_range]
            if frequency_indices.numel() == 0:
                raise ValueError(
                    f"no frequency bins are included in [{f_min:g}, {f_max:g}]"
                )
        if start_idx < 0 or start_idx >= self._nt:
            raise ValueError(f"start_idx must be in the range [0, {self._nt - 1:d}]")
        if n_snapshots < 1:
            raise ValueError("n_snapshots must be at least 1")

        stop_idx = min(start_idx + n_snapshots, self._nt)
        available = stop_idx - start_idx
        if available < n_snapshots:
            warnings.warn(
                f"only {available:d} snapshots are available from "
                f"start_idx={start_idx:d}; reconstructing all available snapshots",
                UserWarning,
            )

        coefficients = self.temporal_coefficients(n_modes)
        n_modes = coefficients.shape[1]
        modes = self._modes[frequency_indices, :, :n_modes]
        coefficients = coefficients[frequency_indices, :, start_idx:stop_idx]
        reconstruction = pt.einsum("fxm,fmt->xt", modes, coefficients)
        if not self._complex:
            reconstruction = reconstruction.real
        if add_mean and self._subtract_mean:
            reconstruction += self._dm.mean(dim=1).unsqueeze(-1)
        return reconstruction

    def mode_reconstruction(
        self,
        f_idx: int,
        eig_idx: int = 0,
        dt: Union[float, None] = None,
        N: Union[int, None] = None,
    ) -> pt.Tensor:
        """Compute a time-domain reconstruction based on a single mode.

        The mode is selected based on the frequency bin and the eigenbasis index.
        Note that the size of the eigenbasis may vary across frequency bins
        if `adaptive=True`. By default, one period is reconstructed at 25
        equally spaced time values, including both period boundaries. For the
        zero-frequency mode, the original time step is used instead.

        :param f_idx: frequency bin index
        :type f_idx: int
        :param eig_idx: index of eigenvector-eigenvalue pair; defaults to 0, i.e.,
            the most dominant mode at the selected frequency
        :type eig_idx: int, optional
        :param dt: optional reconstruction time step; defaults to None (distribute
            the time values over one period)
        :type dt: Union[float, None], optional
        :param N: optional number of time values; defaults to None (use 25)
        :type N: Union[int, None], optional
        :raises ValueError: for invalid frequency bins
        :raises ValueError: for invalid eigenbasis index
        :raises ValueError: if the corresponding mode was not saved due to `keep_n_modes`
        :return: reconstruction based on a single mode
        :rtype: pt.Tensor
        """
        n_freq = self._frequency.shape[0]
        if f_idx >= n_freq or f_idx < 0:
            raise ValueError(
                f"invalid frequency index {int(f_idx):d}; the index must be in the range [0, {n_freq - 1:d}]"
            )
        K = self._log["n_tapers"][f_idx] if self._adaptive else self._max_tapers
        if eig_idx >= K or eig_idx < 0:
            raise ValueError(
                f"invalid taper index {int(eig_idx):d}; the highest possible taper index for frequency bin {f_idx:d} is {K:d}"
            )
        if eig_idx >= self._n_keep:
            raise ValueError(
                f"mode {eig_idx} was not saved (keep_n_modes={self._n_keep})"
            )
        m = self._modes[f_idx, :, eig_idx]
        steps = 25 if N is None else N
        frequency = self._frequency[f_idx]
        if dt is None and frequency != 0.0 and steps > 1:
            t_res = 1.0 / (frequency.abs() * (steps - 1))
        else:
            t_res = self._dt if dt is None else dt
        t = pt.arange(steps) * t_res
        osc = pt.exp(2j * pi * frequency * t)
        if self._complex:
            return pt.outer(m, osc)
        else:
            return pt.outer(m, osc).real


class PAMSPOD(AMSPOD):
    """Perform AMSPOD on POD time coefficients."""

    def __init__(
        self,
        data_matrix: pt.Tensor,
        dt: float,
        nfft: Union[int, None] = None,
        adaptive: bool = True,
        max_tapers: int = 50,
        tolerance: float = 1.0e-5,
        weight: Union[pt.Tensor, None] = None,
        subtract_mean: bool = True,
        keep_n_modes: int = 3,
        device: str = "cpu",
        verbose: bool = False,
        rank: Union[int, None] = None,
    ):
        """Perform POD-projection and compute AMSPOD.

        :param data_matrix: time series of snapshots arranged as 2D tensor with the
            1st dimension (rows, size M) denoting space and the 2nd dimension denoting
            time (columns, size N)
        :type data_matrix: pt.Tensor
        :param dt: time increment between snapshots; assumed to be constant
        :type dt: float
        :param nfft: number of frequency bins for computing the DFT; zero-padding is applied
            if nfft > N; if nfft is None or nfft <= N, then nfft=N is set; defaults to None
        :type nfft: Union[int, None], optional
        :param adaptive: choose the number of tapers adaptively for each frequency bin;
            adaptivity reduces the bias for quickly converging frequencies (see `tolerance`);
            defaults to True
        :type adaptive: bool, optional
        :param max_tapers: max. number of tapers to use when `adaptive=True` or constant number
            of tapers for all frequencies if `adaptive=False`; the minimum number of tapers is 2; defaults to 50
        :type max_tapers: int, optional
        :param tolerance: tolerance at which the leading mode at each frequency is considered
            to be converged; the difference is measured by comparing the leading mode
            with i and i+1 tapers for increasing values of i, defaults to 1.0e-5
        :type tolerance: float, optional
        :param weight: weight vector for reducing the mesh-induced bias when computing inner products;
            should be of length M; defaults to None
        :type weight: Union[pt.Tensor, None], optional
        :param subtract_mean: subtract mean before computing the DFT, defaults to True
        :type subtract_mean: bool, optional
        :param keep_n_modes: number of spatial modes to keep, defaults to 3
        :type keep_n_modes: int, optional
        :param device: device used for computing; defaults to "cpu"
        :type device: str, optional
        :param verbose: extended output of log messages; defaults to False
        :type verbose: str, optional
        :param rank: size of POD basis; if None, the full basis is used; defaults to None
        :type rank: int, optional
        """
        self._dm_org = data_matrix
        if weight is not None:
            logger.info("weighting snapshots with provided weight vector")
            self._weight_org = _prepare_weights(weight, self._dm_org.shape[0])
            self._dm_org *= self._weight_org
        if subtract_mean:
            logger.info("subtracting temporal mean from original data matrix")
            self._mean_org = self._dm_org.mean(dim=-1)
            self._dm_org -= self._mean_org.unsqueeze(-1)
        if rank is None:
            rank = min(self._dm_org.shape)
        logger.info("computing SVD of original data matrix")
        self._svd = SVD(self._dm_org, rank)
        super(PAMSPOD, self).__init__(
            (self._svd.V * self._svd.s).T,
            dt=dt,
            nfft=nfft,
            adaptive=adaptive,
            max_tapers=max_tapers,
            tolerance=tolerance,
            weight=None,
            subtract_mean=False,
            keep_n_modes=keep_n_modes,
            device=device,
            verbose=verbose,
        )

    @property
    def modes(self) -> pt.Tensor:
        """Project modes back to original space and undo weighting.

        :return: SPOD modes in full state space
        :rtype: pt.Tensor
        """
        m = super().modes
        if hasattr(self, "_weight_org"):
            return pt.einsum(
                "m,mr,nrk->nmk",
                1.0 / self._weight_org.squeeze().type(m.dtype),
                self._svd.U.type(m.dtype),
                m,
            )
        else:
            return pt.einsum("mr,nrk->nmk", self._svd.U.type(m.dtype), m)

    @property
    def svd(self) -> SVD:
        """SVD object used for subspace projection.

        :return: economy SVD
        :rtype: SVD
        """
        return self._svd

    def get_mode(self, f_idx: int, mode_idx: int = 0) -> pt.Tensor:
        """Get mode/eigenvector of prescribed frequency bin and mode index.

        :param f_idx: index of the frequency bin
        :type f_idx: int
        :param mode_idx: index of the sorted modes per frequency bin, defaults to 0
        :type mode_idx: int, optional
        :return: mode in full state space
        :rtype: pt.Tensor
        """
        mode = super().modes[f_idx, :, mode_idx]
        if hasattr(self, "_weight_org"):
            return ((self.svd.U / self._weight_org).type(mode.dtype) * mode).sum(dim=1)
        else:
            return (self.svd.U.type(mode.dtype) * mode).sum(dim=1)

    def mode_reconstruction(
        self,
        f_idx: int,
        eig_idx: int = 0,
        dt: Union[float, None] = None,
        N: Union[int, None] = None,
    ) -> pt.Tensor:
        """Compute time-domain reconstruction based on a single mode.

        This method wraps around the corresponding method of the base class
        and projects the reconstruction back to the full state space.

        :param f_idx: frequency bin index
        :type f_idx: int
        :param eig_idx: index of eigenvector-eigenvalue pair; defaults to 0, i.e.,
            the most dominant mode at the selected frequency
        :type eig_idx: int, optional
        :param dt: optional reconstruction time step; defaults to None (distribute
            the time values over one period)
        :type dt: Union[float, None], optional
        :param N: optional number of time values; defaults to None (use 25)
        :type N: Union[int, None], optional
        :return: reconstruction based on a single mode
        :rtype: pt.Tensor
        """
        rec = super().mode_reconstruction(f_idx, eig_idx, dt, N)
        if hasattr(self, "_weight_org"):
            return (self.svd.U / self._weight_org).type(rec.dtype) @ rec
        else:
            return self.svd.U.type(rec.dtype) @ rec

    def partial_reconstruction(
        self,
        f_min: Union[int, float],
        f_max: Union[int, float],
        n_modes: int = 1,
        start_idx: int = 0,
        n_snapshots: int = 100,
        add_mean: bool = False,
    ) -> pt.Tensor:
        """Reconstruct snapshots in full space from selected bins and modes.

        See :meth:`AMSPOD.partial_reconstruction` for the argument definitions.

        :return: partial reconstruction in the original state space
        :rtype: pt.Tensor
        """
        rec = super().partial_reconstruction(
            f_min, f_max, n_modes, start_idx, n_snapshots, add_mean=False
        )
        if hasattr(self, "_weight_org"):
            rec = (self.svd.U / self._weight_org).type(rec.dtype) @ rec
        else:
            rec = self.svd.U.type(rec.dtype) @ rec
        if add_mean and hasattr(self, "_mean_org"):
            mean = self._mean_org
            if hasattr(self, "_weight_org"):
                mean = mean / self._weight_org.squeeze()
            rec += mean.unsqueeze(-1)
        return rec
