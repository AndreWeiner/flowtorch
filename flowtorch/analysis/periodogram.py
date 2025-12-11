"""Periodogram estimation techniques for scalar signals."""

# standard library packages
import logging
from typing import Union, Tuple
from collections import defaultdict
from math import sqrt, log10

# third-party packages
import torch as pt


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AMPS(object):
    """Adaptive multi-taper power spectrum estimation.

    This class is a version of `AMSPOD` (adaptive multi-taper spectral POD)
    adjusted for scaler signals. The result is an estimate of the signal's
    power spectrum, similar to Welch's method. However, multiple realizations
    are estimated using sin-tapers rather than signal blocks (Welch). The main
    references for this implementation are:

    - Yeung, B. C. Y., Schmidt, O. T.: Adaptive spectral proper orthogonal decomposition
    of broadband-tonal flows, Theor. Comput. Fluid Dyn. 38, 355-374, 2024,
    DOI 10.1007/s00162-024-00695-0
    - Riedel, K. S., Sidorenko, A.: Minimum bias multiple taper spectral estimation,
    IEEE transactions of Signal Processing 43, 188-195, 1995,
    DOI 10.1109/78.365298

    """

    def __init__(
        self,
        signal: pt.Tensor,
        dt: float,
        nfft: Union[int, None] = None,
        adaptive: bool = True,
        max_tapers: int = 50,
        tolerance: float = 1.0e-5,
        subtract_mean: bool = True,
        verbose: bool = False,
    ):
        """Instantiate AMPS computation.

        :param signal: time-varying signal sampled at constant frequency; for N samples,
            accepted tensor shapes are (N,) and (1, N)
        :type signal: pt.Tensor
        :param dt: time-increment between samples
        :type dt: float
        :param nfft: number of frequency bins when computing the DFT; zero-padding is applied
            if nfft > N; if nfft is None or nfft <= N, then nfft=N is set; defaults to None
        :type nfft: Union[int, None], optional
        :param adaptive: choose the number of tapers adaptively for each frequency bin;
            adaptivity reduces the bias for quickly converging frequencies (see `tolerance`);
            defaults to True
        :type adaptive: bool, optional
        :param max_tapers: max. number of tapers to use when `adaptive=True` or constant number
            of tapers for all frequencies if `adaptive=False`; the minimum number of tapers is 2; defaults to 50
        :type max_tapers: int, optional
        :param tolerance: tolerance at which the power at each frequency is considered converged;
            the difference is measured by computing and comparing the spectral power
            with i and i+1 tapers for increasing values of i, defaults to 1.0e-5
        :type tolerance: float, optional
        :param subtract_mean: subtract mean before computing the DFT, defaults to True
        :type subtract_mean: bool, optional
        :param verbose: extended output of log messages; defaults to False
        :type verbose: bool, optional
        :raises ValueError: for signals with invalid shape
        """
        shape = signal.shape
        if len(shape) == 1:
            self._signal = signal.unsqueeze(0)
        elif len(shape) == 2 and shape[0] == 1:
            self._signal = signal
        else:
            raise ValueError(
                f"got invalid input signal of size {shape}; allowed shapes are (N,) and (1, N)"
            )
        self._nt = self._signal.shape[1]
        self._complex = pt.is_complex(self._signal)
        self._dt = dt
        if nfft is None:
            self._nfft = self._nt
        else:
            self._nfft = max(self._nt, nfft)
        self._adaptive = adaptive
        self._max_tapers = max(2, min(max_tapers, self._nt))
        self._tol = tolerance
        self._subtract_mean = subtract_mean
        self._verbose = verbose
        self._log = {}
        self._taper_norm = sqrt(2.0 / (self._nfft + 1.0))
        self._power, self._frequency = self._multi_taper_estimate()

    def _multi_taper_estimate(self) -> Tuple[pt.Tensor, pt.Tensor]:
        """Driver routine to estimate the spectral power.

        :return: spectral power per bin and bin frequencies
        :rtype: Tuple[pt.Tensor, pt.Tensor]
        """
        Q_var = self._signal.clone()
        if self._subtract_mean:
            logger.info("computing and subtracting temporal mean")
            Q_var -= Q_var.mean(dim=1).unsqueeze(-1)
        f = (
            pt.fft.fftfreq(self._nfft, self._dt)
            if self._complex
            else pt.fft.rfftfreq(self._nfft, self._dt)
        )
        Q_hat = pt.fft.fft(Q_var, n=2 * self._nfft, dim=1, norm="backward")
        n_freq = f.shape[0]
        n_win = self._determine_n_tapers(Q_hat, n_freq)
        power = pt.zeros(n_freq, dtype=self._signal.dtype)
        logger.info(f"computing power estimate for {n_freq} frequency bins")
        for i in range(n_freq):
            if self._verbose:
                logger.info(
                    f"using {int(n_win[i])} tapers for bin {i:d} (f={f[i].item():1.4f}Hz)"
                )
            power[i] = self._sin_taper_dft(Q_hat, i, int(n_win[i])) / self._nt
        return power, f

    def _parabolic_weights(self, n_tapers: int) -> pt.Tensor:
        """Generate parabolic window weights for a given number of tapers/windows.

        The method implements formula (11) of the first reference. The same factorized
        formula as in the AMSPOD Matlab implementation is used.

        :param n_tapers: number of sin tapers/windows
        :type n_tapers: int
        :return: parabolic window weights decreasing with window order
        :rtype: pt.Tensor
        """
        k = pt.arange(1, n_tapers + 1).type(self._signal.dtype)
        mu = (
            6.0
            / (n_tapers * (4.0 * n_tapers - 1.0) * (n_tapers + 1.0))
            * (n_tapers**2 - (k - 1.0) ** 2)
        )
        return mu

    def _sin_taper_dft(self, Q_hat: pt.Tensor, f_idx: int, n_tapers: int) -> float:
        """Compute the sin-tapered DFT based on the untapered DFT.

        The bin difference formula is used to compute the DFT for various tapers
        all at once rather than computing the DFT of various tapered data matrices.
        Refer to formula (13) of the first reference article.

        :param Q_hat: DFT of the untapered data of size 1 x 2*nfft
        :type Q_hat: pt.Tensor
        :param f_idx: index of the frequency bin of which to compute tapered versions
        :type f_idx: int
        :param n_tapers: number of sin-tapers
        :type n_tapers: int
        :return: spectral power estimate of ith frequency bin based on the specified
            number of tapers (K)
        :rtype: float
        """
        idx_c = 2 * f_idx
        shifts = pt.arange(1, n_tapers + 1)
        idx_l = (idx_c - shifts) % (2 * self._nfft)
        idx_u = (idx_c + shifts) % (2 * self._nfft)
        a = self._taper_norm * (Q_hat[:, idx_l] - Q_hat[:, idx_u]) / (2j)
        return (a.abs()[0]**2 * self._parabolic_weights(n_tapers)).sum().abs().item()

    def _determine_n_tapers(self, Q_hat: pt.Tensor, n_freq: int) -> pt.Tensor:
        """Determine the number of tapers for each frequency bin.

        If the selection is adaptive, the optimal number of tapers is determined for each frequency
        individually by tracking the change in the estimated power between increasing numbers of tapers.
        A frequency-bin is considered converged if:
            1) the change in the power is below a user-defined tolerance
            2) the difference in the taper number between two bins exceeds 1
        If the selection is non-adaptive, the used-defined maximum number of tapers
        if used for each bin

        :param Q_hat: un-tapered DFT of input signal
        :type Q_hat: pt.Tensor
        :param n_freq: number of frequency bins
        :type n_freq: int
        :return: number of tapers per bin to use for the final spectral estimation
        :rtype: pt.Tensor
        """
        n_tapers = pt.ones(n_freq, dtype=pt.int32) * 2
        if self._adaptive:
            logger.info("performing adaptive taper selection")
            converged = pt.zeros(n_freq, dtype=pt.bool)
            prev_power = pt.ones(n_freq, dtype=self._signal.dtype)
            self._log["convergence"] = defaultdict(list)
            eps = pt.finfo(self._signal.dtype).tiny
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
                    power = self._sin_taper_dft(Q_hat, f_idx, K)
                    log_power = log10(power + eps)
                    log_prev = log10(prev_power[f_idx] + eps)
                    change = abs(log_power - log_prev)
                    if itr > 0:
                        self._log["convergence"][f_idx].append(change)
                    prev_power[f_idx] = power
                    if change <= self._tol or K >= self._max_tapers:
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
        """DFT bin frequencies.

        For real-valued input data, only positive frequencies are returned.

        :return: DFT bin frequencies
        :rtype: pt.Tensor
        """
        return self._frequency

    @property
    def spectral_power(self) -> pt.Tensor:
        """Power estimate for all frequency bins.

        The estimate is obtained by averaging the squared amplitudes.
        A correction for one-sided spectra is included.

        :return: spectral power estimate
        :rtype: pt.Tensor
        """
        p = self._power.clone()
        if not self._complex:
            if self._nfft % 2 == 0:
                p[1:-1] *= 2
            else:
                p[1:] *= 2
        return p

    @property
    def spectral_density(self) -> pt.Tensor:
        """Frequency-normalized spectral power estimate.

        :return: power spectral density
        :rtype: pt.Tensor
        """
        df = self._frequency[1] - self._frequency[0]
        return self.spectral_power / df

    @property
    def half_bandwidth(self) -> Union[float, pt.Tensor]:
        """Resulting sin-taper half-bandwidth.

        The half-bandwidth quantifies the effective frequency resolution.
        A larger bandwidth leads to a greater smoothing of the spectrum.
        The effect of smoothing is lower variance (of the eigenvalues) but
        also poorer frequency resolution.

        :return: constant half-bandwidth if the number of windows is constant;
            otherwise, one half-bandwidth per frequency bin
        :rtype: Union[float, pt.Tensor]
        """
        n_tapers = self.log["n_tapers"] if self._adaptive else self._max_tapers
        return 0.5 * (n_tapers + 1) / (self._nfft + 1) / self._dt

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

        :return: residual computed as |power_k+1 - power_k| / power_k+1; since the number
            of tapers varies per bin, a tensor of size n_freq x (n_tapers - 2) is filled
            with NANs, and then the residuals are overwritten if available; (n_tapers - 2)
            is a consequence of the minimum number of tapers (2) and the residual being
            computed from the power change between two consecutive taper values, so in the
            first iteration, there is no sensible value to compare to
        :rtype: pt.Tensor
        """
        if self._adaptive:
            n_freq = self.frequency.shape[0]
            n_max = self._max_tapers - 2
            sim = self._log["convergence"]
            res = pt.full((n_freq, n_max), float("nan"), dtype=pt.float64)
            for key in sim.keys():
                tmp = pt.tensor(sim[key])
                res[int(key), : tmp.shape[0]] = tmp
            return res
        else:
            logger.warning("residuals are only available for adaptive taper selection")
            return None

    def top_power(
        self,
        n: int = 1,
        f_min: float = -float("inf"),
        f_max: float = float("inf"),
    ) -> pt.Tensor:
        """Get the indices of the first n frequency bins with the highest power.

        :param n: number of indices to return; defaults to 1
        :type n: int
        :param f_min: consider only bins with a frequency larger or equal
            to f_min; defaults to -inf
        :type f_min: float, optional
        :param f_max: consider only bins with a frequency smaller than f_max;
            defaults to -inf
        :type f_max: float, optional
        :return: indices of top n frequency bins sorted by spectral power
        :rtype: pt.Tensor
        """
        bins_in_range = pt.logical_and(self.frequency >= f_min, self.frequency < f_max)
        bin_indices = pt.tensor(range(bins_in_range.shape[0]), dtype=pt.int64)[
            bins_in_range
        ]
        n = min(n, bin_indices.shape[0])
        top_n = self.spectral_power[bin_indices].topk(n).indices
        return bin_indices[top_n]
