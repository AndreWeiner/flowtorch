"""Discrete Fourier transformations of snapshot data."""

# standard library packages
from typing import Set
import logging

# third party packages
import torch as pt

# flowtorch packages
from ..constants import FLOAT_TOLERANCE
from .svd import SVD


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

WINDOWS = {
    "boxcar": pt.ones,
    "hann": pt.hann_window,
    "hamming": pt.hamming_window,
    "blackman": pt.blackman_window,
}


class DFT(object):
    """Compute the discrete Fourier transform (DFT) of a data matrix."""

    def __init__(
        self,
        data_matrix: pt.Tensor,
        dt: float,
        nfft: int | None = None,
        window: str = "boxcar",
        device: str = "cpu",
    ):
        """Compute the discrete Fourier transform (DFT) of a data matrix.

        The DFT is computed along the time axis of the data matrix (for every row).

        :param data_matrix: M x N data matrix, where M is the number of spatial points
            and N is the number of time instances
        :type data_matrix: pt.Tensor
        :param dt: timestep between snapshots; must be constant
        :type dt: float
        :param nfft: number of FFT frequency bin; zero-padding is used if nfft > N;
            truncation is not allowed, defaults to N
        :type nfft: int | None, optional
        :param window: weighting used for tapering the signal, defaults to "boxcar"
        :type window: str, optional
        :param device: device on which to perform the DFT, defaults to "cpu"
        :type device: str, optional
        """
        self._dm = data_matrix
        self._mean = data_matrix.mean(dim=-1)
        self._dt = dt
        self._nfft = nfft if nfft is not None else self._dm.shape[-1]
        self._nfft = max(self._nfft, self._dm.shape[-1])
        self._wname = window
        self._wfunc = WINDOWS.get(window)
        if self._wfunc is None:
            raise ValueError(
                f"Unkown window option {window}. Available windows are:"
                + ", ".join(WINDOWS.keys())
            )
        self._window = self._wfunc(self._dm.shape[-1])
        self._device = device
        self._frequency = (
            pt.fft.fftfreq(self._nfft, dt)
            if pt.is_complex(self._dm)
            else pt.fft.rfftfreq(self._nfft, dt)
        )
        fft = pt.fft.fft if pt.is_complex(self._dm) else pt.fft.rfft
        wdm = (self._dm - self._mean.unsqueeze(-1)) * self._window.type(self._dm.dtype)
        self._modes = fft(wdm.to(device), self._nfft, -1, "ortho").cpu()
        self._amplitude = self._modes.norm(dim=0)

    @property
    def amplitude(self) -> pt.Tensor:
        """Vector norm of the spatial DFT modes.

        :return: mode amplitudes (vector norm of the raw modes)
        :rtype: pt.Tensor
        """
        w = self._window.square().mean()
        a = self._amplitude[1:] ** 2
        if not pt.is_complex(self._dm):
            if self._nfft % 2 == 0:
                a[:-1] *= 2.0
            else:
                a *= 2.0
        return a / self._dm.shape[0] / w

    @property
    def frequency(self) -> pt.Tensor:
        """Bin frequency values.

        For real input data, the symmetric part of the spectrum is excluded.
        The zero-frequency bin is always excluded since the data is mean-subtracted.

        :return: frequency bin values
        :rtype: pt.Tensor
        """
        return self._frequency[1:]

    @property
    def modes(self) -> pt.Tensor:
        """Spatial DFT modes normalized to unit length.

        The zero-frequency mode is exluded since the data
        is mean-subtracted.

        :return: complex, spatial DFT modes
        :rtype: pt.Tensor
        """
        return self._modes[:, 1:] / self._amplitude[1:]

    @property
    def spectral_density(self) -> pt.Tensor:
        df = self.frequency[1] - self.frequency[0]
        return self.amplitude / df

    @property
    def reconstruction(self) -> pt.Tensor:
        """Compute the inverse DFT using all modes.

        :return: reconstructed input data matrix
        :rtype: pt.Tensor
        """
        return self.partial_reconstruction(set(range(len(self.amplitude))), True)

    def partial_reconstruction(
        self, mode_indices: Set[int] | int, include_mean: bool = False
    ) -> pt.Tensor:
        """Reconstruct the original data using selected modes.

        :param mode_indices: mode indices to include in the reconstruction;
            the zero-frequency bin is included by default but does not contribute
            to the reconstruction since the data is mean-subtracted.
        :type mode_indices: Set[int] | int
        :param include_mean: add the mean to the reconstruction; can be useful for
            animations, defaults to False
        :type include_mean: bool, optional
        :return: partial reconstruction of the input data matrix
        :rtype: pt.Tensor
        """
        if self._wname in ("hann", "blackman"):
            logger.warning(
                "The first and last snapshots of the reconstruction are unphysical when using the "
                + f"'{self._wname}' window. Consider using the 'hamming' window."
            )
        if isinstance(mode_indices, int):
            mode_indices = {mode_indices}
        indices = pt.tensor(list(mode_indices), dtype=pt.int64) + 1
        mask = pt.zeros_like(self._amplitude).type(self._modes.dtype)
        mask[0] = 1.0
        mask[indices] = 1.0
        options = ((self._modes * mask).to(self._device), self._nfft, -1, "ortho")
        offset = self._mean.unsqueeze(-1) if include_mean else 0
        w_inv = 1.0 / pt.clamp(self._window, FLOAT_TOLERANCE).type(self._dm.dtype)
        N = self._dm.shape[-1]
        if pt.is_complex(self._dm):
            return pt.fft.ifft(*options).cpu()[:, :N] * w_inv + offset
        else:
            return pt.fft.irfft(*options).cpu().real[:, :N] * w_inv + offset

    def top_modes(
        self,
        n: int = 1,
        density: bool = True,
        f_min: float = -float("inf"),
        f_max: float = float("inf"),
    ) -> pt.Tensor:
        """Get the indices of the first n most important modes.

        :param n: number of indices to return; defaults to 1
        :type n: int
        :param density: sorting based on density rather than amplitudes;
            defaults to True
        :type density: bool, optional
        :param f_min: consider only modes with a frequency larger or equal
            to f_min; defaults to -inf
        :type f_min: float, optional
        :param f_max: consider only modes with a frequency smaller than f_max;
            defaults to -inf
        :type f_max: float, optional
        :return: indices of top n modes sorted by amplitude or integral
            contribution
        :rtype: pt.Tensor
        """
        modes_in_range = pt.logical_and(self.frequency >= f_min, self.frequency < f_max)
        mode_indices = pt.tensor(range(modes_in_range.shape[0]), dtype=pt.int64)[
            modes_in_range
        ]
        n = min(n, mode_indices.shape[0])
        imp = self.spectral_density if density else self.amplitude
        top_n = imp[mode_indices].abs().topk(n).indices
        return mode_indices[top_n]


class PDFT(DFT):
    """Compute the DFT of the POD time coefficients."""

    def __init__(
        self,
        data_matrix: pt.Tensor,
        dt: float,
        rank: int | None = None,
        nfft: int | None = None,
        window: str = "boxcar",
        device: str = "cpu",
    ):
        """Compute the DFT of the POD time coefficients.

        First, the POD basis is computed by means of an SVD. The POD basis may
        be truncated to reduce noise in the spectrum. Only the time coefficients
        are Fourier-transformed. Without rank truncation, this yields the DFT
        of the data matrix (the DFT modes need to be projected back onto the POD basis).
        For large data matrices, the projection step reduces the computational
        cost and storage requirements significantly.

        :param data_matrix: M x N data matrix, where M is the number of spatial points
            and N is the number of time instances
        :type data_matrix: pt.Tensor
        :param dt: timestep between snapshots; must be constant
        :type dt: float
        :param rank: truncation parameter for the POD basis, defaults to None
            (automatic selection in the SVD class)
        :type rank: int | None, optional
        :param nfft: number of FFT frequency bin; zero-padding is used if nfft > N;
            truncation is not allowed, defaults to N
        :type nfft: int | None, optional
        :param window: weighting used for tapering the signal, defaults to "boxcar"
        :type window: str, optional
        :param device: device on which to perform the DFT, defaults to "cpu"
        :type device: str, optional
        """
        self._dm_org = data_matrix
        self._svd = SVD(self._dm_org, rank)
        super(PDFT, self).__init__(
            (self._svd.V * self._svd.s).T,
            dt,
            nfft,
            window,
            device,
        )

    @property
    def svd(self) -> SVD:
        """SVD used to obtain the POD basis.

        :return: SVD of the original data matrix
        :rtype: SVD
        """
        return self._svd

    @property
    def amplitude(self) -> pt.Tensor:
        """Rescaled amplitudes consistent with DFT.

        :return: amplitudes considering the size of the full state
        :rtype: pt.Tensor
        """
        return super().amplitude * self._dm.shape[0] / self._dm_org.shape[0]

    @property
    def modes(self) -> pt.Tensor:
        """DFT modes in the original space.

        :return: DFT modes projected onto the POD basis
        :rtype: pt.Tensor
        """
        m = super().modes
        return self._svd.U.type(m.dtype) @ m

    def partial_reconstruction(
        self, mode_indices: Set[int] | int, include_mean: bool = False
    ) -> pt.Tensor:
        """Partial reconstruction projected onto the POD basis."""
        return self._svd.U @ super().partial_reconstruction(mode_indices, include_mean)
