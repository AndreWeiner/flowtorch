"""Classes and functions to compute the dynamic mode decomposition (DMD) of a data matrix.
"""

# standard library packages
from typing import Tuple, Set, Union, List
from math import sqrt
from collections import defaultdict

# third party packages
import torch as pt
from numpy import pi
from scipy.linalg import solve_sylvester

# flowtorch packages
from .svd import SVD
from flowtorch.data.utils import format_byte_size


def _dft_properties(dt: float, n_times: int) -> Tuple[float, float, float]:
    """Compute general properties of a discrete Fourier transformation.

    DFT properties, such as maximum frequency and frequency resolution, can
    be helpful guidance for building sensible data matrices used for
    modal decomposition.

    :param dt: timestep between two samples; assumed constant
    :type dt: float
    :param n_times: number of time steps
    :type n_times: int
    :return: sampling frequency, maximum frequency, frequency resolution
    :rtype: Tuple[float, float, float]
    """
    fs = 1.0 / dt
    return fs, 0.5 * fs, fs / n_times


def _fb_consistent_operator(
    X: pt.Tensor,
    Y: pt.Tensor,
    steps: int = 1000,
    tol_abs: float = 1.0e-8,
    tol_rel: float = 1.0e-6,
    rho: float = 10.0,
    tau: float = 2.0,
    mu: float = 5.0,
):
    """Forward-backward consistent approximation of the DMD operator.

    This implementation closely follows the original Matlab code;
    see https://github.com/azencot/nld.cdmd; the algorithm corresponds
    to algorithm 3.1 of the reference article (see DMD class docs).

    :param X: rank-r projection of the first snapshot matrix
    :type X: pt.Tensor
    :param Y: rank_r projection of the shifted snapshot matrix
    :type Y: pt.Tensor
    :param steps: maximum number of iterations, defaults to 1000
    :type steps: int, optional
    :param tol_abs: absolute stopping tolerance, defaults to 1.0e-8
    :type tol_abs: float, optional
    :param tol_rel: relative stopping tolerance, defaults to 1.0e-6
    :type tol_rel: float, optional
    :param rho: penalty parameter, defaults to 10.0
    :type rho: float, optional
    :param tau: factor to adjust rho, defaults to 2.0
    :type tau: float, optional
    :param mu: maximum allowed ratio between primary and dual residual
        before adjustment of penalty parameter, defaults to 5.0
    :type mu: float, optional
    :return: log file and final operator approximation
    :rtype: Tuple[dict, pt.Tensor]
    """
    # initialize helper tensors and log
    log = defaultdict(list)
    r = X.shape[0]
    A = Y @ pt.linalg.pinv(X)
    B = X @ pt.linalg.pinv(Y)
    Q1, Q2 = pt.zeros_like(A), pt.zeros_like(A)

    # start iterative update
    for i in range(steps):
        A_old = A.clone()
        A = pt.from_numpy(
            solve_sylvester(
                (rho * B.T @ B).numpy(),
                (X @ X.T + rho * B @ B.T).numpy(),
                (Y @ X.T + 2.0 * rho * B.T - rho * Q1 @ B.T - rho * B.T @ Q2).numpy(),
            )
        )
        B_old = B.clone()
        B = pt.from_numpy(
            solve_sylvester(
                (rho * A.T @ A).numpy(),
                (Y @ Y.T + rho * A @ A.T).numpy(),
                (X @ Y.T + 2.0 * rho * A.T - rho * A.T @ Q1 - rho * Q2 @ A.T).numpy(),
            )
        )
        Q1 += A @ B - pt.eye(r)
        Q2 += B @ A - pt.eye(r)

        # log primary and dual residual, forward and backward projection errors
        log["p_res"].append(
            0.5 * ((A @ B - pt.eye(r)).norm() + (B @ A - pt.eye(r)).norm()).item()
        )
        log["d_res"].append(0.5 * rho * pt.cat([A - A_old, B - B_old], dim=0).norm().item())
        log["p_eps"].append(
            sqrt(r) * tol_abs
            + tol_rel * max((A @ B).norm().item(), max((B @ A).norm().item(), sqrt(r)))
        )
        log["d_eps"].append(
            sqrt(2 * r) * tol_abs
            + tol_rel * (rho * pt.cat([Q1, Q2], dim=0)).norm().item()
        )
        log["f_err"].append(((A @ X - Y).norm() / Y.norm()).item())
        log["b_err"].append(((B @ Y - X).norm() / X.norm()).item())

        # update penalty parameter
        log["rho"].append(rho)
        if log["p_res"][-1] > mu * log["d_res"][-1]:
            rho *= tau
            Q1 /= tau
            Q2 /= tau
        elif log["d_res"][-1] > mu * log["p_res"][-1]:
            rho /= tau
            Q1 *= tau
            Q2 *= tau

        # check convergence
        if log["p_res"][-1] < log["p_eps"][-1] and log["d_res"][-1] < log["d_eps"][-1]:
            print(f"ADMM converged after {i+1} steps")
            print("Initial/final mean projection error:")
            e_init = 0.5 * (log["f_err"][0] + log["b_err"][0])
            e_final = 0.5 * (log["f_err"][-1] + log["b_err"][-1])
            print(f"{e_init:1.8e}/{e_final:1.8e}")
            break

    return log, A


class DMD(object):
    """Wrapper class implementing multiple DMD versions.

    Examples

    >>> from flowtorch import DATASETS
    >>> from flowtorch.data import FOAMDataloader
    >>> from flowtorch.analysis import DMD
    >>> path = DATASETS["of_cavity_binary"]
    >>> loader = FOAMDataloader(path)
    >>> data_matrix = loader.load_snapshot("p", loader.write_times)
    >>> dmd = DMD(data_matrix, dt=0.1, rank=3)
    >>> dmd.frequency
    tensor([0., 5., 0.])
    >>> dmd.growth_rate
    tensor([-2.3842e-06, -4.2345e+01, -1.8552e+01])
    >>> dmd.amplitude
    tensor([10.5635+0.j, -0.0616+0.j, -0.0537+0.j])
    >>> dmd = DMD(data_matrix, dt=0.1, rank=3, robust=True)
    >>> dmd = DMD(data_matrix, dt=0.1, rank=3, robust={"tol": 1.0e-5, "verbose" : True})

    """

    def __init__(
        self,
        data_matrix: Union[pt.Tensor, List[pt.Tensor]],
        dt: float,
        rank: Union[int, None] = None,
        robust: Union[bool, dict] = False,
        unitary: bool = False,
        optimal: bool = False,
        tlsq: bool = False,
        forward_backward: Union[bool, dict] = False,
    ) -> None:
        """Create DMD instance based on one or more data matrices.

        :param data_matrix: data matrix or list of data matrices whose columns are
            formed by a sequence of snapshots
        :type data_matrix: pt.Tensor
        :param dt: time step between two snapshots
        :type dt: float
        :param rank: rank for SVD truncation, defaults to None
        :type rank: int, optional
        :param robust: data_matrix is split into low rank and sparse contributions
            if True or if dictionary with options for Inexact ALM algorithm; the SVD
            is computed only on the low rank matrix
        :type robust: Union[bool,dict]
        :param unitary: enforce the linear operator to be unitary; refer to piDMD_
            by Peter Baddoo for more information
        :type unitary: bool, optional
        :param optimal: compute mode amplitudes based on a least-squares problem
            as described in spDMD_ article by M. Janovic et al. (2014); in contrast
            to the original spDMD implementation, the exact DMD modes are used in
            the optimization problem as outlined in an article_ by R. Taylor
        :type optimal: bool, optional
        :param tlsq: de-biasing of the linear operator by solving a total least-squares
            problem instead of a standard least-squares problem; the rank is selected
            automatically or specified by the `rank` parameter; more information can be
            found in the TDMD_ article by M. Hemati et al.
        :type tlsq: bool, optional
        :param forward_backward: enforce consistency between forward and backward
            propagation according to the CDMD_ by O. Azencot et al.
            can be a bool or a dictionary with optimization settings for ADMM; defaults
            to False (no consistency is enforced)
        :type forward_backward: Union[bool,dict], optional

        .. _piDMD: https://github.com/baddoo/piDMD
        .. _spDMD: https://hal-polytechnique.archives-ouvertes.fr/hal-00995141/document
        .. _article: http://www.pyrunner.com/weblog/2016/08/03/spdmd-python/
        .. _TDMD: http://cwrowley.princeton.edu/papers/Hemati-2017a.pdf
        .. _CDMD: https://doi.org/10.1137/18M1233960
        """
        self._dm = [data_matrix] if isinstance(data_matrix, pt.Tensor) else data_matrix
        self._complex = pt.is_complex(self._dm[0])
        self._dt = dt
        self._unitary = unitary
        self._optimal = optimal
        self._tlsq = tlsq
        self._fb = forward_backward
        self._fb_log = None
        X = pt.cat([dm[:, :-1] for dm in self._dm], dim=-1)
        Y = pt.cat([dm[:, 1:] for dm in self._dm], dim=-1)
        if self._tlsq:
            svd = SVD(pt.vstack((X, Y)), rank, robust)
            P = svd.V @ svd.V.conj().T
            self._X = X @ P
            self._Y = Y @ P
            self._svd = SVD(self._X, svd.rank)
            del svd, X, Y
        else:
            self._svd = SVD(X, rank, robust)
            self._X = X
            self._Y = Y
        self._eigvals, self._eigvecs, self._modes = self._compute_mode_decomposition()
        self._amplitude = self._compute_amplitudes()

    def _compute_operator(self) -> pt.Tensor:
        """Compute the DMD operator."""
        if self._unitary:
            Xp = self._svd.U.conj().T @ self._X
            Yp = self._svd.U.conj().T @ self._Y
            U, _, VT = pt.linalg.svd(Yp @ Xp.conj().T, full_matrices=False)
            return U @ VT
        elif bool(self._fb):
            if isinstance(self._fb, dict):
                log, A = _fb_consistent_operator(
                    self._svd.U.conj().T @ self._X,
                    self._svd.U.conj().T @ self._Y,
                    **self._fb,
                )
            else:
                log, A = _fb_consistent_operator(
                    self._svd.U.conj().T @ self._X, self._svd.U.conj().T @ self._Y
                )
            self._fb_log = log
            return A
        else:
            s_inv = pt.diag(1.0 / self._svd.s.type(self._dm[0].dtype))
            return self._svd.U.conj().T @ self._Y @ self._svd.V @ s_inv

    def _compute_mode_decomposition(self) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        """Compute reduced operator, eigen-decomposition, and DMD modes."""
        s_inv = pt.diag(1.0 / self._svd.s)
        self._At = self._compute_operator()
        val, vec = pt.linalg.eig(self._At)
        phi = (
            self._Y.type(val.dtype)
            @ self._svd.V.type(val.dtype)
            @ s_inv.type(val.dtype)
            @ vec
        )
        return val, vec, phi

    def _compute_amplitudes(self) -> Union[List[pt.Tensor], pt.Tensor]:
        """Compute amplitudes for exact DMD modes.

        If *optimal* is False, the amplitudes are computed based on the first
        snapshot in the data matrix; otherwise, a least-square problem as
        introduced by Janovic et al. is solved (refer to the documentation
        in the constructor for more information).
        """
        amp = []
        if self._optimal:
            for dm in self._dm:
                vander = pt.linalg.vander(self.eigvals, N=dm.shape[1] - 1)
                P = (self._modes.conj().T @ self._modes) * (
                    vander @ vander.conj().T
                ).conj()
                q = pt.diag(
                    vander @ dm[:, :-1].type(P.dtype).conj().T @ self._modes
                ).conj()
                amp.append(pt.linalg.lstsq(P, q).solution)
        else:
            for dm in self._dm:
                amp.append(
                    pt.linalg.lstsq(
                        self._modes, dm[:, 0].type(self._modes.dtype)
                    ).solution
                )
        return amp[0] if len(amp) == 1 else amp

    def partial_reconstruction(
        self, mode_indices: Set[int]
    ) -> Union[List[pt.Tensor], pt.Tensor]:
        """Reconstruct data matrix with limited number of modes.

        :param mode_indices: mode indices to keep
        :type mode_indices: Set[int]
        :return: reconstructed data matrix
        :rtype: pt.Tensor
        """
        _, cols = self._modes.shape
        mode_mask = pt.zeros(cols, dtype=pt.complex64)
        mode_indices = pt.tensor(list(mode_indices), dtype=pt.int64)
        mode_mask[mode_indices] = 1.0
        dyn = self.dynamics
        if isinstance(dyn, list):
            rec = []
            for d in dyn:
                rec.append((self.modes * mode_mask) @ d)
            return rec if self._complex else [r.real for r in rec]
        else:
            rec = (self.modes * mode_mask) @ dyn
            return rec if self._complex else rec.real

    def top_modes(
        self,
        n: int = 10,
        integral: bool = False,
        f_min: float = -float("inf"),
        f_max: float = float("inf"),
    ) -> pt.Tensor:
        """Get the indices of the first n most important modes.

        Note that the conjugate complex modes for real data matrices are
        not filtered out by default. However, by setting the lower frequency
        threshold to a positive number, only modes with positive imaginary
        part are considered.

        :param n: number of indices to return; defaults to 10
        :type n: int
        :param integral: if True, the modes are sorted according to their
            integral contribution; defaults to False
        :type integral: bool, optional
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
        importance = self.integral_contribution if integral else self.amplitude
        if isinstance(importance, list):
            importance = pt.vstack(importance).mean(dim=0)
        modes_in_range = pt.logical_and(self.frequency >= f_min, self.frequency < f_max)
        mode_indices = pt.tensor(range(modes_in_range.shape[0]), dtype=pt.int64)[
            modes_in_range
        ]
        n = min(n, mode_indices.shape[0])
        top_n = importance[mode_indices].abs().topk(n).indices
        return mode_indices[top_n]

    def predict(self, initial_condition: pt.Tensor, n_steps: int) -> pt.Tensor:
        """Predict evolution over N steps starting from used-defined initial conditions.

        :param initial_condition: initial state vector
        :type initial_condition: pt.Tensor
        :param n_steps: number of steps to predict
        :type n_steps: int
        :return: predicted evolution, including the initial state (N+1 states are returned)
        :rtype: pt.Tensor
        """
        b = pt.linalg.pinv(self._modes) @ initial_condition.type(self._modes.dtype)
        prediction = (
            self._modes @ pt.diag(b) @ pt.linalg.vander(self.eigvals, N=n_steps + 1)
        )
        if not self._complex:
            prediction = prediction.real
        return prediction

    @property
    def required_memory(self) -> int:
        """Compute the memory size in bytes of the DMD.

        :return: cumulative size of SVD, eigen values/vectors, and
            DMD modes in bytes
        :rtype: int
        """
        return (
            self._svd.required_memory
            + self._eigvals.element_size() * self._eigvals.nelement()
            + self._eigvecs.element_size() * self._eigvecs.nelement()
            + self._modes.element_size() * self._modes.nelement()
        )

    @property
    def fb_log(self) -> Union[dict, None]:
        return self._fb_log

    @property
    def svd(self) -> SVD:
        return self._svd

    @property
    def operator(self) -> pt.Tensor:
        return self._At

    @property
    def modes(self) -> pt.Tensor:
        return self._modes

    @property
    def eigvals(self) -> pt.Tensor:
        return self._eigvals

    @property
    def eigvals_cont(self) -> pt.Tensor:
        return pt.log(self._eigvals) / self._dt

    @property
    def eigvecs(self) -> pt.Tensor:
        return self._eigvecs

    @property
    def frequency(self) -> pt.Tensor:
        return pt.log(self._eigvals).imag / (2.0 * pi * self._dt)

    @property
    def growth_rate(self) -> pt.Tensor:
        return (pt.log(self._eigvals) / self._dt).real

    @property
    def amplitude(self) -> Union[List[pt.Tensor], pt.Tensor]:
        return self._amplitude

    @property
    def dynamics(self) -> Union[List[pt.Tensor], pt.Tensor]:
        if isinstance(self.amplitude, list):
            return [
                pt.diag(a) @ pt.linalg.vander(self.eigvals, N=dm.shape[1])
                for a, dm in zip(self.amplitude, self._dm)
            ]
        else:
            return pt.diag(self.amplitude) @ pt.vander(
                self.eigvals, N=self._dm[0].shape[1]
            )

    @property
    def integral_contribution(self) -> Union[List[pt.Tensor], pt.Tensor]:
        """Integral contribution of individual modes according to J. Kou et al. 2017.

        DOI: https://doi.org/10.1016/j.euromechflu.2016.11.015
        """
        dyn = self.dynamics
        m_norm_sqr = self.modes.norm(dim=0) ** 2
        if isinstance(dyn, list):
            return [m_norm_sqr * d.abs().sum(dim=1) for d in dyn]
        else:
            return m_norm_sqr * dyn.abs().sum(dim=1)

    @property
    def reconstruction(self) -> Union[List[pt.Tensor], pt.Tensor]:
        """Reconstruct an approximation of the original data matrix.

        :return: reconstructed data matrix
        :rtype: pt.Tensor
        """
        dyn = self.dynamics
        if isinstance(dyn, list):
            rec = [self.modes @ d for d in dyn]
            return rec if self._complex else [r.real for r in rec]
        else:
            rec = self.modes @ dyn
            return rec if self._complex else rec.real

    @property
    def reconstruction_error(self) -> Union[List[pt.Tensor], pt.Tensor]:
        """Compute the reconstruction error.

        :return: difference between reconstruction and data matrix
        :rtype: pt.Tensor
        """
        rec = self.reconstruction
        if isinstance(rec, list):
            return [r - dm for r, dm in zip(rec, self._dm)]
        else:
            return rec - self._dm[0]

    @property
    def projection_error(self) -> pt.Tensor:
        """Compute the difference between Y and AX.

        :return: projection error
        :rtype: pt.Tensor
        """
        YH = (self._modes @ pt.diag(self.eigvals)) @ (
            pt.linalg.pinv(self._modes) @ self._X.type(self._modes.dtype)
        )
        Y = pt.cat([dm[:, 1:] for dm in self._dm], dim=-1)
        return YH - Y if self._complex else YH.real - Y

    @property
    def tlsq_error(self) -> Tuple[pt.Tensor, pt.Tensor]:
        """Compute the *noise* in X and Y.

        :return: noise in X and Y
        :rtype: Tuple[pt.Tensor, pt.Tensor]
        """
        if not self._tlsq:
            print("Warning: noise is only removed if tlsq=True")
        X = pt.cat([dm[:, :-1] for dm in self._dm], dim=-1)
        Y = pt.cat([dm[:, 1:] for dm in self._dm], dim=-1)
        return X - self._X, Y - self._Y

    @property
    def dft_properties(self) -> Tuple[float, float, float]:
        return _dft_properties(self._dt, self._X.shape[1])

    def __repr__(self):
        return f"{self.__class__.__qualname__}(data_matrix, rank={self._svd.rank})"

    def __str__(self):
        ms = ["SVD:", str(self.svd), "LSQ:", str(self._tlsq)]
        size, unit = format_byte_size(self.required_memory)
        ms.append("Overall DMD size: {:1.4f}{:s}".format(size, unit))
        ms.append("DFT frequencies (sampling, max., res.):")
        ms.append("{:1.4f}Hz, {:1.4f}Hz, {:1.4f}Hz".format(*self.dft_properties))
        ms.append("")
        return "\n".join(ms)
