"""Implementation of a linear dynamics model with control inputs."""

from typing import Tuple, Union, Callable, Type
from collections import defaultdict
from math import sqrt
from numpy import pi
import torch as pt
from torch.utils.data import DataLoader
from .dmd import _dft_properties
from .utils import (
    unsqueeze_if_1d,
    trajectory_train_test_split,
    EarlyStopping,
    DEFAULT_SCHEDULER_OPT,
)


def _least_squares_operator(X: pt.Tensor, Y: pt.Tensor, G: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
    n_states = X.shape[0]
    AB = Y @ pt.linalg.pinv(pt.cat((X, G), dim=0))
    return AB[:, :n_states], AB[:, n_states:]


def _fro_loss_operator(
    label: pt.Tensor, prediction: pt.Tensor, *parameters: tuple
) -> pt.Tensor:
    return (label - prediction).norm() / sqrt(prediction.numel())


class LinearControlModel(pt.nn.Module):
    def __init__(self, data_matrix: pt.Tensor, control_inputs: pt.Tensor, dt: float):
        super(LinearControlModel, self).__init__()
        self._dm = data_matrix
        self._n_states, self._n_times = self._dm.shape
        self._cm = unsqueeze_if_1d(control_inputs)
        self._n_controls = self._cm.shape[0]
        self._dt = dt
        self._check_input_consistency()
        A, B = _least_squares_operator(self._dm[:, :-1], self._dm[:, 1:], self._cm)
        self._eigvals, self._eigvecs = pt.linalg.eig(A)
        self._amplitude = pt.linalg.inv(self._eigvecs) @ self._dm[:, 0].type(self._eigvecs.dtype)
        self._A = pt.nn.Parameter(A)
        self._B = pt.nn.Parameter(B)
        self._noise = pt.nn.Parameter(pt.zeros_like(self._dm))
        self._log = defaultdict(list)

    def _check_input_consistency(self):
        control_steps = self._cm.shape[1]
        if not control_steps + 1 == self._n_times:
            raise ValueError(
                "The number of control inputs (n_controls) should be\n"
                + "one less than the number of state vectors (n_states).\n"
                + f"Got n_states={self._n_times:d} and n_controls={control_steps:d}"
            )

    def forward(
            self, x0: pt.Tensor, noise_idx: pt.Tensor, c: pt.Tensor, backward: bool
    ) -> pt.Tensor:
        """Predict a batch of controlled trajectories.

        abbreviations
        B - batch size
        M - state vector size
        S - number of control inputs
        N - trajectory length without initial condition

        :param x0: batch of initial states of size B X M
            forward: x0 corresponds to state at time step 0
            backward: x0 corresponds to state at time step N
        :type x0: pt.Tensor
        :param noise_idx: noise indices of size B corresponding to x0 
        :type noise_idx: pt.Tensor
        :param c: batch of control inputs of size B x S X N-1
            forward: control time indices [0, 1, ..., N-1] with x0 at 0
            backward: control time indices [N-1, N-2, ..., 0] with x0 at N
        :type c: pt.Tensor
        :param backward: True for rollouts backward in time
        :type backward: bool
        :return: batch of trajectory predictions of size B x M x N
        :rtype: pt.Tensor
        """
        BS, M = x0.shape
        N = c.shape[-1] + 1
        X = pt.zeros((BS, M, N), dtype=x0.dtype, device=x0.device)
        BT = self._B.T
        if not backward:
            AT = self._A.T
            # X[:, :, 0] = (A @ (x0 - self._noise[:, noise_idx].T).T).T + (B @ c[:, :, 0].T).T
            X[:, :, 0] = (x0 - self._noise[:, noise_idx].T) @ AT + c[:, :, 0] @ BT
            for n in range(1, N):
                # X[:, :, n] = (A @ X[:, :, n-1].T).T + (B @ c[:, :, n-1].T).T
                X[:, :, n] = X[:, :, n-1] @ AT + c[:, :, n-1] @ BT
        else:
            AinvT = pt.linalg.inv(self._A).T
            # X[:, :, 0] = (Ainv @ ((x0 - self._noise[:, noise_idx].T) - (B @ c[:, :, 0].T).T).T).T
            X[:, :, 0] = (x0 - self._noise[:, noise_idx].T - c[:, :, 0].T @ BT) @ AinvT
            for n in range(1, N):
                # X[:, :, n] = (Ainv @ (X[:, :, n-1] - (B @ c[:, :, n].T).T).T).T
                X[:, :, n] = (X[:, :, n-1] - c[:, :, n] @ BT) @ AinvT
        return X
    
    def train(
            self,
            epochs: int = 1000,
            batch_size: Union[int, None] = None,
            loss_function: Union[Callable, None] = None,
            split_options: dict = {},
            scheduler_options: dict = {},
            stopping_options: dict = {},
            optimizer: Type[pt.optim.Optimizer] = pt.optim.AdamW,
            optimizer_options: dict = {},
            loss_key: str = "full_loss",
            device: str = "cpu",
        ) -> None:
        optim = optimizer(self.parameters(), **optimizer_options)
        options = {
            key: scheduler_options[key] if key in scheduler_options else val
            for key, val in DEFAULT_SCHEDULER_OPT.items()
        }
        scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, **options)
        train_set, val_set = trajectory_train_test_split(self._dm, **split_options)
        try:
            n_val = len(val_set)
        except:
            n_val = 0
        batch_size, shuffle = (
            (batch_size, True) if batch_size else (len(train_set), False)
        )
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
        stopper = EarlyStopping(model=self, **stopping_options)
        loss_function = _fro_loss_operator if loss_function is None else loss_function
        e, stop = 0, False
        self.to(device)
    
    def predict(
        self, initial_condition: pt.Tensor, control_inputs: pt.Tensor
    ) -> pt.Tensor:
        """Predict trajectory for given initial condition and control inputs.

        M - size of state vector
        S - number of control inputs
        N - number of prediction steps

        :param initial_condition: initial state vector of size M
        :type initial_condition: pt.Tensor
        :param control_inputs: control inputs of size S x N-1
        :type control_inputs: pt.Tensor
        :raises ValueError: for non-matching initial conditions or control inputs
        :return: predicted trajectory of size N (including initial condition)
        :rtype: pt.Tensor
        """
        cm = unsqueeze_if_1d(control_inputs)
        if not cm.shape[0] == self._n_controls:
            raise ValueError(
                f"Expected {self._n_controls:d} control inputs but got {cm.shape[0]:d}"
            )
        if not initial_condition.shape[0] == self._n_states:
            raise ValueError(
                f"Expected state of length {self._n_states:d} but got {initial_condition.shape[0]:d}"
            )
        b = pt.linalg.pinv(self._eigvecs) @ initial_condition.type(self._eigvecs.dtype)
        n_steps = cm.shape[-1] + 1
        V = pt.linalg.vander(self._eigvals, N=n_steps)
        C = pt.linalg.pinv(self._eigvecs) @ (self._B.detach() @ cm).type(self._eigvecs.dtype)
        forcing = pt.vstack(
            [(C[:, :n] * V[:, :n].flip(-1)).sum(dim=1) for n in range(1, n_steps)]
        ).T
        forcing = pt.cat((pt.zeros(self._n_states).unsqueeze(-1), forcing), dim=1)
        return (self._eigvecs @ (V * b.unsqueeze(-1) + forcing)).type(self._dm.dtype)

    @property
    def A(self) -> pt.Tensor:
        return self._A.detach()

    @property
    def B(self) -> pt.Tensor:
        return self._B.detach()

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
    def amplitude(self) -> pt.Tensor:
        return self._amplitude

    @property
    def modes(self) -> pt.Tensor:
        return self.eigvecs

    @property
    def unforced_dynamics(self) -> pt.Tensor:
        return pt.linalg.vander(
            self._eigvals, N=self._n_times
        ) * self._amplitude.unsqueeze(-1)

    @property
    def forced_dynamics(self) -> pt.Tensor:
        C = pt.linalg.pinv(self._eigvecs) @ (self._B.detach() @ self._cm).type(
            self._eigvals.dtype
        )
        V = pt.linalg.vander(self._eigvals, N=self._n_times - 1)
        dyn = pt.vstack(
            [(C[:, :n] * V[:, :n].flip(-1)).sum(dim=1) for n in range(1, self._n_times)]
        ).T
        return pt.cat((pt.zeros(self._n_states).unsqueeze(-1), dyn), dim=1)

    @property
    def dynamics(self) -> pt.Tensor:
        return self.unforced_dynamics + self.forced_dynamics

    @property
    def reconstruction(self) -> pt.Tensor:
        return (self._eigvecs @ self.dynamics).type(self._dm.dtype)

    @property
    def reconstruction_error(self) -> pt.Tensor:
        return self._dm - self.reconstruction

    @property
    def projection_error(self) -> pt.Tensor:
        X, Y, G = self._dm[:, :-1], self._dm[:, 1:], self._cm
        return Y - (self._A.detach() @ X + self._B.detach() @ G)

    @property
    def dft_properties(self) -> Tuple[float, float, float]:
        return _dft_properties(self._dt, self._n_times - 1)
