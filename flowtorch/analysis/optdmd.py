"""Optimized DMD via gradient descent and backpropagation.
"""

# standard library packages
from typing import Union, Set, Callable, Tuple
from collections import defaultdict
from math import sqrt

# third party packages
import torch as pt
from numpy import pi

# flowtorch packages
from .dmd import DMD
from .utils import EarlyStopping, DEFAULT_SCHEDULER_OPT


def _create_conj_complex_pairs(ev: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
    """Create indices to preserve conjugate complex eigenvalues.

    For real input data, eigenvalues and eigenvectors must come as
    conjugate complex pairs. If there is an odd number of eigenvalues,
    one eigenvalue is reserved to be on the real axis. If there are
    multiple eigenvalues without imaginary part and the number of eigenvalues
    is odd, only the one with real part closest to unity is preserved.

    :param ev: tensor of eigenvalues
    :type ev: pt.Tensor
    :return: indices of eigenvalues to keep and indices of eigenvalues
         with conjugate complex pair
    :rtype: Tuple[pt.Tensor, pt.Tensor]
    """
    sort_imag = ev.imag.sort(descending=True).indices
    zero_imag = ev.imag == 0
    n = ev.size(0)
    indices = pt.tensor(range(n), dtype=pt.int64)
    pairs = indices[: n // 2]
    if zero_imag.sum() <= 1:  # unique real ev
        keep = indices[sort_imag][: n // 2 + n % 2]
    else:  # multiple real ev
        if n % 2 == 1:  # preserve ev with real part closest to unity
            keep_real = (ev[zero_imag].real - 1.0).abs().sort().indices[0]
            keep_real = indices[zero_imag][keep_real]
            keep = pt.cat((indices[sort_imag][: n // 2], keep_real.unsqueeze(-1)))
        else:  # do not preserve any real ev
            keep = indices[sort_imag][: n // 2]
    return keep, pairs


def fro_loss(
    label: pt.Tensor, prediction: pt.Tensor, eigvecs: pt.Tensor, eigvals: pt.Tensor
) -> pt.Tensor:
    """Compute the Frobenius norm of the prediction error.

    Note: in contrast to the default norm function in PyTorch,
    the norm is normalized by the number of elements in the
    prediction tensor.

    Note: the argument list is more generic than necessary for this
    loss function; the eigenvectors and eigenvalues may be used for
    regularization or to enforce physical constraints in more advanced
    loss functions.

    :param label: ground truth
    :type label: pt.Tensor
    :param prediction: predictions
    :type prediction: pt.Tensor
    :return: L2 norm normalized by number of elements
    :rtype: pt.Tensor
    """
    return (label - prediction).norm() / sqrt(prediction.numel())


class OptDMD(pt.nn.Module):
    """Optimized DMD based on backpropagation and gradient descent.

    For a detailed description of this DMD variant, refer to
    `Weiner and Semaan (2023) <https://arxiv.org/abs/2312.12928>`_.

    Examples

    >>> import torch as pt
    >>> from flowtorch.analysis import OptDMD
    >>> dm = pt.rand((200, 100))
    >>> dmd = DMD(dm, dt=1.0)
    >>> dmd.train(stopping_options={'patience' : 80, 'checkpoint' : '/tmp/best_model.pt'})
    >>> dmd.load_state_dict(pt.load('/tmp/best_model.pt'))
    >>> train_loss = dmd.log['train_loss']
    >>> val_loss = dmd.log['val_loss']
    """

    def __init__(self, *dmd_args, **dmd_kwargs):
        """Construct an optimized DMD instance.

        The arguments and keyword arguments passed to the constructor are
        used to create a regular `DMD` instance for initialization. The
        arguments specific to the optimization of eigenvectors and eigenvalues
        are passed to the `train` method.

        Warning: the implementation was only tested rigorously for real input
        data; eigenvectors and eigenvalues are enforced to have complex conjugate
        pairs in case of real input data.
        """
        super(OptDMD, self).__init__()
        self._dmd = DMD(*dmd_args, **dmd_kwargs)
        if self._dmd._complex:
            n_modes = self._dmd.eigvals.size(0)
            keep = pt.tensor(range(n_modes), dtype=pt.int64)
            self._conj_indices = pt.zeros(n_modes, dtype=pt.bool)
        else:
            keep, pairs = _create_conj_complex_pairs(self._dmd.eigvals)
            self._conj_indices = pairs
        scaled_modes = self._dmd._modes[:, keep] * self._dmd.amplitude[keep]
        self._eigvecs = pt.nn.Parameter(scaled_modes)
        self._eigvals = pt.nn.Parameter(self._dmd._eigvals[keep].clone())
        self._log = defaultdict(list)

    def _create_train_val_split(
        self, train_size: Union[int, float], val_size: Union[int, float]
    ) -> tuple:
        """Split the data matrix into training and validation data.

        In contrast to general machine learning tasks, the data is not
        split randomly. The validation data are taken from the end of
        the snapshot sequence. The provided values for training and validation
        size are weighted by their sum such that always a valid split results.

        :param train_size: size of the training data set; can be the number
            of the snapshots or a fraction of the overall number of snapshots
        :type train_size: Union[int, float]
        :param val_size: size of the validation data set; can be the number
            of the snapshots or a fraction of the overall number of snapshots
        :type val_size: Union[int, float]
        :return: training and validation data wrapped as `TensorDataset`
        :rtype: tuple
        """
        data = pt.utils.data.TensorDataset(
            pt.tensor(range(self._dmd._dm[0].shape[1]), dtype=pt.int64))
        n_train = int(len(data) * train_size / (train_size + val_size))
        n_val = len(data) - n_train
        if n_val > 0:
            return pt.utils.data.TensorDataset(
                data[:n_train][0]
            ), pt.utils.data.TensorDataset(data[n_train:][0])
        else:
            return data, None

    def forward(self, time_indices: pt.Tensor) -> pt.Tensor:
        evals = pt.cat((self._eigvals, self._eigvals[self._conj_indices].conj()))
        evecs = pt.cat(
            (self._eigvecs, self._eigvecs[:, self._conj_indices].conj()), dim=1
        )
        vander = pt.vstack([evals ** n.item() for n in time_indices[0]]).T
        return evecs @ vander

    def train(
        self,
        epochs: int = 1000,
        lr: float = 1.0e-3,
        batch_size: int = None,
        train_size: Union[int, float] = 0.75,
        val_size: Union[int, float] = 0.25,
        loss_function: Callable = fro_loss,
        scheduler_options: dict = {},
        stopping_options: dict = {},
        loss_key: str = "val_loss"
    ):
        """Optimize modes and dynamics based on gradient descent.

        :param epochs: number of training iterations, defaults to 1000
        :type epochs: int, optional
        :param lr: initial learning rate, defaults to 1.0e-3
        :type lr: float, optional
        :param batch_size: batch size for batch training, defaults to None
        :type batch_size: int, optional
        :param train_size: fraction or number of snapshots to use for
            training; defaults to 0.75
        :type train_size: Union[int, float], optional
        :param val_size: fraction or number of snapshots to use for
            validation; defaults to 0.25
        :type val_size: Union[int, float], optional
        :param loss_function: user-defined loss function, e.g., to add
            sparsity promotion, defaults to fro_loss
        :type loss_function: Callable, optional
        :param scheduler_options: options passed to learning rate scheduler;
            refer to PyTorch's `ReduceLROnPlateau` documentation; defaults to {}
        :type scheduler_options: dict, optional
        :param stopping_options: options to modify early stopping behavior;
            refer to the `EarlyStopping` class; defaults to {}
        :type stopping_options: dict, optional
        :param loss_key: key of loss value based on which the learning rate schedule
            and early stopping criteria are evaluated; can be 'train_loss',
            'val_loss', or 'full_loss'; defaults to "val_loss"
        :type loss_key: str, optional
        """
        optimizer = pt.optim.AdamW(self.parameters(), lr=lr)
        options = {
            key: scheduler_options[key] if key in scheduler_options else val
            for key, val in DEFAULT_SCHEDULER_OPT.items()
        }
        scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, **options
        )
        train_set, val_set = self._create_train_val_split(train_size, val_size)
        batch_size, shuffle = (
            (batch_size, True) if batch_size else (len(train_set), False)
        )
        train_loader = pt.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=shuffle
        )
        stopper = EarlyStopping(model=self, **stopping_options)
        e, stop = 0, False
        while e < epochs and not stop:
            train_loss = 0.0
            for batch in train_loader:
                pred = self.forward(batch)
                loss = loss_function(
                    self._dmd._dm[0][:, batch[0]], pred, self._eigvecs, self._eigvals
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            self._log["train_loss"].append(train_loss / len(train_loader))
            if val_set is not None:
                with pt.no_grad():
                    batch = val_set[:]
                    self._log["val_loss"].append(
                        loss_function(
                            self._dmd._dm[0][:, batch[0]],
                            self.forward(batch),
                            self._eigvecs,
                            self._eigvals,
                        ).item()
                    )
            self._log["lr"].append(optimizer.param_groups[0]["lr"])
            val_loss = self._log["val_loss"][-1] if val_set is not None else 0.0
            self._log["full_loss"].append(val_loss + self._log["train_loss"][-1])
            scheduler.step(self._log[loss_key][-1])
            print(
                "\rEpoch {:4d} - train loss: {:1.6e}, val loss: {:1.6e}, lr: {:1.6e}".format(
                    e, self._log["train_loss"][-1], val_loss, self._log["lr"][-1]
                ),
                end="",
            )
            e += 1
            stop = stopper(self._log[loss_key][-1])

    def partial_reconstruction(self, mode_indices: Set[int]) -> pt.Tensor:
        modes = self.modes
        mode_mask = pt.zeros(modes.shape[1], dtype=modes.dtype)
        mode_indices = pt.tensor(list(mode_indices), dtype=pt.int64)
        mode_mask[mode_indices] = 1.0
        rec = (modes * mode_mask) @ self.dynamics
        if not self._dmd._complex:
            rec = rec.real
        return rec.type(self._dmd._dm[0].dtype)

    def top_modes(
        self,
        n: int = 10,
        integral: bool = False,
        f_min: float = -float("inf"),
        f_max: float = float("inf"),
    ) -> pt.Tensor:
        importance = self.integral_contribution if integral else self.amplitude
        modes_in_range = pt.logical_and(self.frequency >= f_min, self.frequency < f_max)
        mode_indices = pt.tensor(range(modes_in_range.shape[0]), dtype=pt.int64)[
            modes_in_range
        ]
        n = min(n, mode_indices.shape[0])
        top_n = importance[mode_indices].abs().topk(n).indices
        return mode_indices[top_n]
    
    def predict(self, initial_condition: pt.Tensor, n_steps: int) -> pt.Tensor:
        """Predict evolution over N steps starting from user-defined initial conditions.

        :param initial_condition: initial state vector
        :type initial_condition: pt.Tensor
        :param n_steps: number of steps to predict
        :type n_steps: int
        :return: predicted evolution including the initial state (N+1 states are returned)
        :rtype: pt.Tensor
        """
        modes = self.eigvecs / self.amplitude
        b = pt.linalg.pinv(modes) @ initial_condition.type(modes.dtype)
        prediction = modes @ pt.diag(b) @ pt.linalg.vander(self.eigvals, N=n_steps+1)
        if not self._dmd._complex:
            prediction = prediction.real
        return prediction

    @property
    def dmd_init(self) -> DMD:
        return self._dmd

    @property
    def log(self) -> dict:
        return self._log

    @property
    def modes(self) -> pt.Tensor:
        return self.eigvecs / self.amplitude

    @property
    def eigvals(self) -> pt.Tensor:
        ev = self._eigvals.detach()
        return pt.cat((ev, ev[self._conj_indices].conj()))

    @property
    def eigvals_cont(self) -> pt.Tensor:
        return pt.log(self.eigvals) / self._dmd._dt

    @property
    def eigvecs(self) -> pt.Tensor:
        ev = self._eigvecs.detach()
        return pt.cat((ev, ev[:, self._conj_indices].conj()), dim=1)

    @property
    def frequency(self) -> pt.Tensor:
        return pt.log(self.eigvals).imag / (2.0 * pi * self._dmd._dt)

    @property
    def growth_rate(self) -> pt.Tensor:
        return (pt.log(self.eigvals) / self._dmd._dt).real

    @property
    def amplitude(self) -> pt.Tensor:
        return self.eigvecs.norm(dim=0)

    @property
    def dynamics(self) -> pt.Tensor:
        vander = pt.linalg.vander(self.eigvals, N=self._dmd._dm[0].shape[1])
        return pt.diag(self.amplitude.type(vander.dtype)) @ vander

    @property
    def integral_contribution(self) -> pt.Tensor:
        return self.amplitude**2 * self.dynamics.abs().sum(dim=1)

    @property
    def reconstruction(self) -> pt.Tensor:
        return (self.modes @ self.dynamics).real

    @property
    def reconstruction_error(self) -> pt.Tensor:
        return self._dmd._dm[0] - self.reconstruction
