"""Implements a linear model of the form y = Ax."""

from typing import Any, Union, List, Callable, Type, Tuple
from math import sqrt
from collections import defaultdict
import torch as pt
from torch.utils.data import DataLoader
from numpy import pi
from scipy.ndimage import gaussian_filter1d
from .dmd import _fb_consistent_operator
from .svd import SVD
from .utils import trajectory_train_test_split, EarlyStopping, DEFAULT_SCHEDULER_OPT


def _data_consistent(data: List[pt.Tensor]) -> bool:
    shapes = [dm.shape for dm in data]
    types = [dm.dtype for dm in data]
    return (
        all([shapes[0] == s for s in shapes] + [types[0] == t for t in types])
        and shapes[0][1] > 1
    )


def _least_squares_operator(data: List[pt.Tensor], init_fb) -> pt.Tensor:
    X = pt.cat([dm[:, :-1] for dm in data], dim=-1)
    Y = pt.cat([dm[:, 1:] for dm in data], dim=-1)
    if init_fb:
        _, A = _fb_consistent_operator(X, Y)
        return A
    else:
        return Y @ pt.linalg.pinv(X)


def _initialize_noise(data: List[pt.Tensor], sigma: Union[None, float]) -> pt.Tensor:
    if sigma is not None:
        noise = [
            dm - pt.from_numpy(gaussian_filter1d(dm.numpy(), sigma)) for dm in data
        ]
    else:
        noise = [pt.zeros_like(dm) for dm in data]
    return pt.cat(noise, dim=-1)


def _fro_loss_operator(
    label: pt.Tensor, prediction: pt.Tensor, *parameters: object
) -> pt.Tensor:
    return (label - prediction).norm() / sqrt(prediction.numel())


class LinearModel(pt.nn.Module):
    def __init__(
        self,
        dm: Union[pt.Tensor, List[pt.Tensor]],
        dt: float,
        decompose_operator: bool = False,
        rank: Union[int, None] = None,
        init_forward_backward: bool = True,
        sigma_filter: Union[None, float] = None,
    ):
        super(LinearModel, self).__init__()
        self._dm = [dm] if isinstance(dm, pt.Tensor) else dm
        self._dt = dt
        self._decompose = decompose_operator
        if not _data_consistent(self._dm):
            raise ValueError(
                "Inconsistent input data; data matrices must have\n"
                + "1) the same shape\n"
                + "2) the same data type\n"
                + "3) at least 2 snapshots"
            )
        self._n_states, self._n_times = self._dm[0].shape
        self._noise = pt.nn.Parameter(_initialize_noise(self._dm, sigma_filter))
        self._A: pt.Tensor | None
        if self._decompose:
            dm = pt.cat(self._dm, dim=-1)
            svd = SVD(dm, rank=rank)
            self._rank = svd.rank
            dm_pro = [svd.U.T @ dm for dm in self._dm]
            A = _least_squares_operator(dm_pro, init_forward_backward)
            evals, evecs = pt.linalg.eig(A)
            self._eigvals = pt.nn.Parameter(evals)
            self._eigvecs = pt.nn.Parameter(svd.U.type(evecs.dtype) @ evecs)
            self._A = None
            del dm, svd, A, evals, evecs
        else:
            self._A = pt.nn.Parameter(
                _least_squares_operator(self._dm, init_forward_backward)
            )
            self._eigvals, self._eigvecs = pt.linalg.eig(self._A.detach())
        self._log: dict[str, list[Any]] = defaultdict(list)

    def _operator(self) -> pt.Tensor:
        """Return the full-state operator when it is available."""
        if self._A is None:
            raise RuntimeError("The full-state operator is not explicitly available")
        return self._A

    def forward(
        self, x: pt.Tensor, noise_idx: pt.Tensor, n_steps: int, backward: bool
    ) -> pt.Tensor:
        if self._decompose:
            evecs_inv = pt.linalg.pinv(self._eigvecs)
            evals = 1.0 / self._eigvals if backward else self._eigvals
            vander = pt.linalg.vander(evals, N=n_steps + 1)[:, 1:]
            B = (x - self._noise[:, noise_idx].T).type(evals.dtype) @ evecs_inv.T
            return self._eigvecs.unsqueeze(0) @ (B.unsqueeze(-1) * vander.unsqueeze(0))
        else:
            operator = self._operator()
            A = pt.linalg.inv(operator) if backward else operator
            rollout = [A @ (x - self._noise[:, noise_idx].T).T]
            for _ in range(n_steps - 1):
                rollout.append(A @ rollout[-1])
            rollout = pt.vstack([x.T.flatten() for x in rollout]).split(
                self._n_states, dim=1
            )
            return pt.cat([x.T.unsqueeze(0) for x in rollout], dim=0)

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
        print("Optimization started")
        while e < epochs and not stop:
            if split_options.get("forward_backward"):
                train_loss_f, train_loss_b = [], []
                for batch in train_loader:
                    ffi, ff, lfi, lf, fbi, fb, lbi, lb = [d.to(device) for d in batch]
                    pred_f = self.forward(ff, ffi, lf.shape[-1], False)
                    pred_b = self.forward(fb, fbi, lb.shape[-1], True)
                    noise_f = pt.cat(
                        [self._noise[:, idx].unsqueeze(0) for idx in lfi], dim=0
                    )
                    noise_b = pt.cat(
                        [self._noise[:, idx].unsqueeze(0) for idx in lbi], dim=0
                    )
                    loss_f = loss_function(
                        lf, pred_f, noise_f, self._A, self._eigvecs, self._eigvals
                    )
                    loss_b = loss_function(
                        lb, pred_b, noise_b, self._A, self._eigvecs, self._eigvals
                    )
                    optim.zero_grad()
                    loss = 0.5 * (loss_f + loss_b)
                    loss.backward()
                    optim.step()
                    train_loss_f.append(loss_f.item())
                    train_loss_b.append(loss_b.item())
                self._log["train_loss_f"].append(sum(train_loss_f) / len(train_loader))
                self._log["train_loss_b"].append(sum(train_loss_b) / len(train_loader))
                self._log["train_loss"].append(
                    0.5
                    * (self._log["train_loss_f"][-1] + self._log["train_loss_b"][-1])
                )
                if n_val > 0:
                    with pt.no_grad():
                        batch = val_set[:]
                        ffi, ff, lfi, lf, fbi, fb, lbi, lb = [
                            d.to(device) for d in batch
                        ]
                        noise_f = pt.cat(
                            [self._noise[:, idx].unsqueeze(0) for idx in lfi], dim=0
                        )
                        noise_b = pt.cat(
                            [self._noise[:, idx].unsqueeze(0) for idx in lbi], dim=0
                        )
                        val_loss_f = loss_function(
                            lf,
                            self.forward(ff, ffi, lf.shape[-1], False),
                            noise_f,
                            self._A,
                            self._eigvecs,
                            self._eigvals,
                        ).item()
                        val_loss_b = loss_function(
                            lb,
                            self.forward(fb, fbi, lb.shape[-1], True),
                            noise_b,
                            self._A,
                            self._eigvecs,
                            self._eigvals,
                        ).item()
                        self._log["val_loss_f"].append(val_loss_f)
                        self._log["val_loss_b"].append(val_loss_b)
                        self._log["val_loss"].append(0.5 * (val_loss_f + val_loss_b))
            else:
                train_loss = 0.0
                for batch in train_loader:
                    ffi, ff, lfi, lf = (d.to(device) for d in batch)
                    prediction = self.forward(ff, ffi, lf.shape[-1], False)
                    noise = pt.cat(
                        [self._noise[:, idx].unsqueeze(0) for idx in lfi], dim=0
                    )
                    loss = loss_function(
                        lf, prediction, noise, self._A, self._eigvecs, self._eigvals
                    )
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    train_loss += loss.item()
                self._log["train_loss"].append(train_loss / len(train_loader))
                if n_val > 0:
                    with pt.no_grad():
                        batch = val_set[:]
                        ffi, ff, lfi, lf = [d.to(device) for d in batch]
                        noise = pt.cat(
                            [self._noise[:, idx].unsqueeze(0) for idx in lfi], dim=0
                        )
                        val_loss = loss_function(
                            lf,
                            self.forward(ff, ffi, lf.shape[-1], False),
                            noise,
                            self._A,
                            self._eigvecs,
                            self._eigvals,
                        ).item()
                        self._log["val_loss"].append(val_loss)
            self._log["noise_norm"].append(self._noise.detach().norm().item())
            self._log["lr"].append(optim.param_groups[0]["lr"])
            val_loss = self._log["val_loss"][-1] if n_val > 0 else 0.0
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
        self.to("cpu")
        print("Optimization completed")
        if not self._decompose:
            self._eigvals, self._eigvecs = pt.linalg.eig(self._operator().detach())

    def top_modes(
        self,
        n: int = 10,
        integral: bool = False,
        f_min: float = -float("inf"),
        f_max: float = float("inf"),
    ) -> pt.Tensor:
        importance = self.integral_contribution if integral else self.amplitude
        importance = pt.vstack(importance).mean(dim=0)
        modes_in_range = pt.logical_and(self.frequency >= f_min, self.frequency < f_max)
        mode_indices = pt.tensor(range(modes_in_range.shape[0]), dtype=pt.int64)[
            modes_in_range
        ]
        n = min(n, mode_indices.shape[0])
        top_n = importance[mode_indices].abs().topk(n).indices
        return mode_indices[top_n]

    def predict(
        self,
        initial_condition: pt.Tensor,
        n_steps: int,
        mask: Union[None, pt.Tensor] = None,
    ) -> pt.Tensor:
        b = pt.linalg.pinv(self.eigvecs) @ initial_condition.type(self.eigvecs.dtype)
        if mask is not None:
            b *= mask
        dyn = pt.linalg.vander(self.eigvals, N=n_steps + 1) * b.unsqueeze(-1)
        prediction = self.eigvecs @ dyn
        if not pt.is_complex(initial_condition):
            prediction = prediction.real
        return prediction

    def load_state_dict(self, *args, **kwargs):
        keys = super().load_state_dict(*args, **kwargs)
        if not self._decompose:
            self._eigvals, self._eigvecs = pt.linalg.eig(self._operator().detach())
        return keys

    @property
    def A(self) -> pt.Tensor:
        if self._decompose:
            return (
                self.eigvecs @ pt.diag(self.eigvals) @ pt.linalg.pinv(self.eigvecs)
            ).detach()
        else:
            return self._operator().detach()

    @property
    def noise(self) -> Tuple[pt.Tensor]:
        return self._noise.detach().split(self._n_times, dim=-1)

    @property
    def log(self) -> dict:
        return self._log

    @property
    def eigvals(self) -> pt.Tensor:
        return self._eigvals.detach()

    @property
    def eigvals_cont(self) -> pt.Tensor:
        return pt.log(self.eigvals) / self._dt

    @property
    def eigvecs(self) -> pt.Tensor:
        return self._eigvecs.detach()

    @property
    def frequency(self) -> pt.Tensor:
        return pt.log(self.eigvals).imag / (2.0 * pi * self._dt)

    @property
    def growth_rate(self) -> pt.Tensor:
        return (pt.log(self.eigvals) / self._dt).real

    @property
    def amplitude(self) -> List[pt.Tensor]:
        ev_inv = pt.linalg.pinv(self.eigvecs)
        return [
            ev_inv @ (dm[:, 0] - noise[:, 0]).type(ev_inv.dtype)
            for dm, noise in zip(self._dm, self.noise)
        ]

    @property
    def dynamics(self) -> List[pt.Tensor]:
        vander = pt.linalg.vander(self.eigvals, N=self._n_times)
        return [vander * b.unsqueeze(-1) for b in self.amplitude]

    @property
    def integral_contribution(self) -> List[pt.Tensor]:
        return [dyn.abs().sum(dim=1) for dyn in self.dynamics]

    @property
    def reconstruction(self) -> List[pt.Tensor]:
        rec = [self.eigvecs @ dyn for dyn in self.dynamics]
        if not pt.is_complex(self._dm[0]):
            rec = [r.real for r in rec]
        return rec

    @property
    def reconstruction_error(self) -> List[pt.Tensor]:
        return [dm - rec for dm, rec in zip(self._dm, self.reconstruction)]
