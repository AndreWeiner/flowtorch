"""Common functionality shared by multiple analysis tools."""

from typing import Tuple, List, Union
from random import shuffle as random_shuffle
import torch as pt
from torch.utils.data import TensorDataset

DEFAULT_SCHEDULER_OPT = {"mode": "min", "factor": 0.5, "patience": 20, "min_lr": 1.0e-6}


def trajectory_train_test_split(
    trajectory: Union[List[pt.Tensor], pt.Tensor],
    train_size: Union[int, float] = 0.75,
    test_size: Union[int, float] = 0.25,
    horizon: int = 1,
    n_shift: int = 1,
    forward_backward: bool = False,
    shuffle: bool = False,
) -> Tuple[TensorDataset, TensorDataset]:
    tr = [trajectory] if isinstance(trajectory, pt.Tensor) else trajectory
    if len(tr[0].shape) == 1:
        tr = [tr_i.unsqueeze(0) for tr_i in tr]
    n_features_equal = [tr_i.shape[:-1] == tr[0].shape[:-1] for tr_i in tr]
    if not all(n_features_equal):
        raise ValueError("All trajectories must have the same number of features.")
    tr_long_enough = [tr_i.shape[-1] > horizon for tr_i in tr]
    if not all(tr_long_enough):
        raise ValueError(
            f"At least one trajectory is not long enough for the specified horizon ({horizon}).\n"
            + f"The minimum trajectory length is {horizon + 1}."
        )
    pairs = []
    index_offset = 0
    for tr_i in tr:
        start_at = list(range(0, tr_i.shape[-1] - horizon, n_shift))
        if forward_backward:
            for start in start_at:
                pairs.append(
                    (
                        pt.tensor(start + index_offset, dtype=pt.int64).unsqueeze(0),
                        tr_i[:, start].unsqueeze(0),
                        pt.tensor(
                            range(
                                start + index_offset + 1,
                                start + index_offset + 1 + horizon,
                            ),
                            dtype=pt.int64,
                        ).unsqueeze(0),
                        tr_i[:, start + 1 : start + 1 + horizon].unsqueeze(0),
                        pt.tensor(
                            start + horizon + index_offset, dtype=pt.int64
                        ).unsqueeze(0),
                        tr_i[:, start + horizon].unsqueeze(0),
                        pt.tensor(
                            range(
                                start + index_offset + horizon, start + index_offset, -1
                            ),
                            dtype=pt.int64,
                        ).unsqueeze(0),
                        tr_i[:, start : start + horizon].flip(-1).unsqueeze(0),
                    )
                )
        else:
            for start in start_at:
                pairs.append(
                    (
                        pt.tensor(start + index_offset, dtype=pt.int64).unsqueeze(0),
                        tr_i[:, start].unsqueeze(0),
                        pt.tensor(
                            range(
                                start + index_offset + 1,
                                start + index_offset + 1 + horizon,
                            ),
                            dtype=pt.int64,
                        ).unsqueeze(0),
                        tr_i[:, start + 1 : start + 1 + horizon].unsqueeze(0),
                    )
                )
        index_offset += tr_i.shape[-1]
    if shuffle:
        random_shuffle(pairs)
    n_pairs = len(pairs)
    if n_pairs == 1:
        n_train = 1
    else:
        n_train = int(n_pairs * train_size / (train_size + test_size))
    pairs = list(zip(*pairs))
    train_set = [pt.cat(ti, dim=0)[:n_train] for ti in pairs]
    test_set = [pt.cat(ti, dim=0)[n_train:] for ti in pairs]
    return TensorDataset(*train_set), TensorDataset(*test_set)


class EarlyStopping:
    """Provide stopping control for iterative optimization tasks."""

    def __init__(
        self,
        patience: int = 40,
        min_delta: float = 1.0e-4,
        checkpoint: Union[str, None] = None,
        model: Union[pt.nn.Module, None] = None,
    ):
        """Initialize a new controller instance.

        :param patience: number of iterations to wait for an improved
            loss value before stopping; defaults to 40
        :type patience: int, optional
        :param min_delta: minimum relative reduction of the loss value
            considered as an improvement; avoids overly long optimization
            with marginal improvements per iteration; defaults to 1.0e-4
        :type min_delta: float, optional
        :param checkpoint: path at which to store the best known state
            of the model; the state is not saved if None; defaults to None
        :type checkpoint: Union[str, None], optional
        :param model: instance of PyTorch model; the model's state dict is
            saved upon improvement of the loss function is a valid checkpoint
            is provided; defaults to None
        :type model: Union[pt.nn.Module, None], optional
        """
        self._patience = patience
        self._min_delta = min_delta
        self._chp = checkpoint
        self._model = model
        self._best_loss = float("inf")
        self._counter = 0
        self._stop = False

    def __call__(self, loss: float) -> bool:
        """_summary_

        :param loss: new loss value
        :type loss: float
        :return: boolean flag indicating if the optimization can be stopped
        :rtype: bool
        """
        if loss < self._best_loss * (1.0 - self._min_delta):
            self._best_loss = loss
            self._counter = 0
            if self._chp is not None and self._model is not None:
                pt.save(self._model.state_dict(), self._chp)
        else:
            self._counter += 1
            if self._counter >= self._patience:
                self._stop = True
        return self._stop


def unsqueeze_if_1d(tensor: pt.Tensor, last: bool = False):
    dim = 1 if last else 0
    return tensor.unsqueeze(dim=dim) if len(tensor.shape) < 2 else tensor
