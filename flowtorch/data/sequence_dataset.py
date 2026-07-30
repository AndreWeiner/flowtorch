"""Dataset class for time series forecasting."""

import torch as pt
from torch.utils.data import Dataset


class SequenceTensorDataset(Dataset):
    def __init__(self, sequence: pt.Tensor, delays: int = 1, horizon: int = 1) -> None:
        """Create a time series dataset from a tensor.

        Note: this class takes time series data and creates feature-label pairs
        by rolling a window across the sequence. The window length is the sum of
        time delays (look back) and time horizon (look ahead). The overall number
        of feature-label-pairs is determined by the sequence and window lengths.
        The implementation assumes that the fully inflated time series dataset,
        i.e., all data pairs, fits into memory. Doing so allows very efficient
        batch creation when using the dataset in a dataloader.

        :param sequence: general N-d tensor; by convention, the first dimension must
            be time, e.g., a sequence of N 1d tensor of length M yields a sequence
            tensor of size N x M; this ordering is in contrast to other parts of
            flowTorch but in line with PyTorch modules like RNN, LSTM
        :type sequence: pt.Tensor
        :param delays: number of time delays ('look back'), defaults to 1
        :type delays: int, optional
        :param horizon: length of time horizon, defaults to 1
        :type horizon: int, optional
        :raises ValueError: if the resulting dataset has a length smaller than 1

        Examples

        >>> import torch as pt
        >>> from flowtorch.data import SequenceTensorDataset
        >>> sequence = pt.rand((100, 200))
        >>> ds = SequenceTensorDataset(sequence, delays=6, horizon=2)
        >>> features, labels = ds[:]
        """
        self._sequence = (
            sequence.reshape((-1, 1)) if len(sequence.shape) == 1 else sequence
        )
        self._delays = max(delays, 1)
        self._horizon = max(horizon, 1)
        self._n_pairs = self._sequence.shape[0] - self._delays - self._horizon + 1

        if self._n_pairs < 1:
            raise ValueError(
                "Sequence not long enough to create dataset.\n"
                + f"The sequence must contain at least {self._delays + self._horizon + 1} elements."
            )
        self._features = pt.cat(
            [
                self._sequence[i : i + self._delays].unsqueeze(0)
                for i in range(self._n_pairs)
            ],
            dim=0,
        )
        self._labels = pt.cat(
            [
                self._sequence[
                    i + self._delays : i + self._delays + self._horizon
                ].unsqueeze(0)
                for i in range(self._n_pairs)
            ],
            dim=0,
        )

    def __getitem__(self, index):
        return self._features[index], self._labels[index]

    def __len__(self):
        return self._n_pairs
