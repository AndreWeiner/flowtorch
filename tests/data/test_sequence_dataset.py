"""Unittests for SequenceDataset class.
"""

from pytest import raises
import torch as pt
from flowtorch.data.sequence_dataset import SequenceTensorDataset


class TestSequenceDataset:
    seq_1d = pt.randint(1, 6, (6,))
    seq_2d = pt.randint(1, 6, (6, 2))
    seq_3d = pt.randint(1, 6, (6, 2, 3))

    def test_init(self):
        ds = SequenceTensorDataset(self.seq_1d, delays=0)
        assert ds._delays == 1
        ds = SequenceTensorDataset(self.seq_1d, horizon=0)
        assert ds._horizon == 1
        with raises(ValueError):
            ds = SequenceTensorDataset(self.seq_1d, delays=5, horizon=2)

    def test_sequence(self):
        for seq in (self.seq_1d, self.seq_2d, self.seq_3d):
            # delays = 1, horizon = 1
            ds = SequenceTensorDataset(seq)
            assert len(ds) == 5
            f, l = ds[:]
            assert pt.allclose(f[0].squeeze(), seq[0])
            assert pt.allclose(l[0].squeeze(), seq[1])
            assert pt.allclose(f[-1].squeeze(), seq[-2])
            assert pt.allclose(l[-1].squeeze(), seq[-1])
            # delays = 1, horizon = 2
            ds = SequenceTensorDataset(seq, horizon=2)
            assert len(ds) == 4
            f, l = ds[:]
            assert pt.allclose(f[0], seq[0])
            assert pt.allclose(l[0].squeeze(), seq[1:3])
            assert pt.allclose(f[-1], seq[-3])
            assert pt.allclose(l[-1].squeeze(), seq[-2:])
            # delays = 2, horizon = 1
            ds = SequenceTensorDataset(seq, delays=2)
            assert len(ds) == 4
            f, l = ds[:]
            assert pt.allclose(f[0].squeeze(), seq[:2])
            assert pt.allclose(l[0].squeeze(), seq[2:3])
            assert pt.allclose(f[-1].squeeze(), seq[-3:-1])
            assert pt.allclose(l[-1].squeeze(), seq[-1])
            # delays = 2, horizon = 2
            ds = SequenceTensorDataset(seq, delays=2, horizon=2)
            assert len(ds) == 3
            f, l = ds[:]
            assert pt.allclose(f[0].squeeze(), seq[:2])
            assert pt.allclose(l[0].squeeze(), seq[2:4])
            assert pt.allclose(f[-1].squeeze(), seq[-4:-2])
            assert pt.allclose(l[-1].squeeze(), seq[-2:])
