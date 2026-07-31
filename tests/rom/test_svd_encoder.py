# thirdparty packages
import torch as pt
import pytest

# flowtorch packages
from tests.helpers import requires_datasets

requires_dataset = requires_datasets("of_cylinder2D_binary")
pytestmark = [requires_dataset, pytest.mark.integration]

from flowtorch import DATASETS
from flowtorch.data import FOAMDataloader
from flowtorch.analysis import SVD
from flowtorch.rom import SVDEncoder


class TestSVDEncoder:
    def setup_method(self):
        loader = FOAMDataloader(DATASETS["of_cylinder2D_binary"])
        self.data = loader.load_snapshot("p", loader.write_times[1:11])
        self.rows, self.cols = self.data.shape

    def test_train(self):
        encoder = SVDEncoder(rank=100)
        assert encoder.trained == False
        info = encoder.train(self.data)
        assert "execution_time" in info
        assert encoder.trained == True
        assert encoder.state_shape == (self.rows,)
        assert encoder.reduced_state_size == self.cols

    def test_encode(self):
        encoder = SVDEncoder(rank=self.cols)
        with pytest.raises(Exception):
            encoder.encode(self.data[:, 0])
        encoder.train(self.data)
        with pytest.raises(ValueError):
            encoder.encode(pt.ones(10))
        with pytest.raises(ValueError):
            encoder.encode(pt.ones((3, 3, 3)))
        svd = SVD(self.data, rank=self.cols)
        a0 = encoder.encode(self.data[:, 0])
        assert pt.allclose(a0, svd.V[0, :] * svd.s, rtol=1.0e-2)
        A = encoder.encode(self.data[:, :5])
        assert pt.allclose(A, pt.diag(svd.s) @ svd.V.conj().T[:, :5], rtol=1.0e-2)

    def test_decode(self):
        encoder = SVDEncoder(rank=self.cols)
        with pytest.raises(Exception):
            encoder.decode(pt.ones(10))
        encoder.train(self.data)
        with pytest.raises(ValueError):
            encoder.decode(pt.ones(20))
        with pytest.raises(ValueError):
            encoder.decode(pt.ones((2, 2, 2)))
        a0 = encoder.encode(self.data[:, 0])
        x0 = encoder.decode(a0)
        assert pt.allclose(self.data[:, 0], x0, rtol=1.0e-2)
        A = encoder.encode(self.data)
        X = encoder.decode(A)
        assert pt.allclose(self.data, X, rtol=1.0e-2)
