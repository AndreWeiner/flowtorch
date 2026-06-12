# standard library packages
from collections import deque
# third party packages
import torch as pt
import numpy as np
import pytest
from pytest import raises
# flowtorch packages
from flowtorch.conftest import requires_datasets


requires_dataset = requires_datasets("of_cylinder2D_binary")
pytestmark = [requires_dataset, pytest.mark.integration]

from flowtorch import DATASETS
from flowtorch.data import FOAMDataloader
from flowtorch.rom import SVDEncoder, CNM
from .utils import remove_sequential_duplicates


class TestCNM():
    def setup_method(self):
        loader = FOAMDataloader(DATASETS["of_cylinder2D_binary"])
        self.data = loader.load_snapshot("p", loader.write_times[-10:])
        self.encoder = SVDEncoder(rank=20)
        _ = self.encoder.train(self.data)
        self.encoded_data = self.encoder.encode(self.data)

    def test_init(self):
        with raises(ValueError):
            cnm = CNM(pt.ones(100), self.encoder, 1.0)
        with raises(ValueError):
            cnm = CNM(pt.ones((5, 5)), self.encoder, 1.0)
        with raises(ValueError):
            cnm = CNM(self.encoded_data, self.encoder, 1.0, n_clusters=0)
        with raises(ValueError):
            cnm = CNM(self.encoded_data, self.encoder, 1.0, model_order=0)

    def test_compute_transition_prob(self):
        cnm = CNM(self.encoded_data, self.encoder, 1.0)
        cnm._sequence = np.array([0, 4, 3, 0, 2, 1, 3])
        prob = cnm._compute_transition_prob()
        assert {"0", "1", "2", "3", "4"}.issubset(prob.keys())
        assert np.allclose(prob["0"][:, 0], np.array([2, 4]))
        assert np.allclose(prob["0"][:, 1], np.array([0.5, 0.5]))
        cnm.model_order = 2
        prob = cnm._compute_transition_prob()
        assert {"0,4", "4,3", "3,0", "0,2", "2,1"}.issubset(prob.keys())
        assert np.allclose(prob["0,4"][:, 0], np.array([3]))
        assert np.allclose(prob["0,4"][:, 1], np.array([1]))

    def test_compute_transition_time(self):
        cnm = CNM(self.encoded_data, self.encoder, 1.0)
        cnm._cluster.labels_ = np.array([0, 1, 1, 0, 1, 2, 2, 3, 2, 3])
        cnm._sequence = remove_sequential_duplicates(cnm._cluster.labels_)
        transition = cnm._compute_transition_time()
        reference = {"0,1": 1.25, "1,0": 1.5,
                     "1,2": 1.5, "2,3": 1.25, "3,2": 1.0}
        assert set(reference.keys()).issubset(transition.keys())
        assert np.allclose(list(reference.values()), list(transition.values()))
        cnm.model_order = 2
        transition = cnm._compute_transition_time()
        reference = {"0,1,0": 1.5, "1,0,1": 1.0, "0,1,2": 1.5,
                     "1,2,3": 1.5, "2,3,2": 1.0, "3,2,3": 1.0}
        assert set(reference.keys()).issubset(transition.keys())
        assert np.allclose(list(reference.values()), list(transition.values()))

    def test_find_closest_cluster(self):
        cnm = CNM(self.encoded_data, self.encoder, 1.0)
        cnm._cluster.cluster_centers_ = np.array(
            [
                [0, 1, 0, 1, 0],
                [0, 1.2, 0, 1.2, 0],
                [0, 1.2, 0, 1.2, 0],
                [1, 0, 1, 0, 1],
                [1, 1, 0, 0, 1]
            ]
        )
        assert cnm._find_closest_cluster(0) == 1

    def test_find_history(self):
        cnm = CNM(self.encoded_data, self.encoder, 1.0, model_order=2)
        cnm._sequence = np.array([0, 4, 3, 0, 2, 1, 3])
        cnm._transition_prob = cnm._compute_transition_prob()
        history = list(cnm._find_history(deque([4], 2)))
        assert history == [0, 4]
        history = list(cnm._find_history(deque([1, 4], 2)))
        assert history == [0, 4]

    def test_find_initial_history(self):
        cnm = CNM(self.encoded_data, self.encoder, 1.0, model_order=2)
        # the labels are unique since the snapshots are different and
        # we have as many clusters as snapshots in this test (by design)
        labels = cnm._cluster.predict(self.encoded_data.T.numpy())
        history = cnm._find_initial_history(self.encoded_data[:, 1].numpy())
        assert list(history) == labels[:2].tolist()
        history = cnm._find_initial_history(self.encoded_data[:, :2].numpy())
        assert list(history) == labels[:2].tolist()

    def test_sample_next_cluster(self):
        cnm = CNM(self.encoded_data, self.encoder, 1.0, model_order=2)
        labels = cnm._cluster.predict(self.encoded_data.T.numpy())
        history = deque(labels[:2], 2)
        new_history, t_time = cnm._sample_next_cluster(history)
        assert list(new_history) == labels[1:3].tolist()
        key = ",".join(map(str, labels[:3]))
        assert np.allclose(t_time, cnm.T[key])
        history = deque([labels[-1], labels[1]], 2)
        closest = cnm._find_closest_cluster(history[-1])
        closest_index = np.argmin((labels - closest)**2)
        new_history, t_time = cnm._sample_next_cluster(history)
        assert list(
            new_history) == labels[closest_index:closest_index+2].tolist()

    def test_interpolate_trajectory(self):
        cnm = CNM(self.encoded_data, self.encoder, 1.0, model_order=2)
        cnm._times = [0.0, 1.0, 3.0, 4.0]
        cnm._visited_clusters = [0, 1, 2, 3]
        states = cnm._interpolate_trajectory(0.5)
        assert states.shape == (cnm.encoder.reduced_state_size, 9)
        cnm._times = [0.0, 1.0, 3.0]
        cnm._visited_clusters = [0, 1, 2]
        states = cnm._interpolate_trajectory(0.5)
        assert states.shape == (cnm.encoder.reduced_state_size, 7)

    def test_predict_reduced(self):
        cnm = CNM(self.encoded_data, self.encoder, 1.0, model_order=2)
        prediction = cnm.predict_reduced(self.encoded_data[:, :2], 3.0, 1.0)
        assert prediction.shape == (cnm.encoder.reduced_state_size, 4)
        assert pt.allclose(prediction, self.encoded_data[:, 1:5])
        prediction = cnm.predict_reduced(self.encoded_data[:, :4], 3.0, 1.0)
        assert prediction.shape == (cnm.encoder.reduced_state_size, 4)

    def test_predict(self):
        cnm = CNM(self.encoded_data, self.encoder, 1.0, model_order=2)
        prediction = cnm.predict(self.data[:, :2], 3.0, 1.0)
        assert prediction.shape == (cnm.encoder.state_shape[0], 4)
        prediction = cnm.predict(self.data[:, 0], 3.0, 1.0)
        assert prediction.shape == (cnm.encoder.state_shape[0], 4)

    def test_encode_none(self):
        data = pt.rand((2, 20))
        cnm = CNM(data, None, 1)
        prediction = cnm.predict(data[:, :1], 10, 1)
