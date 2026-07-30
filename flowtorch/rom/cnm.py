"""Implementation of cluster-based network modeling (CNM).

A reference implementation by Daniel Fernex is available on
Github_. Theoretical concepts are covered in the accompanying
publication_.

.. _Github: https://github.com/fernexda/cnm
.. _publication: https://advances.sciencemag.org/content/7/25/eabf5006
"""

# standard library packages
from typing import Dict, Tuple
from collections import defaultdict, deque
from itertools import groupby

# third party packages
import numpy as np
import torch as pt
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from scipy.interpolate import InterpolatedUnivariateSpline

# flowtorch packages
from .base import ROM, Encoder
from .utils import (
    check_larger_than,
    check_int_larger_than,
    remove_sequential_duplicates,
)


class CNM(ROM):
    """Cluster-based network modeling implementation.

    In contrast to the original implementation_, the clustering, the
    computation of transition probabilities, and the propagation in
    time are condensed in a single class. The class also relies on
    Numpy and Scikit-Learn for K-means, KD-tree, and interpolation.
    These components might be replaced with PyTorch implementations
    in future releases.

    The cluster attribute has a member called labels. Each label corresponds
    to the cluster id associated with each sample of the input data. Since
    the input data are sorted according to the time at which they were sampled,
    we can interpret the labels as a sequence of visited clusters over time
    However, if two or more consecutive snapshots belong to the same cluster,
    we are only interested in the next state in the sequence that is different
    from the current cluster because we want to model the temporal behavior as
    a transition between cluster centroids. Therefore, sequential duplicates
    must be removed.

    .. _implementation: https://github.com/fernexda/cnm/tree/main/cnm

    :param dt: time step between two snapshots
    :type dt: float
    :param n_clusters: number of clusters; must be larger than one
    :type n_clusters: int, optional
    :param model_order: number of past cluster to consider when predicting
        the next cluster
    :type model_order: int, optional
    :param Q: transition probabilities
    :type Q: Dict[str, np.ndarray]
    :param T: transition times
    :type T: Dict[str, float]
    :param times: times at which cluster centroids were visited
    :type times: List[float]
    :param visited_clusters: clusters visited during the temporal evolution
    :type visited_clusters: List[int]
    :param cluster_centers: centroids of clusters
    :type cluster_centers: np.ndarray

    Examples

    >>> from flowtorch.rom import SVDEncoder, CNM
    ... assemble data matrix
    >>> encoder = SVDEncoder(rank=20)
    >>> info = encoder.train(data_matrix)
    >>> reduced_state = encoder.encode(data_matrix)
    >>> cnm = CNM(reduced_state, encoder, n_clusters=20, model_order=4)
    >>> prediction = cnm.predict(data_matrix[:, :5], end_time=1.0, step_size=0.1)
    >>> reduced_prediction = cnm.predict_reduced(reduced_state[:, :5], 1.0, 0.1)
    >>> full_prediction = encoder.decode(reduced_prediction)

    """

    def __init__(
        self,
        reduced_state: pt.Tensor,
        encoder: Encoder | None,
        dt: float,
        n_clusters: int = 10,
        model_order: int = 1,
        cluster_config: dict = {},
    ):
        """Create a new CNM instance

        :param reduced_state: time series data in the reduced state space
        :type reduced_state: pt.Tensor
        :param encoder: encoder instance used to map to the reduced state
        :type encoder: Encoder
        :param dt: time step between snapshots
        :type dt: float
        :param n_clusters: number of clusters, defaults to 10
        :type n_clusters: int, optional
        :param model_order: number of past states used to predict the next state;
            defaults to 1
        :type model_order: int, optional
        :param cluster_config: optional parameters for clustering, defaults to {}
        :type cluster_config: dict, optional
        """
        # the base class performs compatibility checks of the input data
        super(CNM, self).__init__(reduced_state, encoder)
        self.dt = dt
        self.n_clusters = n_clusters
        self.model_order = model_order
        self._dtype = reduced_state.dtype
        self._cluster = KMeans(n_clusters, **cluster_config).fit(
            reduced_state.T.numpy()  # batch dimension comes first in Scikit-Learn
        )
        self._sequence = remove_sequential_duplicates(self._cluster.labels_)
        self._transition_prob = self._compute_transition_prob()
        self._transition_time = self._compute_transition_time()
        self._times: list[float] = []
        self._visited_clusters: list[int] = []
        self._tree: KDTree | None = None

    def _compute_transition_prob(self) -> Dict[str, np.ndarray]:
        """Compute the transition probabilities between clusters.

        :raises Exception: if the model order is shorter than or equal to the
            cluster sequence found in the initial dataset
        :return: cluster sequence of length `self.model_order` as key and
            transition probabilities for each potential next cluster as value;
            the probabilities are stored as 2D arrays, where the first column
            corresponds to the id of the next cluster and the second column
            contains the associated probability
        :rtype: Dict[str, np.ndarray]
        """

        if self.model_order >= self._sequence.size:
            raise Exception(
                "Could not compute transition probabilities: "
                + f"length of cluster sequence ({self._sequence.size}) must be higher "
                + f"than chosen model order ({self.model_order})"
            )

        visited_clusters = deque(self._sequence[: self.model_order], self.model_order)
        prob = defaultdict(list)
        for next_cluster in self._sequence[self.model_order :]:
            key = ",".join(map(str, visited_clusters))
            prob[key].append(next_cluster)
            visited_clusters.append(next_cluster)
        for key, next_clusters in prob.items():
            unique, counts = np.unique(next_clusters, return_counts=True)
            prob[key] = np.stack((unique, counts / counts.sum())).T
        return prob

    def _compute_transition_time(self) -> Dict[str, float]:
        """Compute transition time between cluster centroids.

        This function first checks the list of cluster labels for sequential duplicates
        as explained in this post_. Then the transition time from the past clusters to
        the next cluster is computed. The main idea is that if a cluster in the training
        data sequence is duplicated, possibly several times, then the transition time
        from and to that cluster must be higher. For example, if the labels are
        [0, 0, 1, 3, 0, ...], then the sequence of unique clusters is [0, 1, 3, 0, ...]
        and the number of sequential occurrences is [2, 1, 1, ...]. For a model order of
        two (considering two clusters to compute the next one), the transition time for the
        sequence '0,1,3' (having been in 0 then 1, and now going to 3) is half of the
        time spent in cluster one plus half of the time spent in cluster three, which is
        0.5 * (1 + 1) * dt. It might happen that certain transitions occur multiple times
        with different transition times. Therefore, the final transition time is computed
        as the average of all observed transition times for a given sequence.

        .. _post: https://stackoverflow.com/questions/39340345/how-to-count-consecutive-duplicates-in-a-python-list

        :return: dictionary with the sequence of past clusters plus next cluster as key
            and the average transition time as value
        :rtype: Dict[str, float]
        """
        seq_duplicates = np.array(
            [sum(1 for _ in group) for _, group in groupby(self._cluster.labels_)]
        )
        transition = defaultdict(list)
        for i in range(self._sequence.size - self.model_order):
            ip_order = i + self.model_order
            key = ",".join(map(str, self._sequence[i : ip_order + 1]))
            transition[key].append(
                0.5 * self.dt * np.sum(seq_duplicates[ip_order - 1 : ip_order + 1])
            )
        return {key: float(np.mean(value)) for key, value in transition.items()}

    def _find_closest_cluster(self, label: int) -> int:
        """Find the label of the nearest cluster.

        If there are two nearest clusters with the same distance, the KDTree
        returns the index of the first cluster that was found.

        :param label: label of the cluster from which to compute the distance
        :type label: int
        :return: label of the nearest cluster
        :rtype: int
        """
        if self._tree is None:
            self._tree = KDTree(self.cluster_centers)
        _, ind = self._tree.query(
            np.expand_dims(self.cluster_centers[label, :], axis=0), 2
        )
        return int(ind[0, -1])

    def _find_history(self, history: deque) -> deque:
        """Find an alternative history based on a suggested history.

        This method is used in two places:
        1) to create an initial history if the initial reduced state provided
           fewer cluster labels than needed for the given model order
        2) to create an alternative history if the trajectory ended up in a
           dead end cluster (a cluster for which no transition probabilities
           are known); it is assumed that the last cluster of the trajectory
           was replaced with the label of the nearest neighboring cluster

        The following strategy is used to find the most similar trajectory to
        the one provided as input:
        - use as much of the given trajectory is possible and find matching
          histories in the list of available histories
        - if no trajectory is found, the process is repeated is shorter segment
          of the trajectory; this process continues until at least on potential
          history is found
        - if multiple possible histories are found that are equally similar to
          the input trajectory, the final history is sampled at random

        :param history: trajectroy to replace or complete
        :type history: deque
        :return: a trajectory as similar as possible to the input for which a
            transition probability is available
        :rtype: deque
        """
        count = 0
        while count < self.model_order:
            test_history = ",".join(map(str, list(history)[count:]))
            possible_histories = [
                key
                for key in self._transition_prob.keys()
                if key.endswith(test_history)
            ]
            if len(possible_histories) > 0:
                select_key = np.random.choice(possible_histories)
                return deque(map(int, select_key.split(",")), self.model_order)
            count += 1
        raise RuntimeError("Could not find a matching cluster history")

    def _find_initial_history(self, initial_reduced_state: np.ndarray) -> deque:
        """Find a sensible initial history for a given reduced state.

        :param initial_reduced_state: initial reduced state or initial sequence
            of reduced state vectors; if a sequence is given, each column must
            form one reduced state vector
        :type initial_reduced_state: np.ndarray
        :return: initial trajectory of cluster labels with as many elements as
            required by the model order
        :rtype: deque
        """
        if initial_reduced_state.ndim == 1:
            label = self._cluster.predict(np.expand_dims(initial_reduced_state, axis=0))
        else:
            label = self._cluster.predict(initial_reduced_state.T)
        history = deque(remove_sequential_duplicates(label), self.model_order)
        if len(history) < self.model_order:
            return self._find_history(history)
        else:
            return history

    def _sample_next_cluster(self, history: deque) -> Tuple[deque, float]:
        """Sample next cluster and corresponding transition time.

        In the key resulting from the current history is not present in the
        list of available transition probabilities, the last cluster label
        in the given history is replaced with the label of the nearest cluster,
        and a new history is searched.

        :param history: current history/past/trajectory
        :type history: deque
        :return: the history with the new cluster appended and the transition time
        :rtype: Tuple[deque, float]
        """
        key = ",".join(map(str, list(history)))
        if not key in self._transition_prob:
            last_cluster = history.pop()
            history.append(self._find_closest_cluster(last_cluster))
            history = self._find_history(history)
            key = ",".join(map(str, list(history)))
        next_cluster = int(
            np.random.choice(
                self._transition_prob[key][:, 0], p=self._transition_prob[key][:, 1]
            )
        )
        key += ",{:d}".format(next_cluster)
        history.append(next_cluster)
        return history, self._transition_time[key]

    def _interpolate_trajectory(self, step_size: float) -> pt.Tensor:
        """Add interpolated clusters to overall trajectory.

        :param step_size: time step size at which to place clusters
        :type step_size: float
        :raises ValueError: if the trajectory has fewer than two clusters
        :return: trajectory with interpolated clusters
        :rtype: pt.Tensor
        """
        if len(self.times) < 2:
            raise ValueError(
                "At least two predictions must be available "
                + "to interpolate a trajectory"
            )
        times = np.arange(self.times[0], self.times[-1] + 0.5 * step_size, step_size)
        if self.encoder is None:
            state_size = self.cluster_centers.shape[-1]
        else:
            state_size = self.encoder.reduced_state_size
        prediction = pt.empty((state_size, times.size), dtype=self._dtype)
        for dim in range(state_size):
            spline = InterpolatedUnivariateSpline(
                self._times,
                self.cluster_centers[self.visited_clusters][:, dim],
                k=min(3, len(self.times) - 1),
            )
            prediction[dim, :] = pt.from_numpy(spline(times))
        return prediction

    def predict_reduced(
        self, initial_state: pt.Tensor, end_time: float, step_size: float
    ) -> pt.Tensor:
        """Advance given reduced state in time.

        :param initial_state: initial reduced state vector or sequence of
            reduced state vectors; the state vectors must form the columns
            of the input tensor if a sequence is given
        :type initial_state: pt.Tensor
        :param end_time: time at which to step the simulation; the simulation
            time always starts at zero
        :type end_time: float
        :param step_size: time step size to advance simulation time; note that
            the sampling of the next cluster is independent of the step size;
            the step size is only used to interpolate states inbetween the
            sampled clusters
        :type step_size: float
        :return: temporal evolution of the given reduced state vector; each
            column forms a temporal snapshot
        :rtype: pt.Tensor
        """
        history = self._find_initial_history(initial_state.numpy())
        self._visited_clusters = [history[-1]]
        self._times = [0.0]
        while self._times[-1] < end_time:
            history, trans_time = self._sample_next_cluster(history)
            self._times.append(self._times[-1] + trans_time)
            self._visited_clusters.append(history[-1])
        return self._interpolate_trajectory(step_size)

    @property
    def dt(self) -> float:
        return self._dt

    @dt.setter
    def dt(self, value):
        check_larger_than(value, 0.0, "dt")
        self._dt = value

    @property
    def n_clusters(self) -> int:
        return self._n_clusters

    @n_clusters.setter
    def n_clusters(self, value: int):
        check_int_larger_than(value, 1, "n_clusters")
        self._n_clusters = value

    @property
    def model_order(self) -> int:
        return self._model_order

    @model_order.setter
    def model_order(self, value):
        check_int_larger_than(value, 0, "model_order")
        self._model_order = value

    @property
    def cluster_centers(self):
        return self._cluster.cluster_centers_

    @property
    def visited_clusters(self) -> list[int]:
        return self._visited_clusters

    @property
    def times(self) -> list[float]:
        return self._times

    @property
    def Q(self):
        return self._transition_prob

    @property
    def T(self):
        return self._transition_time
