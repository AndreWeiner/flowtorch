"""Definition of a common interface for all reduced-order models (ROMs)."""

# standard library packages
from abc import ABC, abstractmethod, abstractproperty

# third party packages
from torch import Tensor, Size


class Encoder(ABC):
    """Abstract base class for dimensionality reduction algorithms.

    This base class should be used when defineing new algorithms
    for dimensionality reduction.

    :param trained: True if the encoder was trained/set up
    :type trained: bool
    """

    def __init__(self):
        """Base class constructor to initialize common properties."""
        self._trained = False

    def _check_state_shape(self, shape: Size):
        """Check if input and full state shape match.

        For some applications, the encoder input might be multi-dimensional tensor
        like an image or a sequence of images. This function checks that the shape
        of the input matches the one expected by the encoder. The check works for both
        a single state and also a sequence of states.

        :param shape: shape of the input tensor; if a sequence of state vectors is supplied,
            the last entry of the tuple is expected to be the batch dimension.
        :type shape: pt.Size
        :raises ValueError: an error is raised if encoder and input shape don't match
        """
        sequence = len(shape) == len(self.state_shape) + 1
        state_shape = shape[:-1] if sequence else shape
        if not state_shape == self.state_shape:
            raise ValueError(
                f"State shape mismatch: expected shape {tuple(self.state_shape)} "
                + f"but found shape {tuple(state_shape)}"
            )

    def _check_reduced_state_size(self, shape: Size):
        """Check if input and reduced state size match.

        :param shape: shape of the input tensor; if the shape corresponds
            to that of a sequence, the last dimension is expected to be
            the batch dimension
        :type shape: pt.Size
        :raises ValueError: if number of dimensions is not one or two
        :raises ValueError: if the size of input and encoder do not match
        """
        if not len(shape) in (1, 2):
            raise ValueError(
                "Reduced state with wrong number of dimensions:\n"
                + f"expected input with one or two dimensions but got {len(shape)}"
            )
        if not shape[0] == self.reduced_state_size:
            raise ValueError(
                f"Reduced state size mismatch: expected size of {self.reduced_state_size} "
                + f"but got {shape[0]}"
            )

    @abstractmethod
    def train(self, full_state: Tensor) -> dict:
        """Create a mapping from the full to the reduced state space.

        :param full_state: time series data; the size of the last dimension
            equals the number of snapshots (batch dimension)
        :type data: Tensor
        :return: information about the training process
        :rtype: dict
        """
        pass

    @abstractmethod
    def encode(self, full_state: Tensor) -> Tensor:
        """Map the full to the reduced state.

        :param data: snapshot or sequence of snapshots; if the input has
            one more dimension as the state (`state_shape`), the last
            dimension is considered as time/batch dimension
        :type full_state: Tensor
        :return: snapshot or sequence of snapshots in reduced state space
        :rtype: Tensor
        """
        pass

    @abstractmethod
    def decode(self, reduced_state: Tensor) -> Tensor:
        """Map the reduced state back to the full state.

        :param reduced_state: snapshot or sequence of snapshots in reduced
            state space; if there is one more dimension than in
            `reduced_state_shape`, the last dimension is considered as
            time/batch dimension
        :type data: Tensor
        :return: snapshot or sequence of snapshots in full state space
        :rtype: Tensor
        """
        pass

    @abstractproperty
    def state_shape(self) -> Size:
        """Shape of the full state tensor."""
        pass

    @abstractproperty
    def reduced_state_size(self) -> int:
        """Size of the reduced state vector."""
        pass

    @property
    def trained(self) -> bool:
        """Get the training state

        :return: True if training was completed
        :rtype: bool
        """
        return self._trained

    @trained.setter
    def trained(self, value: bool):
        self._trained = value

    @trained.deleter
    def trained(self):
        del self._trained


class ROM(ABC):
    """Abstract base class for reduced-order models.

    This base class should be used when defining new ROMs.
    """

    def __init__(self, reduced_state: Tensor, encoder: Encoder | None):
        """Create a new ROM instance.

        This constructor will be typically called by the base class.

        :param reduced_state: time series data in the reduced state space
        :type reduced_state: Tensor
        :param encoder: encoder used to create the time series data
        :type encoder: Encoder
        """
        self._encoder: Encoder | None = None
        self.encoder = encoder
        self._check_reduced_state(reduced_state)

    def _check_reduced_state(self, reduced_state: Tensor):
        """Check if the reduced state matches the encoder properties.

        :param reduced_state: time series data in the reduced state space
        :type reduced_state: Tensor
        :raises ValueError: if the time series data has more than two dimensions
        :raises ValueError: if data and encoder shapes do not match
        """
        if not len(reduced_state.shape) == 2:
            raise ValueError(
                "The time series of reduced state vectors must have exactly 2 dimensions"
            )
        if self.encoder is not None:
            sd = reduced_state.shape[0]
            se = self.encoder.reduced_state_size
            if not sd == se:
                raise ValueError(
                    f"The size of the reduced state ({sd}) "
                    + f"does not match the one expected by the encoder ({se})"
                )

    def predict(
        self, initial_state: Tensor, end_time: float, step_size: float
    ) -> Tensor:
        """Predict the evolution of a given initial full state vector.

        :param initial_state: state from which to start
        :type initial_state: Tensor
        :param end_time: when to stop the simulation; the corresponding
            start time is always assumed to be zero
        :type end_time: float
        :param step_size: time step size
        :type step_size: float
        :return: evolution of the full state vector; the last dimension
            corresponds to the time/batch dimension
        :rtype: Tensor
        """
        if self.encoder is None:
            return self.predict_reduced(initial_state, end_time, step_size)
        else:
            return self.encoder.decode(
                self.predict_reduced(
                    self.encoder.encode(initial_state), end_time, step_size
                )
            )

    @abstractmethod
    def predict_reduced(
        self, initial_state: Tensor, end_time: float, step_size: float
    ) -> Tensor:
        """Predict the evolution of a given initial reduced state vector.

        :param initial_state: initial reduced state vector
        :type initial_state: Tensor
        :param end_time: when to stop the simulation; the corresponding
            start time is always assumed to be zero
        :type end_time: float
        :param step_size: time step size
        :type step_size: float
        :return: evolution of the reduced state vector; the last dimension
            corresponds to the time/batch dimension
        :rtype: Tensor
        """
        pass

    @property
    def encoder(self) -> Encoder | None:
        """Return encoder instance."""
        return self._encoder

    @encoder.setter
    def encoder(self, encoder: Encoder | None):
        """Set the encoder.

        :param encoder: new encoder; can also be None
        :type encoder: Encoder
        :raises ValueError: if instance is not a subclass of `Encoder`
        :raises ValueError: if the encoder has not been trained
        """
        if encoder is None:
            self._encoder = encoder
        else:
            if not issubclass(type(encoder), Encoder):
                raise ValueError("The encoder must be a subclass of Encoder")
            if not encoder.trained:
                raise ValueError("The encoder must be trained before its usage")
            self._encoder = encoder
