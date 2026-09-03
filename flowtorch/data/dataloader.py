"""Definition of a common interface for all dataloaders.

This abstract base class should be used as parent class when
defining new dataloaders, e.g., to support additional file
formats.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Sequence, Union
from torch import Tensor


def _preallocate_time_series(
    load_snapshot: Callable[[str], Tensor], times: Sequence[str]
) -> Tensor:
    """Load snapshots into a preallocated tensor with time last.

    Only the output tensor and one independently loaded snapshot are resident
    at a time.
    """
    if not times:
        raise ValueError("At least one snapshot time must be provided")
    first = load_snapshot(times[0])
    snapshot_shape = first.shape
    series = first.new_empty((*snapshot_shape, len(times)))
    series[..., 0].copy_(first)
    del first
    for index, time in enumerate(times[1:], start=1):
        snapshot = load_snapshot(time)
        if snapshot.shape != snapshot_shape:
            raise ValueError(
                "All snapshots must have the same shape; "
                f"expected {tuple(snapshot_shape)}, found {tuple(snapshot.shape)} "
                f"at time {time!r}"
            )
        series[..., index].copy_(snapshot)
    return series


class Dataloader(ABC):
    """Abstract base class to define a common interface for dataloaders."""

    @abstractmethod
    def load_snapshot(
        self, field_name: Union[List[str], str], time: Union[List[str], str]
    ) -> Union[List[Tensor], Tensor]:
        """Load one or more snapshots of one or more fields.

        :param field_name: name of the field to load
        :type field_name: Union[List[str], str]
        :param time: snapshot time
        :type time: Union[List[str], str]
        :return: field values
        :rtype: Union[List[Tensor], Tensor]

        """
        pass

    def load_snapshot_slice(
        self,
        field_name: Union[List[str], str],
        time: Union[List[str], str],
        spatial_slice: slice,
    ) -> Union[List[Tensor], Tensor]:
        """Load a slice along the first spatial dimension.

        The default implementation preserves compatibility with existing
        loaders by loading the requested snapshots first and slicing them in
        memory. Array-backed loaders may override this method to avoid reading
        values outside ``spatial_slice``.

        :param field_name: name of one or more fields
        :param time: one or more snapshot times
        :param spatial_slice: slice along the first tensor dimension
        :return: sliced field tensor or list of tensors
        """
        loaded = self.load_snapshot(field_name, time)
        if isinstance(loaded, list):
            return [field[spatial_slice] for field in loaded]
        return loaded[spatial_slice]

    def snapshot_shape(self, field_name: str, time: str) -> Sequence[int]:
        """Return the shape of one field snapshot.

        Loaders with inexpensive metadata access should override this fallback,
        which reads one snapshot.
        """
        loaded = self.load_snapshot(field_name, time)
        if isinstance(loaded, list):
            raise RuntimeError("a single field must return one tensor")
        return tuple(loaded.shape)

    @property
    @abstractmethod
    def write_times(self) -> List[str]:
        """Available write times.

        :return: list of available write times
        :rtype: List[str]

        """
        pass

    @property
    @abstractmethod
    def field_names(self) -> Dict[str, List[str]]:
        """Create a dictionary containing availale fields

        :return: dictionary with write times as keys and
            field names as values
        :rtype: Dict[str, List[str]]

        """
        pass

    @property
    @abstractmethod
    def vertices(self) -> Tensor:
        """Get the vertices at which field values are defined.

        :return: coordinates of vertices
        :rtype: Tensor

        """
        pass

    @property
    @abstractmethod
    def weights(self) -> Tensor:
        """Get the weights for field values.

        In a standard finite volume method, the weights are
        the cell volumes. For other methods, the definition
        of the weight is described in the Dataloader implementation.

        :return: weight for field values
        :rtype: Tensor

        """
        pass
