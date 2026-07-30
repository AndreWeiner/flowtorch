"""Definition of a common interface for all dataloaders.

This abstract base class should be used as parent class when
defining new dataloaders, e.g., to support additional file
formats.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Union
from torch import Tensor


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
