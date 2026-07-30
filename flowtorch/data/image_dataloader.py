"""Load time series of typical image formats."""

# standard library packages
from os.path import join
from glob import glob
from pathlib import Path
from typing import List, NoReturn

# third party packages
import torch as pt
import numpy as np
from PIL import Image

# flowTorch packages
from .dataloader import Dataloader
from ..constants import DEFAULT_DTYPE
from .utils import check_list_or_str


class ImageDataloader(Dataloader):
    def __init__(
        self,
        path: str,
        prefix: str = "",
        suffix: str = ".tif",
        dtype: pt.dtype = DEFAULT_DTYPE,
    ):
        """Load image time series data from folder.

        :param path: folder containing the images
        :type path: str
        :param prefix: constant image base name, defaults to ""
        :type prefix: str, optional
        :param suffix: constant suffix including extension, defaults to ".tif"
        :type suffix: str, optional
        :param dtype: data type, defaults to DEFAULT_DTYPE
        :type dtype: pt.dtype, optional
        """
        self._path = path
        self._prefix = prefix
        self._suffix = suffix
        self._dtype = dtype
        self._write_times = self._parse_times()

    def _parse_times(self) -> List[str]:
        """Extract time/sequence label from file names.

        :return: extracted time values sorted in ascending order
        :rtype: List[str]
        """
        files = glob(join(self._path, f"{self._prefix}*{self._suffix}"))
        len_p, len_s = len(self._prefix), len(self._suffix)
        times = [Path(f).name[len_p:-len_s] for f in files]
        return sorted(times, key=float)

    def _build_file_path(self, time: str) -> str:
        """Build full file path to image at given time.

        :param time: time value to identify the image
        :type time: str
        :return: full path to the image file
        :rtype: str
        """
        return join(self._path, f"{self._prefix}{time}{self._suffix}")

    def _load_single_snapshot(self, time: str) -> pt.Tensor:
        """Load a single image, normalize, and convert to tensor.

        :param time: time value to identify the image
        :type time: str
        :return: normalized image data as 2D tensor
        :rtype: pt.Tensor
        """
        with Image.open(self._build_file_path(time)) as im:
            img = np.array(im)
        if img.dtype == np.uint8:
            norm = 2**8 - 1
        elif img.dtype == np.uint16:
            norm = 2**16 - 1
        else:
            norm = 1
        return (pt.from_numpy(img) / norm).type(self._dtype)

    def load_snapshot(
        self,
        field_name: List[str] | str,
        time: List[str] | str | None = None,
    ) -> pt.Tensor:
        """Load a single image or a sequence of images.

        :param time: time value or list of time values to identify the images
        :type time: List[str] | str
        :return: image or image sequence as 2D/3D tensor, where the last dimension
            represents time (in case an image sequence is loaded)
        :rtype: pt.Tensor
        """
        requested_times = field_name if time is None else time
        check_list_or_str(requested_times, "time")
        if isinstance(requested_times, list):
            first = self._load_single_snapshot(requested_times[0])
            seq = pt.empty((*first.shape, len(requested_times)), dtype=self._dtype)
            seq[:, :, 0] = first
            for i, t_i in enumerate(requested_times[1:]):
                seq[:, :, i + 1] = self._load_single_snapshot(t_i)
            return seq
        else:
            return self._load_single_snapshot(requested_times)

    @property
    def write_times(self) -> List[str]:
        """Write times or labels to identify images in a sequence.

        :return: list of available write times
        :rtype: List[str]
        """
        return self._write_times

    @property
    def field_names(self) -> NoReturn:
        raise NotImplementedError("Field names are not available for image data.")

    @property
    def vertices(self) -> NoReturn:
        raise NotImplementedError("Vertices are not available for image data.")

    @property
    def weights(self) -> NoReturn:
        raise NotImplementedError("Weights are not available for image data.")
