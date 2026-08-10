"""Helper tools to detect and replace outliers in time series data."""

# standard library packages
import logging
from typing import Callable

# third party packages
import torch as pt
from torch.nn import functional as F

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _spatial_median_inward(data: pt.Tensor, window_size: tuple[int, int]) -> pt.Tensor:
    """Return spatial medians using the nearest complete boundary window."""
    if data.ndim != 3:
        raise ValueError("data must have shape (n_fields, nx, ny)")
    window_x, window_y = window_size
    _, nx, ny = data.shape
    if window_x > nx or window_y > ny:
        raise ValueError("window dimensions cannot exceed the spatial dimensions")

    neighborhoods = F.unfold(data.unsqueeze(1), kernel_size=(window_x, window_y))
    median = pt.nanmedian(neighborhoods, dim=1).values
    valid_x = nx - window_x + 1
    valid_y = ny - window_y + 1
    median = median.reshape(-1, 1, valid_x, valid_y)
    radius_x, radius_y = window_x // 2, window_y // 2
    return F.pad(
        median,
        (radius_y, radius_y, radius_x, radius_x),
        mode="replicate",
    )[:, 0]


def iqr_outlier_replacement(
    data: pt.Tensor, k: float = 1.5, nb: int = 3, replace: Callable = pt.median
) -> pt.Tensor:
    """Detect and replace outliers based on the inter-quantile range (IRQ).

    :param data: time series data; time is expected to be the last dimension
    :type data: pt.Tensor
    :param k: factor controlling the detection sensitivity; smaller values
        increase the sensitivity; defaults to 1.5
    :type k: float, optional
    :param nb: number of neighboring points in time to consider when replacing
        an outlier; points in the range i-nb:i+nb are considered for each
        outlier i; defaults to 3
    :type nb: int, optional
    :param replace: function mapping the neighboring values to the value with
        which to replace the outlier, defaults to pt.median
    :type replace: Callable, optional
    :return: clean dataset with the same shape as the input data
    :rtype: pt.Tensor
    """
    initial_shape = data.shape
    if len(initial_shape) > 2:
        data = data.flatten(start_dim=0, end_dim=-2)
    elif len(initial_shape) == 1:
        data = data.unsqueeze(-1).T
    shape = data.shape
    q25, q75 = pt.quantile(data, 0.25, dim=-1), pt.quantile(data, 0.75, dim=-1)
    iqr_k = (q75 - q25) * k
    outliers_low = data < (q25 - iqr_k).unsqueeze(-1)
    outliers_high = data > (q75 + iqr_k).unsqueeze(-1)
    outlier_indices = pt.logical_or(outliers_low, outliers_high).nonzero(as_tuple=True)
    clean_data = data.clone().detach()
    logger.info(
        "Detected {:d} outliers ({:3.2f}%).".format(
            outlier_indices[0].shape[0],
            outlier_indices[0].shape[0] / (data.shape[0] * data.shape[1]) * 100,
        )
    )
    if outlier_indices[0].shape[0] == 0:
        logger.info("Couldn't find any outliers.")
    else:
        logger.info("Start to replace outliers ...")
    for row, col in zip(*outlier_indices):
        i, j = row.item(), col.item()
        clean_data[i, j] = replace(data[i, max(0, j - nb) : min(shape[-1], j + nb + 1)])
    return clean_data.reshape(initial_shape)


def replace_spatial_outliers(
    data: pt.Tensor,
    threshold: float = 3.5,
    window_size: int = 3,
) -> pt.Tensor:
    """Replace spatial outliers using a local median and MAD criterion.

    The first two dimensions are interpreted as spatial axes. Every trailing
    index, such as snapshot and vector-component indices, is processed
    independently. Finite values whose robust local score exceeds
    ``threshold`` are replaced by the neighborhood median. Existing non-finite
    values are preserved.

    :param data: data with two spatial axes and optional trailing dimensions
    :type data: pt.Tensor
    :param threshold: robust-score threshold, defaults to 3.5
    :type threshold: float, optional
    :param window_size: positive odd spatial window size, defaults to 3
    :type window_size: int, optional
    :return: data with detected spatial outliers replaced
    :rtype: pt.Tensor

    **References**

    The robust score and default threshold follow the modified Z-score from
    B. Iglewicz and D. C. Hoaglin, *How to Detect and Handle Outliers*,
    ASQC Quality Press, 1993. Here, the statistic is applied to local 2D
    neighborhoods as a spatial Hampel-style filter, and detected values are
    replaced by the local median.
    """
    if data.ndim < 2:
        raise ValueError("data must have at least two dimensions")
    if not data.is_floating_point():
        raise ValueError("data must have a floating-point dtype")
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    if window_size == 1:
        return data.clone()

    nx, ny = data.shape[:2]
    channels = data.reshape(nx, ny, -1).permute(2, 0, 1).unsqueeze(1)
    radius = window_size // 2
    padding_mode = "reflect" if radius < min(nx, ny) else "replicate"
    padded = F.pad(channels, (radius, radius, radius, radius), mode=padding_mode)
    neighborhoods = F.unfold(padded, kernel_size=window_size)
    median = pt.nanmedian(neighborhoods, dim=1).values
    deviation = pt.abs(neighborhoods - median.unsqueeze(1))
    mad = pt.nanmedian(deviation, dim=1).values

    center = channels.flatten(start_dim=2).squeeze(1)
    scale = 1.4826 * mad + pt.finfo(data.dtype).eps
    outliers = pt.isfinite(center) & pt.isfinite(median)
    outliers &= pt.abs(center - median) > threshold * scale
    clean = pt.where(outliers, median, center)
    return clean.reshape(-1, nx, ny).permute(1, 2, 0).reshape(data.shape)
