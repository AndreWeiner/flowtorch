"""Helper tools to detect and replace outliers in time series data."""

# standard library packages
import logging
from typing import Callable

# third party packages
import torch as pt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
