"""Command-line interface for interactive PSP mask creation.

Examples

Create a mask from the temporal standard deviation of all snapshots::

    flowtoch-mask-psp 0226.hdf5

Use the first 500 snapshots, adjust the color range, and select an output file::

    flowtoch-mask-psp 0226.hdf5 --output 0226_mask.pt \
        --n-snapshots 500 --statistic std --percentile 99

Display the temporal maximum instead of the standard deviation::

    flowtoch-mask-psp 0226.hdf5 --statistic max
"""

# standard library packages
import argparse
import logging
from pathlib import Path
from typing import Sequence

# flowTorch packages
from .selection_tools import mask_psp_interactive


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    :param arguments: arguments without the executable name; uses ``sys.argv``
        when omitted
    :type arguments: Sequence[str], optional
    :return: parsed command-line arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Interactively create a binary keep mask for PSP data."
    )
    parser.add_argument("path", type=Path, help="path to the PSP HDF5 dataset")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output path for the Torch mask file",
    )
    parser.add_argument(
        "-n",
        "--n-snapshots",
        type=int,
        help="number of snapshots used to compute the temporal statistic",
    )
    parser.add_argument(
        "--statistic",
        choices=("std", "max"),
        default="std",
        help="temporal statistic to display (default: std)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="percentile used as the upper color limit (default: 95)",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap used for the statistic (default: viridis)",
    )
    parser.add_argument(
        "--selection-pad",
        type=float,
        default=0.05,
        help="axes padding as a fraction of the image size (default: 0.05)",
    )
    parser.add_argument(
        "--view-update-interval",
        type=float,
        default=0.05,
        help="minimum delay between pan/zoom redraws (default: 0.05)",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    """Run interactive PSP mask creation from the command line.

    Examples

    .. code-block:: console

        $ flowtoch-mask-psp 0226.hdf5
        $ flowtoch-mask-psp 0226.hdf5 -o mask.pt -n 500
        $ flowtoch-mask-psp 0226.hdf5 --statistic max --percentile 99

    :param arguments: arguments without the executable name; uses ``sys.argv``
        when omitted
    :type arguments: Sequence[str], optional
    """
    options = parse_arguments(arguments)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    mask_psp_interactive(
        options.path,
        output_path=options.output,
        n_snapshots=options.n_snapshots,
        statistic=options.statistic,
        percentile=options.percentile,
        cmap=options.cmap,
        selection_pad=options.selection_pad,
        view_update_interval=options.view_update_interval,
    )


if __name__ == "__main__":
    main()
