from pathlib import Path

from flowtorch.data.psp_mask_cli import parse_arguments


def test_parse_arguments_with_defaults():
    arguments = parse_arguments(["data.hdf5"])

    assert arguments.path == Path("data.hdf5")
    assert arguments.output is None
    assert arguments.n_snapshots is None
    assert arguments.statistic == "std"
    assert arguments.percentile == 95.0


def test_parse_arguments_with_options():
    arguments = parse_arguments(
        [
            "data.hdf5",
            "--output",
            "mask.pt",
            "--n-snapshots",
            "20",
            "--statistic",
            "max",
            "--percentile",
            "99",
        ]
    )

    assert arguments.output == Path("mask.pt")
    assert arguments.n_snapshots == 20
    assert arguments.statistic == "max"
    assert arguments.percentile == 99.0
