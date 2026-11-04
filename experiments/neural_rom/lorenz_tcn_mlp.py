"""Small matched-data comparison of causal temporal features plus an MLP.

Reuses immutable inputs from the context/data study. Phases: pilot, run, plot.
The run phase requires a checked evidence gate, as in the original studies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from flowtorch.provenance import capture_environment
from flowtorch.rom import MLPAutoregressor, TCNMLPAutoregressor
from lorenz_context_data import fit, trial_id
from lorenz_benchmark import write_json

NAMES = ["MLP-32", "MLP-21", "TCN+MLP"]


def factory(name, options):
    if name == "TCN+MLP":
        return TCNMLPAutoregressor(
            3, history_length=options["history"], conv_channels=(8,) * 4, hidden_dim=32
        )
    width = {"MLP-32": 32, "MLP-21": 21}[name]
    return MLPAutoregressor(3, history_length=options["history"], hidden_dim=width)


def execute(source, output, phase):
    config = json.loads((source / "config.json").read_text())
    config["budget_seconds"] = 300
    cell = dict(group="context", context=16, budget=8192, count=4, guard=64)
    start = perf_counter()
    deadline = start + 300
    if phase == "run":
        gate = json.loads((output / "gate.json").read_text())
        assert all(x["status"] == "pass" for x in gate["checks"].values())
        assert (
            0
            < gate["proposed_run"]["estimated_seconds"]
            <= gate["proposed_run"]["budget_seconds"]
            <= 300
        )
        for name, digest in gate["fingerprints"].items():
            if hashlib.sha256((output / name).read_bytes()).hexdigest() != digest:
                raise ValueError("changed gate input: " + name)
    folder = output / phase
    folder.mkdir(parents=True, exist_ok=True)
    if list(folder.iterdir()):
        raise ValueError("fresh phase output required")
    rows = []
    replicas = config["data_replicates"] if phase == "run" else [0]
    seeds = config["network_seeds"] if phase == "run" else [101]
    for replica in replicas:
        for seed in seeds:
            for name in NAMES:
                record, raw, model = fit(
                    source,
                    cell,
                    replica,
                    seed,
                    name,
                    config,
                    deadline=deadline,
                    steps=None if phase == "run" else 100,
                    model_factory=factory,
                )
                record["receptive_field"] = (
                    model.temporal.receptive_field if name == "TCN+MLP" else None
                )
                identifier = trial_id(cell, replica, seed, name)
                write_json(folder / (identifier + ".json"), record)
                np.savez_compressed(folder / (identifier + ".npz"), **raw)
                torch.save(model.state_dict(), folder / (identifier + ".pt"))
                rows.append(record)
                print(
                    phase,
                    replica,
                    seed,
                    name,
                    "parameters",
                    record["parameters"],
                    "horizon",
                    round(record["horizon_mean"], 3),
                    flush=True,
                )
    write_json(
        output / (phase + "_record.json"),
        dict(
            trials=len(rows),
            seconds=perf_counter() - start,
            config=config,
            source=str(source.resolve()),
            environment=capture_environment(),
        ),
    )


def plot(source, output, use_tex=True):
    from plot_lorenz_benchmark import STYLE, save
    import matplotlib.pyplot as plt

    config = json.loads((source / "config.json").read_text())
    rows = [json.loads(p.read_text()) for p in (output / "run").glob("*.json")]
    assert len(rows) == 27
    with np.load(source / "held_out.npz") as data:
        physical = data["test"]
    target = (
        ((physical[:, 64:] - np.array(config["mean"])) / np.array(config["scale"]))
        .astype(np.float32)
        .astype(np.float64)
    )
    switching = np.diff(physical[:, 64:, 0] > 0, axis=1).any(axis=1)
    for row in rows:
        stem = trial_id(row["cell"], row["replicate"], row["seed"], row["model"])
        with np.load(output / "run" / (stem + ".npz")) as raw:
            error = np.sqrt(
                np.mean((raw["prediction"].astype(np.float64) - target) ** 2, axis=-1)
            )
            np.testing.assert_allclose(
                error, raw["error"], rtol=1e-12, atol=1e-12, equal_nan=True
            )
            crossed = (error > 0.5) | ~np.isfinite(error)
            horizon = np.array(
                [(np.flatnonzero(x)[0] + 1) * 0.01 if x.any() else 10 for x in crossed]
            )
            np.testing.assert_equal(horizon, raw["horizons"])
            row["horizon_switching"] = float(horizon[switching].mean())
            # Re-running the unchanged baseline should reproduce the earlier experiment.
            if row["model"] == "MLP-32":
                old_stem = trial_id(row["cell"], row["replicate"], row["seed"], "MLP")
                with np.load(source / "trials" / (old_stem + ".npz")) as old:
                    np.testing.assert_allclose(
                        raw["prediction"],
                        old["prediction"],
                        rtol=0,
                        atol=0,
                        equal_nan=True,
                    )
    summary = []
    for name in NAMES:
        subset = [row for row in rows if row["model"] == name]
        result = dict(
            model=name,
            parameters=subset[0]["parameters"],
            nonfinite_test_runs=sum(x["nonfinite_trajectories"] > 0 for x in subset),
        )
        for metric in [
            "validation_mse",
            "horizon_mean",
            "horizon_switching",
            "training_seconds",
        ]:
            data_means = [
                float(np.mean([row[metric] for row in subset if row["replicate"] == r]))
                for r in config["data_replicates"]
            ]
            result[metric] = dict(
                mean=float(np.mean(data_means)), data_replicate_means=data_means
            )
        summary.append(result)
    write_json(output / "summary.json", summary)
    write_json(
        output / "analysis_checks.json",
        dict(
            trials=27,
            first_crossing_times=324,
            errors="independent NumPy vs Torch agreement 1e-12",
            previous_baseline="all nine MLP-32 predictions reproduce previous study exactly",
            switching_subset="post-hoc diagnostic inherited from the previous study, no new selection",
        ),
    )
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    colors = ["#0072B2", "#56B4E9", "#D55E00"]
    with plt.rc_context({**STYLE, "text.usetex": use_tex}):
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), layout="constrained")
        for ax, metric, title in zip(
            axes,
            ["validation_mse", "horizon_mean", "horizon_switching"],
            [
                "One-time-unit validation",
                "All 12 test trajectories",
                "Seven switching trajectories",
            ],
        ):
            for i, row in enumerate(summary):
                values = row[metric]["data_replicate_means"]
                ax.scatter([i] * 3, values, color=colors[i], s=24)
                ax.plot(
                    [i - 0.18, i + 0.18],
                    [row[metric]["mean"]] * 2,
                    color=colors[i],
                    lw=2,
                )
            ax.set_xticks(range(3), NAMES, rotation=25)
            ax.set_title(title, fontsize=11)
            ax.grid(axis="y", alpha=0.15)
            if metric == "validation_mse":
                ax.set_yscale("log")
                ax.set_ylabel("Validation MSE")
            else:
                ax.set_ylim(0, 10 if metric == "horizon_mean" else 5)
                ax.set_ylabel("Mean useful horizon")
        save(fig, figures, "tcn_mlp_comparison")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["pilot", "run", "plot"])
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--no-tex", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.phase == "plot":
        plot(args.source, args.output, not args.no_tex)
    else:
        execute(args.source, args.output, args.phase)


if __name__ == "__main__":
    main()
