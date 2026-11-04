"""Aggregate context/data sweeps with nested dataset/network replication.

Regenerates figures, tables and audit checks from raw saved predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

from plot_lorenz_benchmark import STYLE, COLORS, NAMES, save
from lorenz_context_data import cells


def seed_groups(rows, label):
    """Average network seeds within each independent training-data realization."""
    return np.array(
        [
            np.mean([row[label] for row in rows if row["replicate"] == rep])
            for rep in sorted({row["replicate"] for row in rows})
        ]
    )


def select(rows, group, model, **settings):
    return [
        row
        for row in rows
        if row["cell"]["group"] == group
        and row["model"] == model
        and all(row["cell"][key] == value for key, value in settings.items())
    ]


def summarize(root):
    config = json.loads((root / "config.json").read_text())
    records = []
    with np.load(root / "held_out.npz") as held:
        target = (
            (
                (held["test"][:, max(config["contexts"]) :] - np.array(config["mean"]))
                / np.array(config["scale"])
            )
            .astype(np.float32)
            .astype(np.float64)
        )
    with np.load(root / "held_out.npz") as held:
        physical = held["test"]
    guard = max(config["contexts"])
    switching = np.diff(physical[:, guard:, 0] > 0, axis=1).any(axis=1)
    persistence = (
        (
            (physical[:, guard - 1] - np.array(config["mean"]))
            / np.array(config["scale"])
        )
        .astype(np.float32)
        .astype(np.float64)
    )
    persistence_error = np.sqrt(np.mean((persistence[:, None] - target) ** 2, axis=-1))
    hit = persistence_error > 0.5
    persistence_horizon = np.array(
        [
            (
                (np.flatnonzero(x)[0] + 1) * config["dt"]
                if x.any()
                else config["test_steps"] * config["dt"]
            )
            for x in hit
        ]
    )
    baseline = dict(
        method="hold newest observed state constant; no training or future input",
        all_horizon=float(persistence_horizon.mean()),
        switching_horizon=(
            float(persistence_horizon[switching].mean()) if switching.any() else None
        ),
        switching_test_cases=int(switching.sum()),
        non_switching_test_cases=int((~switching).sum()),
        switching_case_indices=np.flatnonzero(switching).tolist(),
        horizons=persistence_horizon.tolist(),
        status="Added as a post-hoc diagnostic after inspecting test dynamics; no training, checkpoint, or hyperparameter changes",
    )
    (root / "persistence_baseline.json").write_text(
        json.dumps(baseline, indent=2) + "\n"
    )
    audits = []
    for path in sorted((root / "trials").glob("*.json")):
        row = json.loads(path.read_text())
        records.append(row)
        with np.load(path.with_suffix(".npz")) as raw:
            # Recompute error/horizon independently of Torch and the study's fit method.
            error = np.sqrt(
                np.mean((raw["prediction"].astype(np.float64) - target) ** 2, axis=-1)
            )
            np.testing.assert_allclose(
                error, raw["error"], rtol=1.0e-12, atol=1.0e-12, equal_nan=True
            )
            crossing = (error > 0.5) | ~np.isfinite(error)
            horizon = np.array(
                [
                    (
                        (np.flatnonzero(x)[0] + 1) * config["dt"]
                        if x.any()
                        else config["test_steps"] * config["dt"]
                    )
                    for x in crossing
                ]
            )
            np.testing.assert_equal(horizon, raw["horizons"])
            np.testing.assert_allclose(
                horizon.mean(), row["horizon_mean"], atol=1.0e-12
            )
        row["horizon_switching"] = (
            float(horizon[switching].mean()) if switching.any() else None
        )
        row["horizon_non_switching"] = (
            float(horizon[~switching].mean()) if (~switching).any() else None
        )
        assert 0 < row["sampled_unique_origins"] <= row["eligible_windows"]
        assert row["updates"] == config["updates"]
        audits.append(path.stem)
    expected = (
        len(cells(config))
        * len(NAMES)
        * len(config["data_replicates"])
        * len(config["network_seeds"])
    )
    assert len(records) == expected
    summary = []
    for row in records:
        label = (
            row["cell"]["group"],
            row["cell"]["context"],
            row["cell"]["budget"],
            row["cell"]["count"],
            row["model"],
        )
        if any(tuple(x["key"]) == label for x in summary):
            continue
        subset = select(
            records,
            label[0],
            label[4],
            context=label[1],
            budget=label[2],
            count=label[3],
        )
        item = dict(
            key=label,
            parameters=row["parameters"],
            eligible_windows=row["eligible_windows"],
            runs=len(subset),
            nonfinite_test_runs=sum(x["nonfinite_trajectories"] > 0 for x in subset),
        )
        for metric in [
            "horizon_mean",
            *(["horizon_switching"] if switching.any() else []),
            "validation_mse",
            "training_seconds",
            "sampled_unique_origins",
        ]:
            values = seed_groups(subset, metric)
            item[metric] = dict(
                mean=float(values.mean()),
                minimum=float(values.min()),
                maximum=float(values.max()),
                dataset_replicate_means=values.tolist(),
            )
        summary.append(item)
    write_json = root / "summary.json"
    write_json.write_text(json.dumps(summary, indent=2) + "\n")
    comparisons = []
    for name in NAMES:
        for budget in config["budgets"]:
            few = select(records, "data", name, budget=budget, count=1)
            many = select(records, "data", name, budget=budget, count=16)
            # Positive means many-short improves validation; negative means worse.
            a, b = seed_groups(few, "validation_mse"), seed_groups(
                many, "validation_mse"
            )
            changes = 100 * (a - b) / a
            comparisons.append(
                dict(
                    model=name,
                    budget=budget,
                    validation_relative_improvement_percent_by_data_replicate=changes.tolist(),
                    mean_improvement_percent=float(changes.mean()),
                    improving_data_replicates=int((changes > 0).sum()),
                    horizon_change_by_data_replicate=(
                        seed_groups(many, "horizon_mean")
                        - seed_groups(few, "horizon_mean")
                    ).tolist(),
                )
            )
    (root / "many_short_vs_few_long.json").write_text(
        json.dumps(comparisons, indent=2) + "\n"
    )
    (root / "analysis_checks.json").write_text(
        json.dumps(
            dict(
                trials=len(audits),
                horizons_recomputed=len(audits) * config["test_count"],
                errors="NumPy vs Torch agreement to 1e-12, including nonfinite outputs",
                grouping="network seeds averaged within dataset replicate; 3 independent data-replicate means per configuration",
                nonfinite_test_runs=sum(
                    r["nonfinite_trajectories"] > 0 for r in records
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    return config, records, summary, comparisons, baseline


def render(root, output, use_tex=True):
    output.mkdir(parents=True, exist_ok=True)
    config, rows, summary, comparisons, baseline = summarize(root)
    with plt.rc_context({**STYLE, "text.usetex": use_tex}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
        for name in NAMES:
            for ax, metric in zip(axes, ["horizon_mean", "validation_mse"]):
                values = np.stack(
                    [
                        seed_groups(
                            select(rows, "context", name, context=length), metric
                        )
                        for length in config["contexts"]
                    ]
                )
                ax.plot(
                    config["contexts"],
                    values.mean(1),
                    "-o",
                    color=COLORS[name],
                    label=name,
                    ms=4,
                )
                ax.fill_between(
                    config["contexts"],
                    values.min(1),
                    values.max(1),
                    color=COLORS[name],
                    alpha=0.12,
                )
                ax.set_xscale("log", base=2)
                ax.set_xticks(config["contexts"], [str(x) for x in config["contexts"]])
                ax.set_xlabel("Past states in context")
                ax.grid(alpha=0.15)
        axes[0].set(
            title="Context length: held-out forecasts",
            ylabel="Mean useful horizon",
            ylim=(0, 10),
        )
        axes[1].set(
            title="Context length: validation",
            ylabel="One-time-unit validation MSE",
            yscale="log",
        )
        axes[0].axhline(
            baseline["all_horizon"], color="0.35", ls="--", label="Persistence"
        )
        axes[0].legend()
        save(fig, output, "context_sensitivity")
        palette = ["#0072B2", "#D55E00", "#009E73", "#AA4499"]
        count_colors = {
            count: palette[i % len(palette)]
            for i, count in enumerate(config["trajectory_counts"])
        }
        for metric, filename, label in [
            ("horizon_mean", "data_test_horizons", "Mean useful test horizon"),
            ("validation_mse", "data_validation", "One-time-unit validation MSE"),
            *(
                [
                    (
                        "horizon_switching",
                        "data_switching_horizons",
                        "Mean horizon: switching tests",
                    )
                ]
                if baseline["switching_test_cases"]
                else []
            ),
        ]:
            fig, axes = plt.subplots(
                2,
                2,
                figsize=(10, 6.2),
                layout="constrained",
                sharex=True,
                sharey=(metric != "validation_mse"),
            )
            for ax, name in zip(axes.flat, NAMES):
                for count, marker in zip(config["trajectory_counts"], ["o", "s", "D"]):
                    values = np.stack(
                        [
                            seed_groups(
                                select(rows, "data", name, budget=budget, count=count),
                                metric,
                            )
                            for budget in config["budgets"]
                        ]
                    )
                    ax.plot(
                        config["budgets"],
                        values.mean(1),
                        "-" + marker,
                        color=count_colors[count],
                        label=(
                            f"{count} trajectory"
                            if count == 1
                            else f"{count} trajectories"
                        ),
                        ms=4,
                    )
                    ax.fill_between(
                        config["budgets"],
                        values.min(1),
                        values.max(1),
                        color=count_colors[count],
                        alpha=0.12,
                    )
                ax.set(
                    xscale="log",
                    title=name,
                    xlabel="Total retained training states",
                    ylabel=label,
                )
                ax.set_xticks(config["budgets"], [f"{x:,}" for x in config["budgets"]])
                ax.xaxis.set_minor_locator(NullLocator())
                ax.grid(alpha=0.15)
                if metric != "validation_mse":
                    reference_horizon = (
                        baseline["switching_horizon"]
                        if metric == "horizon_switching"
                        else baseline["all_horizon"]
                    )
                    ax.axhline(
                        reference_horizon, color="0.35", ls="--", label="Persistence"
                    )
                if metric == "validation_mse":
                    ax.set_yscale("log")
                else:
                    ax.set_ylim(0, 5 if metric == "horizon_switching" else 10)
            axes[0, 0].legend()
            save(fig, output, filename)
        fig, axes = plt.subplots(
            1, 3, figsize=(10, 3.4), layout="constrained", sharex=True, sharey=True
        )
        with np.load(root / "training_pool_0.npz") as pool:
            for ax, count in zip(axes, config["trajectory_counts"]):
                points = min(config["budgets"]) // count
                for i in range(count):
                    states = pool[f"source_{i}"][:points]
                    ax.plot(
                        states[:, 0], states[:, 2], color="#0072B2", alpha=0.6, lw=0.65
                    )
                ax.set(
                    xlim=(-25, 25),
                    ylim=(0, 55),
                    xlabel=r"$x$",
                    title=(
                        f"{count} trajectory, {points} states"
                        if count == 1
                        else f"{count} trajectories, {points} states each"
                    ),
                )
            axes[0].set_ylabel(r"$z$")
        save(fig, output, "training_coverage")
    # A compact table directly addresses the proposed many-short advantage.
    lines = [
        ".. list-table:: Many short vs few long: validation changes",
        "   :header-rows: 1",
        "",
        "   * - Model",
        "     - Retained states",
        "     - Mean validation improvement (percent)",
        "     - Data realizations favoring 16 trajectories",
    ]
    for row in comparisons:
        lines += [
            f"   * - {row['model']}",
            f"     - {row['budget']:,}",
            f"     - {row['mean_improvement_percent']:+.1f}",
            f"     - {row['improving_data_replicates']}/3",
        ]
    (output / "trajectory_comparison.rst.txt").write_text("\n".join(lines) + "\n")
    lines = [
        ".. list-table:: Mean useful horizon versus context length",
        "   :header-rows: 1",
        "",
        "   * - Model",
    ] + [f"     - L={length}" for length in config["contexts"]]
    for name in NAMES:
        lines.append(f"   * - {name}")
        for length in config["contexts"]:
            vals = seed_groups(
                select(rows, "context", name, context=length), "horizon_mean"
            )
            lines.append(
                f"     - {vals.mean():.2f} [{vals.min():.2f}, {vals.max():.2f}]"
            )
    (output / "context_table.rst.txt").write_text("\n".join(lines) + "\n")
    (root / "figure_settings.json").write_text(
        json.dumps(
            dict(
                use_tex=use_tex,
                style="Reuses original documentation preset: width 10 inches, 11 point text, PNG/PDF",
                bands="minimum and maximum of 3 dataset-replicate means, each averaging 3 network seeds",
                coverage="preselected data replicate 0, smallest retained-state budget; no validation/test points plotted",
            ),
            indent=2,
        )
        + "\n"
    )
    print("Audited and plotted", len(rows), "trials", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--no-tex", action="store_true")
    args = parser.parse_args()
    render(args.input, args.output, not args.no_tex)


if __name__ == "__main__":
    main()
