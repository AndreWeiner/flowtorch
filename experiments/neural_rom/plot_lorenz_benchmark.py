"""Regenerate Lorenz figures and animation from saved raw benchmark arrays.

Default text rendering uses LaTeX. --no-tex is an explicit opt-out for systems
without a working LaTeX backend. No training is performed by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import numpy as np
from scipy.signal import welch

NAMES = ["MLP", "LSTM", "GRU", "Easy attention"]
COLORS = dict(zip(NAMES, ["#0072B2", "#D55E00", "#009E73", "#AA4499"]))
STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "savefig.dpi": 200,
    "figure.dpi": 100,
}


def key(name, seed, strategy):
    return name.replace(" ", "_") + f"_seed{seed}_{strategy}"


def save(fig, output, name):
    fig.savefig(output / (name + ".png"))
    fig.savefig(output / (name + ".pdf"))
    plt.close(fig)


def write_tables(summary, output):
    """Generate documentation tables from the same raw-derived summary."""
    lines = [
        ".. list-table:: Prediction and long-rollout results",
        "   :header-rows: 1",
        "   :widths: 18 18 10 18 18 18",
        "",
        "   * - Model",
        "     - Training",
        "     - Parameters",
        "     - Horizon: median [range]",
        "     - W1: finite-run median",
        "     - Nonfinite long runs",
    ]
    for row in summary:
        h, w = row["horizon"], row["wasserstein"]
        strategy = "One-step" if row["strategy"] == "one_step" else "Curriculum"
        lines += [
            f"   * - {row['model']}",
            f"     - {strategy}",
            f"     - {row['parameters']:,}",
            f"     - {h['median']:.2f} [{h['minimum']:.2f}, {h['maximum']:.2f}]",
            f"     - {w['median']:.3f} ({w['valid_seeds']}/5 finite)",
            f"     - {row['failed_long_runs']}/5",
        ]
    (output / "results_table.rst.txt").write_text("\n".join(lines) + "\n")
    lines = [
        ".. list-table:: Measured CPU costs (median across five seeds)",
        "   :header-rows: 1",
        "",
        "   * - Model",
        "     - One-step training (s)",
        "     - Curriculum training (s)",
        "     - One-step rollout (microseconds/state)",
    ]
    for name in NAMES:
        a, b = [
            next(
                row
                for row in summary
                if row["model"] == name and row["strategy"] == strategy
            )
            for strategy in ("one_step", "curriculum")
        ]
        lines += [
            f"   * - {name}",
            f"     - {a['train_seconds']['median']:.2f}",
            f"     - {b['train_seconds']['median']:.2f}",
            f"     - {1000*a['rollout_ms_per_state']['median']:.2f}",
        ]
    (output / "timing_table.rst.txt").write_text("\n".join(lines) + "\n")


def render(root, output, use_tex=True, animate=True):
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / "config.json").read_text())
    seeds = config["seeds"]
    with np.load(root / "data.npz") as data:
        mean, scale = data["mean"], data["std"]
        reference = data["test"][:, config["history"] :]
        long_reference = data["long"][:, config["history"] + config["long_discard"] :]
    records, arrays = {}, {}
    for name in NAMES:
        for strategy in ("one_step", "curriculum"):
            for seed in seeds:
                trial = key(name, seed, strategy)
                records[trial] = json.loads((root / (trial + ".json")).read_text())
                with np.load(root / (trial + ".npz")) as data:
                    arrays[trial] = {k: data[k] for k in data.files}
    # Aggregation is over seed-level summaries, not thousands of correlated time samples.
    summary = []
    for name in NAMES:
        for strategy in ("one_step", "curriculum"):
            rows = [records[key(name, seed, strategy)] for seed in seeds]
            result = dict(
                model=name, strategy=strategy, parameters=rows[0]["parameters"]
            )
            metrics = {
                "horizon": [r["metrics"]["horizon_mean"] for r in rows],
                "rmse_1_time_unit": [
                    r["metrics"]["short_rmse_first_100"] for r in rows
                ],
                "wasserstein": [r["metrics"]["wasserstein_mean"] for r in rows],
                "train_seconds": [
                    r["warmup"]["seconds"] + r["training"]["seconds"] for r in rows
                ],
                "rollout_ms_per_state": [
                    np.median(r["metrics"]["rollout_seconds"])
                    * 1000
                    / (config["test_steps"] * config["test_count"])
                    for r in rows
                ],
            }
            for label, values in metrics.items():
                good = np.array([v for v in values if v is not None])
                result[label] = (
                    dict(
                        median=float(np.median(good)),
                        minimum=float(good.min()),
                        maximum=float(good.max()),
                        seed_values=values,
                        valid_seeds=int(len(good)),
                    )
                    if len(good)
                    else None
                )
            result["failed_long_runs"] = sum(
                r["metrics"]["nonfinite_long_values"] > 0 for r in rows
            )
            result["long_outside_10_std_max_fraction"] = max(
                r["metrics"]["out_of_reference_range_fraction"] for r in rows
            )
            summary.append(result)
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    write_tables(summary, output)
    with plt.rc_context({**STYLE, "text.usetex": use_tex}):
        # Representative real export verifies the selected text renderer before the full set.
        fig, ax = plt.subplots(figsize=(4, 2.5), layout="constrained")
        ax.plot([0, 1], [0, 1])
        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(r"$\mathrm{RMSE}$")
        save(fig, output, "renderer_check")
        time = np.arange(1, config["test_steps"] + 1) * config["dt"]
        for bounded in (False, True):
            fig, axes = plt.subplots(
                1, 2, figsize=(10, 3.7), layout="constrained", sharey=True
            )
            for ax, strategy, title in zip(
                axes,
                ("one_step", "curriculum"),
                ("One-step training", "Increasing-horizon training"),
            ):
                for name in NAMES:
                    curves = np.stack(
                        [
                            arrays[key(name, seed, strategy)]["error"].mean(axis=0)
                            for seed in seeds
                        ]
                    )
                    # Failure stays a failure, never a silently excluded successful seed.
                    curves = np.where(np.isfinite(curves), curves, np.inf)
                    median = np.median(curves, axis=0)
                    lower, upper = curves.min(axis=0), curves.max(axis=0)
                    ax.plot(
                        time,
                        np.where(np.isfinite(median), median, np.nan),
                        color=COLORS[name],
                        label=name,
                    )
                    if bounded:
                        lower, upper = np.minimum(lower, 10), np.minimum(upper, 10)
                        for curve in curves:
                            outside = np.flatnonzero(curve > 10)
                            if len(outside):
                                ax.plot(
                                    time[outside[0]], 8, "^", color=COLORS[name], ms=5
                                )
                    else:
                        upper = np.where(np.isfinite(upper), upper, np.nan)
                    ax.fill_between(time, lower, upper, color=COLORS[name], alpha=0.12)
                ax.axhline(0.5, color="0.35", lw=1, ls="--", label="Horizon threshold")
                ax.set(
                    xlabel="Forecast time (dimensionless)", yscale="log", title=title
                )
                ax.grid(alpha=0.15)
                if bounded:
                    ax.set_ylim(1.0e-4, 10)
                    ax.text(
                        0.02,
                        0.12,
                        "Triangles: seed error exceeds view",
                        transform=ax.transAxes,
                        va="top",
                        fontsize=9,
                    )
            axes[0].set_ylabel("Mean standardized state error")
            axes[1].legend(loc="lower right")
            save(
                fig,
                output,
                "forecast_errors" if bounded else "forecast_errors_full_range",
            )
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
        for j, (metric, title) in enumerate(
            [
                ("horizon", "Useful prediction horizon"),
                ("wasserstein", "Long-rollout marginal distance"),
            ]
        ):
            for i, name in enumerate(NAMES):
                for offset, strategy, marker in [
                    (-0.12, "one_step", "o"),
                    (0.12, "curriculum", "D"),
                ]:
                    row = next(
                        r
                        for r in summary
                        if r["model"] == name and r["strategy"] == strategy
                    )
                    values = row[metric]["seed_values"]
                    finite_values = [v for v in values if v is not None]
                    if len(finite_values) < len(values):
                        axes[j].scatter(
                            [i + offset],
                            [0.97],
                            transform=axes[j].get_xaxis_transform(),
                            marker="x",
                            color=COLORS[name],
                            s=40,
                        )
                    axes[j].scatter(
                        np.full(len(finite_values), i + offset),
                        finite_values,
                        color=COLORS[name],
                        marker=marker,
                        s=30,
                        alpha=0.65,
                        facecolors="none" if strategy == "one_step" else COLORS[name],
                    )
                    axes[j].plot(
                        [i + offset - 0.07, i + offset + 0.07],
                        [row[metric]["median"]] * 2,
                        color="black",
                        lw=2.0,
                    )
            axes[j].set_xticks(range(4), NAMES, rotation=15)
            axes[j].set_title(title)
            axes[j].grid(axis="y", alpha=0.15)
            if metric == "wasserstein":
                axes[j].set_yscale("log")
                axes[j].set_ylim(0.008, 30)
                axes[j].text(
                    0.98,
                    0.03,
                    "Top crosses: nonfinite runs",
                    transform=axes[j].transAxes,
                    va="bottom",
                    ha="right",
                    fontsize=9,
                )
            else:
                axes[j].set_ylim(bottom=0)
        axes[0].set_ylabel("First threshold crossing time")
        axes[1].set_ylabel("Mean normalized Wasserstein distance")
        axes[0].plot([], [], "o", color="0.3", fillstyle="none", label="One-step")
        axes[0].plot([], [], "D", color="0.3", label="Curriculum")
        axes[0].legend()
        save(fig, output, "seed_comparison")
        # Predeclared representative: first seed and first held-out trajectory.
        seed = seeds[0]
        fig, axes = plt.subplots(
            2, 2, figsize=(10, 6), layout="constrained", sharex=True, sharey=True
        )
        for ax, name in zip(axes.flat, NAMES):
            prediction = (
                arrays[key(name, seed, "curriculum")]["short"][0] * scale + mean
            )
            ax.plot(
                time[:500],
                reference[0, :500, 0],
                color="0.25",
                label="Reference",
                lw=1.4,
            )
            ax.plot(
                time[:500], prediction[:500, 0], color=COLORS[name], label=name, lw=1.2
            )
            ax.set_title(name)
            ax.set_xlabel("Forecast time")
            ax.set_ylabel(r"$x$")
            ax.grid(alpha=0.15)
        axes[0, 0].legend()
        save(fig, output, "short_trajectories")
        fig, axes = plt.subplots(
            2, 2, figsize=(10, 6.3), layout="constrained", sharex=True, sharey=True
        )
        for ax, name in zip(axes.flat, NAMES):
            prediction = (
                arrays[key(name, seed, "curriculum")]["long"][
                    0, config["long_discard"] :
                ]
                * scale
                + mean
            )
            ax.plot(
                long_reference[0, :, 0],
                long_reference[0, :, 2],
                color="0.65",
                lw=0.45,
                alpha=0.6,
                label="Reference",
            )
            ax.plot(
                prediction[:, 0],
                prediction[:, 2],
                color=COLORS[name],
                lw=0.5,
                alpha=0.7,
                label=name,
            )
            ax.set(xlabel=r"$x$", ylabel=r"$z$", title=name)
            # Common physical extent; explicitly count any excluded samples.
            outside = (
                (np.abs(prediction[:, 0]) > 30)
                | (prediction[:, 2] < -10)
                | (prediction[:, 2] > 70)
            ).sum()
            if outside:
                ax.text(
                    0.03,
                    0.95,
                    f"{outside} points outside view",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=9,
                )
            ax.set_xlim(-30, 30)
            ax.set_ylim(-10, 70)
        axes[0, 0].legend(loc="upper right")
        save(fig, output, "long_attractors")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
        bins = np.linspace(-30, 30, 61)
        refx = long_reference[:, :, 0]
        axes[0].hist(
            refx.flatten(),
            bins=bins,
            density=True,
            histtype="step",
            color="0.2",
            label="Reference",
            lw=2.0,
        )
        frequency, power = welch(refx, fs=1 / config["dt"], nperseg=1024, axis=-1)
        axes[1].semilogy(
            frequency, power.mean(0), color="0.2", label="Reference", lw=2.0
        )
        for name in NAMES:
            prediction = (
                arrays[key(name, seed, "curriculum")]["long"][
                    :, config["long_discard"] :, 0
                ]
                * scale[0]
                + mean[0]
            )
            axes[0].hist(
                prediction.flatten(),
                bins=bins,
                density=True,
                histtype="step",
                color=COLORS[name],
                label=name,
            )
            frequency, power = welch(
                prediction, fs=1 / config["dt"], nperseg=1024, axis=-1
            )
            axes[1].semilogy(frequency, power.mean(0), color=COLORS[name], label=name)
        axes[0].set(
            xlabel=r"$x$",
            ylabel="Probability density",
            title="Long-rollout distribution",
        )
        axes[1].set(
            xlabel="Frequency (inverse time)",
            ylabel=r"PSD of $x$",
            title="Long-rollout spectrum",
            xlim=(0, 10),
        )
        axes[0].legend()
        axes[1].grid(alpha=0.15)
        save(fig, output, "long_statistics")
        if animate:
            fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), layout="constrained")
            artists = []
            for ax, name in zip(axes.flat, NAMES):
                (reference_line,) = ax.plot(
                    [], [], color="0.4", lw=1.2, label="Reference"
                )
                (model_line,) = ax.plot([], [], color=COLORS[name], lw=1.5, label=name)
                (reference_dot,) = ax.plot([], [], "o", color="0.4", ms=4)
                (model_dot,) = ax.plot([], [], "o", color=COLORS[name], ms=4)
                status = ax.text(
                    0.03, 0.97, "", transform=ax.transAxes, va="top", fontsize=10
                )
                ax.set(
                    xlim=(-30, 30),
                    ylim=(-10, 70),
                    xlabel=r"$x$",
                    ylabel=r"$z$",
                    title=name,
                )
                ax.legend(loc="lower right")
                artists.append(
                    (reference_line, model_line, reference_dot, model_dot, status)
                )
            title = fig.suptitle("", fontsize=12)

            def update(frame):
                stop = frame + 1
                first = max(0, stop - 150)
                title.set_text(
                    f"Lorenz forecasts: t = {stop*config['dt']:.2f} | seed {seed}, curriculum"
                )
                for name, (rl, ml, rd, md, status) in zip(NAMES, artists):
                    predicted = (
                        arrays[key(name, seed, "curriculum")]["short"][0] * scale + mean
                    )
                    true = reference[0]
                    rl.set_data(true[first:stop, 0], true[first:stop, 2])
                    ml.set_data(predicted[first:stop, 0], predicted[first:stop, 2])
                    rd.set_data([true[frame, 0]], [true[frame, 2]])
                    md.set_data([predicted[frame, 0]], [predicted[frame, 2]])
                    error = arrays[key(name, seed, "curriculum")]["error"][0, frame]
                    text = f"Standardized error: {error:.2f}"
                    if (
                        abs(predicted[frame, 0]) > 30
                        or not -10 <= predicted[frame, 2] <= 70
                    ):
                        text += " | outside view"
                    status.set_text(text)
                return []

            animation = FuncAnimation(
                fig,
                update,
                frames=range(0, config["test_steps"], 5),
                interval=50,
                blit=False,
            )
            animation.save(
                output / "lorenz_rollout.mp4", writer=FFMpegWriter(fps=20, bitrate=1800)
            )
            animation.save(
                output / "lorenz_rollout.gif", writer=PillowWriter(fps=20), dpi=80
            )
            plt.close(fig)
    (root / "figure_settings.json").write_text(
        json.dumps(
            dict(
                use_tex=use_tex,
                preset="documentation: 10 inch width, 11 point body, 200 dpi PNG and vector PDF",
                animation="first seed/first test trajectory; curriculum; 1.5-time-unit trail; every fifth sample at 20 fps",
                uncertainty="min-max over five seed-level mean-error curves, not confidence intervals; bounded error figure clips bands at 10 and marks first exceedance; full-range companion ends at nonfinite outputs",
                spectrum="Welch, fs=100, segment length 1024, default Hann, mean over four trajectories; frequency view 0..10",
                distribution="60 bins over x=-30..30; all-trial out-of-range diagnostic in summary",
            ),
            indent=2,
        )
        + "\n"
    )
    print("Rendered figures and summary", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--no-tex", action="store_true")
    parser.add_argument("--no-animation", action="store_true")
    args = parser.parse_args()
    render(
        args.input, args.output, use_tex=not args.no_tex, animate=not args.no_animation
    )


if __name__ == "__main__":
    main()
