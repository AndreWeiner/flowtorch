"""Bounded CPU Lorenz benchmark using FlowTorch's standalone predictors.

Run prepare, pilot, then run (requires a checked gate.json); plot separately.
Outputs include raw forecasts, checkpoints, tuning decisions and provenance.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import wasserstein_distance
import torch

from flowtorch.provenance import capture_environment
from flowtorch.rom import (
    EasyAttentionAutoregressor,
    GRUAutoregressor,
    LSTMAutoregressor,
    MLPAutoregressor,
)

MODELS = {
    "MLP": MLPAutoregressor,
    "LSTM": LSTMAutoregressor,
    "GRU": GRUAutoregressor,
    "Easy attention": EasyAttentionAutoregressor,
}
CONFIG = dict(
    dt=0.01,
    history=16,
    width=32,
    batch=64,
    train_count=12,
    validation_count=3,
    test_count=12,
    points=1500,
    test_steps=1000,
    long_count=4,
    long_steps=6000,
    long_discard=1000,
    warmup_steps=500,
    branch_steps=500,
    tune_steps=200,
    learning_rates=[0.001, 0.003],
    seeds=[11, 22, 33, 44, 55],
    validation_horizon=16,
    validation_windows=128,
    check_every=100,
    budget_seconds=1200,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rhs(t, xyz):
    x, y, z = xyz
    return [10 * (y - x), x * (28 - z) - y, x * y - 8 * z / 3]


def integrate(initial, count, dt=0.01, *, rtol=1.0e-10, atol=1.0e-12):
    time = np.arange(count) * dt
    solution = solve_ivp(
        rhs,
        (0.0, time[-1]),
        initial,
        t_eval=time,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not solution.success or solution.y.shape != (3, count):
        raise RuntimeError(solution.message)
    if not np.isfinite(solution.y).all():
        raise RuntimeError("nonfinite reference integration")
    return solution.y.T


def prepare(root, config):
    if (root / "data.npz").exists():
        raise ValueError("prepare requires a fresh output directory")
    start = perf_counter()
    rng = np.random.default_rng(20261104)
    burn = 1000
    arrays = {}
    for name, count, points in [
        ("train", config["train_count"], config["points"]),
        ("validation", config["validation_count"], config["points"]),
        ("test", config["test_count"], config["history"] + config["test_steps"]),
        ("long", config["long_count"], config["history"] + config["long_steps"]),
    ]:
        arrays[name] = np.stack(
            [
                integrate(rng.uniform(-10, 10, 3), burn + points)[burn:]
                for _ in range(count)
            ]
        )
    arrays["mean"] = arrays["train"].mean(axis=(0, 1))
    arrays["std"] = arrays["train"].std(axis=(0, 1))
    np.savez_compressed(root / "data.npz", **arrays)
    # Nonchaotic short-interval comparison, independent numerical integration path.
    initial = [1.0, 2.0, 3.0]
    reference = integrate(initial, 101)
    tighter = integrate(initial, 101, rtol=1.0e-12, atol=1.0e-14)
    alternative = solve_ivp(
        rhs,
        (0, 1),
        initial,
        t_eval=np.arange(101) * 0.01,
        method="RK45",
        rtol=1.0e-11,
        atol=1.0e-13,
    ).y.T
    difference = float(np.max(np.abs(reference - tighter)))
    independent = float(np.max(np.abs(reference - alternative)))
    assert difference < 1.0e-6 and independent < 1.0e-6
    write_json(root / "config.json", config)
    write_json(
        root / "reference_checks.json",
        {
            "short_interval_max_abs_tighter": difference,
            "short_interval_max_abs_RK45": independent,
            "finite_all_splits": True,
            "normalization": "training only, population std",
            "generation_seconds": perf_counter() - start,
            "data_sha256": hashlib.sha256((root / "data.npz").read_bytes()).hexdigest(),
            "generation_seed": 20261104,
            "burn_in_time": 10.0,
        },
    )
    print("Prepared independent trajectories and validated reference", flush=True)


def load(root):
    config = json.loads((root / "config.json").read_text())
    with np.load(root / "data.npz") as saved:
        arrays = {key: saved[key] for key in saved.files}
    normalized = {
        key: torch.tensor(
            (arrays[key] - arrays["mean"]) / arrays["std"], dtype=torch.float32
        )
        for key in ("train", "validation", "test", "long")
    }
    return config, arrays, normalized


def make_model(name, config):
    kwargs = dict(state_dim=3, history_length=config["history"])
    if name == "Easy attention":
        kwargs.update(model_dim=config["width"], num_heads=4)
    else:
        kwargs.update(hidden_dim=config["width"])
    return MODELS[name](**kwargs).cpu()


def windows(data, history, horizon, count, generator):
    series = torch.randint(data.shape[0], (count,), generator=generator)
    starts = torch.randint(
        data.shape[1] - history - horizon + 1, (count,), generator=generator
    )
    offsets = torch.arange(history + horizon)
    selected = data[series[:, None], starts[:, None] + offsets]
    return selected[:, :history], selected[:, history:]


def train(
    model,
    optimizer,
    train_data,
    validation_data,
    config,
    steps,
    schedule,
    seed,
    *,
    deadline,
    incumbent=None,
):
    generator = torch.Generator().manual_seed(seed)
    val_generator = torch.Generator().manual_seed(700)
    val_history, val_future = windows(
        validation_data,
        config["history"],
        config["validation_horizon"],
        config["validation_windows"],
        val_generator,
    )
    with torch.no_grad():
        initial_score = float(model.rollout_loss(val_history, val_future))
    best = (
        initial_score,
        deepcopy(model.state_dict()),
        deepcopy(optimizer.state_dict()),
        0,
    )
    if incumbent is not None:
        assert abs(initial_score - incumbent) < 1.0e-7
    log = []
    start = perf_counter()
    for step in range(1, steps + 1):
        if perf_counter() > deadline:
            raise TimeoutError(
                "benchmark wall-clock budget exceeded; retained earlier artifacts"
            )
        horizon = schedule(step, steps)
        history, future = windows(
            train_data, config["history"], horizon, config["batch"], generator
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = model.rollout_loss(history, future)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % config["check_every"] == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                score = float(model.rollout_loss(val_history, val_future))
            log.append(
                dict(
                    step=step,
                    horizon=horizon,
                    train_loss=float(loss.detach()),
                    validation_16_step_mse=score,
                )
            )
            if np.isfinite(score) and score < best[0]:
                best = (
                    score,
                    deepcopy(model.state_dict()),
                    deepcopy(optimizer.state_dict()),
                    step,
                )
    elapsed = perf_counter() - start
    model.load_state_dict(best[1])
    optimizer.load_state_dict(best[2])
    model.eval()
    return dict(
        log=log, seconds=elapsed, validation_loss=best[0], selected_step=best[3]
    )


def evaluate(model, data, arrays, config, *, long=True):
    model.eval()
    history = data["test"][:, : config["history"]]
    with torch.no_grad():
        # Warm up, then record three complete CPU rollout wall times on identical inputs.
        model.rollout(history, 20)
        timings = []
        for _ in range(3):
            start = perf_counter()
            short = model.rollout(history, config["test_steps"])
            timings.append(perf_counter() - start)
        target = data["test"][:, config["history"] :]
        error = (short.double() - target.double()).square().mean(dim=-1).sqrt().numpy()
        finite_short = np.isfinite(error).all(axis=1)
        crossings = (error > 0.5) | ~np.isfinite(error)
        any_crossing = crossings.any(axis=1)
        horizons = np.where(
            any_crossing,
            (crossings.argmax(axis=1) + 1) * config["dt"],
            config["test_steps"] * config["dt"],
        )
        output = dict(short=short.numpy(), error=error, horizons=horizons)
        stats = dict(
            short_rmse_first_100=(
                float(np.sqrt(np.mean(error[:, :100] ** 2)))
                if np.isfinite(error[:, :100]).all()
                else None
            ),
            horizon_mean=float(horizons.mean()),
            horizons=horizons.tolist(),
            horizon_right_censored=int((~any_crossing).sum()),
            nonfinite_short_trajectories=int((~finite_short).sum()),
            rollout_seconds=timings,
        )
        if long:
            long_history = data["long"][:, : config["history"]]
            predicted = model.rollout(long_history, config["long_steps"]).numpy()
            output["long"] = predicted
            samples = (
                predicted[:, config["long_discard"] :].reshape(-1, 3).astype(np.float64)
            )
            reference = (
                data["long"][:, config["history"] + config["long_discard"] :]
                .numpy()
                .reshape(-1, 3)
            )
            finite = np.isfinite(samples).all()
            stats.update(
                nonfinite_long_values=int((~np.isfinite(samples)).sum()),
                out_of_reference_range_fraction=float(
                    (np.abs(samples) > 10).any(axis=1).mean()
                ),
                mean_error=(
                    float(np.linalg.norm(samples.mean(0) - reference.mean(0)))
                    if finite
                    else None
                ),
                std_error=(
                    float(np.linalg.norm(samples.std(0) - reference.std(0)))
                    if finite
                    else None
                ),
                wasserstein_mean=(
                    float(
                        np.mean(
                            [
                                wasserstein_distance(reference[:, j], samples[:, j])
                                for j in range(3)
                            ]
                        )
                    )
                    if finite
                    else None
                ),
            )
    return stats, output


def pilot(root):
    start = perf_counter()
    config, arrays, data = load(root)
    deadline = start + config["budget_seconds"]
    results = []
    for name in MODELS:
        for lr in config["learning_rates"]:
            torch.manual_seed(101)
            model = make_model(name, config)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            training = train(
                model,
                optimizer,
                data["train"],
                data["validation"],
                config,
                config["tune_steps"],
                lambda s, n: 1,
                101,
                deadline=deadline,
            )
            results.append(
                dict(
                    model=name,
                    lr=lr,
                    parameters=sum(p.numel() for p in model.parameters()),
                    **training,
                )
            )
            print(
                name, lr, training["validation_loss"], training["seconds"], flush=True
            )
    selected = {
        name: min(
            (r for r in results if r["model"] == name),
            key=lambda r: r["validation_loss"],
        )["lr"]
        for name in MODELS
    }
    # Exercise real multi-step training and the complete evaluation path.
    torch.manual_seed(102)
    model = make_model("LSTM", config)
    optimizer = torch.optim.Adam(model.parameters(), lr=selected["LSTM"])
    multi = train(
        model,
        optimizer,
        data["train"],
        data["validation"],
        config,
        40,
        lambda s, n: 4,
        102,
        deadline=deadline,
    )
    stats, forecasts = evaluate(model, data, arrays, config)
    np.savez_compressed(root / "pilot_forecasts.npz", **forecasts)
    write_json(
        root / "pilot.json",
        dict(
            tuning=results,
            selected_lr=selected,
            multistep_training=multi,
            evaluation=stats,
            outer_seconds=perf_counter() - start,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            environment=capture_environment(),
            processor=platform.processor(),
            timing_scope="training includes validation/checkpoint copies; rollout excludes normalization and loading",
        ),
    )
    print("Pilot complete", flush=True)


def run(root):
    # Validate the explicit evidence gate without requiring an installed skill at runtime.
    gate = json.loads((root / "gate.json").read_text())
    if any(v["status"] != "pass" for v in gate["checks"].values()):
        raise ValueError("pilot gate checks have not passed")
    for relative, digest in gate["fingerprints"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise ValueError("gate fingerprint changed: " + relative)
    config, arrays, data = load(root)
    budget = gate["proposed_run"]["budget_seconds"]
    if (
        not 0
        < gate["proposed_run"]["estimated_seconds"]
        <= budget
        <= config["budget_seconds"]
    ):
        raise ValueError("invalid gate time budget")
    deadline = perf_counter() + budget
    pilot_record = json.loads((root / "pilot.json").read_text())
    # Randomized architecture order is fixed independently of test outcomes.
    order = list(MODELS)
    np.random.default_rng(812).shuffle(order)
    for seed in config["seeds"]:
        for name in order:
            family = name.replace(" ", "_") + f"_seed{seed}"
            if (root / (family + "_one_step.json")).exists():
                raise ValueError("output already exists; use a fresh run directory")
            torch.manual_seed(seed)
            model = make_model(name, config)
            lr = pilot_record["selected_lr"][name]
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            warm = train(
                model,
                optimizer,
                data["train"],
                data["validation"],
                config,
                config["warmup_steps"],
                lambda s, n: 1,
                seed,
                deadline=deadline,
            )
            weights, optim_state = deepcopy(model.state_dict()), deepcopy(
                optimizer.state_dict()
            )
            for strategy in ("one_step", "curriculum"):
                model.load_state_dict(weights)
                optimizer.load_state_dict(deepcopy(optim_state))
                schedule = (
                    (lambda s, n: 1)
                    if strategy == "one_step"
                    else (lambda s, n: 2 if s <= n // 2 else 4)
                )
                training = train(
                    model,
                    optimizer,
                    data["train"],
                    data["validation"],
                    config,
                    config["branch_steps"],
                    schedule,
                    seed + 1000,
                    deadline=deadline,
                    incumbent=warm["validation_loss"],
                )
                stats, forecasts = evaluate(model, data, arrays, config)
                trial = family + "_" + strategy
                np.savez_compressed(root / (trial + ".npz"), **forecasts)
                torch.save(
                    dict(
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        config=config,
                        model_name=name,
                    ),
                    root / (trial + ".pt"),
                )
                write_json(
                    root / (trial + ".json"),
                    dict(
                        model=name,
                        strategy=strategy,
                        seed=seed,
                        lr=lr,
                        parameters=sum(p.numel() for p in model.parameters()),
                        warmup=warm,
                        training=training,
                        metrics=stats,
                    ),
                )
                print(
                    trial,
                    "horizon",
                    round(stats["horizon_mean"], 3),
                    "val",
                    training["validation_loss"],
                    flush=True,
                )
    write_json(
        root / "complete.json",
        dict(
            environment=capture_environment(),
            elapsed_seconds=budget - (deadline - perf_counter()),
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            trials=40,
            model_order=order,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "pilot", "run"])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    {"prepare": lambda r: prepare(r, CONFIG), "pilot": pilot, "run": run}[args.phase](
        args.output
    )


if __name__ == "__main__":
    main()
