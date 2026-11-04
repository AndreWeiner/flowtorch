"""Context/data sensitivity study, reusing the original Lorenz numerical model.

Phases: prepare independent/nested data; pilot real paths; run reviewed gate;
plot saved results. Original benchmark artifacts are never modified.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np
import torch

from lorenz_benchmark import integrate, make_model, MODELS, write_json
from flowtorch.provenance import capture_environment

CONFIG = dict(
    dt=0.01,
    contexts=[1, 4, 16, 64],
    budgets=[2048, 8192, 32768],
    trajectory_counts=[1, 4, 16],
    data_replicates=[0, 1, 2],
    network_seeds=[11, 22, 33],
    width=32,
    batch=64,
    base_context=16,
    base_budget=8192,
    base_count=4,
    updates=1000,
    validation_steps=100,
    validation_windows=32,
    check_every=250,
    test_steps=1000,
    test_count=12,
    lr=0.003,
    burn=1000,
    mean=[0.0, 0.0, 27.0],
    scale=[10.0, 10.0, 10.0],
    budget_seconds=1200,
)


def cells(config):
    """Two linked, non-factorial sweeps. Groups use different training guard sizes."""
    result = []
    for length in config["contexts"]:
        result.append(
            dict(
                group="context",
                context=length,
                budget=config["base_budget"],
                count=config["base_count"],
                guard=max(config["contexts"]),
            )
        )
    for budget in config["budgets"]:
        for count in config["trajectory_counts"]:
            result.append(
                dict(
                    group="data",
                    context=config["base_context"],
                    budget=budget,
                    count=count,
                    guard=config["base_context"],
                )
            )
    return result


def eligible_windows(cell):
    """Reserve enough future for any training horizon, always at most four."""
    return cell["budget"] - cell["count"] * (cell["guard"] + 4 - 1)


def cell_id(cell):
    return f"{cell['group']}_L{cell['context']}_N{cell['budget']}_K{cell['count']}"


def normalize(array, config):
    return torch.tensor(
        (array - np.asarray(config["mean"])) / np.asarray(config["scale"]),
        dtype=torch.float32,
    )


def prepare(root, config):
    if (root / "config.json").exists():
        raise ValueError("fresh output directory required")
    start = perf_counter()
    maximum = max(config["budgets"])
    max_count = max(config["trajectory_counts"])
    info = []
    for replica in config["data_replicates"]:
        rng = np.random.default_rng(8100 + replica)
        pool = {}
        initial = []
        for i in range(max_count):
            # Source 0 is needed longest; later sources need only shorter prefixes.
            length = max(maximum // k for k in config["trajectory_counts"] if i < k)
            point = rng.uniform(-10, 10, 3)
            initial.append(point.tolist())
            pool[f"source_{i}"] = integrate(point, config["burn"] + length)[
                config["burn"] :
            ]
        np.savez_compressed(root / f"training_pool_{replica}.npz", **pool)
        info.append(
            dict(
                replicate=replica,
                initial_states=initial,
                lengths={name: len(value) for name, value in pool.items()},
            )
        )
    rng = np.random.default_rng(29019)
    reference = {}
    for name, count, length in [
        ("validation", 4, 2000),
        ("test", config["test_count"], max(config["contexts"]) + config["test_steps"]),
    ]:
        reference[name] = np.stack(
            [
                integrate(rng.uniform(-10, 10, 3), config["burn"] + length)[
                    config["burn"] :
                ]
                for _ in range(count)
            ]
        )
    # Common forecast origins: changing L never changes the target or forecast start.
    reference["validation_series"] = rng.integers(
        0, 4, size=config["validation_windows"]
    )
    reference["validation_origins"] = rng.integers(
        max(config["contexts"]),
        reference["validation"].shape[1] - config["validation_steps"] + 1,
        size=config["validation_windows"],
    )
    np.savez_compressed(root / "held_out.npz", **reference)
    write_json(root / "config.json", config)
    write_json(
        root / "data_recipe.json",
        dict(
            training=info,
            generation_seconds=perf_counter() - start,
            fixed_normalization="shift [0,0,27], scales [10,10,10], specified before fitting; no estimated data statistics",
            nested="for each replicate and trajectory count, larger budgets extend the same source trajectories",
            scope="raw retained state counts; spin-up and discarded prefix/target guard overhead are reported separately",
        ),
    )
    print("Prepared nested training pools and independent held-out data", flush=True)


def training_data(root, cell, replica, config):
    points = cell["budget"] // cell["count"]
    with np.load(root / f"training_pool_{replica}.npz") as source:
        array = np.stack([source[f"source_{i}"][:points] for i in range(cell["count"])])
    if array.shape != (cell["count"], points, 3) or eligible_windows(cell) < 1:
        raise ValueError("invalid data split or insufficient context/targets")
    return normalize(array, config)


def held_out(root, context, config):
    with np.load(root / "held_out.npz") as data:
        validation = normalize(data["validation"], config)
        test = normalize(data["test"], config)
        series = torch.tensor(data["validation_series"])
        origins = torch.tensor(data["validation_origins"])
    history = validation[series[:, None], origins[:, None] + torch.arange(-context, 0)]
    target = validation[
        series[:, None], origins[:, None] + torch.arange(config["validation_steps"])
    ]
    guard = max(config["contexts"])
    return history, target, test[:, guard - context : guard], test[:, guard:]


def sample_windows(
    data, context, horizon, batch, generator, guard, *, return_indices=False
):
    if not 1 <= context <= guard or not 1 <= horizon <= 4:
        raise ValueError("context must fit guard, and horizon must be 1..4")
    series = torch.randint(data.shape[0], (batch,), generator=generator)
    origins = torch.randint(guard, data.shape[1] - 4 + 1, (batch,), generator=generator)
    history = data[series[:, None], origins[:, None] + torch.arange(-context, 0)]
    future = data[series[:, None], origins[:, None] + torch.arange(horizon)]
    return (history, future, series, origins) if return_indices else (history, future)


def coverage(data):
    flat = data.double().numpy().reshape(-1, 3)
    # Undo fixed normalization for geometry diagnostics, never for training selection.
    x = flat[:, 0] * 10
    z = flat[:, 2] * 10 + 27
    bins, _, _ = np.histogram2d(x, z, bins=(20, 22), range=((-25, 25), (0, 55)))
    x_series = data[:, :, 0].double().numpy()
    lag1 = float(np.corrcoef(x_series[:, :-1].ravel(), x_series[:, 1:].ravel())[0, 1])
    return dict(
        occupied_xz_bins=int((bins > 0).sum()),
        total_xz_bins=440,
        positive_x_fraction=float((x > 0).mean()),
        lag1_correlation=lag1,
    )


def fit(
    root, cell, replica, seed, name, config, *, deadline, steps=None, model_factory=None
):
    """One fixed scheduled curriculum, common validation duration, best checkpoint."""
    steps = config["updates"] if steps is None else steps
    torch.manual_seed(seed)
    data = training_data(root, cell, replica, config)
    val_h, val_y, test_h, test_y = held_out(root, cell["context"], config)
    factory = make_model if model_factory is None else model_factory
    model = factory(name, dict(history=cell["context"], width=config["width"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    generator = torch.Generator().manual_seed(10000 + seed)
    with torch.no_grad():
        score = float(model.rollout_loss(val_h, val_y))
    best = (score, deepcopy(model.state_dict()), 0)
    trace = []
    seen_origins = torch.zeros(data.shape[:2], dtype=torch.bool)
    start = perf_counter()
    for step in range(1, steps + 1):
        if perf_counter() > deadline:
            raise TimeoutError(
                "study deadline exceeded; completed trial artifacts retained"
            )
        horizon = 1 if step <= steps // 2 else (2 if step <= 3 * steps // 4 else 4)
        h, y, sampled_series, sampled_origins = sample_windows(
            data,
            cell["context"],
            horizon,
            config["batch"],
            generator,
            cell["guard"],
            return_indices=True,
        )
        seen_origins[sampled_series, sampled_origins] = True
        optimizer.zero_grad(set_to_none=True)
        loss = model.rollout_loss(h, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite optimization loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % config["check_every"] == 0 or step == steps:
            with torch.no_grad():
                score = float(model.rollout_loss(val_h, val_y))
            trace.append(
                dict(
                    step=step,
                    horizon=horizon,
                    train_mse=float(loss.detach()),
                    validation_mse=score if np.isfinite(score) else None,
                )
            )
            if np.isfinite(score) and score < best[0]:
                best = (score, deepcopy(model.state_dict()), step)
    training_seconds = perf_counter() - start
    model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad():
        model.rollout(test_h, 10)
        eval_start = perf_counter()
        predicted = model.rollout(test_h, config["test_steps"])
        evaluation_seconds = perf_counter() - eval_start
    errors = (predicted.double() - test_y.double()).square().mean(-1).sqrt().numpy()
    crossing = (errors > 0.5) | ~np.isfinite(errors)
    horizons = np.where(
        crossing.any(1),
        (crossing.argmax(1) + 1) * config["dt"],
        config["test_steps"] * config["dt"],
    )
    record = dict(
        cell=cell,
        replicate=replica,
        seed=seed,
        model=name,
        parameters=sum(p.numel() for p in model.parameters()),
        updates=steps,
        training_seconds=training_seconds,
        evaluation_seconds=evaluation_seconds,
        selected_step=best[2],
        validation_mse=best[0],
        trace=trace,
        horizon_mean=float(horizons.mean()),
        horizons=horizons.tolist(),
        censored=int((~crossing.any(1)).sum()),
        rmse_1_time_unit=(
            float(np.sqrt(np.mean(errors[:, :100] ** 2)))
            if np.isfinite(errors[:, :100]).all()
            else None
        ),
        nonfinite_trajectories=int((~np.isfinite(errors).all(1)).sum()),
        eligible_windows=eligible_windows(cell),
        sampled_unique_origins=int(seen_origins.sum()),
        coverage=coverage(data),
    )
    return (
        record,
        dict(prediction=predicted.numpy(), error=errors, horizons=horizons),
        model,
    )


def trial_id(cell, replica, seed, model):
    return f'{cell_id(cell)}_d{replica}_s{seed}_{model.replace(" ","_")}'


def pilot(root, config):
    start = perf_counter()
    rows = []
    # Exercise extremes and the complete scheduled-training/selection/evaluation paths.
    for length in (
        min(config["contexts"]),
        config["base_context"],
        max(config["contexts"]),
    ):
        cell = dict(
            group="context",
            context=length,
            budget=config["base_budget"],
            count=config["base_count"],
            guard=max(config["contexts"]),
        )
        for name in MODELS:
            record, arrays, _ = fit(
                root, cell, 0, 101, name, config, deadline=start + 1200, steps=100
            )
            rows.append(record)
            np.savez_compressed(
                root / (trial_id(cell, 0, 101, name) + "_pilot.npz"), **arrays
            )
            print(
                "Pilot",
                length,
                name,
                record["training_seconds"],
                record["evaluation_seconds"],
                flush=True,
            )
    write_json(
        root / "pilot.json",
        dict(
            trials=rows,
            outer_seconds=perf_counter() - start,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            environment=capture_environment(),
            purpose="path/resource checks only; no test-driven changes to settings",
        ),
    )


def run(root, config):
    gate = json.loads((root / "gate.json").read_text())
    for check in gate["checks"].values():
        if check["status"] != "pass":
            raise ValueError("unpassed evidence gate")
    for path, digest in gate["fingerprints"].items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != digest:
            raise ValueError("changed gate input: " + path)
    budget = gate["proposed_run"]["budget_seconds"]
    if (
        not 0
        < gate["proposed_run"]["estimated_seconds"]
        <= budget
        <= config["budget_seconds"]
    ):
        raise ValueError("invalid study budget")
    start = perf_counter()
    deadline = start + budget
    trials = [
        (cell, replica, seed, name)
        for cell in cells(config)
        for replica in config["data_replicates"]
        for seed in config["network_seeds"]
        for name in MODELS
    ]
    np.random.default_rng(771).shuffle(trials)
    directory = root / "trials"
    directory.mkdir(exist_ok=True)
    if list(directory.iterdir()):
        raise ValueError("fresh trial output required")
    for i, (cell, replica, seed, name) in enumerate(trials):
        identifier = trial_id(cell, replica, seed, name)
        try:
            row, arrays, model = fit(
                root, cell, replica, seed, name, config, deadline=deadline
            )
        except Exception as error:
            write_json(
                root / "failure.json",
                dict(trial=identifier, error=str(error), completed=i),
            )
            raise
        np.savez_compressed(directory / (identifier + ".npz"), **arrays)
        write_json(directory / (identifier + ".json"), row)
        # Preserve all selected weights, without an unnecessary optimizer-state copy.
        torch.save(model.state_dict(), directory / (identifier + ".pt"))
        print(
            i + 1,
            "/",
            len(trials),
            identifier,
            "horizon",
            round(row["horizon_mean"], 3),
            flush=True,
        )
    write_json(
        root / "complete.json",
        dict(
            trials=len(trials),
            seconds=perf_counter() - start,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            environment=capture_environment(),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "pilot", "run"])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.phase == "prepare":
        prepare(args.output, CONFIG)
    else:
        config = json.loads((args.output / "config.json").read_text())
        {"pilot": pilot, "run": run}[args.phase](args.output, config)


if __name__ == "__main__":
    main()
