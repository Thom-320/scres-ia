#!/usr/bin/env python3
"""Successive-halving reward and hyperparameter tuning for RecurrentPPO Q2."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from scripts.run_program_q2_minimal_learner import evaluate, train  # noqa: E402
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    extract_full_des_skeleton,
    simulate_full_des_frontier,
)
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS  # noqa: E402


STAGES = {
    1: {"timesteps": 60_000, "seeds": (20260901, 20260902, 20260903), "keep": 4},
    2: {"timesteps": 100_000, "seeds": (20260911, 20260912, 20260913), "keep": 2},
    3: {"timesteps": 200_000, "seeds": (20260921, 20260922, 20260923, 20260924, 20260925), "keep": 1},
}
CALIBRATION_RESULT = ROOT / "results/program_o/ret_only_learner_v1/calibration_run/result.json"


def configurations() -> list[dict]:
    axes = (
        (1e-4, 3e-4, 1e-3),
        (5, 10, 20),
        (0.1, 0.2),
        (0.0, 0.01),
        (0.99, 1.0),
        (0.95, 1.0),
        (64, 128),
    )
    samples = qmc.Sobol(d=len(axes), scramble=True, seed=20260722).random_base2(m=4)[:12]
    rewards = ("raw_terminal", "standardized_terminal", "pbrs_terminal") * 4
    rows = []
    for index, (sample, reward) in enumerate(zip(samples, rewards)):
        values = [axis[min(int(value * len(axis)), len(axis) - 1)] for axis, value in zip(axes, sample)]
        rows.append(
            {
                "config_id": f"q2r_{index:02d}",
                "learning_rate": values[0],
                "n_epochs": values[1],
                "clip_range": values[2],
                "entropy": values[3],
                "gamma": values[4],
                "gae_lambda": values[5],
                "hidden_size": values[6],
                "reward_mode": reward,
            }
        )
    return rows


def structured_reference(tapes: tuple[int, ...]) -> dict[tuple[str, int], dict[str, float]]:
    frozen = json.loads(CALIBRATION_RESULT.read_text())
    if tuple(range(frozen["seed_range"][0], frozen["seed_range"][1] + 1)) != tapes:
        raise ValueError("requested evaluation tapes do not match the frozen calibration result")
    sched = scheduler()
    output = {}
    for cell in CONFIRMED_RET_CELLS:
        calendars = frozen["cell_summaries"][cell.cell_id]["best_classical_calendars"]
        for tape, calendar in zip(tapes, calendars):
            skeleton, _ = extract_full_des_skeleton(
                seed=tape,
                scheduler=sched,
                regime_persistence=cell.regime_persistence,
                dominant_share=cell.dominant_share,
                downstream_freight_physics_mode="fixed_clock_physical_v1",
            )
            metrics = simulate_full_des_frontier(
                skeleton=skeleton,
                scheduler=sched,
                calendars=np.asarray([calendar], dtype=np.uint8),
            )
            output[(cell.cell_id, tape)] = {
                "ret_visible": float(metrics["ret_visible"][0]),
                "worst_product_fill": float(metrics["worst_product_fill"][0]),
            }
    return output


def summarize(config_id: str, runs: list[dict], reference: dict, tapes: tuple[int, ...]) -> dict:
    runs = sorted(runs, key=lambda run: run["optimizer_seed"])
    seeds = [run["optimizer_seed"] for run in runs]
    cube = np.empty((len(seeds), len(CONFIRMED_RET_CELLS), len(tapes)), dtype=float)
    fill = np.empty_like(cube)
    for seed_index, run in enumerate(runs):
        lookup = {(row["cell"], row["tape"]): row for row in run["rows"]}
        for cell_index, cell in enumerate(CONFIRMED_RET_CELLS):
            for tape_index, tape in enumerate(tapes):
                row = lookup[(cell.cell_id, tape)]
                comparator = reference[(cell.cell_id, tape)]
                cube[seed_index, cell_index, tape_index] = row["ret_visible"] - comparator["ret_visible"]
                fill[seed_index, cell_index, tape_index] = row["worst_product_fill"] - comparator["worst_product_fill"]
    rng = np.random.default_rng(20260723)
    bootstrap = np.empty(2000)
    for index in range(len(bootstrap)):
        seed_sample = rng.integers(0, len(seeds), len(seeds))
        tape_sample = rng.integers(0, len(tapes), len(tapes))
        bootstrap[index] = cube[seed_sample][:, :, tape_sample].mean()
    return {
        "config_id": config_id,
        "pooled_mean_delta_vs_best_structured": float(cube.mean()),
        "pooled_lcb95_delta_vs_best_structured": float(np.quantile(bootstrap, 0.025)),
        "cell_mean_deltas": {
            cell.cell_id: float(cube[:, index, :].mean())
            for index, cell in enumerate(CONFIRMED_RET_CELLS)
        },
        "favorable_fraction": float(np.mean(cube > 0.0)),
        "positive_optimizer_seed_fraction": float(np.mean(cube.mean(axis=(1, 2)) > 0.0)),
        "worst_product_mean_delta": float(fill.mean()),
        "worst_product_minimum_cell_mean_delta": float(min(fill[:, index, :].mean() for index in range(len(CONFIRMED_RET_CELLS)))),
        "run_count": len(runs),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=STAGES, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--potential-json", type=Path, required=True)
    parser.add_argument("--training-seed-base", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-configs", type=int)
    args = parser.parse_args()
    potential = json.loads(args.potential_json.read_text())
    if len(potential.get("source_tapes", [])) < 12:
        raise ValueError("refusing tuning with a smoke-only PBRS potential")
    stage = STAGES[args.stage]
    configs = configurations()
    if args.stage > 1:
        if args.previous is None:
            raise ValueError("stage 2/3 requires --previous")
        prior = json.loads(args.previous.read_text())
        promoted = {row["config_id"] for row in prior["ranking"][: STAGES[args.stage - 1]["keep"]]}
        configs = [row for row in configs if row["config_id"] in promoted]
    if args.max_configs is not None:
        configs = configs[: args.max_configs]
    calibration = potential["reward_calibration"]
    tapes = tuple(range(7480001, 7480049))
    reference = structured_reference(tapes)
    all_runs: dict[str, list[dict]] = {config["config_id"]: [] for config in configs}
    required_episodes = int(np.ceil(stage["timesteps"] / 8)) + 2
    run_index = 0
    args.output.mkdir(parents=True, exist_ok=False)
    for config in configs:
        for optimizer_seed in stage["seeds"]:
            run_dir = args.output / config["config_id"] / str(optimizer_seed)
            run_dir.mkdir(parents=True)
            seed_start = args.training_seed_base + run_index * required_episodes
            namespace = SimpleNamespace(
                algorithm="recurrent_ppo",
                output=run_dir,
                total_timesteps=stage["timesteps"],
                optimizer_seed=optimizer_seed,
                training_seed_start=seed_start,
                training_seed_end=seed_start + required_episodes - 1,
                evaluation_tapes=tapes,
                reward_mode=config["reward_mode"],
                reward_mean=calibration["mean"],
                reward_standard_deviation=calibration["standard_deviation"],
                potential_json=args.potential_json,
                learning_rate=config["learning_rate"],
                n_steps=512,
                batch_size=64,
                n_epochs=config["n_epochs"],
                clip_range=config["clip_range"],
                entropy=config["entropy"],
                gamma=config["gamma"],
                gae_lambda=config["gae_lambda"],
                hidden_size=config["hidden_size"],
                n_quantiles=200,
                replay_buffer_size=100_000,
                learning_starts=1_000,
            )
            model, elapsed = train(namespace)
            rows = evaluate(model, namespace)
            run = {"optimizer_seed": optimizer_seed, "elapsed_seconds": elapsed, "rows": rows}
            (run_dir / "result.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
            all_runs[config["config_id"]].append(run)
            run_index += 1
    ranking = sorted(
        [summarize(config_id, runs, reference, tapes) for config_id, runs in all_runs.items()],
        key=lambda row: (
            -row["pooled_mean_delta_vs_best_structured"],
            -row["pooled_lcb95_delta_vs_best_structured"],
            -row["worst_product_mean_delta"],
        ),
    )
    payload = {
        "schema_version": "program_q2_recurrent_reward_tuning_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_status": "BURNED_DEVELOPMENT_NO_CLAIM",
        "stage": args.stage,
        "timesteps": stage["timesteps"],
        "optimizer_seeds": list(stage["seeds"]),
        "configs": configs,
        "training_seed_span": [args.training_seed_base, args.training_seed_base + run_index * required_episodes - 1],
        "ranking": ranking,
        "verdict": "STAGE_COMPLETE_NOT_FINAL" if args.stage < 3 else "TUNING_COMPLETE_SELECT_AT_MOST_ONE_LEARNER",
    }
    (args.output / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "configs"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
