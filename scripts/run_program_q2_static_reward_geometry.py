#!/usr/bin/env python3
"""Factorial audit of static PPO reward geometry on burned development tapes.

The PPO architecture is fixed while canonical versus affine-standardized
terminal ReT and SB3 advantage normalization are varied. Identical optimizer
seeds are repeated at increasing budgets. The exhaustive answer key is loaded
only after every requested policy is frozen.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from scripts.run_program_u_static_discovery_benchmark import (  # noqa: E402
    CalendarOracle,
    TAPES,
    ppo_search,
)
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    extract_full_des_skeleton,
    full_action_calendars,
    simulate_full_des_frontier,
)


CALIBRATION = ROOT / "results/program_q2/reward_audit_v1/pbrs_potential_v1.json"
DEFAULT_BUDGETS = (512, 2_048, 8_192)
DEFAULT_SEEDS = (20261001, 20261002, 20261003)


def arms(mean: float, standard_deviation: float) -> tuple[dict, ...]:
    """Return the frozen two-by-two reward/advantage-normalization panel."""
    return (
        {"arm": "raw_advnorm", "reward_center": 0.0, "reward_scale": 1.0, "normalize_advantage": True},
        {"arm": "standardized_advnorm", "reward_center": mean, "reward_scale": standard_deviation, "normalize_advantage": True},
        {"arm": "raw_no_advnorm", "reward_center": 0.0, "reward_scale": 1.0, "normalize_advantage": False},
        {"arm": "standardized_no_advnorm", "reward_center": mean, "reward_scale": standard_deviation, "normalize_advantage": False},
    )


def _run(payload: tuple[dict, int, int]) -> tuple[dict, int, object]:
    arm, budget, seed = payload
    import torch

    torch.set_num_threads(1)
    sched = scheduler()
    skeletons = {
        tape: extract_full_des_skeleton(
            seed=tape,
            scheduler=sched,
            regime_persistence=0.75,
            dominant_share=0.90,
            downstream_freight_physics_mode="fixed_clock_physical_v1",
        )[0]
        for tape in TAPES
    }
    result = ppo_search(
        CalendarOracle(skeletons, sched),
        budget=budget,
        seed=seed,
        learning_rate=1e-3,
        n_steps=128,
        n_epochs=20,
        clip_range=0.1,
        ent_coef=0.005,
        gae_lambda=1.0,
        net_arch=(128, 128),
        reward_center=arm["reward_center"],
        reward_scale=arm["reward_scale"],
        normalize_advantage=arm["normalize_advantage"],
    )
    return arm, budget, result


def _calendar_index(calendar: tuple[int, ...]) -> int:
    return int(sum(action * 4 ** (7 - period) for period, action in enumerate(calendar)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    budgets = tuple(map(int, args.budgets.split(",")))
    seeds = tuple(map(int, args.seeds.split(",")))
    if not budgets or min(budgets) <= 0 or not seeds or args.jobs <= 0:
        raise ValueError("budgets, seeds, and jobs must be positive")
    calibration = json.loads(args.calibration.read_text())
    frozen = calibration["reward_calibration"]
    reward_arms = arms(float(frozen["mean"]), float(frozen["standard_deviation"]))
    tasks = [(arm, budget, seed) for arm in reward_arms for budget in budgets for seed in seeds]

    started = time.perf_counter()
    if args.jobs == 1:
        results = [_run(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(tasks))) as executor:
            results = list(executor.map(_run, tasks))

    # Exact-frontier access begins only after all policies are frozen.
    sched = scheduler()
    skeletons = {
        tape: extract_full_des_skeleton(
            seed=tape,
            scheduler=sched,
            regime_persistence=0.75,
            dominant_share=0.90,
            downstream_freight_physics_mode="fixed_clock_physical_v1",
        )[0]
        for tape in TAPES
    }
    calendars = full_action_calendars()
    mean_scores = np.column_stack([
        simulate_full_des_frontier(skeleton=skeletons[tape], scheduler=sched, calendars=calendars)["ret_visible"]
        for tape in TAPES
    ]).mean(axis=1)
    optimum = float(mean_scores.max())
    rows = []
    for arm, budget, result in results:
        best_index = _calendar_index(result.calendar)
        final = result.frozen_calendar or result.calendar
        final_index = _calendar_index(final)
        rows.append({
            "arm": arm["arm"],
            "budget": budget,
            "optimizer_seed": result.seed,
            "reward_center": arm["reward_center"],
            "reward_scale": arm["reward_scale"],
            "normalize_advantage": arm["normalize_advantage"],
            "best_seen_calendar": list(result.calendar),
            "best_seen_rank": int(1 + np.sum(mean_scores > mean_scores[best_index] + 1e-15)),
            "best_seen_regret": float(optimum - mean_scores[best_index]),
            "frozen_calendar": list(final),
            "frozen_rank": int(1 + np.sum(mean_scores > mean_scores[final_index] + 1e-15)),
            "frozen_regret": float(optimum - mean_scores[final_index]),
            "frozen_entropy": result.frozen_mean_entropy,
        })

    summaries = []
    for arm in reward_arms:
        for budget in budgets:
            members = [row for row in rows if row["arm"] == arm["arm"] and row["budget"] == budget]
            summaries.append({
                "arm": arm["arm"],
                "budget": budget,
                "mean_frozen_regret": float(np.mean([row["frozen_regret"] for row in members])),
                "median_frozen_rank": float(np.median([row["frozen_rank"] for row in members])),
                "mean_best_seen_regret": float(np.mean([row["best_seen_regret"] for row in members])),
                "mean_frozen_entropy": float(np.mean([row["frozen_entropy"] for row in members])),
                "optimum_fraction": float(np.mean([row["frozen_rank"] == 1 for row in members])),
            })
    final_budget = max(budgets)
    final_rows = sorted(
        [row for row in summaries if row["budget"] == final_budget],
        key=lambda row: (row["mean_frozen_regret"], row["median_frozen_rank"]),
    )
    raw = next(row for row in final_rows if row["arm"] == "raw_advnorm")
    standardized = next(row for row in final_rows if row["arm"] == "standardized_advnorm")
    improvement = raw["mean_frozen_regret"] - standardized["mean_frozen_regret"]
    if improvement >= 0.001 and standardized["median_frozen_rank"] < raw["median_frozen_rank"]:
        verdict = "REWARD_STANDARDIZATION_MATERIALLY_IMPROVES_RETENTION"
    elif abs(improvement) < 0.0005:
        verdict = "REWARD_STANDARDIZATION_EFFECT_NEGLIGIBLE"
    elif improvement > 0.0:
        verdict = "REWARD_STANDARDIZATION_SMALL_POSITIVE_EFFECT"
    else:
        verdict = "REWARD_STANDARDIZATION_DOES_NOT_IMPROVE_RETENTION"
    payload = {
        "schema_version": "program_q2_static_reward_geometry_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_status": "BURNED_DEVELOPMENT_NO_CLAIM",
        "tapes": list(TAPES),
        "budgets": list(budgets),
        "optimizer_seeds": list(seeds),
        "calibration_source": str(args.calibration.relative_to(ROOT)),
        "reward_calibration": frozen,
        "answer_key_read_during_training": False,
        "ppo": {"learning_rate": 1e-3, "n_steps": 128, "batch_size": 64, "n_epochs": 20, "clip_range": 0.1, "entropy_coefficient": 0.005, "gae_lambda": 1.0, "gamma": 1.0, "net_arch": [128, 128]},
        "rows": rows,
        "summaries": summaries,
        "final_budget_ranking": final_rows,
        "raw_minus_standardized_regret": float(improvement),
        "elapsed_seconds": float(time.perf_counter() - started),
        "verdict": verdict,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
