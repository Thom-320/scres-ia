#!/usr/bin/env python3
"""Successive-halving audit of PPO convergence to a single static calendar.

The exact frontier is loaded only after every candidate in the requested stage
has frozen its deterministic policy.  Runs are burned development diagnostics.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from scripts.run_program_u_static_discovery_benchmark import (  # noqa: E402
    CalendarOracle,
    TAPES,
    cem_search,
    ppo_search,
)
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    extract_full_des_skeleton,
    full_action_calendars,
    simulate_full_des_frontier,
)


STAGES = {
    1: {"budget": 512, "seeds": tuple(range(20260801, 20260804)), "keep": 4},
    2: {"budget": 2048, "seeds": tuple(range(20260811, 20260816)), "keep": 2},
    3: {"budget": 8192, "seeds": tuple(range(20260821, 20260831)), "keep": 1},
}


def sobol_configurations() -> list[dict]:
    axes = (
        (1e-4, 3e-4, 1e-3),
        (64, 128, 256),
        (5, 10, 20),
        (0.1, 0.2),
        (0.0, 0.005, 0.01),
        (0.95, 1.0),
        ((64, 64), (128, 128)),
    )
    samples = qmc.Sobol(d=len(axes), scramble=True, seed=20260721).random_base2(m=4)
    configs = []
    for index, sample in enumerate(samples):
        selected = [axis[min(int(value * len(axis)), len(axis) - 1)] for axis, value in zip(axes, sample)]
        configs.append(
            {
                "config_id": f"sobol_{index:02d}",
                "learning_rate": selected[0],
                "n_steps": selected[1],
                "n_epochs": selected[2],
                "clip_range": selected[3],
                "ent_coef": selected[4],
                "gae_lambda": selected[5],
                "net_arch": list(selected[6]),
                "gamma": 1.0,
            }
        )
    if len({json.dumps(row, sort_keys=True) for row in configs}) != 16:
        raise AssertionError("Sobol discretization produced a duplicate configuration")
    return configs


def _rank(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["config_id"], []).append(row)
    summaries = []
    for config_id, members in grouped.items():
        summaries.append(
            {
                "config_id": config_id,
                "mean_frozen_policy_regret": float(np.mean([row["frozen_policy_simple_regret"] for row in members])),
                "median_frozen_policy_rank": float(np.median([row["frozen_policy_exact_rank"] for row in members])),
                "frozen_policy_optimum_fraction": float(np.mean([row["frozen_policy_exact_rank"] == 1 for row in members])),
                "mean_frozen_policy_entropy": float(np.mean([row["frozen_policy_mean_entropy"] for row in members])),
                "mean_best_seen_regret": float(np.mean([row["best_seen_simple_regret"] for row in members])),
                "seed_count": len(members),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            row["mean_frozen_policy_regret"],
            row["median_frozen_policy_rank"],
            row["mean_frozen_policy_entropy"],
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=STAGES, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage = STAGES[args.stage]
    configs = sobol_configurations()
    if args.stage > 1:
        if args.previous is None:
            raise ValueError("stage 2/3 requires --previous")
        previous = json.loads(args.previous.read_text())
        keep = STAGES[args.stage - 1]["keep"]
        promoted = {row["config_id"] for row in previous["ranking"][:keep]}
        configs = [row for row in configs if row["config_id"] in promoted]
    if args.max_configs is not None:
        configs = configs[: args.max_configs]

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
    frozen = []
    started = time.perf_counter()
    for config in configs:
        for seed in stage["seeds"]:
            oracle = CalendarOracle(skeletons, sched)
            result = ppo_search(
                oracle,
                budget=stage["budget"],
                seed=seed,
                learning_rate=config["learning_rate"],
                n_steps=config["n_steps"],
                n_epochs=config["n_epochs"],
                clip_range=config["clip_range"],
                ent_coef=config["ent_coef"],
                gae_lambda=config["gae_lambda"],
                net_arch=tuple(config["net_arch"]),
            )
            frozen.append((config, result))

    # Answer-key access starts here, after all requested policies are frozen.
    calendars = full_action_calendars()
    mean_scores = np.column_stack(
        [
            simulate_full_des_frontier(
                skeleton=skeletons[tape], scheduler=sched, calendars=calendars
            )["ret_visible"]
            for tape in TAPES
        ]
    ).mean(axis=1)
    optimum = float(np.max(mean_scores))
    rows = []
    for config, result in frozen:
        frozen_calendar = result.frozen_calendar or result.calendar
        best_index = int(sum(action * 4 ** (7 - period) for period, action in enumerate(result.calendar)))
        frozen_index = int(sum(action * 4 ** (7 - period) for period, action in enumerate(frozen_calendar)))
        rows.append(
            {
                "config_id": config["config_id"],
                "config": config,
                "optimizer_seed": result.seed,
                "best_seen_calendar": list(result.calendar),
                "best_seen_exact_rank": int(1 + np.sum(mean_scores > mean_scores[best_index] + 1e-15)),
                "best_seen_simple_regret": float(optimum - mean_scores[best_index]),
                "frozen_policy_calendar": list(frozen_calendar),
                "frozen_policy_exact_rank": int(1 + np.sum(mean_scores > mean_scores[frozen_index] + 1e-15)),
                "frozen_policy_simple_regret": float(optimum - mean_scores[frozen_index]),
                "frozen_policy_mean_entropy": result.frozen_mean_entropy,
                "proposals": result.proposed,
            }
        )
    ranking = _rank(rows)

    cem_rows = []
    if args.stage == 3:
        for seed in stage["seeds"]:
            result = cem_search(CalendarOracle(skeletons, sched), budget=stage["budget"], seed=seed)
            index = int(sum(action * 4 ** (7 - period) for period, action in enumerate(result.calendar)))
            cem_rows.append({"seed": seed, "regret": float(optimum - mean_scores[index]), "rank": int(1 + np.sum(mean_scores > mean_scores[index] + 1e-15))})
    verdict = "STAGE_COMPLETE_NOT_FINAL"
    if args.stage == 3 and ranking:
        winner = ranking[0]
        cem_mean = float(np.mean([row["regret"] for row in cem_rows]))
        verdict = (
            "PASS_PPO_STATIC_CONVERGENCE"
            if winner["frozen_policy_optimum_fraction"] >= 0.8
            and winner["mean_frozen_policy_regret"] <= cem_mean + 0.0001
            else "STOP_PPO_NOT_BEST_STATIC_SEARCHER"
        )
    payload = {
        "schema_version": "program_q2_static_ppo_successive_halving_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_status": "BURNED_DEVELOPMENT_NO_CLAIM",
        "stage": args.stage,
        "candidate_budget": stage["budget"],
        "optimizer_seeds": list(stage["seeds"]),
        "answer_key_read_during_search": False,
        "configurations_requested": len(configs),
        "elapsed_seconds": float(time.perf_counter() - started),
        "rows": rows,
        "ranking": ranking,
        "cem_reference": cem_rows,
        "verdict": verdict,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
