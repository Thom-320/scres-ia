#!/usr/bin/env python3
"""Burned-development U0 bakeoff with the exhaustive matrix hidden until freeze."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    extract_full_des_skeleton,
    full_action_calendars,
    simulate_full_des_frontier,
)
from supply_chain.program_u_static_search import (  # noqa: E402
    BudgetedPanelObjective,
    autoregressive_policy_gradient_search,
    bayesian_optimization_search,
    cma_es_search,
    cross_entropy_search,
    ppo_calendar_search,
    random_search,
)


BURNED_TAPES = (94800001, 94800002, 94800003)
METHODS = {
    "random": random_search,
    "cem": cross_entropy_search,
    "policy_gradient": autoregressive_policy_gradient_search,
    "cmaes": cma_es_search,
    "bayesian": bayesian_optimization_search,
    "ppo": ppo_calendar_search,
}


def scheduler() -> dict[str, list[str]]:
    contract = json.loads(
        (ROOT / "contracts/program_o_full_des_hpi_translation_v1.json").read_text()
    )
    key = contract["action"]["primary_scheduler"]
    return contract["action"]["within_week_schedulers"][key]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=600)
    parser.add_argument("--seeds", default="20260720,20260721,20260722")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/program_u/static_bakeoff_dev_v1/result.json",
    )
    args = parser.parse_args()
    selected = [name.strip() for name in args.methods.split(",") if name.strip()]
    unknown = set(selected) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    seeds = [int(value) for value in args.seeds.split(",")]
    sched = scheduler()
    skeletons = {
        tape: extract_full_des_skeleton(
            seed=tape,
            scheduler=sched,
            regime_persistence=0.75,
            dominant_share=0.90,
            downstream_freight_physics_mode="fixed_clock_physical_v1",
        )[0]
        for tape in BURNED_TAPES
    }

    def evaluator(calendar: tuple[int, ...], tape: int) -> float:
        panel = simulate_full_des_frontier(
            skeleton=skeletons[int(tape)],
            scheduler=sched,
            calendars=np.asarray([calendar], dtype=np.uint8),
        )
        return float(panel["ret_visible"][0])

    rows = []
    frozen_candidates: set[tuple[int, ...]] = set()
    for method_name in selected:
        for seed in seeds:
            objective = BudgetedPanelObjective(
                evaluator=evaluator,
                tape_ids=BURNED_TAPES,
                call_budget=int(args.budget),
            )
            start = time.perf_counter()
            result = METHODS[method_name](objective, seed=seed)
            elapsed = time.perf_counter() - start
            frozen_candidates.add(result.best_calendar)
            row = asdict(result)
            row["wall_seconds"] = elapsed
            rows.append(row)

    # The answer key is opened only after every method/seed candidate is frozen.
    calendars = full_action_calendars()
    matrix = np.column_stack(
        [
            simulate_full_des_frontier(
                skeleton=skeletons[tape], scheduler=sched, calendars=calendars
            )["ret_visible"]
            for tape in BURNED_TAPES
        ]
    )
    means = matrix.mean(axis=1)
    optimum = float(np.max(means))
    for row in rows:
        score = float(row["best_score"])
        row["simple_regret"] = optimum - score
        row["exact_rank"] = int(1 + np.sum(means > score + 1e-15))
        row["best_so_far_auc"] = float(
            np.trapezoid(
                [point["best_score"] for point in row["trace"]],
                [point["calendar_tape_calls"] for point in row["trace"]],
            )
        )
        for fraction in (0.05, 0.01, 0.005):
            threshold = optimum - fraction * max(abs(optimum), 1e-12)
            calls = next(
                (
                    int(point["calendar_tape_calls"])
                    for point in row["trace"]
                    if float(point["best_score"]) >= threshold
                ),
                None,
            )
            row[f"calls_to_{fraction:g}_relative_gap"] = calls

    payload = {
        "schema_version": "program_u_static_bakeoff_dev_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_status": "BURNED_DEVELOPMENT_NOT_SCIENTIFIC_EVIDENCE",
        "training_may_read_answer_key": False,
        "answer_key_opened_after_all_candidates_frozen": True,
        "tapes": list(BURNED_TAPES),
        "call_budget_per_method_seed": int(args.budget),
        "search_space_size": int(len(calendars)),
        "exact_optimum_mean_ret": optimum,
        "frozen_candidate_count": len(frozen_candidates),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
