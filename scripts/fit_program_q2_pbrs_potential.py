#!/usr/bin/env python3
"""Fit a causal Q21 potential on already-burned Program Q training tapes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    extract_full_des_skeleton,
    simulate_full_des_frontier,
)
from supply_chain.program_o_ret_env import (  # noqa: E402
    CONFIRMED_RET_CELLS,
    normalized_state_rich_observation,
)
from supply_chain.program_o_state_rich import StateRichConfiguration, state_rich_calendar  # noqa: E402


DEFAULT_TAPES = tuple(range(748100001, 748100013))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tapes", default=",".join(map(str, DEFAULT_TAPES)))
    parser.add_argument("--calendars-per-tape", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tapes = tuple(map(int, args.tapes.split(",")))
    rng = np.random.default_rng(args.seed)
    sched = scheduler()
    observations = []
    targets = []
    groups = []
    episode_returns = []
    for cell_index, cell in enumerate(CONFIRMED_RET_CELLS):
        for tape in tapes:
            skeleton, _ = extract_full_des_skeleton(
                seed=tape,
                scheduler=sched,
                regime_persistence=cell.regime_persistence,
                dominant_share=cell.dominant_share,
                downstream_freight_physics_mode="fixed_clock_physical_v1",
            )
            calendars = rng.integers(0, 4, size=(args.calendars_per_tape, 8), dtype=np.uint8)
            outcomes = simulate_full_des_frontier(
                skeleton=skeleton, scheduler=sched, calendars=calendars
            )["ret_visible"]
            episode_returns.extend(map(float, outcomes))
            group = f"{cell.cell_id}:{tape}"
            for calendar, outcome in zip(calendars, outcomes):
                _, decisions = state_rich_calendar(
                    skeleton=skeleton.as_dict(),
                    scheduler=sched,
                    config=StateRichConfiguration("belief_mpc", 3),
                    regime_persistence=0.75,
                    dominant_share=0.90,
                    action_overrides=tuple(map(int, calendar)),
                )
                for decision in decisions:
                    observations.append(normalized_state_rich_observation(decision.observation))
                    targets.append(float(outcome))
                    groups.append(group)
    x = np.asarray(observations, dtype=float)
    y = np.asarray(targets, dtype=float)
    group_values = np.asarray(groups)
    unique_groups = np.unique(group_values)
    fold_scores = []
    for fold in range(min(5, len(unique_groups))):
        held = set(unique_groups[fold::5])
        test = np.asarray([value in held for value in group_values])
        model = Ridge(alpha=args.ridge_alpha).fit(x[~test], y[~test])
        fold_scores.append(float(r2_score(y[test], model.predict(x[test]))))
    final = Ridge(alpha=args.ridge_alpha).fit(x, y)
    returns = np.asarray(episode_returns)
    payload = {
        "schema_version": "program_q2_q21_pbrs_potential_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_status": "BURNED_DEVELOPMENT_NO_CLAIM",
        "source_tapes": list(tapes),
        "source_role": "already_burned_program_q_training",
        "calendars_per_tape_cell": args.calendars_per_tape,
        "training_rows": int(len(x)),
        "weights": list(map(float, final.coef_)),
        "bias": float(final.intercept_),
        "ridge_alpha": args.ridge_alpha,
        "grouped_cross_validated_r2": fold_scores,
        "reward_calibration": {
            "mean": float(np.mean(returns)),
            "standard_deviation": float(np.std(returns, ddof=1)),
            "episode_count": int(len(returns)),
        },
        "contract": {
            "causal_q21_only": True,
            "terminal_potential_forced_to_zero_by_wrapper": True,
            "gamma": 1.0,
            "fit_selection_tapes_disjoint_from_q_calibration_and_confirmation": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "weights"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
