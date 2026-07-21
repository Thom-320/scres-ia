#!/usr/bin/env python3
"""Cross-fitted one-step Q/MPC repair selector on burned common-prefix branches."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.program_o_full_des_transducer import extract_full_des_skeleton, simulate_full_des_frontier  # noqa: E402
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS  # noqa: E402
from supply_chain.program_o_state_rich import StateRichConfiguration, state_rich_calendar  # noqa: E402
from supply_chain.program_t_joint_belief import weekly_product_counts  # noqa: E402

Q_RESULT = ROOT / "results/program_q/confirmation_v1_20260718/artifacts/confirmation/evaluation/result.json"
T0_RESULT = ROOT / "results/program_t/t0_ret_transducer_v1/result.json"


def observation_features(observation, history: tuple[int, ...]) -> np.ndarray:
    padded = np.full(7, -1.0); padded[: len(history)] = history
    return np.asarray([
        *observation.on_hand,
        *observation.locked_pipeline,
        *observation.backlog_quantity,
        *observation.backlog_orders,
        *observation.max_backlog_age,
        *observation.in_flight_quantity,
        observation.belief_c,
        observation.predicted_share_c,
        -1 if observation.previous_action is None else observation.previous_action,
        observation.remaining_decisions,
        *padded,
    ], dtype=float)


def cluster_lcb(values: np.ndarray, groups: np.ndarray) -> float:
    unique = np.unique(groups)
    means = np.asarray([values[groups == value].mean() for value in unique])
    rng = np.random.default_rng(20260720)
    draws = rng.integers(0, len(means), size=(5000, len(means)))
    return float(np.quantile(means[draws].mean(axis=1), 0.025))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tapes", type=int, default=48)
    parser.add_argument("--include-records", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results/program_t/d2_branching_selector_v1/result.json")
    args = parser.parse_args()
    if not 5 <= args.n_tapes <= 48:
        raise ValueError("D2 must remain within burned T0 tapes")
    q = json.loads(Q_RESULT.read_text()); t0 = json.loads(T0_RESULT.read_text()); sched = scheduler()
    features, current_features, targets, groups, cells, rows_meta = [], [], [], [], [], []
    for cell in CONFIRMED_RET_CELLS:
        q_rows = q["trajectory_audits"][cell.cell_id]
        learner_seeds = sorted(q_rows, key=int)
        mpc_calendars = t0["cells"][cell.cell_id]["comparators"]["ret_proxy_scenario_h3_p4"]["calendar"]
        for offset in range(args.n_tapes):
            tape = 7_490_001 + offset
            skeleton, _ = extract_full_des_skeleton(seed=tape, scheduler=sched, regime_persistence=cell.regime_persistence, dominant_share=cell.dominant_share, downstream_freight_physics_mode="fixed_clock_physical_v1")
            counts = weekly_product_counts(order_times=skeleton.order_times, order_products=skeleton.order_products, decision_start=skeleton.decision_start, weeks=skeleton.decision_weeks)
            base = tuple(map(int, mpc_calendars[offset]))
            _calendar, decisions = state_rich_calendar(
                skeleton=skeleton.as_dict(), scheduler=sched,
                config=StateRichConfiguration("belief_mpc", 1),
                regime_persistence=0.75, dominant_share=0.90,
                action_overrides=base,
            )
            q_matrix = np.asarray([q_rows[seed]["calendars"][offset] for seed in learner_seeds], dtype=int)
            for week in range(skeleton.decision_weeks):
                q_action = int(np.argmax(np.bincount(q_matrix[:, week], minlength=4)))
                if q_action == base[week]:
                    continue
                alternative = list(base); alternative[week] = q_action
                metrics = simulate_full_des_frontier(skeleton=skeleton, scheduler=sched, calendars=np.asarray([base, alternative], dtype=np.uint8))
                delta = float(metrics["ret_visible"][1] - metrics["ret_visible"][0])
                observation = decisions[week].observation
                feature = observation_features(observation, counts[:week])
                features.append(feature); current_features.append(feature[:-7]); targets.append(delta)
                groups.append(f"{cell.cell_id}:{tape}"); cells.append(cell.cell_id)
                rows_meta.append({"cell": cell.cell_id, "tape": tape, "week": week, "mpc_action": base[week], "q_modal_action": q_action, "delta": delta})
    x = np.asarray(features); x_current = np.asarray(current_features); y = np.asarray(targets); group = np.asarray(groups); cell_array = np.asarray(cells)
    predictions = {name: np.zeros(len(y)) for name in ("ridge_history", "tree_depth3", "boosting_shallow", "ridge_current")}
    splitter = GroupKFold(n_splits=5)
    for train, test in splitter.split(x, y, group):
        models = {
            "ridge_history": (Ridge(alpha=10.0), x),
            "tree_depth3": (DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=20260720), x),
            "boosting_shallow": (HistGradientBoostingRegressor(max_depth=3, max_iter=60, learning_rate=0.05, min_samples_leaf=20, random_state=20260720), x),
            "ridge_current": (Ridge(alpha=10.0), x_current),
        }
        for name, (model, design) in models.items():
            model.fit(design[train], y[train]); predictions[name][test] = model.predict(design[test])
    summaries = {}
    for name, prediction in predictions.items():
        choose_q = prediction > 0.0
        realized = np.where(choose_q, y, 0.0)
        summaries[name] = {
            "mean_one_step_delta": float(realized.mean()),
            "cluster_lcb95": cluster_lcb(realized, group),
            "override_fraction": float(choose_q.mean()),
            "cells": {
                cell: {
                    "mean_one_step_delta": float(realized[cell_array == cell].mean()),
                    "override_fraction": float(choose_q[cell_array == cell].mean()),
                }
                for cell in sorted(set(cells))
            },
        }
    best_name = max(summaries, key=lambda name: summaries[name]["mean_one_step_delta"])
    best = summaries[best_name]
    promoted = best["cluster_lcb95"] >= 0.012 and min(row["mean_one_step_delta"] for row in best["cells"].values()) >= 0.0
    payload = {
        "schema_version": "program_t_d2_branching_selector_v1",
        "claim_status": "EXPLORATORY_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "burned_seed_range": [7490001, 7490256],
        "n_tapes": args.n_tapes,
        "n_disagreement_branches": len(y),
        "branch_contract": "common_mpc_prefix_and_continuation_one_action_pulse",
        "caveat": "one-step convertibility is not a closed-loop hybrid rollout",
        "models": summaries,
        "selected_model": best_name,
        "verdict": "GO_SELECTOR_CLOSED_LOOP_ROLLOUT" if promoted else "STOP_Q_MPC_SELECTOR_NOT_OBSERVABLY_CONVERTIBLE",
        "closed_loop_hybrid_authorized": False,
        "records": rows_meta if args.include_records else rows_meta[:20],
        "records_truncated": not args.include_records and len(rows_meta) > 20,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
