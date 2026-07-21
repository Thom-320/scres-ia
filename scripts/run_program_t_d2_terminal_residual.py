#!/usr/bin/env python3
"""Cross-fitted H3 plus learned full-DES continuation residual on burned tapes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.program_o_full_des_transducer import extract_full_des_skeleton, simulate_full_des_frontier  # noqa: E402
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS  # noqa: E402
from supply_chain.program_o_state_rich import StateRichConfiguration, state_rich_calendar  # noqa: E402

T0_RESULT = ROOT / "results/program_t/t0_ret_transducer_v1/result.json"


def obs_vector(observation) -> np.ndarray:
    return np.asarray([
        *observation.on_hand, *observation.locked_pipeline,
        *observation.backlog_quantity, *observation.backlog_orders,
        *observation.max_backlog_age, *observation.in_flight_quantity,
        observation.belief_c, observation.predicted_share_c,
        -1 if observation.previous_action is None else observation.previous_action,
        observation.remaining_decisions,
    ], dtype=float)


def scheduler_counts(sched) -> np.ndarray:
    return np.asarray([[sched[str(action)].count("P_C"), sched[str(action)].count("P_H")] for action in range(4)], dtype=float)


def short_proxy(observation, sequence: tuple[int, ...], counts: np.ndarray) -> float:
    net = np.asarray(observation.on_hand) + np.asarray(observation.locked_pipeline) - np.asarray(observation.backlog_quantity)
    belief = float(observation.belief_c); rho = 0.75; share = 0.90
    backlog_area = 0.0; demand_total = 0.0
    for action in sequence:
        p_c = belief * share + (1.0 - belief) * (1.0 - share)
        demand = np.asarray([15000.0 * p_c, 15000.0 * (1.0 - p_c)])
        net = net + 5000.0 * counts[action] - demand
        backlog_area += float(np.maximum(0.0, -net).sum()); demand_total += float(demand.sum())
        belief = rho * belief + (1.0 - rho) * (1.0 - belief)
    return 1.0 - backlog_area / max(demand_total * (len(sequence) + 1), 1.0)


def feature(observation, sequence: tuple[int, ...], counts: np.ndarray) -> tuple[np.ndarray, float]:
    proxy = short_proxy(observation, sequence, counts)
    padded = np.full(3, -1.0); padded[: len(sequence)] = sequence
    return np.concatenate([obs_vector(observation), padded, [proxy]]), proxy


def candidate_calendars(prefix: list[int], weeks: int) -> tuple[list[tuple[int, ...]], np.ndarray]:
    horizon = min(3, weeks - len(prefix))
    sequences = list(product(range(4), repeat=horizon))
    calendars = []
    for sequence in sequences:
        tail = weeks - len(prefix) - len(sequence)
        calendars.append(tuple(prefix) + sequence + (sequence[-1],) * tail)
    return sequences, np.asarray(calendars, dtype=np.uint8)


def lcb(values: np.ndarray) -> float:
    rng = np.random.default_rng(20260720)
    draws = rng.integers(0, len(values), size=(5000, len(values)))
    return float(np.quantile(values[draws].mean(axis=1), 0.025))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tapes", type=int, default=48)
    parser.add_argument("--include-rows", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results/program_t/d2_terminal_residual_v1/result.json")
    args = parser.parse_args()
    if not 10 <= args.n_tapes <= 48:
        raise ValueError("terminal residual uses 10..48 burned tapes")
    t0 = json.loads(T0_RESULT.read_text()); sched = scheduler(); counts = scheduler_counts(sched)
    dataset_x, dataset_y, dataset_group = [], [], []
    skeletons = {}; baselines = {}
    for cell in CONFIRMED_RET_CELLS:
        baseline_rows = t0["cells"][cell.cell_id]["comparators"]["ret_proxy_scenario_h3_p4"]
        for offset in range(args.n_tapes):
            tape = 7_490_001 + offset; group = f"{cell.cell_id}:{tape}"
            skeleton, _ = extract_full_des_skeleton(seed=tape, scheduler=sched, regime_persistence=cell.regime_persistence, dominant_share=cell.dominant_share, downstream_freight_physics_mode="fixed_clock_physical_v1")
            skeletons[group] = skeleton; base = list(map(int, baseline_rows["calendar"][offset])); baselines[group] = base
            _cal, decisions = state_rich_calendar(skeleton=skeleton.as_dict(), scheduler=sched, config=StateRichConfiguration("belief_mpc", 1), regime_persistence=0.75, dominant_share=0.90, action_overrides=base)
            for week in range(skeleton.decision_weeks):
                sequences, calendars = candidate_calendars(base[:week], skeleton.decision_weeks)
                outcomes = simulate_full_des_frontier(skeleton=skeleton, scheduler=sched, calendars=calendars)["ret_visible"]
                for sequence, outcome in zip(sequences, outcomes):
                    x, proxy = feature(decisions[week].observation, sequence, counts)
                    dataset_x.append(x); dataset_y.append(float(outcome) - proxy); dataset_group.append(group)
    x = np.asarray(dataset_x); y = np.asarray(dataset_y); groups = np.asarray(dataset_group)
    unique = np.asarray(sorted(set(dataset_group))); fold_rows = []
    splitter = GroupKFold(n_splits=5)
    for fold, (train_groups, test_groups) in enumerate(splitter.split(unique, groups=unique)):
        train_names = set(unique[train_groups]); test_names = set(unique[test_groups])
        train_mask = np.asarray([name in train_names for name in groups])
        model = HistGradientBoostingRegressor(max_depth=3, max_iter=100, learning_rate=0.05, min_samples_leaf=30, random_state=20260720).fit(x[train_mask], y[train_mask])
        for group in sorted(test_names):
            skeleton = skeletons[group]; prefix: list[int] = []
            for week in range(skeleton.decision_weeks):
                probe = tuple(prefix + [0] * (skeleton.decision_weeks - len(prefix)))
                _cal, decisions = state_rich_calendar(skeleton=skeleton.as_dict(), scheduler=sched, config=StateRichConfiguration("belief_mpc", 1), regime_persistence=0.75, dominant_share=0.90, action_overrides=probe)
                sequences, _calendars = candidate_calendars(prefix, skeleton.decision_weeks)
                rows = [feature(decisions[week].observation, sequence, counts) for sequence in sequences]
                design = np.asarray([row[0] for row in rows]); proxy = np.asarray([row[1] for row in rows])
                scores = proxy + model.predict(design)
                best = max(range(len(sequences)), key=lambda i: (float(scores[i]), *tuple(-value for value in sequences[i])))
                prefix.append(int(sequences[best][0]))
            calendars = np.asarray([baselines[group], prefix], dtype=np.uint8)
            metrics = simulate_full_des_frontier(skeleton=skeleton, scheduler=sched, calendars=calendars)
            cell, tape = group.split(":")
            fold_rows.append({
                "fold": fold, "cell": cell, "tape": int(tape), "calendar": prefix,
                "baseline_calendar": baselines[group],
                "ret_delta": float(metrics["ret_visible"][1] - metrics["ret_visible"][0]),
                "worst_product_delta": float(metrics["worst_product_fill"][1] - metrics["worst_product_fill"][0]),
                "lost_order_delta": float(metrics["lost_orders"][1] - metrics["lost_orders"][0]),
                "resource_delta": float(metrics["gross_production_quantity"][1] - metrics["gross_production_quantity"][0]),
            })
    delta = np.asarray([row["ret_delta"] for row in fold_rows]); cell_names = sorted({row["cell"] for row in fold_rows})
    cell_summary = {}
    for cell in cell_names:
        values = np.asarray([row["ret_delta"] for row in fold_rows if row["cell"] == cell])
        cell_summary[cell] = {"mean": float(values.mean()), "lcb95": lcb(values)}
    change_fraction = float(np.mean([np.mean(np.asarray(row["calendar"]) != np.asarray(row["baseline_calendar"])) for row in fold_rows]))
    guardrails = {
        "worst_product_mean_delta": float(np.mean([row["worst_product_delta"] for row in fold_rows])),
        "lost_orders_nonincrease": max(row["lost_order_delta"] for row in fold_rows) <= 0.0,
        "resources_exact": max(abs(row["resource_delta"]) for row in fold_rows) <= 1e-12,
    }
    pooled = {"mean": float(delta.mean()), "lcb95": lcb(delta)}
    promoted = pooled["lcb95"] >= 0.015 and min(value["mean"] for value in cell_summary.values()) >= 0.0 and change_fraction >= 0.10 and guardrails["lost_orders_nonincrease"] and guardrails["resources_exact"]
    payload = {
        "schema_version": "program_t_d2_terminal_residual_v1",
        "claim_status": "EXPLORATORY_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "burned_seed_range": [7490001, 7490256],
        "n_tapes": args.n_tapes,
        "cross_fitting": "five_fold_grouped_by_complete_cell_tape",
        "method": "H3_short_proxy_plus_cross_fitted_full_DES_residual",
        "pooled": pooled, "cells": cell_summary, "calendar_action_change_fraction": change_fraction,
        "guardrails": guardrails,
        "rows": fold_rows if args.include_rows else fold_rows[:20],
        "rows_truncated": not args.include_rows and len(fold_rows) > 20,
        "verdict": "GO_TERMINAL_RESIDUAL_DEVELOPMENT" if promoted else "STOP_TERMINAL_CONTINUATION_NOT_CONVERTIBLE",
        "hybrid_training_authorized": promoted,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
