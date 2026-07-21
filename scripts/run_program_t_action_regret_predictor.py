#!/usr/bin/env python3
"""Cross-fitted action-regret learning against the strongest H3 comparator.

Unlike D3, this script does not translate a demand forecast into an action.  It
branches all four admissible first actions, obtains their best H3 continuation
under the certified full-DES transducer, and learns the resulting action value.
The final evaluation is a closed-loop calendar rollout on held-out tape groups.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.program_o_full_des_transducer import extract_full_des_skeleton, simulate_full_des_frontier  # noqa: E402
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS  # noqa: E402
from supply_chain.program_o_state_rich import StateRichConfiguration, state_rich_calendar  # noqa: E402
from supply_chain.program_t_full_des_mpc import FullDEST0Config, choose_t0_action  # noqa: E402

T0_RESULT = ROOT / "results/program_t/t0_ret_transducer_v1/result.json"
HISTORY = 8
THRESHOLDS = (0.0, 0.0025, 0.005, 0.01, 0.02)


def observation_vector(observation) -> np.ndarray:
    return np.asarray([
        *observation.on_hand, *observation.locked_pipeline,
        *observation.backlog_quantity, *observation.backlog_orders,
        *observation.max_backlog_age, *observation.in_flight_quantity,
        observation.belief_c, observation.predicted_share_c,
        -1 if observation.previous_action is None else observation.previous_action,
        observation.remaining_decisions,
    ], dtype=float)


def history_vector(history: list[np.ndarray]) -> np.ndarray:
    width = len(history[-1]); out = np.zeros((HISTORY, width), dtype=float)
    selected = history[-HISTORY:]; out[-len(selected):] = selected
    return out.reshape(-1)


def design_row(observation, history: list[np.ndarray], action: int, *, use_history: bool) -> np.ndarray:
    current = observation_vector(observation)
    context = history_vector(history) if use_history else current
    one_hot = np.zeros(4); one_hot[int(action)] = 1.0
    return np.concatenate([context, one_hot])


def candidate_action_values(*, skeleton, sched, prefix: list[int]) -> np.ndarray:
    remaining = skeleton.decision_weeks - len(prefix); horizon = min(3, remaining)
    sequences = list(product(range(4), repeat=horizon)); calendars = []
    for sequence in sequences:
        tail = remaining - len(sequence)
        calendars.append(tuple(prefix) + sequence + (sequence[-1],) * tail)
    values = simulate_full_des_frontier(
        skeleton=skeleton, scheduler=sched, calendars=np.asarray(calendars, dtype=np.uint8)
    )["ret_visible"]
    by_action = np.full(4, -np.inf)
    for sequence, value in zip(sequences, values):
        by_action[sequence[0]] = max(by_action[sequence[0]], float(value))
    if not np.all(np.isfinite(by_action)):
        raise AssertionError("every action must have a finite H3 continuation")
    return by_action


def cluster_lcb(values: np.ndarray) -> float:
    rng = np.random.default_rng(20260721)
    draws = rng.integers(0, len(values), size=(5000, len(values)))
    return float(np.quantile(values[draws].mean(axis=1), 0.025))


def model_factories(seed: int):
    return {
        "ridge_current": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "boosting_current": lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=120, learning_rate=0.05, min_samples_leaf=25, random_state=seed),
        "ridge_history": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "boosting_history": lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=120, learning_rate=0.05, min_samples_leaf=25, random_state=seed),
        "extra_trees_history": lambda: ExtraTreesRegressor(n_estimators=200, max_depth=10, min_samples_leaf=4, max_features=0.7, random_state=seed, n_jobs=1),
        "mlp_history": lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 64), activation="tanh", alpha=1e-4, learning_rate_init=3e-4, max_iter=300, early_stopping=True, random_state=seed)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tapes", type=int, default=48)
    parser.add_argument("--include-rows", action="store_true")
    parser.add_argument("--holdout-split", type=int, default=0, help="train on the first N tapes per cell and evaluate on the remainder")
    parser.add_argument("--fixed-model", choices=tuple(model_factories(0)), default=None)
    parser.add_argument("--fixed-threshold", type=float, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "results/program_t/action_regret_predictor_v1/result.json")
    args = parser.parse_args()
    if not 10 <= args.n_tapes <= 48:
        raise ValueError("use only 10..48 burned T0 tapes")
    t0 = json.loads(T0_RESULT.read_text()); sched = scheduler()
    x_current: list[np.ndarray] = []; x_history: list[np.ndarray] = []; y: list[float] = []
    groups: list[str] = []; skeletons = {}; baselines = {}; training_oracle = {}
    for cell in CONFIRMED_RET_CELLS:
        base_rows = t0["cells"][cell.cell_id]["comparators"]["ret_proxy_scenario_h3_p4"]["calendar"]
        for offset in range(args.n_tapes):
            tape = 7_490_001 + offset; group = f"{cell.cell_id}:{tape}"
            skeleton, _ = extract_full_des_skeleton(seed=tape, scheduler=sched, regime_persistence=cell.regime_persistence, dominant_share=cell.dominant_share, downstream_freight_physics_mode="fixed_clock_physical_v1")
            base = list(map(int, base_rows[offset])); skeletons[group] = skeleton; baselines[group] = base
            _calendar, decisions = state_rich_calendar(skeleton=skeleton.as_dict(), scheduler=sched, config=StateRichConfiguration("belief_mpc", 1), regime_persistence=0.75, dominant_share=0.90, action_overrides=base)
            observed: list[np.ndarray] = []
            for week in range(skeleton.decision_weeks):
                observation = decisions[week].observation; observed.append(observation_vector(observation))
                values = candidate_action_values(skeleton=skeleton, sched=sched, prefix=base[:week])
                training_oracle[f"{group}:{week}"] = values.tolist()
                for action in range(4):
                    x_current.append(design_row(observation, observed, action, use_history=False))
                    x_history.append(design_row(observation, observed, action, use_history=True))
                    # Decision target: zero for the best H3 first action and a
                    # negative regret for every alternative.  Absolute return
                    # differences between states cannot dominate the ranking.
                    y.append(float(values[action] - np.max(values))); groups.append(group)
    current = np.asarray(x_current); history = np.asarray(x_history); target = np.asarray(y); group_array = np.asarray(groups)
    unique = np.asarray(sorted(set(groups)))
    if args.holdout_split:
        if not 10 <= args.holdout_split < args.n_tapes:
            raise ValueError("holdout split must leave at least one evaluation tape")
        train_groups = np.asarray([i for i, name in enumerate(unique) if int(name.split(":")[1]) < 7_490_001 + args.holdout_split])
        test_groups = np.asarray([i for i, name in enumerate(unique) if int(name.split(":")[1]) >= 7_490_001 + args.holdout_split])
        splits = [(train_groups, test_groups)]
        split_label = f"fixed first-{args.holdout_split} calibration versus held-out remainder"
    else:
        splits = GroupKFold(n_splits=5).split(unique, groups=unique)
        split_label = "five_fold_grouped_by_complete_cell_tape"
    fold_rows = []
    for fold, (train_groups, test_groups) in enumerate(splits):
        train_names = set(unique[train_groups]); test_names = set(unique[test_groups])
        train = np.asarray([name in train_names for name in group_array])
        fitted = {}
        factories = model_factories(20260721 + fold)
        if args.fixed_model:
            factories = {args.fixed_model: factories[args.fixed_model]}
        for name, factory in factories.items():
            design = history if name.endswith("history") else current
            fitted[name] = (factory().fit(design[train], target[train]), name.endswith("history"))
        for group in sorted(test_names):
            skeleton = skeletons[group]; baseline = baselines[group]
            for model_name, (model, uses_history) in fitted.items():
                thresholds = (args.fixed_threshold,) if args.fixed_threshold is not None else THRESHOLDS
                for threshold in thresholds:
                    prefix: list[int] = []; observed: list[np.ndarray] = []; overrides = 0
                    for week in range(skeleton.decision_weeks):
                        probe = tuple(prefix + [0] * (skeleton.decision_weeks - len(prefix)))
                        _calendar, decisions = state_rich_calendar(skeleton=skeleton.as_dict(), scheduler=sched, config=StateRichConfiguration("belief_mpc", 1), regime_persistence=0.75, dominant_share=0.90, action_overrides=probe)
                        observation = decisions[week].observation; observed.append(observation_vector(observation))
                        design = np.asarray([design_row(observation, observed, action, use_history=uses_history) for action in range(4)])
                        scores = model.predict(design)
                        best_action = max(range(4), key=lambda a: (float(scores[a]), -a))
                        mpc_action, _diagnostics = choose_t0_action(
                            observation, scheduler=sched,
                            config=FullDEST0Config(horizon=3, mode="scenario", particles=4),
                        )
                        advantage = float(scores[best_action] - scores[mpc_action])
                        action = best_action if best_action != mpc_action and advantage >= threshold else mpc_action
                        overrides += int(action != mpc_action); prefix.append(int(action))
                    metrics = simulate_full_des_frontier(skeleton=skeleton, scheduler=sched, calendars=np.asarray([baseline, prefix], dtype=np.uint8))
                    cell, tape = group.split(":"); key = f"{model_name}__tau_{threshold:g}"
                    fold_rows.append({
                        "fold": fold, "model": key, "base_model": model_name, "threshold": threshold,
                        "cell": cell, "tape": int(tape), "calendar": prefix, "overrides": overrides,
                        "baseline_calendar": baseline, "ret_delta": float(metrics["ret_visible"][1] - metrics["ret_visible"][0]),
                        "worst_product_delta": float(metrics["worst_product_fill"][1] - metrics["worst_product_fill"][0]),
                        "lost_order_delta": float(metrics["lost_orders"][1] - metrics["lost_orders"][0]),
                        "resource_delta": float(metrics["gross_production_quantity"][1] - metrics["gross_production_quantity"][0]),
                    })
    summaries = {}
    for name in sorted({row["model"] for row in fold_rows}):
        rows = [row for row in fold_rows if row["model"] == name]; delta = np.asarray([row["ret_delta"] for row in rows])
        cells = {}
        for cell in sorted({row["cell"] for row in rows}):
            values = np.asarray([row["ret_delta"] for row in rows if row["cell"] == cell]); cells[cell] = {"mean": float(values.mean()), "lcb95": cluster_lcb(values)}
        summaries[name] = {
            "pooled_mean": float(delta.mean()), "pooled_lcb95": cluster_lcb(delta), "cells": cells,
            "favorable_fraction": float(np.mean(delta > 0.0)),
            "action_change_fraction": float(np.mean([np.mean(np.asarray(row["calendar"]) != np.asarray(row["baseline_calendar"])) for row in rows])),
            "override_fraction": float(np.mean([row["overrides"] / 8.0 for row in rows])),
            "worst_product_mean_delta": float(np.mean([row["worst_product_delta"] for row in rows])),
            "lost_orders_nonincrease": max(row["lost_order_delta"] for row in rows) <= 0.0,
            "resources_exact": max(abs(row["resource_delta"]) for row in rows) <= 1e-12,
        }
    selected = max(summaries, key=lambda name: (summaries[name]["pooled_mean"], name)); best = summaries[selected]
    promoted = best["pooled_mean"] >= 0.01 and min(cell["mean"] for cell in best["cells"].values()) >= 0.0 and best["favorable_fraction"] >= 0.70 and best["lost_orders_nonincrease"] and best["resources_exact"] and best["worst_product_mean_delta"] >= -0.02
    payload = {
        "schema_version": "program_t_action_regret_predictor_v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_status": "EXPLORATORY_BURNED_NO_CLAIM", "burned_seed_range": [7490001, 7490256], "n_tapes": args.n_tapes,
        "target": "full-DES H3 first-action regret relative to the best candidate action",
        "confidence_thresholds": list(THRESHOLDS),
        "cross_fitting": split_label, "fixed_model": args.fixed_model, "fixed_threshold": args.fixed_threshold,
        "models": summaries, "selected_model": selected,
        "verdict": "GO_MINIMAL_HYBRID_DEVELOPMENT" if promoted else "STOP_ACTION_REGRET_NOT_CONVERTIBLE",
        "hybrid_training_authorized": promoted,
        "rows": fold_rows if args.include_rows else fold_rows[:24], "rows_truncated": not args.include_rows and len(fold_rows) > 24,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
