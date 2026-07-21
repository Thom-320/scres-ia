#!/usr/bin/env python3
"""Cross-fitted causal-history screen before any recurrent model is authorized."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.program_o_full_des_transducer import extract_full_des_skeleton  # noqa: E402
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS  # noqa: E402
from supply_chain.program_t_joint_belief import ExactJointBelief, THETA_GRID, weekly_product_counts  # noqa: E402


def predictive_count_distribution(belief: ExactJointBelief) -> np.ndarray:
    probability = np.zeros(7, dtype=float)
    for theta_index, (_rho, share) in enumerate(THETA_GRID):
        for regime_index in (0, 1):
            p_c = share if regime_index else 1.0 - share
            for count in range(7):
                probability[count] += belief.probability[theta_index, regime_index] * math.comb(6, count) * p_c**count * (1.0 - p_c) ** (6 - count)
    return probability / probability.sum()


def frozen_belief() -> ExactJointBelief:
    return ExactJointBelief.oracle_parameters((0.75, 0.90))


def history_features(history: list[int], week: int) -> np.ndarray:
    padded = np.full(7, -1.0)
    if history:
        padded[: len(history)] = np.asarray(history, dtype=float)
    ewma = 3.0
    for value in history:
        ewma = 0.5 * value + 0.5 * ewma
    trend = 0.0 if len(history) < 2 else float(history[-1] - history[-2])
    return np.concatenate([padded, [float(week), ewma, trend]])


def current_features(history: list[int], week: int) -> np.ndarray:
    return np.asarray([float(week), float(history[-1]) if history else 3.0], dtype=float)


def _aligned_probability(model, x: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(x)
    output = np.full((len(x), 7), 1e-12, dtype=float)
    for column, label in enumerate(model.classes_):
        output[:, int(label)] = raw[:, column]
    output /= output.sum(axis=1, keepdims=True)
    return output


def grouped_lcb(delta: np.ndarray, groups: np.ndarray, *, resamples: int = 5000) -> float:
    unique = np.unique(groups)
    means = np.asarray([delta[groups == group].mean() for group in unique])
    rng = np.random.default_rng(20260720)
    draws = rng.integers(0, len(means), size=(resamples, len(means)))
    return float(np.quantile(means[draws].mean(axis=1), 0.025))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tapes", type=int, default=48)
    parser.add_argument("--output", type=Path, default=ROOT / "results/program_t/d3_history_predictability_v1/result.json")
    args = parser.parse_args()
    if not 5 <= args.n_tapes <= 48:
        raise ValueError("D3 uses 5..48 already-burned T0 tapes")
    sched = scheduler()
    x_history, x_current, targets, groups, frozen_probability, bayes_probability, metadata = [], [], [], [], [], [], []
    for cell in CONFIRMED_RET_CELLS:
        for offset in range(args.n_tapes):
            tape = 7_490_001 + offset
            skeleton, _ = extract_full_des_skeleton(seed=tape, scheduler=sched, regime_persistence=cell.regime_persistence, dominant_share=cell.dominant_share, downstream_freight_physics_mode="fixed_clock_physical_v1")
            counts = weekly_product_counts(order_times=skeleton.order_times, order_products=skeleton.order_products, decision_start=skeleton.decision_start, weeks=skeleton.decision_weeks)
            adaptive = ExactJointBelief.uniform()
            fixed = frozen_belief()
            history: list[int] = []
            for week, target in enumerate(counts):
                if week:
                    adaptive.observe_previous_week(counts[week - 1]); fixed.observe_previous_week(counts[week - 1])
                x_history.append(history_features(history, week)); x_current.append(current_features(history, week))
                targets.append(target); groups.append(f"{cell.cell_id}:{tape}")
                frozen_probability.append(predictive_count_distribution(fixed)); bayes_probability.append(predictive_count_distribution(adaptive))
                metadata.append({"cell": cell.cell_id, "tape": tape, "week": week})
                history.append(target)
    x_history = np.asarray(x_history); x_current = np.asarray(x_current); y = np.asarray(targets); group = np.asarray(groups)
    frozen_p = np.asarray(frozen_probability); bayes_p = np.asarray(bayes_probability)
    predictions = {"history_logistic": np.zeros((len(y), 7)), "current_logistic": np.zeros((len(y), 7)), "history_boosting": np.zeros((len(y), 7))}
    splitter = GroupKFold(n_splits=5)
    for train, test in splitter.split(x_history, y, group):
        models = {
            "history_logistic": (LogisticRegression(max_iter=2000, C=1.0), x_history),
            "current_logistic": (LogisticRegression(max_iter=2000, C=1.0), x_current),
            "history_boosting": (HistGradientBoostingClassifier(max_depth=3, max_iter=80, learning_rate=0.05, random_state=20260720), x_history),
        }
        for name, (model, x) in models.items():
            model.fit(x[train], y[train])
            predictions[name][test] = _aligned_probability(model, x[test])
    baseline_log = np.log(np.maximum(frozen_p[np.arange(len(y)), y], 1e-12))
    results = {
        "frozen_hmm": {"mean_log_score": float(baseline_log.mean())},
        "adaptive_exact_bayes": {},
    }
    all_predictions = {"adaptive_exact_bayes": bayes_p, **predictions}
    frozen_actions = np.rint((frozen_p * np.arange(7)).sum(axis=1) / 2.0).clip(0, 3).astype(int)
    for name, probability in all_predictions.items():
        log_score = np.log(np.maximum(probability[np.arange(len(y)), y], 1e-12))
        delta = log_score - baseline_log
        actions = np.rint((probability * np.arange(7)).sum(axis=1) / 2.0).clip(0, 3).astype(int)
        cell_rows = {}
        for cell in (row.cell_id for row in CONFIRMED_RET_CELLS):
            mask = np.asarray([row["cell"] == cell for row in metadata])
            cell_rows[cell] = {"log_score_delta": float(delta[mask].mean()), "action_change_fraction": float(np.mean(actions[mask] != frozen_actions[mask]))}
        results[name] = {
            "mean_log_score": float(log_score.mean()),
            "delta_vs_frozen": float(delta.mean()),
            "cluster_lcb95": grouped_lcb(delta, group),
            "action_change_fraction": float(np.mean(actions != frozen_actions)),
            "cells": cell_rows,
        }
    history = results["history_logistic"]
    no_cell_damage = min(value["log_score_delta"] for value in history["cells"].values()) >= 0.0
    predictive_pass = history["cluster_lcb95"] > 0.0 and no_cell_damage and history["action_change_fraction"] >= 0.10
    payload = {
        "schema_version": "program_t_d3_history_predictability_v1",
        "claim_status": "EXPLORATORY_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "burned_seed_range": [7490001, 7490256],
        "n_tapes": args.n_tapes,
        "n_rows": len(y),
        "cross_fitting": "five_fold_grouped_by_complete_cell_tape",
        "results": results,
        "verdict": "PREDICTIVE_SIGNAL_ONLY_AWAITING_CAUSAL_ROLLOUT" if predictive_pass else "STOP_NO_HISTORY_PREDICTIVE_INCREMENT",
        "gru_authorized": False,
        "reason_gru_not_authorized": "decision rollout and structured-model non-reproducibility gates remain required",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"results": results, "verdict": payload["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
