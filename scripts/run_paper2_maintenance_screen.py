#!/usr/bin/env python3
"""Pre-learner periodic frontier, PI headroom and observable-policy screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np

from supply_chain.maintenance_control import (
    ACTIONS, materialize_tape, periodic_policy, periodic_sequences, run_policy,
    truncate_tape, wip_bottleneck_policy, worst_condition_policy,
)

PRIMARY = "ret_excel_full_ledger"
SERVICE = "service_loss_auc_ration_hours"


def ci(values: list[float], seed: int = 20260712) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    a = np.asarray(values, dtype=float)
    draws = rng.choice(a, size=(4000, len(a)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def evaluate(tapes, sequences, cell):
    periodic = {"|".join(seq): [] for seq in sequences}
    observable = {"worst_condition": [], "wip_bottleneck": []}
    oracle = []
    first_actions = []
    for tape in tapes:
        tape_rows = {}
        for seq in sequences:
            name = "|".join(seq)
            row = run_policy(tape, periodic_policy(seq), cell=cell)
            periodic[name].append(row)
            tape_rows[name] = row
        winner = max(tape_rows, key=lambda name: tape_rows[name][PRIMARY])
        oracle.append(tape_rows[winner])
        first_actions.append(winner.split("|")[0])
        observable["worst_condition"].append(run_policy(tape, worst_condition_policy, cell=cell))
        observable["wip_bottleneck"].append(run_policy(tape, wip_bottleneck_policy, cell=cell))
    return periodic, observable, oracle, first_actions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapes", type=int, default=12)
    ap.add_argument("--max-period", type=int, default=6)
    ap.add_argument("--seed-start", type=int, default=1200001)
    ap.add_argument("--output", type=Path, default=Path("results/paper2_maintenance/screen/verdict.json"))
    args = ap.parse_args()
    sequences = periodic_sequences(args.max_period)
    full = [materialize_tape(seed, weeks=8) for seed in range(args.seed_start, args.seed_start + args.tapes)]
    short = [truncate_tape(tape, 4) for tape in full]
    cell = {
        "sensor_balanced_accuracy": 0.75, "pm_restore_fraction": 0.50,
        "wip_capacity_days": 2, "wear_heterogeneity": "high",
        "repair_profile": "current",
    }
    p8, o8, oracle8, first8 = evaluate(full, sequences, cell)
    p4, _, oracle4, first4 = evaluate(short, sequences, cell)
    best_static = max(p8, key=lambda name: mean(row[PRIMARY] for row in p8[name]))
    static_rows = p8[best_static]
    oracle_delta = [a[PRIMARY] - b[PRIMARY] for a, b in zip(oracle8, static_rows)]
    service_reduction = [
        (b[SERVICE] - a[SERVICE]) / max(abs(b[SERVICE]), 1e-9)
        for a, b in zip(oracle8, static_rows)
    ]
    support = {action: first8.count(action) / len(first8) for action in ACTIONS}
    horizon_agreement = mean([a == b for a, b in zip(first8, first4)])
    observable_summary = {}
    best_observable = None
    for name, rows in o8.items():
        deltas = [a[PRIMARY] - b[PRIMARY] for a, b in zip(rows, static_rows)]
        capture = mean(deltas) / mean(oracle_delta) if mean(oracle_delta) > 0 else 0.0
        observable_summary[name] = {
            "ret_delta_mean": mean(deltas), "ret_delta_ci95": ci(deltas),
            "oracle_capture": capture,
            "service_loss_delta_mean": mean([a[SERVICE] - b[SERVICE] for a, b in zip(rows, static_rows)]),
        }
        if best_observable is None or mean(deltas) > observable_summary[best_observable]["ret_delta_mean"]:
            best_observable = name
    gates = {
        "action_diversity": sum(value >= 0.15 for value in support.values()) >= 2 and max(support.values()) <= 0.85,
        "oracle_ret": mean(oracle_delta) >= 0.01 and ci(oracle_delta)[0] > 0.0,
        "service_practical": mean(service_reduction) >= 0.05,
        "horizon_stability": horizon_agreement >= 0.90,
        "observable_conversion": observable_summary[best_observable]["ret_delta_ci95"][0] > 0.0 and observable_summary[best_observable]["oracle_capture"] >= 0.30,
        "equal_scheduled_resources": all(row["scheduled_pm_hours"] == full[0]["weeks"] * 24.0 for rows in p8.values() for row in rows),
    }
    verdict = {
        "contract_id": "paper2_maintenance_control_v1", "stage": "central_cell_prelearner_screen",
        "n_tapes": args.tapes, "n_periodic_calendars": len(sequences), "best_static": best_static,
        "action_support": support, "oracle_ret_delta_mean": mean(oracle_delta),
        "oracle_ret_delta_ci95": ci(oracle_delta), "oracle_service_loss_reduction_mean": mean(service_reduction),
        "horizon_first_action_agreement": horizon_agreement,
        "observable": observable_summary, "gates": gates,
    }
    verdict["verdict"] = "PROMOTE_MAINTENANCE_TO_LEARNER" if all(gates.values()) else "STOP_NO_OBSERVABLE_MAINTENANCE_HEADROOM"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
