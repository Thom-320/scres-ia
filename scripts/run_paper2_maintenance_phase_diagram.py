#!/usr/bin/env python3
"""Learner-free physical envelope screen for Paper 2 maintenance control."""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
from statistics import mean

import numpy as np

from supply_chain.maintenance_control import (
    ACTIONS, materialize_tape, periodic_policy, periodic_sequences, run_policy,
    worst_condition_policy,
)

PRIMARY = "ret_excel_full_ledger"
SERVICE = "service_loss_auc_ration_hours"


def ci_lower(values: list[float]) -> float:
    if not values:
        return 0.0
    rng = np.random.default_rng(20260712)
    a = np.asarray(values)
    draws = rng.choice(a, size=(2000, len(a)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapes", type=int, default=4)
    ap.add_argument("--weeks", type=int, default=8)
    ap.add_argument("--max-period", type=int, default=2)
    ap.add_argument("--output", type=Path, default=Path("results/paper2_maintenance/phase_diagram/verdict.json"))
    args = ap.parse_args()
    tapes = [materialize_tape(1200001 + i, weeks=args.weeks) for i in range(args.tapes)]
    sequences = periodic_sequences(args.max_period)
    cells = product(
        (0.65, 0.75, 0.85), (0.30, 0.50, 0.70), (1, 2, 3),
        ("low", "high"), ("current", "increased"),
    )
    rows = []
    for q, restore, wip, hetero, repair in cells:
        cell = {
            "sensor_balanced_accuracy": q, "pm_restore_fraction": restore,
            "wip_capacity_days": wip, "wear_heterogeneity": hetero,
            "repair_profile": repair,
        }
        periodic = {"|".join(seq): [] for seq in sequences}
        oracle, oracle_actions, observed = [], [], []
        for tape in tapes:
            local = {}
            for seq in sequences:
                name = "|".join(seq)
                outcome = run_policy(tape, periodic_policy(seq), cell=cell)
                periodic[name].append(outcome)
                local[name] = outcome
            winner = max(local, key=lambda name: local[name][PRIMARY])
            oracle.append(local[winner])
            oracle_actions.append(winner.split("|")[0])
            observed.append(run_policy(tape, worst_condition_policy, cell=cell))
        best = max(periodic, key=lambda name: mean(r[PRIMARY] for r in periodic[name]))
        baseline = periodic[best]
        d_oracle = [a[PRIMARY] - b[PRIMARY] for a, b in zip(oracle, baseline)]
        d_obs = [a[PRIMARY] - b[PRIMARY] for a, b in zip(observed, baseline)]
        service = mean([
            (b[SERVICE] - a[SERVICE]) / max(abs(b[SERVICE]), 1e-9)
            for a, b in zip(oracle, baseline)
        ])
        support = {action: oracle_actions.count(action) / len(oracle_actions) for action in ACTIONS}
        rows.append({
            "cell": cell, "best_static": best, "oracle_ret_delta": mean(d_oracle),
            "oracle_ret_lcb95": ci_lower(d_oracle), "oracle_service_reduction": service,
            "observable_ret_delta": mean(d_obs), "observable_ret_lcb95": ci_lower(d_obs),
            "action_support": support,
            "diverse": sum(v >= 0.15 for v in support.values()) >= 2 and max(support.values()) <= 0.85,
        })
    rows.sort(key=lambda row: (row["observable_ret_delta"], row["oracle_ret_delta"]), reverse=True)
    passing = [row for row in rows if row["diverse"] and row["oracle_ret_delta"] >= 0.01 and row["oracle_ret_lcb95"] > 0 and row["oracle_service_reduction"] >= 0.05 and row["observable_ret_lcb95"] > 0]
    verdict = {
        "contract_id": "paper2_maintenance_control_v1", "stage": "learner_free_phase_diagram",
        "n_cells": len(rows), "n_tapes_per_cell": args.tapes,
        "n_periodic_calendars": len(sequences), "passing_cells": len(passing),
        "top_cells": rows[:10], "qualifying_cells": passing,
        "verdict": "PROMOTE_PHASE_REGION_TO_CONFIRMATION" if passing else "STOP_NO_MAINTENANCE_REGION_AT_SCREEN_RESOLUTION",
        "claim_boundary": "Screen resolution only; a passing cell requires fresh calibration and connected-region confirmation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in verdict.items() if k not in {"top_cells", "qualifying_cells"}}, indent=2))
    print(json.dumps(rows[:3], indent=2))


if __name__ == "__main__":
    main()
