#!/usr/bin/env python3
"""Frozen calibration and locked confirmation for Paper 2 bottleneck migration."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.paper2_bottleneck import (
    ACTIONS, ACTION_NAMES, CONTEXTS, materialize_tape, run_policy, signal_policy,
)

CAL_START, N_CAL = 1100001, 60
TEST_START, N_TEST = 1110001, 120
WEEKS = 24


def constant(action):
    return lambda obs: action


def tapes(start, n, split):
    return [materialize_tape(start + i, CONTEXTS[i % 3], split, weeks=WEEKS)
            for i in range(n)]


def ci95(values, seed=20260713, n=4000):
    x = np.asarray(values, float); rng = np.random.default_rng(seed)
    boot = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return [float(x.mean()), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5))]


def main() -> int:
    calibration = tapes(CAL_START, N_CAL, "calibration")
    cal_rows = {a: [run_policy(t, constant(a)) for t in calibration] for a in ACTIONS}
    cal_mean = {a: float(np.mean([r["ret_excel"] for r in rows]))
                for a, rows in cal_rows.items()}
    best = max(ACTIONS, key=lambda a: (cal_mean[a], ACTION_NAMES[a]))

    locked = tapes(TEST_START, N_TEST, "locked")
    policies = {f"constant_{ACTION_NAMES[a]}": constant(a) for a in ACTIONS}
    policies["signal_adaptive"] = signal_policy
    rows = {name: [run_policy(t, policy) for t in locked]
            for name, policy in policies.items()}
    baseline = rows[f"constant_{ACTION_NAMES[best]}"]
    candidate = rows["signal_adaptive"]

    ret_delta = np.asarray([c["ret_excel"] - b["ret_excel"]
                            for c, b in zip(candidate, baseline)])
    service_reduction = np.asarray([
        (b["service_loss_auc_ration_hours"] - c["service_loss_auc_ration_hours"])
        / max(abs(b["service_loss_auc_ration_hours"]), 1.0)
        for c, b in zip(candidate, baseline)
    ])
    lost_delta = np.asarray([c["n_lost"] - b["n_lost"]
                             for c, b in zip(candidate, baseline)])
    ret_ci, service_ci, lost_ci = ci95(ret_delta), ci95(service_reduction), ci95(lost_delta)
    favorable = float(np.mean(ret_delta > 0))
    crn = all(
        len({rows[name][i]["consumed_base_threat_sha256"] for name in rows}) == 1
        and len({rows[name][i]["realized_demand_sha256"] for name in rows}) == 1
        for i in range(N_TEST)
    )
    equal_resource = len({r["total_token_hours"] for values in rows.values() for r in values}) == 1
    max_mass = max(r["mass_residual"] for values in rows.values() for r in values)
    gates = {
        "ret_delta_min_0_01": ret_ci[0] >= 0.01,
        "ret_lcb_positive": ret_ci[1] > 0,
        "service_reduction_min_0_05": service_ci[0] >= 0.05,
        "service_lcb_positive": service_ci[1] > 0,
        "lost_noninferior": lost_ci[2] <= 0,
        "favorable_tapes_min_0_70": favorable >= 0.70,
        "equal_team_hours": equal_resource,
        "crn": crn,
        "mass_conservation": max_mass < 1e-6,
    }
    passed = all(gates.values())
    result = {
        "gate": "PAPER2_BOTTLENECK_MIGRATION_LOCKED_CONFIRMATION",
        "verdict": ("PASS_ADAPTIVE_BOTTLENECK_POLICY" if passed
                    else "STOP_NO_ADAPTIVE_BOTTLENECK_VALUE"),
        "calibration": {"seed_start": CAL_START, "n": N_CAL,
                        "mean_ret_by_constant": {ACTION_NAMES[a]: cal_mean[a] for a in ACTIONS},
                        "frozen_best_constant": ACTION_NAMES[best]},
        "locked": {"seed_start": TEST_START, "n": N_TEST},
        "signal_policy_minus_best_constant": {
            "ret_delta_ci95": ret_ci, "service_reduction_ci95": service_ci,
            "lost_delta_ci95": lost_ci, "favorable_tapes": favorable,
        },
        "means": {name: {
            "ret_excel": float(np.mean([r["ret_excel"] for r in values])),
            "service_loss_auc": float(np.mean([r["service_loss_auc_ration_hours"] for r in values])),
            "n_lost": float(np.mean([r["n_lost"] for r in values])),
        } for name, values in rows.items()},
        "gates": gates, "max_mass_residual": max_mass,
        "learner_1120001_opened": False, "ppo_trained": False,
    }
    out = Path("results/paper2_bottleneck/locked_confirmation")
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
