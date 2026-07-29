#!/usr/bin/env python3
"""Execute the frozen Program G terminal metric audit (stylized adapter only)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.program_g import (
    cover_signal_policy, materialize_tape, metrics_all, mpc_policy, periodic_calendars,
)
from scripts.run_program_g_triangulation import REGION, WEEKS, ARM, boot_ci, fit_tree

CAL_START, N_CAL = 1040001, 200
TEST_START, N_TEST = 1050001, 400


def tape(i: int, start: int):
    return materialize_tape(start + i, REGION[i % len(REGION)], WEEKS, persistent=True)


def evaluate(tapes, policies):
    fields = (
        "ret_order", "ret_quantity", "attended_orders", "worst_cssu_fill",
        "unfulfilled_rations_at_horizon", "cd_sigmoid", "cd_spatial",
    )
    data = {p: {f: [] for f in fields} for p in policies}
    for t in tapes:
        for name, policy in policies.items():
            m = metrics_all(t, policy(t), ARM)
            for field in fields:
                data[name][field].append(float(m[field]))
    return data


def main() -> int:
    calibration = [tape(i, CAL_START) for i in range(N_CAL)]
    locked = [tape(i, TEST_START) for i in range(N_TEST)]

    calendars = periodic_calendars(WEEKS)
    cal_ret = np.asarray([
        [metrics_all(t, calendar, ARM)["ret_order"] for t in calibration]
        for calendar in calendars
    ])
    best_static = calendars[int(cal_ret.mean(axis=1).argmax())]
    service_tree = fit_tree(calibration, "service")
    ret_tree = fit_tree(calibration, "retexcel")
    policies = {
        "best_periodic_static": lambda t: best_static,
        "cover": lambda t: cover_signal_policy(t, ARM),
        "mpc": lambda t: mpc_policy(t, ARM),
        "service_tree_depth3": service_tree,
        "ret_tree_depth3": ret_tree,
    }
    data = evaluate(locked, policies)
    baseline = data["best_periodic_static"]
    means = {p: {f: float(np.mean(v)) for f, v in metrics.items()}
             for p, metrics in data.items()}

    audits = {}
    for policy in policies:
        if policy == "best_periodic_static":
            continue
        p = data[policy]
        ci = {
            "ret_order": boot_ci(np.asarray(p["ret_order"]) - baseline["ret_order"]),
            "ret_quantity": boot_ci(np.asarray(p["ret_quantity"]) - baseline["ret_quantity"]),
            "attended_orders": boot_ci(np.asarray(p["attended_orders"]) - baseline["attended_orders"]),
            "worst_cssu_fill": boot_ci(np.asarray(p["worst_cssu_fill"]) - baseline["worst_cssu_fill"]),
            "unfulfilled_rations_at_horizon": boot_ci(
                np.asarray(p["unfulfilled_rations_at_horizon"])
                - baseline["unfulfilled_rations_at_horizon"]),
        }
        gates = {
            "ret_order_positive": ci["ret_order"][1] > 0,
            "ret_quantity_noninferior": ci["ret_quantity"][1] >= 0,
            "attended_noninferior": ci["attended_orders"][1] >= 0,
            "worst_cssu_fill_noninferior": ci["worst_cssu_fill"][1] >= 0,
            "unfulfilled_noninferior": ci["unfulfilled_rations_at_horizon"][2] <= 0,
            "same_resource_rights": True,
        }
        audits[policy] = {"paired_delta_ci95": ci, "gates": gates,
                          "passes_all": all(gates.values())}

    passing = [p for p, a in audits.items() if a["passes_all"]]
    verdict = ("PASS_PROGRAM_G_STYLIZED_ROBUST_OBSERVABLE_VALUE" if passing else
               "STOP_PROGRAM_G_NO_ROBUST_ADAPTIVE_VALUE_UNDER_STYLIZED_CONTRACT")
    result = {
        "gate": "PROGRAM_G_TERMINAL_METRIC_AUDIT_V1",
        "verdict": verdict,
        "scope": "stylized_program_g_order_adapter_not_full_des",
        "calibration": {"seed_start": CAL_START, "n": N_CAL},
        "locked_terminal_test": {"seed_start": TEST_START, "n": N_TEST},
        "frozen_best_periodic_static": list(best_static),
        "passing_observable_policies": passing,
        "means": means,
        "policy_audits": audits,
        "endpoint_note": ("unfulfilled_rations_at_horizon is terminal unmet quantity, not "
                          "service-loss AUC; Cobb-Douglas fields are secondary inspired indices."),
        "forbidden_claims": ["full_des_confirmation", "cobb_douglas_rescue",
                             "virgin_mfsc_confirmation"],
    }
    out = Path("results/program_g/terminal_metric_audit")
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": verdict, "best_static": best_static,
                      "passing": passing, "audits": audits}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
