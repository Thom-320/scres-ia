#!/usr/bin/env python3
"""Program H: frozen O0 belief-state audit and terminal learner gate."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.program_g import cover_signal_policy, materialize_tape, metrics_all, mpc_policy
from supply_chain.program_h import (
    ARM, belief_rollout_actions, filter_log_loss, fit_regret_q_policy,
    full_information_oracle_actions, regret_q_actions,
)

REGION = [{"cell_id": f"P{p}_Q{int(q*100)}_L{l}_S150", "signal_q": q,
           "lead_weeks": l, "surge_mult": 1.50, "persistence": p,
           "r22_weekly_prob": 0.05}
          for p in ("short", "long") for q in (0.65, 0.75, 0.85) for l in (1, 2)]
WEEKS = 4
CAL_START, N_CAL = 1060001, 200
TEST_START, N_TEST = 1070001, 400


def tape(i: int, start: int):
    return materialize_tape(start + i, REGION[i % len(REGION)], WEEKS, persistent=True)


def ci95(values, *, seed=20260713, n=4000):
    x = np.asarray(values, float); rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return [float(x.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5))]


def main() -> int:
    calibration = [tape(i, CAL_START) for i in range(N_CAL)]
    filter_loss, prior_loss = filter_log_loss(calibration)
    filter_pass = filter_loss < prior_loss
    models = fit_regret_q_policy(calibration)

    locked = [tape(i, TEST_START) for i in range(N_TEST)]
    policies = {
        "ABAB": lambda t: ("A", "B", "A", "B"),
        "cover_reference": lambda t: cover_signal_policy(t, ARM),
        "mpc_reference": lambda t: mpc_policy(t, ARM),
        "regret_fitted_q": lambda t: regret_q_actions(t, models),
        "belief_mpc_2w": lambda t: belief_rollout_actions(t, lookahead=2),
        "belief_point_rollout": lambda t: belief_rollout_actions(t, lookahead=None),
        "full_tape_oracle_ceiling": full_information_oracle_actions,
    }
    fields = ("ret_order", "ret_quantity", "attended_orders", "worst_cssu_fill",
              "unfulfilled_rations_at_horizon")
    values = {p: {f: [] for f in fields} for p in policies}
    action_counts = {p: {a: 0 for a in ("A", "B", "HOLD")} for p in policies}
    for t in locked:
        for name, policy in policies.items():
            actions = policy(t)
            for action in actions:
                action_counts[name][action] += 1
            m = metrics_all(t, actions, ARM)
            for field in fields:
                values[name][field].append(float(m[field]))

    baseline = values["ABAB"]
    means = {p: {f: float(np.mean(v)) for f, v in fs.items()} for p, fs in values.items()}
    pi_delta = np.asarray(values["full_tape_oracle_ceiling"]["ret_order"]) - baseline["ret_order"]
    h_pi = float(pi_delta.mean())
    audits = {}
    candidates = ("regret_fitted_q", "belief_mpc_2w", "belief_point_rollout")
    for name in candidates:
        v = values[name]
        deltas = {
            "ret_order": np.asarray(v["ret_order"]) - baseline["ret_order"],
            "ret_quantity": np.asarray(v["ret_quantity"]) - baseline["ret_quantity"],
            "attended_orders": np.asarray(v["attended_orders"]) - baseline["attended_orders"],
            "worst_cssu_fill": np.asarray(v["worst_cssu_fill"]) - baseline["worst_cssu_fill"],
            "unfulfilled_rations": np.asarray(v["unfulfilled_rations_at_horizon"])
                                    - baseline["unfulfilled_rations_at_horizon"],
        }
        cis = {k: ci95(x) for k, x in deltas.items()}
        favorable = float(np.mean(deltas["ret_order"] > 0))
        eta = float(cis["ret_order"][0] / h_pi) if h_pi > 0 else float("nan")
        gates = {
            "filter_informative": filter_pass,
            "ret_order_delta_at_least_0_01": cis["ret_order"][0] >= 0.01,
            "ret_order_lcb_positive": cis["ret_order"][1] > 0,
            "ret_quantity_noninferior": cis["ret_quantity"][1] >= 0,
            "attended_noninferior": cis["attended_orders"][1] >= 0,
            "worst_cssu_fill_within_0_02": cis["worst_cssu_fill"][1] >= -0.02,
            "unfulfilled_noninferior": cis["unfulfilled_rations"][2] <= 0,
            "resources_equal_rights": True,
            "favorable_tapes_at_least_0_70": favorable >= 0.70,
            "pi_conversion_at_least_0_30": eta >= 0.30,
        }
        audits[name] = {"delta_ci95": cis, "favorable_tapes": favorable,
                        "pi_conversion": eta, "gates": gates,
                        "passes_all": all(gates.values())}

    passing = [p for p, a in audits.items() if a["passes_all"]]
    oracle_ci = ci95(pi_delta)
    if passing:
        verdict = "PROMOTE_PROGRAM_H_TO_FROZEN_LEARNER"
        rl_authorized = True
    elif oracle_ci[2] < 0.01:
        verdict = "STOP_PROGRAM_H_INFORMATION_UPPER_BOUND_BELOW_MCID"
        rl_authorized = False
    else:
        verdict = "STOP_PROGRAM_H_NO_BELIEF_POLICY_PASS_INFORMATION_BOUND_REMAINS_LOOSE"
        rl_authorized = False

    result = {
        "gate": "PROGRAM_H_LOCKED_BELIEF_POLICY_GATE",
        "verdict": verdict,
        "scope": "O0_program_g_exact_stylized_contract",
        "filter": {"log_loss": filter_loss, "prior_log_loss": prior_loss,
                   "informative": filter_pass},
        "calibration": {"seed_start": CAL_START, "n": N_CAL},
        "locked_test": {"seed_start": TEST_START, "n": N_TEST},
        "virgin_1080001_opened": False,
        "rl_authorized": rl_authorized,
        "rigorous_full_information_ceiling_delta_ci95": oracle_ci,
        "qmdp_bound_claimed": False,
        "passing_belief_policies": passing,
        "means": means,
        "action_counts": action_counts,
        "policy_audits": audits,
        "interpretation_boundary": ("The full-tape oracle is a rigorous but loose information "
                                    "relaxation. If belief policies fail while it remains material, "
                                    "formal information insufficiency is unresolved; Program H still "
                                    "terminates under its last-program rule."),
    }
    out = Path("results/program_h/locked_belief_policy_gate")
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": verdict, "filter": result["filter"],
                      "oracle_ci": oracle_ci, "passing": passing,
                      "audits": audits}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
