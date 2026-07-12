#!/usr/bin/env python3
"""Program G — metric triangulation (exploratory, NOT a rescue, NOT virgin-confirmatory).

One trajectory, many lenses. New tapes (calibration 1020001+, locked test 1030001+ — the
G5/bridge universes 1010001+ are retired to development). Frozen policies evaluated under:
service-loss, ret_excel_full_ledger_guardrailed, Cobb-Douglas (repo G24 exponents, sigmoid),
Cobb-Douglas spatial (geometric mean over CSSUs), attended orders, worst-CSSU fill. Reports
the policy ranking induced by each metric -> the metric-induced-policy-reversal result.
Cobb-Douglas is a SECONDARY construct index; it does NOT replace ret_excel or rescue G5.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.program_g import (
    ACTIONS, SB_INITIAL, _week_step, cover_signal_policy, materialize_tape, metrics_all,
    mpc_policy, observe, oracle_action_dataset, periodic_calendars, ret_order_metrics,
    simulate, simulate_orders,
)

REGION = [{"cell_id": f"P{p}_Q{int(q*100)}_L{l}_S150", "signal_q": q, "lead_weeks": l,
           "surge_mult": 1.50, "persistence": p, "r22_weekly_prob": 0.05}
          for p in ("short", "long") for q in (0.65, 0.75, 0.85) for l in (1, 2)]
WEEKS, ARM = 4, "TRS"


def rt(i, base):
    return materialize_tape(base + i, REGION[i % len(REGION)], WEEKS, persistent=True)


def ret_order(t, seq):
    return ret_order_metrics(simulate_orders(t, seq, ARM))["ret_order"]


def boot_ci(x, n=2000, seed=7):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    m = [rng.choice(x, len(x), True).mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fit_tree(train, objective):
    """objective: 'service' or 'retexcel' -> label = that objective's clairvoyant action."""
    import itertools
    X, y = [], []
    for t in train:
        if objective == "service":
            xs, ys = oracle_action_dataset(t, ARM); X += xs; y += ys
        else:
            best = max(itertools.product(ACTIONS, repeat=WEEKS),
                       key=lambda s: ret_order(t, s))
            inv = np.zeros(2); sb = float(SB_INITIAL)
            for w in range(WEEKS):
                X.append(observe(inv, sb, t, w)); y.append(ACTIONS.index(best[w]))
                inv, sb, _ = _week_step(inv, sb, best[w], t.demand[w], t.r22[w], True)
    tr = DecisionTreeClassifier(max_depth=3, random_state=0).fit(np.array(X), y)
    def fn(t):
        inv = np.zeros(2); sb = float(SB_INITIAL); acts = []
        for w in range(WEEKS):
            a = int(tr.predict(observe(inv, sb, t, w).reshape(1, -1))[0]); acts.append(ACTIONS[a])
            inv, sb, _ = _week_step(inv, sb, ACTIONS[a], t.demand[w], t.r22[w], True)
        return tuple(acts)
    return fn


def main() -> int:
    ncal, ntest = 160, 200
    cal = [rt(i, 1020001) for i in range(ncal)]
    test = [rt(i, 1030001) for i in range(ntest)]

    cals = periodic_calendars(WEEKS)
    cl = np.array([[simulate(t, c, arm=ARM).service_loss for t in cal] for c in cals])
    abab = cals[int(cl.mean(axis=1).argmin())]
    svc_tree = fit_tree(cal, "service")
    ret_tree = fit_tree(cal, "retexcel")

    policies = {
        "ABAB_static": lambda t: abab,
        "cover": lambda t: cover_signal_policy(t, ARM),
        "mpc": lambda t: mpc_policy(t, ARM),
        "service_tree": svc_tree,
        "retexcel_tree": ret_tree,
    }
    METRICS = ["unfulfilled_rations_at_horizon_lower_better", "ret_order", "ret_quantity", "cd_sigmoid",
               "cd_spatial", "attended", "worst_cssu_fill"]
    data = {p: {m: [] for m in METRICS} for p in policies}
    for t in test:
        for p, fn in policies.items():
            seq = fn(t); mm = metrics_all(t, seq, ARM); rm = ret_order_metrics(mm["orders"])
            data[p]["unfulfilled_rations_at_horizon_lower_better"].append(
                mm["unfulfilled_rations_at_horizon"])
            data[p]["ret_order"].append(rm["ret_order"])          # each order equal (thesis)
            data[p]["ret_quantity"].append(rm["ret_quantity"])    # each ration equal (mass)
            data[p]["cd_sigmoid"].append(mm["cd_sigmoid"])
            data[p]["cd_spatial"].append(mm["cd_spatial"])
            data[p]["attended"].append(mm["attended_orders"])
            data[p]["worst_cssu_fill"].append(mm["worst_cssu_fill"])

    means = {p: {m: float(np.mean(data[p][m])) for m in METRICS} for p in policies}
    # winner per metric (service_loss lower better; others higher better)
    winners = {}
    for m in METRICS:
        higher = m != "unfulfilled_rations_at_horizon_lower_better"
        winners[m] = (max if higher else min)(policies, key=lambda p: means[p][m])
    # cover vs ABAB per metric (the reversal), CI95
    reversal = {}
    for m in METRICS:
        c = np.array(data["cover"][m]); a = np.array(data["ABAB_static"][m])
        reversal[m] = boot_ci(c - a)   # sign interpreted per metric direction

    out = {"gate": "PROGRAM_G_METRIC_TRIANGULATION", "kind": "exploratory_metric_sensitivity",
           "calibration_seed_start": 1020001, "test_seed_start": 1030001,
           "n_calibration": ncal, "n_test": ntest, "abab": list(abab),
           "cd_exponents_frozen_G24": {"a": 0.024, "b": 0.026, "c": 0.040, "d": 0.060, "n": 0.1771},
           "phi_kappa_held_constant_disclosed": True,
           "means": means, "winner_per_metric": winners, "cover_minus_ABAB_ci95": reversal,
           "note": ("Cobb-Douglas = SECONDARY construct index (repo ReT_garrido2024 exponents); does "
                    "NOT replace ret_excel or rescue G5. phi(spare capacity)/kappa(cost) held constant "
                    "in v1.2 (S1 fixed, resources matched) -> index driven by zeta/epsilon/tau. "
                    "Exploratory, not virgin-confirmatory. Stylized order adapter, not full DES.")}
    # honest interpretation
    same = len(set(winners.values())) == 1
    out["interpretation"] = ("TRIANGULATION_ALL_METRICS_AGREE" if same
                             else "METRIC_INDUCED_POLICY_REVERSAL_CONFIRMED")
    output = Path("results/program_g/triangulation"); output.mkdir(parents=True, exist_ok=True)
    (output / "verdict.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"interpretation": out["interpretation"], "winner_per_metric": winners,
                      "means": means}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
