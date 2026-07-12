#!/usr/bin/env python3
"""Program H — belief audit on DEVELOPMENT tapes 1060001 (O0). Not the locked test.

Builds the augmented semi-Markov belief (validated vs the marginal prior by log-loss),
runs a belief-aware cover policy (J*_obs estimate) and a current-tempo-clairvoyant QMDP
diagnostic, and compares to ABAB and the perfect-information oracle on ret_order. If the
best belief policy does not beat ABAB by delta_min=0.01 (LCB95>0) on development, the
strong prior is Case A (information-limited); the LOCKED 1070001 confirmatory + the rigorous
information-relaxation dual bound are the next, separately-gated steps. No 1070001/1080001.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.program_g import (
    ACTIONS, DEMAND_DAYS, MULT, SB_INITIAL, TEMPO, _week_step, materialize_tape,
    periodic_calendars, ret_order_metrics, simulate, simulate_orders,
)
from supply_chain.program_h_belief import CSSUFilter

REGION = [{"cell_id": f"P{p}_Q{int(q*100)}_L{l}_S150", "signal_q": q, "lead_weeks": l,
           "surge_mult": 1.50, "persistence": p, "r22_weekly_prob": 0.05}
          for p in ("short", "long") for q in (0.65, 0.75, 0.85) for l in (1, 2)]
WEEKS, ARM = 4, "TRS"


def rt(i, base):
    return materialize_tape(base + i, REGION[i % len(REGION)], WEEKS, persistent=True)


def exp_demand(marg, surge_mult):
    means = [2500 / 2 * MULT["low"] * DEMAND_DAYS, 2500 / 2 * 1.0 * DEMAND_DAYS,
             2500 / 2 * surge_mult * DEMAND_DAYS]
    return float(marg @ np.array(means))


def belief_policy(tape, use_demand, use_signal):
    cell = tape.cell; p = cell["persistence"]; sm = cell["surge_mult"]
    q = cell["signal_q"]; lead = cell["lead_weeks"]
    fA = CSSUFilter(p, sm, q, lead); fB = CSSUFilter(p, sm, q, lead)
    inv = np.zeros(2); sb = float(SB_INITIAL); acts = []; margs = []
    for w in range(WEEKS):
        if w > 0:
            if use_demand:
                fA.update(demand=tape.demand[w - 1, 0]); fB.update(demand=tape.demand[w - 1, 1])
            fA.predict(); fB.predict()
        if use_signal:
            fA.update(sig=int(tape.signal[w, 0])); fB.update(sig=int(tape.signal[w, 1]))
        mA, mB = fA.tempo_marginal(), fB.tempo_marginal()
        margs.append((mA.copy(), mB.copy()))
        eA, eB = exp_demand(mA, sm), exp_demand(mB, sm)
        cover = [inv[0] / max(eA, 1.0), inv[1] / max(eB, 1.0)]
        a = "A" if cover[0] <= cover[1] else "B"
        acts.append(a)
        inv, sb, _ = _week_step(inv, sb, a, tape.demand[w], tape.r22[w], True)
    return tuple(acts), margs


def qmdp_policy(tape):
    """Diagnostic ceiling: dispatch to the lower-cover CSSU using the TRUE current tempo
    (knowing current Z, not the future) -- what an operator with a perfect nowcast would do."""
    inv = np.zeros(2); sb = float(SB_INITIAL); acts = []; sm = tape.cell["surge_mult"]
    means = {"low": 2500 / 2 * MULT["low"] * DEMAND_DAYS, "routine": 2500 / 2 * DEMAND_DAYS,
             "surge": 2500 / 2 * sm * DEMAND_DAYS}
    for w in range(WEEKS):
        eA = means[TEMPO[tape.z[w, 0]]]; eB = means[TEMPO[tape.z[w, 1]]]
        cover = [inv[0] / max(eA, 1.0), inv[1] / max(eB, 1.0)]
        a = "A" if cover[0] <= cover[1] else "B"
        acts.append(a)
        inv, sb, _ = _week_step(inv, sb, a, tape.demand[w], tape.r22[w], True)
    return tuple(acts)


def ret(t, seq):
    return ret_order_metrics(simulate_orders(t, seq, ARM))["ret_order"]


def pi_oracle(t):
    return max(ret(t, s) for s in itertools.product(ACTIONS, repeat=WEEKS))


def boot_ci(x, n=2000, seed=7):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    m = [rng.choice(x, len(x), True).mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def filter_logloss(tapes):
    """Validate: filtered tempo posterior log-loss vs the marginal-prior baseline."""
    fl, pr = [], []
    from supply_chain.program_g import STATIONARY
    prior = np.array([STATIONARY[TEMPO[t]] for t in range(3)])
    for t in tapes:
        _, margs = belief_policy(t, use_demand=True, use_signal=True)
        for w in range(WEEKS):
            for i, marg in enumerate(margs[w]):
                true_t = int(t.z[w, i])
                fl.append(-np.log(max(marg[true_t], 1e-9)))
                pr.append(-np.log(max(prior[true_t], 1e-9)))
    return float(np.mean(fl)), float(np.mean(pr))


def main() -> int:
    n = 200
    cal = [rt(i, 1060001) for i in range(n)]                 # DEVELOPMENT only
    cals = periodic_calendars(WEEKS)
    cl = np.array([[simulate(t, c, arm=ARM).service_loss for t in cal] for c in cals])
    abab = cals[int(cl.mean(axis=1).argmin())]

    ll_filter, ll_prior = filter_logloss(cal)
    R = {"ABAB": [], "belief_sig_dem": [], "belief_sig_only": [], "qmdp_nowcast": [], "pi_oracle": []}
    for t in cal:
        R["ABAB"].append(ret(t, abab))
        R["belief_sig_dem"].append(ret(t, belief_policy(t, True, True)[0]))
        R["belief_sig_only"].append(ret(t, belief_policy(t, False, True)[0]))
        R["qmdp_nowcast"].append(ret(t, qmdp_policy(t)))
        R["pi_oracle"].append(pi_oracle(t))
    R = {k: np.array(v) for k, v in R.items()}
    abab_a = R["ABAB"]
    deltas = {k: boot_ci(R[k] - abab_a) for k in R if k != "ABAB"}
    best_belief = max(["belief_sig_dem", "belief_sig_only"], key=lambda k: R[k].mean())
    lo = deltas[best_belief][1]
    interp = ("H_DEV_CASE_B_BELIEF_BEATS_ABAB" if lo > 0.01
              else "H_DEV_CASE_A_INFORMATION_LIMITED_STRONG_PRIOR")
    out = {"gate": "PROGRAM_H_BELIEF_AUDIT_DEVELOPMENT", "universe": "1060001 development (locked 1070001 NOT opened)",
           "n": n, "abab": list(abab), "delta_min": 0.01,
           "filter_logloss": {"filtered": ll_filter, "marginal_prior": ll_prior,
                              "filter_informative": bool(ll_filter < ll_prior)},
           "mean_ret_order": {k: float(v.mean()) for k, v in R.items()},
           "minus_ABAB_ci95": deltas, "best_belief_policy": best_belief,
           "H_PI_dev": boot_ci(R["pi_oracle"] - abab_a),
           "interpretation": interp,
           "note": ("DEVELOPMENT estimate. belief cover policy uses the augmented semi-Markov filter "
                    "(O0: signal +/- inventory-revealed demand). QMDP-nowcast is a current-tempo "
                    "clairvoyant diagnostic. If no belief policy beats ABAB by delta_min here, the "
                    "LOCKED 1070001 confirmatory + rigorous info-relaxation dual bound follow (gated).")}
    output = Path("results/program_h/dev"); output.mkdir(parents=True, exist_ok=True)
    (output / "verdict.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: out[k] for k in ("interpretation", "filter_logloss", "mean_ret_order",
                      "minus_ABAB_ci95", "H_PI_dev")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
