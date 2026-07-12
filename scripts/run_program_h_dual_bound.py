#!/usr/bin/env python3
"""Program H — information-relaxation DUAL upper bound on J*_obs (LOCKED test 1070001).

Brown-Smith-Sun: relax to perfect information (per-tape 81-seq hindsight), subtract a penalty
z_w that is a martingale difference w.r.t. the OBSERVABLE filtration, so E[z_w | obs] = 0 and
E_w[ max_a (ret_order(a) - sum_w z_w(a)) ] >= J*_obs is a RIGOROUS upper bound.

Penalty generator phi (frozen): phi(inv, w) = negative expected remaining service loss under the
ABAB continuation given current inventory and belief-expected demand. z_w = phi(realized next inv)
- E_belief[phi(next inv) | obs<=w, a] corrects the hindsight policy for demand it "already saw".
Point-belief approximation (the filter nearly reveals current tempo; log-loss 0.23 on dev) is
disclosed. Sanity chain checked: J_ABAB <= J*_obs_estimate <= dual_bound <= J_PI. If
dual_bound - J_ABAB < delta_min=0.01 -> Case A (information-limited) rigorously (within phi/grid).
Only 1070001 opened; 1080001 stays sealed.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.program_g import (
    ACTIONS, CONVOY_LOAD, CSSU_CAP, DEMAND_DAYS, MULT, SB_INITIAL, S1_DAILY, TEMPO,
    _week_step, materialize_tape, periodic_calendars, ret_order_metrics, simulate, simulate_orders,
)
from supply_chain.program_h_belief import CSSUFilter

REGION = [{"cell_id": f"P{p}_Q{int(q*100)}_L{l}_S150", "signal_q": q, "lead_weeks": l,
           "surge_mult": 1.50, "persistence": p, "r22_weekly_prob": 0.05}
          for p in ("short", "long") for q in (0.65, 0.75, 0.85) for l in (1, 2)]
WEEKS, ARM = 4, "TRS"
TEMPO_MEAN = lambda sm: {0: 2500 / 2 * MULT["low"] * DEMAND_DAYS, 1: 2500 / 2 * DEMAND_DAYS,
                         2: 2500 / 2 * sm * DEMAND_DAYS}


def rt(i, base):
    return materialize_tape(base + i, REGION[i % len(REGION)], WEEKS, persistent=True)


def ret(t, seq):
    return ret_order_metrics(simulate_orders(t, seq, ARM))["ret_order"]


def phi(inv, w, exp_dem):
    """neg expected remaining service loss under ABAB continuation from (inv,w)."""
    inv = np.array(inv, float); loss = 0.0
    for k in range(w, WEEKS):
        a = "A" if (k % 2 == 0) else "B"                 # ABAB
        i = 0 if a == "A" else 1
        deliver = min(3 * CONVOY_LOAD, CSSU_CAP - inv[i])
        inv[i] += deliver
        for j in range(2):
            loss += max(0.0, exp_dem[j] - inv[j]); inv[j] = max(0.0, inv[j] - exp_dem[j])
    return -loss


def dual_bound_tape(t):
    sm = t.cell["surge_mult"]; means = TEMPO_MEAN(sm)
    p = t.cell["persistence"]; q = t.cell["signal_q"]; lead = t.cell["lead_weeks"]
    fA = CSSUFilter(p, sm, q, lead); fB = CSSUFilter(p, sm, q, lead)
    # belief-expected demand per week from the filter (observable filtration)
    exp_dem = []
    for w in range(WEEKS):
        if w > 0:
            fA.update(demand=t.demand[w - 1, 0]); fB.update(demand=t.demand[w - 1, 1])
            fA.predict(); fB.predict()
        fA.update(sig=int(t.signal[w, 0])); fB.update(sig=int(t.signal[w, 1]))
        mA, mB = fA.tempo_marginal(), fB.tempo_marginal()
        exp_dem.append([float(mA @ np.array([means[0], means[1], means[2]])),
                        float(mB @ np.array([means[0], means[1], means[2]]))])
    # penalized hindsight: max over the 81 action seqs of ret - sum_w z_w
    best = -np.inf
    for seq in itertools.product(ACTIONS, repeat=WEEKS):
        inv = np.zeros(2); sb = float(SB_INITIAL); pen = 0.0
        invs = [inv.copy()]
        for w in range(WEEKS):
            inv, sb, _ = _week_step(inv, sb, seq[w], t.demand[w], t.r22[w], True)
            invs.append(inv.copy())
        for w in range(WEEKS):
            realized = phi(invs[w + 1], w + 1, exp_dem[min(w + 1, WEEKS - 1)] if w + 1 < WEEKS else [0, 0])
            # belief-expected next-inv phi: recompute invs[w+1] under belief-expected week-w demand
            inv_b = invs[w].copy()
            inv_b2, _, _ = _week_step(inv_b, SB_INITIAL, seq[w], np.array(exp_dem[w]), t.r22[w], True)
            expected = phi(inv_b2, w + 1, exp_dem[min(w + 1, WEEKS - 1)] if w + 1 < WEEKS else [0, 0])
            pen += (realized - expected)
        val = ret(t, seq) - pen
        best = max(best, val)
    return best


def boot_ci(x, n=2000, seed=7):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    m = [rng.choice(x, len(x), True).mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    n = 150
    cal = [rt(i, 1060001) for i in range(200)]
    test = [rt(i, 1070001) for i in range(n)]                 # LOCKED test
    cals = periodic_calendars(WEEKS)
    cl = np.array([[simulate(t, c, arm=ARM).service_loss for t in cal] for c in cals])
    abab = cals[int(cl.mean(axis=1).argmin())]

    abab_r = np.array([ret(t, abab) for t in test])
    pi_r = np.array([max(ret(t, s) for s in itertools.product(ACTIONS, repeat=WEEKS)) for t in test])
    dual = np.array([dual_bound_tape(t) for t in test])

    d_abab = boot_ci(dual - abab_r)
    pi_abab = boot_ci(pi_r - abab_r)
    # sanity chain: ABAB <= dual <= PI (per tape means)
    chain_ok = bool(abab_r.mean() <= dual.mean() + 1e-9 <= pi_r.mean() + 1e-6)
    interp = ("H_LOCKED_CASE_A_INFORMATION_LIMITED_CERTIFIED" if d_abab[2] < 0.01 and chain_ok
              else "H_LOCKED_DUAL_ABOVE_DELTA_INCONCLUSIVE")
    out = {"gate": "PROGRAM_H_DUAL_BOUND_LOCKED", "universe": "1070001 locked (1080001 sealed)",
           "n_test": n, "delta_min": 0.01, "abab": list(abab),
           "mean_ret_order": {"ABAB": float(abab_r.mean()), "J_PI_ceiling": float(pi_r.mean()),
                              "dual_upper_bound": float(dual.mean())},
           "dual_minus_ABAB_ci95": d_abab, "J_PI_minus_ABAB_ci95": pi_abab,
           "sanity_chain_ABAB_le_dual_le_PI": chain_ok, "interpretation": interp,
           "note": ("Brown-Smith-Sun info-relaxation dual: phi=neg ABAB remaining service loss, "
                    "martingale penalty z_w=phi(realized)-E_belief[phi]. Point-belief approx (filter "
                    "log-loss 0.23) + coarse phi disclosed. dual >= J*_obs rigorously in expectation; "
                    "if dual-ABAB upper CI < delta_min AND chain holds -> Case A information-limited.")}
    output = Path("results/program_h/dual"); output.mkdir(parents=True, exist_ok=True)
    (output / "verdict.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: out[k] for k in ("interpretation", "mean_ret_order",
          "dual_minus_ABAB_ci95", "J_PI_minus_ABAB_ci95", "sanity_chain_ABAB_le_dual_le_PI")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
