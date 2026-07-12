"""Program I — parameterized headroom estimator for global sensitivity analysis.

headroom_at(theta) returns the clairvoyant headroom H_PI (=EVPI), the observable
adaptive gap H_obs (=VSS analogue), and eta=H_obs/H_PI, all on ret_order, over a
Program-G-style stylized two-CSSU shared-convoy contract whose generative parameters
are set by theta. Reuses program_g (simulate, ret_order_metrics, periodic_calendars,
ACTIONS) and program_h_belief (CSSUFilter) unchanged. The observable policy is the
belief/nowcast cover policy (H_obs upper-ish estimate; matches the H lane).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .program_g import (
    ACTIONS, DEMAND_DAYS, MULT, SB_INITIAL, TEMPO, Tape, _semi_markov, _week_step,
    digest, materialize_tape, periodic_calendars, ret_order_metrics, simulate, simulate_orders,
)
from .program_h_belief import CSSUFilter

# Factor space (name -> (lo, hi)); lead/persistence are quantized inside theta_to_cell.
FACTORS = {
    "signal_q":     (0.50, 0.95),   # signal balanced accuracy (sens=spec)
    "lead":         (1.0, 3.0),     # advance-signal lead in weeks (round)
    "surge_mult":   (1.10, 3.00),   # surge demand multiplier = scarcity vs fixed convoy
    "persistence":  (0.0, 1.0),     # <0.5 short dwell {4,5,6}, else long {6,7,8}
    "commonality":  (0.00, 0.90),   # P(B tracks A's tempo) = A/B co-surge coupling
    "r22_prob":     (0.00, 0.30),   # weekly route-down prob = risk magnitude (control axis)
}
WEEKS, ARM = 4, "TRS"


def theta_to_cell(theta: dict) -> dict:
    return {"cell_id": "gsa", "signal_q": float(np.clip(theta["signal_q"], 0.5, 0.95)),
            "lead_weeks": int(round(np.clip(theta["lead"], 1, 3))),
            "surge_mult": float(np.clip(theta["surge_mult"], 1.1, 3.0)),
            "persistence": "short" if theta["persistence"] < 0.5 else "long",
            "commonality": float(np.clip(theta["commonality"], 0.0, 1.0)),
            "r22_weekly_prob": float(np.clip(theta["r22_prob"], 0.0, 0.3))}


def materialize_tape_theta(seed: int, cell: dict, weeks: int = WEEKS) -> Tape:
    """Base tape delegates EXACTLY to program_g.materialize_tape (so commonality=0 reproduces
    the frozen Program G distribution byte-for-byte); commonality>0 is a post-hoc A/B co-surge
    coupling (B tracks A on a fraction of weeks), with B's demand+signal regenerated for the
    coupled weeks from an independent coupling RNG (base RNG untouched)."""
    g_cell = {"cell_id": "gsa", "signal_q": cell["signal_q"], "lead_weeks": cell["lead_weeks"],
              "surge_mult": cell["surge_mult"], "persistence": cell["persistence"],
              "r22_weekly_prob": cell["r22_weekly_prob"]}
    t = materialize_tape(seed, g_cell, weeks, persistent=True)   # EXACT Program-G base
    com = float(cell.get("commonality", 0.0))
    if com <= 0.0:
        return t
    crng = np.random.default_rng(np.random.SeedSequence([seed, 0x0C0FFEE]))
    sm = cell["surge_mult"]; q = cell["signal_q"]; lead = cell["lead_weeks"]
    for w in range(weeks):
        if crng.random() < com:                                 # B tracks A this week
            t.z[w, 1] = t.z[w, 0]
            base = crng.integers(2400, 2601, size=DEMAND_DAYS)
            m = MULT.get(TEMPO[t.z[w, 1]], sm)
            t.demand[w, 1] = float(np.round(base / 2.0 * m).sum())
    for w in range(weeks):                                       # refresh B signal vs new future tempo
        tgt = w + lead
        fut = (tgt < weeks) and (TEMPO[t.z[tgt, 1]] == "surge")
        t.signal[w, 1] = (1 if crng.random() < q else 0) if fut else (1 if crng.random() > q else 0)
    t.threat_sha256 = digest({"z": t.z.tolist(), "demand": t.demand.tolist()})
    return t


def _belief_policy(t: Tape) -> tuple:
    c = t.cell; fA = CSSUFilter(c["persistence"], c["surge_mult"], c["signal_q"], c["lead_weeks"])
    fB = CSSUFilter(c["persistence"], c["surge_mult"], c["signal_q"], c["lead_weeks"])
    means = np.array([2500/2*MULT["low"]*DEMAND_DAYS, 2500/2*DEMAND_DAYS, 2500/2*c["surge_mult"]*DEMAND_DAYS])
    inv = np.zeros(2); sb = float(SB_INITIAL); acts = []
    for w in range(t.weeks):
        if w > 0:
            fA.update(demand=t.demand[w-1, 0]); fB.update(demand=t.demand[w-1, 1])
            fA.predict(); fB.predict()
        fA.update(sig=int(t.signal[w, 0])); fB.update(sig=int(t.signal[w, 1]))
        eA = float(fA.tempo_marginal() @ means); eB = float(fB.tempo_marginal() @ means)
        a = "A" if inv[0]/max(eA, 1) <= inv[1]/max(eB, 1) else "B"
        acts.append(a); inv, sb, _ = _week_step(inv, sb, a, t.demand[w], t.r22[w], True)
    return tuple(acts)


def _ret(t, seq):
    return ret_order_metrics(simulate_orders(t, seq, ARM))["ret_order"]


@dataclass
class Headroom:
    H_PI: float
    H_obs: float
    eta: float
    n: int


def headroom_at(theta: dict, n_tapes: int = 80, seed0: int = 3_000_001) -> Headroom:
    """Stable GSA headroom estimate. The static baseline is the STRONGEST full-contract
    periodic calendar evaluated on the eval tapes themselves (in-sample static -> the tightest,
    A/B-symmetry-aware, stable baseline; conservative headroom -> no false positives). This
    removes the ABAB<->BABA calibration coin-flip that corrupted a calibration-frozen static."""
    cell = theta_to_cell(theta)
    hold = [materialize_tape_theta(seed0 + i, cell) for i in range(n_tapes)]
    cals = periodic_calendars(WEEKS)
    static_by_cal = np.array([[_ret(t, c) for t in hold] for c in cals])   # (n_cal, n_tape)
    static = static_by_cal[int(static_by_cal.mean(axis=1).argmax())]        # best static on eval
    oracle = np.array([max(_ret(t, s) for s in itertools.product(ACTIONS, repeat=WEEKS)) for t in hold])
    obs = np.array([_ret(t, _belief_policy(t)) for t in hold])
    H_PI = float((oracle - static).mean())
    H_obs = float((obs - static).mean())
    return Headroom(H_PI=H_PI, H_obs=H_obs, eta=float(H_obs / H_PI) if abs(H_PI) > 1e-9 else 0.0, n=n_tapes)
