#!/usr/bin/env python3
"""Program G G2 — 24-cell learnability screen (pre-RL, no learner).

Grid = 2 persistence x 3 signal x 2 lead x 2 surge (V1.2). For each cell: H_PI (spatial
headroom) and H_obs (observable cover-policy conversion), best static frozen on calibration,
evaluated on disjoint holdout. Reports which cells pass (H_obs CI95>0 & eta>=0.30 &
resource-matched) and the largest connected passing component (adjacency = differ by one
level on one axis). Promotion needs a connected component of >=2 cells. Screen tapes 980001+;
no virgin tapes, no PPO.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.program_g import (
    cover_signal_policy, enumerate_oracle, materialize_tape, periodic_calendars, simulate,
)

PERSIST = ["short", "long"]
SIGNAL = [0.65, 0.75, 0.85]
LEAD = [1, 2]
SURGE = [1.25, 1.50]
ARM = "TRS"


def boot_lo(x, n=1500, seed=11):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    return float(np.percentile([rng.choice(x, len(x), True).mean() for _ in range(n)], 2.5))


def cell_id(p, q, l, s):
    return f"P{p}_Q{int(q*100)}_L{l}_S{int(s*100)}"


def eval_cell(cell, n=120, weeks=4, base=980001):
    cal = [materialize_tape(base + i, cell, weeks, persistent=True) for i in range(n)]
    hold = [materialize_tape(base + 500 + i, cell, weeks, persistent=True) for i in range(n)]
    cals = periodic_calendars(weeks)
    cl = np.array([[simulate(t, c, arm=ARM).service_loss for t in cal] for c in cals])
    best_cal = cals[int(cl.mean(axis=1).argmin())]
    static = np.array([simulate(t, best_cal, arm=ARM).service_loss for t in hold])
    oracle = np.array([enumerate_oracle(t, arm=ARM)[0] for t in hold])
    obs = np.array([simulate(t, cover_signal_policy(t, ARM, use_signal=True), arm=ARM).service_loss for t in hold])
    m_obs = np.mean([simulate(t, cover_signal_policy(t, ARM), arm=ARM).convoy_missions for t in hold])
    m_stat = np.mean([simulate(t, best_cal, arm=ARM).convoy_missions for t in hold])
    H_PI, H_obs = static - oracle, static - obs
    eta = H_obs.sum() / max(H_PI.sum(), 1e-9)
    obs_lo = boot_lo(H_obs)
    passes = bool(obs_lo > 0 and eta >= 0.30 and m_obs <= m_stat + 0.2 and boot_lo(H_PI) > 0)
    return {"H_PI_mean": float(H_PI.mean()), "H_obs_mean": float(H_obs.mean()),
            "H_obs_lo": obs_lo, "eta": float(eta), "missions_obs": float(m_obs),
            "missions_static": float(m_stat), "passes": passes}


def largest_connected(passing: set) -> list:
    """Adjacency: two cells adjacent iff they differ by exactly one level on one axis."""
    grid = list(itertools.product(range(2), range(3), range(2), range(2)))
    idx = {g: cell_id(PERSIST[g[0]], SIGNAL[g[1]], LEAD[g[2]], SURGE[g[3]]) for g in grid}
    adj = lambda a, b: sum(x != y for x, y in zip(a, b)) == 1 and all(abs(x - y) <= 1 for x, y in zip(a, b))
    P = [g for g in grid if idx[g] in passing]
    best = []
    seen = set()
    for start in P:
        if start in seen:
            continue
        comp = []; stack = [start]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u); comp.append(u)
            for v in P:
                if v not in seen and adj(u, v):
                    stack.append(v)
        if len(comp) > len(best):
            best = comp
    return [idx[g] for g in best]


def main() -> int:
    cells = {}
    for p, q, l, s in itertools.product(PERSIST, SIGNAL, LEAD, SURGE):
        cid = cell_id(p, q, l, s)
        cell = {"cell_id": cid, "signal_q": q, "lead_weeks": l, "surge_mult": s,
                "persistence": p, "r22_weekly_prob": 0.05}
        cells[cid] = eval_cell(cell)
        print(f"[program-g-screen] {cid} pass={cells[cid]['passes']} "
              f"H_obs_lo={cells[cid]['H_obs_lo']:.1f} eta={cells[cid]['eta']:.2f}", flush=True)
    passing = {c for c, r in cells.items() if r["passes"]}
    component = largest_connected(passing)
    out = {"gate": "PROGRAM_G_G2_24CELL_SCREEN", "arm": ARM, "n_cells": len(cells),
           "n_passing": len(passing), "largest_connected_passing_component": component,
           "promotable": len(component) >= 2, "cells": cells,
           "ppo_trained": False, "virgin_tapes_opened": 0,
           "interpretation": ("G2_PROMOTABLE_CONNECTED_REGION" if len(component) >= 2
                              else "G2_NO_CONNECTED_PASSING_REGION")}
    output = Path("results/program_g/g2"); output.mkdir(parents=True, exist_ok=True)
    (output / "verdict.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: out[k] for k in ("interpretation", "n_passing",
          "largest_connected_passing_component", "promotable")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
