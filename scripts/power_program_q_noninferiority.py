#!/usr/bin/env python3
"""Non-authoritative early approximation to Program Q power.

Burned calibration matrices only. Two-way resampling (learner seeds x tapes) of the empirical
per-tape panels; endpoints per contract: (E1) H_OL LCB >= +0.01 in all 3 cells;
(E2) equivalence CI(Delta_N) within [-0.01,+0.01] in all 3 cells. Critical value frozen at
c=2.24 (approximates the hierarchical simultaneous level; disclosed approximation). This script
does not reselect all 65,536 open-loop calendars and all ten classical controllers inside every
resample, so it cannot select Program Q's contractual N. The authoritative analysis is
``scripts/power_program_q_replication.py``.
"""

import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "results/program_o/ret_only_learner_v1/calibration_run"
CELLS = ["rho75_share90", "rho90_share75", "rho90_share90"]
SEEDS_N, TAPES_N, C_CRIT, M, GRID = 10, 48, 2.24, 2000, [128, 160, 192, 256]


def enc(cal):
    v = 0
    for a in cal:
        v = v * 4 + int(a)
    return v


res = json.load(open(RUN / "result.json"))
panels = {}
for cell in CELLS:
    tapes = sorted((RUN / "raw_calendar_matrix" / cell).glob("tape_*.npz"))
    fr = np.stack([np.load(t)["ret_visible"] for t in tapes])  # (48, 65536)
    summ = res["cell_summaries"][cell]
    ol = fr[:, int(summ["best_open_loop_index"])]  # (48,)
    cls = np.array([fr[i, enc(c)] for i, c in enumerate(summ["best_classical_calendars"])])
    aud = res["trajectory_audits"][cell]
    lrn = np.stack(
        [[fr[i, enc(c)] for i, c in enumerate(aud[s]["calendars"])] for s in sorted(aud)]
    )  # (10, 48)
    panels[cell] = {"H_OL": lrn - ol[None, :], "D_N": lrn - cls[None, :]}
    print(
        cell,
        "obs H_OL",
        round(float((lrn - ol).mean()), 5),
        "obs D_N",
        round(float((lrn - cls).mean()), 5),
    )

rng = np.random.default_rng(20260718)
out = {
    "schema_version": "program_q_power_approximation_v1",
    "status": "NONAUTHORITATIVE_APPROXIMATION",
    "program_q_N_authority": False,
    "authoritative_script": "scripts/power_program_q_replication.py",
    "c_critical": C_CRIT,
    "replications": M,
    "grid": GRID,
    "endpoints": {"E1": "LCB(H_OL)>=0.01 all cells", "E2": "CI(D_N) in [-0.01,+0.01] all cells"},
    "power": {},
}
selected = None
for N in GRID:
    e1 = e2 = 0
    for _ in range(M):
        t_idx = rng.integers(0, TAPES_N, N)
        s_idx = rng.integers(0, SEEDS_N, SEEDS_N)
        ok1 = ok2 = True
        for cell in CELLS:
            for key, rule in (("H_OL", "sup"), ("D_N", "eq")):
                X = panels[cell][key][np.ix_(s_idx, t_idx)]
                per_tape = X.mean(axis=0)
                m, se = per_tape.mean(), per_tape.std(ddof=1) / np.sqrt(N)
                if rule == "sup":
                    ok1 &= (m - C_CRIT * se) >= 0.01
                else:
                    ok2 &= ((m - C_CRIT * se) >= -0.01) and ((m + C_CRIT * se) <= 0.01)
        e1 += ok1
        e2 += ok2
    out["power"][str(N)] = {"E1": e1 / M, "E2": e2 / M}
    print(f"N={N}: power E1={e1 / M:.3f} E2={e2 / M:.3f}")
    if selected is None and e1 / M >= 0.80 and e2 / M >= 0.80:
        selected = N
out["selected_N"] = None
out["diagnostic_selected_N"] = selected
out["verdict"] = "NO_CONTRACTUAL_VERDICT"
Path(ROOT / "research/paper2_exhaustive_search/program_q_power_20260718.json").write_text(
    json.dumps(out, indent=1)
)
print("DIAGNOSTIC ONLY:", selected, out["verdict"])
