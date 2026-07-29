#!/usr/bin/env python3
"""Program I — headroom global sensitivity analysis (Stage 1 Morris [+ optional Sobol/GP]).

Targets H_PI (EVPI) and H_obs (adaptive gap, VSS analogue) as GSA OUTPUTS over the broad
structural factor space (info/tempo/scarcity/risk-control). Estimators validated on Ishigami;
estimator anchored to Program G (H_PI~+0.015, H_obs~-0.02 at commonality=0). Anti-p-hacking:
the whole map is reported; a high-H_obs region only becomes a NEW preregistered lane, never a
managerial claim from the GSA. Usage: run_headroom_gsa.py [--stage morris|sobol|gp] [--n N] [--r R].
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.headroom_sensitivity import FACTORS, headroom_at
from supply_chain.gsa import morris_screen, sobol_indices, gp_locate

BOUNDS = [(lo, hi, name) for name, (lo, hi) in FACTORS.items()]
NAMES = list(FACTORS)


def theta_of(x):
    return {NAMES[i]: float(x[i]) for i in range(len(NAMES))}


def make_f(target, n_tapes, seed0=3_000_001):
    def f(x):
        h = headroom_at(theta_of(x), n_tapes=n_tapes, seed0=seed0)
        return h.H_obs if target == "H_obs" else h.H_PI
    return f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="morris", choices=["morris", "sobol", "gp"])
    ap.add_argument("--n", type=int, default=60)      # tapes per headroom estimate
    ap.add_argument("--r", type=int, default=8)       # Morris trajectories
    ap.add_argument("--sobol-N", type=int, default=64)
    a = ap.parse_args()
    out = {"gate": "PROGRAM_I_HEADROOM_GSA", "stage": a.stage, "n_tapes": a.n,
           "factors": {k: list(v) for k, v in FACTORS.items()},
           "anchor_note": "commonality=0 reproduces Program G (H_PI~+0.015, H_obs~-0.02)"}

    if a.stage == "morris":
        for target in ("H_PI", "H_obs"):
            m = morris_screen(make_f(target, a.n), BOUNDS, r=a.r, seed=1)
            out[f"morris_{target}"] = m
            rank = sorted(m, key=lambda k: -m[k]["mu_star"])
            out[f"rank_{target}"] = rank
            print(f"[{target}] Morris mu*(effect) / sigma(interaction), ranked:")
            for k in rank:
                print(f"   {k:12} mu*={m[k]['mu_star']:.5f}  sigma={m[k]['sigma']:.5f}")
    elif a.stage == "sobol":
        for target in ("H_PI", "H_obs"):
            s = sobol_indices(make_f(target, a.n), BOUNDS, N=a.sobol_N, seed=2)
            out[f"sobol_{target}"] = s
            print(f"[{target}] Sobol S1 / ST / interaction_gap:")
            for k in NAMES:
                print(f"   {k:12} S1={s[k]['S1']:+.3f} ST={s[k]['ST']:+.3f} gap={s[k]['interaction_gap']:+.3f}")
    elif a.stage == "gp":
        g = gp_locate(make_f("H_obs", a.n), BOUNDS, n_init=16, n_iter=24, seed=3)
        out["gp_locate_H_obs"] = {"theta": theta_of(g["x_best"]), **g}
        h = headroom_at(theta_of(g["x_best"]), n_tapes=200)
        out["located_region_headroom"] = {"H_PI": h.H_PI, "H_obs": h.H_obs, "eta": h.eta}
        out["qualifies_new_lane"] = bool(h.H_obs >= 0.01 and h.eta >= 0.30)
        print(json.dumps({"located": out["gp_locate_H_obs"]["theta"],
                          "headroom": out["located_region_headroom"],
                          "qualifies_new_lane": out["qualifies_new_lane"]}, indent=2))

    output = Path("results/headroom_gsa"); output.mkdir(parents=True, exist_ok=True)
    (output / f"verdict_{a.stage}.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
