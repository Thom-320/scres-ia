"""Inner-bias study for the BSS bound: U_pen(M) averaged over all disjoint replicate groups of size k
(M = 8k), then extrapolation to M -> infinity under two bias laws (a/M and a/sqrt(M)).

The finite-M bound is a valid (looser) upper bound for every M: the noisy penalty is still mean zero
given the history, and the max of a noisy objective only adds upward bias (Jensen). The extrapolated
value is an *estimate* of the exact-penalty bound, reported as a diagnostic, not as the certified bound.

Usage: python bss_extrap.py INNER_DIR NREPS OUT_JSON [SEED_FROM-SEED_TO]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bss_bound as BB  # noqa: E402

inner, nrep, outp = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
a, b = (map(int, sys.argv[4].split("-")) if len(sys.argv) > 4 else (7480001, 7480048))
seeds = list(range(a, b + 1))
cache = np.load(BB.B.REPO / "results/program_q/power_preopen_v5_20260718/artifacts/classical_10_cache_v1.npz")
ks = [k for k in (1, 2, 4, 8) if k <= nrep]
out = {"nreps": nrep, "seeds": [a, b], "cells": {}}
for cell in BB.CELLS:
    all_seeds = cache[f"{cell}__tape_seeds"].tolist()
    CL = np.asarray([float(np.load(BB.COTA / "frontier" / f"{cell}__tape_{s}.npz")["ret_visible"][
        int(cache[f"{cell}__calendar_indices"][BB.BEST_K[cell], all_seeds.index(s)])]) for s in seeds])
    row = {}
    for k in ks:
        groups = [list(range(g * k, (g + 1) * k)) for g in range(nrep // k)]
        U = np.asarray([[BB.dp_max(BB.advantages(*BB.load_avg(inner, cell, s, 8, grp))) for s in seeds]
                        for grp in groups])  # groups x tapes
        tape_means = U.mean(0)
        se_inner = (float(U.mean(1).std(ddof=1) / np.sqrt(len(groups))) if len(groups) > 1 else None)
        row[f"M{8 * k}"] = {"n_groups": len(groups), "U_pen": float(tape_means.mean()),
                            "gap": float((tape_means - CL).mean()), "inner_se_of_group_avg": se_inner}
    Ms = np.asarray([8.0 * k for k in ks])
    Us = np.asarray([row[f"M{8 * k}"]["U_pen"] for k in ks])
    fits = {}
    for name, beta in (("inv_M", 1.0), ("inv_sqrtM", 0.5)):
        X = np.c_[np.ones_like(Ms), Ms ** -beta]
        coef, res, *_ = np.linalg.lstsq(X, Us, rcond=None)
        fits[name] = {"U_inf": float(coef[0]), "a": float(coef[1]),
                      "bias_at_max_M": float(coef[1] * Ms.max() ** -beta),
                      "gap_inf": float(coef[0] - CL.mean()),
                      "rss": float(np.sum((X @ coef - Us) ** 2))}
    if len(Ms) >= 3:  # free exponent: U = c + a M^-beta via grid on beta
        best = None
        for beta in np.linspace(0.2, 2.0, 181):
            X = np.c_[np.ones_like(Ms), Ms ** -beta]
            coef, *_ = np.linalg.lstsq(X, Us, rcond=None)
            r = float(np.sum((X @ coef - Us) ** 2))
            if best is None or r < best[0]:
                best = (r, beta, coef)
        fits["free_beta"] = {"beta": float(best[1]), "U_inf": float(best[2][0]),
                             "bias_at_max_M": float(best[2][1] * Ms.max() ** -best[1]),
                             "gap_inf": float(best[2][0] - CL.mean()), "rss": best[0]}
    row["V_CL"] = float(CL.mean())
    row["extrapolation"] = fits
    out["cells"][cell] = row
    print(cell, json.dumps(row))
outp.write_text(json.dumps(out, indent=1))
