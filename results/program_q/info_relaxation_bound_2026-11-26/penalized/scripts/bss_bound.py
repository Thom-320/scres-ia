"""Assemble the BSS penalized bound from inner replicates.

J(c, w) = sum_t [ Q_t(c_0..c_t) - V_t(c_0..c_{t-1}) ]   (telescoped R - sum_t z_t, with V_8 = R exactly)
  Q_t(p_{t+1}) = Fbar_t[p_{t+1}],  V_t(p_t) = Fbar_t[(p_t, kappa_t(p_t))],  V_0 := 0.
U_pen(w) = max_c J(c, w), by exact backward DP over the 4-ary tree (equivalent to enumerating the 65,536 calendars).
Zero penalty check: U0 = mean_w max_c R(c, w) recomputed from the outer frontier.

Usage: python bss_bound.py INNER_DIR M REPS(comma list or a-b) OUT_JSON CELL:SEED_FROM-SEED_TO [...]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bss_lib as B  # noqa: E402

# Zero-penalty frontier tapes (npz, not committed; see zero_penalty/frontier_manifest.sha256).
COTA = Path(os.environ.get("SCRES_COTA_DIR", Path(__file__).resolve().parents[2] / "zero_penalty"))
CELLS = ["rho75_share90", "rho90_share75", "rho90_share90"]
BEST_K = {"rho75_share90": 2, "rho90_share75": 5, "rho90_share90": 2}


def parse_reps(s):
    if "-" in s:
        a, b = map(int, s.split("-"))
        return list(range(a, b + 1))
    return [int(x) for x in s.split(",")]


def load_avg(inner, cell, seed, M, reps):
    zs = [np.load(Path(inner) / f"{cell}__tape_{seed}__M{M}__rep{r}.npz") for r in reps]
    F = [np.mean([z[f"Fbar_{t}"] for z in zs], 0) for t in range(8)]
    K = [zs[0][f"kappa_{t}"].astype(np.int64) for t in range(8)]
    for z in zs[1:]:
        assert all(np.array_equal(z[f"kappa_{t}"], K[t]) for t in range(8))
    return F, K


def advantages(F, K):
    A = [F[0].reshape(1, 4)]
    for t in range(1, 8):
        Ft = F[t].reshape(4 ** t, 4)
        A.append(Ft - Ft[np.arange(4 ** t), K[t]][:, None])
    return A


def dp_max(A):
    G = np.zeros(4 ** 8)
    for t in range(7, -1, -1):
        G = (A[t] + G.reshape(4 ** t, 4)).max(1)
    return float(G[0])


def J_of(A, cal):
    v, idx = 0.0, 0
    for t in range(8):
        v += A[t][idx, int(cal[t])]
        idx = idx * 4 + int(cal[t])
    return float(v)


def boot_ci(x, B_=20000, seed=7):
    rng = np.random.default_rng(seed)
    bt = rng.integers(0, len(x), size=(B_, len(x)))
    m = x[bt].mean(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def main():
    inner, M, reps, outp = sys.argv[1], int(sys.argv[2]), parse_reps(sys.argv[3]), Path(sys.argv[4])
    cache = np.load(B.REPO / "results/program_q/power_preopen_v5_20260718/artifacts/classical_10_cache_v1.npz")
    calres = json.loads((B.REPO / "results/program_o/ret_only_learner_v1/calibration_run/result.json").read_text())
    rng = np.random.default_rng(99)
    fixed = [tuple(rng.integers(0, 4, 8).tolist()) for _ in range(5)]
    out = {"M_per_replicate": M, "replicates": reps, "M_total": M * len(reps), "cells": {}}
    for spec in sys.argv[5:]:
        ci, rs = spec.split(":")
        cell = CELLS[int(ci)]
        a, b = map(int, rs.split("-"))
        seeds = list(range(a, b + 1))
        all_seeds = cache[f"{cell}__tape_seeds"].tolist()
        U, CL, U0, Jcl, stage = [], [], [], [], []
        mz = {"best_open_loop": [], **{f"fixed_{i}": [] for i in range(5)},
              **{f"classical_k{k}": [] for k in range(10)}, "learner_mean_10_seeds": []}
        bol = calres["cell_summaries"][cell]["best_open_loop_index"]
        aud = calres["trajectory_audits"][cell]
        for s in seeds:
            ti = all_seeds.index(s)
            R = np.load(COTA / "frontier" / f"{cell}__tape_{s}.npz")["ret_visible"]
            F, K = load_avg(inner, cell, s, M, reps)
            A = advantages(F, K)
            u = dp_max(A)
            kcal_idx = int(cache[f"{cell}__calendar_indices"][BEST_K[cell], ti])
            kcal = [(kcal_idx >> (2 * (7 - t))) & 3 for t in range(8)]
            U.append(u); CL.append(float(R[kcal_idx])); U0.append(float(R.max()))
            Jcl.append(J_of(A, kcal))
            stage.append([float(np.abs(A[t]).max()) for t in range(8)])
            cal_bol = [(bol >> (2 * (7 - t))) & 3 for t in range(8)]
            mz["best_open_loop"].append(float(R[bol]) - J_of(A, cal_bol))
            for i, c in enumerate(fixed):
                mz[f"fixed_{i}"].append(float(R[B.enc(c)]) - J_of(A, c))
            for k in range(10):
                idx = int(cache[f"{cell}__calendar_indices"][k, ti])
                ck = [(idx >> (2 * (7 - t))) & 3 for t in range(8)]
                mz[f"classical_k{k}"].append(float(R[idx]) - J_of(A, ck))
            mz["learner_mean_10_seeds"].append(float(np.mean(
                [float(R[B.enc(aud[sd]["calendars"][ti])]) - J_of(A, aud[sd]["calendars"][ti]) for sd in aud])))
        U, CL, U0, Jcl = map(np.asarray, (U, CL, U0, Jcl))
        g, g0 = U - CL, U0 - CL
        T = len(seeds)
        res = {"seeds": [a, b], "n_tapes": T,
               "U0": float(U0.mean()), "U0_se": float(U0.std(ddof=1) / np.sqrt(T)),
               "U_pen": float(U.mean()), "U_pen_se": float(U.std(ddof=1) / np.sqrt(T)),
               "V_CL": float(CL.mean()), "V_CL_se": float(CL.std(ddof=1) / np.sqrt(T)),
               "U_pen_minus_V_CL": float(g.mean()), "U_pen_minus_V_CL_se": float(g.std(ddof=1) / np.sqrt(T)),
               "U_pen_minus_V_CL_ci95": boot_ci(g),
               "U0_minus_V_CL_same_tapes": float(g0.mean()), "U0_minus_V_CL_ci95_same_tapes": boot_ci(g0),
               "gap_ratio_pen_over_zero": float(g.mean() / g0.mean()),
               # control-variate estimator: E[J(kappa)] = V_CL exactly (mean-zero penalty along kappa),
               # so U_pen - J(kappa) is unbiased for U_pen - V_CL and does not carry the outer noise of R(kappa).
               "gap_cv_U_pen_minus_J_kappa": float((U - Jcl).mean()),
               "gap_cv_se": float((U - Jcl).std(ddof=1) / np.sqrt(T)),
               "gap_cv_ci95": boot_ci(U - Jcl),
               "sanity_J_at_classical_mean": float(Jcl.mean()),
               "sanity_J_at_classical_minus_V_CL": float((Jcl - CL).mean()),
               "sanity_J_at_classical_minus_V_CL_se": float((Jcl - CL).std(ddof=1) / np.sqrt(T)),
               "sanity_U_pen_ge_J_classical_all_tapes": bool(np.all(U >= Jcl - 1e-12)),
               "mean_zero_tests": {k: {"mean_R_minus_J": float(np.mean(v)),
                                       "se": float(np.std(v, ddof=1) / np.sqrt(T)),
                                       "z": float(np.mean(v) / (np.std(v, ddof=1) / np.sqrt(T)))}
                                   for k, v in mz.items()},
               "max_abs_advantage_by_stage_mean": np.mean(stage, 0).tolist(),
               "per_tape": {"U_pen": U.tolist(), "V_CL": CL.tolist(), "U0": U0.tolist(), "J_kappa": Jcl.tolist()}}
        out["cells"][cell] = res
        print(cell, json.dumps({k: v for k, v in res.items() if k not in ("per_tape", "mean_zero_tests")}))
        print("  mean-zero z:", {k: round(v["z"], 2) for k, v in res["mean_zero_tests"].items()})
    outp.write_text(json.dumps(out, indent=1))
    print("wrote", outp)


if __name__ == "__main__":
    main()
