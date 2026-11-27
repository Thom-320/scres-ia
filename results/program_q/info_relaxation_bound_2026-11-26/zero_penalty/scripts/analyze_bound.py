"""Perfect-information (zero-penalty info relaxation) upper bound on burned calibration tapes.
U0 = mean_t max_c R(c,t); V_OL = max_c mean_t R; V_CL = max_k mean_t R(cal_k(t),t) over 10 classical configs.
Usage: python analyze_bound.py REPO FRONTIER_DIR OUT_JSON"""
import json, sys, hashlib
from pathlib import Path
import numpy as np
repo, fdir, outp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
CELLS = ["rho75_share90", "rho90_share75", "rho90_share90"]
SEEDS = list(range(7480001, 7480049))
cache_p = repo / "results/program_q/power_preopen_v5_20260718/artifacts/classical_10_cache_v1.npz"
cache = np.load(cache_p)
cal = json.loads((repo / "results/program_o/ret_only_learner_v1/calibration_run/result.json").read_text())
sys.path.insert(0, str(repo))
from supply_chain.program_o_state_rich import finite_state_rich_configurations
names = [getattr(c, "name", None) or getattr(c, "config_id", None) or str(i) for i, c in enumerate(finite_state_rich_configurations())]
def enc(c):
    i = 0
    for a in c: i = i * 4 + int(a)
    return i
rng = np.random.default_rng(20261126); B = 20000
out = {"cache_sha256": hashlib.sha256(cache_p.read_bytes()).hexdigest(), "B": B, "cells": {}}
for cell in CELLS:
    R = np.stack([np.load(fdir / f"{cell}__tape_{s}.npz")["ret_visible"] for s in SEEDS])  # (48, 65536)
    my_idx = np.stack([np.load(fdir / f"{cell}__tape_{s}.npz")["classical_indices"] for s in SEEDS], 1)  # (10,48)
    assert cache[f"{cell}__tape_seeds"].tolist() == SEEDS
    cidx = cache[f"{cell}__calendar_indices"]
    T = len(SEEDS)
    pi = R.max(1)                                   # per-tape perfect-information value
    ol_means = R.mean(0); c_ol = int(ol_means.argmax()); ol = R[:, c_ol]
    cl_all = np.stack([R[np.arange(T), cidx[k]] for k in range(cidx.shape[0])])  # (10,48)
    k_cl = int(cl_all.mean(1).argmax()); cl = cl_all[k_cl]
    aud = cal["trajectory_audits"][cell]
    learner = np.stack([[R[t, enc(aud[sd]["calendars"][t])] for t in range(T)] for sd in sorted(aud, key=int)])
    lm = learner.mean(0)  # per-tape mean over learner seeds
    # paired bootstrap over tapes; V_OL and V_CL re-maximized inside each resample (selection-aware)
    bt = rng.integers(0, T, size=(B, T))
    U0_b = pi[bt].mean(1)
    OL_b = np.stack([R[b].mean(0).max() for b in bt[:2000]])  # 2000 for speed (65536-wide argmax)
    CL_b = cl_all[:, bt].mean(2).max(0)
    g_cl_b = U0_b - CL_b; g_ol_b = U0_b[:2000] - OL_b
    g_cl_fixed = pi - cl; g_ol_fixed = pi - ol
    def ci(x): return [float(np.quantile(x, .025)), float(np.quantile(x, .975))]
    res = {
        "n_tapes": T,
        "U0": float(pi.mean()), "U0_se": float(pi.std(ddof=1) / np.sqrt(T)),
        "V_OL": float(ol.mean()), "best_open_loop_index": c_ol,
        "best_open_loop_index_calibration_result": cal["cell_summaries"][cell]["best_open_loop_index"],
        "V_CL": float(cl.mean()), "best_classical_k": k_cl, "best_classical_name": names[k_cl],
        "best_classical_calibration_result": cal["cell_summaries"][cell]["best_classical_config"],
        "classical_means": {names[k]: float(cl_all[k].mean()) for k in range(len(names))},
        "learner_mean_over_10_seeds": float(lm.mean()),
        "learner_minus_V_OL": float(lm.mean() - ol.mean()), "learner_minus_V_CL": float(lm.mean() - cl.mean()),
        "U0_minus_V_CL": float(g_cl_fixed.mean()), "U0_minus_V_CL_se_paired": float(g_cl_fixed.std(ddof=1) / np.sqrt(T)),
        "U0_minus_V_CL_ci95_paired_bootstrap_reselect": ci(g_cl_b),
        "U0_minus_V_CL_lcb95_onesided": float(np.quantile(g_cl_b, .05)),
        "U0_minus_V_OL": float(g_ol_fixed.mean()), "U0_minus_V_OL_ci95_paired_bootstrap_reselect": ci(g_ol_b),
        "U0_minus_learner": float((pi - lm).mean()),
        "tapes_where_classical_attains_pi": int(np.sum(np.isclose(cl, pi, atol=1e-12))),
        "per_tape_gap_pi_minus_cl_quantiles": [float(q) for q in np.quantile(g_cl_fixed, [0, .25, .5, .75, 1])],
        "classical_indices_recomputed_match_cache": bool(np.array_equal(my_idx, cidx)),
        "classical_indices_mismatch_count": int((my_idx != cidx).sum()),
        "n_distinct_pi_argmax": int(len(set(R.argmax(1).tolist()))),
    }
    out["cells"][cell] = res
    print(cell, json.dumps({k: v for k, v in res.items() if k != "classical_means"}))
outp.write_text(json.dumps(out, indent=2)); print("wrote", outp)
