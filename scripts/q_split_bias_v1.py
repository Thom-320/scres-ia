#!/usr/bin/env python3
"""Selection/evaluation split over the sealed Program Q panels — q_split_bias_v1.

Implements docs/PREREGISTRO_SPLIT_PROGRAM_Q_2026-08-27.md (sha256 1821af8b...).
Read-only: no simulation, no seed opened, nothing trained.
"""
from __future__ import annotations

import hashlib
import importlib.util as iu
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SESOI = 0.01
B = 10_000
KFOLD = 8
RNG = np.random.default_rng(20260827)
OUT = Path("results/program_n/q_split_bias_v1/result.json")
CEILING = Path("results/program_n/control_ceiling_v1/result.json")

spec = iu.spec_from_file_location("rd", "scripts/paper_prep/ret_decomposition.py")
rd = iu.module_from_spec(spec)
spec.loader.exec_module(rd)
ROOT = rd.SHARDS_ROOT

evaluation = json.loads((ROOT.parent / "evaluation/result.json").read_text())
estimates = evaluation["inference"]["estimates"]
ceiling_prev = json.loads(CEILING.read_text())["cells"]

CELLS = ("rho75_share90", "rho90_share75", "rho90_share90")


def load(cell):
    ol, cl, lr, seeds = [], [], [], []
    for f in sorted((ROOT / cell).glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        ol.append(np.atleast_2d(np.asarray(z["open_loop__ret_visible"])))
        cl.append(np.atleast_2d(np.asarray(z["classical__ret_visible"])))
        lr.append(np.atleast_2d(np.asarray(z["learner__ret_visible"])))
        n = ol[-1].shape[0]
        key = "tape_seeds" if "tape_seeds" in z else None
        seeds.extend(list(np.asarray(z[key]).ravel()) if key else [f.stem] * n)
    return (np.concatenate(ol), np.concatenate(cl), np.concatenate(lr), seeds)


def best_classical_mean(CL, idx):
    """max over configs of the mean on idx; the argmax is chosen ON idx."""
    means = CL[idx].mean(axis=0)
    c = int(np.argmax(means))
    return float(means[c]), c


out = {
    "schema_version": "q_split_bias_v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "contract_path": "docs/PREREGISTRO_SPLIT_PROGRAM_Q_2026-08-27.md",
    "contract_sha256": "1821af8b2d31419ac5160af5843c3a7c849300d6c2368d5778ce3cca0b476f5a",
    "scope": "READ_ONLY_ON_CONSUMED_PANELS_NO_SEEDS_NO_TRAINING",
    "sesoi": SESOI,
    "cells": {},
}
f1_err = 0.0
f2_rows, f3_rows, f4_rows = {}, {}, {}

for cell in CELLS:
    OL, CL, LR, seeds = load(cell)
    n = OL.shape[0]

    # Deterministic A/B split by sha256 of the tape key: fixed by the seed, not
    # by any outcome.
    order = sorted(range(n), key=lambda i: hashlib.sha256(
        str(seeds[i]).encode()).hexdigest())
    A = np.array(order[: n // 2])
    Bx = np.array(order[n // 2:])

    k_star_A = int(np.argmax(OL[A].mean(axis=0)))
    k_star_all = int(np.argmax(OL.mean(axis=0)))

    cl_A, c_A = best_classical_mean(CL, A)
    cl_B, c_B = best_classical_mean(CL, Bx)

    g_naive = float(OL[A].max(axis=1).mean() - cl_A)
    g_split = float(OL[Bx, k_star_A].mean() - cl_B)
    delta_bias = g_naive - g_split

    # K-fold cross-fitted G_split: select on K-1 folds, evaluate on the held-out
    folds = [np.array(order[i::KFOLD]) for i in range(KFOLD)]
    cf = []
    for i, te in enumerate(folds):
        tr = np.concatenate([f for j, f in enumerate(folds) if j != i])
        k_tr = int(np.argmax(OL[tr].mean(axis=0)))
        cl_te, _ = best_classical_mean(CL, te)
        cf.append(OL[te, k_tr].mean() - cl_te)
    g_split_cf = float(np.mean(cf))

    boot_split = np.empty(B)
    boot_bias = np.empty(B)
    for b in range(B):
        ia = A[RNG.integers(0, len(A), len(A))]
        ib = Bx[RNG.integers(0, len(Bx), len(Bx))]
        k = int(np.argmax(OL[ia].mean(axis=0)))
        cla, _ = best_classical_mean(CL, ia)
        clb, _ = best_classical_mean(CL, ib)
        gs = OL[ib, k].mean() - clb
        gn = OL[ia].max(axis=1).mean() - cla
        boot_split[b] = gs
        boot_bias[b] = gn - gs
    s_lo, s_hi = np.percentile(boot_split, [2.5, 97.5])
    b_lo, b_hi = np.percentile(boot_bias, [2.5, 97.5])

    # F1 anchor
    cl_all, c_all = best_classical_mean(CL, np.arange(n))
    d_here = float(LR.mean() - cl_all)
    d_sealed = float(estimates[f"{cell}::Delta_N"]["point"])
    f1_err = max(f1_err, abs(d_here - d_sealed))

    f2_rows[cell] = {"cl_A": cl_A, "cl_B": cl_B, "abs_diff": abs(cl_A - cl_B),
                     "argmax_c_A": c_A, "argmax_c_B": c_B, "same_argmax": c_A == c_B}
    f3_rows[cell] = {"g_split": g_split, "g_split_crossfit": g_split_cf,
                     "abs_diff": abs(g_split - g_split_cf)}
    f4_rows[cell] = {"k_star_A": k_star_A, "k_star_all": k_star_all,
                     "differs": k_star_A != k_star_all}

    ceil_prev = ceiling_prev[cell]["techo_clarividente"]
    out["cells"][cell] = {
        "n_tapes": n, "n_A": len(A), "n_B": len(Bx),
        "k_star_A": k_star_A,
        "G_naive": g_naive,
        "G_split": g_split, "G_split_ci95": [float(s_lo), float(s_hi)],
        "G_split_crossfit_k8": g_split_cf,
        "delta_bias": delta_bias, "delta_bias_ci95": [float(b_lo), float(b_hi)],
        "ceiling_from_control_ceiling_v1": ceil_prev,
        "bias_covers_ceiling": bool(delta_bias >= ceil_prev),
    }

out["falsifiers"] = {
    "f1_anchor_reproduces_sealed_delta_n": {
        "passed": bool(f1_err <= 1e-9), "max_abs_error": f1_err, "tolerance": 1e-9},
    "f2_partition_is_balanced": {
        "passed": bool(all(r["abs_diff"] < 0.02 and r["same_argmax"]
                           for r in f2_rows.values())), "per_cell": f2_rows},
    "f3_crossfit_agrees_with_single_split": {
        "passed": bool(all(r["abs_diff"] < 0.02 for r in f3_rows.values())),
        "per_cell": f3_rows},
    "f4_k_star_A_is_not_the_global_argmax": {
        "passed": bool(all(r["differs"] for r in f4_rows.values())),
        "per_cell": f4_rows,
        "note": "informative either way; a match would mean the split did not "
                "actually separate selection from evaluation"},
}

cells = out["cells"]
if all(c["bias_covers_ceiling"] for c in cells.values()) and \
        all(c["G_split_ci95"][1] < SESOI for c in cells.values()):
    verdict = "CEILING_IS_BIAS"
elif all(c["G_split_ci95"][0] > SESOI for c in cells.values()):
    verdict = "RESIDUAL_ROOM"
else:
    verdict = "UNDETERMINED"
out["verdict"] = verdict
out["scope_note"] = (
    "G_split bounds picking a FIXED calendar from other tapes' data. It does NOT "
    "bound a state-conditioned policy: the learner beats the best fixed calendar "
    "by H_OL +0.0795/+0.0725/+0.1172 while calendar selection loses to the best "
    "classical.")

OUT.parent.mkdir(parents=True, exist_ok=True)
payload = json.dumps(out, indent=2, sort_keys=True)
OUT.write_text(payload + "\n")
(OUT.parent / "RESULT.sha256").write_text(
    hashlib.sha256((payload + "\n").encode()).hexdigest() + "  result.json\n")

for k, v in out["falsifiers"].items():
    print(f"{k}: passed={v['passed']}")
print()
for cell, o in cells.items():
    print(f"=== {cell} ===  A={o['n_A']} B={o['n_B']}")
    print(f"  G_naive   {o['G_naive']:+.6f}")
    print(f"  G_split   {o['G_split']:+.6f}  CI95 [{o['G_split_ci95'][0]:+.6f}, "
          f"{o['G_split_ci95'][1]:+.6f}]   crossfit K8 {o['G_split_crossfit_k8']:+.6f}")
    print(f"  Δ_bias    {o['delta_bias']:+.6f}  CI95 [{o['delta_bias_ci95'][0]:+.6f}, "
          f"{o['delta_bias_ci95'][1]:+.6f}]")
    print(f"  techo previo {o['ceiling_from_control_ceiling_v1']:+.6f}  "
          f"-> el sesgo lo cubre: {o['bias_covers_ceiling']}")
    print()
print(f"VEREDICTO: {verdict}")
print(f"-> {OUT}")
