#!/usr/bin/env python3
"""Clairvoyant ceiling of the control lane — control_ceiling_v1.

Implements docs/PREREGISTRO_TECHO_CLARIVIDENTE_CONTROL_2026-08-27.md.

Read-only over the sealed Program Q confirmation panels. No simulation, no seeds
opened, nothing trained. Can only CLOSE a lane, never open one.
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
RNG = np.random.default_rng(20260827)
OUT = Path("results/program_n/control_ceiling_v1/result.json")

spec = iu.spec_from_file_location("rd", "scripts/paper_prep/ret_decomposition.py")
rd = iu.module_from_spec(spec)
spec.loader.exec_module(rd)
ROOT = rd.SHARDS_ROOT

evaluation = json.loads((ROOT.parent / "evaluation/result.json").read_text())
estimates = evaluation["inference"]["estimates"]

CELLS = ("rho75_share90", "rho90_share75", "rho90_share90")
out: dict = {
    "schema_version": "control_ceiling_v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "contract_path": "docs/PREREGISTRO_TECHO_CLARIVIDENTE_CONTROL_2026-08-27.md",
    "sesoi": SESOI,
    "endpoint": "ret_visible",
    "scope": "READ_ONLY_ON_CONSUMED_PANELS_NO_SEEDS_NO_TRAINING",
    "cells": {},
    "falsifiers": {},
}

f1_max_err = 0.0
f2_distinct = {}

for cell in CELLS:
    shards = sorted((ROOT / cell).glob("*.npz"))
    ol, cl, lr = [], [], []
    for f in shards:
        z = np.load(f, allow_pickle=True)
        ol.append(np.atleast_2d(np.asarray(z["open_loop__ret_visible"])))
        cl.append(np.atleast_2d(np.asarray(z["classical__ret_visible"])))
        lr.append(np.atleast_2d(np.asarray(z["learner__ret_visible"])))
    OL = np.concatenate(ol, axis=0)          # (tapes, 65536)
    CL = np.concatenate(cl, axis=0)          # (tapes, 10)
    LR = np.concatenate(lr, axis=0)          # (tapes, 10 seeds)
    n_tapes = OL.shape[0]

    per_tape_max = OL.max(axis=1)                    # clairvoyant, per tape
    best_fixed_k = int(np.argmax(OL.mean(axis=0)))
    best_cl_c = int(np.argmax(CL.mean(axis=0)))

    ceiling = float(per_tape_max.mean() - CL[:, best_cl_c].mean())
    room = float(per_tape_max.mean() - LR.mean())
    fixed_gap = float(OL[:, best_fixed_k].mean() - CL[:, best_cl_c].mean())

    # F1: reproduce the sealed Delta_N from these same panels
    delta_n_sealed = float(estimates[f"{cell}::Delta_N"]["point"])
    delta_n_here = float(LR.mean() - CL[:, best_cl_c].mean())
    f1_max_err = max(f1_max_err, abs(delta_n_here - delta_n_sealed))

    # F2: does the clairvoyant argmax vary across tapes?
    argmaxes = OL.argmax(axis=1)
    f2_distinct[cell] = int(len(np.unique(argmaxes)))

    boot_ceiling = np.empty(B)
    boot_room = np.empty(B)
    for b in range(B):
        idx = RNG.integers(0, n_tapes, n_tapes)
        pm = per_tape_max[idx].mean()
        boot_ceiling[b] = pm - CL[idx][:, best_cl_c].mean()
        boot_room[b] = pm - LR[idx].mean()

    c_lo, c_hi = np.percentile(boot_ceiling, [2.5, 97.5])
    r_lo, r_hi = np.percentile(boot_room, [2.5, 97.5])

    out["cells"][cell] = {
        "n_tapes": n_tapes,
        "n_calendars": int(OL.shape[1]),
        "clairvoyant_mean": float(per_tape_max.mean()),
        "best_fixed_mean": float(OL[:, best_fixed_k].mean()),
        "best_classical_mean": float(CL[:, best_cl_c].mean()),
        "learner_mean": float(LR.mean()),
        "techo_clarividente": ceiling,
        "techo_ci95": [float(c_lo), float(c_hi)],
        "margen_sobre_aprendiz": room,
        "margen_ci95": [float(r_lo), float(r_hi)],
        "brecha_fijo": fixed_gap,
        "personalisation_component": ceiling - fixed_gap,
        "distinct_argmax_over_tapes": f2_distinct[cell],
        "delta_n_recomputed": delta_n_here,
        "delta_n_sealed": delta_n_sealed,
    }

out["falsifiers"] = {
    "f1_anchors_reproduce_sealed_delta_n": {
        "passed": bool(f1_max_err <= 1e-9),
        "max_abs_error": f1_max_err,
        "tolerance": 1e-9,
        "why_it_can_fail": "if the panel I read is not the one that produced the "
                           "sealed verdict, I am not measuring this lane's ceiling",
    },
    "f2_clairvoyant_argmax_varies": {
        "passed": bool(all(v > 1 for v in f2_distinct.values())),
        "distinct_argmax_by_cell": f2_distinct,
        "why_it_can_fail": "if every tape picks the same calendar, clairvoyance "
                           "buys nothing beyond the best fixed calendar and the "
                           "ceiling collapses to brecha_fijo",
    },
    "f3_inflation_disclosed": {
        "computed": True,
        "note": "mean_t[max_k] is an IN-SAMPLE maximum over 65,536 calendars and "
                "is therefore inflated. Gate-0 measured that inflation in this same "
                "physics at Delta_bias +0.119 to +0.176. This is a disclosure, not "
                "a test: it cannot fail.",
        "gate0_delta_bias_reference": [0.11855, 0.17552],
    },
}

# Preregistered decision rule
closed = [c for c, o in out["cells"].items() if o["techo_ci95"][1] < SESOI]
at_ceiling = [c for c, o in out["cells"].items() if o["margen_ci95"][1] < SESOI]
if closed:
    verdict = "CLOSED_BY_ARITHMETIC"
elif at_ceiling:
    verdict = "LEARNER_AT_CEILING"
else:
    verdict = "NOT_CLOSED"
out["verdict"] = verdict
out["cells_closed"] = closed
out["cells_learner_at_ceiling"] = at_ceiling
out["verdict_note"] = (
    "NOT_CLOSED is NOT a positive: it means an inflated upper bound was not tight "
    "enough to close the lane, and deciding requires the Gate-0 style "
    "selection/evaluation split."
)

OUT.parent.mkdir(parents=True, exist_ok=True)
payload = json.dumps(out, indent=2, sort_keys=True)
OUT.write_text(payload + "\n")
(OUT.parent / "RESULT.sha256").write_text(
    hashlib.sha256((payload + "\n").encode()).hexdigest() + "  result.json\n")

print(f"F1 anclas reproducen Delta_N sellado: {out['falsifiers']['f1_anchors_reproduce_sealed_delta_n']['passed']} "
      f"(max err {f1_max_err:.3e})")
print(f"F2 argmax clarividente varia: {f2_distinct}")
print()
for cell, o in out["cells"].items():
    print(f"=== {cell} ===  {o['n_tapes']} tapas x {o['n_calendars']} calendarios")
    print(f"  clarividente por tapa     {o['clairvoyant_mean']:.6f}")
    print(f"  mejor calendario fijo     {o['best_fixed_mean']:.6f}")
    print(f"  mejor clasico             {o['best_classical_mean']:.6f}")
    print(f"  aprendiz (RecurrentPPO)   {o['learner_mean']:.6f}")
    print(f"  TECHO sobre clasico       {o['techo_clarividente']:+.6f}  "
          f"CI95 [{o['techo_ci95'][0]:+.6f}, {o['techo_ci95'][1]:+.6f}]")
    print(f"  margen sobre aprendiz     {o['margen_sobre_aprendiz']:+.6f}  "
          f"CI95 [{o['margen_ci95'][0]:+.6f}, {o['margen_ci95'][1]:+.6f}]")
    print(f"  de eso, personalizacion   {o['personalisation_component']:+.6f}")
    print()
print(f"VEREDICTO: {verdict}   cerradas={closed}  aprendiz-en-techo={at_ceiling}")
print(f"-> {OUT}")
