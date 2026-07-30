#!/usr/bin/env python3
"""The order-stock matching sweep, per PREREGISTRO_EMPAREJAMIENTO_ORDEN_STOCK_2026-07-30.

Three binary axes, eight cells per family. No cell may be selected for the result it
produces; the output is the non-dominated set over six moments plus the declared
shape-acceptance flags, both reported for every cell.

The acceptance criteria are about the SHAPE of the CTj distribution, not one moment,
and they were declared before this ran:

    mass at the minimum   < 0.30      (today 0.641)
    CTj p50               within 1.5x of the reference
    CTj p95               within 1.5x of the reference
    rpj_mean              no worse than the current default

The falsifier is also declared: a cell that improves the median while worsening the tail
in the same proportion is redistributing, not correcting.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import (  # noqa: E402
    compute_order_level_ret_excel_request_snapshot_ledger as ledger,
)
from supply_chain.fidelity_moments import (  # noqa: E402
    EPSILON,
    MOMENT_NAMES,
    MomentReference,
    discrepancies,
    epsilon_stability,
    moments_from_rows,
    non_dominated,
)
from supply_chain.provenance import calibration_stamp  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

FAMILIES = {"R1r": ("R11", "R12", "R13", "R14"), "R2r": ("R21", "R22", "R23", "R24")}
ROOTS = tuple(2_200_001 + i for i in range(12))
EPSILONS = (0.25, 0.5, 1.0, 2.0)
# Reference CTj shape, from the canonical workbooks (R1r sheets).
REF_CT_P50 = 101.4
REF_CT_P95 = 2238.6
MASS_AT_MIN_MAX = 0.30
SHAPE_FACTOR = 1.5


def run(*, family, seed, horizon, partial, blocking, fulfil):
    sim = MFSCSimulation(
        shifts=1, initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
        risks_enabled=True, risk_level="current", enabled_risks=set(FAMILIES[family]),
        risk_overrides={r: "increased" for r in FAMILIES[family]},
        partial_fulfilment=partial, queue_blocking=blocking,
        order_fulfillment_mode=fulfil,
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    sim.step(action=None, step_hours=horizon)
    o = [x for x in sim.orders
         if not bool(getattr(x, "metrics_excluded", False))
         and float(getattr(x, "OPTj", 0.0)) >= float(sim.warmup_time)]
    book = ledger(o, current_time=float(sim.env.now))
    ct = np.array([float(x.CTj) for x in o if x.CTj is not None])
    return o, book, ct


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", type=int, default=list(ROOTS))
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--reference", type=Path,
                    default=Path("results/metric_audit/fidelity_reference_v2/result.json"))
    ap.add_argument("--output", type=Path,
                    default=Path("results/metric_audit/matching_sweep_v1/result.json"))
    args = ap.parse_args()

    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    hy = float(args.horizon_weeks) / 52.0
    ref_blob = json.loads(args.reference.read_text())
    reference = {fam: {m: MomentReference(**{k: v[k] for k in ("mean", "spread", "n_sheets")})
                       for m, v in blk.items()}
                 for fam, blk in ref_blob["reference_by_family"].items()}
    started = time.perf_counter()
    out = {}

    for family in ("R1r", "R2r"):
        cells = {}
        base_rpj = None
        for partial, blocking, fulfil in product(
                (False, True), ("blocking", "skip_head"),
                ("legacy_theatre_stock", "op9_linked")):
            per_root, cts = [], []
            for seed in args.roots:
                o, book, ct = run(family=family, seed=seed, horizon=horizon,
                                  partial=partial, blocking=blocking, fulfil=fulfil)
                per_root.append(moments_from_rows(
                    apj=[float(getattr(x, "APj", 0.0) or 0.0) for x in o],
                    rpj=[float(getattr(x, "RPj", 0.0) or 0.0) for x in o],
                    ret=[float(v) for v in book["ret_values"]], horizon_years=hy))
                cts.append(ct)
            allct = np.concatenate([c for c in cts if len(c)]) if any(len(c) for c in cts) else np.array([0.0])
            mean = {m: float(np.mean([r[m] for r in per_root])) for m in MOMENT_NAMES}
            se = {m: float(np.std([r[m] for r in per_root], ddof=1) / np.sqrt(len(per_root)))
                  for m in MOMENT_NAMES}
            name = f"partial{int(partial)}|{blocking}|{fulfil}"
            if name == "partial0|blocking|legacy_theatre_stock":
                base_rpj = mean["rpj_mean"]
            p50, p95 = float(np.median(allct)), float(np.percentile(allct, 95))
            mn = float(allct.min())
            cells[name] = {
                "partial_fulfilment": partial, "queue_blocking": blocking,
                "order_fulfillment_mode": fulfil,
                "moments": mean, "moment_se": se,
                "discrepancies": discrepancies(mean, se, reference[family]),
                "ct_p50": p50, "ct_p95": p95, "ct_min": mn,
                "mass_at_min": float((allct == mn).mean()),
            }
            print(f"  {family} {name} p50={p50:.1f} p95={p95:.0f} "
                  f"mass@min={cells[name]['mass_at_min']:.3f} "
                  f"({time.perf_counter()-started:.0f}s)", flush=True)

        for name, c in cells.items():
            c["accept_mass_at_min"] = bool(c["mass_at_min"] < MASS_AT_MIN_MAX)
            c["accept_p50"] = bool(1/SHAPE_FACTOR <= c["ct_p50"]/REF_CT_P50 <= SHAPE_FACTOR)
            c["accept_p95"] = bool(1/SHAPE_FACTOR <= c["ct_p95"]/REF_CT_P95 <= SHAPE_FACTOR)
            c["accept_rpj_not_worse"] = bool(
                base_rpj is None or c["moments"]["rpj_mean"] <= base_rpj * 1.0000001)
            c["accepted"] = bool(c["accept_mass_at_min"] and c["accept_p50"]
                                 and c["accept_p95"] and c["accept_rpj_not_worse"])
            # Falsifier: median improves while the tail worsens proportionally.
            b = cells["partial0|blocking|legacy_theatre_stock"]
            dm = abs(c["ct_p50"]-REF_CT_P50) - abs(b["ct_p50"]-REF_CT_P50)
            dt = abs(c["ct_p95"]-REF_CT_P95) - abs(b["ct_p95"]-REF_CT_P95)
            c["redistributes_not_corrects"] = bool(dm < 0 and dt > 0 and abs(dt) >= abs(dm))

        d = {n: c["discrepancies"] for n, c in cells.items()}
        out[family] = {
            "cells": cells,
            "non_dominated_set": non_dominated(d, EPSILON),
            "epsilon_stability": epsilon_stability(d, EPSILONS),
            "accepted_cells": sorted(n for n, c in cells.items() if c["accepted"]),
            "reference_ct_p50": REF_CT_P50, "reference_ct_p95": REF_CT_P95,
        }
        print(f"  {family}: accepted {out[family]['accepted_cells']}", flush=True)

    payload = {
        "schema_version": "matching_sweep_v1",
        "calibration_provenance": calibration_stamp(),
        "claim_status": "DEVELOPMENT_SWEEP_NO_DEFAULT_CHANGED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preregistration": "docs/PREREGISTRO_EMPAREJAMIENTO_ORDEN_STOCK_2026-07-30.md",
        "selection_rule": "none -- non-dominated set and declared acceptance flags only",
        "acceptance_declared_before_running": {
            "mass_at_min_max": MASS_AT_MIN_MAX, "shape_factor": SHAPE_FACTOR,
            "reference_ct_p50": REF_CT_P50, "reference_ct_p95": REF_CT_P95,
            "rpj_mean_no_worse_than_current_default": True},
        "roots": list(args.roots), "results": out,
        "elapsed_seconds": time.perf_counter() - started,
    }
    body = json.dumps(payload, indent=1, sort_keys=True)
    payload["self_sha256"] = sha256(body.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"\n-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
