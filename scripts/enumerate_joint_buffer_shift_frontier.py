#!/usr/bin/env python3
"""The joint buffers x shifts static frontier: all 648 postures, enumerated.

Closes the gap no existing run covers. The v2 comparator enumerates all 216 buffer
vectors but pins `shifts=1`. The five-column metric panel crosses buffers with shifts
but over five hand-picked postures on four tapes. Garrido's expanded contract of
2026-07-28 is buffers **and** shifts, so the frontier is 6^3 x 3 = 648.

This is a **static enumeration**, not a controller adjudication. It answers three
questions that need no learner and no confirmation universe:

1. What is the best fixed posture over the *joint* domain, and does adding the shift
   dimension move the buffer part of it?
2. Does the metric disagreement on shifts survive full enumeration? The panel found
   `R_cobb_douglas` picking one shift while all three ReT variants picked three, at
   identical fill -- but over five buffer postures. If that reverses once the buffer
   domain is complete, it was an artefact of the subset.
3. Which postures clear declared service floors, and what does the resource/service
   frontier look like over the whole domain?

Declared before evaluation, because `kappa_dot` is normalised by the comparison set
and every member moves every other member's R: the comparison set **is** the complete
648-posture domain, and the service floors are fixed in `SERVICE_FLOORS` below. All
four metrics are reported for every posture, always; none is selected afterwards.

Post-dates the RPj cadence fix (commit 125b94f), so `ret_excel` is cadence-invariant
here. `step_hours` is recorded anyway.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.cobb_douglas_resilience import (  # noqa: E402
    CobbDouglasRecorder,
    score_comparison_set,
)
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.expanded_contract_controllers import LADDER_HOURS, NODES  # noqa: E402
from supply_chain.expanded_contract_controllers_v2 import posture_targets  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

FAMILIES = {"R1r": ("R11", "R12", "R13", "R14"), "R2r": ("R21", "R22", "R23", "R24")}
SHIFTS = (1, 2, 3)
# Development roots, disjoint from every screen and every confirmation block used so
# far (1.22M buffer gate, 1.31M v1, 1.43M/1.53M v2 development, 1.61M/1.62M C-D,
# 1.71M/1.81M the prospective confirmation).
DEFAULT_ROOTS = tuple(1_900_001 + i for i in range(12))

# Declared before evaluation. Constraints a posture passes or fails, never terms
# inside an objective it can trade away.
SERVICE_FLOORS = {
    "flow_fill_rate_min": 0.90,
    "lost_orders_max": 0.0,
    "backorder_qty_final_max": 50_000.0,
}
# The floor is a decision; a conclusion that moves with it is a conclusion about the
# decision. Swept, and the sweep is persisted.
BACKORDER_CAPS = (20_000.0, 40_000.0, 50_000.0, 60_000.0, 100_000.0, 1e12)

METRICS = ("ret_excel", "ret_excel_full_ledger", "R_cobb_douglas", "ret_excel_cvar10")
PHYSICAL = ("flow_fill_rate", "lost_orders", "backorder_qty_final",
            "delivered_rations", "strategic_injected")


def evaluate(job: tuple) -> dict:
    posture, shifts, family, seed, horizon, period = job
    sim = MFSCSimulation(
        shifts=shifts,
        initial_buffers={n: 0.0 for n in NODES},
        inventory_replenishment_period=168.0,
        seed=seed, horizon=horizon, risks_enabled=True, risk_level="current",
        enabled_risks=set(FAMILIES[family]),
        risk_overrides={r: "increased" for r in FAMILIES[family]},
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    targets = posture_targets(posture)
    rec = CobbDouglasRecorder(period_hours=period)
    elapsed = 0.0
    while elapsed < horizon:
        sim.inventory_buffer_targets.update({k: float(v) for k, v in targets.items()})
        step = min(period, horizon - elapsed)
        sim.step(action=None, step_hours=step)
        elapsed += step
        rec.sample(sim)

    agg = rec.aggregate()
    m = compute_episode_metrics(sim)
    out = {"posture": list(posture), "shifts": shifts, "family": family, "seed": seed,
           "zeta": agg["zeta"], "epsilon": agg["epsilon"], "phi": agg["phi"],
           "tau": agg["tau"], "kappa": agg["kappa"]}
    out.update({k: float(m[k]) for k in (
        "ret_excel", "ret_excel_full_ledger", "ret_thesis", "ret_excel_cvar10",
        "flow_fill_rate", "lost_orders", "backorder_qty_final", "delivered_rations")})
    out["strategic_injected"] = float(sim.total_strategic_raw_injected
                                     + sim.total_strategic_rations_injected)
    return out


def pareto_front(cells: dict) -> list[str]:
    """Non-dominated on (kappa down, fill up, lost down). Three axes only."""
    def dominates(a: dict, b: dict) -> bool:
        ge = (a["kappa"] <= b["kappa"] and a["flow_fill_rate"] >= b["flow_fill_rate"]
              and a["lost_orders"] <= b["lost_orders"])
        gt = (a["kappa"] < b["kappa"] or a["flow_fill_rate"] > b["flow_fill_rate"]
              or a["lost_orders"] < b["lost_orders"])
        return ge and gt
    return sorted(n for n, v in cells.items()
                  if not any(dominates(o, v) for k, o in cells.items() if k != n))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", nargs="+", default=["R1r", "R2r"])
    ap.add_argument("--roots", nargs="+", type=int, default=list(DEFAULT_ROOTS))
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--period-hours", type=float, default=672.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--contract", type=Path,
                    default=Path("contracts/cobb_douglas_calibration_v1.json"))
    ap.add_argument("--output", type=Path,
                    default=Path("results/joint_frontier/"
                                 "buffer_shift_648_v1/result.json"))
    args = ap.parse_args()

    exponents = json.loads(args.contract.read_text())["exponents"]
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    postures = tuple(product(LADDER_HOURS, repeat=len(NODES)))
    assert len(postures) * len(SHIFTS) == 648, len(postures) * len(SHIFTS)
    started = time.perf_counter()
    out: dict = {}

    for family in args.families:
        jobs = [(p, s, family, t, horizon, args.period_hours)
                for p in postures for s in SHIFTS for t in args.roots]
        rows: list[dict] = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, r in enumerate(pool.map(evaluate, jobs, chunksize=8), 1):
                rows.append(r)
                if i % 2000 == 0:
                    print(f"  {family} {i}/{len(jobs)} "
                          f"({time.perf_counter() - started:.0f}s)", flush=True)

        # Aggregate to one row per posture-shift cell, averaged over roots.
        cells: dict[str, dict] = {}
        per_root: dict[str, list[dict]] = {}
        for r in rows:
            name = f"{'/'.join(str(h) for h in r['posture'])}|S{r['shifts']}"
            per_root.setdefault(name, []).append(r)
        keys = ("zeta", "epsilon", "phi", "tau", "kappa", "ret_excel",
                "ret_excel_full_ledger", "ret_thesis", "ret_excel_cvar10") + PHYSICAL
        for name, group in per_root.items():
            cells[name] = {k: float(np.mean([g[k] for g in group])) for k in keys}
            cells[name]["shifts"] = group[0]["shifts"]
            cells[name]["posture"] = group[0]["posture"]

        scored = score_comparison_set(cells, exponents)
        for name, v in cells.items():
            v.update(scored[name])
            v["service_pass"] = bool(
                v["flow_fill_rate"] >= SERVICE_FLOORS["flow_fill_rate_min"]
                and v["lost_orders"] <= SERVICE_FLOORS["lost_orders_max"]
                and v["backorder_qty_final"]
                <= SERVICE_FLOORS["backorder_qty_final_max"])

        winners = {m: max(cells, key=lambda n: cells[n][m]) for m in METRICS}
        # Does the shift dimension move the buffer part of the optimum? Compare the
        # joint winner against the winner restricted to shifts=1, which is what the
        # v2 comparator enumerated.
        s1 = {n: v for n, v in cells.items() if v["shifts"] == 1}
        winners_s1 = {m: max(s1, key=lambda n: s1[n][m]) for m in METRICS}

        eligible = {n for n, v in cells.items() if v["service_pass"]}
        sweep = []
        for cap in BACKORDER_CAPS:
            ok = {n for n, v in cells.items()
                  if v["flow_fill_rate"] >= SERVICE_FLOORS["flow_fill_rate_min"]
                  and v["lost_orders"] <= SERVICE_FLOORS["lost_orders_max"]
                  and v["backorder_qty_final"] <= cap}
            w = {m: (max(ok, key=lambda n: cells[n][m]) if ok else None)
                 for m in METRICS}
            sweep.append({"backorder_cap": cap, "n_pass": len(ok),
                          "winner_by_metric": w,
                          "n_distinct_winners": len(set(w.values()))})

        out[family] = {
            "n_cells": len(cells),
            "winner_by_metric_joint": winners,
            "winner_by_metric_shifts1_only": winners_s1,
            "joint_winner_differs_from_shifts1": {
                m: winners[m] != winners_s1[m] for m in METRICS},
            "best_shift_by_metric": {m: cells[winners[m]]["shifts"] for m in METRICS},
            "all_metrics_agree_joint": len(set(winners.values())) == 1,
            "n_service_pass": len(eligible),
            "winner_by_metric_among_service_pass": {
                m: max(eligible, key=lambda n: cells[n][m]) if eligible else None
                for m in METRICS},
            "service_floor_sweep": sweep,
            "sweep_winner_vectors_stable": len({
                json.dumps(s["winner_by_metric"], sort_keys=True)
                for s in sweep if s["n_pass"]}) == 1,
            "pareto_front_kappa_fill_lost_only": pareto_front(cells),
            "cells": cells,
        }
        print(f"  {family}: {len(cells)} cells, {len(eligible)} pass service "
              f"({time.perf_counter() - started:.0f}s)", flush=True)

    rows_note = "per-cell means over roots; per-root rows are not persisted here"
    payload = {
        "schema_version": "joint_buffer_shift_frontier_648_v1",
        "claim_status": "DEVELOPMENT_SCREEN_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": ("enumerate the joint buffers x shifts frontier that neither the v2 "
                    "comparator (216 buffers, shifts pinned to 1) nor the five-column "
                    "panel (5 buffer postures x 3 shifts) covers"),
        "domain": "6^3 buffer vectors x 3 shift levels = 648 postures",
        "comparison_set_is_complete_domain": True,
        "kappa_dot_is_set_relative": True,
        "service_floors_declared_before_evaluation": SERVICE_FLOORS,
        "backorder_caps_swept": list(BACKORDER_CAPS),
        "metric_panel": list(METRICS),
        "roots": list(args.roots),
        "roots_are_development_only": True,
        "step_hours": args.period_hours,
        "rpj_semantics": "immutable_onset_corrective (commit 125b94f)",
        "aggregation_note": rows_note,
        "results": out,
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
