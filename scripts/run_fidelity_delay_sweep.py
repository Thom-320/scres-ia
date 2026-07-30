#!/usr/bin/env python3
"""The calibration grid, scored against Garrido's own moments by dominance.

Executes `contracts/paper_b_independent_calibration_v2.json`. It does not pick a
winner and cannot: the output is the non-dominated set over six moments, plus the
epsilon sensitivity of that set. If the whole grid survives, that is the answer.

Two axes, and they cost differently, so they are handled differently:

* **the fulfilment delay changes the physics**, so each value needs its own episodes;
* **the branch predicate changes only the classification**, so both semantics are
  evaluated post-hoc on the *same* episodes.

The predicate transformation is exact rather than a re-simulation. `RPj` in
`ret_recovery_period_mode = "disruption"` is the order's total disruption hours, and
`APj` on the autotomy branch is `min(total_disruption, LTj)`. So an order the DES put in
recovery, under a predicate that would call it autotomy, becomes `APj = min(RPj, LTj)`
with `RPj = 0` -- and the official ledger is then re-called on the mutated orders. Same
discipline as the ReT repair variants: mutate the inputs, never reimplement the formula.

Nothing here changes a constant or relabels a frozen result. It measures.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from hashlib import sha256
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
    FAMILY_SHEETS,
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
# Thesis-base configuration, so the comparison is against his Cf1-Cf20 and not a
# regime of ours: one shift, no strategic buffers.
SHIFTS = 1
# Declared in the contract. Disjoint from every previous block.
ROOTS = tuple(2_000_001 + i for i in range(12))
# Tolerance for the thesis-exact predicate. Swept, because the canonical data refutes
# any single value: autotomy rows there run CTj - LT in [0.00744, 0.048] while
# non-autotomy rows that also exceed LT start at exactly 0.00744, so no band on CTj
# reproduces his classification. These are reported side by side, none is "the" rule.
TOLERANCES = (0.0, 0.05, 0.5)
EPSILONS = (0.25, 0.5, 1.0, 2.0)


def scored(sim) -> list:
    return [o for o in sim.orders
            if not bool(getattr(o, "metrics_excluded", False))
            and float(getattr(o, "OPTj", 0.0)) >= float(sim.warmup_time)]


TRANSPORT_MODES = ("skip_wave", "retry_when_ready")


def run_episode(*, delay: float, family: str, seed: int, horizon: float,
                transport: str = "skip_wave"):
    sim = MFSCSimulation(
        shifts=SHIFTS,
        initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0,
        seed=seed, horizon=horizon, risks_enabled=True, risk_level="current",
        enabled_risks=set(FAMILIES[family]),
        risk_overrides={r: "increased" for r in FAMILIES[family]},
        demand_on_hand_fulfillment_delay=delay,
        transport_block_mode=transport,
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    sim.step(action=None, step_hours=horizon)
    return sim


def moments_under(sim, orders: list, *, predicate: str, tol: float,
                  horizon_years: float) -> dict[str, float]:
    """Moments under one branch predicate, by exact reclassification of the orders."""
    now = float(sim.env.now)
    patched = []
    for order in orders:
        clone = copy.copy(order)
        ct = getattr(clone, "CTj", None)
        lt = float(getattr(clone, "LTj", 0.0) or 0.0)
        rpj = float(getattr(clone, "RPj", 0.0) or 0.0)
        if ct is not None and rpj > 0.0:
            ct = float(ct)
            fires = (abs(ct - lt) <= tol if predicate == "thesis_exact_autotomy"
                     else ct <= lt)
            if fires:
                # RPj is the order's total disruption hours in `disruption` mode, and
                # the autotomy branch takes min(total_disruption, LTj).
                clone.APj = min(rpj, lt)
                clone.RPj = 0.0
        patched.append(clone)

    book = ledger(patched, current_time=now)
    return moments_from_rows(
        apj=[float(getattr(o, "APj", 0.0) or 0.0) for o in patched],
        rpj=[float(getattr(o, "RPj", 0.0) or 0.0) for o in patched],
        ret=[float(v) for v in book["ret_values"]],
        horizon_years=horizon_years)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path,
                    default=Path("contracts/paper_b_independent_calibration_v2.json"))
    ap.add_argument("--reference", type=Path,
                    default=Path("results/metric_audit/fidelity_reference_v2/result.json"))
    ap.add_argument("--roots", nargs="+", type=int, default=list(ROOTS))
    ap.add_argument("--delays", nargs="+", type=float, default=None,
                    help="EXPLORATORY override of the contract's frozen grid. The "
                         "contract's no-selection rule still applies: every row is "
                         "reported and none may be chosen for the result it gives.")
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--output", type=Path,
                    default=Path("results/metric_audit/fidelity_delay_sweep_v1/"
                                 "result.json"))
    args = ap.parse_args()

    contract = json.loads(args.contract.read_text())
    grid = contract["fulfillment_delay_decision"]["prospective_robustness_grid"]
    delays = ([float(d) for d in args.delays] if args.delays
              else [float(d) for d in grid["delay_hours"]])
    grid_is_contract = args.delays is None
    ref_blob = json.loads(args.reference.read_text())
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    # Our own calendar, so the population moment is a rate on both sides.
    horizon_years = float(args.horizon_weeks) / 52.0
    started = time.perf_counter()

    reference = {
        fam: {m: MomentReference(**{k: v[k] for k in ("mean", "spread", "n_sheets")})
              for m, v in blk.items()}
        for fam, blk in ref_blob["reference_by_family"].items()}

    out: dict = {}
    for family in FAMILY_SHEETS:
        cells: dict[str, dict] = {}
        for delay in delays:
          for transport in TRANSPORT_MODES:
            sims = [(s, scored(s)) for s in
                    (run_episode(delay=delay, family=family, seed=t, horizon=horizon,
                                 transport=transport)
                     for t in args.roots)]
            for predicate in ("operational_on_time", "thesis_exact_autotomy"):
                for tol in (TOLERANCES if predicate == "thesis_exact_autotomy" else (0.0,)):
                    per_root = [moments_under(s, o, predicate=predicate, tol=tol,
                                              horizon_years=horizon_years)
                                for s, o in sims]
                    mean = {m: float(np.mean([r[m] for r in per_root]))
                            for m in MOMENT_NAMES}
                    se = {m: float(np.std([r[m] for r in per_root], ddof=1)
                                   / np.sqrt(len(per_root))) for m in MOMENT_NAMES}
                    name = (f"delay{delay:g}|{transport}|{predicate}"
                            + (f"|tol{tol:g}" if predicate == "thesis_exact_autotomy"
                               else ""))
                    cells[name] = {
                        "delay_hours": delay, "transport_block_mode": transport,
                        "predicate": predicate, "tolerance": tol,
                        "moments": mean, "moment_se": se,
                        "discrepancies": discrepancies(mean, se, reference[family])}
            print(f"  {family} delay={delay:g} {transport} "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)

        d_only = {n: c["discrepancies"] for n, c in cells.items()}
        stability = epsilon_stability(d_only, EPSILONS)
        out[family] = {
            "n_cells": len(cells),
            "non_dominated_set": non_dominated(d_only, EPSILON),
            "epsilon_declared": EPSILON,
            "epsilon_stability": stability,
            "grid_discriminates": len(non_dominated(d_only, EPSILON)) < len(cells),
            "reference": {m: {"mean": reference[family][m].mean,
                              "spread": reference[family][m].spread}
                          for m in MOMENT_NAMES},
            "cells": cells,
        }
        print(f"  {family}: non-dominated {len(out[family]['non_dominated_set'])}"
              f"/{len(cells)}, epsilon-stable={stability['stable']}", flush=True)

    payload = {
        "schema_version": "fidelity_delay_sweep_v1",
        "calibration_provenance": calibration_stamp(
            note="the swept delay overrides the module constant per cell"),
        "claim_status": "DEVELOPMENT_FIDELITY_SWEEP_NO_CONSTANT_CHANGED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(args.contract),
        "contract_sha256": contract.get("self_sha256"),
        "reference_path": str(args.reference),
        "reference_sha256": ref_blob.get("self_sha256"),
        "selection_rule": "none -- the output is the non-dominated set, never a winner",
        "lead_time_fixed_at": 48.0,
        "lead_time_source": "Garrido 2017 thesis §6.8.2 p.111",
        "roots": list(args.roots),
        "delays_swept": delays,
        "grid_is_the_frozen_contract_grid": grid_is_contract,
        "exploratory_note": (None if grid_is_contract else
            "EXPLORATORY: delays overridden on the command line to resolve the step "
            "between 48 and 49 found by the contract grid. No row may be selected for "
            "the result it produces; the no-selection rule is unchanged."),
        "tolerances_swept": list(TOLERANCES),
        "transport_modes_swept": list(TRANSPORT_MODES),
        "epsilons_swept": list(EPSILONS),
        "r3_excluded": "no reference workbook exists; external validation only",
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
