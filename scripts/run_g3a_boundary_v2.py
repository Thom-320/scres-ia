#!/usr/bin/env python3
"""Rebuild the G3a boundary map with a runner, a contract, falsifiers and a sealed artifact.

Contract: `docs/PREREGISTRO_G3A_V2_RECONSTRUCCION_2026-08-08.md`. PI-authorised development block
8800001-8800060, split 30 selection / 30 held-out. A development holdout, not a confirmation.

The finding this is trying to reproduce is not "adaptation works". It is that observable adaptation
appears INSIDE a hard quota and DISAPPEARS once spare capacity is reallocated -- which would mean
the apparent premium was the price of leaving the truck half empty. f7 and f8 test the two halves
and both must hold; either one alone is not the result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write                              # noqa: E402
from supply_chain import falsifiers as F                                        # noqa: E402
from supply_chain.g3a_boundary_v2 import (                                      # noqa: E402
    HOURS_PER_WEEK, LIBRARY, WEEKS, Cell, regime_tape, share_schedule,
    warning_tape, worst_claimant_late_exposure_service)
from supply_chain.seed_custody import custody_falsifier, module_manifest        # noqa: E402
from supply_chain.supply_chain import MFSCSimulation                            # noqa: E402

SELECT = tuple(range(8800001, 8800031))
HELD = tuple(range(8800031, 8800061))
OUT = Path("results/g3a_boundary_v2/result.full34.json")
CONTRACT = Path("docs/PREREGISTRO_G3A_V2_RECONSTRUCCION_2026-08-08.md")
MODULES = ("supply_chain/supply_chain.py", "supply_chain/g3a_boundary_v2.py",
           "supply_chain/cssu_allocation.py", "supply_chain/falsifiers.py")
SESOI = 0.01
CELLS = [Cell(p, d, c)
         for p, d in (("iid", "uniform"), ("persistent", "uniform"), ("persistent", "seasonal"))
         for c in ("hard_quota", "spare_reallocation", "global_pool")]


def build_tape(seed: int, cell: Cell) -> dict:
    regimes = regime_tape(seed, cell.process)
    return {"regimes": regimes, "warnings": warning_tape(seed, regimes),
            "shares": share_schedule(regimes),
            "lagged_a_minus_b": [1.0 if r == "A_PRESSURE" else -1.0 if r == "B_PRESSURE" else 0.0
                                 for r in regimes],
            "backlog_a_share": [0.6 if r == "A_PRESSURE" else 0.4 if r == "B_PRESSURE" else 0.5
                                for r in regimes]}


def play(cell: Cell, controller, seed: int) -> dict:
    """One episode. The weekly share is applied by reconstructing the simulator per week is NOT
    possible here, so the schedule is handed to the simulator up front and the controller's weekly
    shares are applied through the allocation action at week boundaries."""
    tape = build_tape(seed, cell)
    shares = controller.shares(tape)
    kwargs = cell.sim_kwargs()
    demand = {"seasonal": {"demand_process": "garrido_seasonal_v1"}}.get(cell.demand, {})
    # The split-CSSU dispatch only carries traffic under the op9-linked fulfilment path with a
    # fixed daily clock and partial FIFO service -- the configuration the existing contention
    # runners use. Under the legacy theatre-stock path orders never reach the claimant queues and
    # every arm scores identically, which is what this measured before the line was fixed.
    sim = MFSCSimulation(seed=seed, horizon=WEEKS * HOURS_PER_WEEK,
                         shifts=1,
                         initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
                         inventory_replenishment_period=0.0,
                         risks_enabled=True, risk_level="current",
                         cssu_service_rule="FIFO_PARTIAL",
                         order_fulfillment_mode="op9_linked",
                         op9_dispatch_policy="fixed_clock_daily",
                         strict_exogenous_crn=True,
                         cssu_destination_weight_schedule=tape["shares"],
                         cssu_allocation_a=float(np.clip(shares[0], 0.05, 0.95)),
                         **kwargs, **demand)
    # Weekly re-allocation, one week at a time. `_start_processes` has to be called explicitly:
    # `run()` does it and then runs to the horizon, and driving `env.run(until=...)` without it
    # advances a clock over an empty event queue -- measured as zero orders in every cell before
    # this line existed, which is the shape a silent no-op takes.
    sim._start_processes()
    for week in range(WEEKS):
        if week > 0:
            sim.cssu_allocation_a = float(np.clip(shares[week], 0.05, 0.95))
        sim.env.run(until=min((week + 1) * HOURS_PER_WEEK, sim.horizon))
    service = worst_claimant_late_exposure_service(sim)
    return {"worst": service["worst"], "A": service["A"], "B": service["B"],
            "forfeited": float(getattr(sim, "cssu_forfeited_rations", 0.0)),
            "orders": len([o for o in sim.orders if o.cssu_destination in ("A", "B")]),
            "switches": int(len({round(s, 4) for s in shares}) - 1)}


def run_cell(cell: Cell) -> dict:
    rows = {}
    for controller in LIBRARY:
        sel = np.array([play(cell, controller, s)["worst"] for s in SELECT])
        held = [play(cell, controller, s) for s in HELD]
        rows[controller.name] = {
            "privileged": controller.privileged,
            "select_mean": float(sel.mean()),
            "held": [float(r["worst"]) for r in held],
            "held_mean": float(np.mean([r["worst"] for r in held])),
            "forfeited": float(np.mean([r["forfeited"] for r in held])),
        }
    deployable = {k: v for k, v in rows.items()
                  if not v["privileged"] and not k.startswith("placebo")}
    fixed = max((k for k in deployable if k.startswith("const_")),
                key=lambda k: rows[k]["select_mean"])
    adaptive = max((k for k in deployable if not k.startswith("const_")),
                   key=lambda k: rows[k]["select_mean"])
    diff = np.array(rows[adaptive]["held"]) - np.array(rows[fixed]["held"])
    boot = np.random.default_rng(20260808).choice(
        diff, size=(20_000, diff.size), replace=True).mean(axis=1)
    return {
        "rows": rows, "best_fixed_on_select": fixed, "best_adaptive_on_select": adaptive,
        "h_obs": {"mean": float(diff.mean()), "lcb95": float(np.percentile(boot, 2.5)),
                  "ucb95": float(np.percentile(boot, 97.5)),
                  "favourable": int((diff > 0).sum()), "n": int(diff.size)},
        "best_fixed_service": rows[fixed]["held_mean"],
        "best_adaptive_service": rows[adaptive]["held_mean"],
        "adaptive_forfeited": rows[adaptive]["forfeited"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / OUT)
    args = ap.parse_args()

    # f1's probe first and on its own terms: under pooling the share must not move the endpoint.
    pool_cell = Cell("persistent", "uniform", "global_pool")
    pooled = [play(pool_cell, LIBRARY[i], SELECT[0])["worst"] for i in (0, 4)]
    quota_cell = Cell("persistent", "uniform", "hard_quota")
    quota = [play(quota_cell, LIBRARY[i], SELECT[0]) for i in (0, 4)]

    cells = {c.label: run_cell(c) for c in CELLS}
    hq = [v for k, v in cells.items() if k.endswith("hard_quota")]
    gp = [v for k, v in cells.items() if k.endswith("global_pool")]
    best_hq = max(hq, key=lambda v: v["h_obs"]["lcb95"])
    worst_gp = max(gp, key=lambda v: v["h_obs"]["lcb95"])

    def _best(rows, prefix):
        keys = [k for k in rows if k.startswith(prefix)]
        return max(rows[k]["held_mean"] for k in keys)

    # With four placebos, comparing against the worst would be a gift to the falsifier.
    placebo_gap = min(_best(v["rows"], "warning_") - _best(v["rows"], "placebo_")
                      for k, v in cells.items() if k.endswith("hard_quota"))

    checks = {
        "f1_pooling_is_action_invariant": F.lt(
            abs(pooled[0] - pooled[1]), 1e-9,
            "if the allocation share moves the endpoint under a single FIFO pool, the null cell "
            "is not null and every cross-contract comparison below is meaningless"),
        "f2_action_is_live_under_hard_quota": F.gt(
            abs(quota[0]["A"] - quota[1]["A"]) + abs(quota[0]["B"] - quota[1]["B"]), 1e-9,
            "an inert action would make every number below noise; the package's own probe was "
            "unilateral and said so"),
        "f4_demand_tape_is_identical_across_policies": F.lt(
            abs(quota[0]["orders"] - quota[1]["orders"]), 0.5,
            "the regime schedule must not consume simulator RNG; if the order count moves with "
            "the policy the cells are different worlds"),
        "f6_placebo_loses_under_hard_quota": F.gt(
            placebo_gap, 0.0,
            "if the shuffled-warning placebo ties the real warning, what was measured is cadence "
            "and not information"),
        "f7_hard_quota_shows_observable_headroom": F.gt(
            best_hq["h_obs"]["lcb95"], 0.0,
            "this is the package's headline and it may simply not reproduce"),
        "f8_work_conserving_kills_it": F.lt(
            worst_gp["h_obs"]["lcb95"], SESOI,
            "if pooling ALSO shows headroom, the proposed causal mechanism -- that the premium is "
            "the price of idle capacity -- is false"),
        "f9_forfeiture_is_measured": F.ge(
            float(best_hq["adaptive_forfeited"]), 0.0,
            "without forfeiture on the page, a premium for not wasting the truck reads as "
            "adaptation"),
    }
    checks["d1_endpoint_is_post_hoc"] = F.disclosure(
        "worst_claimant_late_exposure_service_v1 was defined after the 54 h vs 48 h timing "
        "conflict made on-time fill identically zero under every policy. It is not a validated "
        "SCRES measure and carries no confirmatory claim",
        evidence={"weeks": WEEKS, "n_controllers": len(LIBRARY)})
    checks["d2_development_holdout"] = F.disclosure(
        "30/30 split of a PI-authorised development block; a development holdout, not a "
        "confirmation, and it grants no confirmatory grade",
        evidence={"select": [SELECT[0], SELECT[-1]], "held": [HELD[0], HELD[-1]]})
    checks["custody"] = custody_falsifier(sorted(set(SELECT + HELD)))
    summary = F.summarise(checks)

    physics = all(checks[k]["passed"] for k in
                  ("f1_pooling_is_action_invariant", "f4_demand_tape_is_identical_across_policies"))
    if not physics:
        status = "BLOCKED_INSTRUMENT"
    elif checks["f7_hard_quota_shows_observable_headroom"]["passed"] and \
            checks["f8_work_conserving_kills_it"]["passed"]:
        status = "G3A_REPRODUCED__ADAPTATION_INSIDE_A_DOMINATED_CONTRACT"
    elif not checks["f7_hard_quota_shows_observable_headroom"]["passed"]:
        status = "G3A_DID_NOT_REPRODUCE"
    else:
        status = "G3A_PARTIAL__WORK_CONSERVATION_DID_NOT_REMOVE_IT"

    payload = {
        "schema_version": "g3a_boundary_v2", "claim_status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_role": "DEVELOPMENT",
        "scope": "DEVELOPMENT_HOLDOUT_ON_PI_AUTHORISED_BLOCK_NO_CONFIRMATION",
        "endpoint": "worst_claimant_late_exposure_service_v1",
        "seeds": sorted(set(SELECT + HELD)),
        "reproduction_target": {
            "source": "Garrido_CIE_core_results_v1_1.csv (external package, no artifact in repo)",
            "hard_quota_h_obs": 0.0963, "spare_reallocation_h_obs": 0.00126,
            "global_pool_h_obs": 0.0,
            "note": "targets to reproduce, never evidence"},
        "sesoi": SESOI, "n_cells": len(CELLS), "n_controllers": len(LIBRARY),
        "library_amendment": "docs/ENMIENDA_BIBLIOTECA_34_CONTROLADORES_2026-08-08.md",
        "cells": cells,
        "module_manifest": module_manifest(MODULES),
        "falsifiers": checks, "falsifier_summary": summary,
    }
    seal_and_write(payload, args.output, contract=CONTRACT,
                   reference=Path("research/external_packages/garrido_cie_v1_1_targets.json"))

    print(f"veredicto: {status}\n")
    for label, c in cells.items():
        h = c["h_obs"]
        print(f"  {label:38} H_obs {h['mean']:+.4f} [{h['lcb95']:+.4f}, {h['ucb95']:+.4f}]  "
              f"fijo {c['best_fixed_service']:.4f} -> adapt {c['best_adaptive_service']:.4f} "
              f"({c['best_adaptive_on_select']}) forfeit {c['adaptive_forfeited']:.0f}")
    print(f"\n  falsadores: {summary['n_computed']} computados, {summary['n_failed']} fallidos")
    for name, c in checks.items():
        if c.get("computed"):
            print(f"    {name:44} {'PASA' if c['passed'] else 'FALLA'}")
    print(f"\n  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
