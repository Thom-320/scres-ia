#!/usr/bin/env python3
"""Where does upstream storage have to bind before a supplier decision reaches service?

Contract: `docs/PREREGISTRO_ESCASEZ_MATERIA_PRIMA_2026-08-09.md`, frozen before this file.
Development replay on burned seeds 8600001-8600060. No virgin block is opened.

THE ANSWER IS A FRONTIER, NOT A LEVEL. The obvious way to make the Program V port stop returning
zero is to squeeze raw material until it does, which would be outcome engineering with extra steps.
So the cap is expressed in days of supply -- the unit doctrine sizes a depot in -- swept across a
declared range, and reported whole. The unlimited cell is the control: if headroom appears there,
the instrument built it.

AND A BINDING CAP IS NOT AUTOMATICALLY A FINDING. A chain starved badly enough will show a large
gap between policies simply because it is broken. `f5` therefore requires the best constant to be
still serving before any cell is read as headroom.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_program_v_des_port_v1 import HELD, SELECT, paired, play                # noqa: E402
from supply_chain.arm_runner import seal_and_write                              # noqa: E402
from supply_chain import falsifiers as F                                        # noqa: E402
from supply_chain.program_v_supplier_memory import policy_library               # noqa: E402
from supply_chain.seed_custody import custody_falsifier, module_manifest        # noqa: E402

#: Measured on the frozen DES BEFORE the contract was written, and cited in it.
UNITS_PER_DAY = 19_011.0
CELLS = {"unlimited": None, "d180": 180, "d90": 90, "d60": 60, "d30": 30, "d14": 14}
SERVICE_FLOOR = 0.50
OUT = Path("results/raw_scarcity_boundary/result.json")
CONTRACT = Path("docs/PREREGISTRO_ESCASEZ_MATERIA_PRIMA_2026-08-09.md")
MODULES = ("supply_chain/supply_chain.py", "supply_chain/program_v_supplier_memory.py",
           "supply_chain/falsifiers.py")


def run_cell(days) -> dict:
    cap = None if days is None else days * UNITS_PER_DAY
    rows = {}
    for policy in policy_library():
        sel = [play(policy, s, cap=cap) for s in SELECT]
        held = [play(policy, s, cap=cap) for s in HELD]
        rows[policy.name] = {
            "deployable": bool(policy.deployable),
            "select_mean": float(np.mean([r["service"] for r in sel])),
            "held": [float(r["service"]) for r in held],
            "held_mean": float(np.mean([r["service"] for r in held])),
            "blocked": float(np.mean([r["blocked"] for r in held])),
            "orders": float(np.mean([r["orders"] for r in held])),
            "mass_residual_rel": float(max(r["mass_residual_rel"] for r in held)),
        }
    arr = lambda n: np.array(rows[n]["held"])                      # noqa: E731
    constants = [k for k in rows if k.startswith("constant_")]
    best_constant = max(constants, key=lambda k: rows[k]["select_mean"])
    observable = [k for k in rows if rows[k]["deployable"] and not k.startswith("constant_")
                  and not k.startswith("placebo")]
    best_observable = max(observable, key=lambda k: rows[k]["select_mean"])
    return {
        "days_of_supply": days, "cap_units": cap,
        "best_constant": best_constant, "best_observable": best_observable,
        "best_constant_service": rows[best_constant]["held_mean"],
        "blocked_units": rows[best_constant]["blocked"],
        "H_priv": paired(arr("privileged_true_state"), arr(best_constant)),
        "H_obs": paired(arr(best_observable), arr(best_constant)),
        "H_ret": paired(arr("bayes_retained"), arr("bayes_reset")),
        "retained_vs_delayed": paired(arr("bayes_retained"), arr("placebo_delayed")),
        "retained_vs_shuffled": paired(arr("bayes_retained"), arr("placebo_shuffled")),
        "max_mass_residual_rel": float(max(v["mass_residual_rel"] for v in rows.values())),
        "order_count_spread": float(np.ptp([v["orders"] for v in rows.values()])),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / OUT)
    args = ap.parse_args()

    cells = {name: run_cell(days) for name, days in CELLS.items()}
    unlimited = cells["unlimited"]
    # Only cells whose best constant is still serving may be read as headroom.
    readable = {k: v for k, v in cells.items()
                if v["best_constant_service"] >= SERVICE_FLOOR and k != "unlimited"}
    best = max(readable, key=lambda k: readable[k]["H_priv"]["lcb95"]) if readable else None

    checks = {
        "f1_unlimited_cell_is_flat": F.lt(
            max(abs(unlimited[e]["mean"]) for e in ("H_priv", "H_obs", "H_ret")), 1e-9,
            "with unlimited storage the port returned exactly zero; headroom appearing here would "
            "mean the cap layer changed the base physics rather than adding an opt-in roof"),
        "f2_cap_binds_when_tight": F.gt(
            cells["d14"]["blocked_units"], 0.0,
            "a roof that is never hit is not scarcity and the cell would be mislabelled"),
        "f3_blocked_is_never_destroyed": F.lt(
            max(v["max_mass_residual_rel"] for v in cells.values()), 1e-6,
            "what does not fit must never have entered; this is the defect retracted on 2026-08-08"),
        "f4_same_tape_same_risks": F.lt(
            max(v["order_count_spread"] for v in cells.values()), 0.5,
            "if the order count moves with the policy the cap consumed simulator RNG"),
        "f5_best_constant_still_serves": F.check(
            bool(readable),
            "a chain starved into collapse separates policies because it is broken, not because a "
            "decision matters; with no cell above the floor there is nothing legible",
            computed_from={"floor": SERVICE_FLOOR, "n_readable": len(readable)}),
        "f6_H_priv_material": F.ge(
            cells[best]["H_priv"]["lcb95"] if best else 0.0, 0.02,
            "scarcity may simply degrade every policy equally, leaving no decision to make"),
        "f7_H_ret_positive": F.gt(
            cells[best]["H_ret"]["lcb95"] if best else 0.0, 0.0,
            "an unverifiable external report predicts this is EXACTLY zero because 24-72h yields "
            "reveal the regime too early for retention to add anything; written down before the run"),
    }
    checks["d1_domain_justification"] = F.disclosure(
        "the cap exists because the source declares unlimited WDC/AL/SB storage as an explicit "
        "simplification, not as a fact; the level is in days of supply from consumption measured "
        "on the frozen DES before the contract was written",
        evidence={"units_per_day": UNITS_PER_DAY, "cells": {k: v for k, v in CELLS.items()}})
    checks["custody"] = custody_falsifier(sorted(set(SELECT + HELD)))
    summary = F.summarise(checks)

    if not (checks["f1_unlimited_cell_is_flat"]["passed"]
            and checks["f3_blocked_is_never_destroyed"]["passed"]):
        status = "BLOCKED_INSTRUMENT"
    elif not checks["f5_best_constant_still_serves"]["passed"]:
        status = "NO_LEGIBLE_CELL__SCARCITY_ONLY_BREAKS_THE_CHAIN"
    elif not checks["f6_H_priv_material"]["passed"]:
        status = "SCARCITY_DOES_NOT_CREATE_PHYSICAL_HEADROOM"
    elif checks["f7_H_ret_positive"]["passed"]:
        status = "SCARCITY_MAKES_HEADROOM_PHYSICAL_AND_HISTORY_ADDS_VALUE"
    else:
        status = "SCARCITY_MAKES_HEADROOM_PHYSICAL_BUT_HISTORY_ADDS_NOTHING"

    payload = {
        "schema_version": "raw_scarcity_boundary_v1", "claim_status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_role": "DEVELOPMENT",
        "scope": "DEVELOPMENT_REPLAY_ON_BURNED_SEEDS_NO_VIRGIN_BLOCK_NO_LEARNER",
        "endpoint": "theatre_fill_rate_delivered_over_demanded",
        "seeds": sorted(set(SELECT + HELD)),
        "units_per_day": UNITS_PER_DAY, "service_floor": SERVICE_FLOOR,
        "readable_cells": sorted(readable), "best_readable_cell": best,
        "cells": cells,
        "external_prediction": {
            "source": "unverifiable Program W report, commits not published",
            "claim": "H_ret is exactly 0 under scarcity because 24-72h yields reveal the regime "
                     "too early for retention to add anything",
            "written_before_this_run": True},
        "module_manifest": module_manifest(MODULES),
        "falsifiers": checks, "falsifier_summary": summary,
    }
    seal_and_write(payload, args.output, contract=CONTRACT,
                   reference=Path("results/program_v/des_port_v1/result.json"))

    print(f"veredicto: {status}\n")
    print(f"  {'celda':10}{'servicio cte':>13}{'bloqueado':>13}   H_priv                  H_ret")
    for name, c in cells.items():
        print(f"  {name:10}{c['best_constant_service']:>13.4f}{c['blocked_units']:>13,.0f}   "
              f"{c['H_priv']['mean']:+.4f} [{c['H_priv']['lcb95']:+.4f}]   "
              f"{c['H_ret']['mean']:+.4f} [{c['H_ret']['lcb95']:+.4f}]")
    print(f"\n  celdas legibles: {sorted(readable)} | mejor: {best}")
    print(f"  falsadores: {summary['n_computed']} computados, {summary['n_failed']} fallidos")
    for name, c in checks.items():
        if c.get("computed"):
            print(f"    {name:36} {'PASA' if c['passed'] else 'FALLA'}")
    print(f"\n  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
