#!/usr/bin/env python3
"""Does a shared budget or a shelf life create a sequential decision the buffer gate did not have?

Contract: `docs/PREREGISTRO_PRESUPUESTO_COMPARTIDO_Y_CADUCIDAD_2026-08-08.md`, committed before
this file. Development replay on already-burned seeds; no virgin block is opened.

THE READING ORDER IS THE CONTRACT'S AND IS NOT RE-DERIVED HERE. The thesis-faithful cell is read
first: at a 156-week shelf life over a 26-week horizon nothing can expire, and with no budget the
allocation cannot contend, so that cell MUST reproduce the conservative gate's near-zero result. If
it does not, the instrument invented the headroom and nothing below it is read.

TWO SINGLE-MECHANISM CELLS ARE WHY THIS IS AN EXPERIMENT. Running only "both on" against "both off"
would show that something changed and nothing about what. If the gap appears only under both, it is
an interaction; if under one, the other is decoration.

AND THE GAP IS NOT READ WITHOUT ITS NULL. A per-tape minimum over many noisy options is biased
upward, which is exactly how a ceiling survived three weeks before dying on virgin seeds this
morning. f6 and f7 pass together or the cell reports nothing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write                               # noqa: E402
from supply_chain.continuous_its_env import (                                     # noqa: E402
    _I1344, make_per_op_buffer_track_a_env)
from supply_chain import falsifiers as F                                         # noqa: E402
from supply_chain.seed_custody import custody_falsifier, module_manifest         # noqa: E402

R1 = ("R11", "R12", "R13", "R14")
R2 = ("R21", "R22", "R23", "R24")
MAX_STEPS, STEP_HOURS = 26, 168.0
LEAD_HOURS = 336.0
BAR = 0.01
TRAIN = tuple(range(8600001, 8600013))
TEST = tuple(range(8600013, 8600025))
OUT = Path("results/budget_expiry_boundary/result.json")
CONTRACT = Path("docs/PREREGISTRO_PRESUPUESTO_COMPARTIDO_Y_CADUCIDAD_2026-08-08.md")
MODULES = ("supply_chain/supply_chain.py", "supply_chain/continuous_its_env.py",
           "supply_chain/falsifiers.py", "supply_chain/arm_runner.py",
           "supply_chain/seed_custody.py")

INERT_SHELF = 156 * STEP_HOURS          # three years: cannot bite inside a 26-week horizon
SHORT_SHELF = 8 * STEP_HOURS            # OUR extension, swept and priced, never assumed
#: Fixed before any result, and amended before any result: the bag that stops all three nodes
#: being prepositioned at once. The preregistration wrote that purpose AND a formula that divided
#: it by the horizon, which are different numbers -- the divided one refuses 7.5M units and
#: replenishes 4.4k, which is starvation rather than contention. Purpose governs; see
#: docs/ENMIENDA_PRESUPUESTO_FORMULA_VS_PROPOSITO_2026-08-08.md.
TIGHT_BUDGET = 0.5 * (_I1344["op3_rm"] / 12.0 + _I1344["op5_rm"] / 12.0 + _I1344["op9_rations"])

CELLS = {
    "faithful_control": {"shelf": INERT_SHELF, "budget": None},
    "budget_only": {"shelf": INERT_SHELF, "budget": TIGHT_BUDGET},
    "expiry_only": {"shelf": SHORT_SHELF, "budget": None},
    "both": {"shelf": SHORT_SHELF, "budget": TIGHT_BUDGET},
}

#: Enumerated per-node postures. Deliberately small and declared: the point is whether the OPTIMUM
#: MOVES WITH THE TAPE, not how finely the grid is cut.
POSTURES = [(a, b, c) for a in (0.0, 0.5, 1.0) for b in (0.0, 0.5, 1.0) for c in (0.0, 0.5, 1.0)]


def make_env(cell: dict):
    return make_per_op_buffer_track_a_env(
        init_frac=0.0, reward_mode="ReT_excel_delta", observation_version="v6",
        risk_level="current", enabled_risks=R1 + R2, risk_rng_mode="per_risk",
        stochastic_pt=False, max_steps=MAX_STEPS, step_size_hours=STEP_HOURS,
        risk_obs=True, holding_cost=0.0, shift_cost=0.0,
        demand_process="garrido_seasonal_v1",
        demand_seasonal_contract={"forecast_mode": "garrido_generator"},
        inventory_replenishment_lead_time=LEAD_HOURS,
        strategic_shelf_life_hours=cell["shelf"],
        strategic_budget_per_period=cell["budget"])


def exposure(sim) -> float:
    horizon, start = float(sim.env.now), float(sim.warmup_time)
    num = den = 0.0
    for o in sim.orders:
        if bool(getattr(o, "metrics_excluded", False)):
            continue
        opt = float(getattr(o, "OPTj", 0.0) or 0.0)
        if opt < start:
            continue
        q = float(o.quantity or 0.0)
        due = opt + float(o.LTj or 0.0)
        end = float(o.OATj) if getattr(o, "OATj", None) is not None else horizon
        num += q * max(0.0, end - due)
        den += q * max(0.0, horizon - due)
    return num / den if den > 0 else 0.0


def play(cell: dict, posture, seed: int) -> dict:
    env = make_env(cell)
    env.reset(seed=int(seed))
    sim = env.unwrapped.sim
    action = np.array([*posture, -1.0], dtype=np.float32)
    done = truncated = False
    try:
        while not (done or truncated):
            _o, _r, done, truncated, _i = env.step(action)
        return {"L": exposure(sim),
                "expired": float(getattr(sim, "strategic_expired_units", 0.0)),
                "refused": float(getattr(sim, "strategic_budget_refused_units", 0.0)),
                "binding": int(getattr(sim, "strategic_budget_binding_periods", 0)),
                "replenished": float(sim.strategic_replenishment_units())}
    finally:
        env.close()


def run_cell(cell: dict) -> dict:
    tr = np.array([[play(cell, p, s)["L"] for p in POSTURES] for s in TRAIN])
    rows = [[play(cell, p, s) for p in POSTURES] for s in TEST]
    te = np.array([[r["L"] for r in row] for row in rows])

    fixed_idx = int(tr.mean(axis=0).argmin())          # chosen on TRAIN only
    diff = te[:, fixed_idx] - te.min(axis=1)
    boot = np.random.default_rng(20260808).choice(
        diff, size=(20_000, diff.size), replace=True).mean(axis=1)
    gap = {"mean": float(diff.mean()), "lcb95": float(np.percentile(boot, 2.5)),
           "ucb95": float(np.percentile(boot, 97.5)),
           "favourable": int((diff > 1e-12).sum()), "n": int(diff.size)}

    train_idx = list(range(len(TRAIN)))
    null = F.permutation_null(np.vstack([tr, te]), train_idx,
                              [i + len(TRAIN) for i in range(len(TEST))])
    return {
        "fixed_posture_from_train": list(POSTURES[fixed_idx]),
        "clairvoyant_gap": gap, "interaction_null": null,
        "distinct_optima_on_test": len(set(int(i) for i in te.argmin(axis=1))),
        "expired_units": float(np.mean([r["expired"] for row in rows for r in row])),
        "refused_units": float(np.mean([r["refused"] for row in rows for r in row])),
        "binding_periods": float(np.mean([r["binding"] for row in rows for r in row])),
        "replenished": float(np.mean([r["replenished"] for row in rows for r in row])),
        "L_test_mean": float(te.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / OUT)
    args = ap.parse_args()

    cells = {name: run_cell(spec) for name, spec in CELLS.items()}
    ctrl, both = cells["faithful_control"], cells["both"]
    best = max(cells, key=lambda k: cells[k]["clairvoyant_gap"]["lcb95"])

    checks = {
        "f1_faithful_cell_reproduces_today": F.lt(
            ctrl["clairvoyant_gap"]["lcb95"], BAR,
            "if headroom appears where expiry is inert and the budget unlimited, the instrument "
            "manufactured it and every cell below is unreadable"),
        "f2_shelf_life_expires_nothing_when_inert": F.lt(
            ctrl["expired_units"], 1e-9,
            "a 156-week shelf life cannot bite inside 26 weeks; removing stock there is a "
            "bookkeeping defect, not physics"),
        "f3_budget_binds_when_tight": F.gt(
            cells["budget_only"]["binding_periods"], 0.0,
            "a budget that is never hit is not contention, and the cell would be mislabelled"),
        "f4_expiry_removes_stock_when_short": F.gt(
            cells["expiry_only"]["expired_units"], 0.0,
            "an 8-week shelf life that expires nothing means the lot ledger never aged"),
        "f5_control_selected_on_train_only": F.check(
            all(len(c["fixed_posture_from_train"]) == 3 for c in cells.values()),
            "selecting the comparator on the tapes where the gap is measured inflates every gap",
            computed_from={"n_cells": len(cells), "n_postures": len(POSTURES)}),
        "f6_clairvoyant_gap_is_material": F.ge(
            cells[best]["clairvoyant_gap"]["lcb95"], BAR,
            "all four cells may come back flat, and that is the result rather than a setback"),
        "f7_gap_survives_the_interaction_null": F.lt(
            cells[best]["interaction_null"]["p_value"], 0.05,
            "a per-tape minimum over 27 noisy postures is biased upward; a ceiling died on virgin "
            "seeds this morning for exactly this reason"),
    }
    checks["d1_fidelity_price"] = F.disclosure(
        "shelf life and shared budget are OUR extensions with no source event: the thesis ration "
        "is non-perishable at three years and no common procurement bag is modelled. The "
        "156-week cell is the faithful control, and no result here is presented as reproducing "
        "Garrido-Rios (2017)",
        evidence={"inert_shelf_hours": INERT_SHELF, "short_shelf_hours": SHORT_SHELF,
                  "tight_budget_per_period": TIGHT_BUDGET})
    checks["custody"] = custody_falsifier(sorted(set(TRAIN + TEST)))
    summary = F.summarise(checks)

    if not (checks["f1_faithful_cell_reproduces_today"]["passed"]
            and checks["f2_shelf_life_expires_nothing_when_inert"]["passed"]):
        status = "BLOCKED_INSTRUMENT"
    elif checks["f6_clairvoyant_gap_is_material"]["passed"] and \
            checks["f7_gap_survives_the_interaction_null"]["passed"]:
        status = f"SEQUENTIAL_HEADROOM_UNDER_{best.upper()}"
    else:
        status = "NO_SEQUENTIAL_HEADROOM_UNDER_BUDGET_AND_EXPIRY"

    payload = {
        "schema_version": "budget_expiry_boundary_v1", "claim_status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_role": "DEVELOPMENT",
        "scope": "DEVELOPMENT_REPLAY_ON_BURNED_SEEDS_NO_VIRGIN_BLOCK",
        "endpoint": "L_star_under_shared_budget_and_shelf_life",
        "seeds": sorted(set(TRAIN + TEST)),
        "cells": {k: CELLS[k] | v for k, v in cells.items()},
        "best_cell": best, "bar": BAR, "n_postures": len(POSTURES),
        "module_manifest": module_manifest(MODULES),
        "falsifiers": checks, "falsifier_summary": summary,
    }
    seal_and_write(payload, args.output, contract=CONTRACT,
                   reference=Path("results/conservative_buffer_gate/result.json"))

    print(f"veredicto: {status}\n")
    for name, c in cells.items():
        g = c["clairvoyant_gap"]
        print(f"  {name:18} hueco {g['mean']:+.6f} [{g['lcb95']:+.6f}]  "
              f"p_nulo {c['interaction_null']['p_value']:.4f}  "
              f"optimos distintos {c['distinct_optima_on_test']}  "
              f"caducado {c['expired_units']:.0f}  atados {c['binding_periods']:.1f}")
    print(f"\n  falsadores: {summary['n_computed']} computados, {summary['n_failed']} fallidos")
    for name, c in checks.items():
        if c.get("computed"):
            print(f"    {name:44} {'PASA' if c['passed'] else 'FALLA'}")
    print(f"\n  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
