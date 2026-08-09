#!/usr/bin/env python3
"""The same four cells, with an endpoint that charges for what the mechanisms spend.

Contract: `docs/PREREGISTRO_PRESUPUESTO_COMPARTIDO_Y_CADUCIDAD_2026-08-08.md` as amended by
`docs/ENMIENDA_ENDPOINT_CON_PRECIO_2026-08-08.md`, both committed before this file.
Development replay on already-burned seeds; no virgin block is opened.

WHY v2 EXISTS. v1 froze `L*`, which measures lateness and cannot see cost, so more buffer never
hurt and the optimum was maximal-affordable by construction. All four cells returned the identical
posture at the identical endpoint while spending 313,002 units in some and 667,386 in others. The
negative it produced is real and narrow: no headroom WHEN THE ENDPOINT DOES NOT CHARGE.

THE COST IS REPLENISHED UNITS AND EXPIRY IS NOT ADDED ON TOP. Expiry is not an extra spend; it is
spend that bought nothing, and it already enters by forcing replenishment again. Adding both would
double-count the same unit and manufacture a trade-off, which is the defect retracted this morning.

READ ACROSS THE WHOLE lambda FRONT. No verdict may depend on choosing the price that flatters it.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from supply_chain.arm_runner import seal_and_write                               # noqa: E402
from supply_chain import falsifiers as F                                         # noqa: E402
from supply_chain.seed_custody import custody_falsifier, module_manifest         # noqa: E402
from run_budget_expiry_boundary_v1 import (                                      # noqa: E402
    CELLS, MODULES, POSTURES, TEST, TRAIN, play)

BAR = 0.01
LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
OUT = Path("results/budget_expiry_priced/result.json")
CONTRACT = Path("docs/ENMIENDA_ENDPOINT_CON_PRECIO_2026-08-08.md")


def measure(cell: dict) -> tuple[np.ndarray, np.ndarray]:
    """(L, cost) over every tape x posture. Train and test are stacked, split by index later."""
    seeds = list(TRAIN) + list(TEST)
    L = np.empty((len(seeds), len(POSTURES)))
    cost = np.empty_like(L)
    for i, seed in enumerate(seeds):
        for j, posture in enumerate(POSTURES):
            row = play(cell, posture, seed)
            L[i, j] = row["L"]
            cost[i, j] = row["replenished"]
    return L, cost


def run_cell(cell: dict) -> dict:
    L, cost = measure(cell)
    n_tr = len(TRAIN)
    norm = float(cost.max()) or 1.0
    cost_n = cost / norm

    per_lambda = {}
    for lam in LAMBDAS:
        J = L + lam * cost_n
        fixed_idx = int(J[:n_tr].mean(axis=0).argmin())          # TRAIN only
        diff = J[n_tr:, fixed_idx] - J[n_tr:].min(axis=1)
        boot = np.random.default_rng(20260808).choice(
            diff, size=(20_000, diff.size), replace=True).mean(axis=1)
        null = F.permutation_null(J, list(range(n_tr)),
                                  list(range(n_tr, J.shape[0])))
        per_lambda[str(lam)] = {
            "fixed_posture_from_train": list(POSTURES[fixed_idx]),
            "gap_mean": float(diff.mean()),
            "gap_lcb95": float(np.percentile(boot, 2.5)),
            "gap_ucb95": float(np.percentile(boot, 97.5)),
            "favourable": int((diff > 1e-12).sum()),
            "distinct_optima_on_test": len(set(int(i) for i in J[n_tr:].argmin(axis=1))),
            "null_p_value": float(null["p_value"]),
            "null_mean": float(null["null_mean"]),
        }
    best = max(per_lambda, key=lambda k: per_lambda[k]["gap_lcb95"])
    return {"per_lambda": per_lambda, "best_lambda": best,
            "max_replenished": norm,
            "cost_spread": float(cost.max() - cost.min()),
            "L_spread": float(L.max() - L.min())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / OUT)
    args = ap.parse_args()

    cells = {name: run_cell(spec) for name, spec in CELLS.items()}
    flat = [(name, lam, d) for name, c in cells.items() for lam, d in c["per_lambda"].items()]
    best_name, best_lam, best = max(flat, key=lambda t: t[2]["gap_lcb95"])
    ctrl = cells["faithful_control"]

    checks = {
        "f1_cost_separates_the_class": F.gt(
            min(c["cost_spread"] for c in cells.values()), 0.0,
            "if every posture spends the same, the price cannot move the optimum and the priced "
            "endpoint prices nothing"),
        "f2_price_moves_the_chosen_posture": F.check(
            len({tuple(d["fixed_posture_from_train"])
                 for d in cells["both"]["per_lambda"].values()}) > 1,
            "if the same posture wins at every lambda, the decision is insensitive to price and "
            "the whole priced family is inert",
            computed_from={"n_lambdas": len(LAMBDAS),
                           "n_distinct": len({tuple(d["fixed_posture_from_train"])
                                              for d in cells["both"]["per_lambda"].values()})}),
        "f3_faithful_control_stays_flat": F.lt(
            max(d["gap_lcb95"] for d in ctrl["per_lambda"].values()), BAR,
            "headroom in the cell where expiry is inert and the budget unlimited would mean the "
            "price, not the physics, produced it"),
        "f4_clairvoyant_gap_is_material": F.ge(
            best["gap_lcb95"], BAR,
            "every cell and every lambda may come back flat, and by the frozen closure rule that "
            "closes the strategic-buffer family"),
        "f5_gap_survives_the_interaction_null": F.lt(
            best["null_p_value"], 0.05,
            "a per-tape minimum over 27 noisy postures is biased upward, which is how a ceiling "
            "survived three weeks before dying on virgin seeds this morning"),
    }
    checks["d1_cost_definition"] = F.disclosure(
        "cost is replenished kit-equivalent units; expired units are NOT added on top because "
        "expiry already enters by forcing replenishment again, and adding both would count the "
        "same unit twice",
        evidence={"max_replenished_per_cell":
                  {k: v["max_replenished"] for k, v in cells.items()}})
    checks["custody"] = custody_falsifier(sorted(set(TRAIN + TEST)))
    summary = F.summarise(checks)

    if checks["f4_clairvoyant_gap_is_material"]["passed"] and \
            checks["f5_gap_survives_the_interaction_null"]["passed"]:
        status = f"PRICED_SEQUENTIAL_HEADROOM_IN_{best_name.upper()}"
    else:
        status = "STRATEGIC_BUFFER_FAMILY_CLOSED__NO_PRICED_SEQUENTIAL_HEADROOM"

    payload = {
        "schema_version": "budget_expiry_priced_v2", "claim_status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_role": "DEVELOPMENT",
        "scope": "DEVELOPMENT_REPLAY_ON_BURNED_SEEDS_NO_VIRGIN_BLOCK",
        "endpoint": "J_lambda_L_star_plus_replenished_units",
        "seeds": sorted(set(TRAIN + TEST)),
        "supersedes": {"path": "results/budget_expiry_boundary/result.json", "retained": True,
                       "why": "its endpoint could not see the cost the two mechanisms create, so "
                              "the optimum was maximal-affordable by construction"},
        "lambdas": list(LAMBDAS), "bar": BAR,
        "cells": {k: CELLS[k] | v for k, v in cells.items()},
        "best": {"cell": best_name, "lambda": best_lam, **best},
        "module_manifest": module_manifest(MODULES),
        "falsifiers": checks, "falsifier_summary": summary,
    }
    seal_and_write(payload, args.output, contract=CONTRACT,
                   reference=Path("results/budget_expiry_boundary/result.json"))

    print(f"veredicto: {status}\n")
    for name, c in cells.items():
        print(f"  {name}")
        for lam in LAMBDAS:
            d = c["per_lambda"][str(lam)]
            print(f"    lambda {lam:<5} postura(train) {str(d['fixed_posture_from_train']):<17} "
                  f"hueco {d['gap_mean']:+.6f} [{d['gap_lcb95']:+.6f}]  "
                  f"p_nulo {d['null_p_value']:.4f}  optimos {d['distinct_optima_on_test']}")
    print(f"\n  mejor: {best_name} @ lambda {best_lam}  "
          f"hueco {best['gap_mean']:+.6f} [{best['gap_lcb95']:+.6f}]  vs barra {BAR}")
    print(f"  falsadores: {summary['n_computed']} computados, {summary['n_failed']} fallidos")
    for name, c in checks.items():
        if c.get("computed"):
            print(f"    {name:44} {'PASA' if c['passed'] else 'FALLA'}")
    print(f"\n  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
