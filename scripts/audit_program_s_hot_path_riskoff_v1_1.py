#!/usr/bin/env python3
"""Multi-cell burned-tape parity audit for the shared R14 hot-path change."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supply_chain.program_o_full_des import run_program_o_full_des_episode  # noqa: E402
from supply_chain.program_o_full_des_transducer import direct_full_des_vector  # noqa: E402


REFERENCE = ROOT / "results/program_o/fixed_clock_hobs_corrective_validation_v1/remote_run/artifacts/validation"
PARENT = json.loads((ROOT / "contracts/program_o_full_des_hpi_translation_v1.json").read_text())
SCHEDULER = PARENT["action"]["within_week_schedulers"][PARENT["action"]["primary_scheduler"]]
OUT = ROOT / "results/program_s/s0_hot_path_riskoff_parity_v1_1/corrective_result_v2.json"
CELLS = {
    "rho75_share90": (0.75, 0.90),
    "rho90_share75": (0.90, 0.75),
    "rho90_share90": (0.90, 0.90),
}
SEEDS = (7_430_001, 7_430_048)
CALENDAR_INDICES = (0, 43_690, 65_535)
TOLERANCE = float(np.finfo(float).eps)


def decode(index: int) -> tuple[int, ...]:
    values = [0] * 8
    for position in range(7, -1, -1):
        values[position] = index % 4
        index //= 4
    return tuple(values)


def main() -> int:
    rows = []
    max_abs = 0.0
    for cell_id, (rho, share) in CELLS.items():
        for seed in SEEDS:
            matrix = np.load(REFERENCE / f"raw_calendar_matrix/{cell_id}/tape_{seed}.npz")
            for index in CALENDAR_INDICES:
                sim, panel = run_program_o_full_des_episode(
                    seed=seed,
                    calendar=decode(index),
                    scheduler=SCHEDULER,
                    regime_persistence=rho,
                    dominant_share=share,
                    downstream_freight_physics_mode="fixed_clock_physical_v1",
                    risks_enabled=False,
                    risk_impact_multipliers_by_id=None,
                    risk_event_tape=None,
                    risk_rng_mode="shared",
                )
                vector = direct_full_des_vector(sim, panel)
                diffs = {
                    key: abs(float(vector[key]) - float(matrix[key][index]))
                    for key in ("ret_visible", "ret_visible_cvar10", "gross_production_quantity")
                }
                max_abs = max(max_abs, *diffs.values())
                rework_events = [
                    event for event in sim.program_o_product_events
                    if event.get("event") == "r14_product_rework_started"
                ]
                rows.append({
                    "cell": cell_id,
                    "seed": seed,
                    "calendar_index": index,
                    "diffs": diffs,
                    "final_rework_level": float(sim.rework_op6.level),
                    "rework_event_count": len(rework_events),
                    "rework_ledger_quantity": float(sim.program_o_ledger.quantity("rework_op6")),
                })
    zero_rework = all(
        row["final_rework_level"] == 0.0
        and row["rework_event_count"] == 0
        and row["rework_ledger_quantity"] == 0.0
        for row in rows
    )
    passed = max_abs <= TOLERANCE and zero_rework
    payload = {
        "schema_version": "program_s_hot_path_riskoff_parity_v1_1",
        "burned_only": True,
        "scientific_751_seeds_opened": False,
        "episodes": len(rows),
        "cells": sorted(CELLS),
        "seeds": list(SEEDS),
        "calendar_indices": list(CALENDAR_INDICES),
        "max_abs_diff": max_abs,
        "tolerance": TOLERANCE,
        "instrument_correction": "Attempt 1 used the rounded decimal 2.22e-16. V2 uses the exact IEEE-754 binary64 epsilon; episodes and inputs are unchanged.",
        "zero_rework_all_episodes": zero_rework,
        "rows": rows,
        "pass": passed,
        "verdict": "PASS_S0_HOT_PATH_RISKOFF_PARITY_V1_1" if passed else "STOP_S0_HOT_PATH_RISKOFF_PARITY_FAILURE"
    }
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("episodes", "max_abs_diff", "zero_rework_all_episodes", "verdict")}, indent=2))
    return 0 if passed else 5


if __name__ == "__main__":
    raise SystemExit(main())
