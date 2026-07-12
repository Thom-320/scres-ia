#!/usr/bin/env python3
"""Fail-closed physical preflight for Program F before calibration tapes."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.program_f import ACTIONS, ConstantPortfolio, CONTEXTS, materialize_tape, run_policy


def duration(row: dict, risk_ids: set[str]) -> float:
    return sum(
        float(event["realized_duration_hours"])
        for event in row["damage_events"] if event["risk_id"] in risk_ids
    )


def main() -> int:
    output = Path("results/program_f/preflight")
    output.mkdir(parents=True, exist_ok=True)
    tapes = [
        materialize_tape(939100 + 10 * i + replicate, context, "disposable-preflight", weeks=12)
        for i, context in enumerate(CONTEXTS) for replicate in range(6)
    ]
    rows = []
    for tape in tapes:
        for action in ACTIONS:
            result = run_policy(tape, ConstantPortfolio(action))
            rows.append({"tape_id": tape["tape_id"], "context": tape["first_context"], "action": action, **result})
    def selected(context: str, action: tuple[int, int, int]) -> list[dict]:
        return [row for row in rows if row["context"] == context and tuple(row["action"]) == action]

    equipment_m2 = selected("equipment_pressure", (2, 0, 0))
    equipment_m0 = selected("equipment_pressure", (0, 2, 0))
    interdiction_t2 = selected("interdiction_campaign", (0, 2, 0))
    interdiction_t0 = selected("interdiction_campaign", (2, 0, 0))
    surge_r2 = selected("mission_surge", (0, 0, 2))
    surge_r0 = selected("mission_surge", (2, 0, 0))
    total_duration = lambda group, ids: sum(duration(row, ids) for row in group)
    total = lambda group, key: sum(float(row[key]) for row in group)
    gates = {
        "equal_budget": len({row["total_token_hours"] for row in rows}) == 1,
        "mass_conservation": max(abs(row["mass_residual"]) for row in rows) < 1e-6,
        "threat_crn": all(
            len({row["threat_sha256"] for row in rows if row["tape_id"] == tape["tape_id"]}) == 1
            for tape in tapes
        ),
        "manufacturing_live": total_duration(equipment_m2, {"R11"}) < total_duration(equipment_m0, {"R11"}),
        "manufacturing_cost_live": total(equipment_m2, "maintenance_downtime_hours") > total(equipment_m0, "maintenance_downtime_hours"),
        "transport_eligible_events_observed": total_duration(interdiction_t0, {"R22", "R23"}) > 0,
        "transport_live": total_duration(interdiction_t2, {"R22", "R23"}) < total_duration(interdiction_t0, {"R22", "R23"}),
        "reserve_live": total(surge_r2, "reserve_units_issued") > total(surge_r0, "reserve_units_issued"),
        "reserve_finite": all(row["reserve_units_issued"] <= 10000 + row["reserve_units_replenished"] + 1e-9 for row in surge_r2),
    }
    verdict = {
        "gate": "PROGRAM_F_PHYSICAL_PREFLIGHT",
        "disposable_tapes": len(tapes), "policies": len(ACTIONS),
        "gates": gates, "all_pass": all(gates.values()),
        "calibration_tapes_opened": 0, "ppo_trained": False,
        "diagnostics": {
            "r11_hours_m2": total_duration(equipment_m2, {"R11"}),
            "r11_hours_m0": total_duration(equipment_m0, {"R11"}),
            "transport_hours_t2": total_duration(interdiction_t2, {"R22", "R23"}),
            "transport_hours_t0": total_duration(interdiction_t0, {"R22", "R23"}),
            "reserve_issued_r2": total(surge_r2, "reserve_units_issued"),
            "reserve_issued_r0": total(surge_r0, "reserve_units_issued"),
        },
        "interpretation": "PASS_PROGRAM_F_PHYSICAL_PREFLIGHT" if all(gates.values()) else "STOP_PROGRAM_F_PHYSICAL_PREFLIGHT",
    }
    (output / "verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
