#!/usr/bin/env python3
"""Execute Program S S0 without opening any 751/752/753 scientific seed.

Deterministic fixtures establish causal liveness.  A separate burned-tape
panel reports native incidence without allowing rare-event absence to veto
liveness.  Risk-off identity is checked against the custodied Program O matrix.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.paper2_exhaustive_search.program_s_design import (  # noqa: E402
    build_program_s_risk_tape,
)
from supply_chain.program_o_full_des import run_program_o_full_des_episode  # noqa: E402
from supply_chain.program_o_full_des_transducer import direct_full_des_vector  # noqa: E402
from supply_chain.program_s_risk_interaction import (  # noqa: E402
    ProgramSCell,
    ProgramSRiskAwareSimulation,
)


CONTRACT = json.loads(
    (ROOT / "contracts/program_s_product_mix_risk_interaction_gsa_v1.json").read_text()
)
PARENT = json.loads(
    (ROOT / "contracts/program_o_full_des_hpi_translation_v1.json").read_text()
)
SCHEDULER = PARENT["action"]["within_week_schedulers"][
    PARENT["action"]["primary_scheduler"]
]
REFERENCE = (
    ROOT
    / "results/program_o/fixed_clock_hobs_corrective_validation_v1/remote_run/artifacts/validation"
)
OUT = ROOT / "results/program_s/s0_preflight_v1"
BURNED_FIXTURE_SEED = 7430001
BURNED_INCIDENCE_SEEDS = tuple(range(7430001, 7430013))


def decode_calendar(index: int, weeks: int = 8) -> tuple[int, ...]:
    actions = [0] * weeks
    value = int(index)
    for position in range(weeks - 1, -1, -1):
        actions[position] = value % 4
        value //= 4
    if value:
        raise ValueError("calendar index exceeds horizon")
    return tuple(actions)


def make_cell(mask: str) -> ProgramSCell:
    risks = CONTRACT["physical_masks"][mask]
    return ProgramSCell(
        stratum="THESIS_NATIVE_INDEPENDENT",
        mask=mask,
        coupling="independent",
        phi_by_risk={risk: 1.0 for risk in risks},
        psi_by_risk={risk: 1.0 for risk in risks},
        r14_probability_multiplier=1.0,
        baseline_capacity_multiplier=1.0,
        regime_persistence=0.75,
        dominant_share=0.90,
        alarm_lead_hours=0.0,
        alarm_balanced_accuracy=0.50,
    )


def event(risk_id: str, affected_ops: list[int], *, start=48.0, duration=4.0, magnitude=1.0, unit="incidents"):
    return {
        "risk_id": risk_id,
        "start_time": float(start),
        "end_time": float(start + duration),
        "duration": float(duration),
        "affected_ops": list(affected_ops),
        "magnitude": float(magnitude),
        "unit": unit,
        "description": f"program_s_s0_forced_{risk_id}",
    }


def fixture_episode(mask: str, rows: list[dict[str, Any]], *, weeks=1):
    return ProgramSRiskAwareSimulation(
        seed=BURNED_FIXTURE_SEED,
        calendar=[2] * int(weeks),
        scheduler=SCHEDULER,
        cell=make_cell(mask),
        risk_event_tape=rows,
        downstream_freight_physics_mode="fixed_clock_physical_v1",
    ).run_contract()


def deterministic_liveness() -> dict[str, Any]:
    fixtures: dict[str, tuple[str, list[dict[str, Any]], int]] = {
        "R11_op5": ("PRODUCTION_QUALITY_SURGE", [event("R11", [5])], 1),
        "R11_op6": ("PRODUCTION_QUALITY_SURGE", [event("R11", [6])], 1),
        "R14_op7_rework": (
            "PRODUCTION_QUALITY_SURGE",
            [event("R14", [7], start=120, duration=0, magnitude=30, unit="defective_products")],
            2,
        ),
        "R21_simultaneous": (
            "CROSS_ECHELON_SURGE",
            [event("R21", [3, 5, 6, 7, 9], duration=12)],
            1,
        ),
        "R22_op4": ("LOC_SURGE", [event("R22", [4], duration=8)], 1),
        "R22_op8": ("LOC_SURGE", [event("R22", [8], duration=8)], 1),
        "R22_op10": ("LOC_SURGE", [event("R22", [10], duration=8)], 1),
        "R22_op12": ("LOC_SURGE", [event("R22", [12], duration=8)], 1),
        "R23_op11": ("CROSS_ECHELON_SURGE", [event("R23", [11], duration=12)], 1),
        "R24_product_preservation": (
            "PRODUCTION_QUALITY_SURGE",
            [event("R24", [13], start=40, duration=0, magnitude=2500, unit="rations")],
            1,
        ),
    }
    results: dict[str, Any] = {}
    for fixture_id, (mask, rows, weeks) in fixtures.items():
        sim = fixture_episode(mask, rows, weeks=weeks)
        expected = rows[0]
        matches = [
            record
            for record in sim.risk_events
            if record.risk_id == expected["risk_id"]
            and set(record.affected_ops) == set(expected["affected_ops"])
        ]
        panel = sim.product_outcome_panel()
        row: dict[str, Any] = {
            "risk_id": expected["risk_id"],
            "expected_ops": expected["affected_ops"],
            "matching_event_count": len(matches),
            "event_start_relative": (
                None
                if not matches
                else float(matches[0].start_time) - float(sim.program_o_decision_start)
            ),
            "product_mass_residual": panel["conservation"]["max_abs_product_residual"],
            "partition_residual": panel["conservation"]["max_abs_partition_residual"],
        }
        if expected["risk_id"] == "R14":
            started = [
                item for item in sim.program_o_product_events
                if item["event"] == "r14_product_rework_started"
            ]
            returned = [
                item for item in sim.program_o_product_events
                if item["event"] == "r14_rework_returned_to_pending_batch"
            ]
            row["rework_started_quantity"] = sum(item["quantity"] for item in started)
            row["rework_returned_quantity"] = sum(item["quantity"] for item in returned)
            row["rework_product_labels"] = sorted(
                {token["product_id"] for item in started for token in item["tokens"]}
            )
        if expected["risk_id"] == "R24":
            contingent = [order for order in sim.orders if bool(order.contingent)]
            row["contingent_orders"] = len(contingent)
            row["contingent_product_labels"] = sorted(
                {order.requested_product_id for order in contingent}
            )
            row["r24_generated_quantity"] = float(sim.r24_generated_surge_quantity)
        row["pass"] = bool(
            len(matches) == 1
            and row["product_mass_residual"] <= 1e-8
            and row["partition_residual"] <= 1e-8
        )
        if expected["risk_id"] == "R14":
            row["pass"] = row["pass"] and abs(
                row["rework_started_quantity"] - row["rework_returned_quantity"]
            ) <= 1e-8
        if expected["risk_id"] == "R24":
            row["pass"] = row["pass"] and bool(row["contingent_orders"]) and set(
                row["contingent_product_labels"]
            ).issubset({"P_C", "P_H"})
        results[fixture_id] = row
    return {"fixtures": results, "pass": all(row["pass"] for row in results.values())}


def riskoff_identity() -> dict[str, Any]:
    result = json.loads((REFERENCE / "result.json").read_text())
    cell_id = "rho90_share90"
    static_index = int(result["cells"][cell_id]["static_index"])
    calendar = decode_calendar(static_index)
    reference = np.load(REFERENCE / f"raw_calendar_matrix/{cell_id}/tape_{BURNED_FIXTURE_SEED}.npz")
    common = dict(
        seed=BURNED_FIXTURE_SEED,
        calendar=calendar,
        scheduler=SCHEDULER,
        regime_persistence=0.90,
        dominant_share=0.90,
        downstream_freight_physics_mode="fixed_clock_physical_v1",
    )
    default_sim, default_panel = run_program_o_full_des_episode(**common)
    sim, panel = run_program_o_full_des_episode(
        **common,
        risks_enabled=False,
        risk_impact_multipliers_by_id=None,
        risk_event_tape=None,
        risk_rng_mode="shared",
    )
    default_vector = direct_full_des_vector(default_sim, default_panel)
    vector = direct_full_des_vector(sim, panel)
    keys = ("ret_visible", "ret_visible_cvar10", "gross_production_quantity")
    checks = {
        key: {
            "program_s_adapter": float(vector[key]),
            "custodied_program_o": float(reference[key][static_index]),
            "abs_diff": abs(float(vector[key]) - float(reference[key][static_index])),
        }
        for key in keys
    }
    return {
        "cell": cell_id,
        "seed": BURNED_FIXTURE_SEED,
        "static_index": static_index,
        "checks": checks,
        "adapter_defaults_bitwise_equal": vector == default_vector,
        "custodied_numeric_tolerance": 1e-12,
        "custodied_one_ulp_disclosure": (
            "The parent c2fa5cb checkout already differs from the older custodied "
            "ret_visible cell by one IEEE-754 ULP; Program S introduces zero additional drift."
        ),
        "pass": bool(
            vector == default_vector
            and all(row["abs_diff"] <= 1e-12 for row in checks.values())
        ),
    }


def thesis_incidence() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for mask in CONTRACT["physical_masks"]:
        current_cell = make_cell(mask)
        counts = {risk_id: [] for risk_id in CONTRACT["physical_masks"][mask]}
        operations = {risk_id: set() for risk_id in counts}
        for seed in BURNED_INCIDENCE_SEEDS:
            tape = build_program_s_risk_tape(
                current_cell, tape_id=seed, horizon_hours=8 * 168
            )
            for risk_id in counts:
                events = [row for row in tape["events"] if row["risk_id"] == risk_id]
                counts[risk_id].append(len(events))
                for item in events:
                    operations[risk_id].update(item["affected_ops"])
        rows[mask] = {
            risk_id: {
                "per_tape_counts": values,
                "total": int(sum(values)),
                "zero_tapes": int(sum(value == 0 for value in values)),
                "operations_seen": sorted(operations[risk_id]),
                "liveness_gate": "NOT_APPLICABLE_INCIDENT_RATE_ONLY",
            }
            for risk_id, values in counts.items()
        }
    return {
        "burned_seed_range": [BURNED_INCIDENCE_SEEDS[0], BURNED_INCIDENCE_SEEDS[-1]],
        "masks": rows,
        "rare_zero_events_do_not_invalidate_deterministic_liveness": True,
    }


def main() -> int:
    liveness = deterministic_liveness()
    identity = riskoff_identity()
    incidence = thesis_incidence()
    passed = bool(liveness["pass"] and identity["pass"])
    verdict = (
        "PASS_S0_RISK_ADAPTER_LIVE_AND_RISKOFF_IDENTICAL"
        if passed
        else "STOP_S0_RISK_ADAPTER_OR_PARITY_FAILURE"
    )
    payload = {
        "schema_version": "program_s_s0_preflight_result_v1",
        "contract_id": CONTRACT["contract_id"],
        "scientific_seed_blocks_opened": [],
        "burned_fixture_seed": BURNED_FIXTURE_SEED,
        "riskoff_identity": identity,
        "deterministic_liveness": liveness,
        "thesis_incidence_report_only": incidence,
        "verdict": verdict,
        "pass": passed,
        "next_authority": (
            "FREEZE_AND_INDEPENDENTLY_AUDIT_S1_DESIGN_BEFORE_7510001"
            if passed
            else "NO_PROGRAM_S_SCREEN_AUTHORIZED"
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
