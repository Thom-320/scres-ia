#!/usr/bin/env python3
"""Direct-SimPy 2x2 Garrido-native R22/R24 discovery factorial.

This is an exploratory, no-claim screen on already-burned tapes.  The four
physical cells are the exact current/increased occurrence frequencies from
Garrido-Rios Table 6.12: R22 4032->1344 hours (x3), R24 672->336 hours (x2).
No learner return participates in cell selection.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.paper2_exhaustive_search.program_s_design import build_program_s_risk_tape  # noqa: E402
from research.paper2_exhaustive_search.program_s_transducer import run_program_s_direct  # noqa: E402
from scripts.run_program_u1_direct_classical_conversion import (  # noqa: E402
    adaptive_calendars,
    demand_counts,
    scheduler,
    static_calendars,
)
from supply_chain.program_o_full_des_transducer import direct_full_des_vector  # noqa: E402
from supply_chain.program_s_risk_interaction import ProgramSCell  # noqa: E402


BURNED_TAPES = tuple(range(7_430_001, 7_430_013))
PRODUCT_CELLS = {
    "rho75_share90": (0.75, 0.90),
    "rho90_share75": (0.90, 0.75),
    "rho90_share90": (0.90, 0.90),
}
RISK_LEVELS = {
    "r22_current__r24_current": (1.0, 1.0),
    "r22_increased__r24_current": (3.0, 1.0),
    "r22_current__r24_increased": (1.0, 2.0),
    "r22_increased__r24_increased": (3.0, 2.0),
}
METRICS = (
    "ret_visible",
    "worst_product_fill",
    "lost_orders",
    "gross_production_quantity",
    "mass_residual",
    "partition_residual",
)


def make_native_cell(r22_frequency: float, r24_frequency: float, product_cell: str) -> ProgramSCell:
    rho, share = PRODUCT_CELLS[product_cell]
    return ProgramSCell(
        stratum="THESIS_NATIVE_INDEPENDENT",
        mask="LOC_SURGE",
        coupling="independent",
        phi_by_risk={"R22": float(r22_frequency), "R24": float(r24_frequency)},
        psi_by_risk={"R22": 1.0, "R24": 1.0},
        r14_probability_multiplier=1.0,
        baseline_capacity_multiplier=1.0,
        regime_persistence=rho,
        dominant_share=share,
        alarm_lead_hours=0.0,
        alarm_balanced_accuracy=0.50,
    )


def paired_lcb(values: np.ndarray, *, seed: int = 20260721) -> float:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(10_000, len(values)))
    return float(np.quantile(values[draws].mean(axis=1), 0.025))


def evaluate_cell(level_id: str, product_cell: str) -> dict:
    r22_frequency, r24_frequency = RISK_LEVELS[level_id]
    cell = make_native_cell(r22_frequency, r24_frequency, product_cell)
    sched = scheduler()
    statics = static_calendars()
    per_tape: list[dict[str, dict[str, float]]] = []
    risk_counts: list[dict[str, int]] = []

    for tape in BURNED_TAPES:
        built = build_program_s_risk_tape(cell, tape_id=tape, horizon_hours=8 * 168)
        counts_by_risk = {risk_id: 0 for risk_id in ("R22", "R24")}
        for event in built["events"]:
            counts_by_risk[str(event["risk_id"])] += 1
        risk_counts.append(counts_by_risk)

        reference = run_program_s_direct(
            seed=tape,
            calendar=[0] * 8,
            scheduler=sched,
            cell=cell,
            risk_event_tape=built["events"],
        )
        policies = {
            **{f"static_{index}": calendar for index, calendar in enumerate(statics)},
            **adaptive_calendars(demand_counts(reference)),
        }
        metrics: dict[str, dict[str, float]] = {}
        for policy_id, calendar in policies.items():
            sim = run_program_s_direct(
                seed=tape,
                calendar=calendar,
                scheduler=sched,
                cell=cell,
                risk_event_tape=built["events"],
            )
            vector = direct_full_des_vector(sim, sim.product_outcome_panel())
            metrics[policy_id] = {key: float(vector[key]) for key in METRICS}
        per_tape.append(metrics)

    static_ids = sorted(name for name in per_tape[0] if name.startswith("static_"))
    rule_ids = sorted(name for name in per_tape[0] if not name.startswith("static_"))
    adaptive_delta: list[float] = []
    fill_delta: list[float] = []
    hpi: list[float] = []
    selected_rules: list[str] = []
    selected_statics: list[str] = []
    oracle_statics: list[str] = []

    for test in range(len(BURNED_TAPES)):
        train = [index for index in range(len(BURNED_TAPES)) if index != test]
        rule = max(rule_ids, key=lambda name: np.mean([per_tape[index][name]["ret_visible"] for index in train]))
        static = max(static_ids, key=lambda name: np.mean([per_tape[index][name]["ret_visible"] for index in train]))
        oracle = max(static_ids, key=lambda name: per_tape[test][name]["ret_visible"])
        adaptive_delta.append(per_tape[test][rule]["ret_visible"] - per_tape[test][static]["ret_visible"])
        fill_delta.append(per_tape[test][rule]["worst_product_fill"] - per_tape[test][static]["worst_product_fill"])
        hpi.append(per_tape[test][oracle]["ret_visible"] - per_tape[test][static]["ret_visible"])
        selected_rules.append(rule)
        selected_statics.append(static)
        oracle_statics.append(oracle)

    adaptive = np.asarray(adaptive_delta)
    hpi_array = np.asarray(hpi)
    mass_ok = all(
        abs(metrics[name]["mass_residual"]) <= 1e-8
        and abs(metrics[name]["partition_residual"]) <= 1e-8
        for metrics in per_tape
        for name in metrics
    )
    resources_exact = all(
        len({metrics[name]["gross_production_quantity"] for name in metrics}) == 1
        for metrics in per_tape
    )
    lost_orders_zero = all(
        metrics[name]["lost_orders"] == 0.0 for metrics in per_tape for name in metrics
    )
    passed = (
        paired_lcb(hpi_array) >= 0.02
        and paired_lcb(adaptive) >= 0.015
        and float(np.mean(adaptive > 0.0)) >= 0.70
        and float(np.mean(fill_delta)) >= -0.02
        and resources_exact
        and mass_ok
        and lost_orders_zero
        and len(set(oracle_statics)) >= 2
    )
    return {
        "level_id": level_id,
        "product_cell": product_cell,
        "physical": {
            "r22_occurrence_frequency_multiplier": r22_frequency,
            "r24_occurrence_frequency_multiplier": r24_frequency,
            "r22_duration_multiplier": 1.0,
            "r24_quantity_multiplier": 1.0,
        },
        "risk_event_counts": risk_counts,
        "h_pi_sampled_mean": float(hpi_array.mean()),
        "h_pi_sampled_lcb95": paired_lcb(hpi_array),
        "classical_h_obs_mean": float(adaptive.mean()),
        "classical_h_obs_lcb95": paired_lcb(adaptive),
        "favorable_fraction": float(np.mean(adaptive > 0.0)),
        "worst_product_delta_mean": float(np.mean(fill_delta)),
        "selected_rules": selected_rules,
        "selected_statics": selected_statics,
        "oracle_statics": oracle_statics,
        "ranking_reversal": len(set(oracle_statics)) >= 2,
        "resources_exact": resources_exact,
        "mass_conservation_passed": mass_ok,
        "lost_orders_zero": lost_orders_zero,
        "cell_pass": passed,
    }


def connected_components(rows: list[dict]) -> list[list[str]]:
    passing = {(row["level_id"], row["product_cell"]) for row in rows if row["cell_pass"]}
    components: list[list[str]] = []
    while passing:
        start = passing.pop()
        stack = [start]
        component = [start]
        while stack:
            level_id, product_cell = stack.pop()
            r22, r24 = RISK_LEVELS[level_id]
            neighbors = {
                (other_id, product_cell)
                for other_id, (other_r22, other_r24) in RISK_LEVELS.items()
                if (other_r22 == r22) ^ (other_r24 == r24)
            }
            for neighbor in list(neighbors & passing):
                passing.remove(neighbor)
                stack.append(neighbor)
                component.append(neighbor)
        components.append(sorted(f"{level}:{product}" for level, product in component))
    return sorted(components, key=lambda rows_: (-len(rows_), rows_))


def main() -> int:
    started = time.perf_counter()
    rows = [
        evaluate_cell(level_id, product_cell)
        for product_cell in PRODUCT_CELLS
        for level_id in RISK_LEVELS
    ]
    components = connected_components(rows)
    largest = max((len(component) for component in components), default=0)
    passing_products = sorted({row["product_cell"] for row in rows if row["cell_pass"]})
    verdict = (
        "PASS_Q2_GARRIDO_NATIVE_LOC_REGION"
        if largest >= 3 and len(passing_products) >= 2
        else "STOP_Q2_NO_GARRIDO_NATIVE_LOC_REGION"
    )
    payload = {
        "schema_version": "q2_garrido_native_loc_factorial_v1",
        "claim_status": "EXPLORATORY_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine": "direct_simpy_only",
        "source_levels": {
            "R22": "Table 6.12 current 4032h vs increased 1344h => frequency x1/x3",
            "R24": "Table 6.12 current 672h vs increased 336h => frequency x1/x2",
        },
        "burned_tapes": list(BURNED_TAPES),
        "new_scientific_seeds_opened": [],
        "learner_return_used_for_selection": False,
        "interpolated_or_extrapolated_severity_used": False,
        "rows": rows,
        "passing_cell_count": sum(bool(row["cell_pass"]) for row in rows),
        "connected_components": components,
        "largest_connected_component": largest,
        "passing_product_cells": passing_products,
        "elapsed_seconds": time.perf_counter() - started,
        "verdict": verdict,
        "learner_training_authorized": verdict == "PASS_Q2_GARRIDO_NATIVE_LOC_REGION",
    }
    output = ROOT / "results/program_q2/garrido_native_loc_factorial_v1/result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
