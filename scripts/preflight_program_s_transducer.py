#!/usr/bin/env python3
"""Burned-seed Program S transducer gate; opens no 751 scientific tape."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.paper2_exhaustive_search.program_s_transducer import (  # noqa: E402
    exact_short_horizon_gate,
    extract_program_s_skeleton,
    run_program_s_direct,
)
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    MATRIX_KEYS,
    direct_full_des_vector,
    full_action_calendars,
    simulate_full_des_frontier,
)
from supply_chain.program_s_risk_interaction import ProgramSCell  # noqa: E402


PARENT = json.loads(
    (ROOT / "contracts/program_o_full_des_hpi_translation_v1.json").read_text()
)
CONTRACT = json.loads(
    (ROOT / "contracts/program_s_product_mix_risk_interaction_gsa_v1.json").read_text()
)
SCHEDULER = PARENT["action"]["within_week_schedulers"][
    PARENT["action"]["primary_scheduler"]
]
OUT = ROOT / "results/program_s/s1_transducer_preflight_v1/result.json"
BURNED_SEED = 7430001


def cell(mask: str) -> ProgramSCell:
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
        alarm_lead_hours=0,
        alarm_balanced_accuracy=0.5,
    )


def event(risk_id: str, start: float, duration: float, ops, magnitude=1.0, unit="incidents"):
    return {
        "risk_id": risk_id,
        "start_time": float(start),
        "end_time": float(start + duration),
        "duration": float(duration),
        "affected_ops": list(ops),
        "magnitude": float(magnitude),
        "unit": unit,
        "description": f"program_s_transducer_forced_{risk_id}",
    }


def forced_tape(mask: str, weeks: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for week in range(int(weeks)):
        offset = week * 168.0
        if mask == "PRODUCTION_QUALITY_SURGE":
            rows.extend(
                [
                    event("R24", offset + 40, 0, [13], 2500, "rations"),
                    event("R11", offset + 48, 2, [5]),
                    event("R11", offset + 72, 2, [6]),
                    event("R14", offset + 120, 0, [7], 30, "defective_products"),
                ]
            )
        elif mask == "LOC_SURGE":
            rows.append(event("R24", offset + 40, 0, [13], 2500, "rations"))
            for index, op_id in enumerate((4, 8, 10, 12)):
                rows.append(event("R22", offset + 48 + 24 * index, 4, [op_id]))
        else:
            rows.extend(
                [
                    event("R24", offset + 40, 0, [13], 2500, "rations"),
                    event("R21", offset + 48, 12, [3, 5, 6, 7, 9]),
                    event("R23", offset + 96, 12, [11]),
                ]
            )
    return rows


def h8_replay(mask: str) -> dict[str, Any]:
    current = cell(mask)
    tape = forced_tape(mask, 8)
    reference = run_program_s_direct(
        seed=BURNED_SEED,
        calendar=[2] * 8,
        scheduler=SCHEDULER,
        cell=current,
        risk_event_tape=tape,
    )
    skeleton = extract_program_s_skeleton(reference)
    calendars = full_action_calendars(8)
    frontier = simulate_full_des_frontier(skeleton=skeleton, scheduler=SCHEDULER)
    oracle_index = int(np.argmax(frontier["ret_visible"]))
    replay_indices = sorted({0, 65535, oracle_index, 31743, 39599})
    max_error = 0.0
    rows = []
    for index in replay_indices:
        sim = run_program_s_direct(
            seed=BURNED_SEED,
            calendar=calendars[index].tolist(),
            scheduler=SCHEDULER,
            cell=current,
            risk_event_tape=tape,
        )
        direct = direct_full_des_vector(sim, sim.product_outcome_panel())
        errors = {
            key: abs(float(direct[key]) - float(frontier[key][index]))
            for key in MATRIX_KEYS
        }
        max_error = max(max_error, max(errors.values()))
        rows.append({"calendar_index": index, "max_abs_error": max(errors.values())})
    return {
        "oracle_index": oracle_index,
        "replays": rows,
        "max_matrix_abs_error": max_error,
        "pass": max_error <= 1e-10,
    }


def main() -> int:
    masks: dict[str, Any] = {}
    for mask in CONTRACT["physical_masks"]:
        try:
            short = exact_short_horizon_gate(
                seed=BURNED_SEED,
                scheduler=SCHEDULER,
                cell=cell(mask),
                risk_tapes_by_horizon={h: forced_tape(mask, h) for h in (1, 2, 3)},
            )
            long = h8_replay(mask) if short["pass"] else {"pass": False, "not_run": True}
            masks[mask] = {
                "short_horizon": short,
                "h8_replay": long,
                "eligible": bool(short["pass"] and long["pass"]),
                "error": None,
            }
        except Exception as exc:
            masks[mask] = {
                "short_horizon": None,
                "h8_replay": None,
                "eligible": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    eligible = [mask for mask, row in masks.items() if row["eligible"]]
    payload = {
        "schema_version": "program_s_transducer_preflight_result_v1",
        "burned_seed": BURNED_SEED,
        "scientific_seed_blocks_opened": [],
        "masks": masks,
        "eligible_masks": eligible,
        "pass": bool(eligible),
        "verdict": (
            "PASS_S1_TRANSDUCER_PREFLIGHT_ALL_MASKS_ELIGIBLE"
            if len(eligible) == len(CONTRACT["physical_masks"])
            else "PASS_S1_TRANSDUCER_PREFLIGHT_SOME_MASKS_ELIGIBLE"
            if eligible
            else "STOP_S1_ALL_MASKS_ACTION_DEPENDENT_OR_INEXACT"
        ),
        "scientific_seed_authorization": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if eligible else 1


if __name__ == "__main__":
    raise SystemExit(main())
