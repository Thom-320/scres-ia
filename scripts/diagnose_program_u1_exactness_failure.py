#!/usr/bin/env python3
"""Locate the fail-closed U1 transducer mismatch on already-burned tapes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.paper2_exhaustive_search.program_s_design import build_program_s_risk_tape
from research.paper2_exhaustive_search.program_s_transducer import (
    extract_program_s_skeleton,
    run_program_s_direct,
)
from scripts.run_program_s_s1_shard import make_cell, resolve_point
from scripts.run_program_u1_native_risk_stage_a import FRESH_SEEDS, _scheduler, tasks
from supply_chain.program_o_full_des_transducer import (
    MATRIX_KEYS,
    direct_full_des_vector,
    full_action_calendars,
    simulate_full_des_frontier,
)


CALENDAR_INDICES = (0, 21_845, 43_690, 65_535)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scheduler = _scheduler()
    calendars = full_action_calendars()
    checked = 0
    failures = []
    failed_masks: set[str] = set()
    for group_id, trajectory_id, point_id, product_cell, seed in tasks():
        identity = (
            f"g{group_id:02d}__t{trajectory_id:02d}__p{point_id:02d}"
            f"__{product_cell}__seed{seed}.npz"
        )
        if (args.failed_root / "matrices" / identity).exists():
            continue
        group, point = resolve_point(group_id, trajectory_id, point_id)
        cell = make_cell(group, point, product_cell)
        if cell.mask in failed_masks:
            continue
        built = build_program_s_risk_tape(cell, tape_id=seed, horizon_hours=8 * 168)
        reference = run_program_s_direct(
            seed=seed,
            calendar=[2] * 8,
            scheduler=scheduler,
            cell=cell,
            risk_event_tape=built["events"],
        )
        skeleton = extract_program_s_skeleton(reference)
        selected = calendars[np.asarray(CALENDAR_INDICES)]
        frontier = simulate_full_des_frontier(
            skeleton=skeleton, scheduler=scheduler, calendars=selected
        )
        task_max = 0.0
        worst = None
        differences = []
        for position, calendar_index in enumerate(CALENDAR_INDICES):
            direct_sim = run_program_s_direct(
                seed=seed,
                calendar=calendars[calendar_index].tolist(),
                scheduler=scheduler,
                cell=cell,
                risk_event_tape=built["events"],
            )
            direct = direct_full_des_vector(direct_sim, direct_sim.product_outcome_panel())
            for key in MATRIX_KEYS:
                error = abs(float(direct[key]) - float(frontier[key][position]))
                if error > task_max:
                    task_max = error
                    worst = {
                        "calendar_index": calendar_index,
                        "field": key,
                        "direct": float(direct[key]),
                        "transducer": float(frontier[key][position]),
                    }
                if error > 1e-12:
                    differences.append({
                        "calendar_index": calendar_index,
                        "field": key,
                        "abs_error": error,
                        "direct": float(direct[key]),
                        "transducer": float(frontier[key][position]),
                    })
        checked += 1
        if task_max > 1e-10:
            failures.append({
                "group": group_id,
                "trajectory": trajectory_id,
                "point": point_id,
                "product_cell": product_cell,
                "seed": seed,
                "mask": cell.mask,
                "cell_id": cell.cell_id,
                "max_abs_error": task_max,
                "worst": worst,
                "differences": sorted(
                    differences, key=lambda row: row["abs_error"], reverse=True
                ),
                "risk_event_count": len(built["events"]),
                "risk_event_tape_sha256": built["event_tape_sha256"],
            })
            failed_masks.add(cell.mask)
            if failed_masks == {"LOC_SURGE", "CROSS_ECHELON_SURGE"}:
                break
    payload = {
        "schema_version": "program_u1_exactness_failure_diagnostic_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "burned_seed_block": list(FRESH_SEEDS),
        "checked_missing_tasks": checked,
        "failures_by_mask": {row["mask"]: row for row in failures},
        "status": (
            "FAILURES_LOCALIZED_FOR_ALL_SCREENED_MASKS"
            if failed_masks == {"LOC_SURGE", "CROSS_ECHELON_SURGE"}
            else "PARTIAL_FAILURE_LOCALIZATION"
            if failures
            else "NO_FAILURE_IN_EXTREME_CALENDAR_DIAGNOSTIC"
        ),
        "claim_limit": "Instrument diagnostic on already-burned tapes; no risk-headroom estimate.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
