#!/usr/bin/env python3
"""Direct-SimPy feasibility/liveness preflight for the last Garrido-native route."""

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
from scripts.run_program_s_s1_shard import make_cell, resolve_point  # noqa: E402
from supply_chain.program_o_full_des_transducer import direct_full_des_vector  # noqa: E402
from supply_chain.program_s_risk_interaction import PROGRAM_S_MASKS  # noqa: E402

DESIGN = ROOT / "research/paper2_exhaustive_search/program_s_native_morris_design_v1_1.json"
PARENT = ROOT / "contracts/program_o_full_des_hpi_translation_v1.json"
BURNED_SEED = 7_430_001
CALENDARS = {
    "all_h": [0] * 8,
    "minority_c": [1] * 8,
    "minority_h": [2] * 8,
    "all_c": [3] * 8,
    "alternating": [0, 3] * 4,
}


def scheduler():
    parent = json.loads(PARENT.read_text())
    return parent["action"]["within_week_schedulers"][parent["action"]["primary_scheduler"]]


def least_severe_points():
    design = json.loads(DESIGN.read_text())
    selected = []
    for group_index, group in enumerate(design["groups"]):
        candidates = []
        for trajectory_index, trajectory in enumerate(group["trajectories"]):
            for point_index, point in enumerate(trajectory["points"]):
                distance = sum(
                    abs(float(value) - 1.0)
                    for key, value in point["physical"].items()
                    if key != "baseline_capacity"
                )
                candidates.append((distance, trajectory_index, point_index))
        distance, trajectory_index, point_index = min(candidates)
        selected.append((group_index, trajectory_index, point_index, group["mask"], distance))
    return selected


def main() -> int:
    sched = scheduler(); design = json.loads(DESIGN.read_text()); rows = []; started = time.perf_counter()
    for group_index, trajectory_index, point_index, mask, distance in least_severe_points():
        group, point = resolve_point(group_index, trajectory_index, point_index)
        for product_cell in sorted(design["product_cells"]):
            cell = make_cell(group, point, product_cell)
            built = build_program_s_risk_tape(cell, tape_id=BURNED_SEED, horizon_hours=8 * 168)
            metrics = {}; observed_risks = set(); runtimes = []
            for name, calendar in CALENDARS.items():
                run_started = time.perf_counter()
                sim = run_program_s_direct(seed=BURNED_SEED, calendar=calendar, scheduler=sched, cell=cell, risk_event_tape=built["events"])
                runtimes.append(time.perf_counter() - run_started)
                vector = direct_full_des_vector(sim, sim.product_outcome_panel())
                metrics[name] = {key: float(vector[key]) for key in ("ret_visible", "worst_product_fill", "lost_orders", "gross_production_quantity", "mass_residual", "partition_residual")}
                observed_risks.update(event.risk_id for event in sim.risk_events)
            production = [row["gross_production_quantity"] for row in metrics.values()]
            rows.append({
                "mask": mask, "group": group_index, "trajectory": trajectory_index, "point": point_index,
                "least_severe_distance": distance, "product_cell": product_cell,
                "risk_ids_expected": list(PROGRAM_S_MASKS[cell.mask]), "risk_ids_observed": sorted(observed_risks),
                "all_expected_risks_observed_on_fixture": set(PROGRAM_S_MASKS[cell.mask]).issubset(observed_risks),
                "resource_range": max(production) - min(production),
                "ret_range": max(row["ret_visible"] for row in metrics.values()) - min(row["ret_visible"] for row in metrics.values()),
                "mean_direct_seconds": float(np.mean(runtimes)), "metrics": metrics,
            })
    s0 = json.loads((ROOT / "results/program_s/s0_preflight_v1/result.json").read_text())
    live = bool(s0["deterministic_liveness"]["pass"])
    resources = all(row["resource_range"] <= 1e-12 for row in rows)
    mean_seconds = float(np.mean([row["mean_direct_seconds"] for row in rows]))
    payload = {
        "schema_version": "program_u1_direct_simpy_preflight_v1",
        "claim_status": "EXPLORATORY_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "burned_fixture_seed": BURNED_SEED,
        "new_scientific_seeds_opened": [],
        "engine": "direct_simpy_only",
        "masks": sorted({row["mask"] for row in rows}),
        "calendar_count": len(CALENDARS), "cell_count": len(rows),
        "deterministic_liveness_inherited_from_s0": live,
        "fixture_incidence_is_not_a_liveness_gate": True,
        "resources_exact": resources,
        "mean_direct_seconds": mean_seconds,
        "projected_seconds_for_3_tapes_12_points_3_cells_32_calendars": mean_seconds * 3 * 12 * 3 * 32,
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
        "verdict": "PASS_U1_DIRECT_PREFLIGHT_READY_FOR_BOUNDED_DISCOVERY" if live and resources else "STOP_U1_DIRECT_PREFLIGHT_LIVENESS_OR_RESOURCE_FAILURE",
        "scientific_screen_authorized": False,
    }
    output = ROOT / "results/program_u1/direct_simpy_preflight_v1/result.json"
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
