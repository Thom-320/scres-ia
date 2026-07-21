#!/usr/bin/env python3
"""Bounded direct-SimPy Garrido-native headroom screen on already-burned tapes."""

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

DESIGN = ROOT / "research/paper2_exhaustive_search/program_s_native_morris_design_v1_1.json"
PARENT = ROOT / "contracts/program_o_full_des_hpi_translation_v1.json"
BURNED_TAPES = (7_430_001, 7_430_002, 7_430_003)


def scheduler():
    parent = json.loads(PARENT.read_text())
    return parent["action"]["within_week_schedulers"][parent["action"]["primary_scheduler"]]


def calendar_panel() -> tuple[tuple[int, ...], ...]:
    rows = {(value,) * 8 for value in range(4)}
    rows.update({(0, 3) * 4, (3, 0) * 4, (1, 2) * 4, (2, 1) * 4})
    rng = np.random.default_rng(20260720)
    while len(rows) < 32:
        rows.add(tuple(map(int, rng.integers(0, 4, size=8))))
    return tuple(sorted(rows))


def selected_points():
    design = json.loads(DESIGN.read_text()); output = []
    for group_index, group in enumerate(design["groups"]):
        candidates = []
        for trajectory_index, trajectory in enumerate(group["trajectories"]):
            for point_index, point in enumerate(trajectory["points"]):
                distance = sum(abs(float(value) - 1.0) for key, value in point["physical"].items() if key != "baseline_capacity")
                candidates.append((distance, trajectory_index, point_index))
        for distance, trajectory_index, point_index in sorted(candidates)[:4]:
            output.append((group_index, trajectory_index, point_index, group["mask"], distance))
    return output


def main() -> int:
    sched = scheduler(); design = json.loads(DESIGN.read_text()); calendars = calendar_panel(); rows = []; started = time.perf_counter()
    for group_index, trajectory_index, point_index, mask, distance in selected_points():
        group, point = resolve_point(group_index, trajectory_index, point_index)
        for product_cell in sorted(design["product_cells"]):
            cell = make_cell(group, point, product_cell)
            ret = np.empty((len(BURNED_TAPES), len(calendars))); fill = np.empty_like(ret); lost = np.empty_like(ret); resource = np.empty_like(ret)
            for tape_index, tape in enumerate(BURNED_TAPES):
                built = build_program_s_risk_tape(cell, tape_id=tape, horizon_hours=8 * 168)
                for calendar_index, calendar in enumerate(calendars):
                    sim = run_program_s_direct(seed=tape, calendar=calendar, scheduler=sched, cell=cell, risk_event_tape=built["events"])
                    vector = direct_full_des_vector(sim, sim.product_outcome_panel())
                    ret[tape_index, calendar_index] = vector["ret_visible"]
                    fill[tape_index, calendar_index] = vector["worst_product_fill"]
                    lost[tape_index, calendar_index] = vector["lost_orders"]
                    resource[tape_index, calendar_index] = vector["gross_production_quantity"]
            static_index = int(np.argmax(ret.mean(axis=0))); oracle_index = np.argmax(ret, axis=1)
            static = ret[:, static_index]; oracle = ret[np.arange(len(BURNED_TAPES)), oracle_index]
            rows.append({
                "mask": mask, "group": group_index, "trajectory": trajectory_index, "point": point_index,
                "distance": distance, "product_cell": product_cell,
                "h_pi_sampled_mean": float(np.mean(oracle - static)),
                "best_static_calendar_index": static_index,
                "oracle_calendar_indices": list(map(int, oracle_index)),
                "ranking_reversal": len(set(map(int, oracle_index))) >= 2,
                "worst_product_fill_mean_at_static": float(fill[:, static_index].mean()),
                "lost_orders_max_at_static": float(lost[:, static_index].max()),
                "resource_range": float(resource.max() - resource.min()),
            })
    candidates = [row for row in rows if row["h_pi_sampled_mean"] >= 0.02 and row["ranking_reversal"] and row["resource_range"] <= 1e-12]
    candidates.sort(key=lambda row: (row["distance"], -row["h_pi_sampled_mean"], row["mask"], row["product_cell"]))
    payload = {
        "schema_version": "program_u1_direct_bounded_discovery_v1",
        "claim_status": "EXPLORATORY_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine": "direct_simpy_only",
        "burned_tapes": list(BURNED_TAPES), "new_scientific_seeds_opened": [],
        "calendar_panel_size": len(calendars), "point_count": len(rows),
        "selection_uses_learner_returns": False,
        "elapsed_seconds": time.perf_counter() - started,
        "candidate_count": len(candidates), "candidate_rows": candidates[:12], "rows": rows,
        "verdict": "CANDIDATES_REQUIRE_CLASSICAL_OBSERVABLE_CONVERSION" if candidates else "STOP_U1_DIRECT_BOUNDED_NO_SAMPLED_HEADROOM",
        "hybrid_training_authorized": False,
    }
    output = ROOT / "results/program_u1/direct_bounded_discovery_v1/result.json"
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
