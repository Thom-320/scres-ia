#!/usr/bin/env python3
"""Twelve-tape connected-region check for the sole U1 classical candidate."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.paper2_exhaustive_search.program_s_design import build_program_s_risk_tape  # noqa: E402
from research.paper2_exhaustive_search.program_s_transducer import run_program_s_direct  # noqa: E402
from scripts.run_program_s_s1_shard import make_cell, resolve_point  # noqa: E402
from scripts.run_program_u1_direct_classical_conversion import adaptive_calendars, demand_counts, scheduler, static_calendars  # noqa: E402
from supply_chain.program_o_full_des_transducer import direct_full_des_vector  # noqa: E402

BURNED_TAPES = tuple(range(7_430_001, 7_430_013))
GROUP = 1
TRAJECTORY = 8
PRODUCT_CELL = "rho75_share90"


def lcb(values: np.ndarray) -> float:
    rng = np.random.default_rng(20260720)
    draws = rng.integers(0, len(values), size=(5000, len(values)))
    return float(np.quantile(values[draws].mean(axis=1), 0.025))


def main() -> int:
    sched = scheduler(); statics = static_calendars(); rows = []
    for point_index in range(5):
        group, point = resolve_point(GROUP, TRAJECTORY, point_index); cell = make_cell(group, point, PRODUCT_CELL); per_tape = []
        for tape in BURNED_TAPES:
            built = build_program_s_risk_tape(cell, tape_id=tape, horizon_hours=8 * 168)
            reference = run_program_s_direct(seed=tape, calendar=[0] * 8, scheduler=sched, cell=cell, risk_event_tape=built["events"])
            rules = adaptive_calendars(demand_counts(reference)); policies = {f"static_{i}": calendar for i, calendar in enumerate(statics)} | rules; metrics = {}
            for policy_id, calendar in policies.items():
                sim = run_program_s_direct(seed=tape, calendar=calendar, scheduler=sched, cell=cell, risk_event_tape=built["events"])
                vector = direct_full_des_vector(sim, sim.product_outcome_panel())
                metrics[policy_id] = {key: float(vector[key]) for key in ("ret_visible", "worst_product_fill", "lost_orders", "gross_production_quantity")}
            per_tape.append(metrics)
        rule_ids = sorted(name for name in per_tape[0] if not name.startswith("static_")); static_ids = sorted(name for name in per_tape[0] if name.startswith("static_")); adaptive_delta=[]; fill_delta=[]; hpi=[]
        for test in range(len(BURNED_TAPES)):
            train = [index for index in range(len(BURNED_TAPES)) if index != test]
            rule = max(rule_ids, key=lambda name: np.mean([per_tape[index][name]["ret_visible"] for index in train]))
            static = max(static_ids, key=lambda name: np.mean([per_tape[index][name]["ret_visible"] for index in train]))
            adaptive_delta.append(per_tape[test][rule]["ret_visible"] - per_tape[test][static]["ret_visible"])
            fill_delta.append(per_tape[test][rule]["worst_product_fill"] - per_tape[test][static]["worst_product_fill"])
            hpi.append(max(per_tape[test][name]["ret_visible"] for name in static_ids) - per_tape[test][static]["ret_visible"])
        adaptive_delta=np.asarray(adaptive_delta); hpi=np.asarray(hpi)
        rows.append({
            "point": point_index, "physical": point["physical"],
            "h_pi_sampled_mean": float(hpi.mean()), "h_pi_sampled_lcb95": lcb(hpi),
            "classical_h_obs_mean": float(adaptive_delta.mean()), "classical_h_obs_lcb95": lcb(adaptive_delta),
            "favorable_fraction": float(np.mean(adaptive_delta > 0.0)),
            "worst_product_delta_mean": float(np.mean(fill_delta)),
            "resources_exact": all(len({metrics[name]["gross_production_quantity"] for name in metrics}) == 1 for metrics in per_tape),
        })
    promoted = [row for row in rows if row["h_pi_sampled_lcb95"] >= 0.02 and row["classical_h_obs_lcb95"] >= 0.015 and row["worst_product_delta_mean"] >= -0.02 and row["resources_exact"]]
    payload = {
        "schema_version": "program_u1_direct_connected_region_v1",
        "claim_status": "EXPLORATORY_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine": "direct_simpy_only", "burned_tapes": list(BURNED_TAPES), "new_scientific_seeds_opened": [],
        "group": GROUP, "trajectory": TRAJECTORY, "product_cell": PRODUCT_CELL,
        "rows": rows, "promoted_point_count": len(promoted), "promoted_points": [row["point"] for row in promoted],
        "verdict": "PASS_U1_CONNECTED_REGION" if len(promoted) >= 3 else "STOP_U1_NO_CONNECTED_CLASSICAL_CONVERSION_REGION",
        "hybrid_training_authorized": False,
    }
    output = ROOT / "results/program_u1/direct_connected_region_v1/result.json"
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
