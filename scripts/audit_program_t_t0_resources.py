#!/usr/bin/env python3
"""Audit matched demand, mass and scheduled resources for T0 versus Q."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.program_o_full_des_transducer import extract_full_des_skeleton, simulate_full_des_frontier  # noqa: E402
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS  # noqa: E402

RESOURCE_KEYS = ("gross_policy_batch_slots", "gross_production_quantity", "charged_daily_dispatch_slots", "charged_downstream_vehicle_hours")
IDENTITY_KEYS = ("generated_orders", "mass_residual", "partition_residual", "aggregate_ration_residual", "raw_material_residual")
Q_RESULT = ROOT / "results/program_q/confirmation_v1_20260718/artifacts/confirmation/evaluation/result.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("adjudication", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    adjudication = json.loads(args.adjudication.read_text())
    q = json.loads(Q_RESULT.read_text())
    maximum = {key: 0.0 for key in (*RESOURCE_KEYS, "generated_orders")}
    invariant_maximum = {key: 0.0 for key in IDENTITY_KEYS[1:]}
    sched = scheduler()
    rows = 0
    for cell in CONFIRMED_RET_CELLS:
        selected = adjudication["cells"][cell.cell_id]["selected_comparator"]
        mpc_rows = matrix["cells"][cell.cell_id]["comparators"][selected]
        learners = q["trajectory_audits"][cell.cell_id]
        learner_seeds = sorted(learners, key=int)
        for tape_index in range(24, 48):
            tape = 7_490_001 + tape_index
            skeleton, _ = extract_full_des_skeleton(seed=tape, scheduler=sched, regime_persistence=cell.regime_persistence, dominant_share=cell.dominant_share, downstream_freight_physics_mode="fixed_clock_physical_v1")
            q_calendars = np.asarray([learners[str(seed)]["calendars"][tape_index] for seed in learner_seeds], dtype=np.uint8)
            q_metrics = simulate_full_des_frontier(skeleton=skeleton, scheduler=sched, calendars=q_calendars)
            mpc_metrics = simulate_full_des_frontier(skeleton=skeleton, scheduler=sched, calendars=np.asarray([mpc_rows["calendar"][tape_index]], dtype=np.uint8))
            for key in (*RESOURCE_KEYS, "generated_orders"):
                maximum[key] = max(maximum[key], float(np.max(np.abs(q_metrics[key] - mpc_metrics[key][0]))))
            for key in IDENTITY_KEYS[1:]:
                invariant_maximum[key] = max(invariant_maximum[key], float(np.max(np.abs(q_metrics[key]))), abs(float(mpc_metrics[key][0])))
            rows += 1
    out = {"schema_version": "program_t_t0_resource_identity_audit_v1", "created_at": datetime.now(timezone.utc).isoformat(), "tape_cell_rows": rows, "maximum_q_mpc_difference": maximum, "maximum_invariant_residual": invariant_maximum, "passed": max(maximum.values()) <= 1e-12 and max(invariant_maximum.values()) <= 1e-9}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
