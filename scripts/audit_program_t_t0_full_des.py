#!/usr/bin/env python3
"""Direct-SimPy parity audit for selected T0 calendars."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.program_o_full_des import run_program_o_full_des_episode  # noqa: E402
from supply_chain.program_o_full_des_transducer import MATRIX_KEYS, direct_full_des_vector  # noqa: E402
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("adjudication", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--all-evaluation", action="store_true")
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    adjudication = json.loads(args.adjudication.read_text())
    maximum = {key: 0.0 for key in MATRIX_KEYS}
    failures = []
    replay_count = 0
    for cell in CONFIRMED_RET_CELLS:
        selected = adjudication["cells"][cell.cell_id]["selected_comparator"]
        rows = matrix["cells"][cell.cell_id]["comparators"][selected]
        tape_indices = range(24, 48) if args.all_evaluation else (24, 47)
        for tape_index in tape_indices:
            tape = 7_490_001 + tape_index
            calendar = tuple(map(int, rows["calendar"][tape_index]))
            sim, panel = run_program_o_full_des_episode(
                seed=tape,
                calendar=calendar,
                scheduler=scheduler(),
                regime_persistence=cell.regime_persistence,
                dominant_share=cell.dominant_share,
                downstream_freight_physics_mode="fixed_clock_physical_v1",
            )
            direct = direct_full_des_vector(sim, panel)
            replay_count += 1
            for key in MATRIX_KEYS:
                if key not in rows:
                    continue
                error = abs(float(direct[key]) - float(rows[key][tape_index]))
                maximum[key] = max(maximum[key], error)
                if error > args.atol:
                    failures.append({"cell": cell.cell_id, "tape": tape, "metric": key, "error": error})
    out = {
        "schema_version": "program_t_t0_direct_full_des_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "replay_count": replay_count,
        "maximum_absolute_error": maximum,
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
