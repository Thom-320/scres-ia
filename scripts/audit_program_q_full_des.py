#!/usr/bin/env python3
"""Direct SimPy replay audit for every Program Q promoted trajectory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import CONTRACT, scheduler  # noqa: E402
from supply_chain.program_o_eval_custody import (  # noqa: E402
    sha256,
    verify_sha256_manifest,
    write_sha256_manifest,
)
from supply_chain.program_o_full_des import run_program_o_full_des_episode  # noqa: E402
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    MATRIX_KEYS,
    direct_full_des_vector,
)
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS  # noqa: E402


def calendar_index(calendar: tuple[int, ...]) -> int:
    value = 0
    for action in calendar:
        value = 4 * value + int(action)
    return value


def audit(*, evaluation: Path, shards: Path, atol: float) -> dict:
    result_path = evaluation / "result.json"
    evaluation_manifest = verify_sha256_manifest(
        evaluation, evaluation / "evaluation_files.sha256"
    )
    shard_manifest = verify_sha256_manifest(shards, evaluation / "shard_files.sha256")
    if "result.json" not in evaluation_manifest:
        raise RuntimeError("Program Q result is absent from evaluation manifest")
    result = json.loads(result_path.read_text())
    if result.get("schema_version") != "program_q_frozen_policy_replication_evaluation_v1":
        raise RuntimeError("Program Q evaluator schema mismatch")
    if result.get("contract_sha256") != sha256(CONTRACT):
        raise RuntimeError("Program Q result contract hash mismatch")
    seed_start, seed_end = map(int, result["seed_range"])
    seeds = list(range(seed_start, seed_end + 1))
    maximum_error = {key: 0.0 for key in MATRIX_KEYS}
    failures: list[dict[str, object]] = []
    replay_count = 0
    for cell in CONFIRMED_RET_CELLS:
        summary = result["cell_summaries"][cell.cell_id]
        audit_rows = result["trajectory_audits"][cell.cell_id]
        for tape_index, tape_seed in enumerate(seeds):
            calendars = {
                tuple(map(int, summary["best_open_loop_calendar"])),
                tuple(map(int, summary["best_classical_calendars"][tape_index])),
            }
            for seed_rows in audit_rows.values():
                calendars.add(tuple(map(int, seed_rows["calendars"][tape_index])))
            shard = shards / cell.cell_id / f"tape_{tape_seed}.npz"
            relative = shard.relative_to(shards).as_posix()
            if relative not in shard_manifest:
                raise RuntimeError(f"Program Q shard missing from manifest: {relative}")
            with np.load(shard, allow_pickle=False) as payload:
                for calendar in sorted(calendars):
                    replay_count += 1
                    simulation, panel = run_program_o_full_des_episode(
                        seed=tape_seed,
                        calendar=calendar,
                        scheduler=scheduler(),
                        regime_persistence=cell.regime_persistence,
                        dominant_share=cell.dominant_share,
                        downstream_freight_physics_mode="fixed_clock_physical_v1",
                    )
                    direct = direct_full_des_vector(simulation, panel)
                    index = calendar_index(calendar)
                    for key in MATRIX_KEYS:
                        expected = float(payload[f"open_loop__{key}"][index])
                        error = abs(float(direct[key]) - expected)
                        maximum_error[key] = max(maximum_error[key], error)
                        if error > atol:
                            failures.append(
                                {
                                    "cell": cell.cell_id,
                                    "tape_seed": tape_seed,
                                    "calendar": list(calendar),
                                    "metric": key,
                                    "direct": float(direct[key]),
                                    "transducer": expected,
                                    "absolute_error": error,
                                }
                            )
    return {
        "schema_version": "program_q_direct_full_des_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_result_sha256": sha256(result_path),
        "contract_sha256": sha256(CONTRACT),
        "seed_range": [seed_start, seed_end],
        "direct_unique_replays": replay_count,
        "atol": atol,
        "max_absolute_error_by_metric": maximum_error,
        "failure_count": len(failures),
        "failures": failures[:100],
        "passed": not failures,
        "terminal_status": (
            "DIRECT_FULL_DES_AUDIT_PASS_ELIGIBLE_FOR_ADJUDICATION"
            if not failures
            else "STOP_Q_DIRECT_FULL_DES_PARITY_FAILURE"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = audit(evaluation=args.evaluation, shards=args.shards, atol=args.atol)
    args.output.mkdir(parents=True)
    result = args.output / "independent_full_des_audit.json"
    result.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_sha256_manifest(args.output, [result], args.output / "audit_files.sha256")
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
