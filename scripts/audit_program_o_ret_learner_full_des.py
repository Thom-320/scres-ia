#!/usr/bin/env python3
"""Independently replay every promoted Program O-R trajectory in direct SimPy."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supply_chain.program_o_full_des import run_program_o_full_des_episode  # noqa: E402
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    MATRIX_KEYS,
    direct_full_des_vector,
)
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS  # noqa: E402
from supply_chain.program_o_ret_freeze import verify_execution_freeze  # noqa: E402
from supply_chain.program_o_eval_custody import (  # noqa: E402
    sha256,
    verify_sha256_manifest,
    write_sha256_manifest,
)


CONTRACT = ROOT / "contracts/program_o_ret_only_learner_v1.json"


def scheduler() -> dict[str, list[str]]:
    parent = json.loads(
        (ROOT / "contracts/program_o_full_des_hpi_translation_v1.json").read_text()
    )
    key = parent["action"]["primary_scheduler"]
    return parent["action"]["within_week_schedulers"][key]


def calendar_index(calendar: tuple[int, ...]) -> int:
    value = 0
    for action in calendar:
        value = 4 * value + int(action)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    contract = json.loads(CONTRACT.read_text())
    if contract["status"] != "FROZEN_BEFORE_748_SCIENTIFIC_SEEDS":
        raise SystemExit("audit blocked until source/execution freeze")
    verify_execution_freeze(ROOT, CONTRACT)
    result_path = args.evaluation / "result.json"
    evaluation_files = verify_sha256_manifest(
        args.evaluation, args.evaluation / "evaluation_files.sha256"
    )
    raw_files = verify_sha256_manifest(
        args.evaluation, args.evaluation / "raw_files.sha256"
    )
    if "result.json" not in evaluation_files:
        raise SystemExit("audit blocked: result.json absent from evaluation manifest")
    expected_raw = {
        relative for relative in evaluation_files if relative.startswith("raw_calendar_matrix/")
    }
    if expected_raw != set(raw_files):
        raise SystemExit("audit blocked: raw and evaluation manifests disagree")
    evaluation = json.loads(result_path.read_text())
    if evaluation.get("schema_version") != "program_o_ret_only_learner_evaluation_v1_2":
        raise SystemExit("audit blocked: evaluator schema is not frozen v1.2")
    if evaluation.get("raw_matrix_count") != evaluation.get("raw_matrix_expected_count"):
        raise SystemExit("audit blocked: incomplete raw matrix count")
    seeds = list(range(int(evaluation["seed_range"][0]), int(evaluation["seed_range"][1]) + 1))
    learner_seeds = list(map(int, contract["learner"]["learner_seeds"]))
    cell_lookup = {cell.cell_id: cell for cell in CONFIRMED_RET_CELLS}
    maximum_error = {key: 0.0 for key in MATRIX_KEYS}
    failures: list[dict[str, object]] = []
    unique_replays = 0
    for cell_id, summary in evaluation["cell_summaries"].items():
        cell = cell_lookup[cell_id]
        calendars_by_tape: list[set[tuple[int, ...]]] = [set() for _ in seeds]
        open_calendar = tuple(map(int, summary["best_open_loop_calendar"]))
        for tape_index in range(len(seeds)):
            calendars_by_tape[tape_index].add(open_calendar)
            calendars_by_tape[tape_index].add(
                tuple(map(int, summary["best_classical_calendars"][tape_index]))
            )
        for learner_seed in learner_seeds:
            rows = evaluation["trajectory_audits"][cell_id][str(learner_seed)]["calendars"]
            if len(rows) != len(seeds):
                raise AssertionError("learner calendar/tape length mismatch")
            for tape_index, row in enumerate(rows):
                calendars_by_tape[tape_index].add(tuple(map(int, row)))
        for tape_index, tape_seed in enumerate(seeds):
            shard_path = (
                args.evaluation / "raw_calendar_matrix" / cell_id / f"tape_{tape_seed}.npz"
            )
            with np.load(shard_path) as shard:
                for calendar in sorted(calendars_by_tape[tape_index]):
                    unique_replays += 1
                    sim, panel = run_program_o_full_des_episode(
                        seed=tape_seed,
                        calendar=calendar,
                        scheduler=scheduler(),
                        regime_persistence=cell.regime_persistence,
                        dominant_share=cell.dominant_share,
                        downstream_freight_physics_mode="fixed_clock_physical_v1",
                    )
                    direct = direct_full_des_vector(sim, panel)
                    index = calendar_index(calendar)
                    for key in MATRIX_KEYS:
                        error = abs(float(direct[key]) - float(shard[key][index]))
                        maximum_error[key] = max(maximum_error[key], error)
                        if error > args.atol:
                            failures.append(
                                {
                                    "cell": cell_id,
                                    "tape_seed": tape_seed,
                                    "calendar": list(calendar),
                                    "metric": key,
                                    "direct": float(direct[key]),
                                    "transducer": float(shard[key][index]),
                                    "absolute_error": error,
                                }
                            )
    args.output.mkdir(parents=True)
    passed = not failures
    audit = {
        "schema_version": "program_o_ret_only_learner_direct_full_des_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_result": str(result_path),
        "evaluation_result_sha256": sha256(result_path),
        "evaluation_files_manifest_sha256": sha256(
            args.evaluation / "evaluation_files.sha256"
        ),
        "raw_files_manifest_sha256": sha256(args.evaluation / "raw_files.sha256"),
        "phase": evaluation["phase"],
        "seed_range": evaluation["seed_range"],
        "direct_unique_replays": unique_replays,
        "atol": args.atol,
        "max_absolute_error_by_metric": maximum_error,
        "failure_count": len(failures),
        "failures": failures[:100],
        "passed": passed,
        "terminal_status": (
            "DIRECT_FULL_DES_AUDIT_PASS_ELIGIBLE_FOR_INDEPENDENT_VERDICT"
            if passed
            else "STOP_DIRECT_FULL_DES_PARITY_FAILURE"
        ),
        "claim_boundary": "This audit establishes replay parity only; it cannot promote a failed statistical gate.",
    }
    audit_path = args.output / "independent_full_des_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    write_sha256_manifest(
        args.output, [audit_path], args.output / "audit_files.sha256"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
