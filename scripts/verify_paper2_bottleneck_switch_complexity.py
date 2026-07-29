#!/usr/bin/env python3
"""Reverify, optionally by full replay, the <=3-switch calibration gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_paper2_bottleneck_full_frontier import (
    PRIMARY_CONTRACT_PATH,
    _contract_seed_rows,
    calendar_index,
)
from scripts.search_paper2_bottleneck_switch_complexity import (
    CONTRACT_PATH,
    DEPENDENCIES,
    EXPECTED_COUNTS,
    OUTPUT_SCHEMA,
    PROGRESS_SCHEMA,
    ROOT,
    atomic_json,
    calendar_name,
    candidate_calendars,
    evaluate_calibration_tape,
    evaluate_selected_tape,
    exact_selection,
    file_sha256,
    json_sha256,
    parallel_map,
    reserve_artifact,
    switch_count,
)
from supply_chain.paper2_bottleneck import materialize_tape


VERIFICATION_SCHEMA = "paper2_bottleneck_switch_complexity_verification_v1"


def git_blob_sha256(commit: str, relative: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return sha256(result.stdout).hexdigest() if result.returncode == 0 else None


def same_float(left: Any, right: Any) -> bool:
    try:
        return float(left).hex() == float(right).hex()
    except (TypeError, ValueError):
        return False


def validate_payload(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    content = dict(payload)
    recorded = content.pop("content_sha256", None)
    if recorded != json_sha256(content):
        failures.append("content hash mismatch")
    if payload.get("schema_version") != OUTPUT_SCHEMA:
        failures.append("result schema mismatch")
    if payload.get("contract_id") != "paper2_bottleneck_switch_complexity_screen_v1":
        failures.append("contract id mismatch")
    if payload.get("contract_sha256") != file_sha256(CONTRACT_PATH):
        failures.append("contract hash mismatch")
    if payload.get("canonical_metric") != "ret_excel_visible_v1":
        failures.append("canonical metric mismatch")
    if payload.get("launch_git_status_porcelain") != []:
        failures.append("result was not launched from a clean worktree")

    expected_dependencies = {
        str(path.relative_to(ROOT)): file_sha256(path) for path in DEPENDENCIES
    }
    if payload.get("dependency_sha256") != expected_dependencies:
        failures.append("dependency hashes mismatch")
    commit = str(payload.get("git_head", ""))
    if not commit or any(
        git_blob_sha256(commit, relative) != digest
        for relative, digest in expected_dependencies.items()
    ):
        failures.append("producing commit does not contain dependency bytes")

    candidates = candidate_calendars()
    family = payload.get("candidate_family", {})
    candidate_rows = family.get("rows")
    if (
        family.get("count") != 11_611
        or family.get("counts_by_switches")
        != {str(key): value for key, value in EXPECTED_COUNTS.items()}
        or not isinstance(candidate_rows, list)
        or len(candidate_rows) != 11_611
    ):
        failures.append("candidate family is incomplete")
        candidate_rows = []
    if family.get("rows_sha256") != json_sha256(candidate_rows):
        failures.append("candidate row hash mismatch")

    calibration = payload.get("calibration", {})
    matrix_hex = calibration.get("score_matrix_float_hex")
    matrix: np.ndarray | None = None
    if (
        not isinstance(matrix_hex, list)
        or len(matrix_hex) != 60
        or any(
            not isinstance(row, list) or len(row) != 11_611
            for row in matrix_hex
        )
    ):
        failures.append("score matrix shape mismatch")
    else:
        try:
            matrix = np.asarray(
                [[float.fromhex(value) for value in row] for row in matrix_hex],
                dtype=float,
            )
        except (TypeError, ValueError):
            failures.append("score matrix contains invalid float hex")
        if calibration.get("score_matrix_float_hex_sha256") != json_sha256(matrix_hex):
            failures.append("score matrix hash mismatch")

    primary = json.loads(PRIMARY_CONTRACT_PATH.read_text())
    specs = _contract_seed_rows(primary, "calibration")
    tape_rows = calibration.get("tapes")
    if not isinstance(tape_rows, list) or len(tape_rows) != 60:
        failures.append("calibration tape rows are incomplete")
        tape_rows = []
    for index, (spec, row) in enumerate(zip(specs, tape_rows)):
        tape = materialize_tape(
            int(spec["seed"]), str(spec["context_0"]), "calibration", weeks=24
        )
        if (
            row.get("index") != index
            or row.get("seed") != spec["seed"]
            or row.get("context") != spec["context_0"]
            or row.get("split") != "calibration"
            or row.get("tape_sha256") != tape["threat_sha256"]
            or matrix_hex is not None
            and index < len(matrix_hex)
            and row.get("scores_float_hex_sha256") != json_sha256(matrix_hex[index])
        ):
            failures.append(f"calibration tape identity mismatch at {index}")

    selected_index = calibration.get("selected_index")
    if matrix is not None and isinstance(selected_index, int):
        exact_sums, expected_index = exact_selection(matrix)
        if selected_index != expected_index:
            failures.append("selected index is not the exact winner")
        selected = candidates[expected_index]
        expected_switches = switch_count(selected)
        if (
            calibration.get("selected_calendar") != calendar_name(selected)
            or calibration.get("selected_full_frontier_calendar_index")
            != calendar_index(selected)
            or calibration.get("selected_switch_count") != expected_switches
            or calibration.get("selected_exact_sum_numerator")
            != str(exact_sums[expected_index].numerator)
            or calibration.get("selected_exact_sum_denominator")
            != str(exact_sums[expected_index].denominator)
            or not same_float(
                calibration.get("selected_mean_ret_excel"),
                exact_sums[expected_index] / 60,
            )
        ):
            failures.append("selected calendar fields mismatch")
        if len(candidate_rows) == 11_611:
            for index, (sequence, row) in enumerate(zip(candidates, candidate_rows)):
                if (
                    row.get("candidate_index") != index
                    or row.get("calendar") != calendar_name(sequence)
                    or row.get("switch_count") != switch_count(sequence)
                    or row.get("full_frontier_calendar_index")
                    != calendar_index(sequence)
                    or row.get("exact_sum_numerator")
                    != str(exact_sums[index].numerator)
                    or row.get("exact_sum_denominator")
                    != str(exact_sums[index].denominator)
                ):
                    failures.append(f"candidate aggregate mismatch at {index}")
                    break
        expected_status = (
            "CALIBRATION_SWITCH_COMPLEXITY_BOUNDARY_ACTIVE"
            if expected_switches == 3
            else "CALIBRATION_SWITCH_COMPLEXITY_INTERIOR"
        )
        if (
            payload.get("scientific_status") != expected_status
            or payload.get("boundary_hit") is not (expected_switches == 3)
        ):
            failures.append("boundary decision mismatch")
        ledgers = calibration.get(
            "selected_calendar_guardrail_and_resource_ledgers"
        )
        if not isinstance(ledgers, list) or len(ledgers) != 60:
            failures.append("selected ledger rows are incomplete")
            ledgers = []
        if calibration.get(
            "selected_calendar_guardrail_and_resource_ledgers_sha256"
        ) != json_sha256(ledgers):
            failures.append("selected ledger hash mismatch")
        required = set(json.loads(CONTRACT_PATH.read_text())["selected_calendar_ledger"]["required"])
        for index, row in enumerate(ledgers):
            guardrails = row.get("guardrails_and_resources", {})
            if (
                row.get("index") != index
                or row.get("seed") != specs[index]["seed"]
                or row.get("context") != specs[index]["context_0"]
                or row.get("split") != "calibration"
                or row.get("calendar") != calendar_name(selected)
                or row.get("ret_excel_float_hex") != matrix_hex[index][selected_index]
                or not required.issubset(guardrails)
                or row.get("guardrails_and_resources_sha256")
                != json_sha256(guardrails)
                or float(guardrails.get("total_token_hours", -1)) != 4032.0
                or float(guardrails.get("mass_residual", -1)) != 0.0
            ):
                failures.append(f"selected ledger mismatch at {index}")
                break
    else:
        failures.append("selected index or matrix is unavailable")

    if payload.get("locked_tapes_accessed") is not False:
        failures.append("result claims locked-tape access")
    if payload.get("virgin_tapes_accessed") is not False:
        failures.append("result claims virgin-tape access")
    for field in (
        "h_pi_computed",
        "h_obs_computed",
        "w24_authorized",
        "learner_authorized",
        "paper2_authorized",
        "paper3_authorized",
    ):
        if payload.get(field) is not False:
            failures.append(f"forbidden claim field is not false: {field}")
    return sorted(set(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--deep", action="store_true")
    args = parser.parse_args()
    launch_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        text=True, stdout=subprocess.PIPE,
    ).stdout.splitlines()
    if launch_status:
        parser.error("verification requires a clean immutable worktree")
    launch_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        text=True, stdout=subprocess.PIPE,
    ).stdout.strip()
    result_path = args.result.resolve(strict=True)
    payload = json.loads(result_path.read_text())
    failures = validate_payload(payload)
    started = time.perf_counter()
    deep_summary: dict[str, Any] = {"performed": False}
    progress_path: Path | None = None
    if args.deep and not failures:
        if args.progress is None:
            parser.error("--deep requires --progress")
        progress_path = reserve_artifact(args.progress, label="verification progress")
        candidates = candidate_calendars()
        primary = json.loads(PRIMARY_CONTRACT_PATH.read_text())
        specs = _contract_seed_rows(primary, "calibration")
        jobs = [
            (index, int(row["seed"]), str(row["context_0"]), candidates)
            for index, row in enumerate(specs)
        ]
        replayed = parallel_map(
            jobs,
            evaluate_calibration_tape,
            workers=args.workers,
            progress_path=progress_path,
            stage="deep_replay_all_candidates",
            started=started,
        )
        replayed_hex = [
            [float(value).hex() for value in row["scores"]] for row in replayed
        ]
        if replayed_hex != payload["calibration"]["score_matrix_float_hex"]:
            failures.append("deep score-matrix replay mismatch")
        selected = candidates[int(payload["calibration"]["selected_index"])]
        selected_jobs = [
            (index, int(row["seed"]), str(row["context_0"]), selected)
            for index, row in enumerate(specs)
        ]
        ledgers = parallel_map(
            selected_jobs,
            evaluate_selected_tape,
            workers=args.workers,
            progress_path=progress_path,
            stage="deep_replay_selected_ledgers",
            started=started,
        )
        if ledgers != payload["calibration"][
            "selected_calendar_guardrail_and_resource_ledgers"
        ]:
            failures.append("deep selected-ledger replay mismatch")
        deep_summary = {
            "performed": True,
            "calibration_tapes_replayed": 60,
            "candidate_rollouts_replayed": 60 * 11_611,
            "selected_ledger_rollouts_replayed": 60,
        }

    final_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        text=True, stdout=subprocess.PIPE,
    ).stdout.splitlines()
    final_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        text=True, stdout=subprocess.PIPE,
    ).stdout.strip()
    if final_status != launch_status or final_head != launch_head:
        failures.append("worktree or source commit drifted during verification")
    output_path = reserve_artifact(args.output, label="verification output")
    verification = {
        "schema_version": VERIFICATION_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_git_head": launch_head,
        "verification_launch_git_status_porcelain": launch_status,
        "verifier_sha256": file_sha256(Path(__file__).resolve()),
        "result_path": str(result_path),
        "result_sha256": file_sha256(result_path),
        "deep_replay": deep_summary,
        "failures": sorted(set(failures)),
        "passed": not failures,
        "elapsed_seconds": time.perf_counter() - started,
    }
    verification["content_sha256"] = json_sha256(verification)
    atomic_json(output_path, verification)
    if progress_path is not None:
        atomic_json(progress_path, {
            "schema_version": PROGRESS_SCHEMA,
            "stage": "complete",
            "completed": 120,
            "total": 120,
            "output": str(output_path),
            "output_sha256": file_sha256(output_path),
            "elapsed_seconds": time.perf_counter() - started,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        })
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
