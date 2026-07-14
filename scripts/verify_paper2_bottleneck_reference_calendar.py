#!/usr/bin/env python3
"""Independently reverify the frozen M/T/R reference-calendar screen."""
from __future__ import annotations

import argparse
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_paper2_bottleneck_full_frontier import (
    PRIMARY_CONTRACT_PATH,
    _contract_seed_rows,
)
from scripts.search_paper2_bottleneck_reference_calendar import (
    CONTRACT_PATH,
    DEPENDENCIES,
    ROOT,
    _atomic_json,
    _bootstrap,
    _calendar_name,
    _evaluate_calibration_tape,
    _evaluate_locked_tape,
    _file_sha256,
    _json_sha256,
    _parallel_map,
    _reserve_artifact,
    candidate_calendars,
)
from supply_chain.paper2_bottleneck import materialize_tape


VERIFICATION_SCHEMA = "paper2_bottleneck_reference_calendar_verification_v1"


def _same_float(left: Any, right: Any) -> bool:
    try:
        return float(left).hex() == float(right).hex()
    except (TypeError, ValueError):
        return False


def _git_blob_sha256(commit: str, relative: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return sha256(result.stdout).hexdigest() if result.returncode == 0 else None


def validate_payload(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    content = dict(payload)
    recorded_content_sha = content.pop("content_sha256", None)
    if recorded_content_sha != _json_sha256(content):
        failures.append("content_sha256 mismatch")
    if payload.get("schema_version") != "paper2_bottleneck_reference_calendar_screen_v1":
        failures.append("result schema mismatch")
    if payload.get("contract_id") != "paper2_bottleneck_reference_calendar_screen_v1":
        failures.append("contract id mismatch")
    if payload.get("contract_sha256") != _file_sha256(CONTRACT_PATH):
        failures.append("contract hash mismatch")
    if payload.get("canonical_metric") != "ret_excel_visible_v1":
        failures.append("canonical metric mismatch")
    if payload.get("launch_git_status_porcelain") != []:
        failures.append("result was not launched from a clean worktree")

    dependencies = payload.get("dependency_sha256")
    expected_dependencies = {
        str(path.relative_to(ROOT)): _file_sha256(path) for path in DEPENDENCIES
    }
    if dependencies != expected_dependencies:
        failures.append("current dependency hashes mismatch")
    commit = str(payload.get("git_head", ""))
    if not commit or any(
        _git_blob_sha256(commit, relative) != digest
        for relative, digest in expected_dependencies.items()
    ):
        failures.append("producing commit does not contain dependency bytes")

    candidates = candidate_calendars()
    family = payload.get("candidate_family", {})
    candidate_rows = family.get("rows")
    if family.get("count") != 507 or not isinstance(candidate_rows, list) or len(candidate_rows) != 507:
        failures.append("candidate family is incomplete")
        candidate_rows = []
    if family.get("rows_sha256") != _json_sha256(candidate_rows):
        failures.append("candidate rows hash mismatch")

    calibration = payload.get("calibration", {})
    matrix_hex = calibration.get("score_matrix_float_hex")
    matrix: np.ndarray | None = None
    if (
        not isinstance(matrix_hex, list)
        or len(matrix_hex) != 60
        or any(not isinstance(row, list) or len(row) != 507 for row in matrix_hex)
    ):
        failures.append("calibration score matrix shape mismatch")
    else:
        try:
            matrix = np.asarray(
                [[float.fromhex(value) for value in row] for row in matrix_hex],
                dtype=float,
            )
        except (TypeError, ValueError):
            failures.append("calibration score matrix contains invalid float hex")
        if calibration.get("score_matrix_float_hex_sha256") != _json_sha256(matrix_hex):
            failures.append("calibration score matrix hash mismatch")

    primary = json.loads(PRIMARY_CONTRACT_PATH.read_text())
    calibration_specs = _contract_seed_rows(primary, "calibration")
    locked_specs = _contract_seed_rows(primary, "locked_bound")
    tape_rows = calibration.get("tapes")
    if not isinstance(tape_rows, list) or len(tape_rows) != len(calibration_specs):
        failures.append("calibration tape identity rows are incomplete")
        tape_rows = []
    for index, (spec, row) in enumerate(zip(calibration_specs, tape_rows)):
        tape = materialize_tape(
            int(spec["seed"]), str(spec["context_0"]), str(spec["split"]), weeks=24
        )
        if (
            row.get("index") != index
            or row.get("seed") != spec["seed"]
            or row.get("context") != spec["context_0"]
            or row.get("tape_sha256") != tape["threat_sha256"]
            or matrix_hex is not None
            and index < len(matrix_hex)
            and row.get("scores_sha256") != _json_sha256(matrix_hex[index])
        ):
            failures.append(f"calibration tape identity mismatch at index {index}")

    selected_index = calibration.get("selected_index")
    if matrix is not None and isinstance(selected_index, int):
        exact_sums = [
            sum(
                (Fraction.from_float(float(value)) for value in matrix[:, index]),
                Fraction(0),
            )
            for index in range(507)
        ]
        expected_index = min(
            index for index, total in enumerate(exact_sums) if total == max(exact_sums)
        )
        if selected_index != expected_index:
            failures.append("calibration selected index is not the exact winner")
        if calibration.get("selected_calendar") != _calendar_name(candidates[expected_index]):
            failures.append("calibration selected calendar mismatch")
        if not _same_float(
            calibration.get("selected_mean_ret_excel"),
            exact_sums[expected_index] / 60,
        ):
            failures.append("calibration selected mean mismatch")
        if not _same_float(
            calibration.get("constant_M_mean_ret_excel"), exact_sums[0] / 60
        ):
            failures.append("calibration constant-M mean mismatch")
        if len(candidate_rows) == 507:
            for index, row in enumerate(candidate_rows):
                if (
                    row.get("calendar_index") != index
                    or row.get("calendar") != _calendar_name(candidates[index])
                    or row.get("exact_sum_numerator") != str(exact_sums[index].numerator)
                    or row.get("exact_sum_denominator") != str(exact_sums[index].denominator)
                    or not _same_float(row.get("mean_ret_excel"), exact_sums[index] / 60)
                ):
                    failures.append(f"candidate aggregate mismatch at index {index}")
                    break
    else:
        failures.append("selected index or matrix is unavailable")

    locked = payload.get("locked_diagnostic", {})
    locked_rows = locked.get("rows")
    if not isinstance(locked_rows, list) or len(locked_rows) != 119:
        failures.append("locked diagnostic rows are incomplete")
        locked_rows = []
    lower_values: list[float] = []
    for index, (spec, row) in enumerate(zip(locked_specs, locked_rows)):
        tape = materialize_tape(
            int(spec["seed"]), str(spec["context_0"]), str(spec["split"]), weeks=24
        )
        policies = row.get("policies", {})
        try:
            envelope = max(
                float(policies[name]["ret_excel"])
                for name in ("constant_M", "constant_T", "constant_R")
            )
            lower = max(
                0.0,
                envelope - float(policies["selected_reference"]["ret_excel"]),
            )
        except (KeyError, TypeError, ValueError):
            failures.append(f"locked policy rows malformed at index {index}")
            continue
        if (
            row.get("index") != index
            or row.get("seed") != spec["seed"]
            or row.get("context") != spec["context_0"]
            or row.get("tape_sha256") != tape["threat_sha256"]
            or not _same_float(row.get("best_constant_envelope"), envelope)
            or not _same_float(
                row.get("oracle_minus_reference_lower_bound"), lower
            )
            or any(
                float(policy.get("total_token_hours", -1)) != 4032.0
                for policy in policies.values()
            )
        ):
            failures.append(f"locked identity/arithmetic mismatch at index {index}")
        lower_values.append(lower)

    if len(lower_values) == 119:
        inference = _bootstrap(np.asarray(lower_values, dtype=float))
        if locked.get("lower_bound_inference") != inference:
            failures.append("locked bootstrap inference mismatch")
        precluded = inference["ucb95"] >= 0.01
        if locked.get("reference_precluded") is not precluded:
            failures.append("locked preclusion decision mismatch")
        expected_status = (
            "SELECTED_REFERENCE_PRECLUDED_NO_W24_LAUNCH"
            if precluded
            else "SELECTED_REFERENCE_ELIGIBLE_FOR_EXACT_ORACLE_CEILING_ONLY"
        )
        if payload.get("scientific_status") != expected_status:
            failures.append("scientific status mismatch")
    if payload.get("h_pi_computed") is not False or payload.get("h_obs_computed") is not False:
        failures.append("result attempts an H_PI or H_obs claim")
    if payload.get("learner_authorized") is not False or payload.get("paper3_authorized") is not False:
        failures.append("result attempts learner or Paper 3 authorization")
    return sorted(set(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--deep", action="store_true")
    args = parser.parse_args()
    verification_launch_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        text=True, stdout=subprocess.PIPE,
    ).stdout.splitlines()
    if verification_launch_status:
        parser.error("verification requires a clean immutable worktree")
    verification_git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        text=True, stdout=subprocess.PIPE,
    ).stdout.strip()
    result_path = args.result.resolve()
    payload = json.loads(result_path.read_text())
    failures = validate_payload(payload)
    deep_summary: dict[str, Any] = {"performed": False}
    progress: Path | None = None
    started = time.perf_counter()
    if args.deep and not failures:
        if args.progress is None:
            parser.error("--deep requires --progress")
        progress = _reserve_artifact(args.progress, label="verification progress")
        candidates = candidate_calendars()
        selected_index = int(payload["calibration"]["selected_index"])
        primary = json.loads(PRIMARY_CONTRACT_PATH.read_text())
        calibration_specs = _contract_seed_rows(primary, "calibration")
        locked_specs = _contract_seed_rows(primary, "locked_bound")
        calibration_jobs = [
            (
                index,
                int(row["seed"]),
                str(row["context_0"]),
                candidates,
            )
            for index, row in enumerate(calibration_specs)
        ]
        replayed_calibration = _parallel_map(
            calibration_jobs,
            _evaluate_calibration_tape,
            workers=args.workers,
            progress_path=progress,
            stage="deep_replay_calibration",
            started=started,
        )
        expected_matrix = payload["calibration"]["score_matrix_float_hex"]
        for index, row in enumerate(replayed_calibration):
            if [float(value).hex() for value in row["scores"]] != expected_matrix[index]:
                failures.append(f"deep calibration replay mismatch at index {index}")
                break
        selected = candidates[selected_index]
        locked_jobs = [
            (
                index,
                int(row["seed"]),
                str(row["context_0"]),
                selected,
            )
            for index, row in enumerate(locked_specs)
        ]
        replayed_locked = _parallel_map(
            locked_jobs,
            _evaluate_locked_tape,
            workers=args.workers,
            progress_path=progress,
            stage="deep_replay_locked",
            started=started,
        )
        if replayed_locked != payload["locked_diagnostic"]["rows"]:
            failures.append("deep locked replay mismatch")
        deep_summary = {
            "performed": True,
            "calibration_tapes_replayed": len(replayed_calibration),
            "calibration_rollouts_replayed": len(replayed_calibration) * 507,
            "locked_tapes_replayed": len(replayed_locked),
            "locked_rollouts_replayed": len(replayed_locked) * 4,
        }

    verification_final_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        text=True, stdout=subprocess.PIPE,
    ).stdout.splitlines()
    verification_final_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        text=True, stdout=subprocess.PIPE,
    ).stdout.strip()
    if (
        verification_final_status != verification_launch_status
        or verification_final_head != verification_git_head
    ):
        failures.append("worktree or source commit drifted during verification")
    verification_path = _reserve_artifact(args.output, label="verification output")
    verification = {
        "schema_version": VERIFICATION_SCHEMA,
        "verifier_sha256": _file_sha256(Path(__file__).resolve()),
        "verification_git_head": verification_git_head,
        "verification_launch_git_status_porcelain": verification_launch_status,
        "result_path": str(result_path),
        "result_sha256": _file_sha256(result_path),
        "result_content_sha256": payload.get("content_sha256"),
        "deep_replay": deep_summary,
        "failures": sorted(set(failures)),
        "passed": not failures,
        "elapsed_seconds": time.perf_counter() - started,
    }
    verification["verification_content_sha256"] = _json_sha256(verification)
    _atomic_json(verification_path, verification)
    if progress is not None:
        _atomic_json(progress, {
            "schema_version": "paper2_reference_screen_progress_v1",
            "stage": "complete",
            "completed": deep_summary.get("calibration_tapes_replayed", 0)
            + deep_summary.get("locked_tapes_replayed", 0),
            "total": 179,
            "output": str(verification_path),
            "output_sha256": _file_sha256(verification_path),
            "elapsed_seconds": time.perf_counter() - started,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        })
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
