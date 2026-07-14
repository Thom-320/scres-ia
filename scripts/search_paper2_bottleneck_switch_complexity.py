#!/usr/bin/env python3
"""Exact calibration-only screen of W24 M/T/R calendars with <=3 switches."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_paper2_bottleneck_exact_transducer import certification_environment
from scripts.run_paper2_bottleneck_full_frontier import (
    PRIMARY_CONTRACT_PATH,
    REPLAY_KEYS,
    _contract_seed_rows,
    active_calendar_policy,
    calendar_index,
)
from supply_chain.paper2_bottleneck import (
    ACTIONS,
    ACTION_NAMES,
    materialize_tape,
    run_policy,
)


ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = (
    ROOT / "contracts" / "paper2_bottleneck_switch_complexity_screen_v1.json"
)
OUTPUT_SCHEMA = "paper2_bottleneck_switch_complexity_screen_v1"
PROGRESS_SCHEMA = "paper2_bottleneck_switch_complexity_progress_v1"
RESULT_ROOT = (
    ROOT / "results" / "paper2_bound_harness" / "switch_complexity_screen"
)
DEPENDENCIES = (
    CONTRACT_PATH,
    PRIMARY_CONTRACT_PATH,
    Path(__file__).resolve(),
    ROOT / "scripts" / "run_paper2_bottleneck_full_frontier.py",
    ROOT / "supply_chain" / "paper2_bottleneck.py",
    ROOT / "supply_chain" / "program_f.py",
    ROOT / "supply_chain" / "supply_chain.py",
    ROOT / "supply_chain" / "episode_metrics.py",
    ROOT / "supply_chain" / "ret_thesis.py",
)
EXPECTED_COUNTS = {0: 1, 1: 46, 2: 924, 3: 10_640}


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def json_sha256(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def reserve_artifact(path: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=False)
    allowed = RESULT_ROOT.resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"{label} must be under {allowed}") from exc
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(resolved, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    return resolved


def switch_count(sequence: Sequence[int]) -> int:
    return sum(
        int(sequence[index]) != int(sequence[index - 1])
        for index in range(1, len(sequence))
    )


def candidate_calendars(
    *, weeks: int = 24, max_switches: int = 3
) -> list[tuple[int, ...]]:
    """Enumerate the complete pruned family without walking the 11m frontier."""
    if weeks != 24 or max_switches != 3:
        raise ValueError("this frozen gate is defined only for W24 and <=3 switches")
    rows: list[tuple[int, ...]] = []

    def visit(prefix: tuple[int, ...], switched_previous: bool, switches: int) -> None:
        if len(prefix) == weeks:
            rows.append(prefix)
            return
        last = prefix[-1]
        choices = (last,) if switched_previous else (0, 1, 2)
        for action in choices:
            changed = int(action) != int(last)
            next_switches = switches + int(changed)
            if next_switches <= max_switches:
                visit(prefix + (int(action),), changed, next_switches)

    visit((0,), False, 0)
    counts = {number: 0 for number in EXPECTED_COUNTS}
    for row in rows:
        counts[switch_count(row)] += 1
    if counts != EXPECTED_COUNTS or len(rows) != 11_611 or len(set(rows)) != 11_611:
        raise AssertionError("frozen <=3-switch family is incomplete")
    return rows


def calendar_name(sequence: Sequence[int]) -> str:
    return "".join(ACTION_NAMES[ACTIONS[int(action)]] for action in sequence)


def validate_rollout(
    row: dict[str, Any],
    *,
    tape_sha256: str,
    requested_sequence: Sequence[int],
) -> tuple[str, str]:
    if row.get("ret_excel_contract_version") != "ret_excel_visible_v1":
        raise AssertionError("canonical metric contract drifted")
    if not np.isfinite(float(row["ret_excel"])):
        raise AssertionError("canonical ReT is non-finite")
    if str(row.get("threat_sha256")) != str(tape_sha256):
        raise AssertionError("rollout tape hash differs from materialized tape")
    active = tuple(
        ACTIONS.index(tuple(int(value) for value in event["action"]))
        for event in row["action_events"]
    )
    if active != tuple(map(int, requested_sequence)):
        raise AssertionError("executed active calendar differs from requested calendar")
    token_sum = sum(
        float(row[key]) for key in ("token_hours_m", "token_hours_t", "token_hours_r")
    )
    invariant = (
        float(row["total_token_hours"]),
        token_sum,
        float(row["reserve_inventory_initial"]),
        float(row["reserve_capacity"]),
        float(row["reserve_target_terminal"]),
        float(row["reserve_replenishment_lead_time"]),
        float(row["reserve_issue_delay"]),
        float(row["mass_residual"]),
        float(row["reserve_stock_balance_residual"]),
    )
    if invariant != (4032.0, 4032.0, 10_000.0, 10_000.0, 10_000.0, 168.0, 24.0, 0.0, 0.0):
        raise AssertionError("team, reserve-envelope or mass invariant failed")
    hashes = (
        str(row.get("consumed_base_threat_sha256")),
        str(row.get("realized_demand_sha256")),
    )
    if any(len(value) != 64 for value in hashes):
        raise AssertionError("CRN hash is missing")
    return hashes


def evaluate_calibration_tape(
    index: int,
    seed: int,
    context: str,
    candidates: Sequence[tuple[int, ...]],
) -> dict[str, Any]:
    tape = materialize_tape(seed, context, "calibration", weeks=24)
    scores: list[float] = []
    reference_hashes: tuple[str, str] | None = None
    for sequence in candidates:
        row = run_policy(tape, active_calendar_policy(sequence))
        hashes = validate_rollout(
            row,
            tape_sha256=tape["threat_sha256"],
            requested_sequence=sequence,
        )
        if reference_hashes is None:
            reference_hashes = hashes
        elif hashes != reference_hashes:
            raise AssertionError("calibration CRN hashes changed across calendars")
        scores.append(float(row["ret_excel"]))
    score_hex = [value.hex() for value in scores]
    return {
        "index": index,
        "seed": seed,
        "context": context,
        "split": "calibration",
        "tape_sha256": tape["threat_sha256"],
        "exogenous_hashes": list(reference_hashes or ()),
        "scores": scores,
        "scores_float_hex_sha256": json_sha256(score_hex),
    }


def selected_guardrail_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in REPLAY_KEYS}


def evaluate_selected_tape(
    index: int,
    seed: int,
    context: str,
    sequence: tuple[int, ...],
) -> dict[str, Any]:
    tape = materialize_tape(seed, context, "calibration", weeks=24)
    row = run_policy(tape, active_calendar_policy(sequence))
    hashes = validate_rollout(
        row,
        tape_sha256=tape["threat_sha256"],
        requested_sequence=sequence,
    )
    guardrails = selected_guardrail_row(row)
    return {
        "index": index,
        "seed": seed,
        "context": context,
        "split": "calibration",
        "tape_sha256": tape["threat_sha256"],
        "exogenous_hashes": list(hashes),
        "calendar": calendar_name(sequence),
        "full_frontier_calendar_index": calendar_index(sequence),
        "ret_excel_float_hex": float(row["ret_excel"]).hex(),
        "guardrails_and_resources": guardrails,
        "guardrails_and_resources_sha256": json_sha256(guardrails),
    }


def parallel_map(
    jobs: Sequence[tuple[Any, ...]],
    function: Any,
    *,
    workers: int,
    progress_path: Path,
    stage: str,
    started: float,
) -> list[dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {pool.submit(function, *job): int(job[0]) for job in jobs}
        for future in as_completed(futures):
            index = futures[future]
            completed[index] = future.result()
            atomic_json(
                progress_path,
                {
                    "schema_version": PROGRESS_SCHEMA,
                    "stage": stage,
                    "completed": len(completed),
                    "total": len(jobs),
                    "completed_indices": sorted(completed),
                    "elapsed_seconds": time.perf_counter() - started,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
    return [completed[index] for index in range(len(jobs))]


def exact_selection(matrix: np.ndarray) -> tuple[list[Fraction], int]:
    if matrix.shape != (60, 11_611) or not np.all(np.isfinite(matrix)):
        raise ValueError("calibration score matrix must be finite and 60x11611")
    sums = [
        sum(
            (Fraction.from_float(float(value)) for value in matrix[:, index]),
            Fraction(0),
        )
        for index in range(matrix.shape[1])
    ]
    maximum = max(sums)
    selected = min(index for index, value in enumerate(sums) if value == maximum)
    return sums, selected


def _git_status() -> list[str]:
    return subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(6, os.cpu_count() or 1))
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.output.resolve(strict=False) == args.progress.resolve(strict=False):
        parser.error("--output and --progress must be distinct")
    launch_status = _git_status()
    if launch_status:
        parser.error("scientific screen requires a clean immutable worktree")
    try:
        output_path = reserve_artifact(args.output, label="output")
        progress_path = reserve_artifact(args.progress, label="progress")
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    git_head = _git_head()
    dependency_sha256 = {
        str(path.relative_to(ROOT)): file_sha256(path) for path in DEPENDENCIES
    }
    contract = json.loads(CONTRACT_PATH.read_text())
    if not (
        contract.get("contract_id")
        == "paper2_bottleneck_switch_complexity_screen_v1"
        and contract.get("parent_contract") == "paper2_bottleneck_primary_bound_v2"
        and contract.get("physics", {}).get("weeks") == 24
        and contract.get("candidate_family", {}).get("candidate_count") == 11_611
        and contract.get("candidate_family", {}).get("counts_by_switches")
        == {str(key): value for key, value in EXPECTED_COUNTS.items()}
        and contract.get("calibration", {}).get("seed_start") == 1_100_001
        and contract.get("calibration", {}).get("seed_end") == 1_100_060
        and contract.get("calibration", {}).get("n") == 60
        and contract.get("calibration", {}).get("locked_seed_access_forbidden") is True
        and contract.get("decision_rules", {}).get("h_pi_computed") is False
        and contract.get("decision_rules", {}).get("h_obs_computed") is False
        and contract.get("decision_rules", {}).get("w24_authorized") is False
        and contract.get("decision_rules", {}).get("learner_authorized") is False
        and contract.get("decision_rules", {}).get("paper2_authorized") is False
        and contract.get("decision_rules", {}).get("paper3_authorized") is False
    ):
        raise AssertionError("executable constants differ from frozen contract")

    candidates = candidate_calendars()
    primary = json.loads(PRIMARY_CONTRACT_PATH.read_text())
    calibration_specs = _contract_seed_rows(primary, "calibration")
    if len(calibration_specs) != 60 or any(row["split"] != "calibration" for row in calibration_specs):
        raise AssertionError("calibration seed block drifted")
    started = time.perf_counter()
    jobs = [
        (index, int(row["seed"]), str(row["context_0"]), candidates)
        for index, row in enumerate(calibration_specs)
    ]
    evaluated = parallel_map(
        jobs,
        evaluate_calibration_tape,
        workers=args.workers,
        progress_path=progress_path,
        stage="calibration_all_candidates",
        started=started,
    )
    matrix = np.asarray([row["scores"] for row in evaluated], dtype=float)
    exact_sums, selected_index = exact_selection(matrix)
    selected = candidates[selected_index]
    selected_jobs = [
        (index, int(row["seed"]), str(row["context_0"]), selected)
        for index, row in enumerate(calibration_specs)
    ]
    selected_ledgers = parallel_map(
        selected_jobs,
        evaluate_selected_tape,
        workers=args.workers,
        progress_path=progress_path,
        stage="selected_calendar_guardrail_replay",
        started=started,
    )
    selected_hex = [float(value).hex() for value in matrix[:, selected_index]]
    if [row["ret_excel_float_hex"] for row in selected_ledgers] != selected_hex:
        raise AssertionError("selected canonical replay differs from score matrix")

    means = [float(value / 60) for value in exact_sums]
    candidate_rows = [
        {
            "candidate_index": index,
            "full_frontier_calendar_index": calendar_index(sequence),
            "calendar": calendar_name(sequence),
            "switch_count": switch_count(sequence),
            "exact_sum_numerator": str(exact_sums[index].numerator),
            "exact_sum_denominator": str(exact_sums[index].denominator),
            "mean_ret_excel": means[index],
        }
        for index, sequence in enumerate(candidates)
    ]
    matrix_hex = [[float(value).hex() for value in row] for row in matrix]
    selected_switches = switch_count(selected)
    status = (
        "CALIBRATION_SWITCH_COMPLEXITY_BOUNDARY_ACTIVE"
        if selected_switches == 3
        else "CALIBRATION_SWITCH_COMPLEXITY_INTERIOR"
    )
    payload: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_id": contract["contract_id"],
        "contract_sha256": file_sha256(CONTRACT_PATH),
        "git_head": git_head,
        "launch_git_status_porcelain": launch_status,
        "command": list(sys.argv),
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "certification_environment": certification_environment(),
        },
        "dependency_sha256": dependency_sha256,
        "canonical_metric": "ret_excel_visible_v1",
        "candidate_family": {
            "count": len(candidates),
            "counts_by_switches": {
                str(key): value for key, value in EXPECTED_COUNTS.items()
            },
            "rows": candidate_rows,
            "rows_sha256": json_sha256(candidate_rows),
        },
        "calibration": {
            "seed_start": 1_100_001,
            "seed_end": 1_100_060,
            "n": 60,
            "split": "calibration",
            "selected_index": selected_index,
            "selected_calendar": calendar_name(selected),
            "selected_full_frontier_calendar_index": calendar_index(selected),
            "selected_switch_count": selected_switches,
            "selected_exact_sum_numerator": str(exact_sums[selected_index].numerator),
            "selected_exact_sum_denominator": str(exact_sums[selected_index].denominator),
            "selected_mean_ret_excel": means[selected_index],
            "primary_tie_count": sum(value == exact_sums[selected_index] for value in exact_sums),
            "score_matrix_float_hex": matrix_hex,
            "score_matrix_float_hex_sha256": json_sha256(matrix_hex),
            "tapes": [
                {key: row[key] for key in (
                    "index", "seed", "context", "split", "tape_sha256",
                    "exogenous_hashes", "scores_float_hex_sha256",
                )}
                for row in evaluated
            ],
            "selected_calendar_guardrail_and_resource_ledgers": selected_ledgers,
            "selected_calendar_guardrail_and_resource_ledgers_sha256": json_sha256(selected_ledgers),
        },
        "locked_tapes_accessed": False,
        "virgin_tapes_accessed": False,
        "scientific_status": status,
        "boundary_hit": selected_switches == 3,
        "h_pi_computed": False,
        "h_obs_computed": False,
        "w24_authorized": False,
        "learner_authorized": False,
        "paper2_authorized": False,
        "paper3_authorized": False,
        "claim_limit": (
            "Calibration comparator-development result only. Boundary means only "
            "that a separately frozen richer comparator gate may be considered; "
            "interior does not prove global optimality or family closure."
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    final_dependencies = {
        str(path.relative_to(ROOT)): file_sha256(path) for path in DEPENDENCIES
    }
    if (
        _git_status() != launch_status
        or _git_head() != git_head
        or final_dependencies != dependency_sha256
    ):
        raise RuntimeError("worktree, source commit or dependency bytes drifted")
    payload["content_sha256"] = json_sha256(payload)
    atomic_json(output_path, payload)
    atomic_json(
        progress_path,
        {
            "schema_version": PROGRESS_SCHEMA,
            "stage": "complete",
            "completed": 120,
            "total": 120,
            "output": str(output_path),
            "output_sha256": file_sha256(output_path),
            "elapsed_seconds": time.perf_counter() - started,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(json.dumps({
        "scientific_status": status,
        "selected_calendar": calendar_name(selected),
        "selected_switch_count": selected_switches,
        "output": str(output_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
