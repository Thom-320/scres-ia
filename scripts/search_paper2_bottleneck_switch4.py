#!/usr/bin/env python3
"""Frozen calibration-only W24 M/T/R screen with at most four switches."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from fractions import Fraction
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_paper2_bottleneck_exact_transducer import certification_environment
from scripts.run_paper2_bottleneck_full_frontier import (
    PRIMARY_CONTRACT_PATH,
    _contract_seed_rows,
    calendar_index,
)
from scripts.search_paper2_bottleneck_switch_complexity import (
    calendar_name,
    evaluate_calibration_tape,
    evaluate_selected_tape,
    file_sha256,
    json_sha256,
    switch_count,
)


ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = (
    ROOT / "contracts" / "paper2_bottleneck_switch_complexity_screen_v2.json"
)
OUTPUT_SCHEMA = "paper2_bottleneck_switch4_screen_v1"
PREFLIGHT_SCHEMA = "paper2_bottleneck_switch4_preflight_v1"
PROGRESS_SCHEMA = "paper2_bottleneck_switch4_progress_v1"
RESULT_ROOT = (
    ROOT / "results" / "paper2_bound_harness" / "switch_complexity_screen_v2"
)
DEPENDENCIES = (
    CONTRACT_PATH,
    PRIMARY_CONTRACT_PATH,
    Path(__file__).resolve(),
    ROOT / "scripts" / "search_paper2_bottleneck_switch_complexity.py",
    ROOT / "scripts" / "run_paper2_bottleneck_full_frontier.py",
    ROOT / "supply_chain" / "paper2_bottleneck.py",
    ROOT / "supply_chain" / "program_f.py",
    ROOT / "supply_chain" / "supply_chain.py",
    ROOT / "supply_chain" / "episode_metrics.py",
    ROOT / "supply_chain" / "ret_thesis.py",
)
EXPECTED_COUNTS = {0: 1, 1: 46, 2: 924, 3: 10_640, 4: 77_520}
CANDIDATE_COUNT = 89_131
CALIBRATION_TAPES = 60
_WORKER_CANDIDATES: Sequence[tuple[int, ...]] | None = None


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


def candidate_calendars() -> list[tuple[int, ...]]:
    """Enumerate the complete frozen W24, <=4-switch feasible family."""
    rows: list[tuple[int, ...]] = []

    def visit(
        prefix: tuple[int, ...], switched_previous: bool, switches: int
    ) -> None:
        if len(prefix) == 24:
            rows.append(prefix)
            return
        last = prefix[-1]
        choices = (last,) if switched_previous else (0, 1, 2)
        for action in choices:
            changed = int(action) != int(last)
            next_switches = switches + int(changed)
            if next_switches <= 4:
                visit(prefix + (int(action),), changed, next_switches)

    visit((0,), False, 0)
    counts = {number: 0 for number in EXPECTED_COUNTS}
    for row in rows:
        counts[switch_count(row)] += 1
    if (
        counts != EXPECTED_COUNTS
        or len(rows) != CANDIDATE_COUNT
        or len(set(rows)) != CANDIDATE_COUNT
    ):
        raise AssertionError("frozen <=4-switch family is incomplete")
    return rows


def initialize_candidate_worker(
    candidates: Sequence[tuple[int, ...]],
) -> None:
    global _WORKER_CANDIDATES
    _WORKER_CANDIDATES = candidates


def evaluate_calibration_tape_from_worker(
    index: int, seed: int, context: str
) -> dict[str, Any]:
    if _WORKER_CANDIDATES is None:
        raise RuntimeError("candidate worker was not initialized")
    return evaluate_calibration_tape(
        index, seed, context, _WORKER_CANDIDATES
    )


def parallel_map(
    jobs: Sequence[tuple[Any, ...]],
    function: Any,
    *,
    workers: int,
    progress_path: Path,
    stage: str,
    started: float,
    initializer: Any = None,
    initargs: tuple[Any, ...] = (),
) -> list[dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=initializer,
        initargs=initargs,
    ) as pool:
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


def exact_argmax(matrix: np.ndarray) -> tuple[list[Fraction], int]:
    """Return exact binary64 column sums and the minimum-index maximizer."""
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("score matrix must be finite and two-dimensional")
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


def exact_selection(matrix: np.ndarray) -> tuple[list[Fraction], int]:
    if matrix.shape != (CALIBRATION_TAPES, CANDIDATE_COUNT):
        raise ValueError("calibration score matrix must be 60x89131")
    return exact_argmax(matrix)


def git_status() -> list[str]:
    return subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def validate_contract(contract: dict[str, Any]) -> None:
    rules = contract.get("decision_rules", {})
    if not (
        contract.get("contract_id")
        == "paper2_bottleneck_switch_complexity_screen_v2"
        and contract.get("parent_contract") == "paper2_bottleneck_primary_bound_v2"
        and contract.get("physics", {}).get("weeks") == 24
        and contract.get("candidate_family", {}).get("candidate_count")
        == CANDIDATE_COUNT
        and contract.get("candidate_family", {}).get("counts_by_switches")
        == {str(key): value for key, value in EXPECTED_COUNTS.items()}
        and contract.get("calibration", {}).get("seed_start") == 1_100_001
        and contract.get("calibration", {}).get("seed_end") == 1_100_060
        and contract.get("calibration", {}).get("n") == CALIBRATION_TAPES
        and contract.get("calibration", {}).get("locked_seed_access_forbidden")
        is True
        and contract.get("vps_preflight", {}).get("required_before_producer")
        is True
        and contract.get("vps_preflight", {}).get("workers") == 6
        and contract.get("vps_preflight", {}).get("tapes") == 6
        and contract.get("vps_preflight", {}).get("preflight_rollouts")
        == 534_786
        and contract.get("execution_discipline", {}).get("expected_hostname")
        == "vps-f733423b"
        and all(
            rules.get(field) is False
            for field in (
                "h_pi_computed",
                "h_obs_computed",
                "w24_authorized",
                "learner_authorized",
                "paper2_authorized",
                "paper3_authorized",
            )
        )
    ):
        raise AssertionError("executable constants differ from frozen v2 contract")


def environment_payload() -> dict[str, Any]:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "numpy": np.__version__,
        "certification_environment": certification_environment(),
    }


def finalize(
    *,
    output_path: Path,
    progress_path: Path,
    payload: dict[str, Any],
    launch_status: list[str],
    launch_head: str,
    dependency_sha256: dict[str, str],
    completed: int,
    total: int,
    started: float,
) -> None:
    final_dependencies = {
        str(path.relative_to(ROOT)): file_sha256(path) for path in DEPENDENCIES
    }
    if (
        git_status() != launch_status
        or git_head() != launch_head
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
            "completed": completed,
            "total": total,
            "output": str(output_path),
            "output_sha256": file_sha256(output_path),
            "elapsed_seconds": time.perf_counter() - started,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(6, os.cpu_count() or 1))
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.preflight_only and args.workers != 6:
        parser.error("the frozen preflight requires the six-worker production pool")
    if args.output.resolve(strict=False) == args.progress.resolve(strict=False):
        parser.error("--output and --progress must be distinct")
    launch_status = git_status()
    if launch_status:
        parser.error("scientific screen requires a clean immutable worktree")
    try:
        output_path = reserve_artifact(args.output, label="output")
        progress_path = reserve_artifact(args.progress, label="progress")
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))

    launch_head = git_head()
    dependency_sha256 = {
        str(path.relative_to(ROOT)): file_sha256(path) for path in DEPENDENCIES
    }
    contract = json.loads(CONTRACT_PATH.read_text())
    validate_contract(contract)
    expected_hostname = contract["execution_discipline"]["expected_hostname"]
    if socket.gethostname() != expected_hostname:
        raise RuntimeError(
            f"frozen switch4 search requires hostname {expected_hostname}"
        )
    candidates = candidate_calendars()
    candidate_sequence_sha256 = json_sha256(candidates)
    primary = json.loads(PRIMARY_CONTRACT_PATH.read_text())
    calibration_specs = _contract_seed_rows(primary, "calibration")
    if len(calibration_specs) != CALIBRATION_TAPES or any(
        row["split"] != "calibration" for row in calibration_specs
    ):
        raise AssertionError("calibration seed block drifted")
    started = time.perf_counter()

    common = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_id": contract["contract_id"],
        "contract_sha256": file_sha256(CONTRACT_PATH),
        "git_head": launch_head,
        "launch_git_status_porcelain": launch_status,
        "command": [sys.executable, *sys.argv],
        "environment": environment_payload(),
        "dependency_sha256": dependency_sha256,
        "canonical_metric": "ret_excel_visible_v1",
        "candidate_count": CANDIDATE_COUNT,
        "candidate_sequence_sha256": candidate_sequence_sha256,
        "locked_tapes_accessed": False,
        "virgin_tapes_accessed": False,
        "h_pi_computed": False,
        "h_obs_computed": False,
        "w24_authorized": False,
        "learner_authorized": False,
        "paper2_authorized": False,
        "paper3_authorized": False,
    }

    if args.preflight_only:
        preflight_specs = calibration_specs[:6]
        evaluated = parallel_map(
            [
                (index, int(row["seed"]), str(row["context_0"]))
                for index, row in enumerate(preflight_specs)
            ],
            evaluate_calibration_tape_from_worker,
            workers=6,
            progress_path=progress_path,
            stage="vps_six_tape_all_candidates_memory_preflight",
            started=started,
            initializer=initialize_candidate_worker,
            initargs=(candidates,),
        )
        score_hex = [
            [float(value).hex() for value in row["scores"]]
            for row in evaluated
        ]
        payload = {
            **common,
            "schema_version": PREFLIGHT_SCHEMA,
            "preflight_only": True,
            "seed_start": int(preflight_specs[0]["seed"]),
            "seed_end": int(preflight_specs[-1]["seed"]),
            "n_tapes": 6,
            "split": "calibration",
            "tapes": [
                {key: row[key] for key in (
                    "index",
                    "seed",
                    "context",
                    "split",
                    "tape_sha256",
                    "exogenous_hashes",
                    "scores_float_hex_sha256",
                )}
                for row in evaluated
            ],
            "scores_evaluated": sum(len(row) for row in score_hex),
            "score_matrix_float_hex_sha256": json_sha256(score_hex),
            "elapsed_seconds": time.perf_counter() - started,
            "claim_limit": "Execution and memory preflight only; no policy selection or scientific estimand is computed.",
        }
        finalize(
            output_path=output_path,
            progress_path=progress_path,
            payload=payload,
            launch_status=launch_status,
            launch_head=launch_head,
            dependency_sha256=dependency_sha256,
            completed=6,
            total=6,
            started=started,
        )
        print(json.dumps({
            "status": "SWITCH4_SIX_TAPE_MEMORY_PREFLIGHT_COMPLETE",
            "scores_evaluated": sum(len(row) for row in score_hex),
            "output": str(output_path),
        }, indent=2, sort_keys=True))
        return 0

    jobs = [
        (index, int(row["seed"]), str(row["context_0"]))
        for index, row in enumerate(calibration_specs)
    ]
    evaluated = parallel_map(
        jobs,
        evaluate_calibration_tape_from_worker,
        workers=args.workers,
        progress_path=progress_path,
        stage="calibration_all_switch4_candidates",
        started=started,
        initializer=initialize_candidate_worker,
        initargs=(candidates,),
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
        stage="selected_switch4_calendar_guardrail_replay",
        started=started,
    )
    selected_hex = [float(value).hex() for value in matrix[:, selected_index]]
    if [row["ret_excel_float_hex"] for row in selected_ledgers] != selected_hex:
        raise AssertionError("selected canonical replay differs from score matrix")

    means = [float(value / CALIBRATION_TAPES) for value in exact_sums]
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
        "CALIBRATION_SWITCH4_BOUNDARY_ACTIVE"
        if selected_switches == 4
        else "CALIBRATION_SWITCH4_INTERIOR"
    )
    payload = {
        **common,
        "schema_version": OUTPUT_SCHEMA,
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
            "n": CALIBRATION_TAPES,
            "split": "calibration",
            "selected_index": selected_index,
            "selected_calendar": calendar_name(selected),
            "selected_full_frontier_calendar_index": calendar_index(selected),
            "selected_switch_count": selected_switches,
            "selected_exact_sum_numerator": str(exact_sums[selected_index].numerator),
            "selected_exact_sum_denominator": str(exact_sums[selected_index].denominator),
            "selected_mean_ret_excel": means[selected_index],
            "primary_tie_count": sum(
                value == exact_sums[selected_index] for value in exact_sums
            ),
            "score_matrix_float_hex": matrix_hex,
            "score_matrix_float_hex_sha256": json_sha256(matrix_hex),
            "tapes": [
                {key: row[key] for key in (
                    "index",
                    "seed",
                    "context",
                    "split",
                    "tape_sha256",
                    "exogenous_hashes",
                    "scores_float_hex_sha256",
                )}
                for row in evaluated
            ],
            "selected_calendar_guardrail_and_resource_ledgers": selected_ledgers,
            "selected_calendar_guardrail_and_resource_ledgers_sha256": json_sha256(
                selected_ledgers
            ),
        },
        "scientific_status": status,
        "boundary_hit": selected_switches == 4,
        "claim_limit": (
            "Calibration comparator-development result only. Boundary does not "
            "automatically authorize a richer gate; interior does not prove global "
            "optimality or family closure."
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    finalize(
        output_path=output_path,
        progress_path=progress_path,
        payload=payload,
        launch_status=launch_status,
        launch_head=launch_head,
        dependency_sha256=dependency_sha256,
        completed=120,
        total=120,
        started=started,
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
