#!/usr/bin/env python3
"""Frozen cheap screen for a strong M/T/R open-loop reference calendar."""
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

from scripts.run_paper2_bottleneck_full_frontier import (
    PRIMARY_CONTRACT_PATH,
    _contract_seed_rows,
    active_calendar_policy,
)
from scripts.run_paper2_bottleneck_exact_transducer import (
    certification_environment,
)
from supply_chain.paper2_bottleneck import (
    ACTIONS,
    ACTION_NAMES,
    CONTEXTS,
    materialize_tape,
    run_policy,
)


ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = (
    ROOT / "contracts" / "paper2_bottleneck_reference_calendar_screen_v1.json"
)
OUTPUT_SCHEMA = "paper2_bottleneck_reference_calendar_screen_v1"
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


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _reserve_artifact(path: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=False)
    allowed = (
        ROOT / "results" / "paper2_bound_harness" / "reference_calendar_screen"
    ).resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"{label} must be under {allowed}") from exc
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(resolved, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    return resolved


def candidate_calendars() -> list[tuple[int, ...]]:
    candidates = [(0,) * 24]
    for action in (1, 2):
        for start in range(1, 23):
            for end in range(start + 1, 24):
                sequence = [0] * 24
                sequence[start : end + 1] = [action] * (end - start + 1)
                candidates.append(tuple(sequence))
    if len(candidates) != 507 or len(set(candidates)) != 507:
        raise AssertionError("frozen single-excursion family is not 507 calendars")
    return candidates


def _calendar_name(sequence: Sequence[int]) -> str:
    return "".join(ACTION_NAMES[ACTIONS[int(action)]] for action in sequence)


def _validate_rollout(
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
    invariant = (
        float(row["total_token_hours"]),
        float(row["reserve_inventory_initial"]),
        float(row["reserve_capacity"]),
        float(row["reserve_target_terminal"]),
        float(row["mass_residual"]),
        str(row["consumed_base_threat_sha256"]),
        str(row["realized_demand_sha256"]),
    )
    if invariant[:5] != (4032.0, 10_000.0, 10_000.0, 10_000.0, 0.0):
        raise AssertionError("team, initial-reserve or mass invariant failed")
    return invariant[-2], invariant[-1]


def _evaluate_calibration_tape(
    index: int,
    seed: int,
    context: str,
    candidates: Sequence[tuple[int, ...]],
) -> dict[str, Any]:
    tape = materialize_tape(seed, context, "calibration", weeks=24)
    scores: list[float] = []
    reference_hashes: tuple[str, str] | None = None
    for sequence in candidates:
        row = run_policy(
            tape,
            active_calendar_policy(sequence),
            ret_excel_contract_version="ret_excel_visible_v1",
        )
        hashes = _validate_rollout(
            row,
            tape_sha256=tape["threat_sha256"],
            requested_sequence=sequence,
        )
        if reference_hashes is None:
            reference_hashes = hashes
        elif hashes != reference_hashes:
            raise AssertionError("calibration CRN hashes changed across calendars")
        scores.append(float(row["ret_excel"]))
    return {
        "index": index,
        "seed": seed,
        "context": context,
        "tape_sha256": tape["threat_sha256"],
        "scores": scores,
        "scores_sha256": _json_sha256([float(value).hex() for value in scores]),
    }


def _evaluate_locked_tape(
    index: int,
    seed: int,
    context: str,
    selected: tuple[int, ...],
) -> dict[str, Any]:
    tape = materialize_tape(seed, context, "locked", weeks=24)
    policies = {
        "selected_reference": selected,
        "constant_M": (0,) * 24,
        "constant_T": (0,) + (1,) * 23,
        "constant_R": (0,) + (2,) * 23,
    }
    rows: dict[str, dict[str, Any]] = {}
    reference_hashes: tuple[str, str] | None = None
    for name, sequence in policies.items():
        row = run_policy(
            tape,
            active_calendar_policy(sequence),
            ret_excel_contract_version="ret_excel_visible_v1",
        )
        hashes = _validate_rollout(
            row,
            tape_sha256=tape["threat_sha256"],
            requested_sequence=sequence,
        )
        if reference_hashes is None:
            reference_hashes = hashes
        elif hashes != reference_hashes:
            raise AssertionError("locked CRN hashes changed across policies")
        rows[name] = {
            "ret_excel": float(row["ret_excel"]),
            "lost_orders": float(row["lost_orders"]),
            "service_loss_auc_ration_hours": float(
                row["service_loss_auc_ration_hours"]
            ),
            "total_token_hours": float(row["total_token_hours"]),
            "reserve_units_issued": float(row["reserve_units_issued"]),
            "reserve_units_replenished": float(
                row["reserve_units_replenished"]
            ),
            "reserve_replenishment_requests": float(
                row["reserve_replenishment_requests"]
            ),
            "reserve_inventory_terminal": float(
                row["reserve_inventory_terminal"]
            ),
            "reserve_committed_pending_terminal": float(
                row["reserve_committed_pending_terminal"]
            ),
        }
    envelope = max(rows[name]["ret_excel"] for name in (
        "constant_M", "constant_T", "constant_R"
    ))
    lower_gap = max(0.0, envelope - rows["selected_reference"]["ret_excel"])
    return {
        "index": index,
        "seed": seed,
        "context": context,
        "tape_sha256": tape["threat_sha256"],
        "exogenous_hashes": list(reference_hashes or ()),
        "policies": rows,
        "best_constant_envelope": envelope,
        "oracle_minus_reference_lower_bound": lower_gap,
    }


def _parallel_map(
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
            _atomic_json(progress_path, {
                "schema_version": "paper2_reference_screen_progress_v1",
                "stage": stage,
                "completed": len(completed),
                "total": len(jobs),
                "completed_indices": sorted(completed),
                "elapsed_seconds": time.perf_counter() - started,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            })
    return [completed[index] for index in range(len(jobs))]


def _bootstrap(values: np.ndarray) -> dict[str, Any]:
    if values.shape != (119,):
        raise ValueError("locked bootstrap requires exactly 119 paired tapes")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("locked lower-bound vector must be finite and non-negative")
    rng = np.random.default_rng(20260713)
    indices = rng.integers(0, len(values), size=(10_000, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975], method="linear")
    return {
        "mean": float(values.mean()),
        "lcb95": float(lower),
        "ucb95": float(upper),
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": 20260713,
        "method": "paired_tape_percentile_with_replacement_numpy_quantile_linear",
        "values_sha256": _json_sha256([float(value).hex() for value in values]),
    }


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
    launch_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        text=True, stdout=subprocess.PIPE,
    ).stdout.splitlines()
    if launch_status:
        parser.error("scientific reference screen requires a clean immutable worktree")
    try:
        output_path = _reserve_artifact(args.output, label="output")
        progress_path = _reserve_artifact(args.progress, label="progress")
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        text=True, stdout=subprocess.PIPE,
    ).stdout.strip()
    dependency_sha256 = {
        str(path.relative_to(ROOT)): _file_sha256(path)
        for path in DEPENDENCIES
    }
    contract = json.loads(CONTRACT_PATH.read_text())
    if not (
        contract.get("contract_id")
        == "paper2_bottleneck_reference_calendar_screen_v1"
        and contract.get("parent_contract")
        == "paper2_bottleneck_primary_bound_v2"
        and contract.get("physics", {}).get("weeks") == 24
        and contract.get("physics", {}).get("actions") == ["M", "T", "R"]
        and contract.get("candidate_family", {}).get("candidate_count") == 507
        and contract.get("calibration", {}).get("seed_start") == 1_100_001
        and contract.get("calibration", {}).get("seed_end") == 1_100_060
        and contract.get("calibration", {}).get("n") == 60
        and contract.get("locked_diagnostic", {}).get("seed_start") == 1_110_002
        and contract.get("locked_diagnostic", {}).get("seed_end") == 1_110_120
        and contract.get("locked_diagnostic", {}).get("n") == 119
        and contract.get("locked_diagnostic", {}).get("bootstrap_resamples")
        == 10_000
        and contract.get("locked_diagnostic", {}).get("bootstrap_seed")
        == 20260713
        and contract.get("locked_diagnostic", {}).get("practical_gate") == 0.01
        and contract.get("resource_diagnostic", {}).get("classification")
        == "RESOURCE_RELAXED_DIAGNOSTIC"
        and contract.get("decision_rules", {}).get("learner_authorized") is False
        and contract.get("decision_rules", {}).get("paper2_authorized") is False
        and contract.get("decision_rules", {}).get("paper3_authorized") is False
    ):
        raise AssertionError("executable constants differ from frozen contract")
    candidates = candidate_calendars()
    started = time.perf_counter()
    primary_contract = json.loads(PRIMARY_CONTRACT_PATH.read_text())
    calibration_seed_rows = _contract_seed_rows(primary_contract, "calibration")
    locked_seed_rows = _contract_seed_rows(primary_contract, "locked_bound")
    calibration_jobs = [
        (
            index,
            int(row["seed"]),
            str(row["context_0"]),
            candidates,
        )
        for index, row in enumerate(calibration_seed_rows)
    ]
    calibration = _parallel_map(
        calibration_jobs,
        _evaluate_calibration_tape,
        workers=args.workers,
        progress_path=progress_path,
        stage="calibration_candidate_screen",
        started=started,
    )
    matrix = np.asarray([row["scores"] for row in calibration], dtype=float)
    if matrix.shape != (60, 507):
        raise AssertionError("calibration score matrix is incomplete")
    exact_sums = [
        sum((Fraction.from_float(float(value)) for value in matrix[:, index]), Fraction(0))
        for index in range(matrix.shape[1])
    ]
    maximum_sum = max(exact_sums)
    selected_index = min(
        index for index, total in enumerate(exact_sums) if total == maximum_sum
    )
    means = np.asarray([float(total / 60) for total in exact_sums], dtype=float)
    selected = candidates[selected_index]
    locked_jobs = [
        (
            index,
            int(row["seed"]),
            str(row["context_0"]),
            selected,
        )
        for index, row in enumerate(locked_seed_rows)
    ]
    locked = _parallel_map(
        locked_jobs,
        _evaluate_locked_tape,
        workers=args.workers,
        progress_path=progress_path,
        stage="locked_preclusion_diagnostic",
        started=started,
    )
    lower_values = np.asarray([
        row["oracle_minus_reference_lower_bound"] for row in locked
    ], dtype=float)
    inference = _bootstrap(lower_values)
    candidate_rows = [
        {
            "calendar_index": index,
            "calendar": _calendar_name(sequence),
            "mean_ret_excel": float(means[index]),
            "exact_sum_numerator": str(exact_sums[index].numerator),
            "exact_sum_denominator": str(exact_sums[index].denominator),
            "delta_vs_constant_M": float(means[index] - means[0]),
        }
        for index, sequence in enumerate(candidates)
    ]
    score_matrix_float_hex = [
        [float(value).hex() for value in row] for row in matrix
    ]
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_id": contract["contract_id"],
        "contract_sha256": _file_sha256(CONTRACT_PATH),
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
            "rows": candidate_rows,
            "rows_sha256": _json_sha256(candidate_rows),
        },
        "calibration": {
            "seed_start": 1_100_001,
            "n": 60,
            "selected_index": selected_index,
            "selected_calendar": _calendar_name(selected),
            "selected_mean_ret_excel": float(means[selected_index]),
            "constant_M_mean_ret_excel": float(means[0]),
            "selected_delta_vs_constant_M": float(
                means[selected_index] - means[0]
            ),
            "score_matrix_float_hex": score_matrix_float_hex,
            "score_matrix_float_hex_sha256": _json_sha256(
                score_matrix_float_hex
            ),
            "tapes": [{key: row[key] for key in (
                "index", "seed", "context", "tape_sha256", "scores_sha256"
            )} for row in calibration],
        },
        "locked_diagnostic": {
            "seed_start": 1_110_002,
            "n": 119,
            "rows": locked,
            "lower_bound_inference": inference,
            "reference_precluded": inference["ucb95"] >= 0.01,
            "interpretation": (
                "RESOURCE_RELAXED_DIAGNOSTIC: the full oracle-minus-reference "
                "UCB95 is no smaller than this common-resample lower-bound "
                "UCB95. This can preclude only the selected reference."
            ),
        },
        "scientific_status": (
            "SELECTED_REFERENCE_PRECLUDED_NO_W24_LAUNCH"
            if inference["ucb95"] >= 0.01
            else "SELECTED_REFERENCE_ELIGIBLE_FOR_EXACT_ORACLE_CEILING_ONLY"
        ),
        "h_pi_computed": False,
        "h_obs_computed": False,
        "learner_authorized": False,
        "paper3_authorized": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    final_git_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        text=True, stdout=subprocess.PIPE,
    ).stdout.splitlines()
    if (
        final_git_status != launch_status
        or subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            text=True, stdout=subprocess.PIPE,
        ).stdout.strip()
        != git_head
        or {
            str(path.relative_to(ROOT)): _file_sha256(path)
            for path in DEPENDENCIES
        }
        != dependency_sha256
    ):
        raise RuntimeError(
            "worktree, source commit or dependency bytes drifted during execution"
        )
    payload["content_sha256"] = _json_sha256(payload)
    _atomic_json(output_path, payload)
    _atomic_json(progress_path, {
        "schema_version": "paper2_reference_screen_progress_v1",
        "stage": "complete",
        "completed": 179,
        "total": 179,
        "output": str(output_path),
        "output_sha256": _file_sha256(output_path),
        "elapsed_seconds": time.perf_counter() - started,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps({
        "scientific_status": payload["scientific_status"],
        "selected_calendar": payload["calibration"]["selected_calendar"],
        "selected_delta_vs_constant_M": payload["calibration"][
            "selected_delta_vs_constant_M"
        ],
        "locked_lower_bound_inference": inference,
        "output": str(output_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
