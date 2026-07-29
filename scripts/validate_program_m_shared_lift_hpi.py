#!/usr/bin/env python3
"""Fresh, frozen Program M H_PI validation producer.

Only the six cells frozen by the screen-selection artifact are evaluated.  A
scientific CLI run opens the burned H_PI-validation block 7,300,025--7,300,048;
tests use explicit synthetic seeds and a fake evaluator.  Completion establishes
at most an H_PI validation result.  It cannot authorize H_obs, a learner,
Paper 2, Paper 3, or virgin tapes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.screen_program_m_shared_lift_hpi import (  # noqa: E402
    METRIC_ID,
    PRACTICAL_GATE,
    Cell,
    adjacent,
    all_calendars,
    atomic_json,
    digest_json,
    evaluate_cell_tape,
    file_sha256,
    frozen_cells,
)


SELECTION_PATH = (
    ROOT / "research/paper2_exhaustive_search/program_m_hpi_screen_selection_20260714.json"
)
VALIDATION_SEEDS = tuple(range(7_300_025, 7_300_049))
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_RNG_SEED = 2_026_071_401
SELECTED_CELL_IDS = (
    "h50_d120_s70",
    "h50_d120_s85",
    "h75_d120_s70",
    "h75_d120_s85",
    "h75_d72_s70",
    "h75_d72_s85",
)
SCREEN_RESULT_SHA256 = "54d26a8c2e8651159d33694ad56c311f7cef8e6483d9caeb1a449a14b14e8101"
SCIENTIFIC_COMMIT = "d2adb8a2bfecd76ba2f40f988bce1184f144cce0"
SOURCE_PATHS = (
    "research/paper2_exhaustive_search/program_m_hpi_screen_selection_20260714.json",
    "scripts/validate_program_m_shared_lift_hpi.py",
    "scripts/screen_program_m_shared_lift_hpi.py",
    "contracts/program_m_shared_lift_reservation_v1.json",
    "supply_chain/program_m_shared_lift.py",
    "supply_chain/episode_metrics.py",
    "supply_chain/ret_thesis.py",
    "supply_chain/supply_chain.py",
    "supply_chain/data/garrido_proxy_v1_freeze_2026-07-10.json",
)


def selection_artifact() -> dict[str, Any]:
    payload = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if payload["scientific_commit"] != SCIENTIFIC_COMMIT:
        raise RuntimeError("selection scientific commit binding drift")
    if payload["screen_result"]["sha256"] != SCREEN_RESULT_SHA256:
        raise RuntimeError("selection screen-result binding drift")
    if payload["screen_result"]["completed_shards"] != 456:
        raise RuntimeError("selection does not bind a complete 456-shard screen")
    if tuple(payload["selected_cell_ids"]) != SELECTED_CELL_IDS:
        raise RuntimeError("frozen selected-cell order or membership drift")
    if payload["h_pi_validation"]["state"] != "SEALED_NOT_OPENED":
        raise RuntimeError("selection artifact no longer marks validation seeds sealed")
    return payload


def selected_cells() -> tuple[Cell, ...]:
    selection_artifact()
    by_id = {cell.cell_id: cell for cell in frozen_cells()}
    cells = tuple(by_id[cell_id] for cell_id in SELECTED_CELL_IDS)
    if len(cells) != 6 or any(cell.is_null for cell in cells):
        raise RuntimeError("invalid frozen validation cell set")
    return cells


def _matrix(shards: Sequence[Mapping[str, Any]]) -> np.ndarray:
    matrix = np.asarray(
        [
            [float(row["ret_request_snapshot_v2"]) for row in shard["evaluations"]]
            for shard in shards
        ],
        dtype=float,
    )
    if matrix.shape != (len(shards), 256) or not np.isfinite(matrix).all():
        raise AssertionError("incomplete or non-finite 256-calendar validation frontier")
    return matrix


def observed_hpi(matrix: np.ndarray) -> dict[str, Any]:
    means = matrix.mean(axis=0)
    best_static = int(np.flatnonzero(means == means.max())[0])
    oracle_indices = np.argmax(matrix, axis=1)
    deltas = matrix[np.arange(matrix.shape[0]), oracle_indices] - matrix[:, best_static]
    return {
        "observed_h_pi": float(deltas.mean()),
        "best_static_calendar_index": best_static,
        "best_static_calendar": list(all_calendars()[best_static]),
        "oracle_calendar_indices": oracle_indices.astype(int).tolist(),
        "unique_oracle_calendars": int(len(set(map(int, oracle_indices)))),
        "h_pi_per_tape": deltas.tolist(),
    }


def bootstrap_simultaneous_lcbs(
    matrices: Mapping[str, np.ndarray],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    rng_seed: int = BOOTSTRAP_RNG_SEED,
) -> dict[str, Any]:
    """Basic max-error one-sided simultaneous 95% LCBs.

    The same resampled tape indices are used across cells (paired CRN).  The
    best tape-independent static calendar is reselected separately in every
    cell and bootstrap resample, with lowest-index exact-tie resolution.
    """

    cell_ids = tuple(sorted(matrices))
    if not cell_ids:
        raise ValueError("at least one cell matrix is required")
    n_tapes = matrices[cell_ids[0]].shape[0]
    if n_tapes < 2 or any(matrix.shape != (n_tapes, 256) for matrix in matrices.values()):
        raise ValueError("all cell matrices must have one paired n-tape by 256 frontier")
    observed = {cell_id: observed_hpi(matrices[cell_id])["observed_h_pi"] for cell_id in cell_ids}
    rng = np.random.default_rng(int(rng_seed))
    maxima = np.empty(int(n_resamples), dtype=float)
    bootstrap_values = {cell_id: np.empty(int(n_resamples), dtype=float) for cell_id in cell_ids}
    for draw in range(int(n_resamples)):
        indices = rng.integers(0, n_tapes, size=n_tapes)
        errors = []
        for cell_id in cell_ids:
            sampled = matrices[cell_id][indices]
            means = sampled.mean(axis=0)
            static_index = int(np.flatnonzero(means == means.max())[0])
            oracle = sampled.max(axis=1)
            theta_star = float((oracle - sampled[:, static_index]).mean())
            bootstrap_values[cell_id][draw] = theta_star
            errors.append(float(observed[cell_id]) - theta_star)
        maxima[draw] = max(errors)
    critical = float(np.quantile(maxima, 0.95, method="higher"))
    return {
        "method": "paired_basic_max_error_one_sided_familywise_95",
        "resamples": int(n_resamples),
        "rng_seed": int(rng_seed),
        "quantile": {"probability": 0.95, "numpy_method": "higher"},
        "static_reselected_in_every_cell_and_resample": True,
        "shared_paired_tape_indices_across_cells": True,
        "critical_max_error": critical,
        "simultaneous_lcb95": {
            cell_id: float(observed[cell_id] - critical) for cell_id in cell_ids
        },
        "bootstrap_distribution_sha256": digest_json(
            {cell_id: values.tolist() for cell_id, values in bootstrap_values.items()}
        ),
    }


def passing_components(cell_rows: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    by_id = {cell.cell_id: cell for cell in selected_cells()}
    eligible = {
        str(row["cell_id"])
        for row in cell_rows
        if float(row["simultaneous_lcb95"]) >= PRACTICAL_GATE
    }
    components: list[list[str]] = []
    while eligible:
        first = min(eligible)
        eligible.remove(first)
        pending = [first]
        component: list[str] = []
        while pending:
            current = pending.pop()
            component.append(current)
            neighbors = {other for other in eligible if adjacent(by_id[current], by_id[other])}
            eligible -= neighbors
            pending.extend(sorted(neighbors))
        cells = [by_id[cell_id] for cell_id in component]
        if (
            len(cells) >= 3
            and len({cell.hazard for cell in cells}) >= 2
            and len({cell.duration_hours for cell in cells}) >= 2
        ):
            components.append(sorted(component))
    return sorted(components)


def select_least_observed(
    cell_rows: Sequence[Mapping[str, Any]], components: Sequence[Sequence[str]]
) -> str | None:
    eligible = {cell_id for component in components for cell_id in component}
    rows = [row for row in cell_rows if row["cell_id"] in eligible]
    if not rows:
        return None
    return str(
        min(rows, key=lambda row: (float(row["observed_h_pi"]), str(row["cell_id"])))["cell_id"]
    )


def source_manifest() -> dict[str, str]:
    return {path: file_sha256(ROOT / path) for path in SOURCE_PATHS}


def build_run_contract(*, seeds: Sequence[int]) -> dict[str, Any]:
    selection = selection_artifact()
    seed_role = (
        "burned_h_pi_validation"
        if tuple(seeds) == VALIDATION_SEEDS
        else "synthetic_test_or_explicit_noncanonical"
    )
    payload = {
        "schema_version": "program_m_shared_lift_hpi_validation_run_contract_v1",
        "scientific_commit": SCIENTIFIC_COMMIT,
        "screen_result_sha256": SCREEN_RESULT_SHA256,
        "selection_artifact_sha256": file_sha256(SELECTION_PATH),
        "governing_metric": METRIC_ID,
        "scientific_role": "fresh_h_pi_validation_only_no_hobs_no_promotion",
        "seed_role": seed_role,
        "seeds": list(map(int, seeds)),
        "cells": [asdict(cell) for cell in selected_cells()],
        "calendars": [list(calendar) for calendar in all_calendars()],
        "inference": selection["inference"],
        "source_sha256": source_manifest(),
    }
    payload["content_sha256"] = digest_json(payload)
    return payload


def validate_resume(run_dir: Path, expected_contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    contract_path = run_dir / "run_contract.json"
    frozen = json.loads(contract_path.read_text(encoding="utf-8"))
    if frozen != expected_contract:
        raise RuntimeError("resume run contract, selection, or source hash mismatch")
    progress_path = run_dir / "progress.json"
    if not progress_path.exists():
        raise FileNotFoundError("--resume requires progress.json")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("run_contract_sha256") != file_sha256(contract_path):
        raise RuntimeError("resume progress-to-contract custody mismatch")
    allowed = {
        (cell.cell_id, int(seed))
        for cell in selected_cells()
        for seed in expected_contract["seeds"]
    }
    seen: set[tuple[str, int]] = set()
    for record in progress.get("completed_shards", []):
        key = (str(record["cell_id"]), int(record["seed"]))
        if key not in allowed or key in seen:
            raise RuntimeError(f"resume shard key custody mismatch: {key}")
        seen.add(key)
        path = run_dir / record["path"]
        if not path.is_file() or file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"resume shard custody mismatch: {record['path']}")
        shard = json.loads(path.read_text(encoding="utf-8"))
        if (
            (str(shard["cell"]["cell_id"]), int(shard["seed"])) != key
            or int(shard["n_calendars"]) != 256
            or len(shard["evaluations"]) != 256
        ):
            raise RuntimeError(f"resume shard payload custody mismatch: {record['path']}")
    return list(progress.get("completed_shards", []))


def execute(
    *,
    run_dir: Path,
    seeds: Sequence[int],
    workers: int,
    resume: bool,
    evaluator: Callable[[Mapping[str, Any]], dict[str, Any]] = evaluate_cell_tape,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    run_contract = build_run_contract(seeds=seeds)
    contract_path = run_dir / "run_contract.json"
    progress_path = run_dir / "progress.json"
    if resume:
        completed = validate_resume(run_dir, run_contract)
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError("refusing to overwrite non-empty run directory")
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(contract_path, run_contract)
        completed = []
        atomic_json(
            progress_path,
            {
                "schema_version": "program_m_shared_lift_hpi_validation_progress_v1",
                "run_contract_sha256": file_sha256(contract_path),
                "completed_shards": [],
                "complete": False,
            },
        )
    completed_keys = {(row["cell_id"], int(row["seed"])) for row in completed}
    tasks = [
        {"cell": asdict(cell), "seed": int(seed)}
        for cell in selected_cells()
        for seed in seeds
        if (cell.cell_id, int(seed)) not in completed_keys
    ]

    def persist(payload: Mapping[str, Any]) -> None:
        cell_id = str(payload["cell"]["cell_id"])
        seed = int(payload["seed"])
        relative = Path("raw") / cell_id / f"seed_{seed}.json"
        path = run_dir / relative
        atomic_json(path, payload)
        completed.append(
            {
                "cell_id": cell_id,
                "seed": seed,
                "path": relative.as_posix(),
                "sha256": file_sha256(path),
            }
        )
        completed.sort(key=lambda row: (row["cell_id"], int(row["seed"])))
        atomic_json(
            progress_path,
            {
                "schema_version": "program_m_shared_lift_hpi_validation_progress_v1",
                "run_contract_sha256": file_sha256(contract_path),
                "completed_shards": completed,
                "completed_count": len(completed),
                "total_count": len(selected_cells()) * len(seeds),
                "complete": False,
            },
        )

    if int(workers) <= 1 or evaluator is not evaluate_cell_tape:
        for task in tasks:
            persist(evaluator(task))
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futures = {pool.submit(evaluate_cell_tape, task): task for task in tasks}
            for future in as_completed(futures):
                persist(future.result())

    expected_keys = {(cell.cell_id, int(seed)) for cell in selected_cells() for seed in seeds}
    completed_keys = {(str(row["cell_id"]), int(row["seed"])) for row in completed}
    if completed_keys != expected_keys or len(completed) != len(expected_keys):
        raise RuntimeError("validation frontier incomplete or duplicated before inference")

    matrices: dict[str, np.ndarray] = {}
    observed: dict[str, dict[str, Any]] = {}
    for cell in selected_cells():
        records = sorted(
            (row for row in completed if row["cell_id"] == cell.cell_id),
            key=lambda row: int(row["seed"]),
        )
        shards = [
            json.loads((run_dir / row["path"]).read_text(encoding="utf-8")) for row in records
        ]
        matrices[cell.cell_id] = _matrix(shards)
        observed[cell.cell_id] = observed_hpi(matrices[cell.cell_id])
    inference = bootstrap_simultaneous_lcbs(
        matrices, n_resamples=int(n_resamples), rng_seed=BOOTSTRAP_RNG_SEED
    )
    rows = []
    for cell in selected_cells():
        row = {"cell_id": cell.cell_id, "cell": asdict(cell), **observed[cell.cell_id]}
        row["simultaneous_lcb95"] = inference["simultaneous_lcb95"][cell.cell_id]
        row["passes_h_pi_gate"] = bool(row["simultaneous_lcb95"] >= PRACTICAL_GATE)
        rows.append(row)
    components = passing_components(rows)
    selected = select_least_observed(rows, components)
    result = {
        "schema_version": "program_m_shared_lift_hpi_validation_v1",
        "status": (
            "HPI_VALIDATED_CONNECTED_REGION__NO_HOBS_OR_LEARNER_AUTHORIZED"
            if selected is not None
            else "HPI_VALIDATION_NO_PASSING_CONNECTED_REGION__STOP_BEFORE_HOBS"
        ),
        "run_contract_sha256": file_sha256(contract_path),
        "scientific_commit": SCIENTIFIC_COMMIT,
        "screen_result_sha256": SCREEN_RESULT_SHA256,
        "governing_metric": METRIC_ID,
        "n_cells": 6,
        "n_tapes_per_cell": len(seeds),
        "n_calendars": 256,
        "n_des_evaluations": 6 * len(seeds) * 256,
        "cell_results": rows,
        "simultaneous_inference": inference,
        "passing_connected_components": components,
        "selected_least_observed_h_pi_cell": selected,
        "authorization": {
            "h_obs": False,
            "learner": False,
            "paper2": False,
            "paper3": False,
            "virgin_tapes": False,
        },
    }
    atomic_json(run_dir / "result.json", result)
    atomic_json(
        progress_path,
        {
            "schema_version": "program_m_shared_lift_hpi_validation_progress_v1",
            "run_contract_sha256": file_sha256(contract_path),
            "completed_shards": completed,
            "completed_count": len(completed),
            "total_count": len(selected_cells()) * len(seeds),
            "complete": True,
            "result_sha256": file_sha256(run_dir / "result.json"),
        },
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "outputs/program_m_shared_lift_hpi_validation_v1",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    execute(
        run_dir=args.run_dir.resolve(),
        seeds=VALIDATION_SEEDS,
        workers=max(1, int(args.workers)),
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()
