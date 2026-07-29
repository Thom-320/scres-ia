#!/usr/bin/env python3
"""Aggregate Q-R1 factorial v4 workers without opening or simulating data."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "contracts/q_r1_matched_retention_factorial_v4.json"
FULL_PHASE_FREEZE_PATH = (
    ROOT
    / "contracts/q_r1_factorial_v4_full_phase_runner_amendment_v1_freeze_receipt.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_workers(
    paths: list[Path], *, expected_phase: str
) -> list[tuple[Path, dict[str, Any]]]:
    workers: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("schema_version") != "q_r1_matched_retention_factorial_v4_run":
            raise ValueError(f"{path} is not a factorial v4 worker")
        if payload.get("mode") != "development-worker":
            raise ValueError(f"{path} is not a development worker")
        if payload.get("contract_sha256") != sha256(CONTRACT_PATH):
            raise ValueError(f"{path} contract hash mismatch")
        if payload.get("confirmation_roots_opened") is not False:
            raise ValueError(f"{path} opened confirmation")
        if payload.get("same_checkpoint_hash_all_neural_arms") is not True:
            raise ValueError(f"{path} did not use the same neural checkpoint")
        actual_phase = payload.get("development_phase", "screen")
        if actual_phase != expected_phase:
            raise ValueError(
                f"{path} is phase {actual_phase}, expected {expected_phase}"
            )
        expected_limit = 96_000 if expected_phase == "screen" else 240_000
        expected_steps = list(range(0, expected_limit + 1, 24_000))
        actual_steps = [int(row["timesteps"]) for row in payload["checkpoints"]]
        if actual_steps != expected_steps:
            raise ValueError(
                f"{path} checkpoint schedule does not match {expected_phase}"
            )
        workers.append((path, payload))
    return workers


def score(payloads: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    selected_scores = []
    for payload in payloads:
        step = str(int(payload["selected_checkpoint"]["timesteps"]))
        selected_scores.append(payload["checkpoint_selection_scores"][step])
    matrix = np.asarray(selected_scores, dtype=float)
    return tuple(map(float, matrix[:, :4].mean(axis=0)))


def rank(
    workers: list[tuple[Path, dict[str, Any]]],
    *,
    expected_configs: set[str],
    expected_seeds: set[int],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for path, payload in workers:
        config = str(payload["config_id"])
        seed = int(payload["optimizer_seed"])
        key = (config, seed)
        if key in seen:
            raise ValueError(f"duplicate worker for {key}")
        seen.add(key)
        grouped[config].append((path, payload))
    if set(grouped) != expected_configs:
        raise ValueError("worker configuration coverage is incomplete")
    for config, rows in grouped.items():
        seeds = {int(payload["optimizer_seed"]) for _path, payload in rows}
        if seeds != expected_seeds:
            raise ValueError(f"worker seed coverage is incomplete for {config}")

    ranking: list[dict[str, Any]] = []
    for config, rows in grouped.items():
        metrics = score([payload for _path, payload in rows])
        ranking.append(
            {
                "config_id": config,
                "optimizer_seeds": sorted(
                    int(payload["optimizer_seed"]) for _path, payload in rows
                ),
                "mean_selected_checkpoint_primary": metrics[0],
                "mean_neural_premium": metrics[1],
                "mean_total_retained_neural_treatment": metrics[2],
                "negative_mean_absolute_iid_effect": metrics[3],
                "worker_results": [
                    {
                        "path": str(path),
                        "sha256": sha256(path),
                    }
                    for path, _payload in sorted(rows)
                ],
            }
        )
    return sorted(
        ranking,
        key=lambda row: (
            -float(row["mean_selected_checkpoint_primary"]),
            -float(row["mean_neural_premium"]),
            -float(row["mean_total_retained_neural_treatment"]),
            -float(row["negative_mean_absolute_iid_effect"]),
            str(row["config_id"]),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("screen", "full"), required=True)
    parser.add_argument("--worker-results", type=Path, nargs="+", required=True)
    parser.add_argument("--screen-selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    contract = json.loads(CONTRACT_PATH.read_text())
    configs = {
        str(row["id"]) for row in contract["training_protocol"]["screen_configurations"]
    }
    if args.phase == "screen":
        expected_configs = configs
        expected_seeds = set(
            map(
                int,
                contract["training_protocol"][
                    "development_screen_optimizer_seeds"
                ],
            )
        )
    else:
        full_freeze = json.loads(FULL_PHASE_FREEZE_PATH.read_text())
        if full_freeze.get("status") != "FROZEN_BEFORE_FULL_RESULTS":
            raise ValueError("full phase is not frozen")
        if full_freeze.get("aggregator_sha256") != sha256(Path(__file__)):
            raise ValueError("full phase aggregator hash mismatch")
        if args.screen_selection is None or not args.screen_selection.is_file():
            raise ValueError("full aggregation requires the screen selection")
        screen = json.loads(args.screen_selection.read_text())
        if screen.get("phase") != "screen":
            raise ValueError("screen selection has the wrong phase")
        if screen.get("contract_sha256") != sha256(CONTRACT_PATH):
            raise ValueError("screen selection contract hash mismatch")
        expected_configs = set(map(str, screen["advanced_config_ids"]))
        expected_seeds = set(
            map(int, contract["training_protocol"]["full_optimizer_seeds"])
        )

    workers = load_workers(args.worker_results, expected_phase=args.phase)
    ranking = rank(
        workers,
        expected_configs=expected_configs,
        expected_seeds=expected_seeds,
    )
    advance = (
        int(contract["training_protocol"]["configuration_selection"]["screen_advances"])
        if args.phase == "screen"
        else 1
    )
    result = {
        "schema_version": "q_r1_factorial_v4_configuration_selection",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "claim_status": "DEVELOPMENT_SELECTION_NO_CONFIRMATORY_CLAIM",
        "contract_sha256": sha256(CONTRACT_PATH),
        "worker_count": len(workers),
        "ranking": ranking,
        "advanced_config_ids": [
            str(row["config_id"]) for row in ranking[:advance]
        ],
        "confirmation_return_used": False,
        "oracle_return_used": False,
        "confirmation_roots_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
