#!/usr/bin/env python3
"""Fail-closed, read-only salvage for factorial-v4 packaging failures."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_q_r1_matched_retention_factorial_v4 import (  # noqa: E402
    CONTRACT_PATH,
    FACTORIAL_ARMS,
    KAPPAS,
    PRIMARY,
    RHO,
    build_histories,
    estimands,
    integer_range,
    sha256,
    static_rows,
    validate_shared_static_bar,
    write_json,
)


AMENDMENT_PATH = (
    ROOT / "contracts/q_r1_matched_retention_factorial_v4_1_salvage.json"
)
FREEZE_PATH = (
    ROOT / "contracts/q_r1_matched_retention_factorial_v4_1_salvage_freeze_receipt.json"
)
SERVICE_FIELDS = (
    "worst_product_fill",
    "unresolved_orders",
    "unresolved_quantity",
    "lost_orders",
    "lost_quantity",
    "service_loss",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _expected_workers(amendment: dict[str, Any]) -> set[tuple[str, int]]:
    return {
        (str(row["config_id"]), int(row["optimizer_seed"]))
        for row in amendment["salvage_source_workers"]
    }


def _worker_files(source: Path) -> list[Path]:
    return sorted(
        path
        for path in source.iterdir()
        if path.is_file()
        and (
            path.name in {
                "worker_opening_reference.json",
                "checkpoint_progress.json",
                "structured_rows.json",
                "structured_progress.json",
                "static_bar_reference.json",
            }
            or path.name.startswith("checkpoint_t")
            or path.name.startswith("checkpoint_rows_t")
        )
    )


def inventory(source_root: Path, amendment: dict[str, Any]) -> dict[str, Any]:
    workers: list[dict[str, Any]] = []
    found: set[tuple[str, int]] = set()
    for source in sorted(path for path in source_root.iterdir() if path.is_dir()):
        if not source.name.startswith("s"):
            continue
        config, seed_text = source.name.split("_", 1)
        key = (config, int(seed_text))
        if key not in _expected_workers(amendment):
            continue
        found.add(key)
        progress = _load(source / "checkpoint_progress.json")
        structured = _load(source / "structured_progress.json")
        workers.append(
            {
                "source_name": source.name,
                "config_id": config,
                "optimizer_seed": int(seed_text),
                "completed_timesteps": list(progress["completed_timesteps"]),
                "checkpoint_count": len(progress["checkpoint_receipts"]),
                "structured_rows": int(structured["rows"]),
                "files": [
                    {
                        "name": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                    for path in _worker_files(source)
                ],
            }
        )
    if found != _expected_workers(amendment):
        raise RuntimeError("salvage source worker coverage mismatch")
    return {
        "schema_version": "q_r1_factorial_v4_salvage_source_inventory",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_status": "STRUCTURAL_INVENTORY_NO_OUTCOME_INSPECTION",
        "base_contract_sha256": sha256(CONTRACT_PATH),
        "amendment_sha256": sha256(AMENDMENT_PATH),
        "source_root": str(source_root.resolve()),
        "workers": workers,
        "confirmation_roots_opened": False,
    }


def _validate_authority(
    amendment_path: Path,
    freeze_path: Path,
    inventory_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    amendment = _load(amendment_path)
    freeze = _load(freeze_path)
    manifest = _load(inventory_path)
    if freeze.get("status") != "FROZEN_CORRECTIVE_SALVAGE_UNOPENED":
        raise RuntimeError("salvage authority is not frozen")
    if freeze.get("base_contract_sha256") != sha256(CONTRACT_PATH):
        raise RuntimeError("base contract hash mismatch")
    if freeze.get("amendment_sha256") != sha256(amendment_path):
        raise RuntimeError("salvage amendment hash mismatch")
    if freeze.get("source_inventory_sha256") != sha256(inventory_path):
        raise RuntimeError("salvage source inventory hash mismatch")
    if freeze.get("returns_or_scores_inspected_before_freeze") is not False:
        raise RuntimeError("salvage was not frozen before outcome inspection")
    if freeze.get("confirmation_roots_opened") is not False:
        raise RuntimeError("confirmation roots were opened")
    return amendment, freeze, manifest


def _manifest_worker(
    manifest: dict[str, Any], config: str, seed: int
) -> dict[str, Any]:
    matches = [
        row
        for row in manifest["workers"]
        if row["config_id"] == config and int(row["optimizer_seed"]) == seed
    ]
    if len(matches) != 1:
        raise RuntimeError("worker is absent or duplicated in source inventory")
    return matches[0]


def _verify_source_hashes(source: Path, manifest_worker: dict[str, Any]) -> None:
    expected = {
        row["name"]: row["sha256"] for row in manifest_worker["files"]
    }
    actual_files = _worker_files(source)
    if {path.name for path in actual_files} != set(expected):
        raise RuntimeError("source file inventory changed")
    for path in actual_files:
        if sha256(path) != expected[path.name]:
            raise RuntimeError(f"source hash mismatch: {path.name}")


def _row_key(row: dict[str, Any]) -> tuple[int, float, int]:
    return (
        int(row["history_root"]),
        float(row["kappa"]),
        int(row["campaign_index"]),
    )


def _validate_service(rows: list[dict[str, Any]]) -> None:
    for field in SERVICE_FIELDS:
        if any(field not in row for row in rows):
            raise RuntimeError(f"mandatory service field missing: {field}")


def _validate_checkpoint_rows(
    rows: list[dict[str, Any]],
    *,
    config: str,
    seed: int,
    step: int,
    checkpoint_sha: str,
    evaluation_roots: list[int],
) -> None:
    expected_arms = {row[0] for row in FACTORIAL_ARMS}
    counts = Counter(str(row["arm"]) for row in rows)
    expected_per_arm = len(evaluation_roots) * len(KAPPAS) * 12
    if counts != Counter({arm: expected_per_arm for arm in expected_arms}):
        raise RuntimeError("factorial row coverage mismatch")
    if {int(row["history_root"]) for row in rows} != set(evaluation_roots):
        raise RuntimeError("factorial root coverage mismatch")
    if {float(row["kappa"]) for row in rows} != set(KAPPAS):
        raise RuntimeError("factorial kappa coverage mismatch")
    if {int(row["campaign_index"]) for row in rows} != set(range(12)):
        raise RuntimeError("factorial campaign coverage mismatch")
    if {str(row["config_id"]) for row in rows} != {config}:
        raise RuntimeError("factorial config mismatch")
    if {int(row["optimizer_seed"]) for row in rows} != {seed}:
        raise RuntimeError("factorial optimizer seed mismatch")
    if {int(row["timesteps"]) for row in rows} != {step}:
        raise RuntimeError("factorial checkpoint step mismatch")
    if {str(row["checkpoint_sha256"]) for row in rows} != {checkpoint_sha}:
        raise RuntimeError("factorial checkpoint identity mismatch")
    _validate_service(rows)


def _selection_key(
    rows: list[dict[str, Any]],
    structured_mean: dict[tuple[int, float, int], float],
    step: int,
) -> tuple[float, float, float, float, int]:
    retained = [row for row in rows if row["arm"] == "P1_H1"]
    primary = float(np.mean([row[PRIMARY] for row in retained]))
    premium_values = [
        float(row[PRIMARY]) - structured_mean[_row_key(row)]
        for row in retained
        if _row_key(row) in structured_mean
    ]
    premium = float(np.mean(premium_values)) if premium_values else float("-inf")
    point_estimands = estimands(rows)
    total = float(point_estimands["total_retained_neural_treatment"]["mean"])
    iid_rows = [row for row in rows if float(row["kappa"]) == 0.5]
    iid_effect = estimands(iid_rows)["total_retained_neural_treatment"]["mean"]
    return primary, premium, total, -abs(float(iid_effect)), -step


def salvage(
    *,
    source: Path,
    output: Path,
    inventory_path: Path,
    static_bar_path: Path,
    static_completion_path: Path,
    static_opening_path: Path,
    amendment_path: Path = AMENDMENT_PATH,
    freeze_path: Path = FREEZE_PATH,
) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    started = time.perf_counter()
    amendment, freeze, manifest = _validate_authority(
        amendment_path, freeze_path, inventory_path
    )
    config, seed_text = source.name.split("_", 1)
    seed = int(seed_text)
    if (config, seed) not in _expected_workers(amendment):
        raise RuntimeError("worker is not authorized for salvage")
    manifest_worker = _manifest_worker(manifest, config, seed)
    _verify_source_hashes(source, manifest_worker)

    contract = _load(CONTRACT_PATH)
    roots = integer_range(contract["data_splits"]["checkpoint_selection_history_roots"])
    static_bar = validate_shared_static_bar(
        static_bar_path=static_bar_path,
        completion_receipt_path=static_completion_path,
        opening_receipt_path=static_opening_path,
        expected_contract_sha256=sha256(CONTRACT_PATH),
        expected_roots=roots,
        expected_campaigns=len(roots) * len(KAPPAS) * 12,
    )
    opening_ref = _load(source / "worker_opening_reference.json")
    if opening_ref.get("contract_sha256") != sha256(CONTRACT_PATH):
        raise RuntimeError("worker contract mismatch")
    if opening_ref.get("static_bar_completion_receipt_sha256") != sha256(
        static_completion_path
    ):
        raise RuntimeError("worker static-bar receipt mismatch")

    progress = _load(source / "checkpoint_progress.json")
    checkpoint_interval = int(
        contract["training_protocol"]["checkpoint_interval_timesteps"]
    )
    total = int(contract["training_protocol"]["screen_timesteps_per_seed"])
    checkpoints = list(range(0, total + 1, checkpoint_interval))
    if list(map(int, progress["completed_timesteps"])) != checkpoints:
        raise RuntimeError("checkpoint sequence is incomplete")
    if progress.get("confirmation_roots_opened") is not False:
        raise RuntimeError("checkpoint progress opened confirmation")

    checkpoint_rows: dict[int, list[dict[str, Any]]] = {}
    checkpoint_receipts: list[dict[str, Any]] = []
    for receipt in progress["checkpoint_receipts"]:
        step = int(receipt["timesteps"])
        archive = source / str(receipt["path"])
        rows_path = source / str(receipt["rows_path"])
        if sha256(archive) != receipt["sha256"]:
            raise RuntimeError("checkpoint archive hash mismatch")
        if sha256(rows_path) != receipt["rows_sha256"]:
            raise RuntimeError("checkpoint rows hash mismatch")
        rows = _load(rows_path)
        _validate_checkpoint_rows(
            rows,
            config=config,
            seed=seed,
            step=step,
            checkpoint_sha=str(receipt["sha256"]),
            evaluation_roots=roots,
        )
        checkpoint_rows[step] = rows
        checkpoint_receipts.append(dict(receipt))
    if sorted(checkpoint_rows) != checkpoints:
        raise RuntimeError("checkpoint receipt coverage mismatch")

    structured_path = source / "structured_rows.json"
    structured_progress = _load(source / "structured_progress.json")
    if structured_progress.get("complete") is not True:
        raise RuntimeError("structured evaluation is incomplete")
    if structured_progress.get("rows_sha256") != sha256(structured_path):
        raise RuntimeError("structured rows hash mismatch")
    structured = _load(structured_path)
    if len(structured) != 192:
        raise RuntimeError("structured row count mismatch")
    if Counter(row["arm"] for row in structured) != Counter(
        {"structured_reset": 96, "structured_retained": 96}
    ):
        raise RuntimeError("structured arm coverage mismatch")
    if {int(row["history_root"]) for row in structured} != set(roots):
        raise RuntimeError("structured root coverage mismatch")
    if {float(row["kappa"]) for row in structured} != set(KAPPAS):
        raise RuntimeError("structured kappa coverage mismatch")
    if {int(row["campaign_index"]) for row in structured} != {0, 1}:
        raise RuntimeError("structured campaign coverage mismatch")
    _validate_service(structured)

    structured_mean = {
        _row_key(row): float(row[PRIMARY])
        for row in structured
        if row["arm"] == "structured_retained"
    }
    scores = {
        step: _selection_key(checkpoint_rows[step], structured_mean, step)
        for step in checkpoints
    }
    selected_step = max(checkpoints, key=lambda step: scores[step])
    selected_checkpoint = next(
        row for row in checkpoint_receipts if int(row["timesteps"]) == selected_step
    )
    histories = build_histories(roots, KAPPAS)
    bar_rows = static_rows(histories, calendar=list(map(int, static_bar["calendar"])))
    selected_rows = checkpoint_rows[selected_step] + structured + bar_rows
    all_checkpoint_rows = [
        row for step in checkpoints for row in checkpoint_rows[step]
    ]

    output.mkdir(parents=True)
    write_json(output / "checkpoint_rows.json", all_checkpoint_rows)
    write_json(output / "rows.json", selected_rows)
    result = {
        "schema_version": "q_r1_matched_retention_factorial_v4_run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_status": "DEVELOPMENT_SELECTION_NO_CONFIRMATORY_CLAIM",
        "mode": "development-worker",
        "contract_sha256": sha256(CONTRACT_PATH),
        "config_id": config,
        "optimizer_seed": seed,
        "rho": RHO,
        "kappa_cells": list(KAPPAS),
        "training_roots": integer_range(
            contract["data_splits"]["training_history_roots"]
        ),
        "evaluation_roots": roots,
        "static_bar": {
            "calendar": static_bar["calendar"],
            "frontier_row": static_bar["frontier_row"],
            "sha256": sha256(static_bar_path),
        },
        "checkpoints": checkpoint_receipts,
        "selected_checkpoint": selected_checkpoint,
        "checkpoint_selection_scores": {
            str(step): list(map(float, scores[step])) for step in checkpoints
        },
        "selection_rule": contract["training_protocol"]["checkpoint_selection"],
        "structured_comparator": structured_progress,
        "estimands": estimands(selected_rows),
        "arm_counts": {
            arm: sum(row["arm"] == arm for row in selected_rows)
            for arm in (
                "P0_H0",
                "P1_H0",
                "P0_H1",
                "P1_H1",
                "structured_reset",
                "structured_retained",
                "best_static_frozen",
            )
        },
        "same_checkpoint_hash_all_neural_arms": len(
            {
                row["checkpoint_sha256"]
                for row in selected_rows
                if str(row["arm"]).startswith("P")
            }
        )
        == 1,
        "salvage": {
            "status": "MECHANICAL_PACKAGING_SALVAGE",
            "source_attempt": str(source.resolve()),
            "source_inventory_sha256": sha256(inventory_path),
            "amendment_sha256": sha256(amendment_path),
            "freeze_receipt_sha256": sha256(freeze_path),
            "source_read_only": True,
            "scientific_rules_changed": False,
            "thresholds_changed": False,
            "configuration_changed": False,
            "action_eligible": True,
        },
        "confirmation_roots_opened": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    result["checkpoint_rows_sha256"] = sha256(output / "checkpoint_rows.json")
    result["rows_sha256"] = sha256(output / "rows.json")
    write_json(output / "result.json", result)
    write_json(
        output / "salvage_completion_receipt.json",
        {
            "schema_version": "q_r1_factorial_v4_salvage_completion_receipt",
            "status": "COMPLETE_MECHANICAL_SALVAGE",
            "result_sha256": sha256(output / "result.json"),
            "rows_sha256": result["rows_sha256"],
            "checkpoint_rows_sha256": result["checkpoint_rows_sha256"],
            "source_inventory_sha256": sha256(inventory_path),
            "source_files_rewritten": False,
            "confirmation_roots_opened": False,
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--source-root", type=Path, required=True)
    inv.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("salvage")
    run.add_argument("--source", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--inventory", type=Path, required=True)
    run.add_argument("--static-bar", type=Path, required=True)
    run.add_argument("--static-completion", type=Path, required=True)
    run.add_argument("--static-opening", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inventory":
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        payload = inventory(args.source_root, _load(AMENDMENT_PATH))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    result = salvage(
        source=args.source,
        output=args.output,
        inventory_path=args.inventory,
        static_bar_path=args.static_bar,
        static_completion_path=args.static_completion,
        static_opening_path=args.static_opening,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
