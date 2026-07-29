#!/usr/bin/env python3
"""Audit a completed burned-data Program Q power preflight bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


CELL_IDS = ("rho75_share90", "rho90_share75", "rho90_share90")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_selected_N(
    power: dict[str, dict[str, float]], grid: list[int], minimum: float
) -> int | None:
    for tape_count in grid:
        row = power[str(tape_count)]
        if all(float(row[key]) >= minimum for key in ("H_OL", "Delta_N_equivalence", "joint")):
            return tape_count
    return None


def verify_manifest(run_dir: Path, manifest: Path) -> list[str]:
    failures: list[str] = []
    if not manifest.is_file():
        return ["missing_remote_checksum_manifest"]
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            failures.append(f"malformed_manifest_line:{line_number}")
            continue
        path = run_dir / relative
        if not path.is_file():
            failures.append(f"manifest_file_missing:{relative}")
        elif sha256(path) != expected:
            failures.append(f"manifest_hash_mismatch:{relative}")
    return failures


def audit(run_dir: Path, contract_path: Path) -> dict[str, Any]:
    failures: list[str] = []
    contract = read_json(contract_path)
    result_path = run_dir / "artifacts/program_q_power_v1.json"
    cache_path = run_dir / "artifacts/classical_10_cache_v1.npz"
    shards_dir = run_dir / "artifacts/classical_10_cache_v1_shards"
    ready_path = run_dir / "custody/late_watcher_ready.json"
    watcher_path = run_dir / "custody/late_watcher_state.json"
    manifest_path = run_dir / "custody/late_watcher_remote_files.sha256"

    if not result_path.is_file():
        failures.append("missing_power_result")
    if not cache_path.is_file():
        failures.append("missing_classical_cache")
    if not ready_path.is_file():
        failures.append("missing_watcher_ready")
    if not watcher_path.is_file():
        failures.append("missing_watcher_terminal")
    if failures:
        return {
            "schema_version": "program_q_power_preopen_audit_v1",
            "pass": False,
            "failures": failures,
        }

    result = read_json(result_path)
    ready = read_json(ready_path)
    watcher = read_json(watcher_path)
    if result.get("schema_version") != "program_q_power_analysis_v1":
        failures.append("wrong_power_schema")
    if result.get("status") != "BURNED_DATA_ONLY":
        failures.append("power_not_burned_data_only")
    if result.get("749_or_950_seed_opened") is not False:
        failures.append("reserved_seed_opened_or_unattested")
    expected_reselection = {
        "open_loop_65536": True,
        "classical_10": True,
        "inside_every_resample": True,
    }
    if result.get("comparator_reselection") != expected_reselection:
        failures.append("comparator_reselection_not_complete")
    if result.get("cell_ids") != list(CELL_IDS):
        failures.append("cell_identity_mismatch")

    grid = list(contract["power"]["candidate_N"])
    minimum = float(contract["power"]["minimum_joint_power"])
    if result.get("grid") != grid:
        failures.append("power_grid_mismatch")
    else:
        selected = expected_selected_N(result["power"], grid, minimum)
        if result.get("selected_N") != selected:
            failures.append("selected_N_not_minimum_passing_N")
        expected_verdict = (
            f"SELECT_N_{selected}"
            if selected is not None
            else "STOP_PROGRAM_Q_UNDERPOWERED_WITHIN_CAP"
        )
        if result.get("verdict") != expected_verdict:
            failures.append("power_verdict_mismatch")

    if ready.get("watcher_started_before_producer") is not True:
        failures.append("watcher_not_started_before_producer")
    if watcher.get("terminal") != "COMPLETE_PENDING_RETRIEVAL":
        failures.append("watcher_not_complete")
    if watcher.get("output_sha256") != sha256(result_path):
        failures.append("watcher_output_hash_mismatch")
    if watcher.get("cache_sha256") != sha256(cache_path):
        failures.append("watcher_cache_hash_mismatch")

    shard_paths = sorted(shards_dir.glob("*.npz"))
    if len(shard_paths) != 144:
        failures.append(f"classical_shard_count:{len(shard_paths)}")
    with np.load(cache_path, allow_pickle=False) as cache:
        for cell_index, cell_id in enumerate(CELL_IDS):
            seeds_key = f"{cell_id}__tape_seeds"
            indices_key = f"{cell_id}__calendar_indices"
            if seeds_key not in cache or indices_key not in cache:
                failures.append(f"cache_keys_missing:{cell_id}")
                continue
            seeds = cache[seeds_key].astype(np.int64)
            indices = cache[indices_key].astype(np.int64)
            if seeds.shape != (48,) or indices.shape != (10, 48):
                failures.append(f"cache_shape:{cell_id}:{seeds.shape}:{indices.shape}")
                continue
            if np.any(indices < 0) or np.any(indices >= 65_536):
                failures.append(f"cache_calendar_index_range:{cell_id}")
            for tape_index, tape_seed in enumerate(seeds.tolist()):
                shard = shards_dir / f"{cell_id}__tape_{tape_seed}.npz"
                if not shard.is_file():
                    failures.append(f"missing_shard:{cell_id}:{tape_seed}")
                    continue
                with np.load(shard, allow_pickle=False) as payload:
                    if int(payload["cell_index"]) != cell_index:
                        failures.append(f"shard_cell_identity:{cell_id}:{tape_seed}")
                    if int(payload["tape_seed"]) != tape_seed:
                        failures.append(f"shard_tape_identity:{cell_id}:{tape_seed}")
                    if not np.array_equal(
                        payload["calendar_indices"].astype(np.int64),
                        indices[:, tape_index],
                    ):
                        failures.append(f"shard_cache_mismatch:{cell_id}:{tape_seed}")

    failures.extend(verify_manifest(run_dir, manifest_path))
    return {
        "schema_version": "program_q_power_preopen_audit_v1",
        "run_dir": str(run_dir),
        "pass": not failures,
        "failures": failures,
        "selected_N": result.get("selected_N"),
        "power_verdict": result.get("verdict"),
        "result_sha256": sha256(result_path),
        "cache_sha256": sha256(cache_path),
        "shard_count": len(shard_paths),
        "reserved_seed_opened": result.get("749_or_950_seed_opened"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "contracts/program_q_frozen_policy_replication_v1.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = audit(args.run_dir, args.contract)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not payload["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
