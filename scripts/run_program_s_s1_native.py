#!/usr/bin/env python3
"""Custodied two-worker producer for the 5,760 frozen Program S S1 shards."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supply_chain.program_o_full_des_transducer import MATRIX_KEYS  # noqa: E402


DESIGN_PATH = ROOT / "research/paper2_exhaustive_search/program_s_native_morris_design_v1_1.json"
AUDIT_PATH = ROOT / "research/paper2_exhaustive_search/program_s_s1_preopen_audit_v1_2.json"
SHARD_RUNNER = ROOT / "scripts/run_program_s_s1_shard.py"
EXPECTED_SHARDS = 5_760


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def tasks() -> list[tuple[int, int, int, str, int]]:
    design = json.loads(DESIGN_PATH.read_text())
    rows = []
    for group_index, group in enumerate(design["groups"]):
        if group["stratum"] != "THESIS_NATIVE_INDEPENDENT":
            raise RuntimeError("S1 producer accepts S-NATIVE groups only")
        for trajectory_index, trajectory in enumerate(group["trajectories"]):
            for point_index, _point in enumerate(trajectory["points"]):
                for product_cell in sorted(design["product_cells"]):
                    for seed in range(7_510_001, 7_510_013):
                        rows.append(
                            (group_index, trajectory_index, point_index, product_cell, seed)
                        )
    if len(rows) != EXPECTED_SHARDS or len(set(rows)) != EXPECTED_SHARDS:
        raise AssertionError(f"frozen S1 task identity mismatch: {len(rows)}")
    return rows


def shard_path(output_root: Path, task: tuple[int, int, int, str, int]) -> Path:
    group, trajectory, point, product_cell, seed = task
    identity = (
        f"g{group:02d}__t{trajectory:02d}__p{point:02d}"
        f"__{product_cell}__seed{seed}"
    )
    return output_root / "matrices" / f"{identity}.npz"


def run_one(task: tuple[int, int, int, str, int], output_root: str) -> dict:
    group, trajectory, point, product_cell, seed = task
    command = [
        sys.executable,
        str(SHARD_RUNNER),
        "--group", str(group),
        "--trajectory", str(trajectory),
        "--point", str(point),
        "--product-cell", product_cell,
        "--seed", str(seed),
        "--output-root", output_root,
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"S1 shard failed {task}: stdout={completed.stdout[-2000:]} "
            f"stderr={completed.stderr[-4000:]}"
        )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def pending_tasks(output_root: Path, rows: Iterable[tuple[int, int, int, str, int]]) -> list:
    pending = []
    for task in rows:
        path = shard_path(output_root, task)
        if path.exists():
            try:
                if path.stat().st_size <= 0:
                    raise RuntimeError(f"zero-byte S1 shard: {path}")
                with np.load(path, allow_pickle=False) as shard:
                    required = set(MATRIX_KEYS) | {
                        "classical_calendar_index",
                        "classical_calendar",
                        "oracle_calendar_index",
                        "risk_event_tape_sha256",
                        "base_stream_sha256",
                        "skeleton_sha256",
                        "cell_id",
                        "observation_sha256",
                        "direct_replay_max_abs_error",
                    }
                    missing = required - set(shard.files)
                    if missing:
                        raise RuntimeError(
                            f"incomplete existing S1 shard {path}: {sorted(missing)}"
                        )
                    if np.asarray(shard["ret_visible"]).shape != (65_536,):
                        raise RuntimeError(f"invalid frontier shape in {path}")
                    if np.asarray(shard["classical_calendar"]).shape != (8,):
                        raise RuntimeError(f"invalid classical calendar in {path}")
                    if float(shard["direct_replay_max_abs_error"]) > 1e-10:
                        raise RuntimeError(f"replay-invalid existing shard {path}")
            except (OSError, ValueError) as error:
                raise RuntimeError(f"cannot validate existing shard {path}") from error
        else:
            pending.append(task)
    return pending


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--custody-dir", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers != 2:
        raise ValueError("Program S freezes exactly two workers")
    audit = json.loads(AUDIT_PATH.read_text())
    if audit["verdict"] != "PASS_S1_PREOPEN_AUTHORIZED" or not audit["scientific_seed_authorization"]:
        raise RuntimeError("Program S S1 lacks post-Q preopening authorization")
    args.output_root.mkdir(parents=True, exist_ok=True)
    custody_dir = args.custody_dir or args.output_root
    output_resolved = args.output_root.resolve()
    custody_resolved = custody_dir.resolve()
    if custody_resolved != output_resolved and output_resolved not in custody_resolved.parents:
        raise ValueError("custody-dir must be the run root or a child of it")
    if args.resume and custody_resolved == output_resolved:
        raise ValueError("resume requires a new immutable custody-dir")
    custody_dir.mkdir(parents=True, exist_ok=True)
    exit_path = custody_dir / "producer_exit.json"
    if exit_path.exists():
        raise FileExistsError("terminal Program S run cannot be resumed or overwritten")
    all_tasks = tasks()
    pending = pending_tasks(args.output_root, all_tasks)
    if args.resume and len(pending) == len(all_tasks):
        raise RuntimeError("resume requires at least one valid preserved shard")
    if len(pending) != len(all_tasks) and not args.resume:
        raise RuntimeError("existing shards require explicit fail-closed --resume")
    progress_path = custody_dir / "progress.json"
    started = utc_now()
    atomic_json(progress_path, {
        "status": "RUNNING",
        "started_at": started,
        "expected_shards": EXPECTED_SHARDS,
        "completed_shards": EXPECTED_SHARDS - len(pending),
        "pending_shards": len(pending),
        "workers": args.workers,
    })
    failure: str | None = None
    completed_count = EXPECTED_SHARDS - len(pending)
    try:
        with ProcessPoolExecutor(max_workers=2, max_tasks_per_child=1) as executor:
            futures = {executor.submit(run_one, task, str(args.output_root)): task for task in pending}
            for future in as_completed(futures):
                future.result()
                completed_count += 1
                atomic_json(progress_path, {
                    "status": "RUNNING",
                    "started_at": started,
                    "updated_at": utc_now(),
                    "expected_shards": EXPECTED_SHARDS,
                    "completed_shards": completed_count,
                    "pending_shards": EXPECTED_SHARDS - completed_count,
                    "last_task": list(futures[future]),
                    "workers": args.workers,
                })
    except BaseException as error:
        failure = repr(error)
        raise
    finally:
        files = sorted((args.output_root / "matrices").glob("*.npz"))
        manifest = custody_dir / "shard_files.sha256"
        if not manifest.exists():
            manifest.write_text("".join(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(args.output_root)}\n"
                for path in files
            ))
        atomic_json(exit_path, {
            "status": "COMPLETE" if failure is None and len(files) == EXPECTED_SHARDS else "FAILED",
            "finished_at": utc_now(),
            "expected_shards": EXPECTED_SHARDS,
            "completed_shards": len(files),
            "failure": failure,
            "shard_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
