#!/usr/bin/env python3
"""Custodied Program Q producer: shards, reduction, direct replay, adjudication."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.audit_program_q_full_des import audit as direct_audit  # noqa: E402
from scripts.adjudicate_program_q import adjudicate  # noqa: E402
from scripts.evaluate_program_q_replication import (  # noqa: E402
    CONTRACT,
    _contract,
    _frozen_seeds,
    produce_shard,
    reduce_shards,
    verify_authorization,
    verify_model_hashes,
)
from supply_chain.program_o_eval_custody import (  # noqa: E402
    sha256,
    verify_sha256_manifest,
    write_sha256_manifest,
)
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS  # noqa: E402


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _task(arguments: tuple[int, int, str, str]) -> str:
    cell_index, tape_seed, models, shards = arguments
    return str(
        produce_shard(
            cell_index=cell_index,
            tape_seed=tape_seed,
            models_dir=Path(models),
            output=Path(shards),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--watcher-ready-sha256", required=True)
    parser.add_argument("--attempt-id", type=int, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    custody = run_dir / "custody"
    artifacts = run_dir / "artifacts/confirmation"
    shards = artifacts / "shards"
    evaluation = artifacts / "evaluation"
    audit_dir = artifacts / "direct_audit"
    adjudication_path = artifacts / "adjudication.json"
    failures: list[str] = []
    returncode = 1
    try:
        if not (custody / "watcher_ready.json").is_file():
            raise RuntimeError("Program Q producer refuses to start before watcher readiness")
        if sha256(custody / "watcher_ready.json") != args.watcher_ready_sha256:
            raise RuntimeError("Program Q watcher readiness hash mismatch")
        verify_authorization(args.authorization)
        verify_model_hashes(args.models)
        contract = _contract()
        seeds = _frozen_seeds(contract)
        tasks = [
            (cell_index, seed, str(args.models.resolve()), str(shards))
            for cell_index in range(len(CONFIRMED_RET_CELLS))
            for seed in seeds
        ]
        completed = len(list(shards.rglob("*.npz"))) if shards.exists() else 0
        write_json_atomic(
            artifacts / "progress.json",
            {
                "updated_at": now_utc(),
                "stage": "shards",
                "completed": completed,
                "expected": len(tasks),
            },
        )
        with ProcessPoolExecutor(
            max_workers=int(args.workers), max_tasks_per_child=1
        ) as executor:
            futures = [executor.submit(_task, task) for task in tasks]
            completed = 0
            for future in as_completed(futures):
                future.result()
                completed += 1
                write_json_atomic(
                    artifacts / "progress.json",
                    {
                        "updated_at": now_utc(),
                        "stage": "shards",
                        "completed": completed,
                        "expected": len(tasks),
                    },
                )
        direct_path = audit_dir / "independent_full_des_audit.json"
        result_path = evaluation / "result.json"
        if not result_path.is_file():
            write_json_atomic(
                artifacts / "progress.json",
                {"updated_at": now_utc(), "stage": "reduction", "completed": 0, "expected": 1},
            )
            reduce_shards(
                shards=shards,
                output=evaluation,
                resamples=int(contract["confirmation"]["bootstrap_resamples"]),
            )
        else:
            verify_sha256_manifest(evaluation, evaluation / "evaluation_files.sha256")
            verify_sha256_manifest(shards, evaluation / "shard_files.sha256")
        if not direct_path.is_file():
            write_json_atomic(
                artifacts / "progress.json",
                {"updated_at": now_utc(), "stage": "direct_replay", "completed": 0, "expected": 1},
            )
            direct = direct_audit(evaluation=evaluation, shards=shards, atol=1e-8)
            audit_dir.mkdir(parents=True, exist_ok=True)
            direct_path.write_text(json.dumps(direct, indent=2, sort_keys=True) + "\n")
            write_sha256_manifest(audit_dir, [direct_path], audit_dir / "audit_files.sha256")
        else:
            verify_sha256_manifest(audit_dir, audit_dir / "audit_files.sha256")
            direct = json.loads(direct_path.read_text())
            if (
                direct.get("passed") is not True
                or direct.get("contract_sha256") != sha256(CONTRACT)
                or direct.get("evaluation_result_sha256") != sha256(result_path)
            ):
                raise RuntimeError("existing Program Q direct audit failed custody binding")
        result = json.loads(result_path.read_text())
        terminal = adjudicate(result, contract, direct)
        terminal.update(
            {
                "evaluation_result_sha256": sha256(result_path),
                "direct_audit_sha256": sha256(direct_path),
                "contract_sha256": sha256(CONTRACT),
            }
        )
        adjudication_path.write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n")
        write_json_atomic(
            artifacts / "progress.json",
            {"updated_at": now_utc(), "stage": "complete", "completed": 1, "expected": 1},
        )
        returncode = 0
    except BaseException as error:
        failures.append(f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json_atomic(
            custody / "producer_exit.json",
            {
                "finished_at": now_utc(),
                "returncode": returncode,
                "failures": failures,
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "attempt_id": int(args.attempt_id),
            },
        )
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
