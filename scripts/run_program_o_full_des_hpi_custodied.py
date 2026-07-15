#!/usr/bin/env python3
"""Run the Program O producer and always emit an exit custody manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("development", "validation"), required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--execution-freeze", type=Path, required=True)
    parser.add_argument("--seed-claim", type=Path, required=True)
    parser.add_argument("--development-result", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    command = [
        sys.executable,
        str(ROOT / "scripts/screen_program_o_full_des_hpi.py"),
        "--stage",
        str(args.stage),
        "--workers",
        str(args.workers),
        "--contract",
        str(args.contract.resolve()),
        "--output-root",
        str(run_dir / "artifacts"),
        "--run-dir",
        str(run_dir),
        "--run-id",
        str(args.run_id),
        "--execution-freeze",
        str(args.execution_freeze.resolve()),
        "--seed-claim",
        str(args.seed_claim.resolve()),
    ]
    if args.development_result is not None:
        command.extend(["--development-result", str(args.development_result.resolve())])
    started = now_utc()
    returncode = 1
    error = None
    try:
        returncode = int(subprocess.run(command, cwd=ROOT, check=False).returncode)
    except BaseException:
        error = traceback.format_exc()
        returncode = 1
    exit_state = {
        "schema_version": "program_o_full_des_producer_exit_v1",
        "started_at_utc": started,
        "finished_at_utc": now_utc(),
        "returncode": returncode,
        "runner_pid": os.getpid(),
        "runner_pgid": os.getpgid(0),
        "runner_sid": os.getsid(0),
        "command": command,
        "error": error,
    }
    write_json_atomic(run_dir / "custody" / "producer_exit.json", exit_state)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
