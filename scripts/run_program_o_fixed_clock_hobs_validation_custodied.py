#!/usr/bin/env python3
"""Run frozen Program O H_obs validation and always record producer exit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.program_o_full_des_guard import verify_seed_claim  # noqa: E402


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
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--execution-freeze", type=Path, required=True)
    parser.add_argument("--seed-claim", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    contract_path = args.contract.resolve()
    freeze_path = args.execution_freeze.resolve()
    freeze = json.loads(freeze_path.read_text())
    authorization = {
        **freeze["authorization"],
        "freeze_sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
    }
    verify_seed_claim(
        claim_path=args.seed_claim.resolve(),
        authorization=authorization,
        contract_sha256=str(freeze["contract_sha256"]),
    )
    packages = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    ).splitlines()
    write_json_atomic(
        run_dir / "custody" / "environment_manifest.json",
        {
            "schema_version": "program_o_fixed_clock_hobs_environment_v1",
            "captured_at_utc": now_utc(),
            "run_id": str(args.run_id),
            "scientific_commit": str(freeze["scientific_commit"]),
            "executable": sys.executable,
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": packages,
        },
    )
    command = [
        sys.executable,
        str(ROOT / "scripts/screen_program_o_fixed_clock_hobs_validation.py"),
        "--contract",
        str(contract_path),
        "--output",
        str(run_dir / "artifacts" / "validation"),
        "--workers",
        str(args.workers),
    ]
    started = now_utc()
    returncode = 1
    error = None
    try:
        returncode = int(subprocess.run(command, cwd=ROOT, check=False).returncode)
    except BaseException:
        error = traceback.format_exc()
    write_json_atomic(
        run_dir / "custody" / "producer_exit.json",
        {
            "schema_version": "program_o_fixed_clock_hobs_validation_exit_v1",
            "started_at_utc": started,
            "finished_at_utc": now_utc(),
            "returncode": returncode,
            "runner_pid": os.getpid(),
            "runner_pgid": os.getpgid(0),
            "runner_sid": os.getsid(0),
            "command": command,
            "error": error,
            "run_id": str(args.run_id),
            "execution_freeze": str(freeze_path),
        },
    )
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
