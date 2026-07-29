#!/usr/bin/env python3
"""Independently validate retrieved Program O execution custody.

The producer can only emit a scientific PASS_PENDING_CUSTODY.  This validator
replays remote and stage checksums, verifies watcher-first whole-session
custody, and is the only component allowed to promote a completed validation
run to full-DES H_PI evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.program_o_full_des_guard import verify_tracked_freeze  # noqa: E402


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def verify_checksum_manifest(root: Path, manifest: Path) -> list[str]:
    failures: list[str] = []
    for line_number, line in enumerate(manifest.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            failures.append(f"malformed checksum line {line_number}")
            continue
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            failures.append(f"unsafe checksum path: {relative}")
            continue
        path = root / candidate
        if not path.is_file():
            failures.append(f"missing checksummed file: {relative}")
        elif sha256(path) != expected:
            failures.append(f"checksum mismatch: {relative}")
    return failures


def validate(
    *,
    run_dir: Path,
    stage: str,
    execution_freeze: Path,
    contract: Path,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    custody = run_dir / "custody"
    stage_root = run_dir / "artifacts" / stage
    failures: list[str] = []
    required = {
        "launch": custody / "launch_manifest.json",
        "ready": custody / "watcher_ready.json",
        "control": custody / "producer_control.json",
        "exit": custody / "producer_exit.json",
        "watcher": custody / "watcher_state.json",
        "journal": custody / "watcher_state.jsonl",
        "remote_checksums": custody / "remote_files.sha256",
        "seed_claim": custody / "seed_claim.json",
        "seed_reference": custody / "seed_claim_reference.json",
        "producer_stderr": custody / "producer.stderr.log",
        "watcher_stderr": custody / "watcher.stderr.log",
        "result": stage_root / "result.json",
        "stage_checksums": stage_root / "checksums.sha256",
    }
    for label, path in required.items():
        if not path.is_file():
            failures.append(f"missing {label}: {path}")
    if failures:
        return {
            "status": "INVALID_PROGRAM_O_FULL_DES_NO_EVIDENCE",
            "passed": False,
            "failures": failures,
        }

    launch = load_json(required["launch"])
    ready = load_json(required["ready"])
    control = load_json(required["control"])
    exit_state = load_json(required["exit"])
    watcher = load_json(required["watcher"])
    seed_reference = load_json(required["seed_reference"])
    result = load_json(required["result"])
    contract_data = load_json(contract)
    seed_range = list(map(int, contract_data["tape_blocks"][stage]["range"]))
    remote_run_dir = Path(str(launch.get("run_dir", "")))
    run_id = str(launch.get("run_id", ""))

    try:
        authorization = verify_tracked_freeze(
            freeze_path=execution_freeze,
            contract_path=contract,
            stage=stage,
            run_id=run_id,
            run_dir=remote_run_dir,
            seed_range=seed_range,
            expected_commit=str(control.get("scientific_commit", "")),
        )
    except RuntimeError as exc:
        authorization = None
        failures.append(str(exc))

    if launch.get("passed") is not True or launch.get("failures"):
        failures.append("launch preflight did not pass")
    if launch.get("stage") != stage or control.get("stage") != stage:
        failures.append("stage binding")
    if launch.get("run_id") != control.get("run_id"):
        failures.append("run id binding")
    if not (
        int(control.get("producer_pid", -1))
        == int(control.get("producer_pgid", -2))
        == int(control.get("producer_sid", -3))
    ):
        failures.append("producer was not isolated as PID=PGID=SID")
    if int(exit_state.get("returncode", 1)) != 0:
        failures.append("producer exit was nonzero")
    if watcher.get("status") != "COMPLETE_PENDING_RETRIEVAL":
        failures.append("watcher did not reach terminal retrieval state")
    if watcher.get("custody_scope_alive") is not False:
        failures.append("custody scope was not empty at terminal state")
    if int(watcher.get("group_member_count", -1)) != 0:
        failures.append("producer process group was not empty")
    if int(watcher.get("session_member_count", -1)) != 0:
        failures.append("producer session was not empty")
    if watcher.get("result_exists") is not True:
        failures.append("watcher did not observe result")
    if watcher.get("result_sha256") != sha256(required["result"]):
        failures.append("watcher result checksum mismatch")
    if watcher.get("remote_checksums_sha256") != sha256(required["remote_checksums"]):
        failures.append("remote checksum manifest hash mismatch")

    journal = [
        json.loads(line)
        for line in required["journal"].read_text().splitlines()
        if line.strip()
    ]
    if not journal or journal[0].get("status") != "AWAITING_PRODUCER_CONTROL":
        failures.append("watcher-first prestart heartbeat absent")
    if not any(row.get("status") == "RUNNING" for row in journal):
        failures.append("watcher never observed the running session")
    if ready.get("ready_at_utc", "") > control.get("launched_at_utc", ""):
        failures.append("watcher readiness was not before producer launch")

    failures.extend(verify_checksum_manifest(run_dir, required["remote_checksums"]))
    failures.extend(verify_checksum_manifest(stage_root, required["stage_checksums"]))
    producer_stderr = required["producer_stderr"]
    watcher_stderr = required["watcher_stderr"]
    if producer_stderr.stat().st_size:
        failures.append("producer stderr was nonempty")
    if watcher_stderr.stat().st_size:
        failures.append("watcher stderr was nonempty")

    if seed_reference.get("sha256") != sha256(required["seed_claim"]):
        failures.append("retrieved seed claim checksum mismatch")
    claim = load_json(required["seed_claim"])
    if authorization is not None:
        for key in (
            "scientific_commit",
            "freeze_sha256",
            "run_id",
            "run_dir",
            "stage",
            "seed_range",
        ):
            if claim.get(key) != authorization.get(key):
                failures.append(f"seed claim binding: {key}")
    if claim.get("contract_sha256") != sha256(contract):
        failures.append("seed claim contract hash")

    expected_pending = contract_data["terminal_labels"][f"{stage}_pass_pending_custody"]
    if result.get("status") != expected_pending or result.get("passed") is not True:
        failures.append("scientific result did not pass pending custody")
    if result.get("claim_boundary", {}).get("full_des_h_pi_established") is not False:
        failures.append("producer improperly promoted full-DES H_PI")
    if result.get("scientific_commit") != control.get("scientific_commit"):
        failures.append("result commit binding")
    if result.get("contract_sha256") != sha256(contract):
        failures.append("result contract binding")
    if result.get("run_id") != run_id:
        failures.append("result run id binding")
    if result.get("seeds") != list(range(seed_range[0], seed_range[1] + 1)):
        failures.append("result seed block binding")

    passed = not failures
    status = (
        contract_data["terminal_labels"][f"{stage}_pass_after_custody"]
        if passed
        else contract_data["terminal_labels"]["invalid"]
    )
    return {
        "schema_version": "program_o_full_des_independent_custody_verdict_v1",
        "generated_at_utc": now_utc(),
        "status": status,
        "passed": passed,
        "stage": stage,
        "run_id": run_id,
        "retrieved_run_dir": str(run_dir),
        "remote_run_dir": str(remote_run_dir),
        "scientific_result": str(required["result"]),
        "scientific_result_sha256": sha256(required["result"]),
        "remote_checksums_sha256": sha256(required["remote_checksums"]),
        "failures": failures,
        "claim_boundary": {
            "full_des_h_pi_established": bool(passed and stage == "validation"),
            "h_obs_authorized": False,
            "learner_authorized": False,
            "paper2_confirmed": False,
            "paper3_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("development", "validation"), required=True)
    parser.add_argument("--execution-freeze", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verdict = validate(
        run_dir=args.run_dir,
        stage=str(args.stage),
        execution_freeze=args.execution_freeze.resolve(),
        contract=args.contract.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
