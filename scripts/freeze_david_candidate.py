#!/usr/bin/env python3
"""Freeze one David sandbox finalist before blind qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--model-kind", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = json.loads(args.report.read_text())
    if report.get("status") != "SANDBOX_DEVELOPMENT_ONLY_NOT_PROMOTABLE":
        raise RuntimeError("candidate source must be a sandbox report")
    if args.model_kind not in report.get("models", []):
        raise RuntimeError("model kind is absent from the report")
    architecture = report.get("architecture", {}).get(args.model_kind)
    if not architecture or not architecture.get("source_sha256"):
        raise RuntimeError("candidate architecture audit is missing")
    payload = {
        "schema_version": "program_q_candidate_freeze_v1",
        "status": "FROZEN_BEFORE_950_BLIND_QUALIFICATION",
        "model_kind": args.model_kind,
        "notebook": str(args.notebook),
        "notebook_sha256": sha256(args.notebook),
        "development_report": str(args.report),
        "development_report_sha256": sha256(args.report),
        "architecture": architecture,
        "preset": report["preset"],
        "timesteps_per_seed": report["total_timesteps_per_seed"],
        "optimizer_seeds": report["optimizer_seeds"],
        "history_length": report["history_length"],
        "blind_qualification": [950100001, 950100096],
        "blind_qualification_opened": False,
        "post_freeze_changes_invalidate_candidate": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
