#!/usr/bin/env python3
"""Resolve historical `code_ref: HEAD` values in existing artifact bundles.

This script infers the most likely git commit for a run directory from the
bundle timestamp and replaces any JSON field named `code_ref` whose value is
`HEAD`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent
DEFAULT_RUN_DIR = REPO / "outputs" / "track_b_benchmarks" / "track_b_ret_seq_k020_500k_rerun1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pin historical code_ref fields inside a run bundle.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--timestamp", type=str, default=None, help="Optional explicit UTC timestamp.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def infer_timestamp(run_dir: Path) -> str:
    status_path = run_dir / "status.json"
    if status_path.exists():
        status = load_json(status_path)
        started = status.get("started_at_utc")
        if started:
            return str(started)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        generated = summary.get("benchmark_metadata", {}).get("generated_at_utc")
        if generated:
            return str(generated)
    raise ValueError(f"Could not infer timestamp from {run_dir}")


def resolve_commit(timestamp: str) -> str:
    commit = (
        subprocess.check_output(
            ["git", "rev-list", "-n", "1", f"--before={timestamp}", "HEAD"],
            cwd=REPO,
        )
        .decode()
        .strip()
    )
    if not commit:
        raise ValueError(f"No commit found before {timestamp}")
    return commit


def replace_code_ref(payload: Any, commit: str) -> int:
    replacements = 0
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "code_ref" and value == "HEAD":
                payload[key] = commit
                replacements += 1
            else:
                replacements += replace_code_ref(value, commit)
    elif isinstance(payload, list):
        for item in payload:
            replacements += replace_code_ref(item, commit)
    return replacements


def main() -> None:
    args = build_parser().parse_args()
    timestamp = args.timestamp or infer_timestamp(args.run_dir)
    commit = resolve_commit(timestamp)

    patched = []
    for path in sorted(args.run_dir.rglob("*.json")):
        payload = load_json(path)
        replacements = replace_code_ref(payload, commit)
        if replacements:
            patched.append((path, replacements))
            if not args.dry_run:
                write_json(path, payload)

    print(f"Timestamp: {timestamp}")
    print(f"Resolved commit: {commit}")
    if not patched:
        print("No code_ref=HEAD fields found.")
        return
    for path, replacements in patched:
        display = path.resolve().relative_to(REPO)
        print(f"{display}: patched {replacements} field(s)")


if __name__ == "__main__":
    main()
