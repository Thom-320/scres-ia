#!/usr/bin/env python3
"""Scan a candidate Q-R1 history-root block before registering it."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def scan(commit: str, low: int, high: int) -> dict:
    collisions = []
    for value in range(low, high + 1):
        pattern = rf"(?<![0-9]){value}(?![0-9])"
        process = subprocess.run(
            ["git", "grep", "-n", "-P", pattern, commit],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if process.returncode not in (0, 1):
            raise RuntimeError(process.stderr.strip() or "git grep failed")
        for line in process.stdout.splitlines():
            # Decimal substrings such as 0.337572001 are not integer seed
            # tokens and are rejected by the same numeric-boundary regex.
            if re.search(pattern, line):
                collisions.append({"value": value, "hit": line})
    return {
        "schema_version": "q_r1_successor_seed_collision_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit_scanned": commit,
        "range_scanned_inclusively": [low, high],
        "values_scanned": high - low + 1,
        "scan_method": "git grep PCRE exact numeric boundaries against immutable parent commit for every integer",
        "collisions": collisions,
        "passed": not collisions,
        "roots_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--low", type=int, required=True)
    parser.add_argument("--high", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    payload = scan(args.commit, args.low, args.high)
    if not payload["passed"]:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
