#!/usr/bin/env python3
"""Fail closed if the reserved Program Q confirmation namespace appears opened."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


RESERVED_LOW = 7_490_001
RESERVED_HIGH = 7_490_256
DECLARATION_ALLOWLIST = {
    "contracts/program_q_frozen_policy_replication_v1.json",
    "research/paper2_exhaustive_search/program_q_historical_recurrentppo_fallback_freeze_20260717.json",
    "research/paper2_exhaustive_search/program_q_power_preopen_attempts_20260717.json",
    "research/paper2_exhaustive_search/program_q_power_preopen_v5_verdict_20260718.json",
    "research/paper2_exhaustive_search/program_q_primary_candidate_independence_v1.json",
    "research/paper2_exhaustive_search/program_q_seed_custody_preopen_20260717.json",
    "scripts/audit_program_q_seed_custody.py",
}
NUMBER = re.compile(r"(?<!\d)(\d{7,9})(?!\d)")


def scan(root: Path) -> dict:
    suspicious = []
    declarations = []
    for base in ("contracts", "docs", "research", "results", "scripts"):
        directory = root / base
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix not in {".json", ".md", ".py", ".txt", ".log"}:
                continue
            relative = path.relative_to(root).as_posix()
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            hits = sorted(
                {
                    int(value)
                    for value in NUMBER.findall(text)
                    if RESERVED_LOW <= int(value) <= RESERVED_HIGH
                }
            )
            name_hit = any(
                RESERVED_LOW <= int(value) <= RESERVED_HIGH
                for value in re.findall(r"\d+", path.name)
            )
            if hits or name_hit:
                row = {"path": relative, "seeds": hits, "seed_in_filename": name_hit}
                if relative in DECLARATION_ALLOWLIST:
                    declarations.append(row)
                else:
                    suspicious.append(row)
    return {
        "schema_version": "program_q_seed_custody_audit_v1",
        "reserved": [RESERVED_LOW, RESERVED_HIGH],
        "declarations": declarations,
        "suspicious": suspicious,
        "pass": not suspicious,
        "status": "PROGRAM_Q_SEEDS_VIRGIN" if not suspicious else "STOP_PROGRAM_Q_SEED_COLLISION",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = scan(args.root)
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if not payload["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
