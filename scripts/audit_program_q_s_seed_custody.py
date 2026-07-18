#!/usr/bin/env python3
"""Unified numeric-range seed custody audit for Program Q and Program S."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "research/paper2_exhaustive_search/program_q_s_seed_registry_v1_1.json"
NUMBER = re.compile(r"(?<!\d)(\d{7,9})(?!\d)")
TEXT_SUFFIXES = {".json", ".md", ".py", ".txt", ".log", ".yaml", ".yml", ".sh"}


SEED_CONTEXT = re.compile(
    r"seed|tape|reserved|unopened|opened|block|range|training|calibration|confirmation|morris",
    re.IGNORECASE,
)


def _range_hits(text: str, ranges: list[dict], *, require_context: bool) -> dict[str, list[int]]:
    values = set()
    for match in NUMBER.finditer(text):
        if require_context:
            context = text[max(0, match.start() - 120): min(len(text), match.end() + 120)]
            if SEED_CONTEXT.search(context) is None:
                continue
        values.add(int(match.group(1)))
    return {
        row["id"]: sorted(value for value in values if int(row["low"]) <= value <= int(row["high"]))
        for row in ranges
        if any(int(row["low"]) <= value <= int(row["high"]) for value in values)
    }


def scan(root: Path = ROOT) -> dict:
    registry = json.loads(REGISTRY_PATH.read_text())
    ranges = [
        row for row in registry["ranges"]
        if row["status"] in {"RESERVED_UNOPENED", "SEALED_UNAUTHORIZED"}
    ]
    allowlist = set(registry["declaration_allowlist"])
    declarations: list[dict] = []
    suspicious: list[dict] = []
    for base in ("contracts", "docs", "research", "results", "scripts", "tests"):
        directory = root / base
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            name_hits = _range_hits(path.name, ranges, require_context=False)
            content_hits = {}
            if path.suffix.lower() in TEXT_SUFFIXES:
                content_hits = _range_hits(
                    path.read_text(errors="ignore"), ranges, require_context=True
                )
            hits = {
                key: sorted(set(name_hits.get(key, ())) | set(content_hits.get(key, ())))
                for key in set(name_hits) | set(content_hits)
            }
            if not hits:
                continue
            row = {"path": relative, "range_hits": hits}
            (declarations if relative in allowlist else suspicious).append(row)
    return {
        "schema_version": "program_q_s_seed_custody_audit_v1_2",
        "registry": str(REGISTRY_PATH.relative_to(root)),
        "numeric_interval_semantics": True,
        "declarations": declarations,
        "suspicious": suspicious,
        "pass": not suspicious,
        "opened_ranges_excluded_from_virginity_scan": [
            row["id"] for row in registry["ranges"] if row["status"].startswith("BURNED_")
        ],
        "verdict": "PASS_PROGRAM_S_RESERVED_SEED_CUSTODY_POST_Q" if not suspicious else "STOP_PROGRAM_S_SEED_COLLISION"
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = scan(args.root)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
