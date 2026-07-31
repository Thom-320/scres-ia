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
    "research/paper2_exhaustive_search/program_q_s_seed_registry_v1.json",
    "research/paper2_exhaustive_search/program_q_s_seed_custody_preopen_v1_1.json",
    "scripts/audit_program_q_seed_custody.py",
}
NUMBER = re.compile(r"(?<!\d)(\d{7,9})(?!\d)")

# CORRECTED 2026-07-31. The scan classified by PATH alone, so any file outside the allowlist
# that so much as NAMED the reserved range was a collision. It fired on
# `scripts/build_david_sandbox_notebook.py:866`, prose telling the reader what he may not
# touch: "las semillas reservadas 7480101-7480148, `7490001-7490256` y 950100001-950100096".
# Declaring a reservation is the opposite of opening it, and a guard that reddens on correct
# text trains the reader to ignore it -- a real collision would then hide behind the same
# failure. The seeds are virgin; the status was not.
#
# A hit is now judged by CONTEXT, not by which file it sits in:
#
# * a **consumption cue** -- `seed`/`root`/`tape`/`semilla` immediately before the number, or
#   the seed in the filename -- is always suspicious, in any file, allowlisted or not. This is
#   how an opened seed actually looks: `{"seed": 7490001}`, `seed=7490001`, `..._7490001.npz`.
# * a **bounds mention** -- the value is exactly an endpoint of the reserved namespace AND the
#   other endpoint appears beside it -- is a declaration. An INTERIOR seed (7490137, say) is
#   never declarable this way, so the narrow reading cannot be widened into a loophole.
#
# The path allowlist survives with its original meaning -- these files ARE the custody record,
# so a seed mentioned in their text is a declaration by construction. Measured, all nine of
# them declare bounds only; the sole consumption-shaped hits in the whole repository are this
# script's own constants. What the allowlist never excuses is a reserved seed in a FILENAME:
# an opened seed materializes as an artifact named after it, and that stays suspicious
# everywhere, allowlisted or not.
CONSUMPTION_CUE = re.compile(r"(?:seed|root|tape|semilla)[\w-]{0,12}\W{0,4}$", re.IGNORECASE)
CUE_WINDOW = 32
BOUNDS_WINDOW = 48
# Digit runs may carry `_`, and the two uses are NOT the same: `7_490_001` is one Python
# literal, while `"7490001_7490256"` is a range key holding two. Stripping `_` globally merges
# the second into a 14-digit run that no 7-9 digit pattern matches, so the custody scan would
# silently MISS a declared range -- caught while writing this. Group-of-three separators are
# removed; any other `_` splits the token.
DIGIT_TOKEN = re.compile(r"(?<![\dA-Za-z])[\d][\d_]*(?<![_])")
PYTHON_GROUPED = re.compile(r"\d{1,3}(?:_\d{3})+$")


def _candidates(text: str) -> list[tuple[int, int, int]]:
    """Every integer literal in `text`, as `(value, start, end)` in ORIGINAL coordinates."""
    out: list[tuple[int, int, int]] = []
    for token in DIGIT_TOKEN.finditer(text):
        raw = token.group(0)
        if PYTHON_GROUPED.fullmatch(raw):
            out.append((int(raw.replace("_", "")), token.start(), token.end()))
            continue
        offset = token.start()
        for part in raw.split("_"):
            if part:
                out.append((int(part), offset, offset + len(part)))
            offset += len(part) + 1
    return out


def _classify_hit(text: str, start: int, end: int, value: int) -> str:
    """`consumption` when a seed is being used, `bounds` when the namespace is named."""
    if CONSUMPTION_CUE.search(text[max(0, start - CUE_WINDOW):start]):
        return "consumption"
    if value not in (RESERVED_LOW, RESERVED_HIGH):
        return "consumption"
    partner = str(RESERVED_HIGH if value == RESERVED_LOW else RESERVED_LOW)
    near = text[max(0, start - BOUNDS_WINDOW):min(len(text), end + BOUNDS_WINDOW)]
    return "bounds" if partner in near.replace("_", "") else "consumption"


def classify_text(text: str) -> dict[str, list[int]]:
    """Split one file's reserved-namespace hits into declared bounds and used seeds."""
    out: dict[str, set[int]] = {"bounds": set(), "consumption": set()}
    for value, start, end in _candidates(text):
        if not RESERVED_LOW <= value <= RESERVED_HIGH:
            continue
        out[_classify_hit(text, start, end, value)].add(value)
    return {key: sorted(values) for key, values in out.items()}


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
            found = classify_text(text)
            name_hit = any(
                RESERVED_LOW <= int(value) <= RESERVED_HIGH
                for value in re.findall(r"\d+", path.name.replace("_", ""))
            )
            used = found["consumption"]
            declared = found["bounds"]
            if not (used or declared or name_hit):
                continue
            row = {"path": relative, "seeds": sorted(set(used) | set(declared)),
                   "seeds_used": used, "seeds_declared_as_bounds": declared,
                   "seed_in_filename": name_hit}
            # A filename carrying a reserved seed is a consumption in every file; the path
            # allowlist covers only the custody artifacts' textual declarations.
            if name_hit or (used and relative not in DECLARATION_ALLOWLIST):
                suspicious.append(row)
            else:
                declarations.append(row)
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
