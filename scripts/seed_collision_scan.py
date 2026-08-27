#!/usr/bin/env python3
"""Repository-wide seed collision scan.

The ACTA of 2026-08-25 records that a scan of this kind was run before Gate-0
opened its block, but no script was ever tracked, so the scan could not be
reproduced or audited. This is that instrument.

It reads every tracked file (``git ls-files``), harvests every 6-to-9 digit
integer that could be a simulation seed, and reports whether a candidate block
is free. It answers one question and refuses to guess: a seed is USED if it
appears anywhere in the tracked tree, because a seed mentioned in a result, a
manifest or a contract has been reasoned about and is no longer virgin.

Usage
-----
    python scripts/seed_collision_scan.py --block 9700001 9700048
    python scripts/seed_collision_scan.py --propose 48
    python scripts/seed_collision_scan.py --block 9700001 9700048 --json out.json

Exit code is 0 when the block is free, 1 when it collides.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# A seed in this repository is a 6-to-9 digit integer. Bounding the range keeps
# the scan from harvesting timestamps, byte counts and SHA fragments.
SEED_RE = re.compile(r"(?<![0-9.])([1-9][0-9]{5,8})(?![0-9.])")
SEED_MIN, SEED_MAX = 100_000, 999_999_999

# Extensions worth scanning. Binary artefacts (.npz, .pdf, .png) are skipped:
# a seed that only exists inside a binary was still declared somewhere in text.
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".py", ".txt", ".yaml", ".yml",
                 ".toml", ".cfg", ".sh", ".csv"}


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [ROOT / line for line in out.stdout.splitlines() if line]


def harvest(paths: list[Path]) -> tuple[dict[int, list[str]], int]:
    """seed -> the files that mention it, plus a count of files actually read."""
    used: dict[int, list[str]] = defaultdict(list)
    read = 0
    for path in paths:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        read += 1
        rel = str(path.relative_to(ROOT))
        for match in SEED_RE.finditer(text):
            value = int(match.group(1))
            if SEED_MIN <= value <= SEED_MAX:
                if len(used[value]) < 6:  # keep the report bounded
                    used[value].append(rel)
    return used, read


def scan_block(used: dict[int, list[str]], low: int, high: int) -> dict:
    collisions = {
        seed: used[seed] for seed in range(low, high + 1) if seed in used
    }
    return {
        "block": [low, high],
        "size": high - low + 1,
        "n_collisions": len(collisions),
        "free": not collisions,
        "collisions": {str(k): v for k, v in sorted(collisions.items())[:40]},
    }


def propose(used: dict[int, list[str]], size: int, start: int, stop: int,
            stride: int) -> list[int] | None:
    """First aligned block of `size` free seeds, searching upward from `start`."""
    for low in range(start, stop, stride):
        high = low + size - 1
        if not any(seed in used for seed in range(low, high + 1)):
            return [low, high]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block", type=int, nargs=2, metavar=("LOW", "HIGH"),
                    help="check this inclusive block for collisions")
    ap.add_argument("--propose", type=int, metavar="N",
                    help="propose the first free aligned block of N seeds")
    ap.add_argument("--search-from", type=int, default=9_700_001)
    ap.add_argument("--search-to", type=int, default=9_999_001)
    ap.add_argument("--stride", type=int, default=1000)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    if not args.block and not args.propose:
        ap.error("give --block or --propose")

    files = tracked_files()
    used, read = harvest(files)
    report: dict = {
        "schema_version": "seed_collision_scan_v1",
        "tracked_files": len(files),
        "files_read": read,
        "distinct_seed_like_integers": len(used),
    }
    print(f"tracked files: {len(files)}  read: {read}  "
          f"distinct seed-like integers: {len(used)}")

    ok = True
    if args.block:
        low, high = args.block
        result = scan_block(used, low, high)
        report["block_scan"] = result
        if result["free"]:
            print(f"BLOCK {low}-{high}: FREE ({result['size']} seeds, 0 collisions)")
        else:
            ok = False
            print(f"BLOCK {low}-{high}: {result['n_collisions']} COLLISIONS")
            for seed, where in list(result["collisions"].items())[:10]:
                print(f"  {seed} <- {where[0]}")

    if args.propose:
        found = propose(used, args.propose, args.search_from, args.search_to,
                        args.stride)
        report["proposed"] = found
        if found:
            print(f"PROPOSED free block of {args.propose}: {found[0]}-{found[1]}")
        else:
            ok = False
            print(f"no free aligned block of {args.propose} in "
                  f"[{args.search_from}, {args.search_to}]")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"report: {args.json}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
