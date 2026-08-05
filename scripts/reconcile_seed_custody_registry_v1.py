#!/usr/bin/env python3
"""Reconcile the seed-custody registry against every sealed artifact that consumed seeds.

The registry declares itself `BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED`, and it is:
it lists ten blocks while a scan of `results/` finds dozens of consumed ranges. That matters
because `custody_falsifier` reports `NO_KNOWN_COLLISION` -- a verdict whose strength depends
entirely on how much the registry knows. An incomplete registry does not produce wrong answers,
it produces weak ones, and a reproducibility audit would find the gap before we did.

This is ADDITIVE and conservative:
  * existing entries are never modified or removed;
  * every newly discovered range is registered as USED_DEVELOPMENT_NOT_VIRGIN, because anything a
    sealed artifact already consumed cannot be virgin;
  * every entry records the artifacts that consumed it, so the claim is checkable.

OPTIMIZER SEEDS ARE NOT TAPE SEEDS. `results/program_e/ppo/training_verdict.json` lists 9301-9310;
those index a torch initialisation, not a CRN tape, and registering them as tape blocks would
manufacture collisions that do not exist. Only values at or above `TAPE_FLOOR` are treated as
tapes, and everything below is reported separately rather than silently dropped.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REGISTRY = Path("research/seed_custody_registry.json")
#: Every CRN tape block this project has ever opened is in the millions; optimizer seeds,
#: replicate indices and fold numbers are not.
TAPE_FLOOR = 1_000_000
MAX_GAP = 1                     # ranges are contiguous; a gap of more than one starts a new block


def scan(results_root: Path) -> tuple[dict[int, set[str]], dict[int, set[str]]]:
    tapes: dict[int, set[str]] = {}
    non_tapes: dict[int, set[str]] = {}
    for path in sorted(results_root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        seeds = payload.get("seeds")
        if not (isinstance(seeds, list) and seeds and all(isinstance(s, int) for s in seeds)):
            continue
        for seed in seeds:
            bucket = tapes if seed >= TAPE_FLOOR else non_tapes
            bucket.setdefault(seed, set()).add(str(path))
    return tapes, non_tapes


def to_blocks(seeds: dict[int, set[str]]) -> list[dict]:
    blocks, current = [], None
    for seed in sorted(seeds):
        if current and seed - current["end"] <= MAX_GAP:
            current["end"] = seed
            current["sources"] |= seeds[seed]
        else:
            if current:
                blocks.append(current)
            current = {"start": seed, "end": seed, "sources": set(seeds[seed])}
    if current:
        blocks.append(current)
    return blocks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--registry", type=Path, default=REGISTRY)
    ap.add_argument("--receipt", type=Path,
                    default=Path("results/custody/registry_reconciliation.json"))
    ap.add_argument("--apply", action="store_true",
                    help="write the registry; without it the run is a dry report")
    args = ap.parse_args()

    registry = json.loads(args.registry.read_text())
    known: set[int] = set()
    for block in registry["blocks"]:
        known |= set(range(int(block["start"]), int(block["end"]) + 1))

    tapes, non_tapes = scan(args.results)
    unknown = {s: src for s, src in tapes.items() if s not in known}
    new_blocks = to_blocks(unknown)

    added = []
    for block in new_blocks:
        sources = sorted(block["sources"])
        added.append({
            # start and end adjacent, for the same reason the registry is not key-sorted.
            "id": f"reconciled_{block['start']}",
            "start": int(block["start"]), "end": int(block["end"]),
            "status": "USED_DEVELOPMENT_NOT_VIRGIN",
            "source": sources[0],
            "all_sources": sources,
            "purpose": ("Discovered by reconciling the registry against sealed artifacts. It was "
                        "consumed but never registered, so a replay could not be declared against "
                        "it by name and its absence weakened every NO_KNOWN_COLLISION verdict."),
            "registered_by": "scripts/reconcile_seed_custody_registry_v1.py",
        })

    print(f"  bloques ya registrados: {len(registry['blocks'])}")
    print(f"  semillas de cinta vistas en artefactos: {len(tapes)}")
    print(f"  no registradas: {len(unknown)} en {len(added)} bloques contiguos")
    for block in added:
        print(f"    {block['start']}-{block['end']:<10} ({len(block['all_sources'])} artefactos)"
              f"  {Path(block['source']).parent.name}")
    if non_tapes:
        lo, hi = min(non_tapes), max(non_tapes)
        print(f"\n  ignoradas por no ser cintas (< {TAPE_FLOOR:,}): {len(non_tapes)} valores "
              f"en [{lo}, {hi}] — semillas de optimizador o índices de réplica")

    receipt = {
        "schema_version": "seed_custody_reconciliation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "applied": bool(args.apply),
        "registry": str(args.registry),
        "blocks_before": len(registry["blocks"]),
        "blocks_added": len(added),
        "tape_floor": TAPE_FLOOR,
        "added": added,
        "non_tape_values_ignored": sorted(non_tapes),
        "rule": ("additive only: existing entries are never modified; discovered ranges are "
                 "registered as USED_DEVELOPMENT_NOT_VIRGIN because a sealed artifact consumed "
                 "them"),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=1, sort_keys=True))
    print(f"\n  -> recibo {args.receipt}")

    if args.apply and added:
        registry["blocks"] = registry["blocks"] + added
        registry["reconciled_at"] = receipt["created_at"]
        registry["reconciliation_receipt"] = str(args.receipt)
        # NOT sort_keys: the Program Q custody auditor classifies a reserved-range endpoint as a
        # DECLARATION only when its partner endpoint is within a short window of text. Sorting the
        # keys puts `end` first and `start` last, separated by the long `purpose` string, so a
        # declared range starts reading as a consumption. Reformatting a file you are only
        # appending to is how a cosmetic write becomes a false collision.
        args.registry.write_text(json.dumps(registry, indent=1))
        print(f"  -> registro actualizado con {len(added)} bloques")
    elif not args.apply:
        print("  (dry run: usa --apply para escribir)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
