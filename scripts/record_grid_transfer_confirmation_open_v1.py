#!/usr/bin/env python3
"""Record that the confirmation block is OPEN, and stop the registry saying otherwise.

The preflight artifact is scoped `RESERVED_BLOCK_NOT_OPENED_NO_SIMULATION`, which was true when it
ran and is false now: workers are executing and slices are on disk. Leaving the registry at
`RESERVED_NOT_OPENED` while a block is being consumed is the same defect that made the previous
attempt's entry false -- a custody record that describes an intention instead of the world.

The preflight stays exactly as it is. It is a historical pre-opening receipt, not a status.

Contract: docs/ENMIENDA_CONFIRMACION_BLOQUE_LIMPIO_2026-08-05.md
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402

BLOCK_ID = "garrido_grid_transfer_v2_confirmation"
START, END = 8_200_001, 8_200_060
CONTEXTS = ("R1r", "R2r", "R1r+R2r", "R1r|esc", "R2r|esc", "R1r+R2r|esc")
MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")
#: 4,608 extended configurations per (context, seed). The 288-cell base grid is PROJECTED from the
#: extended surface -- it is the subgrid op3_rm = op5_rm = 0 -- so it costs no DES episodes.
EXT_CONFIGS, BASE_CONFIGS = 4_608, 288


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", type=Path, default=Path("research/seed_custody_registry.json"))
    ap.add_argument("--ext-cache", type=Path,
                    default=Path("results/surface_cache/garrido_transfer_confirmation_v2_ext"))
    ap.add_argument("--base-cache", type=Path,
                    default=Path("results/surface_cache/garrido_transfer_confirmation_v2_base"))
    ap.add_argument("--contract", type=Path,
                    default=Path("docs/ENMIENDA_CONFIRMACION_BLOQUE_LIMPIO_2026-08-05.md"))
    ap.add_argument("--output", type=Path,
                    default=Path("results/custody/garrido_grid_transfer_confirmation_v2_open.json"))
    args = ap.parse_args()

    ext = sorted(args.ext_cache.rglob("*.json"))
    base = sorted(args.base_cache.rglob("*.json"))
    seeds_seen = sorted({int(json.loads(p.read_text())["seed"]) for p in ext})
    n_seeds = END - START + 1

    registry = json.loads(args.registry.read_text())
    block = next((b for b in registry["blocks"] if b["id"] == BLOCK_ID), None)
    if block is None:
        raise SystemExit(f"missing block {BLOCK_ID}")
    previous = block["status"]
    block["status"] = "OPEN_IN_PROGRESS"
    block["opened_at"] = datetime.now(timezone.utc).isoformat()
    block["opening_receipt"] = str(args.output)
    args.registry.write_text(json.dumps(registry, indent=1))   # NOT sort_keys: start/end adjacent

    payload = {
        "schema_version": "grid_transfer_confirmation_open_v1",
        "claim_status": "CONFIRMATION_BLOCK_OPEN_NO_RESULT_YET",
        "scope": "CUSTODY_STATE_ONLY_NO_SCIENTIFIC_CLAIM",
        "run_role": "CUSTODY_RECEIPT",
        "module_manifest": module_manifest(MODULES, script=__file__),
        "block": {"id": BLOCK_ID, "start": START, "end": END, "n_seeds": n_seeds},
        "registry_status": {"previous": previous, "now": "OPEN_IN_PROGRESS"},
        "execution": {
            "pool": "local M1 Pro, 10 cores",
            "workers": len(CONTEXTS),
            "sharding": "one worker per context, seeds shared, disjoint output paths",
            "command": ("scripts/build_transfer_confirmation_cache_v1.py --seed-start 8200001 "
                        "--seeds 60 --context <ctx> --contract "
                        "docs/ENMIENDA_CONFIRMACION_BLOQUE_LIMPIO_2026-08-05.md"),
            "routing_rationale": (
                "Kaggle is excluded: a 9-hour session cap plus session death yields a partial "
                "confirmation, which this contract refuses to rescue. The VPS is excluded for "
                "THIS run: it is ~4x slower per episode and its checkout is missing "
                "supply_chain/service_first_metric.py -- the same absence that put the H3-prime "
                "VPS slice into HOLD_SOURCE_AUDIT with its source identity unprovable."),
        },
        "cost": {
            "des_episodes": EXT_CONFIGS * len(CONTEXTS) * n_seeds,
            "stored_cells": (EXT_CONFIGS + BASE_CONFIGS) * len(CONTEXTS) * n_seeds,
            "note": ("The 288-cell base grid is PROJECTED from the extended surface, not "
                     "re-simulated: it is the subgrid op3_rm = op5_rm = 0. Reporting stored cells "
                     "as episodes overstates the run by 103,680."),
        },
        "progress_at_receipt": {
            "extended_slices": len(ext), "base_slices": len(base),
            "seeds_with_a_complete_slice": seeds_seen,
            "caveat": ("Only complete slices are written, so this counts what was WRITTEN, not "
                       "what was consumed. Any seed the workers touched is spent whether or not "
                       "it reached disk."),
        },
        "preflight_note": ("results/custody/garrido_grid_transfer_confirmation_v2_preflight.json "
                           "keeps its RESERVED_BLOCK_NOT_OPENED_NO_SIMULATION scope. It is a "
                           "pre-opening receipt and was true when it ran; it is not a status."),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/grid_transfer_ordered_v1/result.json"))
    print(f"  {BLOCK_ID}: {previous} -> OPEN_IN_PROGRESS")
    print(f"  episodios DES: {payload['cost']['des_episodes']:,} "
          f"(celdas almacenadas: {payload['cost']['stored_cells']:,})")
    print(f"  rebanadas escritas: {len(ext)} ext / {len(base)} base · semillas completas "
          f"{len(seeds_seen)}/{n_seeds}")
    print(f"  -> {args.output} (sello {digest[:16]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
