#!/usr/bin/env python3
"""Quarantine the partially opened grid-transfer block without promoting a result."""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supply_chain.arm_runner import seal_and_write  # noqa: E402

BLOCK_ID = "garrido_grid_transfer_v1_confirmation"
START, END = 8_100_001, 8_100_060
CONTRACT = Path("docs/PREREGISTRO_CONFIRMACION_TRANSFERENCIA_REJILLA_2026-08-05.md")
REGISTRY = Path("research/seed_custody_registry.json")
REFERENCE = Path("results/custody/garrido_grid_transfer_confirmation_preflight.json")
OUTPUT = Path("results/custody/garrido_grid_transfer_confirmation_abort.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text())
    block = next((b for b in registry["blocks"] if b.get("id") == BLOCK_ID), None)
    if block is None:
        raise SystemExit(f"missing block {BLOCK_ID}")
    if block.get("status") != "RESERVED_NOT_OPENED":
        raise SystemExit(f"unexpected block status: {block.get('status')!r}")

    ext_root = Path("results/surface_cache/garrido_transfer_confirmation_ext_v1")
    base_root = Path("results/surface_cache/garrido_transfer_confirmation_base_v1")
    ext_paths = sorted(str(p) for p in ext_root.glob("*/*.json"))
    base_paths = sorted(str(p) for p in base_root.glob("*/*.json"))
    complete_seeds = sorted({int(Path(p).stem) for p in ext_paths})
    complete_contexts = sorted({Path(p).parent.name for p in ext_paths})
    now = datetime.now(timezone.utc).isoformat()

    block["status"] = "ATTEMPTED_NO_SEALED_ARTIFACT"
    block["authorization"] = {
        "granted_by": "UNRECORDED",
        "note": "No explicit PI authorization for opening this virgin block was recorded. The phrase 'freedom as PI' was not a seed-opening receipt.",
    }
    block["attempt_audit"] = {
        "status": "ABORTED_AFTER_PARTIAL_RAW_CACHE",
        "stopped_at": now,
        "termination": "explicit SIGKILL of the named workers after custody/order audit",
        "complete_extended_slices_on_disk": len(ext_paths),
        "complete_base_slices_on_disk": len(base_paths),
        "complete_slice_seed_names_seen": complete_seeds,
        "complete_slice_context_names_seen": complete_contexts,
        "raw_slices_are_unsealed": True,
        "result_artifact_present": False,
        "why_entire_block_is_quarantined": (
            "workers were killed without an execution receipt for every episode; a missing raw "
            "slice is not evidence that its seed was untouched, so the reserved block cannot be "
            "reused as virgin evidence"
        ),
    }
    block["purpose"] = (
        "Aborted unauthorized/invalid confirmation attempt. Entire range is quarantined; no "
        "result from this attempt is confirmatory and the range is not reusable as virgin evidence."
    )
    args.registry.write_text(json.dumps(registry, indent=1) + "\n")

    payload = {
        "schema_version": "garrido_grid_transfer_confirmation_abort_v1",
        "claim_status": "CONFIRMATION_BLOCK_QUARANTINED_NO_SCIENTIFIC_RESULT",
        "scope": "PARTIAL_RAW_CACHE_NO_CONFIRMATION_NO_MANUSCRIPT_CLAIM",
        "block_id": BLOCK_ID,
        "attempted_seed_range": [START, END],
        "complete_extended_slices_on_disk": len(ext_paths),
        "complete_base_slices_on_disk": len(base_paths),
        "complete_slice_seed_names_seen": complete_seeds,
        "complete_slice_context_names_seen": complete_contexts,
        "raw_slice_paths": ext_paths + base_paths,
        "result_artifact_present": False,
        "registry_status_after_closeout": "ATTEMPTED_NO_SEALED_ARTIFACT",
        "stopped_at": now,
        "rule": "partial execution and missing logs cannot be converted into virginity or confirmation",
    }
    digest = seal_and_write(payload, args.output, contract=args.contract, reference=args.reference)
    print(f"  block quarantined · {len(ext_paths)} extended slices · receipt {digest[:16]}…")
    print(f"  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
