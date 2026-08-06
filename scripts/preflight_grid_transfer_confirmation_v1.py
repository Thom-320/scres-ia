#!/usr/bin/env python3
"""Preflight the reserved virgin block before the grid-transfer confirmation opens."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import (  # noqa: E402
    NO_KNOWN_COLLISION,
    check_seeds,
    module_manifest,
    seeds_used_by_sealed_artifacts,
)

DEFAULT_CONTRACT = Path("docs/PREREGISTRO_CONFIRMACION_TRANSFERENCIA_REJILLA_2026-08-05.md")
DEFAULT_REGISTRY = Path("research/seed_custody_registry.json")
DEFAULT_REFERENCE = Path("results/grid_transfer_v2/result.json")
DEFAULT_OUTPUT = Path("results/custody/garrido_grid_transfer_confirmation_preflight.json")
BLOCK_ID = "garrido_grid_transfer_v1_confirmation"
START, END = 8_100_001, 8_100_060


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.contract.is_file():
        raise SystemExit(f"missing contract: {args.contract}")
    if not args.reference.is_file():
        raise SystemExit(f"missing development reference: {args.reference}")
    registry = json.loads(args.registry.read_text())
    block = next((row for row in registry.get("blocks", []) if row.get("id") == BLOCK_ID), None)
    if block is None:
        raise SystemExit(f"missing reservation block: {BLOCK_ID}")
    if (int(block["start"]), int(block["end"])) != (START, END):
        raise SystemExit("reservation range does not match the contract")
    if block.get("status") != "RESERVED_NOT_OPENED":
        raise SystemExit(f"reservation is not unopened: {block.get('status')!r}")
    if block.get("source") != str(args.contract):
        raise SystemExit("reservation source does not point to the contract")

    seeds = list(range(START, END + 1))
    custody = check_seeds(seeds, registry_path=args.registry, results_root=Path("results"))
    used = seeds_used_by_sealed_artifacts(Path("results"), exclude=args.output)
    fresh_artifact_overlap = sorted(set(seeds) & used)
    filename_hits = sorted(
        str(path) for path in Path("results").rglob("*")
        if path.is_file() and any(path.name == f"{seed}.json" for seed in seeds)
    )
    checks = {
        "f_contract_exists": args.contract.is_file(),
        "f_reference_exists": args.reference.is_file(),
        "f_registry_reserves_exact_block": True,
        "f_no_known_seed_collision": custody["status"] == NO_KNOWN_COLLISION,
        "f_no_sealed_artifact_overlap": not fresh_artifact_overlap,
        "f_no_seed_named_cache_slice": not filename_hits,
    }
    if not all(checks.values()):
        raise SystemExit(json.dumps({"checks": checks, "custody": custody}, indent=1))

    payload = {
        "schema_version": "garrido_grid_transfer_confirmation_preflight_v1",
        "claim_status": "PREOPEN_CONFIRMATION_PREFLIGHT_PASS",
        "scope": "RESERVED_BLOCK_NOT_OPENED_NO_SIMULATION",
        "contract": str(args.contract),
        "contract_sha256_local": sha256(args.contract),
        "reference": str(args.reference),
        "reserved_seed_block": {"id": BLOCK_ID, "start": START, "end": END, "n": len(seeds)},
        "checks": checks,
        "custody": custody,
        "fresh_artifact_overlap": fresh_artifact_overlap,
        "seed_named_cache_slice_hits": filename_hits,
        "module_manifest": module_manifest(
            ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py"), script=__file__
        ),
    }
    digest = seal_and_write(payload, args.output, contract=args.contract, reference=args.reference)
    print(f"  preflight PASS · block {START}-{END} · sello {digest[:16]}…")
    print(f"  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
