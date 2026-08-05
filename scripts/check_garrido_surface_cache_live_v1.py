#!/usr/bin/env python3
"""Recompute eight cached cells against the live DES and seal the comparison.

This is an external cache-vs-simulator check.  It uses only the already-consumed 5.3-million seed
block and compares the complete observable panel plus the primary scalar.  It does not open a new
seed and it does not train a learner.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_meta_learner_normaliser_audit_v1 import (  # noqa: E402
    CONFIGS,
    evaluate,
)
from scripts.seal_garrido_surface_cache_v1 import (  # noqa: E402
    observable_key,
    verify_sealed_slice,
)
from supply_chain.arm_runner import seal_and_write  # noqa: E402

SAMPLES = (
    ("R1r", 5_300_001, 0),
    ("R1r", 5_300_007, 137),
    ("R2r", 5_300_002, 42),
    ("R2r", 5_300_012, 281),
    ("R1r+R2r", 5_300_003, 73),
    ("R1r|esc", 5_300_004, 211),
    ("R2r|esc", 5_300_005, 19),
    ("R1r+R2r|esc", 5_300_006, 166),
)


def compare_cell(cache_root: Path, context: str, seed: int, index: int) -> dict[str, Any]:
    path = cache_root / context.replace("|", "_").replace("+", "_") / f"{seed}.json"
    payload = json.loads(path.read_text())
    verify_sealed_slice(payload)
    cached = payload["cells"][index]
    live = evaluate(CONFIGS[index], context, seed, float(payload["horizon_hours"]))
    cached_key = observable_key(cached)
    live_key = observable_key(live)
    return {
        "path": str(path),
        "context": context,
        "seed": seed,
        "config_index": index,
        "observable_key_identical": cached_key == live_key,
        "panel_identical": cached["panel"] == live["panel"],
        "value_identical": float(cached["value"]) == float(live["value"]),
        "cached_observable_key": cached_key,
        "live_observable_key": live_key,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("results/surface_cache/wrap288_v1"))
    parser.add_argument("--contract", type=Path,
                        default=Path("docs/PREREGISTRO_AUDITORIA_NORMALIZADOR_2026-08-05.md"))
    parser.add_argument("--reference", type=Path,
                        default=Path("results/garrido_normaliser_audit/result.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/surface_cache_custody/wrap288_v1/live_check.json"))
    args = parser.parse_args()
    checks = [compare_cell(args.cache, *sample) for sample in SAMPLES]
    passed = all(
        check["observable_key_identical"] and check["panel_identical"] and check["value_identical"]
        for check in checks
    )
    payload = {
        "schema_version": "garrido_surface_cache_live_check_v1",
        "claim_status": "CACHE_REPRODUCES_LIVE_DES" if passed else "CACHE_DERIVA_FROM_LIVE_DES",
        "scope": "BURNED_REPLAY_EIGHT_CELL_CHECK_NO_NEW_SEEDS_NO_LEARNER",
        "run_role": "CACHE_CUSTODY_AUDIT",
        "cache": str(args.cache),
        "samples": list(SAMPLES),
        "checks": checks,
        "f_cache_reproduces_the_simulator": {
            "passed": passed,
            "evidence": {
                "why_it_can_fail": "a silent physics, metric, CRN, or interpreter drift changes the panel or endpoint",
                "n_cells": len(checks),
                "comparison": "observable_key, complete panel, and primary scalar",
            },
        },
        "f_no_fresh_seeds": {
            "passed": None,
            "not_applicable": True,
            "evidence": {
                "status": "DECLARED_REPLAY",
                "seeds": sorted({seed for _, seed, _ in SAMPLES}),
            },
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["falsifiers"] = {
        "all_passed": passed,
        "f_cache_reproduces_the_simulator": payload["f_cache_reproduces_the_simulator"],
        "f_no_fresh_seeds": payload["f_no_fresh_seeds"],
    }
    seal_and_write(payload, args.output, contract=args.contract, reference=args.reference)
    print(f"live cache check: {'PASS' if passed else 'FAIL'} ({len(checks)} cells)")
    print(f"result: {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
