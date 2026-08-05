#!/usr/bin/env python3
"""Seal the already-computed Garrido surface cache without rerunning the DES.

The cache is a declared replay of burned seeds.  This utility adds an envelope seal and a
scientific-payload hash to each existing slice, records the runtime used to seal it, and emits a
separate manifest.  It deliberately does not change any cell value and it never opens a seed.

The cache's endpoint is the scalar ``ret_excel_risk_conditional`` used by the normaliser/search
lane.  ``observable_key`` is therefore the scalar plus the panel fields that the cache contract
requires; it is not the service-first lexicographic endpoint of the separate DES-288 confirmation
runner.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supply_chain.arm_runner import (  # noqa: E402
    canonical_payload_sha256,
    seal_and_write,
)
from supply_chain.seed_custody import module_manifest  # noqa: E402

SCHEMA = "garrido_surface_cache_v1"
PANEL_KEYS = (
    "delivered_rations",
    "demanded_rations",
    "flow_fill_rate",
    "lost_orders",
    "ret_excel",
    "ret_excel_full_ledger",
    "ret_excel_risk_conditional",
)
SEALER_MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")


def _runtime_manifest() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "kan"):
        try:
            from importlib.metadata import version

            versions[name] = version(name)
        except Exception:
            versions[name] = None
    return {
        "python": sys.version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


def _script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def observable_key(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Return the endpoint identity that can be compared against a live recomputation."""
    panel = cell.get("panel")
    if not isinstance(panel, Mapping):
        raise ValueError("cache cell has no panel")
    missing = [key for key in PANEL_KEYS if key not in panel]
    if missing:
        raise ValueError(f"cache cell panel is missing {missing}")
    return {
        "metric": "ret_excel_risk_conditional",
        "components": list(PANEL_KEYS),
        "values": [float(cell["value"])]
        + [float(panel[key]) for key in PANEL_KEYS],
    }


def validate_slice(payload: Mapping[str, Any], *, expected_cells: int = 288,
                   expected_grid_id: str | None = None) -> None:
    if payload.get("schema_version") not in (None, SCHEMA):
        raise ValueError(f"unsupported cache schema: {payload.get('schema_version')!r}")
    required = ("grid_id", "context", "seed", "horizon_hours", "metric", "cells")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"cache slice is missing {missing}")
    if expected_grid_id is not None and payload.get("grid_id") != expected_grid_id:
        raise ValueError(
            f"cache slice grid_id {payload.get('grid_id')!r} != {expected_grid_id!r}"
        )
    cells = payload["cells"]
    if not isinstance(cells, list) or len(cells) != expected_cells:
        raise ValueError(f"cache slice has {len(cells) if isinstance(cells, list) else 'non-list'} cells")
    if payload["metric"] != "ret_excel_risk_conditional":
        raise ValueError(f"unexpected cache metric: {payload['metric']!r}")
    for index, cell in enumerate(cells):
        if not isinstance(cell, Mapping) or "value" not in cell or "panel" not in cell:
            raise ValueError(f"cell {index} is not a full value/panel row")
        observable_key(cell)


def scientific_hash(payload: Mapping[str, Any]) -> str:
    return canonical_payload_sha256(
        payload,
        extra_exclude=frozenset({
            "cache_custody", "contract_sha256", "reference_sha256", "scientific_payload_sha256"
        }),
    )


def verify_sealed_slice(payload: Mapping[str, Any], *, expected_cells: int = 288,
                        expected_grid_id: str | None = None) -> None:
    """Fail closed if either the envelope or scientific payload seal is wrong."""
    validate_slice(payload, expected_cells=expected_cells, expected_grid_id=expected_grid_id)
    stored_science = payload.get("scientific_payload_sha256")
    if not stored_science or stored_science != scientific_hash(payload):
        raise ValueError("cache scientific_payload_sha256 is invalid")
    stored_self = payload.get("self_sha256")
    if not stored_self:
        raise ValueError("cache slice has no self_sha256")
    body = json.dumps({key: value for key, value in payload.items() if key != "self_sha256"},
                      indent=1, sort_keys=True, default=str)
    if hashlib.sha256(body.encode()).hexdigest() != stored_self:
        raise ValueError("cache self_sha256 is invalid")


def seal_slice(payload: Mapping[str, Any], *, source_artifact: Path,
               contract: Path, sealer_sha256: str, expected_cells: int = 288,
               expected_grid_id: str | None = None) -> dict[str, Any]:
    out = dict(payload)
    validate_slice(out, expected_cells=expected_cells, expected_grid_id=expected_grid_id)
    # A retry may be repairing a partially sealed slice.  Remove old self-referential fields
    # before recomputing the scientific payload hash.
    out.pop("scientific_payload_sha256", None)
    out.pop("self_sha256", None)
    out["schema_version"] = SCHEMA
    out["cache_custody"] = {
        "source_artifact": str(source_artifact),
        "source_artifact_sha256": json.loads(source_artifact.read_text())["self_sha256"],
        "sealer_script": "scripts/seal_garrido_surface_cache_v1.py",
        "sealer_script_sha256": sealer_sha256,
        "runtime": _runtime_manifest(),
        "observable_key_schema": "value_plus_panel_v1",
        "expected_cells_per_slice": int(expected_cells),
    }
    out["scientific_payload_sha256"] = scientific_hash(out)
    return out


def seal_cache(root: Path, *, contract: Path, reference: Path, manifest_output: Path,
               live_check: bool = False, expected_cells: int = 288,
               expected_grid_id: str | None = None) -> dict[str, Any]:
    source_artifact = reference
    files = sorted(path for path in root.rglob("*.json") if path.is_file())
    if not files:
        raise ValueError(f"no cache slices found below {root}")
    sealer_sha256 = _script_sha256()
    rows: list[dict[str, Any]] = []
    inferred_grid_id = expected_grid_id
    for path in files:
        payload = json.loads(path.read_text())
        if inferred_grid_id is None:
            inferred_grid_id = str(payload.get("grid_id"))
        out = seal_slice(payload, source_artifact=source_artifact, contract=contract,
                         sealer_sha256=sealer_sha256, expected_cells=expected_cells,
                         expected_grid_id=inferred_grid_id)
        seal_and_write(out, path, contract=contract, reference=reference)
        sealed = json.loads(path.read_text())
        verify_sealed_slice(sealed, expected_cells=expected_cells,
                            expected_grid_id=inferred_grid_id)
        rows.append({
            "path": str(path),
            "context": sealed["context"],
            "seed": int(sealed["seed"]),
            "n_cells": len(sealed["cells"]),
            "scientific_payload_sha256": sealed["scientific_payload_sha256"],
            "self_sha256": sealed["self_sha256"],
        })

    contexts = sorted({row["context"] for row in rows})
    seeds = sorted({row["seed"] for row in rows})
    summary: dict[str, Any] = {
        "schema_version": "garrido_surface_cache_manifest_v1",
        "claim_status": "CACHE_SEALED_DECLARED_REPLAY",
        "scope": "BURNED_REPLAY_CUSTODY_NO_SIMULATION_NEW_SEEDS_OR_LEARNER",
        "grid_id": inferred_grid_id,
        "expected_cells_per_slice": int(expected_cells),
        "cache_root": str(root),
        "contract": str(contract),
        "reference": str(reference),
        "contexts": contexts,
        "seeds": seeds,
        "n_slices": len(rows),
        "n_cells": sum(row["n_cells"] for row in rows),
        "module_manifest": module_manifest(SEALER_MODULES, script=__file__),
        "runtime": _runtime_manifest(),
        "files": rows,
        "live_check": {
            "requested": bool(live_check),
            "status": "NOT_RUN_BY_SEALER" if not live_check else "REQUIRES_EXPLICIT_CHECK",
            "note": "Sealing does not claim that cache cells reproduce the live DES; run the separate eight-cell check.",
        },
    }
    summary["scientific_payload_sha256"] = scientific_hash(summary)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    seal_and_write(summary, manifest_output, contract=contract, reference=reference)
    return json.loads(manifest_output.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("results/surface_cache/wrap288_v1"))
    parser.add_argument("--contract", type=Path,
                        default=Path("docs/PREREGISTRO_AUDITORIA_NORMALIZADOR_2026-08-05.md"))
    parser.add_argument("--reference", type=Path,
                        default=Path("results/garrido_normaliser_audit/result.json"))
    parser.add_argument("--manifest-output", type=Path,
                        default=Path("results/surface_cache_custody/wrap288_v1/result.json"))
    parser.add_argument("--expected-cells", type=int, default=288,
                        help="number of full-panel cells expected in each slice")
    parser.add_argument("--grid-id", default=None,
                        help="expected grid_id; defaults to the first slice's grid_id")
    parser.add_argument("--live-check", action="store_true",
                        help="record that a live check was requested; this command does not run it")
    args = parser.parse_args()
    result = seal_cache(args.cache, contract=args.contract, reference=args.reference,
                        manifest_output=args.manifest_output, live_check=args.live_check,
                        expected_cells=args.expected_cells, expected_grid_id=args.grid_id)
    print(f"sealed {result['n_slices']} slices / {result['n_cells']} cells")
    print(f"manifest: {args.manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
