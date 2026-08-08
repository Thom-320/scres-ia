#!/usr/bin/env python3
"""Build a resumable development surface for the bounded seasonal-demand sensitivity.

This runner has no default seed block. A caller must provide explicit, already-used seeds and a
declared replay label. It writes raw slices only; the resulting sensitivity is never a confirmation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_transfer_confirmation_cache_v1 import (  # noqa: E402
    BASE_CONFIGS, BASE_FACTORS, BASE_GRID_ID, CONTEXTS, EXT_CONFIGS, EXT_FACTORS,
    EXT_GRID_ID, _project_base, _slug, evaluate,
)
from supply_chain.seed_custody import module_manifest  # noqa: E402

CONTRACT = Path("docs/ENMIENDA_DEMANDA_ESTACIONAL_P2_2026-08-08.md")
MODULES = (
    "supply_chain/config.py", "supply_chain/episode_metrics.py",
    "supply_chain/seed_custody.py", "supply_chain/supply_chain.py",
    "supply_chain/demand_seasonal.py",
)


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def build_slice(context: str, seed: int, *, grid: str, horizon: float,
                output: Path, replay_of: str, visibility: str) -> bool:
    configs = BASE_CONFIGS if grid == "base" else EXT_CONFIGS
    grid_id = BASE_GRID_ID if grid == "base" else EXT_GRID_ID
    factors = BASE_FACTORS if grid == "base" else EXT_FACTORS
    path = output / _slug(context) / f"{seed}.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("grid_id") != grid_id or existing.get("seed") != seed:
            raise ValueError(f"existing seasonal slice has incompatible identity: {path}")
        return False
    cells = [
        evaluate(
            config, context, seed, horizon,
            demand_process="garrido_seasonal_v1",
            demand_seasonal_contract={"forecast_mode": "holt_winters_observable"},
            demand_forecast_visibility=visibility,
        )
        for config in configs
    ]
    write(path, {
        "schema_version": "seasonal_sensitivity_surface_v1",
        "grid_id": grid_id, "context": context, "seed": int(seed),
        "metric": "ret_excel_risk_conditional", "horizon_hours": float(horizon),
        "factors": {key: list(values) for key, values in factors.items()},
        "demand_process": "garrido_seasonal_v1",
        "demand_seasonal_contract": {"forecast_mode": "holt_winters_observable"},
        "demand_forecast_visibility": visibility,
        "replay_of": replay_of, "contract": str(CONTRACT),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "cells": cells,
    })
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, action="append", required=True,
                    help="explicit seed already used by a declared development/replay block")
    ap.add_argument("--replay-of", required=True,
                    help="registry label proving that the explicit seeds are not virgin")
    ap.add_argument("--context", action="append", choices=tuple(CONTEXTS), default=None)
    ap.add_argument("--grid", choices=("base", "ext", "both"), default="both")
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--visibility", choices=("visible", "shuffled"), default="visible")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--output", type=Path,
                    default=Path("results/surface_cache/seasonal_sensitivity_v1"))
    args = ap.parse_args()
    if args.of < 1 or not 0 <= args.shard < args.of:
        raise SystemExit("--shard must lie in [0, --of)")
    contexts = args.context or list(CONTEXTS)
    grids = ("base", "ext") if args.grid == "both" else (args.grid,)
    jobs = [(grid, context, seed) for grid in grids for context in contexts for seed in args.seed]
    jobs = [job for i, job in enumerate(jobs) if i % args.of == args.shard]
    horizon = float(args.horizon_weeks * 168.0)
    started = time.perf_counter()
    manifest = module_manifest(MODULES, script=__file__)
    print(f"development sensitivity: {len(jobs)} slices · replay_of={args.replay_of}", flush=True)
    made = 0
    for grid, context, seed in jobs:
        out = args.output / grid
        if build_slice(context, seed, grid=grid, horizon=horizon, output=out,
                       replay_of=args.replay_of, visibility=args.visibility):
            made += 1
        print(f"  {grid} {context} seed {seed} ({time.perf_counter() - started:.0f}s)",
              flush=True)
    print(json.dumps({"run_role": "DEVELOPMENT_SENSITIVITY", "execution_role": "CACHE_BUILD",
                      "replay_of": args.replay_of, "seeds": args.seed,
                      "n_jobs": len(jobs), "n_created": made, "module_manifest": manifest},
                     indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
