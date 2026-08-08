#!/usr/bin/env python3
"""Build the virgin 4,608-cell surface for the Garrido grid-transfer confirmation.

The 288 projection is written from the same six-factor evaluations, so the confirmation does not
run the null subgrid twice.  The script only creates raw panel-complete slices.  A separate sealing
step must run after all slices exist; this keeps an unfinished worker from looking like a sealed
scientific result.

Contract: docs/PREREGISTRO_CONFIRMACION_TRANSFERENCIA_REJILLA_2026-08-05.md
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

R1R = ("R11", "R12", "R13", "R14")
R2R = ("R21", "R22", "R23", "R24")
CONTEXTS = {
    "R1r": (R1R, {}),
    "R2r": (R2R, {}),
    "R1r+R2r": (R1R + R2R, {}),
    "R1r|esc": (R1R, {r: 3.0 for r in R1R}),
    "R2r|esc": (R2R, {r: 3.0 for r in R2R}),
    "R1r+R2r|esc": (R1R + R2R, {r: 3.0 for r in R1R + R2R}),
}
BASE_FACTORS = {
    "buffer_hours": (0.0, 168.0, 336.0, 504.0, 672.0, 1344.0),
    "shifts": (1, 2, 3),
    "op9_rop": (12.0, 24.0, 36.0, 48.0),
    "op12_rop": (12.0, 24.0, 36.0, 48.0),
}
RAW_LEVELS = (0.0, 17_500.0, 70_000.0, 140_000.0)
EXT_FACTORS = dict(BASE_FACTORS, op3_rm=RAW_LEVELS, op5_rm=RAW_LEVELS)
BASE_NAMES = tuple(BASE_FACTORS)
EXT_NAMES = tuple(EXT_FACTORS)
BASE_CONFIGS = tuple(
    dict(zip(BASE_NAMES, values))
    for values in itertools.product(*BASE_FACTORS.values())
)
EXT_CONFIGS = tuple(
    dict(zip(EXT_NAMES, values))
    for values in itertools.product(*EXT_FACTORS.values())
)
BASE_GRID_ID = "wrap288_v1"
EXT_GRID_ID = "wrap288_compat_extended_v1"
METRIC = "ret_excel_risk_conditional"
CONTRACT = Path("docs/PREREGISTRO_CONFIRMACION_TRANSFERENCIA_REJILLA_2026-08-05.md")
MODULES = (
    "supply_chain/config.py",
    "supply_chain/episode_metrics.py",
    "supply_chain/seed_custody.py",
    "supply_chain/supply_chain.py",
)
PANEL_KEYS = (
    "ret_excel",
    "ret_excel_full_ledger",
    "ret_excel_risk_conditional",
    "flow_fill_rate",
    "lost_orders",
    "delivered_rations",
    "demanded_rations",
)


def evaluate(config: dict[str, float | int], context: str, seed: int, horizon: float,
             *, demand_process: str = "thesis_uniform",
             demand_seasonal_contract: dict[str, Any] | None = None,
             demand_forecast_visibility: str = "visible") -> dict[str, Any]:
    """Evaluate one frozen configuration and retain the complete observable panel."""
    risks, freq = CONTEXTS[context]
    sim = MFSCSimulation(
        shifts=int(config["shifts"]),
        initial_buffers={
            "op3_rm": float(config.get("op3_rm", 0.0)),
            "op5_rm": float(config.get("op5_rm", 0.0)),
            "op9_rations": float(config["buffer_hours"]) * 2_500.0 / 24.0,
        },
        inventory_replenishment_period=0.0,
        seed=int(seed),
        horizon=float(horizon),
        risks_enabled=True,
        risk_level="current",
        enabled_risks=set(risks),
        risk_frequency_multipliers_by_id=dict(freq) or None,
        strict_exogenous_crn=True,
        year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"],
        r14_defect_mode=P["r14_defect_mode"],
        demand_process=demand_process,
        demand_seasonal_contract=demand_seasonal_contract,
        demand_forecast_visibility=demand_forecast_visibility,
    )
    sim.params["op9_rop"] = float(config["op9_rop"])
    sim.params["op12_rop"] = float(config["op12_rop"])
    sim.run()
    panel = compute_episode_metrics(sim)
    return {
        "value": float(panel[METRIC]),
        "drivers": [
            float(panel["excel_case_pct_autotomy"]) / 100.0,
            float(panel["excel_case_pct_recovery"]) / 100.0,
            float(panel["excel_case_pct_risk_no_recovery"]) / 100.0,
            float(panel["excel_case_pct_fill_rate"]) / 100.0,
        ],
        "panel": {key: float(panel[key]) for key in PANEL_KEYS},
    }


def _slug(context: str) -> str:
    return context.replace("+", "_").replace("|", "_")


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))


def _project_base(ext_payload: dict[str, Any], *, contract: str) -> dict[str, Any]:
    """Project the null subgrid without evaluating it a second time."""
    base_keys = {tuple(sorted(config.items())) for config in BASE_CONFIGS}
    base_cells = []
    for config, cell in zip(EXT_CONFIGS, ext_payload["cells"]):
        if config["op3_rm"] == 0.0 and config["op5_rm"] == 0.0:
            base_cells.append(cell)
    if len(base_cells) != len(BASE_CONFIGS):
        raise ValueError(f"null projection has {len(base_cells)} cells, expected {len(BASE_CONFIGS)}")
    # Assert the order, not just the count: a reordered projection would be a silent cache bug.
    projected_keys = [
        tuple(sorted({key: value for key, value in config.items() if key in BASE_FACTORS}.items()))
        for config in EXT_CONFIGS
        if config["op3_rm"] == 0.0 and config["op5_rm"] == 0.0
    ]
    expected_keys = [tuple(sorted(config.items())) for config in BASE_CONFIGS]
    if projected_keys != expected_keys or set(projected_keys) != base_keys:
        raise ValueError("null projection order does not match the frozen 288 grid")
    return {
        "schema_version": "garrido_surface_cache_v1",
        "grid_id": BASE_GRID_ID,
        "context": ext_payload["context"],
        "seed": int(ext_payload["seed"]),
        "metric": METRIC,
        "horizon_hours": float(ext_payload["horizon_hours"]),
        "module_manifest": ext_payload["module_manifest"],
        "factors": {key: list(values) for key, values in BASE_FACTORS.items()},
        "replay_of": "garrido_grid_transfer_confirmation",
        "projection_of": EXT_GRID_ID,
        "contract": contract,
        "cells": base_cells,
    }


def _load_or_build(
    *,
    context: str,
    seed: int,
    horizon: float,
    ext_path: Path,
    base_path: Path,
    contract: str,
    manifest: dict[str, Any],
) -> bool:
    """Return True when a new DES surface was evaluated."""
    if ext_path.exists():
        ext_payload = json.loads(ext_path.read_text())
        if len(ext_payload.get("cells", [])) != len(EXT_CONFIGS):
            raise ValueError(f"existing extended slice is incomplete: {ext_path}")
        if ext_payload.get("grid_id") != EXT_GRID_ID:
            raise ValueError(f"wrong extended grid id in {ext_path}")
    else:
        cells = [evaluate(config, context, seed, horizon) for config in EXT_CONFIGS]
        ext_payload = {
            "schema_version": "garrido_surface_cache_v1",
            "grid_id": EXT_GRID_ID,
            "context": context,
            "seed": int(seed),
            "metric": METRIC,
            "horizon_hours": float(horizon),
            "module_manifest": manifest,
            "factors": {key: list(values) for key, values in EXT_FACTORS.items()},
            "replay_of": "garrido_grid_transfer_confirmation",
            "contract": contract,
            "cells": cells,
        }
        _write(ext_path, ext_payload)
        made = True
    if not base_path.exists():
        _write(base_path, _project_base(ext_payload, contract=contract))
    return locals().get("made", False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=8_100_001)
    parser.add_argument("--seeds", type=int, default=60)
    parser.add_argument("--horizon-weeks", type=int, default=52)
    parser.add_argument("--context", action="append", default=None)
    parser.add_argument("--extended-out", type=Path,
                        default=Path("results/surface_cache/garrido_transfer_confirmation_ext_v1"))
    parser.add_argument("--base-out", type=Path,
                        default=Path("results/surface_cache/garrido_transfer_confirmation_base_v1"))
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    args = parser.parse_args()
    contexts = args.context or list(CONTEXTS)
    unknown = [context for context in contexts if context not in CONTEXTS]
    if unknown:
        raise SystemExit(f"unknown contexts: {unknown}")
    if args.seeds <= 0:
        raise SystemExit("--seeds must be positive")
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.seeds)))
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    manifest = module_manifest(MODULES, script=__file__)
    started = time.perf_counter()
    print(
        f"  {len(EXT_CONFIGS):,} configuraciones extendidas x {len(contexts)} contextos x "
        f"{len(seeds)} semillas = {len(EXT_CONFIGS) * len(contexts) * len(seeds):,} episodios",
        flush=True,
    )
    made = 0
    for context in contexts:
        for seed in seeds:
            ext_path = args.extended_out / _slug(context) / f"{seed}.json"
            base_path = args.base_out / _slug(context) / f"{seed}.json"
            if _load_or_build(
                context=context,
                seed=seed,
                horizon=horizon,
                ext_path=ext_path,
                base_path=base_path,
                contract=str(args.contract),
                manifest=manifest,
            ):
                made += 1
            if made and made % 1 == 0:
                print(f"  {context} seed {seed} ({time.perf_counter() - started:.0f}s)", flush=True)
    print(f"  listo en {time.perf_counter() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
