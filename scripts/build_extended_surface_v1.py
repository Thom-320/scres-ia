#!/usr/bin/env python3
"""Build the extended design surface `wrap288_compat_extended_v1` (4,608 configurations).

The upstream buffers were already wired -- `initial_buffers` feeds `op3_rm` at the WDC and
`op5_rm` at the AL. All that was missing was exposing them as FACTORS. That is Garrido's 28 July
instruction (add decision variables, not longer episodes) executed where it can pay: the surface
gates closed cross-context transfer (H_regime +0.0038), so grid transfer is the only axis left.

NO NEW SEEDS. The extension adds configurations, not tapes: it runs on the same burned block
5_300_001-012 as a declared replay. That is why this axis needs no opening signature -- there is
nothing to open.

The null subgrid `op3_rm = op5_rm = 0` IS the frozen 288 surface by construction, so f1 anchors
every one of its 20,736 cells against the sealed cache written by an earlier run.

Contract: docs/ENMIENDA_REJILLA_EXTENDIDA_4608_2026-08-05.md
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

R1R = ("R11", "R12", "R13", "R14")
R2R = ("R21", "R22", "R23", "R24")
CONTEXTS = {
    "R1r": (R1R, {}), "R2r": (R2R, {}), "R1r+R2r": (R1R + R2R, {}),
    "R1r|esc": (R1R, {r: 3.0 for r in R1R}),
    "R2r|esc": (R2R, {r: 3.0 for r in R2R}),
    "R1r+R2r|esc": (R1R + R2R, {r: 3.0 for r in R1R + R2R}),
}
#: Same hours-to-units convention the runner already uses for op9_rations: h * 2500 / 24.
#: 0, 168 h, 672 h, 1344 h of raw material. Frozen in the amendment before this ran.
RAW_LEVELS = (0.0, 17_500.0, 70_000.0, 140_000.0)
FACTORS = {
    "buffer_hours": (0.0, 168.0, 336.0, 504.0, 672.0, 1344.0),
    "shifts": (1, 2, 3),
    "op9_rop": (12.0, 24.0, 36.0, 48.0),
    "op12_rop": (12.0, 24.0, 36.0, 48.0),
    "op3_rm": RAW_LEVELS,
    "op5_rm": RAW_LEVELS,
}
FACTOR_NAMES = tuple(FACTORS)
CONFIGS = tuple(dict(zip(FACTOR_NAMES, combo)) for combo in itertools.product(*FACTORS.values()))
METRIC = "ret_excel_risk_conditional"
GRID_ID = "wrap288_compat_extended_v1"
SEED_BASE = 5_300_001
MODULES = ("supply_chain/supply_chain.py", "supply_chain/config.py",
           "supply_chain/episode_metrics.py", "supply_chain/seed_custody.py")


def evaluate(config: dict, context: str, seed: int, horizon: float) -> dict:
    """Byte-for-byte the Q2 runner's semantics, with op3_rm and op5_rm no longer pinned to zero."""
    risks, freq = CONTEXTS[context]
    sim = MFSCSimulation(
        shifts=int(config["shifts"]),
        initial_buffers={"op3_rm": float(config["op3_rm"]),
                         "op5_rm": float(config["op5_rm"]),
                         "op9_rations": float(config["buffer_hours"]) * 2_500.0 / 24.0},
        inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
        risks_enabled=True, risk_level="current", enabled_risks=set(risks),
        risk_frequency_multipliers_by_id=dict(freq) or None,
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    sim.params["op9_rop"] = float(config["op9_rop"])
    sim.params["op12_rop"] = float(config["op12_rop"])
    sim.run()
    panel = compute_episode_metrics(sim)
    return {
        "value": float(panel[METRIC]),
        "drivers": [float(panel["excel_case_pct_autotomy"]) / 100.0,
                    float(panel["excel_case_pct_recovery"]) / 100.0,
                    float(panel["excel_case_pct_risk_no_recovery"]) / 100.0,
                    float(panel["excel_case_pct_fill_rate"]) / 100.0],
        "panel": {k: float(panel[k]) for k in
                  ("ret_excel", "ret_excel_full_ledger", "ret_excel_risk_conditional",
                   "flow_fill_rate", "lost_orders", "delivered_rations", "demanded_rations")},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--out", type=Path, default=Path("results/surface_cache") / GRID_ID)
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    seeds = [SEED_BASE + i for i in range(args.seeds)]
    manifest = module_manifest(MODULES, script=__file__)
    started = time.perf_counter()
    print(f"  {len(CONFIGS)} configs x {len(CONTEXTS)} contextos x {len(seeds)} semillas "
          f"= {len(CONFIGS) * len(CONTEXTS) * len(seeds):,} episodios")

    for ctx in CONTEXTS:
        for seed in seeds:
            path = args.out / ctx.replace("+", "_").replace("|", "_") / f"{seed}.json"
            if path.exists():
                continue
            cells = [evaluate(cfg, ctx, seed, horizon) for cfg in CONFIGS]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "grid_id": GRID_ID, "context": ctx, "seed": seed, "metric": METRIC,
                "horizon_hours": horizon, "module_manifest": manifest,
                "factors": {k: list(v) for k, v in FACTORS.items()},
                "replay_of": "garrido_q2_des288",
                "contract": "docs/ENMIENDA_REJILLA_EXTENDIDA_4608_2026-08-05.md",
                "cells": cells}, indent=1))
            print(f"  {ctx} seed {seed} ({time.perf_counter() - started:.0f}s)", flush=True)
    print(f"  listo en {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
