#!/usr/bin/env python3
"""L(e-1) program, Gate 1: static 18-policy panel (no RL).

Evaluates the thesis-native factorial EXTENSION — 6 strategic buffer levels
(B0, I168..I1344, replenishment period tied to the level per Table 6.16) x
3 constant shift levels — CRN-paired on the frozen calibration tape universe
(700001-700060; 20 tapes per family R1/R2/mixed at thesis Table 6.12 '+'
levels). Physics: garrido_proxy_v1 (frozen sim kwargs). No gym wrapper, no
actions: constant policies run on the raw simulation, which makes the panel
exact and cheap.

Outputs (results/headroom/):
  static_18_policy_panel.csv   - one row per (buffer, shift, tape)
  panel_summary.json           - best fixed global/per-family (calibration
                                 only), clairvoyant per-tape bound (NOT a
                                 baseline), resource frontier rows.

Contract: configs/garrido_learning_v1.json; claim doc
docs/CLAIM_AND_IDENTIFICATION_2026-07-11.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import INVENTORY_BUFFERS  # noqa: E402
from supply_chain.ret_thesis import (  # noqa: E402
    compute_order_level_ret_excel_visible_ledger,
)
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

CONTRACT = json.loads(Path("configs/garrido_learning_v1.json").read_text())
PROXY = json.loads(
    Path("supply_chain/data/garrido_proxy_v1_freeze_2026-07-10.json").read_text()
)

BUFFER_LEVELS: dict[str, float | None] = {
    "B0": None,
    "I168": 168.0,
    "I336": 336.0,
    "I504": 504.0,
    "I672": 672.0,
    "I1344": 1344.0,
}
SHIFTS = (1, 2, 3)
FAMILIES = {
    "R1": (700_001, 700_020),
    "R2": (700_021, 700_040),
    "mixed": (700_041, 700_060),
}
CAMPAIGN_WEEKS = int(CONTRACT["campaign_horizon_weeks_provisional"])
WEEK = 168.0


def sim_kwargs_for(family: str, buffer_key: str, shifts: int, tape: int) -> dict[str, Any]:
    base = dict(PROXY["sim_kwargs"])
    base.pop("r24_attribution_window_hours", None)  # keep the disclosed proxy default
    overrides = dict(CONTRACT["campaign_families"][family]["risk_overrides"])
    kwargs: dict[str, Any] = {
        "shifts": shifts,
        "seed": tape,
        "horizon": CAMPAIGN_WEEKS * WEEK + 4_000.0,  # warm-up margin
        "risks_enabled": True,
        "enabled_risks": {"R11", "R12", "R13", "R14", "R21", "R22", "R23", "R24", "R3"},
        "risk_overrides": overrides,
        "r24_attribution_window_hours": 168.0,  # disclosed Tier-2 proxy
        **{k: v for k, v in base.items() if k not in ("risk_level",)},
        "risk_level": base.get("risk_level", "current"),
    }
    # Comparability across cells: buffered cells warm up earlier (stock is
    # present), so demand-start-at-warmup would give them MORE demand weeks.
    # The panel uses a common fixed metrics window instead (t0 below).
    kwargs["demand_start_after_warmup"] = False
    period = BUFFER_LEVELS[buffer_key]
    if period is not None:
        levels = INVENTORY_BUFFERS[int(period)]
        kwargs["initial_buffers"] = {
            "op3_rm": float(levels["op3_rm"]),
            "op5_rm": float(levels["op5_rm"]),
            "op9_rations": float(levels["op9_rations"]),
        }
        kwargs["inventory_replenishment_period"] = float(period)
    return kwargs


def run_cell(task: tuple[str, str, int, int]) -> dict[str, Any]:
    family, buffer_key, shifts, tape = task
    sim = MFSCSimulation(**sim_kwargs_for(family, buffer_key, shifts, tape))
    sim._start_processes()
    # Weekly slicing to sample backlog for the AUC panel metric.
    backlog_samples: list[float] = []
    horizon = float(sim.horizon)
    t = 0.0
    while t < horizon:
        t = min(t + WEEK, horizon)
        sim.env.run(until=t)
        backlog_samples.append(float(sim.pending_backorder_qty))
    # Common metrics window for ALL cells (see sim_kwargs_for comparability note).
    t0 = 12.0 * WEEK
    placed = [o for o in sim.orders if float(o.OPTj) >= t0]
    ret = compute_order_level_ret_excel_visible_ledger(
        placed, current_time=float(sim.env.now)
    )
    delivered = [
        o for o in placed if o.CTj is not None and not getattr(o, "lost", False)
    ]
    demanded_qty = sum(float(o.quantity) for o in placed)
    delivered_qty = sum(float(o.quantity) for o in delivered)
    return {
        "family": family,
        "buffer": buffer_key,
        "shifts": shifts,
        "tape": tape,
        "ret_excel": float(ret["mean_ret_excel"]),
        "placed": len(placed),
        "attended": len(delivered),
        "lost": int(ret["final_unattended"]),
        "fill_qty": delivered_qty / max(demanded_qty, 1e-9),
        "backlog_auc_qty_weeks": float(np.sum(backlog_samples)),
        "backlog_peak_qty": float(np.max(backlog_samples)) if backlog_samples else 0.0,
        "shift_hours_per_week": float(shifts * 8 * 6),
        "warmup_hours": float(sim.warmup_time),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/headroom"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--tapes-per-family", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, str, int, int]] = []
    for family, (lo, _hi) in FAMILIES.items():
        for i in range(args.tapes_per_family):
            tape = lo + i
            for buffer_key in BUFFER_LEVELS:
                for shifts in SHIFTS:
                    tasks.append((family, buffer_key, shifts, tape))
    print(f"panel cells: {len(tasks)}", flush=True)

    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(run_cell, tasks, chunksize=4)):
            rows.append(row)
            if (i + 1) % 90 == 0:
                print(f"  {i + 1}/{len(tasks)}", flush=True)

    out_csv = args.output_dir / "static_18_policy_panel.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Summary: best fixed (calibration-only by construction), per family,
    # and the clairvoyant per-tape bound (diagnostic upper bound only).
    def key(r):
        return (r["buffer"], r["shifts"])

    by_policy: dict[tuple, list[dict]] = {}
    for r in rows:
        by_policy.setdefault(key(r), []).append(r)
    policy_means = {
        k: {
            "ret_excel": float(np.mean([x["ret_excel"] for x in v])),
            "fill_qty": float(np.mean([x["fill_qty"] for x in v])),
            "backlog_auc": float(np.mean([x["backlog_auc_qty_weeks"] for x in v])),
            "shift_hours_per_week": v[0]["shift_hours_per_week"],
        }
        for k, v in by_policy.items()
    }
    best_global = max(policy_means, key=lambda k: policy_means[k]["ret_excel"])
    best_by_family: dict[str, Any] = {}
    for family in FAMILIES:
        fam_rows = [r for r in rows if r["family"] == family]
        fam_means: dict[tuple, float] = {}
        for r in fam_rows:
            fam_means.setdefault(key(r), []).append(r["ret_excel"])  # type: ignore[arg-type]
        fam_avg = {k: float(np.mean(v)) for k, v in fam_means.items()}
        b = max(fam_avg, key=lambda k: fam_avg[k])
        best_by_family[family] = {"policy": list(b), "ret_excel": fam_avg[b]}
    # Clairvoyant per-tape bound
    per_tape_best: list[float] = []
    tapes = sorted({r["tape"] for r in rows})
    for tape in tapes:
        per_tape_best.append(max(r["ret_excel"] for r in rows if r["tape"] == tape))
    clairvoyant = float(np.mean(per_tape_best))
    summary = {
        "contract": "configs/garrido_learning_v1.json",
        "n_cells": len(rows),
        "best_fixed_global": {
            "policy": list(best_global),
            **policy_means[best_global],
        },
        "best_fixed_by_family": best_by_family,
        "clairvoyant_per_tape_bound_NOT_A_BASELINE": clairvoyant,
        "clairvoyant_minus_best_fixed": clairvoyant
        - policy_means[best_global]["ret_excel"],
        "policy_means": {str(k): v for k, v in policy_means.items()},
    }
    (args.output_dir / "panel_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in (
        "best_fixed_global", "best_fixed_by_family",
        "clairvoyant_per_tape_bound_NOT_A_BASELINE",
        "clairvoyant_minus_best_fixed")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
