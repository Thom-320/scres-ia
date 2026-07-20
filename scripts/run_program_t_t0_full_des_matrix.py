#!/usr/bin/env python3
"""Run the burned-data full-DES T0 matrix; never opens a new seed."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    MATRIX_KEYS, extract_full_des_skeleton, simulate_full_des_frontier,
)
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS  # noqa: E402
from supply_chain.program_t_full_des_mpc import (  # noqa: E402
    ret_transducer_t0_calendar,
    t0_calendar,
    t0_grid,
)
from supply_chain.program_t_t0_gate import adjudicate_t0_residual  # noqa: E402

Q_RESULT = ROOT / "results/program_q/confirmation_v1_20260718/artifacts/confirmation/evaluation/result.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tapes", type=int, default=48)
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--horizons", default="1,3,4,6,8")
    parser.add_argument("--modes", default="nominal,scenario,robust,constraint_aware")
    parser.add_argument("--cells", default="rho75_share90,rho90_share75,rho90_share90")
    parser.add_argument("--provider", choices=("compact", "ret_transducer"), default="compact")
    parser.add_argument("--output", type=Path, default=ROOT / "results/program_t/t0_full_des_matrix_v1/result.json")
    args = parser.parse_args()
    if not 1 <= args.n_tapes <= 256:
        raise ValueError("T0 may use only the 256 already-burned Q tapes")
    wanted_h = {int(x) for x in args.horizons.split(",")}
    wanted_m = {x.strip() for x in args.modes.split(",")}
    wanted_c = {x.strip() for x in args.cells.split(",")}
    configs = [c for c in t0_grid(particles=args.particles) if c.horizon in wanted_h and c.mode in wanted_m]
    q = json.loads(Q_RESULT.read_text())
    sched = scheduler()
    output = {"schema_version": "program_t_t0_full_des_matrix_v1", "created_at": datetime.now(timezone.utc).isoformat(), "claim_status": "BURNED_T0_ROUTING_EVIDENCE", "planning_provider": args.provider, "seed_range": [7490001, 7490256], "n_tapes": args.n_tapes, "cells": {}}
    for cell_index, cell in enumerate(CONFIRMED_RET_CELLS):
        if cell.cell_id not in wanted_c:
            continue
        q_calendars = q["trajectory_audits"][cell.cell_id]
        learner_seeds = sorted(q_calendars, key=int)
        rows = {c.config_id: {key: [] for key in ("ret_visible", "worst_product_fill", "lost_orders", "gross_production_quantity")} | {"calendar": [], "online_ms": []} for c in configs}
        learner = {key: [] for key in ("ret_visible", "worst_product_fill", "lost_orders", "gross_production_quantity")}
        for tape_offset in range(args.n_tapes):
            tape = 7_490_001 + tape_offset
            skeleton, _ = extract_full_des_skeleton(seed=tape, scheduler=sched, regime_persistence=cell.regime_persistence, dominant_share=cell.dominant_share, downstream_freight_physics_mode="fixed_clock_physical_v1")
            q_cals = np.asarray([q_calendars[str(seed)]["calendars"][tape_offset] for seed in learner_seeds], dtype=np.uint8)
            q_metrics = simulate_full_des_frontier(skeleton=skeleton, scheduler=sched, calendars=q_cals)
            for key in learner:
                learner[key].append(float(np.mean(q_metrics[key])))
            for config in configs:
                if args.provider == "ret_transducer":
                    calendar, diagnostics = ret_transducer_t0_calendar(skeleton=skeleton, scheduler=sched, config=config)
                else:
                    calendar, diagnostics = t0_calendar(skeleton=skeleton.as_dict(), scheduler=sched, config=config)
                metrics = simulate_full_des_frontier(skeleton=skeleton, scheduler=sched, calendars=np.asarray([calendar], dtype=np.uint8))
                target = rows[config.config_id]
                target["calendar"].append(list(calendar)); target["online_ms"].append(float(diagnostics["online_ms"]))
                for key in ("ret_visible", "worst_product_fill", "lost_orders", "gross_production_quantity"):
                    target[key].append(float(metrics[key][0]))
            print(json.dumps({"cell": cell.cell_id, "tape": tape, "completed": tape_offset + 1, "total": args.n_tapes}), flush=True)
        eligible = []
        for config_id, values in rows.items():
            eligible.append((float(np.mean(values["ret_visible"])), float(np.mean(values["worst_product_fill"])), config_id))
        _mean, _fill, best_id = max(eligible)
        best = rows[best_id]
        worst_delta = np.asarray(learner["worst_product_fill"]) - np.asarray(best["worst_product_fill"])
        lost_delta = np.asarray(learner["lost_orders"]) - np.asarray(best["lost_orders"])
        resource_delta = np.asarray(learner["gross_production_quantity"]) - np.asarray(best["gross_production_quantity"])
        adjudication = (
            adjudicate_t0_residual(best_observable_ret=learner["ret_visible"], reinforced_mpc_ret=best["ret_visible"], worst_product_delta=worst_delta, lost_order_delta=lost_delta, resource_delta=resource_delta)
            if args.n_tapes >= 2
            else {"status": "PREFLIGHT_ONLY_INSUFFICIENT_TAPES_FOR_INFERENCE"}
        )
        output["cells"][cell.cell_id] = {"best_reinforced_mpc": best_id, "learner_mean_ret": float(np.mean(learner["ret_visible"])), "best_mpc_mean_ret": float(np.mean(best["ret_visible"])), "adjudication": adjudication, "learner": learner, "comparators": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({cell: data["adjudication"] for cell, data in output["cells"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
