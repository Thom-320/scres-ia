#!/usr/bin/env python3
"""Burned-data D1 probe for demand-belief misspecification; opens no new seed."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.program_o_full_des_transducer import extract_full_des_skeleton, simulate_full_des_frontier  # noqa: E402
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS  # noqa: E402
from supply_chain.program_t_full_des_mpc import (  # noqa: E402
    FullDEST0Config,
    joint_belief_ret_transducer_calendar,
    ret_transducer_t0_calendar,
)
from supply_chain.program_t_joint_belief import ExactJointBelief  # noqa: E402


ARMS = ("frozen_belief", "oracle_parameters", "adaptive_joint_posterior", "shuffled_history", "wrong_product")
METRICS = ("ret_visible", "worst_product_fill", "lost_orders", "gross_production_quantity")


def paired_lcb(values: np.ndarray, *, resamples: int = 5000) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan")
    rng = np.random.default_rng(20260720)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025))


def _calendar(arm, *, skeleton, sched, config, theta):
    if arm == "frozen_belief":
        return ret_transducer_t0_calendar(skeleton=skeleton, scheduler=sched, config=config)
    if arm == "oracle_parameters":
        oracle_config = FullDEST0Config(
            horizon=config.horizon,
            mode=config.mode,
            particles=config.particles,
            regime_persistence=theta[0],
            dominant_share=theta[1],
            worst_product_floor=config.worst_product_floor,
        )
        return ret_transducer_t0_calendar(
            skeleton=skeleton, scheduler=sched, config=oracle_config
        )
    initial = ExactJointBelief.uniform()
    transform = "shuffled" if arm == "shuffled_history" else "wrong_product" if arm == "wrong_product" else "real"
    return joint_belief_ret_transducer_calendar(
        skeleton=skeleton, scheduler=sched, config=config, belief=initial, history_transform=transform
    )


def summarize(rows: dict[str, dict[str, list[float]]], cells: list[str]) -> dict[str, object]:
    output: dict[str, object] = {"cells": {}}
    pooled: dict[str, list[float]] = {"c_missp": [], "c_adaptive": [], "g_remaining": []}
    for cell in cells:
        arm = rows[cell]
        frozen = np.asarray(arm["frozen_belief"]["ret_visible"])
        oracle = np.asarray(arm["oracle_parameters"]["ret_visible"])
        adaptive = np.asarray(arm.get("adaptive_joint_posterior", {}).get("ret_visible", []))
        estimands = {"c_missp": oracle - frozen}
        if len(adaptive):
            estimands.update({"c_adaptive": adaptive - frozen, "g_remaining": oracle - adaptive})
        cell_out = {
            "arms": {
                name: {
                    **{key: float(np.mean(metrics[key])) for key in METRICS},
                    "mean_online_ms": float(np.mean(metrics["online_ms"])),
                }
                for name, metrics in arm.items()
            },
            "estimands": {},
        }
        for name, value in estimands.items():
            cell_out["estimands"][name] = {"mean": float(np.mean(value)), "lcb95": paired_lcb(value)}
            pooled[name].extend(map(float, value))
        output["cells"][cell] = cell_out
    pooled_out = {}
    for name, value in pooled.items():
        if value:
            pooled_out[name] = {"mean": float(np.mean(value)), "lcb95": paired_lcb(np.asarray(value))}
    output["pooled"] = pooled_out
    c_cell = [output["cells"][cell]["estimands"]["c_missp"]["mean"] for cell in cells]
    c_pooled = pooled_out["c_missp"]["mean"]
    if c_pooled >= 0.02 and sum(value >= 0.01 for value in c_cell) >= 2:
        status = "CRACK_MATERIAL"
    elif c_pooled >= 0.01 and min(c_cell) >= 0.0:
        status = "CRACK_USABLE"
    elif max(c_cell) < 0.01:
        status = "CRACK_NEGLIGIBLE"
    else:
        status = "CRACK_INCONCLUSIVE"
    if "c_adaptive" in pooled_out:
        recovery = pooled_out["c_adaptive"]["mean"] / c_pooled if c_pooled > 0 else 0.0
        output["adaptive_recovery_fraction"] = float(recovery)
        if recovery >= 0.80:
            status = "ADAPTIVE_MPC_CLOSES_CRACK"
        if pooled_out["g_remaining"]["mean"] >= 0.01 and min(
            output["cells"][cell]["estimands"]["g_remaining"]["mean"] for cell in cells
        ) >= 0.0:
            status = "LEARNABLE_RESIDUAL"
    output["verdict"] = status
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tapes", type=int, choices=(1, 24, 48), default=24)
    parser.add_argument("--arms", default="frozen_belief,oracle_parameters")
    parser.add_argument("--modes", default="scenario")
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--hard-cap-seconds", type=float, default=1800.0)
    parser.add_argument("--output", type=Path, default=ROOT / "results/program_t/d1_belief_misspecification_v1/result.json")
    args = parser.parse_args()
    arms = [value.strip() for value in args.arms.split(",")]
    if any(value not in ARMS for value in arms) or "frozen_belief" not in arms or "oracle_parameters" not in arms:
        raise ValueError("arms must be known and include frozen_belief,oracle_parameters")
    modes = [value.strip() for value in args.modes.split(",")]
    sched = scheduler()
    started = time.perf_counter()
    all_results = {}
    for mode in modes:
        config = FullDEST0Config(horizon=3, mode=mode, particles=args.particles, worst_product_floor=0.70)
        rows = {cell.cell_id: {arm: {key: [] for key in METRICS} | {"calendar": [], "online_ms": []} for arm in arms} for cell in CONFIRMED_RET_CELLS}
        for cell in CONFIRMED_RET_CELLS:
            theta = (cell.regime_persistence, cell.dominant_share)
            for offset in range(args.n_tapes):
                if time.perf_counter() - started > args.hard_cap_seconds:
                    raise TimeoutError("D1 hard cap exceeded before completion")
                tape = 7_490_001 + offset
                skeleton, _ = extract_full_des_skeleton(seed=tape, scheduler=sched, regime_persistence=theta[0], dominant_share=theta[1], downstream_freight_physics_mode="fixed_clock_physical_v1")
                for arm in arms:
                    calendar, diagnostics = _calendar(arm, skeleton=skeleton, sched=sched, config=config, theta=theta)
                    metrics = simulate_full_des_frontier(skeleton=skeleton, scheduler=sched, calendars=np.asarray([calendar], dtype=np.uint8))
                    target = rows[cell.cell_id][arm]
                    target["calendar"].append(list(calendar)); target["online_ms"].append(float(diagnostics["online_ms"]))
                    for key in METRICS:
                        target[key].append(float(metrics[key][0]))
                print(json.dumps({"mode": mode, "cell": cell.cell_id, "tape": tape}), flush=True)
        all_results[mode] = {"rows": rows, "summary": summarize(rows, [cell.cell_id for cell in CONFIRMED_RET_CELLS])}
    payload = {
        "schema_version": "program_t_d1_belief_misspecification_v1",
        "claim_status": "EXPLORATORY_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "burned_seed_range": [7490001, 7490256],
        "n_tapes": args.n_tapes,
        "arms": arms,
        "particles": args.particles,
        "elapsed_seconds": time.perf_counter() - started,
        "results": all_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({mode: value["summary"] for mode, value in all_results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
