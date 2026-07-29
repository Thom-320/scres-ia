#!/usr/bin/env python3
"""Benchmark risk-aware S1 shards on burned tapes and freeze feasibility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.paper2_exhaustive_search.program_s_design import build_program_s_risk_tape  # noqa: E402
from research.paper2_exhaustive_search.program_s_transducer import extract_program_s_skeleton, run_program_s_direct  # noqa: E402
from supply_chain.program_o_full_des_transducer import simulate_full_des_frontier  # noqa: E402
from supply_chain.program_s_risk_interaction import ProgramSCell  # noqa: E402


DESIGN_PATH = ROOT / "research/paper2_exhaustive_search/program_s_native_morris_design_v1_1.json"
PARENT_PATH = ROOT / "contracts/program_o_full_des_hpi_translation_v1.json"
OUT = ROOT / "results/program_s/s1_compute_benchmark_v1_1/result.json"
BURNED_SEED = 7_430_001
WALLTIME_LIMIT_SECONDS = 7 * 24 * 3600
WORKERS = 2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_cell(group: dict, point: dict, product: dict) -> ProgramSCell:
    return ProgramSCell(
        stratum="THESIS_NATIVE_INDEPENDENT",
        mask=group["mask"],
        coupling="independent",
        phi_by_risk=point["phi_by_risk"],
        psi_by_risk=point["psi_by_risk"],
        r14_probability_multiplier=point["r14_probability_multiplier"],
        baseline_capacity_multiplier=point["baseline_capacity_multiplier"],
        regime_persistence=product["rho"],
        dominant_share=product["share"],
        alarm_lead_hours=0.0,
        alarm_balanced_accuracy=0.5,
    )


def main() -> int:
    design = json.loads(DESIGN_PATH.read_text())
    parent = json.loads(PARENT_PATH.read_text())
    scheduler = parent["action"]["within_week_schedulers"][parent["action"]["primary_scheduler"]]
    product = design["product_cells"]["rho90_share90"]
    rows = []
    for group in design["groups"]:
        point = group["trajectories"][0]["points"][0]
        cell = make_cell(group, point, product)
        started = time.perf_counter()
        tape = build_program_s_risk_tape(cell, tape_id=BURNED_SEED, horizon_hours=8 * 168)
        after_tape = time.perf_counter()
        direct = run_program_s_direct(
            seed=BURNED_SEED,
            calendar=[2] * 8,
            scheduler=scheduler,
            cell=cell,
            risk_event_tape=tape["events"],
        )
        after_direct = time.perf_counter()
        skeleton = extract_program_s_skeleton(direct)
        frontier = simulate_full_des_frontier(skeleton=skeleton, scheduler=scheduler)
        finished = time.perf_counter()
        rows.append({
            "mask": group["mask"],
            "risk_event_count": len(tape["events"]),
            "risk_tape_seconds": after_tape - started,
            "direct_and_skeleton_seconds": after_direct - after_tape,
            "frontier_65536_seconds": finished - after_direct,
            "total_seconds": finished - started,
            "frontier_rows": int(np.asarray(frontier["ret_visible"]).size),
        })
    native_points = sum(len(trajectory["points"]) for group in design["groups"] for trajectory in group["trajectories"])
    projected_shards = native_points * len(design["product_cells"]) * 12
    conservative_seconds_per_shard = max(row["total_seconds"] for row in rows) * 1.25
    projected_worker_seconds = projected_shards * conservative_seconds_per_shard
    projected_wall_seconds = projected_worker_seconds / WORKERS
    passed = projected_wall_seconds <= WALLTIME_LIMIT_SECONDS
    payload = {
        "schema_version": "program_s_s1_compute_benchmark_v1_1",
        "burned_seed": BURNED_SEED,
        "scientific_751_seeds_opened": False,
        "design_sha256": sha256(DESIGN_PATH),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "rows": rows,
        "native_design_points": native_points,
        "product_cells": len(design["product_cells"]),
        "tapes": 12,
        "projected_shards": projected_shards,
        "conservative_multiplier": 1.25,
        "workers": WORKERS,
        "projected_wall_seconds": projected_wall_seconds,
        "walltime_limit_seconds": WALLTIME_LIMIT_SECONDS,
        "reduction_ladder_if_failed": [
            "reduce optimized trajectories from 10 to 6 before design refreeze",
            "retain all three masks but keep R23 fixed as a negative control",
            "do not execute S-WARTIME under the S-NATIVE budget",
            "stop before scientific execution if the reduced projection still exceeds seven days"
        ],
        "pass": passed,
        "verdict": "PASS_S1_COMPUTE_BENCHMARK_FEASIBLE" if passed else "STOP_S1_COMPUTE_BENCHMARK_INFEASIBLE_REFREEZE_REQUIRED"
    }
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
