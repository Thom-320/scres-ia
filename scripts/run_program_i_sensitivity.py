#!/usr/bin/env python3
"""Run Program I's Garrido-realistic Morris screen on exposed DES factors.

This is a development/screen runner only. It cannot open branching,
confirmation, or virgin-RL universes and cannot authorize RL.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import HOURS_PER_WEEK, THESIS_FAITHFUL_PROTOCOL as P
from supply_chain.decision_right_discovery import (
    NumericFactor, morris_effects, morris_trajectories,
)
from supply_chain.episode_metrics import compute_episode_metrics
from supply_chain.supply_chain import MFSCSimulation


FACTORS = (
    NumericFactor("op3_inventory_target", 0.0, 122_880.0, "inventory"),
    NumericFactor("op3_order_quantity", 7_750.0, 47_000.0, "replenishment"),
    NumericFactor("op3_review_period", 84.0, 336.0, "replenishment"),
    NumericFactor("op5_inventory_target", 0.0, 122_880.0, "inventory"),
    NumericFactor("op5_capacity_posture", 1.0, 3.0, "production"),
    NumericFactor("op7_release_period", 24.0, 72.0, "batching"),
    NumericFactor("op9_inventory_target", 0.0, 126_000.0, "inventory"),
    NumericFactor("op9_release_period", 12.0, 48.0, "dispatch"),
    NumericFactor("op10_dispatch_quantity", 1_200.0, 5_200.0, "transport"),
    NumericFactor("op10_dispatch_period", 12.0, 48.0, "transport"),
    NumericFactor("op12_dispatch_quantity", 1_200.0, 5_200.0, "transport"),
    NumericFactor("op12_dispatch_period", 12.0, 48.0, "transport"),
    NumericFactor("risk_frequency_scale", 0.5, 2.0, "risk", "environment_uncertainty"),
    NumericFactor("risk_impact_scale", 0.5, 2.0, "risk", "environment_uncertainty"),
    NumericFactor("demand_level", 0.75, 1.5, "demand", "environment_uncertainty"),
)


def run_des(params: dict[str, float], tape: int, horizon_weeks: int) -> dict[str, float]:
    buffers = {
        "op3_rm": params["op3_inventory_target"],
        "op5_rm": params["op5_inventory_target"],
        "op9_rations": params["op9_inventory_target"],
    }
    sim = MFSCSimulation(
        shifts=1,
        initial_buffers=buffers,
        seed=tape,
        horizon=float(horizon_weeks) * HOURS_PER_WEEK,
        risks_enabled=True,
        risk_level="current",
        strict_exogenous_crn=True,
        risk_frequency_multiplier=params["risk_frequency_scale"],
        risk_impact_multiplier=params["risk_impact_scale"],
        demand_mean_multiplier=params["demand_level"],
        year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"],
        r14_defect_mode=P["r14_defect_mode"],
    )
    sim.step(
        action={
            "op3_q": params["op3_order_quantity"],
            "op3_rop": params["op3_review_period"],
            "assembly_shifts": int(np.clip(np.rint(params["op5_capacity_posture"]), 1, 3)),
            "op8_rop": params["op7_release_period"],
            "op9_rop": params["op9_release_period"],
            "op10_q_min": params["op10_dispatch_quantity"],
            "op10_q_max": params["op10_dispatch_quantity"],
            "op10_rop": params["op10_dispatch_period"],
            "op12_q_min": params["op12_dispatch_quantity"],
            "op12_q_max": params["op12_dispatch_quantity"],
            "op12_rop": params["op12_dispatch_period"],
        },
        step_hours=float(horizon_weeks) * HOURS_PER_WEEK,
    )
    metrics = compute_episode_metrics(sim)
    return {
        "ret_excel": float(metrics["ret_excel"]),
        "service_loss_auc": float(metrics["service_loss_auc_ration_hours"]),
        "lost_orders": float(metrics["lost_orders"]),
        "backlog_auc_proxy": float(metrics["backorder_qty_final"]),
        "flow_fill_rate": float(metrics["flow_fill_rate"]),
        "strategic_inventory": float(sum(buffers.values())),
    }


def write_design(path: Path, design: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([factor.name for factor in FACTORS])
        writer.writerows(design)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=int, default=20)
    parser.add_argument("--levels", type=int, default=8)
    parser.add_argument("--tapes", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=1_090_001)
    parser.add_argument("--horizon-weeks", type=int, default=52)
    parser.add_argument("--design-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/program_i/morris"))
    args = parser.parse_args()
    if args.seed_start >= 1_110_001:
        raise SystemExit("Refusing to open Program I confirmation/virgin universes")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    design, edges = morris_trajectories(FACTORS, args.trajectories, args.levels, 20260713)
    write_design(args.output_dir / "design.csv", design)
    design_sha = sha256((args.output_dir / "design.csv").read_bytes()).hexdigest()
    metadata = {
        "contract_id": "global_sensitivity_v1",
        "phase": "physical_sensitivity_only",
        "n_factors_exposed_now": len(FACTORS),
        "catalog_factors_not_exposed_are_not_screened": True,
        "design_rows": len(design),
        "tapes": args.tapes,
        "seed_start": args.seed_start,
        "confirmation_opened": False,
        "virgin_rl_opened": False,
        "rl_authorized": False,
    }
    if args.design_only:
        (args.output_dir / "design_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata, indent=2))
        return 0
    tapes = range(args.seed_start, args.seed_start + args.tapes)
    partial_path = args.output_dir / "partial_rows.jsonl"
    completed = {}
    if partial_path.exists():
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("design_sha256") != design_sha:
                raise RuntimeError("Checkpoint design hash does not match current frozen design")
            completed[int(row["design_index"])] = row
    tape_ids = tuple(tapes)
    for index, design_row in enumerate(design):
        if index in completed:
            continue
        params = {factor.name: float(value) for factor, value in zip(FACTORS, design_row)}
        tape_results = [run_des(params, tape, args.horizon_weeks) for tape in tape_ids]
        row = {
            "design_index": index,
            "design_sha256": design_sha,
            "metrics": {
                key: float(np.mean([result[key] for result in tape_results]))
                for key in tape_results[0]
            },
        }
        with partial_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        completed[index] = row
        print(f"[program-i] design {index + 1}/{len(design)} complete", flush=True)
    if set(completed) != set(range(len(design))):
        raise RuntimeError("Resumable screen is incomplete")
    metric_names = completed[0]["metrics"]
    outputs = {
        metric: np.asarray([completed[index]["metrics"][metric] for index in range(len(design))])
        for metric in metric_names
    }
    analyses = {metric: morris_effects(design, values, FACTORS, edges) for metric, values in outputs.items()}
    ranked = sorted(
        analyses["ret_excel"],
        key=lambda name: analyses["ret_excel"][name]["mu_star"], reverse=True,
    )
    verdict = {
        **metadata,
        "design_sha256": design_sha,
        "resumable_checkpoint": str(partial_path),
        "metrics": analyses,
        "ret_excel_sensitivity_ranking": ranked,
        "interpretation": "PHYSICAL_SENSITIVITY_ONLY_REQUIRES_COUNTERFACTUAL_BRANCHING",
        "promote_to_rl": False,
    }
    (args.output_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.output_dir / "outputs.npz", **outputs)
    print(json.dumps({"interpretation": verdict["interpretation"], "ranking": ranked}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
