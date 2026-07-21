#!/usr/bin/env python3
"""Adjudicate burned Q2 warm-start and QR-DQN probes on matched tapes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    extract_full_des_skeleton,
    simulate_full_des_frontier,
)
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS  # noqa: E402


def _hierarchical_ci(rows: list[dict], metric: str, *, draws: int, seed: int) -> list[float]:
    optimizer_seeds = sorted({int(row["optimizer_seed"]) for row in rows})
    tapes = sorted({int(row["tape"]) for row in rows})
    cells = sorted({str(row["cell"]) for row in rows})
    index = {(int(row["optimizer_seed"]), int(row["tape"]), str(row["cell"])): float(row[metric]) for row in rows}
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled_seeds = rng.choice(optimizer_seeds, len(optimizer_seeds), replace=True)
        sampled_tapes = rng.choice(tapes, len(tapes), replace=True)
        estimates[draw] = np.mean([
            index[(int(optimizer_seed), int(tape), cell)]
            for optimizer_seed in sampled_seeds
            for tape in sampled_tapes
            for cell in cells
        ])
    return [float(value) for value in np.quantile(estimates, (0.025, 0.5, 0.975))]


def _summary(rows: list[dict], *, draws: int, seed: int) -> dict:
    deltas = np.asarray([row["delta_vs_structured"] for row in rows], dtype=float)
    result = {
        "n": len(rows),
        "mean_delta_vs_static": float(np.mean([row["delta_vs_static"] for row in rows])),
        "mean_delta_vs_structured": float(np.mean(deltas)),
        "hierarchical_ci95_vs_structured": _hierarchical_ci(rows, "delta_vs_structured", draws=draws, seed=seed),
        "favorable_vs_structured": float(np.mean(deltas > 0.0)),
        "mean_worst_product_delta_vs_structured": float(np.mean([row["worst_product_delta_vs_structured"] for row in rows])),
        "lost_orders_max": float(max(row["lost_orders"] for row in rows)),
        "resource_spread_max": float(max(row["resource_spread"] for row in rows)),
    }
    result["by_cell"] = {}
    for cell in sorted({row["cell"] for row in rows}):
        members = [row for row in rows if row["cell"] == cell]
        result["by_cell"][cell] = {
            "mean_delta_vs_structured": float(np.mean([row["delta_vs_structured"] for row in members])),
            "favorable_vs_structured": float(np.mean([row["delta_vs_structured"] > 0.0 for row in members])),
            "mean_worst_product_delta_vs_structured": float(np.mean([row["worst_product_delta_vs_structured"] for row in members])),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmstart", type=Path, default=ROOT / "results/program_q2/dynamic_warmstart_probe_v1/result.json")
    parser.add_argument("--qrdqn-glob", default=str(ROOT / "results/program_q2/qrdqn_dynamic_probe_v1/seed_*/result.json"))
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    warm = json.loads(args.warmstart.read_text())
    final_scratch = next(run for run in warm["runs"] if run["arm"] == "scratch")
    comparator_rows = {
        (row["cell"], int(row["tape"])): row
        for row in final_scratch["checkpoints"][-1]["rows"]
    }
    warm_rows = []
    for run in warm["runs"]:
        for row in run["checkpoints"][-1]["rows"]:
            warm_rows.append({**row, "arm": run["arm"], "optimizer_seed": int(run["optimizer_seed"])})

    qrdqn_rows = []
    parity_max = 0.0
    for result_path in sorted(glob.glob(args.qrdqn_glob)):
        result = json.loads(Path(result_path).read_text())
        optimizer_seed = int(result["optimizer_seed"])
        for row in result["rows"]:
            cell_index = next(index for index, cell in enumerate(CONFIRMED_RET_CELLS) if cell.cell_id == row["cell"])
            cell = CONFIRMED_RET_CELLS[cell_index]
            tape = int(row["tape"])
            comparator = comparator_rows[(cell.cell_id, tape)]
            skeleton, _sim = extract_full_des_skeleton(
                seed=tape,
                scheduler=scheduler(),
                regime_persistence=float(cell.regime_persistence),
                dominant_share=float(cell.dominant_share),
                downstream_freight_physics_mode="fixed_clock_physical_v1",
            )
            panel = simulate_full_des_frontier(
                skeleton=skeleton,
                scheduler=scheduler(),
                calendars=np.asarray([
                    row["calendar"],
                    comparator["static_calendar"],
                    comparator["structured_calendar"],
                ], dtype=np.uint8),
            )
            parity_max = max(parity_max, abs(float(panel["ret_visible"][0]) - float(row["ret_visible"])))
            qrdqn_rows.append({
                "optimizer_seed": optimizer_seed,
                "cell": cell.cell_id,
                "tape": tape,
                "calendar": row["calendar"],
                "ret_visible": float(panel["ret_visible"][0]),
                "delta_vs_static": float(panel["ret_visible"][0] - panel["ret_visible"][1]),
                "delta_vs_structured": float(panel["ret_visible"][0] - panel["ret_visible"][2]),
                "worst_product_delta_vs_structured": float(panel["worst_product_fill"][0] - panel["worst_product_fill"][2]),
                "lost_orders": float(panel["lost_orders"][0]),
                "resource_spread": float(max(panel["gross_policy_batch_slots"]) - min(panel["gross_policy_batch_slots"])),
            })

    warm_summaries = {
        arm: _summary([row for row in warm_rows if row["arm"] == arm], draws=args.bootstrap_draws, seed=20260721 + index)
        for index, arm in enumerate(("scratch", "static_bc", "structured_bc"))
    }
    qrdqn_summary = _summary(qrdqn_rows, draws=args.bootstrap_draws, seed=20260731)
    payload = {
        "schema_version": "program_q2_dynamic_probe_adjudication_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_status": "EXPLORATORY_NO_CLAIM",
        "warmstart_verdict": "STOP_WARMSTART_NO_STRUCTURED_PREMIUM",
        "qrdqn_verdict": "STOP_QRDQN_NOT_COMPETITIVE",
        "warmstart": warm_summaries,
        "qrdqn": qrdqn_summary,
        "qrdqn_direct_replay_parity_max_abs": parity_max,
        "qrdqn_rows": qrdqn_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "qrdqn_rows"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
