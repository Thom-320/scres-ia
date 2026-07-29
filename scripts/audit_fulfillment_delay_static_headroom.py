#!/usr/bin/env python3
"""Custody the *static-library* headroom sensitivity to fulfilment delay.

This diagnostic enumerates the full 216-posture buffer library on already-open
development roots. Its headroom is the value of choosing one fixed posture per
tape with hindsight. It is not an epoch-level dynamic oracle and cannot authorize
or rule out MPC, RL, or neural premium.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.expanded_contract_controllers_v2 import (  # noqa: E402
    ALL_POSTURES,
    posture_name,
)
from scripts.run_expanded_contract_comparators_v2 import (  # noqa: E402
    apply_posture,
    make_replay_sim,
    materialize_tape,
)


def canonical_sha(payload: Any) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run_cell(job: dict[str, Any]) -> dict[str, Any]:
    tape = job["tape"]
    horizon = float(job["horizon"])
    epoch_hours = float(job["epoch_hours"])
    posture = tuple(job["posture"])
    delay = float(job["delay"])
    sim = make_replay_sim(
        seed=int(tape["seed"]),
        horizon=horizon,
        family="R1r",
        tape=tape,
    )
    sim.demand_on_hand_fulfillment_delay = delay
    while float(sim.env.now) < horizon - 1e-9:
        apply_posture(sim, posture)
        sim.step(
            action=None,
            step_hours=min(epoch_hours, horizon - float(sim.env.now)),
        )
    metric = compute_episode_metrics(sim)
    return {
        "delay": delay,
        "tape_seed": int(tape["seed"]),
        "posture": list(posture),
        "posture_name": posture_name(posture),
        "ret_excel": float(metric["ret_excel"]),
        "flow_fill_rate": float(metric["flow_fill_rate"]),
        "lost_orders": float(metric["lost_orders"]),
        "delivered_rations": float(metric["delivered_rations"]),
        "unresolved": float(
            metric.get("unresolved_orders", metric.get("unresolved", 0.0))),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    roots = sorted({int(row["tape_seed"]) for row in rows})
    names = sorted({str(row["posture_name"]) for row in rows})
    by_name = {
        name: [row for row in rows if row["posture_name"] == name]
        for name in names
    }
    mean_by_name = {
        name: float(np.mean([row["ret_excel"] for row in values]))
        for name, values in by_name.items()
    }
    incumbent = min(
        mean_by_name,
        key=lambda name: (-mean_by_name[name], name),
    )
    best_by_tape: dict[int, dict[str, Any]] = {}
    for root in roots:
        candidates = [row for row in rows if row["tape_seed"] == root]
        candidates.sort(key=lambda row: (-row["ret_excel"], row["posture_name"]))
        best_by_tape[root] = candidates[0]
    pi_values = [best_by_tape[root]["ret_excel"] for root in roots]
    incumbent_values = [
        next(row["ret_excel"] for row in by_name[incumbent]
             if row["tape_seed"] == root)
        for root in roots
    ]
    deltas = np.asarray(pi_values) - np.asarray(incumbent_values)
    return {
        "roots": roots,
        "posture_count": len(names),
        "incumbent": incumbent,
        "incumbent_mean": mean_by_name[incumbent],
        "static_tape_selection_headroom_mean": float(deltas.mean()),
        "static_tape_selection_positive_tapes": int((deltas > 0).sum()),
        "static_tape_selection_deltas": deltas.tolist(),
        "distinct_best_postures_by_tape": sorted({
            best_by_tape[root]["posture_name"] for root in roots
        }),
        "ret_excel_grid_span": (
            max(mean_by_name.values()) - min(mean_by_name.values())),
        "interpretation": (
            "Perfect-hindsight selection of one fixed posture per tape within "
            "the enumerated 216-posture library; not within-tape dynamic headroom."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delays", nargs="+", type=float, default=[54.0, 47.0])
    parser.add_argument("--root-start", type=int, default=1_430_001)
    parser.add_argument("--tapes", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--horizon-weeks", type=int, default=52)
    parser.add_argument("--epoch-weeks", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/metric_audit/"
            "fulfillment_delay_static_headroom_v1/result.json"),
    )
    args = parser.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    epoch_hours = float(args.epoch_weeks * HOURS_PER_WEEK)
    roots = [args.root_start + i for i in range(args.tapes)]
    tapes = [materialize_tape(root, horizon, "R1r") for root in roots]
    jobs = [
        {
            "delay": delay,
            "tape": tape,
            "posture": list(posture),
            "horizon": horizon,
            "epoch_hours": epoch_hours,
        }
        for delay in args.delays
        for posture in ALL_POSTURES
        for tape in tapes
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        rows = list(pool.map(run_cell, jobs, chunksize=8))
    payload = {
        "schema_version": "fulfillment_delay_static_headroom_v1",
        "claim_status": "DEVELOPMENT_STATIC_LIBRARY_DIAGNOSTIC",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_code_note": (
            "RPj immutable-onset corrective present; git_commit alone does not "
            "identify uncommitted code, so script and result hashes are required"),
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "simulation_sha256": sha256(
            Path("supply_chain/supply_chain.py").read_bytes()).hexdigest(),
        "confirmation_roots_opened": False,
        "roots_are_previously_opened_v2_development_roots": True,
        "family": "R1r",
        "posture_count": len(ALL_POSTURES),
        "tapes": len(roots),
        "epoch_hours": epoch_hours,
        "results_by_delay": {
            str(delay): summarize(
                [row for row in rows if row["delay"] == delay])
            for delay in args.delays
        },
        "neural_authorization": False,
        "dynamic_headroom_adjudicated": False,
        "interpretation_boundary": (
            "A zero result bounds only tape-level selection among fixed postures. "
            "It does not bound state-contingent changes between epochs."),
        "rows_sha256": canonical_sha(rows),
    }
    payload["self_sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    args.output.with_name("rows.json").write_text(
        json.dumps(rows, indent=1, sort_keys=True) + "\n")
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
