#!/usr/bin/env python3
"""Prospective U1 Stage-A screen on fresh tapes.

The failed Program S run is immutable.  This runner opens a disjoint seed block
and evaluates only the two masks whose transducer is independently certified.
The R14/production mask is stopped before screening and never reaches this
runner.  Three tapes per point are used because a one-tape sample-average
perfect-information gap is identically zero when its static comparator is
reselected on that same tape.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.paper2_exhaustive_search.program_s_design import build_program_s_risk_tape
from research.paper2_exhaustive_search.program_s_transducer import (
    extract_program_s_skeleton,
    run_program_s_direct,
)
from scripts.run_program_s_s1_shard import make_cell, resolve_point
from scripts.screen_program_o_full_des_hpi import profile_summary
from supply_chain.program_o_full_des_transducer import (
    MATRIX_KEYS,
    direct_full_des_vector,
    full_action_calendars,
    simulate_full_des_frontier,
)
from supply_chain.program_o_state_rich import StateRichConfiguration, state_rich_calendar
from supply_chain.program_t_full_des_mpc import FullDEST0Config, ret_transducer_t0_calendar


DESIGN_PATH = ROOT / "research/paper2_exhaustive_search/program_s_native_morris_design_v1_1.json"
CONTRACT_PATH = ROOT / "contracts/program_s_product_mix_risk_interaction_gsa_v1.json"
PARENT_PATH = ROOT / "contracts/program_o_full_des_hpi_translation_v1.json"
FRESH_SEEDS = (7_540_001, 7_540_002, 7_540_003)
ALLOWED_GROUPS = (1, 2)
T0_CONFIG = FullDEST0Config(horizon=3, mode="scenario", particles=4)


def _scheduler() -> dict[str, list[str]]:
    parent = json.loads(PARENT_PATH.read_text())
    key = parent["action"]["primary_scheduler"]
    return parent["action"]["within_week_schedulers"][key]


def point_id(group: int, trajectory: int, point: int, product_cell: str) -> str:
    return f"g{group:02d}__t{trajectory:02d}__p{point:02d}__{product_cell}"


def tasks() -> list[tuple[int, int, int, str, int]]:
    design = json.loads(DESIGN_PATH.read_text())
    rows: list[tuple[int, int, int, str, int]] = []
    for group_index in ALLOWED_GROUPS:
        group = design["groups"][group_index]
        if group["mask"] not in {"LOC_SURGE", "CROSS_ECHELON_SURGE"}:
            raise AssertionError("U1 Stage A may not route an uncertified mask")
        for trajectory_index, trajectory in enumerate(group["trajectories"]):
            for point_index, point in enumerate(trajectory["points"]):
                if float(point["baseline_capacity_multiplier"]) != 1.0:
                    raise AssertionError("U1 may not select baseline environment capacity")
                for product_cell in sorted(design["product_cells"]):
                    for seed in FRESH_SEEDS:
                        rows.append((group_index, trajectory_index, point_index, product_cell, seed))
    if len(rows) != 900 or len(set(rows)) != 900:
        raise AssertionError(f"unexpected U1 task family: {len(rows)}")
    return rows


def _calendar_index(calendar: tuple[int, ...] | list[int]) -> int:
    value = 0
    for action in calendar:
        value = value * 4 + int(action)
    return value


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_one(task: tuple[int, int, int, str, int], output_root: str) -> dict[str, object]:
    group_id, trajectory_id, point_index, product_cell, seed = task
    group, point = resolve_point(group_id, trajectory_id, point_index)
    cell = make_cell(group, point, product_cell)
    if cell.mask not in {"LOC_SURGE", "CROSS_ECHELON_SURGE"}:
        raise RuntimeError("uncertified mask reached U1 transducer route")
    built = build_program_s_risk_tape(cell, tape_id=seed, horizon_hours=8 * 168)
    scheduler = _scheduler()
    reference = run_program_s_direct(
        seed=seed,
        calendar=[2] * 8,
        scheduler=scheduler,
        cell=cell,
        risk_event_tape=built["events"],
    )
    skeleton = extract_program_s_skeleton(reference)
    frontier = simulate_full_des_frontier(skeleton=skeleton, scheduler=scheduler)
    classical_calendar, _ = state_rich_calendar(
        skeleton=skeleton.as_dict(),
        scheduler=scheduler,
        config=StateRichConfiguration("belief_mpc", 4),
        regime_persistence=cell.regime_persistence,
        dominant_share=cell.dominant_share,
    )
    reinforced_calendar, reinforced_diagnostics = ret_transducer_t0_calendar(
        skeleton=skeleton,
        scheduler=scheduler,
        config=FullDEST0Config(
            horizon=T0_CONFIG.horizon,
            mode=T0_CONFIG.mode,
            particles=T0_CONFIG.particles,
            regime_persistence=cell.regime_persistence,
            dominant_share=cell.dominant_share,
        ),
    )
    classical_index = _calendar_index(classical_calendar)
    reinforced_index = _calendar_index(reinforced_calendar)
    oracle_index = int(np.argmax(frontier["ret_visible"]))
    replay_indices = sorted({0, 65_535, classical_index, reinforced_index, oracle_index})
    max_error = 0.0
    for index in replay_indices:
        direct_sim = run_program_s_direct(
            seed=seed,
            calendar=full_action_calendars()[index].tolist(),
            scheduler=scheduler,
            cell=cell,
            risk_event_tape=built["events"],
        )
        direct = direct_full_des_vector(direct_sim, direct_sim.product_outcome_panel())
        max_error = max(
            max_error,
            max(abs(float(direct[key]) - float(frontier[key][index])) for key in MATRIX_KEYS),
        )
    if max_error > 1e-10:
        raise AssertionError(f"U1 certified-mask replay error {max_error}")
    identity = point_id(group_id, trajectory_id, point_index, product_cell)
    path = Path(output_root) / "matrices" / f"{identity}__seed{seed}.npz"
    arrays = {key: np.asarray(frontier[key]) for key in MATRIX_KEYS}
    arrays.update(
        classical_calendar_index=np.asarray(classical_index, dtype=np.int32),
        reinforced_calendar_index=np.asarray(reinforced_index, dtype=np.int32),
        classical_calendar=np.asarray(classical_calendar, dtype=np.uint8),
        reinforced_calendar=np.asarray(reinforced_calendar, dtype=np.uint8),
        oracle_calendar_index=np.asarray(oracle_index, dtype=np.int32),
        direct_replay_max_abs_error=np.asarray(max_error),
        cell_id=np.asarray(cell.cell_id),
        mask=np.asarray(cell.mask),
        risk_event_tape_sha256=np.asarray(built["event_tape_sha256"]),
        skeleton_sha256=np.asarray(skeleton.skeleton_sha256),
        reinforced_online_ms=np.asarray(float(reinforced_diagnostics["online_ms"])),
    )
    _atomic_npz(path, arrays)
    return {"identity": identity, "seed": seed, "path": str(path), "max_error": max_error}


def reduce(output_root: Path) -> dict[str, object]:
    contract = json.loads(CONTRACT_PATH.read_text())
    design = json.loads(DESIGN_PATH.read_text())
    rows = []
    for group_id in ALLOWED_GROUPS:
        group = design["groups"][group_id]
        for trajectory_id, trajectory in enumerate(group["trajectories"]):
            for point_index, point in enumerate(trajectory["points"]):
                for product_cell in sorted(design["product_cells"]):
                    identity = point_id(group_id, trajectory_id, point_index, product_cell)
                    paths = [output_root / "matrices" / f"{identity}__seed{seed}.npz" for seed in FRESH_SEEDS]
                    panel = {key: [] for key in MATRIX_KEYS}
                    classical_indices = []
                    reinforced_indices = []
                    online_ms = []
                    replay_errors = []
                    for path in paths:
                        with np.load(path, allow_pickle=False) as shard:
                            for key in MATRIX_KEYS:
                                panel[key].append(np.asarray(shard[key]))
                            classical_indices.append(int(shard["classical_calendar_index"]))
                            reinforced_indices.append(int(shard["reinforced_calendar_index"]))
                            online_ms.append(float(shard["reinforced_online_ms"]))
                            replay_errors.append(float(shard["direct_replay_max_abs_error"]))
                    stacked = {key: np.stack(values) for key, values in panel.items()}
                    summary = profile_summary(stacked, contract)
                    tape_index = np.arange(len(FRESH_SEEDS))
                    static_index = int(summary["best_static_calendar_index"])
                    static = stacked["ret_visible"][:, static_index]
                    classical = stacked["ret_visible"][tape_index, classical_indices]
                    reinforced = stacked["ret_visible"][tape_index, reinforced_indices]
                    rows.append({
                        "point_id": identity,
                        "group": group_id,
                        "trajectory": trajectory_id,
                        "point": point_index,
                        "mask": group["mask"],
                        "product_cell": product_cell,
                        "physical": point["physical"],
                        "safe_h_pi_mean": float(summary["safe_h_pi"]),
                        "classical_h_obs_mean": float(np.mean(classical - static)),
                        "residual_classical_minus_reinforced_mean": float(np.mean(classical - reinforced)),
                        "reinforced_minus_static_mean": float(np.mean(reinforced - static)),
                        "reinforced_online_ms_mean": float(np.mean(online_ms)),
                        "maximum_direct_replay_abs_error": float(max(replay_errors)),
                    })
    promoted = [
        row for row in rows
        if row["safe_h_pi_mean"] >= 0.02
        and row["classical_h_obs_mean"] >= 0.015
        and row["maximum_direct_replay_abs_error"] <= 1e-10
    ]
    promoted.sort(
        key=lambda row: (
            sum(abs(float(v) - 1.0) for v in row["physical"].values() if isinstance(v, (int, float))),
            row["point_id"],
        )
    )
    payload = {
        "schema_version": "program_u1_native_risk_stage_a_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_U1_STAGE_A_CANDIDATES" if promoted else "STOP_U1_NATIVE_NO_PRELIMINARY_HEADROOM",
        "fresh_seed_block": list(FRESH_SEEDS),
        "production_quality_route": "STOP_MASK_BEFORE_INEXACT_SCREEN",
        "evaluated_masks": ["LOC_SURGE", "CROSS_ECHELON_SURGE"],
        "point_count": len(rows),
        "task_count": 900,
        "preliminary_thresholds": {"safe_h_pi_mean": 0.02, "classical_h_obs_mean": 0.015},
        "promoted_point_ids": [row["point_id"] for row in promoted[:12]],
        "promotion_count": min(12, len(promoted)),
        "rows": rows,
        "claim_limit": "Three-tape development routing only; no confidence bound, connected region, U2 authorization, or Paper 2 claim.",
    }
    result_path = output_root / "result.json"
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, result_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    manifest = {
        "schema_version": "program_u1_native_risk_stage_a_launch_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "fresh_seed_block": list(FRESH_SEEDS),
        "task_count": len(tasks()),
        "workers": int(args.workers),
        "design_sha256": hashlib.sha256(DESIGN_PATH.read_bytes()).hexdigest(),
        "production_quality_route": "STOP_MASK_BEFORE_INEXACT_SCREEN",
    }
    (args.output_root / "launch_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    completed = 0
    started = datetime.now(timezone.utc).isoformat()
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_one, task, str(args.output_root)): task for task in tasks()}
            for future in as_completed(futures):
                future.result()
                completed += 1
                if completed % 10 == 0:
                    progress = {"status": "RUNNING", "started_at": started, "completed": completed, "expected": 900}
                    (args.output_root / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    except BaseException as error:
        receipt = {"status": "FAILED", "completed": completed, "expected": 900, "failure": repr(error)}
        (args.output_root / "producer_exit.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        raise
    payload = reduce(args.output_root)
    receipt = {"status": "COMPLETE", "completed": completed, "expected": 900, "verdict": payload["status"]}
    (args.output_root / "producer_exit.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "promotion_count": payload["promotion_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
