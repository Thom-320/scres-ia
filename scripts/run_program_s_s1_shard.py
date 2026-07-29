#!/usr/bin/env python3
"""Produce one atomic Program S S1 matrix shard (cell x tape).

This launcher is intentionally inert until called with a reserved 751 seed.
Program Q retains VPS priority and this file itself authorizes no opening.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.paper2_exhaustive_search.program_s_design import build_program_s_risk_tape  # noqa: E402
from research.paper2_exhaustive_search.program_s_transducer import (  # noqa: E402
    extract_program_s_skeleton,
    run_program_s_direct,
)
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    MATRIX_KEYS,
    direct_full_des_vector,
    full_action_calendars,
    simulate_full_des_frontier,
)
from supply_chain.program_o_state_rich import (  # noqa: E402
    StateRichConfiguration,
    state_rich_calendar,
)
from supply_chain.program_s_risk_interaction import ProgramSCell  # noqa: E402


CONTRACT = json.loads(
    (ROOT / "contracts/program_s_product_mix_risk_interaction_gsa_v1.json").read_text()
)
PARENT = json.loads(
    (ROOT / "contracts/program_o_full_des_hpi_translation_v1.json").read_text()
)
DESIGN = json.loads(
    (ROOT / "research/paper2_exhaustive_search/program_s_native_morris_design_v1_1.json").read_text()
)
SCHEDULER = PARENT["action"]["within_week_schedulers"][
    PARENT["action"]["primary_scheduler"]
]


def calendar_index(calendar) -> int:
    value = 0
    for action in calendar:
        value = value * 4 + int(action)
    return int(value)


def resolve_point(group_id: int, trajectory_id: int, point_id: int) -> tuple[dict, dict]:
    group = DESIGN["groups"][int(group_id)]
    trajectory = group["trajectories"][int(trajectory_id)]
    point = trajectory["points"][int(point_id)]
    return group, {**point, "coupling": trajectory["coupling"]}


def make_cell(group: dict, point: dict, product_cell_id: str) -> ProgramSCell:
    product = DESIGN["product_cells"][product_cell_id]
    return ProgramSCell(
        stratum=group["stratum"],
        mask=group["mask"],
        coupling=point["coupling"],
        phi_by_risk=point["phi_by_risk"],
        psi_by_risk=point["psi_by_risk"],
        r14_probability_multiplier=point["r14_probability_multiplier"],
        baseline_capacity_multiplier=point["baseline_capacity_multiplier"],
        regime_persistence=product["rho"],
        dominant_share=product["share"],
        alarm_lead_hours=0,
        alarm_balanced_accuracy=0.5,
    )


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite shard: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--trajectory", type=int, required=True)
    parser.add_argument("--point", type=int, required=True)
    parser.add_argument("--product-cell", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not 7510001 <= args.seed <= 7510012:
        raise ValueError("S1 seed must be in the frozen 7510001-7510012 block")
    group, point = resolve_point(args.group, args.trajectory, args.point)
    if group["stratum"] != "THESIS_NATIVE_INDEPENDENT":
        raise RuntimeError("the S1 scientific runner accepts S-NATIVE only")
    current_cell = make_cell(group, point, args.product_cell)
    built = build_program_s_risk_tape(
        current_cell, tape_id=args.seed, horizon_hours=8 * 168
    )
    reference = run_program_s_direct(
        seed=args.seed,
        calendar=[2] * 8,
        scheduler=SCHEDULER,
        cell=current_cell,
        risk_event_tape=built["events"],
    )
    skeleton = extract_program_s_skeleton(reference)
    frontier = simulate_full_des_frontier(skeleton=skeleton, scheduler=SCHEDULER)
    calendar, decisions = state_rich_calendar(
        skeleton=skeleton.as_dict(),
        scheduler=SCHEDULER,
        config=StateRichConfiguration("belief_mpc", 4),
        regime_persistence=current_cell.regime_persistence,
        dominant_share=current_cell.dominant_share,
    )
    classical_index = calendar_index(calendar)
    oracle_index = int(np.argmax(frontier["ret_visible"]))
    replay_indices = sorted({0, 65535, classical_index, oracle_index})
    max_error = 0.0
    for index in replay_indices:
        direct_sim = run_program_s_direct(
            seed=args.seed,
            calendar=full_action_calendars()[index].tolist(),
            scheduler=SCHEDULER,
            cell=current_cell,
            risk_event_tape=built["events"],
        )
        direct = direct_full_des_vector(
            direct_sim, direct_sim.product_outcome_panel()
        )
        max_error = max(
            max_error,
            max(abs(float(direct[key]) - float(frontier[key][index])) for key in MATRIX_KEYS),
        )
    if max_error > 1e-10:
        raise AssertionError(f"risk-aware transducer replay error {max_error}")
    identity = (
        f"g{args.group:02d}__t{args.trajectory:02d}__p{args.point:02d}"
        f"__{args.product_cell}__seed{args.seed}"
    )
    shard = args.output_root / "matrices" / f"{identity}.npz"
    arrays = {key: np.asarray(frontier[key]) for key in MATRIX_KEYS}
    arrays.update(
        classical_calendar_index=np.asarray(classical_index, dtype=np.int32),
        classical_calendar=np.asarray(calendar, dtype=np.uint8),
        oracle_calendar_index=np.asarray(oracle_index, dtype=np.int32),
        risk_event_tape_sha256=np.asarray(built["event_tape_sha256"]),
        base_stream_sha256=np.asarray(built["base_stream_sha256"]),
        skeleton_sha256=np.asarray(skeleton.skeleton_sha256),
        cell_id=np.asarray(current_cell.cell_id),
        observation_sha256=np.asarray(
            [decision.observation.observation_sha256 for decision in decisions]
        ),
        direct_replay_max_abs_error=np.asarray(max_error),
    )
    atomic_npz(shard, arrays)
    print(
        json.dumps(
            {
                "identity": identity,
                "shard": str(shard),
                "cell_id": current_cell.cell_id,
                "classical_index": classical_index,
                "oracle_index": oracle_index,
                "direct_replay_max_abs_error": max_error,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
