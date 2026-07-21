#!/usr/bin/env python3
"""Run the C/D action diagnostics on already-burned Program Q tapes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.program_o_full_des_transducer import extract_full_des_skeleton  # noqa: E402
from supply_chain.program_o_state_rich import StateRichConfiguration, state_rich_calendar  # noqa: E402
from supply_chain.program_q2_action_discovery import (  # noqa: E402
    CENTERED_ACTIONS,
    count4_calendar_to_sequence8,
    greedy_full_rollout,
    perfect_information_timing_identity,
    pulse_branching,
)


CELLS = {
    "rho90_share90": (0.90, 0.90),
    "rho75_share90": (0.75, 0.90),
}


def run(args: argparse.Namespace) -> dict:
    rho, share = CELLS[args.cell]
    sched4 = scheduler()
    state_rows = []
    rollout_rows = []
    for seed in range(args.seed_start, args.seed_start + args.tapes):
        skeleton, _sim = extract_full_des_skeleton(
            seed=seed,
            scheduler=sched4,
            regime_persistence=rho,
            dominant_share=share,
            downstream_freight_physics_mode="fixed_clock_physical_v1",
        )
        baseline4, _decisions = state_rich_calendar(
            skeleton=skeleton.as_dict(),
            scheduler=sched4,
            config=StateRichConfiguration("belief_mpc", 3),
            regime_persistence=rho,
            dominant_share=share,
        )
        baseline8 = count4_calendar_to_sequence8(baseline4)
        pulse = pulse_branching(skeleton=skeleton, baseline_sequence8=baseline8)
        for row in pulse["states"]:
            state_rows.append({"seed": seed, **row})
        calendar8, result8 = greedy_full_rollout(
            skeleton=skeleton,
            baseline_sequence8=baseline8,
            allowed_actions=tuple(range(8)),
        )
        calendar4, result4 = greedy_full_rollout(
            skeleton=skeleton,
            baseline_sequence8=baseline8,
            allowed_actions=CENTERED_ACTIONS,
        )
        rollout_rows.append(
            {
                "seed": seed,
                "greedy8_calendar": list(calendar8),
                "greedy4_calendar": list(calendar4),
                "ret_gain": result8["ret_visible"] - result4["ret_visible"],
                "worst_product_fill_delta": result8["worst_product_fill"] - result4["worst_product_fill"],
                "programmed_resource_error": max(
                    abs(result8[key] - result4[key])
                    for key in (
                        "gross_policy_batch_slots",
                        "gross_production_quantity",
                        "charged_daily_dispatch_slots",
                        "charged_downstream_vehicle_hours",
                    )
                ),
            }
        )
    omitted_fraction = float(np.mean([row["omitted_action_optimal"] for row in state_rows]))
    mean_pulse_gain = float(np.mean([row["ret_gain"] for row in state_rows]))
    mean_rollout_gain = float(np.mean([row["ret_gain"] for row in rollout_rows]))
    pass_c = omitted_fraction >= 0.15 and mean_pulse_gain >= 0.015
    timing = perfect_information_timing_identity()
    payload = {
        "schema_version": "program_q2_action_discovery_result_v1",
        "status": "EXPLORATORY_NO_CLAIM",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cell": args.cell,
        "burned_tapes": [args.seed_start, args.seed_start + args.tapes - 1],
        "n_states": len(state_rows),
        "omitted_action_optimal_fraction": omitted_fraction,
        "mean_pulse_ret_gain": mean_pulse_gain,
        "mean_greedy_full_rollout_ret_gain": mean_rollout_gain,
        "max_programmed_resource_error": max(row["programmed_resource_error"] for row in rollout_rows),
        "C_verdict": "PASS_SEQUENCE8_ACTION_HEADROOM" if pass_c else "STOP_NO_SEQUENCE8_ACTION_HEADROOM",
        "D_timing_identification": timing,
        "D_verdict": "STOP_PI_TIMING_NOT_IDENTIFIABLE_WITH_EQUIVALENT_FEASIBLE_SETS",
    }
    if args.include_rows:
        payload["state_rows"] = state_rows
        payload["rollout_rows"] = rollout_rows
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", default="rho90_share90", choices=sorted(CELLS))
    parser.add_argument("--seed-start", type=int, default=7_480_001)
    parser.add_argument("--tapes", type=int, default=24)
    parser.add_argument("--include-rows", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results/program_q2/action_discovery_v1/result.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("C_verdict", "D_verdict", "omitted_action_optimal_fraction", "mean_pulse_ret_gain", "mean_greedy_full_rollout_ret_gain")}, indent=2))


if __name__ == "__main__":
    main()
