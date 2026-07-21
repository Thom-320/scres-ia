#!/usr/bin/env python3
"""Run A/B observation diagnostics on burned Program Q tapes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from supply_chain.program_q2_action_discovery import count4_calendar_to_sequence8, pulse_branching  # noqa: E402
from supply_chain.program_q2_observation_discovery import (  # noqa: E402
    Q21_NAMES,
    cross_tape_nearest_regret,
    exact_collision_summary,
    make_audit_row,
)


def run(args: argparse.Namespace) -> dict:
    rho, share = (0.90, 0.90)
    sched = scheduler()
    rows = []
    for seed in range(args.seed_start, args.seed_start + args.tapes):
        skeleton, _sim = extract_full_des_skeleton(
            seed=seed,
            scheduler=sched,
            regime_persistence=rho,
            dominant_share=share,
            downstream_freight_physics_mode="fixed_clock_physical_v1",
        )
        baseline4, decisions = state_rich_calendar(
            skeleton=skeleton.as_dict(),
            scheduler=sched,
            config=StateRichConfiguration("belief_mpc", 3),
            regime_persistence=rho,
            dominant_share=share,
        )
        pulse = pulse_branching(
            skeleton=skeleton,
            baseline_sequence8=count4_calendar_to_sequence8(baseline4),
        )
        for decision, target in zip(decisions, pulse["states"], strict=True):
            rows.append(
                make_audit_row(
                    seed=seed,
                    skeleton=skeleton,
                    observation=decision.observation,
                    best_action=target["best8_action"],
                    action_ret=target["action_ret"],
                )
            )
    preclip = np.asarray([row.q21_preclip for row in rows], dtype=float)
    clipped = np.asarray([row.q21_clipped for row in rows], dtype=float)
    feature_rows = []
    for index, name in enumerate(Q21_NAMES):
        feature_rows.append(
            {
                "feature": name,
                "preclip_min": float(preclip[:, index].min()),
                "preclip_p50": float(np.median(preclip[:, index])),
                "preclip_p95": float(np.quantile(preclip[:, index], 0.95)),
                "preclip_max": float(preclip[:, index].max()),
                "fraction_at_zero_after_clip": float(np.mean(clipped[:, index] == 0.0)),
                "fraction_at_one_after_clip": float(np.mean(clipped[:, index] == 1.0)),
                "fraction_above_one_before_clip": float(np.mean(preclip[:, index] > 1.0)),
            }
        )
    q21_regret = cross_tape_nearest_regret(rows, "q21_clipped")
    rich_regret = cross_tape_nearest_regret(rows, "rich")
    attributable = q21_regret - rich_regret
    promote = attributable >= 0.015
    payload = {
        "schema_version": "program_q2_observation_discovery_result_v1",
        "status": "EXPLORATORY_NO_CLAIM",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "burned_tapes": [args.seed_start, args.seed_start + args.tapes - 1],
        "n_states": len(rows),
        "feature_saturation": feature_rows,
        "collision_summary": exact_collision_summary(rows),
        "q21_cross_tape_nearest_regret": q21_regret,
        "rich_cross_tape_nearest_regret": rich_regret,
        "regret_attributable_to_q21_compression": attributable,
        "verdict": "PASS_OBSERVATION_ALIASING_HEADROOM" if promote else "STOP_OBSERVATION_ALIASING_BELOW_THRESHOLD",
        "full_state_probe_authorized": promote,
    }
    if args.include_rows:
        payload["rows"] = [asdict(row) for row in rows]
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=7_480_001)
    parser.add_argument("--tapes", type=int, default=24)
    parser.add_argument("--include-rows", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results/program_q2/observation_discovery_v1/result.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("verdict", "q21_cross_tape_nearest_regret", "rich_cross_tape_nearest_regret", "regret_attributable_to_q21_compression", "full_state_probe_authorized")}, indent=2))


if __name__ == "__main__":
    main()
