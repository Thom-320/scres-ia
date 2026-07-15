#!/usr/bin/env python3
"""Burned-tape parity audit for Program O's physical fixed-clock extension."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supply_chain.program_o_full_des import run_program_o_full_des_episode
from supply_chain.program_o_full_des_transducer import extract_full_des_skeleton
from supply_chain.program_o_state_rich import (
    StateRichConfiguration,
    state_rich_calendar,
)


CONTRACT_PATH = ROOT / "contracts/program_o_fixed_clock_physical_hobs_validation_v1.json"
PARENT_CONTRACT_PATH = ROOT / "contracts/program_o_full_des_hpi_translation_v1.json"
CELLS = {
    "rho75_share90": (0.75, 0.90),
    "rho90_share75": (0.90, 0.75),
    "rho90_share90": (0.90, 0.90),
}
SEEDS = tuple(range(7420001, 7420049))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/program_o/fixed_clock_physical_preflight_v1/result.json",
    )
    args = parser.parse_args()

    parent = json.loads(PARENT_CONTRACT_PATH.read_text())
    scheduler = parent["action"]["within_week_schedulers"][
        parent["action"]["primary_scheduler"]
    ]
    config = StateRichConfiguration("belief_mpc", 3)
    failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    scheduled_resource_vectors: set[tuple[float, ...]] = set()

    for cell_id, (rho, share) in CELLS.items():
        for seed in SEEDS:
            skeleton, _ = extract_full_des_skeleton(
                seed=seed,
                scheduler=scheduler,
                regime_persistence=rho,
                dominant_share=share,
            )
            calendar, decisions = state_rich_calendar(
                skeleton=skeleton.as_dict(),
                scheduler=scheduler,
                config=config,
                regime_persistence=rho,
                dominant_share=share,
            )
            base_sim, base = run_program_o_full_des_episode(
                seed=seed,
                calendar=calendar,
                scheduler=scheduler,
                regime_persistence=rho,
                dominant_share=share,
                downstream_freight_physics_mode="loaded_only",
            )
            physical_sim, physical = run_program_o_full_des_episode(
                seed=seed,
                calendar=calendar,
                scheduler=scheduler,
                regime_persistence=rho,
                dominant_share=share,
                downstream_freight_physics_mode="fixed_clock_physical_v1",
            )
            parity = {
                "metrics": physical["metrics"] == base["metrics"],
                "products": physical["products"] == base["products"],
                "worst_product_fill": physical["worst_product_fill"]
                == base["worst_product_fill"],
                "conservation": physical["conservation"] == base["conservation"],
                "aggregate_state_hash": physical["aggregate_state_hash"]
                == base["aggregate_state_hash"],
                "action_sequence": tuple(calendar)
                == tuple(event["action"] for event in physical_sim.program_o_action_events if event["status"] == "requested"),
            }
            resources = physical["resources"]
            scheduled_vector = (
                resources["committed_action_batch_slots"],
                resources["gross_action_production_quantity"],
                resources["scheduled_downstream_missions"],
                resources["scheduled_downstream_vehicle_hours"],
                resources["scheduled_downstream_crew_hours"],
                resources["scheduled_payload_capacity"],
                resources["setup_hours"],
            )
            scheduled_resource_vectors.add(scheduled_vector)
            physical_checks = {
                "scheduled_missions_112": resources["scheduled_downstream_missions"]
                == 112,
                "loaded_plus_empty_112": resources["actual_loaded_departures"]
                + resources["empty_downstream_missions"]
                == 112,
                "vehicle_hours_5376": resources[
                    "scheduled_downstream_vehicle_hours"
                ]
                == 5376,
                "crew_hours_5376": resources["scheduled_downstream_crew_hours"]
                == 5376,
                "payload_capacity_291200": resources["scheduled_payload_capacity"]
                == 291200,
                "product_residual": physical["conservation"][
                    "max_abs_product_residual"
                ]
                <= 1e-8,
                "partition_residual": physical["conservation"][
                    "max_abs_partition_residual"
                ]
                <= 1e-8,
            }
            if not all(parity.values()) or not all(physical_checks.values()):
                failures.append(
                    {
                        "cell_id": cell_id,
                        "seed": seed,
                        "parity": parity,
                        "physical_checks": physical_checks,
                    }
                )
            rows.append(
                {
                    "cell_id": cell_id,
                    "seed": seed,
                    "calendar": list(calendar),
                    "observation_hashes": [
                        decision.observation.observation_sha256 for decision in decisions
                    ],
                    "ret_excel": physical["metrics"]["ret_excel"],
                    "loaded_departures": resources["actual_loaded_departures"],
                    "empty_missions": resources["empty_downstream_missions"],
                    "scheduled_resource_vector": list(scheduled_vector),
                    "parity": parity,
                    "physical_checks": physical_checks,
                    "base_actual_payload": base_sim.program_o_actual_payload,
                    "physical_actual_payload": physical_sim.program_o_actual_payload,
                }
            )

    passed = not failures and len(scheduled_resource_vectors) == 1
    result = {
        "schema_version": "program_o_fixed_clock_physical_preflight_v1",
        "status": (
            "PASS_PROGRAM_O_FIXED_CLOCK_PHYSICAL_PREFLIGHT"
            if passed
            else "STOP_PROGRAM_O_FIXED_CLOCK_PHYSICAL_PREFLIGHT"
        ),
        "development_tapes_only": [SEEDS[0], SEEDS[-1]],
        "sealed_validation_accessed": False,
        "primary_policy": config.config_id,
        "cells": list(CELLS),
        "episodes_per_mode": len(rows),
        "total_direct_episodes": len(rows) * 2,
        "unique_scheduled_resource_vectors": len(scheduled_resource_vectors),
        "contract_sha256": digest(CONTRACT_PATH),
        "parent_contract_sha256": digest(PARENT_CONTRACT_PATH),
        "failures": failures,
        "rows": rows,
        "claim_boundary": {
            "preflight_passed": passed,
            "h_obs_confirmed": False,
            "learner_authorized": False,
            "paper2_confirmed": False,
            "paper3_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "episodes_per_mode", "total_direct_episodes", "unique_scheduled_resource_vectors", "sealed_validation_accessed")}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
