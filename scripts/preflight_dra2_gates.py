#!/usr/bin/env python3
"""Fail-closed DRA-2 G-A/G-B preflight using disposable fixtures only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.dra2_experiment import advance_including, make_sim, materialize_tape
from supply_chain.episode_metrics import compute_episode_metrics
from supply_chain.supply_chain import MFSCSimulation


def seeded_sim(qty: float) -> MFSCSimulation:
    sim = MFSCSimulation(
        seed=909,
        horizon=10_000,
        risks_enabled=False,
        op8_dispatch_mode="finite_convoy_v1",
    )
    sim.rations_al.put(float(qty))
    sim.op8_staging_first_ready_at = float(sim.env.now)
    return sim


def future_signature(sim: MFSCSimulation) -> dict[str, float | bool]:
    return {
        "convoy_available": bool(sim.op8_convoy_available),
        "dispatch_feasible": bool(sim.op8_convoy_dispatch_feasible()),
        "staged": float(sim.rations_al.level),
        "in_transit": float(sim._in_transit),
        "departures": float(sim.op8_convoy_departures),
        "vehicle_hours": float(sim.op8_convoy_vehicle_hours),
    }


def commitment_fixture(qty: float) -> dict:
    dispatch = seeded_sim(qty)
    hold = seeded_sim(qty)
    dispatch.apply_op8_convoy_action("DISPATCH_NOW", source="preflight")
    hold.apply_op8_convoy_action("HOLD", source="preflight")
    rows = []
    for hour in (24.001, 48.001):
        dispatch.env.run(until=hour)
        hold.env.run(until=hour)
        d = future_signature(dispatch)
        h = future_signature(hold)
        rows.append({"hour": hour, "dispatch": d, "hold": h, "different": d != h})
    return {
        "staged_quantity": qty,
        "both_actions_admissible_at_t0": True,
        "future_states": rows,
        "strong_live": all(row["different"] for row in rows),
        "return_gap_hours": float(
            dispatch.op8_convoy_actual_return_at - 48.0
        ),
    }


def integrated_cycle_fixture() -> dict:
    tape = materialize_tape(990001, "routine", 2, "disposable_preflight")
    sim, start = make_sim(tape)
    while not sim.op8_convoy_dispatch_feasible() and sim.env.now < start + 1_000:
        advance_including(sim, sim.env.now + 1.0)
    departed_at = float(sim.env.now)
    event = sim.apply_op8_convoy_action("DISPATCH_NOW", source="preflight")
    nominal = float(sim.op8_convoy_nominal_return_at)
    advance_including(sim, departed_at + 48.001)
    actual = float(sim.op8_convoy_actual_return_at)
    return {
        "tape_split": tape["split"],
        "departed": bool(event["departed"]),
        "departed_at": departed_at,
        "nominal_return_at": nominal,
        "actual_return_at": actual,
        "cycle_hours": actual - departed_at,
        "route_wait_hours": float(sim.op8_convoy_route_wait_hours),
        "pass_exact_48h_without_route_wait": abs(actual - nominal) <= 1e-9
        and abs(actual - departed_at - 48.0) <= 1e-9
        and abs(sim.op8_convoy_route_wait_hours) <= 1e-9,
    }


def default_identity_fixture() -> dict:
    kwargs = {"seed": 919, "horizon": 5_000, "risks_enabled": False}
    default = MFSCSimulation(**kwargs)
    explicit = MFSCSimulation(**kwargs, op8_dispatch_mode="thesis_full_batch")
    default.run()
    explicit.run()
    default_ret = compute_episode_metrics(default)["ret_excel_visible"]
    explicit_ret = compute_episode_metrics(explicit)["ret_excel_visible"]
    passed = (
        default.total_produced == explicit.total_produced
        and default.total_delivered == explicit.total_delivered
        and default.total_demanded == explicit.total_demanded
        and [order.CTj for order in default.orders]
        == [order.CTj for order in explicit.orders]
        and default_ret == explicit_ret
    )
    return {
        "pass": bool(passed),
        "default_ret_excel_visible_v1": float(default_ret),
        "explicit_ret_excel_visible_v1": float(explicit_ret),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/program_d/dra2_preflight/verdict.json"),
    )
    args = parser.parse_args()

    commitments = [commitment_fixture(qty) for qty in (1_000.0, 2_500.0, 5_000.0)]
    integrated = integrated_cycle_fixture()
    identity = default_identity_fixture()
    ga_pass = all(row["strong_live"] for row in commitments) and integrated[
        "pass_exact_48h_without_route_wait"
    ]
    verdict = {
        "contract_id": "op7_op8_finite_convoy_v1",
        "scope": "disposable preflight only; no calibration, holdout, virgin or PPO data",
        "g_a_intertemporal_commitment": {
            "pass": bool(ga_pass),
            "fixtures": commitments,
            "integrated_cycle": integrated,
        },
        "g_b_strong_liveness_definition": {
            "pass": bool(all(row["strong_live"] for row in commitments)),
            "definition": "both actions admissible at t0 and exact CRN branches differ physically at t+1 and t+2",
            "confirmatory_fraction_computed": False,
        },
        "g_e_default_identity": identity,
        "resource_rule_frozen": True,
        "garrido_face_validation": "PENDING",
        "calibration_opened": False,
        "ppo_trained": False,
        "verdict": "PASS_SOFTWARE_PREFLIGHT_AWAITING_GARRIDO_FACE_VALIDATION"
        if ga_pass and identity["pass"]
        else "FAIL_SOFTWARE_PREFLIGHT",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    if verdict["verdict"] == "FAIL_SOFTWARE_PREFLIGHT":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
