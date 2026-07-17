#!/usr/bin/env python3
"""Fail-closed G0/G1 preflight for relevant-risk sensitivity v1.1.

This runner deliberately has no G2 command.  The secondary robustness map is
blocked until an independent verifier signs the emitted G0 and G1 artifacts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.screen_program_o_fixed_clock_hobs_validation import decode_calendar
from scripts.screen_program_o_hobs_fit import load_cell_panel, primary_scheduler
from scripts.screen_program_o_state_rich_fit import evaluate_configuration
from supply_chain.program_o_full_des import run_program_o_full_des_episode
from supply_chain.program_o_full_des_transducer import MATRIX_KEYS, direct_full_des_vector
from supply_chain.program_o_state_rich import StateRichConfiguration


CONTRACT = ROOT / "contracts/program_o_relevant_risk_sensitivity_v1_1.json"
PARENT_FULL = ROOT / "contracts/program_o_full_des_hpi_translation_v1.json"
PARENT_HOBS = ROOT / "contracts/program_o_hobs_prelearner_v1.json"
PARENT_STATE = ROOT / "contracts/program_o_state_rich_comparator_fit_v1.json"
VALIDATION = ROOT / "results/program_o/fixed_clock_hobs_corrective_validation_v1/remote_run/artifacts/validation"
OUT = ROOT / "results/program_o/relevant_risk_sensitivity_v1_1"
CELL_ID = "rho90_share90"
TOLERANCE = 1e-9


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def frozen_context() -> dict[str, Any]:
    contract = read(CONTRACT)
    full = read(PARENT_FULL)
    hobs = read(PARENT_HOBS)
    state = read(PARENT_STATE)
    cell = next(row for row in hobs["stability_cells"] if row["id"] == CELL_ID)
    return {
        "contract": contract,
        "scheduler": primary_scheduler(full),
        "cell": cell,
        "model": state["observation_contract"]["model_parameters"],
        "seeds": [int(seed) for seed in contract["burned_preflight_tapes"]],
    }


def episode(seed: int, calendar: tuple[int, ...], *, risk: str | None = None, phi: float = 1.0):
    context = frozen_context()
    return run_program_o_full_des_episode(
        seed=int(seed),
        calendar=calendar,
        scheduler=context["scheduler"],
        regime_persistence=float(context["cell"]["regime_persistence"]),
        dominant_share=float(context["cell"]["dominant_product_share"]),
        downstream_freight_physics_mode="fixed_clock_physical_v1",
        risks_enabled=risk is not None,
        enabled_risks={risk} if risk else None,
        risk_frequency_multipliers_by_id={risk: float(phi)} if risk else None,
    )


def reconstruct_controller() -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
    context = frozen_context()
    seeds = context["seeds"]
    panel = load_cell_panel(VALIDATION, CELL_ID, seeds)
    skeletons = [
        read(VALIDATION / "skeletons" / CELL_ID / f"tape_{seed}.json")
        for seed in seeds
    ]
    return evaluate_configuration(
        config=StateRichConfiguration("belief_mpc", 3),
        cell=context["cell"],
        seeds=seeds,
        skeletons=skeletons,
        panel=panel,
        scheduler=context["scheduler"],
        model=context["model"],
    )


def vector_differences(seed: int, calendar_index: int) -> dict[str, float]:
    calendar = decode_calendar(int(calendar_index))
    sim, panel = episode(seed, calendar)
    direct = direct_full_des_vector(sim, panel)
    matrix = np.load(VALIDATION / "raw_calendar_matrix" / CELL_ID / f"tape_{seed}.npz")
    return {
        key: abs(float(direct[key]) - float(matrix[key][int(calendar_index)]))
        for key in MATRIX_KEYS
    }


def g0() -> dict[str, Any]:
    context = frozen_context()
    historical = read(VALIDATION / "result.json")["cells"][CELL_ID]
    row, _audits = reconstruct_controller()
    expected_indices = [int(value) for value in historical["calendar_indices"][: len(context["seeds"])]]
    observed_indices = [int(value) for value in row["calendar_indices"]]
    controller_identity = observed_indices == expected_indices
    static_index = int(historical["static_index"])
    replay_rows = []
    maximum = 0.0
    for seed, policy_index in zip(context["seeds"], observed_indices, strict=True):
        policy_diffs = vector_differences(seed, policy_index)
        static_diffs = vector_differences(seed, static_index)
        local_max = max(max(policy_diffs.values()), max(static_diffs.values()))
        maximum = max(maximum, local_max)
        replay_rows.append({
            "seed": seed,
            "policy_index": policy_index,
            "static_index": static_index,
            "maximum_abs_diff": local_max,
        })
    passed = controller_identity and maximum <= TOLERANCE
    return {
        "schema_version": "program_o_relevant_risk_sensitivity_v1_1_g0",
        "gate": "G0",
        "cell": CELL_ID,
        "seeds": context["seeds"],
        "controller": "belief_mpc__3",
        "controller_model_parameters": context["model"],
        "controller_indices_match_historical": controller_identity,
        "observed_indices": observed_indices,
        "expected_indices": expected_indices,
        "static_index": static_index,
        "replays": replay_rows,
        "maximum_abs_diff_all_metrics": maximum,
        "tolerance": TOLERANCE,
        "pass": bool(passed),
    }


def event_value(event: Any, name: str) -> Any:
    return event.get(name) if isinstance(event, Mapping) else getattr(event, name, None)


def g1_task(seed: int, risk_id: str, static_index: int) -> dict[str, Any]:
    """Run one isolated risk fixture and return only its auditable ledger."""
    sim, _panel = episode(
        int(seed), decode_calendar(int(static_index)), risk=str(risk_id), phi=1.0
    )
    events = []
    for event in list(getattr(sim, "risk_events", [])):
        events.append(
            {
                "risk_id": str(event_value(event, "risk_id")),
                "affected_ops": [
                    int(op) for op in (event_value(event, "affected_ops") or [])
                ],
            }
        )
    contingent_products: dict[str, int] = {"P_C": 0, "P_H": 0}
    invalid_products = []
    if risk_id == "R24":
        for order in getattr(sim, "orders", []):
            if bool(getattr(order, "contingent", False)):
                product = str(getattr(order, "requested_product_id", ""))
                if product in contingent_products:
                    contingent_products[product] += 1
                else:
                    invalid_products.append(product)
    return {
        "seed": int(seed),
        "risk_id": str(risk_id),
        "events": events,
        "contingent_products": contingent_products,
        "invalid_products": invalid_products,
    }


def g1() -> dict[str, Any]:
    context = frozen_context()
    historical = read(VALIDATION / "result.json")["cells"][CELL_ID]
    static_calendar = decode_calendar(int(historical["static_index"]))
    configured = {
        key: {int(op) for op in value}
        for key, value in context["contract"]["risks"]["affected_ops"].items()
    }
    per_risk: dict[str, Any] = {}
    overall_pass = True
    tasks = [
        (seed, risk_id)
        for risk_id in configured
        for seed in context["seeds"]
    ]
    task_rows: dict[tuple[str, int], dict[str, Any]] = {}
    # R14 full-DES fixtures peak near 3.3 GiB per worker on the reference Mac.
    # Keep this memory-safe and portable; CPU oversubscription caused swapping
    # in the discarded preflight and produced no scientific artifact.
    with ProcessPoolExecutor(max_workers=min(2, len(tasks))) as pool:
        futures = {
            pool.submit(g1_task, seed, risk_id, int(historical["static_index"])): (
                risk_id,
                seed,
            )
            for seed, risk_id in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            risk_id, seed = futures[future]
            task_rows[(risk_id, seed)] = future.result()
            print(f"G1 progress {completed}/{len(tasks)}", file=sys.stderr, flush=True)

    for risk_id, allowed in configured.items():
        counts: dict[str, int] = {}
        union: set[int] = set()
        invalid_events = []
        contingent_products = {"P_C": 0, "P_H": 0}
        per_seed = []
        for seed in context["seeds"]:
            task = task_rows[(risk_id, seed)]
            events = task["events"]
            local = 0
            for event in events:
                actual_id = str(event_value(event, "risk_id"))
                counts[actual_id] = counts.get(actual_id, 0) + 1
                if actual_id != risk_id:
                    continue
                local += 1
                ops = {int(op) for op in (event_value(event, "affected_ops") or [])}
                union.update(ops)
                if not ops or not ops.issubset(allowed):
                    invalid_events.append({"seed": seed, "affected_ops": sorted(ops)})
            if risk_id == "R24":
                for product, count in task["contingent_products"].items():
                    contingent_products[product] += int(count)
                for product in task["invalid_products"]:
                    invalid_events.append({"seed": seed, "invalid_contingent_product": product})
            per_seed.append({"seed": seed, "event_count": local})
        foreign = {key: value for key, value in counts.items() if key != risk_id}
        risk_pass = (
            counts.get(risk_id, 0) > 0
            and not foreign
            and not invalid_events
            and union == allowed
            and counts.get("R3", 0) == 0
        )
        overall_pass = overall_pass and risk_pass
        per_risk[risk_id] = {
            "phi": 1.0,
            "event_count": counts.get(risk_id, 0),
            "foreign_events": foreign,
            "allowed_affected_ops": sorted(allowed),
            "observed_affected_ops_union": sorted(union),
            "invalid_events": invalid_events,
            "per_seed": per_seed,
            "contingent_orders_by_frozen_product_label": contingent_products if risk_id == "R24" else None,
            "pass": bool(risk_pass),
        }
    return {
        "schema_version": "program_o_relevant_risk_sensitivity_v1_1_g1",
        "gate": "G1",
        "phi": 1.0,
        "seeds": context["seeds"],
        "r3_event_count": sum(row["foreign_events"].get("R3", 0) for row in per_risk.values()),
        "per_risk": per_risk,
        "high_frequency_fixture_can_rescue": False,
        "pass": bool(overall_pass),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gate", choices=("g0", "g1"))
    args = parser.parse_args()
    result = g0() if args.gate == "g0" else g1()
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"{args.gate}_result.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
