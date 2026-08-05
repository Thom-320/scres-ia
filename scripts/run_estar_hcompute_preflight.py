#!/usr/bin/env python3
"""Burned-only E* planning-cost preflight against the historical DES bridge.

This runner is intentionally fail-closed.  It measures planning work using the
source-conserving E* adapter on one burned exogenous tape, verifies the
historical M000 golden vector, and records the bridge conservation receipt.  It
never opens a root, allocates a seed, or trains a learner.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supply_chain.arm_runner import canonical_payload_sha256
from supply_chain.estar_bridge import (
    check_expanded_bridge_smoke,
    check_flags_off_golden,
    load_tape,
    make_expanded_sim,
    make_m000_adapter_sim,
)
from supply_chain.estar_kernel import (
    DISPATCH_LANES,
    EStarAction,
    EStarKernel,
    MASKS,
    SUPPLIER_LANES,
    flags_off_bridge_descriptor,
)
from supply_chain.seed_custody import module_manifest
from scripts.validate_estar_hcompute_contract import load, validate_contract


DEFAULT_CONTRACT = ROOT / "contracts/garrido_expanded_des_e_star_v2_hcompute.json"
DEFAULT_TAPE = ROOT / (
    "results/expanded_contract_comparators_v2_preflight_1dc40c1/preflight/"
    "R1r_actual_tapes.json"
)
DEFAULT_OUTPUT = ROOT / "results/estar_hcompute_preflight_v1/result.json"
SEED_REGISTRY = ROOT / "research/seed_custody_registry.json"
RUN_ROLE = "BURNED_COMPUTE_PREFLIGHT"
DES_STEP_HOURS = 168.0


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate percentile of empty timing sample")
    index = int(round((len(ordered) - 1) * q))
    return float(ordered[index])


def _fixture_kernel(mask_id: str, supplier_count: int, dispatch_count: int) -> EStarKernel:
    initial = {node: 0.0 for node in ("wdc", "al", "sb", "cssu_a", "cssu_b")}
    initial["sb"] = 20_000.0
    return EStarKernel(
        mask_id=mask_id,
        initial_inventory=initial,
        supplier_capacity={lane: 20_000.0 for lane in SUPPLIER_LANES},
        source_stock={lane: 1_000_000.0 for lane in SUPPLIER_LANES},
        transport_capacity={
            lane: 20_000.0 for lane in (*SUPPLIER_LANES, *DISPATCH_LANES)
        },
    )


def _kernel_action(
    mask_id: str, index: int, supplier_count: int, dispatch_count: int
) -> EStarAction:
    mask = MASKS[mask_id]
    suppliers = tuple(SUPPLIER_LANES[:supplier_count]) if mask["P"] else ()
    dispatch = tuple(DISPATCH_LANES[:dispatch_count]) if mask["D"] else ()
    procurement = {lane: float(100.0 + index) for lane in suppliers}
    targets = {
        node: float(500.0 + index)
        if mask["U"] or node in ("cssu_a", "cssu_b") and mask["D"]
        else 0.0
        for node in ("wdc", "al", "sb", "cssu_a", "cssu_b")
    }
    dispatch_qty = {lane: float(50.0 + index) for lane in dispatch}
    return EStarAction(
        procurement_qty=procurement,
        buffer_targets=targets,
        dispatch_qty=dispatch_qty,
        active_supplier_lanes=suppliers,
        active_dispatch_lanes=dispatch,
    )


def _des_action(
    mask_id: str, index: int, supplier_count: int, dispatch_count: int
) -> EStarAction:
    """Create a valid DES action without inventing dispatch stock.

    The burned tape does not guarantee claimant backlog or SB stock at the
    instant of this timing fixture.  Dispatch lanes are therefore activated
    with zero quantity; their combinatorial right is still part of the measured
    planning problem, while the conservation bridge remains honest.
    """
    mask = MASKS[mask_id]
    suppliers = tuple(SUPPLIER_LANES[:supplier_count]) if mask["P"] else ()
    dispatch = tuple(DISPATCH_LANES[:dispatch_count]) if mask["D"] else ()
    procurement = {lane: float(100.0 + index) for lane in suppliers}
    targets = {
        node: float(500.0 + index)
        if mask["U"] or node in ("cssu_a", "cssu_b") and mask["D"]
        else 0.0
        for node in ("wdc", "al", "sb", "cssu_a", "cssu_b")
    }
    return EStarAction(
        procurement_qty=procurement,
        buffer_targets=targets,
        dispatch_qty={lane: 0.0 for lane in dispatch},
        active_supplier_lanes=suppliers,
        active_dispatch_lanes=dispatch,
    )


def _measure(
    name: str,
    fn: Callable[[], tuple[int, int]],
    repetitions: int,
    warmups: int,
) -> dict[str, Any]:
    cold_started = time.perf_counter()
    cold_count, cold_calls = fn()
    cold_seconds = time.perf_counter() - cold_started
    for _ in range(warmups):
        fn()
    durations: list[float] = []
    rollouts: list[int] = []
    des_calls: list[int] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        count, calls = fn()
        durations.append(time.perf_counter() - started)
        rollouts.append(int(count))
        des_calls.append(int(calls))
    return {
        "planner": name,
        "repetitions": repetitions,
        "warmups": warmups,
        "wall_seconds": durations,
        "cold_seconds": cold_seconds,
        "cold_kernel_rollouts": int(cold_count),
        "cold_des_calls": int(cold_calls),
        "cold_p50_seconds": cold_seconds,
        "cold_p95_seconds": cold_seconds,
        "p50_seconds": _percentile(durations, 0.50),
        "p95_seconds": _percentile(durations, 0.95),
        "kernel_rollouts": int(sum(rollouts) / len(rollouts)),
        "des_calls": int(sum(des_calls) / len(des_calls)),
        "solver_iterations": 0,
        "peak_memory_bytes": None,
    }


def _run_des_once(
    tape: dict[str, Any],
    mask_id: str,
    index: int,
    supplier_count: int,
    dispatch_count: int,
) -> None:
    step_hours = min(DES_STEP_HOURS, float(tape["horizon"]))
    if mask_id == "M000":
        sim = make_m000_adapter_sim(tape)
        sim.step(action=None, step_hours=step_hours)
        return
    sim = make_expanded_sim(tape, mask_id)
    sim.step_e_star(
        _des_action(mask_id, index, supplier_count, dispatch_count),
        step_hours=step_hours,
    )


def _benchmark_level(
    level: dict[str, Any],
    tape: dict[str, Any],
    repetitions: int,
    warmups: int,
) -> dict[str, Any]:
    mask_id = str(level["mask_id"])
    supplier_count = int(level["active_supplier_lanes"])
    dispatch_count = int(level["active_dispatch_lanes"])
    complexity = max(1, supplier_count + dispatch_count + 1)
    candidate_count = 2**complexity

    def one_action() -> tuple[int, int]:
        _run_des_once(tape, mask_id, 0, supplier_count, dispatch_count)
        return 1, 1

    def rollout_search(multiplier: int) -> tuple[int, int]:
        count = candidate_count * multiplier
        for index in range(count):
            _run_des_once(tape, mask_id, index, supplier_count, dispatch_count)
        return count, count

    return {
        "level_id": str(level["id"]),
        "mask_id": mask_id,
        "active_supplier_lanes": supplier_count,
        "active_dispatch_lanes": dispatch_count,
        "candidate_count": candidate_count,
        "planner_backend": "EStarDESAdapter",
        "step_hours": min(DES_STEP_HOURS, float(tape["horizon"])),
        "planners": [
            _measure("constant", lambda: (0, 0), repetitions, warmups),
            _measure("lookup_order_up_to", one_action, repetitions, warmups),
            _measure("threshold_hysteresis", one_action, repetitions, warmups),
            _measure(
                "dp_rollout",
                lambda: rollout_search(1),
                repetitions,
                warmups,
            ),
            _measure(
                "mpc_direct",
                lambda: rollout_search(3),
                repetitions,
                warmups,
            ),
        ],
    }


def _falsifiers(
    contract: dict[str, Any],
    tape: dict[str, Any],
    bridge: dict[str, Any],
    expanded_bridge: dict[str, Any],
) -> dict[str, Any]:
    changed = _fixture_kernel("M111", 3, 2)
    unchanged = _fixture_kernel("M111", 3, 2)
    changed.step(
        _kernel_action("M111", 0, 3, 2),
        demand={"cssu_a": 100.0, "cssu_b": 100.0},
    )
    unchanged.step(
        EStarAction(), demand={"cssu_a": 100.0, "cssu_b": 100.0}
    )
    evidence = changed._ledger.as_evidence(changed._inventory, tuple(changed._transit))
    f1 = abs(float(evidence["physical_residual"])) <= 1e-9
    f2 = "future_demand" not in changed.observe() and "future_risk" not in changed.observe()
    f3 = changed.state().buffer_targets != unchanged.state().buffer_targets
    try:
        EStarKernel(mask_id="M000").step(
            EStarAction(
                procurement_qty={"supplier_wdc": 1.0},
                active_supplier_lanes=("supplier_wdc",),
            )
        )
    except ValueError:
        f4 = True
    else:
        f4 = False
    f5 = float(evidence["physical_residual"]) == 0.0
    f6 = all(value >= 0.0 for value in changed._inventory.values())
    f7 = set(changed.observe()) <= {
        "time",
        "inventory",
        "on_order",
        "buffer_targets",
        "backlog",
        "mask_id",
    }
    rows = contract.get("factorial_masks", [])
    f8 = [row.get("mask_id") for row in rows] == list(MASKS) and all(
        set(row) >= {"mask_id", "P", "U", "D"} for row in rows
    )
    levels = contract.get("h_compute", {}).get("size_ladder", [])
    complexities = [
        int(row.get("active_supplier_lanes", 0))
        + int(row.get("active_dispatch_lanes", 0))
        for row in levels
    ]
    f9 = (
        len({row.get("id") for row in levels}) == len(levels)
        and levels[0].get("mask_id") == "M000"
        and levels[-1].get("mask_id") == "M111"
        and sum(b > a for a, b in zip(complexities, complexities[1:])) >= 2
    )
    checks = {
        "f1_conservation": {"passed": f1, "evidence": evidence},
        "f2_no_future_observation": {"passed": f2, "evidence": changed.observe()},
        "f3_action_is_live": {
            "passed": f3,
            "evidence": {
                "changed_targets": changed.state().buffer_targets,
                "unchanged_targets": unchanged.state().buffer_targets,
            },
        },
        "f4_masked_action_rejected": {"passed": f4, "evidence": {"mask": "M000"}},
        "f5_no_physical_creation": {
            "passed": f5,
            "evidence": {"physical_residual": evidence["physical_residual"]},
        },
        "f6_nonnegative_inventory": {"passed": f6, "evidence": changed._inventory},
        "f7_observation_schema_closed": {
            "passed": f7,
            "evidence": sorted(changed.observe()),
        },
        "f8_factorial_masks_complete": {
            "passed": f8,
            "evidence": list(MASKS),
        },
        "f9_flags_off_golden": {"passed": bridge["passed"], "evidence": bridge},
        "f10_expanded_bridge_smoke": {
            "passed": expanded_bridge["passed"],
            "evidence": {
                "observed_digest": expanded_bridge["observed_digest"],
                "expected_digest": expanded_bridge["expected_digest"],
                "residuals": expanded_bridge["residuals"],
            },
        },
    }
    checks["all_passed"] = all(
        item["passed"] for name, item in checks.items() if name != "all_passed"
    )
    return checks


def _two_consecutive_increases(values: list[float]) -> bool:
    return any(
        right > middle and middle > left
        for left, middle, right in zip(values, values[1:], values[2:])
    )


def _adjudicate_h_compute(
    timing: list[dict[str, Any]], h_compute: dict[str, Any]
) -> dict[str, Any]:
    mpc = [
        next(row for row in level["planners"] if row["planner"] == "mpc_direct")
        for level in timing
    ]
    p95_seconds = [float(row["p95_seconds"]) for row in mpc]
    des_calls = [int(row["des_calls"]) for row in mpc]
    cadence_budget = (
        float(h_compute["native_cadence_fraction_budget"])
        * float(h_compute["decision_cadence_hours"])
        * 3600.0
    )
    m000_calls = max(1, des_calls[0])
    call_budget = m000_calls * float(h_compute["relative_call_budget_vs_m000"])
    latency_gate = any(value >= cadence_budget for value in p95_seconds[1:])
    calls_gate = max(des_calls[1:], default=0) >= call_budget
    latency_trend = _two_consecutive_increases(p95_seconds)
    calls_trend = _two_consecutive_increases([float(value) for value in des_calls])
    passes = bool(
        (latency_gate or calls_gate) and (latency_trend or calls_trend)
    )
    return {
        "passed": passes,
        "latency_budget_seconds": cadence_budget,
        "call_budget": call_budget,
        "mpc_p95_seconds_by_level": p95_seconds,
        "mpc_des_calls_by_level": des_calls,
        "latency_gate": latency_gate,
        "calls_gate": calls_gate,
        "latency_two_consecutive_increases": latency_trend,
        "calls_two_consecutive_increases": calls_trend,
        "rule": h_compute["pass_rule"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-role", choices=(RUN_ROLE,), required=True)
    parser.add_argument("--replay-of", required=True)
    parser.add_argument("--tape-file", type=Path, default=DEFAULT_TAPE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    contract = load(args.contract)
    registry = load(SEED_REGISTRY)
    validation = validate_contract(contract, registry)
    if not validation["ok"]:
        raise SystemExit(json.dumps(validation, indent=2))
    tape = load_tape(args.tape_file)
    started = time.perf_counter()

    bridge = check_flags_off_golden(
        tape,
        contract.get("flags_off_bridge", {}).get("golden_payload_sha256"),
    )
    expanded_bridge = check_expanded_bridge_smoke(
        tape,
        contract.get("expanded_des_bridge", {}).get("smoke_payload_sha256"),
        m000_expected_digest=contract.get("flags_off_bridge", {}).get(
            "golden_payload_sha256"
        ),
    )
    levels = contract["h_compute"]["size_ladder"]
    timing = [
        _benchmark_level(
            level,
            tape,
            int(contract["h_compute"]["timing_repetitions"]),
            int(contract["h_compute"]["warmup_repetitions"]),
        )
        for level in levels
    ]
    falsifiers = _falsifiers(contract, tape, bridge, expanded_bridge)
    expanded_bridge_ready = bool(
        contract.get("expanded_des_bridge", {}).get("status")
        == "PASS_BURNED_SOURCE_CONSERVING_SMOKE"
        and expanded_bridge["passed"]
    )
    adjudication = _adjudicate_h_compute(timing, contract["h_compute"])
    if not bridge["passed"]:
        claim_status = "STOP_ESTAR_FLAGS_OFF_NON_EQUIVALENT"
    elif not falsifiers["all_passed"]:
        claim_status = "STOP_ESTAR_FALSIFIER_FAILED"
    elif not expanded_bridge_ready:
        claim_status = "STOP_ESTAR_DES_BRIDGE_NOT_READY"
    elif adjudication["passed"]:
        claim_status = "H_COMPUTE_PASS_NEURAL_AMORTIZATION_ELIGIBLE"
    else:
        claim_status = "STOP_ESTAR_PLANNER_NOT_BINDING"

    payload: dict[str, Any] = {
        "schema_version": "estar_hcompute_preflight_v2",
        "claim_status": claim_status,
        "run_role": args.run_role,
        "replay_of": args.replay_of,
        "engineering_only": True,
        "scientific_execution_authorized": False,
        "fresh_seeds_opened": False,
        "learner_trained": False,
        "fixture": {
            "path": str(args.tape_file),
            "seed": int(tape["seed"]),
            "family": str(tape["family"]),
            "horizon": float(tape["horizon"]),
        },
        "command_argv": [str(value) for value in sys.argv],
        "hardware_and_protocol": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "protocol": {
                "clock": "time.perf_counter",
                "warmups_excluded_from_hot_statistics": True,
                "cold_measurement": "one pre-warmup invocation",
                "hot_measurements": int(contract["h_compute"]["timing_repetitions"]),
                "warmup_measurements": int(contract["h_compute"]["warmup_repetitions"]),
                "horizon_hours": float(contract["h_compute"]["horizon_hours"]),
                "decision_cadence_hours": float(
                    contract["h_compute"]["decision_cadence_hours"]
                ),
                "backend": "historical MFSCSimulation through EStarDESAdapter",
                "des_step_hours": DES_STEP_HOURS,
            },
        },
        "bridge_descriptor": flags_off_bridge_descriptor(),
        "bridge_check": bridge,
        "expanded_bridge_check": expanded_bridge,
        "expanded_des_bridge_ready": expanded_bridge_ready,
        "timing": timing,
        "h_compute_adjudication": adjudication,
        "h_compute_adjudicated": bool(expanded_bridge_ready and falsifiers["all_passed"]),
        "falsifiers": falsifiers,
        "contract_path": str(args.contract),
        "contract_sha256": hashlib.sha256(args.contract.read_bytes()).hexdigest(),
        "module_manifest": module_manifest(
            modules=(
                "supply_chain/estar_kernel.py",
                "supply_chain/estar_des_adapter.py",
                "supply_chain/estar_bridge.py",
                "supply_chain/supply_chain.py",
                "supply_chain/seed_custody.py",
                "supply_chain/arm_runner.py",
            ),
            script=Path(__file__),
        ),
        "git_commit": _git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload["canonical_payload_sha256"] = canonical_payload_sha256(payload)
    payload["self_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "claim_status": claim_status,
                "output": str(args.output),
                "self_sha256": payload["self_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
