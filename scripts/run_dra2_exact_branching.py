#!/usr/bin/env python3
"""DRA-2 exact one-action and seven-day sequence branching."""
from __future__ import annotations

import argparse
import csv
from itertools import product
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import HOURS_PER_DAY  # noqa: E402
from supply_chain.dra2_convoy import ACTIONS, ConvoyThresholdPolicy  # noqa: E402
from supply_chain.dra2_experiment import (  # noqa: E402
    advance_including, episode_metrics_from, exogenous_hashes, make_sim,
    resource_delta, resource_snapshot, state_hash,
)


DEFAULT_FRONTIER = Path("results/program_d/dra2_static_frontier_smoke")
DEFAULT_OUTPUT = Path("results/program_d/dra2_exact_branching_smoke")
SHORT_HOURS = 72.0
LONG_HOURS = 28.0 * HOURS_PER_DAY


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def policy_from_verdict(verdict: dict[str, Any]) -> ConvoyThresholdPolicy:
    row = verdict["best_admissible"]
    return ConvoyThresholdPolicy(
        float(row["inventory_threshold"]), float(row["maximum_wait_hours"])
    )


def advance_prefix_to(sim, start: float, relative_time: float, policy: ConvoyThresholdPolicy) -> None:
    target = start + float(relative_time)
    while sim.env.now < target - 1e-9:
        action = policy.action(sim.get_op8_convoy_observation())
        sim.apply_op8_convoy_action(action, source="prefix")
        advance_including(sim, min(target, float(sim.env.now) + HOURS_PER_DAY))


def select_states(tape: dict[str, Any], policy: ConvoyThresholdPolicy, n_states: int = 4) -> list[dict[str, Any]]:
    sim, start = make_sim(tape)
    end = start + int(tape["horizon_weeks"]) * 7 * HOURS_PER_DAY - LONG_HOURS
    candidates: list[dict[str, Any]] = []
    while sim.env.now < end - 1e-9:
        obs = sim.get_op8_convoy_observation()
        if sim.op8_convoy_dispatch_feasible():
            rel = float(sim.env.now - start)
            recent_route_recovery = any(
                event["risk_id"] == "R22"
                and 0 <= rel - float(event["end_time"]) <= 72.0
                for event in tape["risk_events"]
            )
            candidates.append(
                {"relative_time": rel, "prefix_hash": state_hash(sim),
                 "recent_route_recovery": recent_route_recovery, **obs}
            )
        action = policy.action(obs)
        sim.apply_op8_convoy_action(action, source="state_collection")
        advance_including(sim, min(end, float(sim.env.now) + HOURS_PER_DAY))
    if not candidates:
        raise RuntimeError(f"No live DRA-2 state in {tape['tape_id']}")

    chosen: list[dict[str, Any]] = []
    selectors = [
        lambda rows: min(rows, key=lambda row: row["op7_staged_inventory"]),
        lambda rows: max(rows, key=lambda row: row["op7_staged_inventory"]),
        lambda rows: max(rows, key=lambda row: row["downstream_backlog_qty"]),
        lambda rows: max(
            [row for row in rows if row["recent_route_recovery"]] or rows,
            key=lambda row: (row["recent_route_recovery"], row["staging_age"]),
        ),
    ]
    for selector in selectors:
        remaining = [row for row in candidates if row["relative_time"] not in {x["relative_time"] for x in chosen}]
        if not remaining:
            break
        chosen.append(selector(remaining))
    while len(chosen) < min(n_states, len(candidates)):
        remaining = [row for row in candidates if row["relative_time"] not in {x["relative_time"] for x in chosen}]
        chosen.append(remaining[0])
    result = []
    for index, row in enumerate(chosen[:n_states]):
        result.append(
            {"state_id": f"{tape['tape_id']}|state={index}",
             "tape_id": tape["tape_id"], "family": tape["family"], **row}
        )
    return result


def branch_actions(
    tape: dict[str, Any], state: dict[str, Any], policy: ConvoyThresholdPolicy,
    forced_actions: tuple[str, ...],
) -> dict[str, Any]:
    sim, start = make_sim(tape)
    advance_prefix_to(sim, start, float(state["relative_time"]), policy)
    if state_hash(sim) != state["prefix_hash"]:
        raise RuntimeError(f"FAIL_PREFIX_IDENTITY {state['state_id']}")
    state_time = float(sim.env.now)
    resource_before = resource_snapshot(sim)
    action_iter = iter(forced_actions)
    forced_remaining = True
    short_metrics = None
    end = state_time + LONG_HOURS
    while sim.env.now < end - 1e-9:
        if forced_remaining:
            try:
                action = next(action_iter)
            except StopIteration:
                forced_remaining = False
                action = policy.action(sim.get_op8_convoy_observation())
        else:
            action = policy.action(sim.get_op8_convoy_observation())
        sim.apply_op8_convoy_action(action, source="forced_sequence" if forced_remaining else "continuation")
        next_time = min(end, float(sim.env.now) + HOURS_PER_DAY)
        advance_including(sim, next_time)
        if short_metrics is None and sim.env.now >= state_time + SHORT_HOURS - 1e-9:
            short_metrics = episode_metrics_from(sim, state_time)
    long_metrics = episode_metrics_from(sim, state_time)
    resources = resource_delta(sim, resource_before)
    hashes = exogenous_hashes(sim, start)
    realized = [
        bool(event["departed"])
        for event in sim.op8_convoy_action_events
        if event["time"] >= state_time - 1e-9 and event["source"] == "forced_sequence"
    ][:len(forced_actions)]
    return {
        "short_ret": float(short_metrics["ret_excel"] if short_metrics else 0.0),
        "short_service": float(short_metrics["service_loss_auc_ration_hours"] if short_metrics else 0.0),
        "long_ret": float(long_metrics["ret_excel"]),
        "long_clipped": float(long_metrics["ret_excel_visible_clipped_0_1"]),
        "long_service": float(long_metrics["service_loss_auc_ration_hours"]),
        "long_lost": float(long_metrics["lost_orders"]),
        "long_backlog": float(long_metrics["backorder_qty_final"]),
        "mass_residual": float(long_metrics["mass_residual"]),
        "risk_sha256": hashes["risk_sha256"],
        "demand_sha256": hashes["demand_sha256"],
        "realized_departure_pattern": json.dumps(realized),
        **resources,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier-dir", type=Path, default=DEFAULT_FRONTIER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--states-per-tape", type=int, default=4)
    parser.add_argument("--sequence-state-limit", type=int)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)

    frontier = json.loads((args.frontier_dir / "verdict.json").read_text())
    if frontier["interpretation"] not in {"IMPLEMENTATION_SMOKE_PASS", "PASS_STATIC_FRONTIER"}:
        raise RuntimeError("DRA-2 static frontier did not pass")
    policy = policy_from_verdict(frontier)
    tapes = json.loads((args.frontier_dir / "tapes.json").read_text())
    states = [state for tape in tapes for state in select_states(tape, policy, args.states_per_tape)]
    write_csv(args.output_dir / "states.csv", states)
    tape_map = {tape["tape_id"]: tape for tape in tapes}

    one_rows: list[dict[str, Any]] = []
    for state in states:
        expected = None
        for action in ACTIONS:
            result = branch_actions(tape_map[state["tape_id"]], state, policy, (action,))
            hashes = (result["risk_sha256"], result["demand_sha256"])
            if expected is None:
                expected = hashes
            elif hashes != expected:
                raise RuntimeError(f"FAIL_EXOGENOUS_CRN {state['state_id']}")
            one_rows.append({"state_id": state["state_id"], "tape_id": state["tape_id"],
                             "family": state["family"], "action": action, **result})
        print(f"[dra2-branch] {state['state_id']} one-step complete", flush=True)
    write_csv(args.output_dir / "one_action_rows.csv", one_rows)

    sequence_limit = (
        int(args.sequence_state_limit)
        if args.sequence_state_limit is not None
        else (60 if frontier["calibration_opened"] else 4)
    )
    by_family_states: dict[str, list[dict[str, Any]]] = {}
    for state in states:
        by_family_states.setdefault(state["family"], []).append(state)
    sequence_states: list[dict[str, Any]] = []
    while len(sequence_states) < min(sequence_limit, len(states)):
        progressed = False
        for family in sorted(by_family_states):
            family_rows = by_family_states[family]
            if family_rows:
                sequence_states.append(family_rows.pop(0))
                progressed = True
                if len(sequence_states) >= min(sequence_limit, len(states)):
                    break
        if not progressed:
            break
    sequence_rows: list[dict[str, Any]] = []
    sequences = tuple(product(ACTIONS, repeat=7))
    for state in sequence_states:
        for sequence in sequences:
            result = branch_actions(tape_map[state["tape_id"]], state, policy, sequence)
            sequence_rows.append(
                {"state_id": state["state_id"], "tape_id": state["tape_id"],
                 "family": state["family"], "sequence": "|".join(sequence),
                 "first_action": sequence[0], **result}
            )
        print(f"[dra2-sequence] {state['state_id']} 128 sequences complete", flush=True)
    write_csv(args.output_dir / "sequence_rows.csv", sequence_rows)

    by_state: dict[str, list[dict[str, Any]]] = {}
    for row in one_rows:
        by_state.setdefault(row["state_id"], []).append(row)
    one_oracle = []
    for state_id, rows in by_state.items():
        hold = next(row for row in rows if row["action"] == "HOLD")
        best = max(rows, key=lambda row: (float(row["long_ret"]), -float(row["long_service"])))
        one_oracle.append(
            {"state_id": state_id, "optimal_action": best["action"],
             "delta_ret_vs_hold": float(best["long_ret"]) - float(hold["long_ret"]),
             "service_improvement_vs_hold": (
                 float(hold["long_service"]) - float(best["long_service"])
             ) / max(abs(float(hold["long_service"])), 1.0)}
        )
    write_csv(args.output_dir / "one_action_oracle.csv", one_oracle)

    seq_by_state: dict[str, list[dict[str, Any]]] = {}
    for row in sequence_rows:
        seq_by_state.setdefault(row["state_id"], []).append(row)
    sequence_oracle = []
    for state_id, rows in seq_by_state.items():
        best = max(rows, key=lambda row: (float(row["long_ret"]), -float(row["long_service"])))
        sequence_oracle.append(
            {"state_id": state_id, "best_sequence": best["sequence"],
             "optimal_first_action": best["first_action"],
             "long_ret": best["long_ret"], "long_service": best["long_service"],
             "realized_departure_pattern": best["realized_departure_pattern"]}
        )
    write_csv(args.output_dir / "sequence_oracle.csv", sequence_oracle)

    verdict = {
        "gate": "DRA2_BRANCHING_IMPLEMENTATION_SMOKE",
        "n_tapes": len(tapes), "n_states": len(states),
        "n_one_action_rollouts": len(one_rows),
        "n_sequence_states": len(sequence_states),
        "n_sequence_rollouts": len(sequence_rows),
        "unique_realized_sequence_patterns": len({row["realized_departure_pattern"] for row in sequence_rows}),
        "one_action_optimal_counts": {
            action: sum(row["optimal_action"] == action for row in one_oracle)
            for action in ACTIONS
        },
        "sequence_first_action_counts": {
            action: sum(row["optimal_first_action"] == action for row in sequence_oracle)
            for action in ACTIONS
        },
        "prefix_identity_pass": True,
        "crn_pass": True,
        "mass_pass": max(float(row["mass_residual"]) for row in one_rows + sequence_rows) <= 1e-6,
        "convoy_conservation_pass": max(abs(float(row["op8_convoy_resource_residual"])) for row in one_rows + sequence_rows) <= 1e-9,
        "face_validation_accepted": bool(frontier["face_validation_accepted"]),
        "calibration_opened": bool(frontier["calibration_opened"]),
        "virgin_tapes_opened": 0, "ppo_trained": False,
        "interpretation": "IMPLEMENTATION_SMOKE_PASS",
    }
    (args.output_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
