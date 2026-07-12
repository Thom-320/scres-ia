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
PREFIX_POLICIES = (
    ConvoyThresholdPolicy(1_000.0, 24.0),
    ConvoyThresholdPolicy(2_500.0, 48.0),
    ConvoyThresholdPolicy(5_000.0, 48.0),
    ConvoyThresholdPolicy(5_000.0, 72.0),
)
PRIMARY_SEQUENCE_DAYS = 7
SENSITIVITY_SEQUENCE_DAYS = 10
SENSITIVITY_STATE_LIMIT = 12
RET_TIE_TOLERANCE = 1e-9
SERVICE_TIE_TOLERANCE = 1e-9


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


def select_state(tape: dict[str, Any], policy: ConvoyThresholdPolicy) -> dict[str, Any]:
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

    if policy.policy_id == "threshold_1000__wait_24h":
        row = min(candidates, key=lambda item: item["op7_staged_inventory"])
    elif policy.policy_id == "threshold_2500__wait_48h":
        row = max(candidates, key=lambda item: item["downstream_backlog_qty"])
    elif policy.policy_id == "threshold_5000__wait_48h":
        row = max(
            [item for item in candidates if item["recent_route_recovery"]] or candidates,
            key=lambda item: (item["recent_route_recovery"], item["staging_age"]),
        )
    else:
        row = max(candidates, key=lambda item: item["op7_staged_inventory"])
    return {
        "state_id": f"{tape['tape_id']}|prefix={policy.policy_id}",
        "tape_id": tape["tape_id"],
        "family": tape["family"],
        "prefix_policy_id": policy.policy_id,
        "prefix_inventory_threshold": policy.inventory_threshold,
        "prefix_maximum_wait_hours": policy.maximum_wait_hours,
        **row,
    }


def state_policy(state: dict[str, Any]) -> ConvoyThresholdPolicy:
    return ConvoyThresholdPolicy(
        float(state["prefix_inventory_threshold"]),
        float(state["prefix_maximum_wait_hours"]),
    )


def branch_actions(
    tape: dict[str, Any], state: dict[str, Any], policy: ConvoyThresholdPolicy,
    forced_actions: tuple[str, ...],
    *,
    continuation_policy: ConvoyThresholdPolicy | None = None,
) -> dict[str, Any]:
    sim, start = make_sim(tape)
    advance_prefix_to(sim, start, float(state["relative_time"]), policy)
    if state_hash(sim) != state["prefix_hash"]:
        raise RuntimeError(f"FAIL_PREFIX_IDENTITY {state['state_id']}")
    state_time = float(sim.env.now)
    resource_before = resource_snapshot(sim)
    action_iter = iter(forced_actions)
    continuation_policy = continuation_policy or policy
    forced_remaining = True
    short_metrics = None
    t1_signature = None
    end = state_time + LONG_HOURS
    while sim.env.now < end - 1e-9:
        if forced_remaining:
            try:
                action = next(action_iter)
            except StopIteration:
                forced_remaining = False
                action = continuation_policy.action(sim.get_op8_convoy_observation())
        else:
            action = continuation_policy.action(sim.get_op8_convoy_observation())
        sim.apply_op8_convoy_action(action, source="forced_sequence" if forced_remaining else "continuation")
        next_time = min(end, float(sim.env.now) + HOURS_PER_DAY)
        advance_including(sim, next_time)
        if t1_signature is None:
            obs = sim.get_op8_convoy_observation()
            t1_signature = json.dumps(
                {
                    "convoy_available": obs["convoy_available"],
                    "dispatch_feasible": sim.op8_convoy_dispatch_feasible(),
                    "staged": obs["op7_staged_inventory"],
                    "in_transit": float(sim._in_transit),
                    "departures": float(sim.op8_convoy_departures),
                    "unavailable_hours": float(sim.op8_convoy_vehicle_hours),
                },
                sort_keys=True,
            )
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
        "t1_physical_signature": t1_signature,
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
    tapes = json.loads((args.frontier_dir / "tapes.json").read_text())
    if args.states_per_tape != 4:
        raise RuntimeError("DRA-2 contract requires exactly four balanced prefixes per tape")
    states = [
        select_state(tape, policy)
        for tape in tapes
        for policy in PREFIX_POLICIES
    ]
    write_csv(args.output_dir / "states.csv", states)
    tape_map = {tape["tape_id"]: tape for tape in tapes}

    one_rows: list[dict[str, Any]] = []
    for state in states:
        policy = state_policy(state)
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
    by_tape: dict[str, list[dict[str, Any]]] = {}
    for state in states:
        by_tape.setdefault(state["tape_id"], []).append(state)
    selected_by_family: dict[str, list[dict[str, Any]]] = {}
    for family in sorted({state["family"] for state in states}):
        family_tapes = sorted(
            tape_id for tape_id, tape_states in by_tape.items()
            if tape_states[0]["family"] == family
        )
        rows = []
        for tape_index, tape_id in enumerate(family_tapes):
            prefix = PREFIX_POLICIES[tape_index % len(PREFIX_POLICIES)].policy_id
            rows.append(next(
                state for state in by_tape[tape_id]
                if state["prefix_policy_id"] == prefix
            ))
        selected_by_family[family] = rows
    sequence_states: list[dict[str, Any]] = []
    while len(sequence_states) < min(sequence_limit, len(by_tape)):
        progressed = False
        for family in sorted(selected_by_family):
            if selected_by_family[family]:
                sequence_states.append(selected_by_family[family].pop(0))
                progressed = True
                if len(sequence_states) >= min(sequence_limit, len(by_tape)):
                    break
        if not progressed:
            break
    if len({row["tape_id"] for row in sequence_states}) != len(sequence_states):
        raise RuntimeError("FAIL_SEQUENCE_TAPE_BALANCE")
    sequence_rows: list[dict[str, Any]] = []
    for state_index, state in enumerate(sequence_states):
        policy = state_policy(state)
        horizons = [PRIMARY_SEQUENCE_DAYS]
        if state_index < min(SENSITIVITY_STATE_LIMIT, len(sequence_states)):
            horizons.append(SENSITIVITY_SEQUENCE_DAYS)
        for days in horizons:
            for sequence in product(ACTIONS, repeat=days):
                result = branch_actions(tape_map[state["tape_id"]], state, policy, sequence)
                sequence_rows.append(
                    {"state_id": state["state_id"], "tape_id": state["tape_id"],
                     "family": state["family"], "sequence_days": days,
                     "sequence": "|".join(sequence),
                     "first_action": sequence[0], **result}
                )
        print(f"[dra2-sequence] {state['state_id']} 7d primary"
              f"{' + 10d sensitivity' if len(horizons) > 1 else ''} complete", flush=True)
    write_csv(args.output_dir / "sequence_rows.csv", sequence_rows)

    by_state: dict[str, list[dict[str, Any]]] = {}
    for row in one_rows:
        by_state.setdefault(row["state_id"], []).append(row)
    one_oracle = []
    for state_id, rows in by_state.items():
        hold = next(row for row in rows if row["action"] == "HOLD")
        dispatch = next(row for row in rows if row["action"] == "DISPATCH_NOW")
        best = max(rows, key=lambda row: (float(row["long_ret"]), -float(row["long_service"])))
        ret_gap = float(best["long_ret"]) - float(
            dispatch["long_ret"] if best["action"] == "HOLD" else hold["long_ret"]
        )
        other = dispatch if best["action"] == "HOLD" else hold
        service_relative = (
            float(other["long_service"]) - float(best["long_service"])
        ) / max(abs(float(other["long_service"])), 1.0)
        supported = (
            best["action"]
            if ret_gap > RET_TIE_TOLERANCE
            or (
                abs(ret_gap) <= RET_TIE_TOLERANCE
                and service_relative > SERVICE_TIE_TOLERANCE
            )
            else "NONE_TIE_OR_ZERO"
        )
        one_oracle.append(
            {"state_id": state_id, "optimal_action": best["action"],
             "diversity_supported_action": supported,
             "ret_gap_vs_other": ret_gap,
             "service_improvement_vs_other": service_relative,
             "strong_live": hold["t1_physical_signature"] != next(
                 row["t1_physical_signature"] for row in rows
                 if row["action"] == "DISPATCH_NOW"
             ),
             "delta_ret_vs_hold": float(best["long_ret"]) - float(hold["long_ret"]),
             "service_improvement_vs_hold": (
                 float(hold["long_service"]) - float(best["long_service"])
             ) / max(abs(float(hold["long_service"])), 1.0)}
        )
    write_csv(args.output_dir / "one_action_oracle.csv", one_oracle)

    seq_by_state: dict[str, list[dict[str, Any]]] = {}
    for row in sequence_rows:
        seq_by_state.setdefault(f"{row['state_id']}|days={row['sequence_days']}", []).append(row)
    sequence_oracle = []
    for state_day_id, rows in seq_by_state.items():
        state_id, days_text = state_day_id.rsplit("|days=", 1)
        best = max(rows, key=lambda row: (float(row["long_ret"]), -float(row["long_service"])))
        all_hold = next(row for row in rows if set(row["sequence"].split("|")) == {"HOLD"})
        sequence_oracle.append(
            {"state_id": state_id, "sequence_days": int(days_text),
             "family": best["family"],
             "best_sequence": best["sequence"],
             "optimal_first_action": best["first_action"],
             "long_ret": best["long_ret"], "long_service": best["long_service"],
             "delta_ret_vs_all_hold": float(best["long_ret"]) - float(all_hold["long_ret"]),
             "realized_departure_pattern": best["realized_departure_pattern"]}
        )
    write_csv(args.output_dir / "sequence_oracle.csv", sequence_oracle)

    oracle_by_state_day = {
        (row["state_id"], int(row["sequence_days"])): row
        for row in sequence_oracle
    }
    sensitivity_pairs = []
    for state in sequence_states[:min(SENSITIVITY_STATE_LIMIT, len(sequence_states))]:
        primary = oracle_by_state_day[(state["state_id"], PRIMARY_SEQUENCE_DAYS)]
        sensitivity = oracle_by_state_day[(state["state_id"], SENSITIVITY_SEQUENCE_DAYS)]
        h7 = float(primary["delta_ret_vs_all_hold"])
        h10 = float(sensitivity["delta_ret_vs_all_hold"])
        sensitivity_pairs.append(
            {
                "state_id": state["state_id"],
                "family": state["family"],
                "first_action_agrees": (
                    primary["optimal_first_action"] == sensitivity["optimal_first_action"]
                ),
                "headroom_7d": h7,
                "headroom_10d": h10,
                "absolute_headroom_change": abs(h10 - h7),
                "relative_headroom_change": (
                    abs(h10 - h7) / max(abs(h7), 1e-12)
                ),
                "sign_flip": (h7 > 0) != (h10 > 0),
            }
        )
    write_csv(args.output_dir / "sequence_sufficiency.csv", sensitivity_pairs)
    first_action_agreement = sum(
        row["first_action_agrees"] for row in sensitivity_pairs
    ) / max(len(sensitivity_pairs), 1)
    stable_headroom = all(
        row["absolute_headroom_change"] <= 0.002
        and row["relative_headroom_change"] <= 0.20
        for row in sensitivity_pairs
    )
    no_family_sign_flip = not any(row["sign_flip"] for row in sensitivity_pairs)

    verdict = {
        "gate": (
            "DRA2_BRANCHING_CALIBRATION"
            if frontier["calibration_opened"]
            else "DRA2_BRANCHING_IMPLEMENTATION_SMOKE"
        ),
        "n_tapes": len(tapes), "n_states": len(states),
        "n_one_action_rollouts": len(one_rows),
        "n_sequence_states": len(sequence_states),
        "n_sequence_rollouts": len(sequence_rows),
        "sequence_days": [PRIMARY_SEQUENCE_DAYS, SENSITIVITY_SEQUENCE_DAYS],
        "unique_realized_sequence_patterns": len({row["realized_departure_pattern"] for row in sequence_rows}),
        "one_action_optimal_counts": {
            action: sum(row["optimal_action"] == action for row in one_oracle)
            for action in ACTIONS
        },
        "one_action_diversity_supported_counts": {
            action: sum(row["diversity_supported_action"] == action for row in one_oracle)
            for action in (*ACTIONS, "NONE_TIE_OR_ZERO")
        },
        "tie_tolerances": {
            "ret": RET_TIE_TOLERANCE,
            "service_relative": SERVICE_TIE_TOLERANCE,
        },
        "sequence_first_action_counts_7d": {
            action: sum(
                row["optimal_first_action"] == action
                and row["sequence_days"] == PRIMARY_SEQUENCE_DAYS
                for row in sequence_oracle
            )
            for action in ACTIONS
        },
        "strong_live_fraction": sum(row["strong_live"] for row in one_oracle) / max(len(one_oracle), 1),
        "g_b_strong_liveness_pass": (
            sum(row["strong_live"] for row in one_oracle) / max(len(one_oracle), 1)
        ) >= 0.20,
        "restricted_sequence_oracle": True,
        "sequence_sufficiency": {
            "n_states": len(sensitivity_pairs),
            "first_action_agreement": first_action_agreement,
            "headroom_stability_pass": stable_headroom,
            "no_sign_flip_pass": no_family_sign_flip,
            "pass": (
                first_action_agreement >= 0.90
                and stable_headroom
                and no_family_sign_flip
            ),
        },
        "prefix_identity_pass": True,
        "crn_pass": True,
        "mass_pass": max(float(row["mass_residual"]) for row in one_rows + sequence_rows) <= 1e-6,
        "convoy_conservation_pass": max(abs(float(row["op8_convoy_resource_residual"])) for row in one_rows + sequence_rows) <= 1e-9,
        "authorization_record": frontier.get("authorization_record"),
        "authorization_decision": frontier.get("authorization_decision"),
        "calibration_opened": bool(frontier["calibration_opened"]),
        "virgin_tapes_opened": 0, "ppo_trained": False,
        "interpretation": (
            "CALIBRATION_DIAGNOSTIC_COMPLETE"
            if frontier["calibration_opened"]
            else "IMPLEMENTATION_SMOKE_PASS"
        ),
    }
    (args.output_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
