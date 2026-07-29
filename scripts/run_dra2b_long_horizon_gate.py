#!/usr/bin/env python3
"""Exact canonical 14-day DRA-2b oracle and pre-tree gate."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from functools import lru_cache
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_dra2_exact_branching import (  # noqa: E402
    PREFIX_POLICIES, branch_actions, select_state, state_policy,
)
from supply_chain.dra2_convoy import ACTIONS, static_policies  # noqa: E402
from supply_chain.dra2_experiment import resource_dominance_static_comparator  # noqa: E402


PRIMARY_DAYS = 14
SENSITIVITY_DAYS = 10
SHORT_HOURS = 28.0 * 24.0
LONG_HOURS = 56.0 * 24.0
RET_TOL = 1e-9


@lru_cache(maxsize=None)
def canonical_sequences(days: int) -> tuple[tuple[str, ...], ...]:
    result: list[tuple[str, ...]] = []

    def visit(prefix: tuple[str, ...]) -> None:
        if len(prefix) == days:
            result.append(prefix)
            return
        visit((*prefix, "HOLD"))
        if not prefix or prefix[-1] != "DISPATCH_NOW":
            visit((*prefix, "DISPATCH_NOW"))

    visit(())
    return tuple(result)


def better(row: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    return (float(row["long_ret"]), -float(row["long_service"])) > (
        float(incumbent["long_ret"]), -float(incumbent["long_service"])
    )


def evaluate_horizon(
    tape: dict[str, Any], state: dict[str, Any], days: int
) -> dict[str, Any]:
    prefix = state_policy(state)
    best = None
    best_by_first: dict[str, dict[str, Any] | None] = {action: None for action in ACTIONS}
    all_hold = None
    patterns: set[str] = set()
    for sequence in canonical_sequences(days):
        row = branch_actions(
            tape, state, prefix, sequence,
            short_hours=SHORT_HOURS, long_hours=LONG_HOURS,
        )
        row = {"sequence": "|".join(sequence), "first_action": sequence[0], **row}
        patterns.add(row["realized_departure_pattern"])
        if set(sequence) == {"HOLD"}:
            all_hold = row
        if better(row, best):
            best = row
        if better(row, best_by_first[sequence[0]]):
            best_by_first[sequence[0]] = row
    assert best is not None and all_hold is not None
    headroom = float(best["long_ret"]) - float(all_hold["long_ret"])
    return {
        "days": days,
        "n_canonical_sequences": len(canonical_sequences(days)),
        "n_realized_patterns": len(patterns),
        "best": best,
        "all_hold": all_hold,
        "best_by_first": best_by_first,
        "headroom": headroom,
        "normalized_headroom": headroom / float(days),
    }


def oracle_worker(payload: tuple[dict, dict, bool]) -> dict[str, Any]:
    tape, state, sensitivity = payload
    primary = evaluate_horizon(tape, state, PRIMARY_DAYS)
    result = {"state": state, "primary": primary}
    if sensitivity:
        result["sensitivity"] = evaluate_horizon(tape, state, SENSITIVITY_DAYS)
    return result


def static_worker(payload: tuple[dict, dict]) -> list[dict[str, Any]]:
    tape, state = payload
    prefix = state_policy(state)
    rows = []
    for policy in static_policies():
        result = branch_actions(
            tape, state, prefix, (), continuation_policy=policy,
            short_hours=SHORT_HOURS, long_hours=LONG_HOURS,
        )
        rows.append({"policy_id": policy.policy_id, **result})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def bootstrap(values: np.ndarray, seed: int, n_boot: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--tape-limit", type=int)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)

    tapes = json.loads((args.frontier_dir / "tapes.json").read_text())
    if args.tape_limit is not None:
        tapes = tapes[:args.tape_limit]
    tape_map = {tape["tape_id"]: tape for tape in tapes}
    by_family_index: dict[str, int] = {}
    states: list[dict[str, Any]] = []
    for tape in tapes:
        index = by_family_index.get(tape["family"], 0)
        by_family_index[tape["family"]] = index + 1
        for offset in (0, 2):
            policy = PREFIX_POLICIES[(index + offset) % len(PREFIX_POLICIES)]
            states.append(select_state(tape, policy))
    if len({state["state_id"] for state in states}) != len(states):
        raise RuntimeError("FAIL_DRA2B_STATE_IDENTITY")
    write_csv(args.output_dir / "states.csv", states)

    sensitivity_ids: set[str] = set()
    for family in sorted({state["family"] for state in states}):
        family_states = []
        seen_tapes = set()
        for state in states:
            if state["family"] == family and state["tape_id"] not in seen_tapes:
                family_states.append(state); seen_tapes.add(state["tape_id"])
            if len(family_states) == 3:
                break
        sensitivity_ids.update(state["state_id"] for state in family_states)

    jobs = [
        (tape_map[state["tape_id"]], state, state["state_id"] in sensitivity_ids)
        for state in states
    ]
    oracle_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(oracle_worker, job): job[1]["state_id"] for job in jobs}
        for future in as_completed(futures):
            oracle_results.append(future.result())
            print(f"[dra2b-oracle] {futures[future]} complete", flush=True)
    oracle_results.sort(key=lambda row: row["state"]["state_id"])

    static_results: dict[str, list[dict[str, Any]]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(static_worker, (tape_map[state["tape_id"]], state)): state
            for state in states
        }
        for future in as_completed(futures):
            state = futures[future]
            static_results[state["state_id"]] = future.result()

    summary_rows = []
    static_rows = []
    for result in oracle_results:
        state = result["state"]; primary = result["primary"]
        hold_best = primary["best_by_first"]["HOLD"]
        dispatch_best = primary["best_by_first"]["DISPATCH_NOW"]
        first_gap = abs(float(hold_best["long_ret"]) - float(dispatch_best["long_ret"]))
        supported = primary["best"]["first_action"] if first_gap > RET_TOL else "NONE_TIE_OR_ZERO"
        row = {
            "state_id": state["state_id"], "tape_id": state["tape_id"],
            "family": state["family"], "prefix_policy_id": state["prefix_policy_id"],
            "optimal_first_action_14d": primary["best"]["first_action"],
            "supported_first_action_14d": supported,
            "headroom_14d": primary["headroom"],
            "normalized_headroom_14d": primary["normalized_headroom"],
            "ret_28d": primary["best"]["short_ret"],
            "service_28d": primary["best"]["short_service"],
            "ret_56d": primary["best"]["long_ret"],
            "service_56d": primary["best"]["long_service"],
            "lost_56d": primary["best"]["long_lost"],
            "departures_56d": primary["best"]["op8_convoy_departures"],
            "unavailable_hours_56d": primary["best"]["op8_convoy_unavailable_hours"],
            "strong_live": (
                hold_best["t1_physical_signature"] != dispatch_best["t1_physical_signature"]
            ),
            "n_patterns_14d": primary["n_realized_patterns"],
            "sensitivity_state": "sensitivity" in result,
        }
        if "sensitivity" in result:
            sens = result["sensitivity"]
            relative = abs(primary["normalized_headroom"] - sens["normalized_headroom"]) / max(
                abs(sens["normalized_headroom"]), 1e-12
            )
            row.update({
                "optimal_first_action_10d": sens["best"]["first_action"],
                "headroom_10d": sens["headroom"],
                "normalized_headroom_10d": sens["normalized_headroom"],
                "first_action_agrees_10d_14d": (
                    sens["best"]["first_action"] == primary["best"]["first_action"]
                ),
                "normalized_headroom_relative_change": relative,
                "normalized_headroom_absolute_change": abs(
                    primary["normalized_headroom"] - sens["normalized_headroom"]
                ),
                "positive_to_negative_reversal": (
                    sens["headroom"] > RET_TOL and primary["headroom"] < -RET_TOL
                ),
                "service_direction_28d_ok": (
                    float(primary["all_hold"]["short_service"])
                    - float(primary["best"]["short_service"])
                ) >= 0,
                "service_direction_56d_ok": (
                    float(primary["all_hold"]["long_service"])
                    - float(primary["best"]["long_service"])
                ) >= 0,
            })
        summary_rows.append(row)
        for static in static_results[state["state_id"]]:
            static_rows.append({
                "state_id": state["state_id"], "tape_id": state["tape_id"],
                "family": state["family"], **static,
            })
    write_csv(args.output_dir / "oracle_state_summary.csv", summary_rows)
    write_csv(args.output_dir / "resource_static_rows.csv", static_rows)

    candidate = {
        "mean_departures": float(np.mean([row["departures_56d"] for row in summary_rows])),
        "mean_unavailable_hours": float(np.mean([row["unavailable_hours_56d"] for row in summary_rows])),
    }
    static_summaries = []
    for policy in static_policies():
        selected = [row for row in static_rows if row["policy_id"] == policy.policy_id]
        static_summaries.append({
            "policy_id": policy.policy_id,
            "mean_ret": float(np.mean([row["long_ret"] for row in selected])),
            "mean_service": float(np.mean([row["long_service"] for row in selected])),
            "mean_departures": float(np.mean([row["op8_convoy_departures"] for row in selected])),
            "mean_unavailable_hours": float(np.mean([row["op8_convoy_unavailable_hours"] for row in selected])),
        })
    comparator = resource_dominance_static_comparator(candidate, static_summaries)
    static_index = {
        (row["state_id"], row["policy_id"]): row for row in static_rows
    }
    ret_delta = [] ; service_reduction = [] ; lost_delta = []
    for row in summary_rows:
        base = static_index[(row["state_id"], comparator["policy_id"])]
        ret_delta.append(float(row["ret_56d"]) - float(base["long_ret"]))
        service_reduction.append(
            (float(base["long_service"]) - float(row["service_56d"]))
            / max(abs(float(base["long_service"])), 1.0)
        )
        lost_delta.append(float(row["lost_56d"]) - float(base["long_lost"]))
    ret_ci = bootstrap(np.array(ret_delta), 20260720, args.n_boot)
    service_ci = bootstrap(np.array(service_reduction), 20260721, args.n_boot)

    sensitivity = [row for row in summary_rows if row["sensitivity_state"]]
    agreement = np.mean([row["first_action_agrees_10d_14d"] for row in sensitivity])
    horizon_pass = bool(
        agreement >= .90
        and all(row["normalized_headroom_relative_change"] <= .20 for row in sensitivity)
        and all(row["normalized_headroom_absolute_change"] <= .0005 for row in sensitivity)
        and not any(row["positive_to_negative_reversal"] for row in sensitivity)
        and all(row["service_direction_28d_ok"] and row["service_direction_56d_ok"] for row in sensitivity)
    )
    counts = {
        action: sum(row["supported_first_action_14d"] == action for row in summary_rows)
        for action in (*ACTIONS, "NONE_TIE_OR_ZERO")
    }
    n = len(summary_rows)
    gates = {
        "strong_liveness": bool(np.mean([row["strong_live"] for row in summary_rows]) >= .20),
        "diversity": bool(all(counts[action] / n >= .15 for action in ACTIONS)),
        "ret_practical": bool(ret_ci[0] >= .01 and ret_ci[1] > 0),
        "service_practical": bool(service_ci[0] >= .05 and service_ci[1] > 0),
        "horizon_sufficiency": bool(horizon_pass),
        "lost_orders": bool(float(np.mean(lost_delta)) <= 0),
        "resource_envelope": bool(
            comparator["mean_departures"] <= candidate["mean_departures"] + 1e-9
            and comparator["mean_unavailable_hours"] <= candidate["mean_unavailable_hours"] + 1e-9
        ),
    }
    promote = all(gates.values())
    verdict = {
        "gate": "DRA2B_LONG_HORIZON_PRE_TREE_GATE",
        "n_tapes": len(tapes), "n_states": n,
        "canonical_sequences_14d": len(canonical_sequences(14)),
        "canonical_sequences_10d": len(canonical_sequences(10)),
        "supported_action_counts": counts,
        "strong_live_fraction": float(np.mean([row["strong_live"] for row in summary_rows])),
        "horizon_sufficiency": {
            "n_states": len(sensitivity),
            "first_action_agreement": float(agreement),
            "pass": horizon_pass,
        },
        "resource_candidate": candidate,
        "resource_dominance_static_comparator": comparator,
        "ret_delta_mean_ci95": ret_ci,
        "service_loss_reduction_mean_ci95": service_ci,
        "lost_orders_delta_mean": float(np.mean(lost_delta)),
        "gates": gates,
        "observable_tree_authorized": promote,
        "holdout_opened": False,
        "ppo_authorized": False,
        "ppo_trained": False,
        "virgin_tapes_opened": 0,
        "interpretation": "PROMOTE_DRA2B_TO_OBSERVABLE_TREE" if promote else "STOP_DRA2B_PRE_TREE_GATE",
        "failed_gates": [name for name, passed in gates.items() if not passed],
    }
    (args.output_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if promote else 2


if __name__ == "__main__":
    raise SystemExit(main())
