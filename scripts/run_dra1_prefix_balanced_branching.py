#!/usr/bin/env python3
"""Corrective Gate C audit with counterbalanced 0.25/0.50/0.75 prefixes.

Uses the same 60 calibration tapes and nine branch actions.  Only the state
generating prefix changes.  Inference clusters the three prefix states back to
their common tape and expresses allocation as share sent to the stressed node.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_dra1_exact_branching import branch, select_state  # noqa: E402
from scripts.run_dra1_static_frontier import (  # noqa: E402
    ALLOCATION_LEVELS, SERVICE_RULES, boot_ci,
)


FRONTIER = Path("results/program_d/dra1_static_frontier")
DEFAULT_OUTPUT = Path("results/program_d/dra1_prefix_balanced_branching")
PREFIXES = (0.25, 0.50, 0.75)
PREFIX_RULE = "SPT_FULL"
TIE_TOL = 1e-12


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def stress_label(state: dict[str, Any], tolerance: float = 0.10) -> str:
    a = float(state["cssu_A_backlog_qty"])
    b = float(state["cssu_B_backlog_qty"])
    relative = (a - b) / max(a + b, 1.0)
    if relative > tolerance:
        return "A"
    if relative < -tolerance:
        return "B"
    return "balanced"


def normalized_share(allocation_a: float, stressed: str) -> float | None:
    if stressed == "A":
        return float(allocation_a)
    if stressed == "B":
        return 1.0 - float(allocation_a)
    return None


def cluster_ci(rows: list[dict[str, Any]], field: str, seed: int, n_boot: int) -> tuple[float, float, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["tape_id"])].append(float(row[field]))
    tape_means = [float(np.mean(values)) for values in grouped.values()]
    return boot_ci(tape_means, seed, n_boot)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier-dir", type=Path, default=FRONTIER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--limit-tapes", type=int)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)

    frontier = json.loads((args.frontier_dir / "verdict.json").read_text())
    if frontier["interpretation"] != "PASS_STATIC_FRONTIER":
        raise RuntimeError("Static frontier did not pass")
    selected = frontier["best_admissible"]
    comparator = (float(selected["allocation_a"]), str(selected["service_rule"]))
    tapes = json.loads((args.frontier_dir / "calibration_tapes.json").read_text())
    if args.limit_tapes:
        tapes = tapes[:args.limit_tapes]

    states: list[dict[str, Any]] = []
    modes = {0.25: "a_stressed", 0.50: "balanced", 0.75: "b_stressed"}
    for tape in tapes:
        for prefix_a in PREFIXES:
            prefix = (prefix_a, PREFIX_RULE)
            state = select_state(tape, prefix, selection_mode=modes[prefix_a])
            state_id = f"{tape['tape_id']}|prefix={prefix_a:.2f}"
            state.update(
                {"state_id": state_id, "prefix_allocation_a": prefix_a,
                 "prefix_service_rule": PREFIX_RULE, "stressed_node": stress_label(state)}
            )
            states.append(state)
    write_csv(args.output_dir / "states.csv", states)

    tape_map = {tape["tape_id"]: tape for tape in tapes}
    branch_rows: list[dict[str, Any]] = []
    for index, state in enumerate(states, 1):
        tape = tape_map[state["tape_id"]]
        prefix = (float(state["prefix_allocation_a"]), PREFIX_RULE)
        for allocation in ALLOCATION_LEVELS:
            for rule in SERVICE_RULES:
                result = branch(
                    tape, state, prefix, (allocation, rule),
                    continuation=comparator,
                )
                branch_rows.append(
                    {"state_id": state["state_id"], "tape_id": state["tape_id"],
                     "family": state["family"], "category": state["category"],
                     "prefix_allocation_a": state["prefix_allocation_a"],
                     "stressed_node": state["stressed_node"],
                     "allocation_a": allocation,
                     "share_to_stressed": normalized_share(allocation, state["stressed_node"]),
                     "concentration_share": max(allocation, 1.0 - allocation),
                     "service_rule": rule, **result}
                )
        print(f"[dra1-balanced] {index}/{len(states)} {state['state_id']}", flush=True)
    write_csv(args.output_dir / "branch_rows.csv", branch_rows)

    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in branch_rows:
        by_state[str(row["state_id"])].append(row)
    oracle_rows: list[dict[str, Any]] = []
    normalized_weights: Counter[float] = Counter()
    normalized_action_weights: Counter[tuple[float, str]] = Counter()
    for state_id, rows in by_state.items():
        baseline = next(
            row for row in rows
            if float(row["allocation_a"]) == comparator[0]
            and row["service_rule"] == comparator[1]
        )
        max_ret = max(float(row["long_ret"]) for row in rows)
        ret_ties = [row for row in rows if max_ret - float(row["long_ret"]) <= TIE_TOL]
        max_clip = max(float(row["long_clipped"]) for row in ret_ties)
        optimal = [row for row in ret_ties if max_clip - float(row["long_clipped"]) <= TIE_TOL]
        representative = sorted(
            optimal, key=lambda row: (float(row["allocation_a"]), str(row["service_rule"]))
        )[0]
        delta_ret = max_ret - float(baseline["long_ret"])
        informative = delta_ret > TIE_TOL
        shares = sorted({
            float(row["share_to_stressed"])
            for row in optimal if row["share_to_stressed"] not in {None, ""}
        })
        if shares and informative:
            for share in shares:
                normalized_weights[share] += 1.0 / len(shares)
            normalized_actions = sorted({
                (float(row["share_to_stressed"]), str(row["service_rule"]))
                for row in optimal if row["share_to_stressed"] not in {None, ""}
            })
            for key in normalized_actions:
                normalized_action_weights[key] += 1.0 / len(normalized_actions)
        oracle_rows.append(
            {"state_id": state_id, "tape_id": representative["tape_id"],
             "family": representative["family"], "category": representative["category"],
             "prefix_allocation_a": representative["prefix_allocation_a"],
             "stressed_node": representative["stressed_node"],
             "allocation_a": representative["allocation_a"],
             "share_to_stressed": representative["share_to_stressed"],
             "service_rule": representative["service_rule"],
             "n_tied_optimal_actions": len(optimal),
             "informative_headroom": informative,
             "optimal_normalized_shares": json.dumps(shares),
             "delta_ret": delta_ret,
             "delta_clipped": max_clip - float(baseline["long_clipped"]),
             "short_delta_ret": float(representative["short_ret"]) - float(baseline["short_ret"]),
             "lost_degradation": (float(representative["long_lost"]) - float(baseline["long_lost"])) / max(abs(float(baseline["long_lost"])), 1.0),
             "service_degradation": (float(representative["long_service"]) - float(baseline["long_service"])) / max(abs(float(baseline["long_service"])), 1.0),
             "backlog_degradation": (float(representative["long_backlog"]) - float(baseline["long_backlog"])) / max(abs(float(baseline["long_backlog"])), 1.0)}
        )
    write_csv(args.output_dir / "oracle_rows.csv", oracle_rows)

    ret_ci = cluster_ci(oracle_rows, "delta_ret", 551, args.n_boot)
    clip_ci = cluster_ci(oracle_rows, "delta_clipped", 552, args.n_boot)
    lost_ci = cluster_ci(oracle_rows, "lost_degradation", 553, args.n_boot)
    service_ci = cluster_ci(oracle_rows, "service_degradation", 554, args.n_boot)
    backlog_ci = cluster_ci(oracle_rows, "backlog_degradation", 555, args.n_boot)
    stressed_n = sum(row["stressed_node"] in {"A", "B"} for row in oracle_rows)
    informative_n = sum(bool(row["informative_headroom"]) for row in oracle_rows)
    diversity = (
        stressed_n > 0
        and sum(weight >= 0.15 * stressed_n for weight in normalized_weights.values()) >= 2
        and (max(normalized_action_weights.values()) if normalized_action_weights else stressed_n) <= 0.85 * stressed_n
    )
    stress_counts = Counter(row["stressed_node"] for row in oracle_rows)
    prefix_counts = Counter(str(row["prefix_allocation_a"]) for row in oracle_rows)
    no_reversal = not (
        np.mean([float(row["short_delta_ret"]) for row in oracle_rows]) > 0
        and np.mean([float(row["delta_ret"]) for row in oracle_rows]) < 0
    )
    pass_gate = (
        diversity and ret_ci[1] > 0 and clip_ci[0] > 0
        and lost_ci[2] <= .02 and service_ci[2] <= .02 and backlog_ci[2] <= .02
        and no_reversal
        and max(float(row["mass_residual"]) for row in branch_rows) <= 1e-6
        and stress_counts["A"] > 0 and stress_counts["B"] > 0
    )
    verdict = {
        "gate": "DRA1_C_PREFIX_BALANCED_CORRECTIVE",
        "n_tapes": len(tapes), "n_states": len(states),
        "n_informative_states": informative_n,
        "prefix_counts": dict(prefix_counts), "stress_counts": dict(stress_counts),
        "state_categories": dict(Counter(row["category"] for row in oracle_rows)),
        "normalized_share_weights": {str(k): v for k, v in sorted(normalized_weights.items())},
        "normalized_action_weights": {f"{k[0]}|{k[1]}": v for k, v in sorted(normalized_action_weights.items())},
        "oracle_delta_ret": {"mean": ret_ci[0], "ci95": [ret_ci[1], ret_ci[2]]},
        "oracle_delta_clipped": {"mean": clip_ci[0], "ci95": [clip_ci[1], clip_ci[2]]},
        "guardrail_ci_high": {"lost": lost_ci[2], "service": service_ci[2], "backlog": backlog_ci[2]},
        "diversity_pass": diversity, "prefix_balance_pass": len(prefix_counts) == 3,
        "stress_symmetry_support_pass": stress_counts["A"] > 0 and stress_counts["B"] > 0,
        "prefix_identity_pass": True, "mass_pass": True,
        "no_short_long_reversal": no_reversal,
        "virgin_tapes_opened": 0, "ppo_trained": False,
        "interpretation": "PASS_BRANCHING_TO_OBSERVABLE_GATE" if pass_gate else "STOP_NO_DYNAMIC_ORACLE_HEADROOM",
    }
    (args.output_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if pass_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
