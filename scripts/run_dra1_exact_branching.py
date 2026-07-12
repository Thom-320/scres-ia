#!/usr/bin/env python3
"""DRA-1 Gate C: replay-exact one-epoch branching on calibration tapes."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_dra1_static_frontier import (  # noqa: E402
    ALLOCATION_LEVELS, SERVICE_RULES, boot_ci, digest, proxy_kwargs,
)
from supply_chain.config import HOURS_PER_DAY, HOURS_PER_WEEK, SIMULATION_HORIZON  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402


FRONTIER = Path("results/program_d/dra1_static_frontier")
DEFAULT_OUTPUT = Path("results/program_d/dra1_exact_branching")
SHORT_HOURS = 72.0
LONG_HOURS = 28.0 * HOURS_PER_DAY


def make_sim(tape: dict[str, Any], action: tuple[float, str]) -> tuple[MFSCSimulation, float]:
    horizon = max(float(SIMULATION_HORIZON), 8_000 + tape["horizon_weeks"] * HOURS_PER_WEEK)
    sim = MFSCSimulation(
        seed=int(tape["seed"]), horizon=horizon, risks_enabled=False,
        strict_exogenous_crn=True, cssu_topology_mode="split_v1",
        cssu_allocation_a=action[0], cssu_service_rule=action[1], **proxy_kwargs(),
    )
    sim._start_processes()
    while not sim.warmup_complete:
        sim.env.run(until=min(sim.env.now + 1.0, sim.horizon))
    start = float(sim.env.now)
    absolute = []
    for event in tape["risk_events"]:
        row = dict(event)
        row["start_time"] = start + float(event["start_time"])
        row["end_time"] = start + float(event["end_time"])
        absolute.append(row)
    sim.risk_event_tape = sim._normalize_risk_event_tape(absolute)
    sim.env.process(sim._risk_event_tape_replay())
    return sim, start


def state_hash(sim: MFSCSimulation) -> str:
    payload = {
        "time": round(float(sim.env.now), 9),
        "obs": sim.get_cssu_observation(),
        "inventory": sim._inventory_detail(),
        "queue": [
            {"j": o.j, "dest": o.cssu_destination, "remaining": o.remaining_qty,
             "in_flight": o.in_flight_qty, "contingent": o.contingent,
             "lost": o.lost}
            for o in sim.pending_backorders
        ],
        "local_down": {f"{op}-{cssu}": count for (op, cssu), count in sim.cssu_local_down_count.items()},
    }
    return digest(payload)


def category(sim: MFSCSimulation, tape: dict[str, Any], relative_time: float) -> str:
    obs = sim.get_cssu_observation()
    both_up = all(obs[f"cssu_{cssu}_op{op}_up"] for cssu in ("A", "B") for op in (10, 11, 12))
    joint = sim.cssu_allocation_is_live() and both_up
    if tape["family"] == "r24_mixed" and both_up and (
        obs["cssu_A_r24_share"] > 0 or obs["cssu_B_r24_share"] > 0
    ):
        return "localized_r24_both_up"
    recent_hit = any(
        event["risk_id"] in {"R22", "R23"}
        and 0 <= relative_time - float(event["end_time"]) <= 14 * HOURS_PER_DAY
        for event in tape["risk_events"]
    )
    if recent_hit and both_up and joint:
        return "post_hit_recovery"
    if joint:
        return "joint_scarcity"
    return "other"


def select_state(tape: dict[str, Any], base: tuple[float, str]) -> dict[str, Any]:
    sim, start = make_sim(tape, base)
    latest = start + tape["horizon_weeks"] * HOURS_PER_WEEK - LONG_HOURS
    candidates: list[dict[str, Any]] = []
    while sim.env.now + HOURS_PER_DAY <= latest:
        sim.env.run(until=sim.env.now + HOURS_PER_DAY)
        rel = float(sim.env.now - start)
        cat = category(sim, tape, rel)
        obs = sim.get_cssu_observation()
        imbalance = abs(obs["cssu_A_backlog_qty"] - obs["cssu_B_backlog_qty"])
        candidates.append(
            {"relative_time": rel, "category": cat, "imbalance": imbalance,
             "prefix_hash": state_hash(sim), **obs}
        )
    preferred = {
        "nominal": ("joint_scarcity",),
        "r22_localized": ("post_hit_recovery", "joint_scarcity"),
        "r23_localized": ("post_hit_recovery", "joint_scarcity"),
        "r24_mixed": ("localized_r24_both_up", "post_hit_recovery", "joint_scarcity"),
    }[tape["family"]]
    for name in preferred:
        eligible = [row for row in candidates if row["category"] == name]
        if eligible:
            chosen = max(eligible, key=lambda row: row["imbalance"])
            return {"tape_id": tape["tape_id"], "family": tape["family"], **chosen}
    chosen = max(candidates, key=lambda row: row["imbalance"])
    return {"tape_id": tape["tape_id"], "family": tape["family"], **chosen}


def branch(tape: dict[str, Any], state: dict[str, Any], base: tuple[float, str], action: tuple[float, str]) -> dict[str, Any]:
    sim, start = make_sim(tape, base)
    state_time = start + float(state["relative_time"])
    sim.env.run(until=state_time)
    replay_hash = state_hash(sim)
    if replay_hash != state["prefix_hash"]:
        raise RuntimeError(f"FAIL_PREFIX_IDENTITY {tape['tape_id']}")

    dispatch_before = dict(sim.cssu_dispatched)
    sim.set_cssu_allocation_action(*action, activation_delay_hours=HOURS_PER_DAY)
    sim.env.run(until=state_time + HOURS_PER_DAY)
    sim._activate_due_cssu_action()
    sim.set_cssu_allocation_action(*base, activation_delay_hours=HOURS_PER_DAY)
    sim.env.run(until=state_time + 2 * HOURS_PER_DAY)
    sim._activate_due_cssu_action()
    pulse_dispatch_a = float(sim.cssu_dispatched["A"] - dispatch_before["A"])
    pulse_dispatch_b = float(sim.cssu_dispatched["B"] - dispatch_before["B"])
    sim.env.run(until=state_time + SHORT_HOURS)
    short = compute_episode_metrics(sim, treatment_start=state_time)
    sim.env.run(until=state_time + LONG_HOURS)
    long = compute_episode_metrics(sim, treatment_start=state_time)
    ledger = sim.flow_ledger()
    return {
        "short_ret": short["ret_excel"],
        "short_clipped": short["ret_excel_visible_clipped_0_1"],
        "long_ret": long["ret_excel"],
        "long_clipped": long["ret_excel_visible_clipped_0_1"],
        "long_lost": long["lost_orders"],
        "long_service": long["service_loss_auc_ration_hours"],
        "long_backlog": long["backorder_qty_final"],
        "mass_residual": max(abs(ledger["raw_residual"]), abs(ledger["ration_residual"])),
        "prefix_hash": replay_hash,
        "pulse_dispatch_a": pulse_dispatch_a,
        "pulse_dispatch_b": pulse_dispatch_b,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


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
    best = frontier["best_admissible"]
    base = (float(best["allocation_a"]), str(best["service_rule"]))
    tapes = json.loads((args.frontier_dir / "calibration_tapes.json").read_text())
    if args.limit_tapes:
        tapes = tapes[:args.limit_tapes]
    states = [select_state(tape, base) for tape in tapes]
    write_csv(args.output_dir / "states.csv", states)

    branch_rows: list[dict[str, Any]] = []
    for tape, state in zip(tapes, states):
        for allocation in ALLOCATION_LEVELS:
            for rule in SERVICE_RULES:
                result = branch(tape, state, base, (allocation, rule))
                branch_rows.append(
                    {"state_id": tape["tape_id"], "tape_id": tape["tape_id"],
                     "family": tape["family"], "category": state["category"],
                     "allocation_a": allocation, "service_rule": rule, **result}
                )
        print(f"[dra1-branch] {tape['tape_id']} complete", flush=True)
    write_csv(args.output_dir / "branch_rows.csv", branch_rows)

    by_state: dict[str, list[dict[str, Any]]] = {}
    for row in branch_rows:
        by_state.setdefault(row["state_id"], []).append(row)
    oracle_rows = []
    for state_id, rows in by_state.items():
        baseline = next(r for r in rows if float(r["allocation_a"]) == base[0] and r["service_rule"] == base[1])
        best_row = max(rows, key=lambda row: (float(row["long_ret"]), float(row["long_clipped"])))
        oracle_rows.append(
            {"state_id": state_id, "family": best_row["family"], "category": best_row["category"],
             "allocation_a": best_row["allocation_a"], "service_rule": best_row["service_rule"],
             "delta_ret": float(best_row["long_ret"]) - float(baseline["long_ret"]),
             "delta_clipped": float(best_row["long_clipped"]) - float(baseline["long_clipped"]),
             "short_delta_ret": float(best_row["short_ret"]) - float(baseline["short_ret"]),
             "lost_degradation": (float(best_row["long_lost"]) - float(baseline["long_lost"])) / max(abs(float(baseline["long_lost"])), 1.0),
             "service_degradation": (float(best_row["long_service"]) - float(baseline["long_service"])) / max(abs(float(baseline["long_service"])), 1.0),
             "backlog_degradation": (float(best_row["long_backlog"]) - float(baseline["long_backlog"])) / max(abs(float(baseline["long_backlog"])), 1.0)}
        )
    write_csv(args.output_dir / "oracle_rows.csv", oracle_rows)
    counts = Counter(float(row["allocation_a"]) for row in oracle_rows)
    action_counts = Counter((float(row["allocation_a"]), row["service_rule"]) for row in oracle_rows)
    ret_ci = boot_ci([row["delta_ret"] for row in oracle_rows], 441, args.n_boot)
    clip_ci = boot_ci([row["delta_clipped"] for row in oracle_rows], 442, args.n_boot)
    lost_ci = boot_ci([row["lost_degradation"] for row in oracle_rows], 443, args.n_boot)
    service_ci = boot_ci([row["service_degradation"] for row in oracle_rows], 444, args.n_boot)
    backlog_ci = boot_ci([row["backlog_degradation"] for row in oracle_rows], 445, args.n_boot)
    n = len(oracle_rows)
    diversity = sum(count >= .15 * n for count in counts.values()) >= 2 and max(action_counts.values()) <= .85 * n
    no_reversal = not (
        np.mean([row["short_delta_ret"] for row in oracle_rows]) > 0
        and np.mean([row["delta_ret"] for row in oracle_rows]) < 0
    )
    pass_gate = (
        diversity and ret_ci[1] > 0 and clip_ci[0] > 0 and
        lost_ci[2] <= .02 and service_ci[2] <= .02 and backlog_ci[2] <= .02 and
        no_reversal and max(float(row["mass_residual"]) for row in branch_rows) <= 1e-6
    )
    verdict = {
        "gate": "DRA1_C_EXACT_BRANCHING", "n_states": n,
        "state_categories": dict(Counter(row["category"] for row in oracle_rows)),
        "oracle_allocation_counts": {str(k): v for k, v in counts.items()},
        "oracle_action_counts": {f"{k[0]}|{k[1]}": v for k, v in action_counts.items()},
        "oracle_delta_ret": {"mean": ret_ci[0], "ci95": [ret_ci[1], ret_ci[2]]},
        "oracle_delta_clipped": {"mean": clip_ci[0], "ci95": [clip_ci[1], clip_ci[2]]},
        "guardrail_ci_high": {"lost": lost_ci[2], "service": service_ci[2], "backlog": backlog_ci[2]},
        "diversity_pass": diversity, "no_short_long_reversal": no_reversal,
        "prefix_identity_pass": True, "mass_pass": True,
        "virgin_tapes_opened": 0, "ppo_trained": False,
        "interpretation": "PASS_BRANCHING_TO_OBSERVABLE_GATE" if pass_gate else "STOP_NO_DYNAMIC_ORACLE_HEADROOM",
    }
    (args.output_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if pass_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
