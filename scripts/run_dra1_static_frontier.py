#!/usr/bin/env python3
"""DRA-1 Gate B: nine-policy same-contract static frontier.

The runner materializes calibration-only exogenous calendars once, localizes
R22/R23/R24 by an event-keyed hash, and replays identical demand/risk futures
under every allocation x service-rule constant.  It opens no observable-policy
holdout or PPO-confirmatory tape.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import HOURS_PER_DAY, HOURS_PER_WEEK, SIMULATION_HORIZON  # noqa: E402
from supply_chain.cssu_allocation import ALLOCATION_LEVELS, SERVICE_RULES  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402


PROXY = Path("supply_chain/data/garrido_proxy_v1_freeze_2026-07-10.json")
DEFAULT_OUTPUT = Path("results/program_d/dra1_static_frontier")
FAMILIES = ("nominal", "r22_localized", "r23_localized", "r24_mixed")
THESIS_ACTION = (0.50, "SPT_FULL")


def digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def local_target(seed: int, event_index: int, risk_id: str) -> str:
    raw = sha256(f"dra1-local-v1:{seed}:{event_index}:{risk_id}".encode()).digest()
    return "A" if raw[0] & 1 == 0 else "B"


def proxy_kwargs() -> dict[str, Any]:
    payload = json.loads(PROXY.read_text(encoding="utf-8"))
    kwargs = dict(payload["sim_kwargs"])
    kwargs.pop("risk_level", None)
    kwargs.pop("seed_stream_mode", None)
    return kwargs


def materialize_tape(seed: int, family: str, horizon_weeks: int) -> dict[str, Any]:
    if family not in FAMILIES:
        raise ValueError(family)
    horizon = max(float(SIMULATION_HORIZON), 8_000 + horizon_weeks * HOURS_PER_WEEK)
    sim = MFSCSimulation(
        seed=seed, horizon=horizon, risks_enabled=False,
        strict_exogenous_crn=True, **proxy_kwargs(),
    )
    sim._start_processes()
    while not sim.warmup_complete:
        sim.env.run(until=min(sim.env.now + 1.0, sim.horizon))
    start = float(sim.env.now)
    processes = []
    if family == "r22_localized":
        processes = [sim._risk_R22]
    elif family == "r23_localized":
        processes = [sim._risk_R23]
    elif family == "r24_mixed":
        processes = [sim._risk_R21, sim._risk_R22, sim._risk_R23, sim._risk_R24]
    for process in processes:
        sim.env.process(process())
    end = start + horizon_weeks * HOURS_PER_WEEK
    sim.env.run(until=end)

    rows: list[dict[str, Any]] = []
    for idx, event in enumerate(sim.risk_events):
        if event.start_time < start or event.start_time >= end:
            continue
        affected_ops = [int(op) for op in event.affected_ops]
        affected_cssu = None
        if event.risk_id == "R22":
            # DRA-1 is the downstream spatial extension. Preserve occurrence,
            # duration and intensity; map the thesis LOC draw onto Op10/12.
            op = affected_ops[0] if affected_ops else 10
            affected_ops = [10 if op in {4, 10} else 12]
            affected_cssu = local_target(seed, idx, "R22")
        elif event.risk_id in {"R23", "R24"}:
            affected_cssu = local_target(seed, idx, event.risk_id)
        rows.append(
            {
                "risk_id": str(event.risk_id),
                "start_time": float(event.start_time - start),
                "end_time": float(min(event.end_time, end) - start),
                "duration": float(min(event.end_time, end) - event.start_time),
                "affected_ops": affected_ops,
                "affected_cssu": affected_cssu,
                "description": str(event.description),
                "magnitude": float(event.magnitude),
                "unit": str(event.unit),
            }
        )
    rows.sort(key=lambda row: (row["start_time"], row["risk_id"]))
    return {
        "tape_id": f"dra1-cal-{family}-{seed}",
        "split": "calibration",
        "family": family,
        "seed": seed,
        "horizon_weeks": horizon_weeks,
        "risk_events": rows,
        "risk_sha256": digest(rows),
    }


def exogenous_hashes(sim: MFSCSimulation, start: float) -> tuple[str, str]:
    risks = [
        {
            "risk_id": e.risk_id,
            "start": round(float(e.start_time - start), 9),
            "duration": round(float(e.duration), 9),
            "ops": list(map(int, e.affected_ops)),
            "cssu": e.affected_cssu,
            "magnitude": round(float(e.magnitude), 9),
        }
        for e in sim.risk_events if e.start_time >= start - 1e-9
    ]
    demand = [
        (round(float(t - start), 9), round(float(q), 9))
        for t, q in sim.daily_demand if t >= start - 1e-9
    ]
    return digest(risks), digest(demand)


def run_constant(tape: dict[str, Any], allocation: float, rule: str) -> dict[str, Any]:
    horizon = max(float(SIMULATION_HORIZON), 8_000 + tape["horizon_weeks"] * HOURS_PER_WEEK)
    sim = MFSCSimulation(
        seed=int(tape["seed"]), horizon=horizon, risks_enabled=False,
        strict_exogenous_crn=True, cssu_topology_mode="split_v1",
        cssu_allocation_a=float(allocation), cssu_service_rule=rule,
        **proxy_kwargs(),
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
    end = start + int(tape["horizon_weeks"]) * HOURS_PER_WEEK
    backlog_auc = 0.0
    while sim.env.now < end - 1e-9:
        step_end = min(end, sim.env.now + HOURS_PER_DAY)
        sim.env.run(until=step_end)
        backlog_auc += float(sim.pending_backorder_qty) * (step_end - max(start, step_end - HOURS_PER_DAY))
    metrics = compute_episode_metrics(sim, treatment_start=start)
    risk_hash, demand_hash = exogenous_hashes(sim, start)
    ledger = sim.flow_ledger()
    metrics.update(
        {
            "backlog_auc": float(backlog_auc),
            "risk_sha256": risk_hash,
            "demand_sha256": demand_hash,
            "mass_residual": max(abs(float(ledger["raw_residual"])), abs(float(ledger["ration_residual"]))),
            "allocation_live_epochs": int(sim.cssu_allocation_live_epochs),
            "allocation_moot_epochs": int(sim.cssu_allocation_moot_epochs),
            "dispatch_a": float(sim.cssu_dispatched["A"]),
            "dispatch_b": float(sim.cssu_dispatched["B"]),
        }
    )
    return metrics


def boot_ci(values: list[float], seed: int, n_boot: int) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed-start", type=int, default=830001)
    parser.add_argument("--n-tapes", type=int, default=60)
    parser.add_argument("--horizon-weeks", type=int, default=104)
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args()
    if args.n_tapes % 4:
        raise ValueError("n-tapes must be divisible by four")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tapes = [
        materialize_tape(args.seed_start + i, FAMILIES[i // (args.n_tapes // 4)], args.horizon_weeks)
        for i in range(args.n_tapes)
    ]
    (args.output_dir / "calibration_tapes.json").write_text(
        json.dumps(tapes, indent=2, sort_keys=True), encoding="utf-8"
    )
    rows: list[dict[str, Any]] = []
    for tape in tapes:
        expected_hashes = None
        for allocation in ALLOCATION_LEVELS:
            for rule in SERVICE_RULES:
                metrics = run_constant(tape, allocation, rule)
                pair = (metrics["risk_sha256"], metrics["demand_sha256"])
                if expected_hashes is None:
                    expected_hashes = pair
                elif pair != expected_hashes:
                    raise RuntimeError(f"FAIL_EXOGENOUS_CRN {tape['tape_id']}")
                rows.append(
                    {"tape_id": tape["tape_id"], "family": tape["family"],
                     "seed": tape["seed"], "allocation_a": allocation,
                     "service_rule": rule, **metrics}
                )
        print(f"[dra1-static] {tape['tape_id']} complete", flush=True)
    write_csv(args.output_dir / "policy_tape_rows.csv", rows)

    baseline = {
        row["tape_id"]: row for row in rows
        if (float(row["allocation_a"]), row["service_rule"]) == THESIS_ACTION
    }
    summary: list[dict[str, Any]] = []
    for allocation in ALLOCATION_LEVELS:
        for rule in SERVICE_RULES:
            selected = [r for r in rows if float(r["allocation_a"]) == allocation and r["service_rule"] == rule]
            ret = boot_ci([float(r["ret_excel"]) for r in selected], 991, args.n_boot)
            clip = boot_ci([float(r["ret_excel_visible_clipped_0_1"]) for r in selected], 992, args.n_boot)
            guardrails = {}
            for key, metric in (("lost", "lost_orders"), ("service", "service_loss_auc_ration_hours"), ("backlog", "backlog_auc")):
                degradation = [
                    (float(r[metric]) - float(baseline[r["tape_id"]][metric])) /
                    max(abs(float(baseline[r["tape_id"]][metric])), 1.0)
                    for r in selected
                ]
                guardrails[key] = boot_ci(degradation, 1000 + len(summary), args.n_boot)
            admissible = (
                max(float(r["mass_residual"]) for r in selected) <= 1e-6
                and all(guardrails[key][2] <= 0.02 for key in guardrails)
            )
            summary.append(
                {"allocation_a": allocation, "service_rule": rule,
                 "mean_ret": ret[0], "ret_ci_low": ret[1], "ret_ci_high": ret[2],
                 "mean_ret_clipped": clip[0], "clipped_ci_low": clip[1], "clipped_ci_high": clip[2],
                 "lost_deg_ci_high": guardrails["lost"][2],
                 "service_deg_ci_high": guardrails["service"][2],
                 "backlog_deg_ci_high": guardrails["backlog"][2],
                 "mean_live_epochs": float(np.mean([r["allocation_live_epochs"] for r in selected])),
                 "admissible": admissible}
            )
    write_csv(args.output_dir / "policy_summary.csv", summary)
    admissible = [row for row in summary if row["admissible"]]
    best = max(admissible, key=lambda row: row["mean_ret"]) if admissible else None
    verdict = {
        "gate": "DRA1_B_STATIC_FRONTIER",
        "n_tapes": len(tapes), "virgin_tapes_opened": 0, "ppo_trained": False,
        "crn_pass": True, "mass_pass": max(float(r["mass_residual"]) for r in rows) <= 1e-6,
        "allocation_live": sum(float(r["allocation_live_epochs"]) for r in rows) > 0,
        "best_admissible": best,
        "interpretation": "PASS_STATIC_FRONTIER" if best else "FAIL_NO_ADMISSIBLE_STATIC",
    }
    (args.output_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if best else 2


if __name__ == "__main__":
    raise SystemExit(main())
