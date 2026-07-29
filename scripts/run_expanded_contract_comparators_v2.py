#!/usr/bin/env python3
"""Corrective expanded-contract comparator experiment.

This is a new instrument.  It never overwrites the v1 diagnostic result.

Key guarantees:
* enumerates all 6^3 static postures;
* materializes policy-independent demand/risk tapes;
* every MPC branch replays the realized prefix and must match a frozen state hash;
* DDMRP is projected onto the same 6^3 action domain;
* raw tape rows, action/state traces, candidate values, and paired intervals persist;
* a greedy perfect-information policy is reported as a best-found diagnostic, never
  as an exact dynamic ceiling.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.expanded_contract_controllers_v2 import (  # noqa: E402
    ALL_POSTURES,
    NODES,
    Posture,
    ProjectedDDMRPController,
    posture_name,
    posture_targets,
)
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

FAMILIES = {
    "R1r": ("R11", "R12", "R13", "R14"),
    "R2r": ("R21", "R22", "R23", "R24"),
}
SCHEMA = "expanded_contract_comparators_v2"


def canonical_sha(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def file_sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def risk_row(event: Any) -> dict[str, Any]:
    return {
        "risk_id": str(event.risk_id),
        "start_time": float(event.start_time),
        "end_time": float(event.end_time),
        "duration": float(event.duration),
        "affected_ops": [int(op) for op in event.affected_ops],
        "description": str(event.description or ""),
        "magnitude": float(event.magnitude),
        "unit": str(event.unit or "incidents"),
        "affected_cssu": getattr(event, "affected_cssu", None),
    }


def order_row(order: Any) -> dict[str, Any]:
    return {
        "j": int(order.j),
        "OPTj": float(order.OPTj),
        "Q": float(order.quantity),
        "contingent": bool(order.contingent),
    }


def make_generated_sim(seed: int, horizon: float, family: str) -> MFSCSimulation:
    return MFSCSimulation(
        shifts=1,
        initial_buffers={node: 0.0 for node in NODES},
        inventory_replenishment_period=168.0,
        seed=seed,
        horizon=horizon,
        risks_enabled=True,
        risk_level="current",
        enabled_risks=set(FAMILIES[family]),
        risk_overrides={risk: "increased" for risk in FAMILIES[family]},
        strict_exogenous_crn=True,
        year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"],
        r14_defect_mode=P["r14_defect_mode"],
    )


def materialize_tape(seed: int, horizon: float, family: str) -> dict[str, Any]:
    sim = make_generated_sim(seed, horizon, family)
    sim.inventory_buffer_targets.update(posture_targets((168, 168, 168)))
    sim.step(action=None, step_hours=horizon)
    orders = [order_row(order) for order in sim.orders]
    risks = [risk_row(event) for event in sim.risk_events]
    payload = {
        "seed": int(seed),
        "family": family,
        "horizon": float(horizon),
        "orders": orders,
        "risks": risks,
    }
    payload["sha256"] = canonical_sha(payload)
    return payload


def make_replay_sim(
    *,
    seed: int,
    horizon: float,
    family: str,
    tape: dict[str, Any],
) -> MFSCSimulation:
    return MFSCSimulation(
        shifts=1,
        initial_buffers={node: 0.0 for node in NODES},
        inventory_replenishment_period=168.0,
        seed=seed,
        horizon=horizon,
        risks_enabled=True,
        risk_level="current",
        enabled_risks=set(FAMILIES[family]),
        risk_overrides={risk: "increased" for risk in FAMILIES[family]},
        strict_exogenous_crn=True,
        year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"],
        r14_defect_mode=P["r14_defect_mode"],
        demand_source="excel_order_tape",
        excel_order_tape=list(tape["orders"]),
        risk_event_tape=list(tape["risks"]),
    )


def state_payload(sim: MFSCSimulation) -> dict[str, Any]:
    pending = sorted(
        (
            int(order.j),
            round(float(order.remaining_qty), 9),
            round(float(order.in_flight_qty), 9),
            bool(order.backorder),
            bool(order.lost),
        )
        for order in sim.pending_backorders
    )
    return {
        "time": round(float(sim.env.now), 9),
        "inventory": {
            key: round(float(value), 9)
            for key, value in sim._inventory_detail().items()
        },
        "targets": {
            key: round(float(sim.inventory_buffer_targets.get(key, 0.0)), 9)
            for key in NODES
        },
        "pending": pending,
        "op_down": {str(k): int(v) for k, v in sorted(sim.op_down_count.items())},
        "raw_in_transit": round(float(sim._raw_material_in_transit), 9),
        "ration_in_transit": round(float(sim._in_transit), 9),
        "pending_batch": round(float(sim._pending_batch), 9),
        "demanded": round(float(sim.total_demanded), 9),
        "fulfilled": round(float(sim.total_order_fulfilled), 9),
        "produced": round(float(sim.total_produced), 9),
        "strategic_raw": round(float(sim.total_strategic_raw_injected), 9),
        "strategic_rations": round(float(sim.total_strategic_rations_injected), 9),
        "orders": len(sim.orders),
        "risk_events": len(sim.risk_events),
    }


def state_hash(sim: MFSCSimulation) -> str:
    return canonical_sha(state_payload(sim))


def splice_tapes(
    actual: dict[str, Any], future: dict[str, Any], branch_time: float
) -> dict[str, Any]:
    """Use the realized tape through the branch and a scenario tape afterwards."""
    orders = [
        dict(row) for row in actual["orders"]
        if float(row["OPTj"]) <= branch_time + 1e-9
    ]
    orders.extend(
        dict(row) for row in future["orders"]
        if float(row["OPTj"]) > branch_time + 1e-9
    )
    orders.sort(key=lambda row: (float(row["OPTj"]), int(row["j"])))
    for index, row in enumerate(orders, start=1):
        row["j"] = index

    risks = [
        dict(row) for row in actual["risks"]
        if float(row["start_time"]) <= branch_time + 1e-9
    ]
    risks.extend(
        dict(row) for row in future["risks"]
        if float(row["start_time"]) > branch_time + 1e-9
    )
    risks.sort(key=lambda row: (float(row["start_time"]), str(row["risk_id"])))
    return {
        "seed": int(future["seed"]),
        "family": actual["family"],
        "horizon": actual["horizon"],
        "orders": orders,
        "risks": risks,
        "sha256": canonical_sha({"orders": orders, "risks": risks}),
    }


def apply_posture(sim: MFSCSimulation, posture: Posture) -> None:
    sim.inventory_buffer_targets.update(posture_targets(posture))


def replay_prefix(
    *,
    tape: dict[str, Any],
    seed: int,
    horizon: float,
    family: str,
    prefix: Sequence[Posture],
    epoch_hours: float,
) -> MFSCSimulation:
    sim = make_replay_sim(seed=seed, horizon=horizon, family=family, tape=tape)
    for posture in prefix:
        apply_posture(sim, posture)
        sim.step(action=None, step_hours=min(epoch_hours, horizon - sim.env.now))
    return sim


def finish_with_posture(
    sim: MFSCSimulation,
    posture: Posture,
    horizon: float,
    epoch_hours: float,
) -> dict[str, float]:
    while float(sim.env.now) < horizon - 1e-9:
        apply_posture(sim, posture)
        sim.step(
            action=None,
            step_hours=min(epoch_hours, horizon - float(sim.env.now)),
        )
    return episode_row(sim)


def episode_row(sim: MFSCSimulation) -> dict[str, float]:
    metric = compute_episode_metrics(sim)
    return {
        "ret_excel": float(metric["ret_excel"]),
        "ret_excel_full_ledger": float(metric["ret_excel_full_ledger"]),
        "ret_thesis": float(metric["ret_thesis"]),
        "flow_fill_rate": float(metric["flow_fill_rate"]),
        "lost_orders": float(metric["lost_orders"]),
        "delivered_rations": float(metric["delivered_rations"]),
        "unresolved": float(metric.get("unresolved_orders", metric.get("unresolved", 0.0))),
        "strategic_injected": float(
            sim.total_strategic_raw_injected
            + sim.total_strategic_rations_injected
        ),
        "terminal_stock": float(sum(sim._inventory_detail().values())),
    }


def run_static(
    posture: Posture,
    tape: dict[str, Any],
    horizon: float,
    family: str,
    epoch_hours: float,
) -> dict[str, Any]:
    sim = make_replay_sim(
        seed=int(tape["seed"]), horizon=horizon, family=family, tape=tape
    )
    while float(sim.env.now) < horizon - 1e-9:
        apply_posture(sim, posture)
        sim.step(
            action=None,
            step_hours=min(epoch_hours, horizon - float(sim.env.now)),
        )
    return {
        "family": family,
        "tape_seed": int(tape["seed"]),
        "arm": posture_name(posture),
        "posture": list(posture),
        **episode_row(sim),
    }


def run_ddmrp(
    tape: dict[str, Any],
    horizon: float,
    family: str,
    epoch_hours: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sim = make_replay_sim(
        seed=int(tape["seed"]), horizon=horizon, family=family, tape=tape
    )
    controller = ProjectedDDMRPController()
    trace: list[dict[str, Any]] = []
    epoch = 0
    while float(sim.env.now) < horizon - 1e-9:
        before = state_hash(sim)
        targets = controller.act(sim, epoch)
        posture = tuple(controller.last_diagnostic["posture"])
        apply_posture(sim, posture)  # type: ignore[arg-type]
        trace.append(
            {
                "epoch": epoch,
                "time": float(sim.env.now),
                "state_hash": before,
                "posture": list(posture),
                "targets": targets,
                "ddmrp": controller.last_diagnostic,
            }
        )
        sim.step(
            action=None,
            step_hours=min(epoch_hours, horizon - float(sim.env.now)),
        )
        epoch += 1
    return (
        {
            "family": family,
            "tape_seed": int(tape["seed"]),
            "arm": controller.name,
            **episode_row(sim),
        },
        trace,
    )


@dataclass
class BranchEvaluation:
    posture: list[int]
    scenario_seed: int
    scenario_tape_sha256: str
    replay_state_hash: str
    actual_state_hash: str
    state_hash_match: bool
    value: float


def branch_value(
    *,
    actual_tape: dict[str, Any],
    future_tape: dict[str, Any],
    actual_hash: str,
    prefix: Sequence[Posture],
    candidate: Posture,
    epoch_hours: float,
    horizon: float,
    family: str,
    metric: str,
) -> BranchEvaluation:
    branch_time = len(prefix) * epoch_hours
    hybrid = splice_tapes(actual_tape, future_tape, branch_time)
    replay = replay_prefix(
        tape=hybrid,
        seed=int(actual_tape["seed"]),
        horizon=horizon,
        family=family,
        prefix=prefix,
        epoch_hours=epoch_hours,
    )
    replay_hash = state_hash(replay)
    matched = replay_hash == actual_hash
    if not matched:
        raise RuntimeError(
            "PREFIX_STATE_HASH_MISMATCH "
            f"family={family} tape={actual_tape['seed']} epoch={len(prefix)} "
            f"expected={actual_hash} observed={replay_hash}"
        )
    result = finish_with_posture(replay, candidate, horizon, epoch_hours)
    return BranchEvaluation(
        posture=list(candidate),
        scenario_seed=int(future_tape["seed"]),
        scenario_tape_sha256=str(hybrid["sha256"]),
        replay_state_hash=replay_hash,
        actual_state_hash=actual_hash,
        state_hash_match=matched,
        value=float(result[metric]),
    )


def run_replay_controller(
    *,
    actual_tape: dict[str, Any],
    scenario_tapes: Sequence[dict[str, Any]],
    candidates: Sequence[Posture],
    horizon: float,
    family: str,
    epoch_hours: float,
    metric: str,
    perfect_information: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sim = make_replay_sim(
        seed=int(actual_tape["seed"]),
        horizon=horizon,
        family=family,
        tape=actual_tape,
    )
    prefix: list[Posture] = []
    trace: list[dict[str, Any]] = []
    while float(sim.env.now) < horizon - 1e-9:
        actual_hash = state_hash(sim)
        futures = [actual_tape] if perfect_information else list(scenario_tapes)
        candidate_rows: list[dict[str, Any]] = []
        for candidate in candidates:
            evaluations = [
                branch_value(
                    actual_tape=actual_tape,
                    future_tape=future,
                    actual_hash=actual_hash,
                    prefix=prefix,
                    candidate=candidate,
                    epoch_hours=epoch_hours,
                    horizon=horizon,
                    family=family,
                    metric=metric,
                )
                for future in futures
            ]
            values = [row.value for row in evaluations]
            candidate_rows.append(
                {
                    "posture": list(candidate),
                    "mean_value": float(np.mean(values)),
                    "scenario_values": [asdict(row) for row in evaluations],
                }
            )
        candidate_rows.sort(
            key=lambda row: (-float(row["mean_value"]), tuple(row["posture"]))
        )
        selected = tuple(candidate_rows[0]["posture"])
        trace.append(
            {
                "epoch": len(prefix),
                "time": float(sim.env.now),
                "actual_state_hash": actual_hash,
                "selected_posture": list(selected),
                "selected_value": float(candidate_rows[0]["mean_value"]),
                "runner_up_value": float(candidate_rows[1]["mean_value"]),
                "candidate_rows": candidate_rows,
            }
        )
        prefix.append(selected)  # type: ignore[arg-type]
        apply_posture(sim, selected)  # type: ignore[arg-type]
        sim.step(
            action=None,
            step_hours=min(epoch_hours, horizon - float(sim.env.now)),
        )
    arm = "greedy_pi_best_found_v2" if perfect_information else "replay_mpc_v2"
    return (
        {
            "family": family,
            "tape_seed": int(actual_tape["seed"]),
            "arm": arm,
            "action_sequence": [list(row) for row in prefix],
            **episode_row(sim),
        },
        trace,
    )


def paired_interval(
    treatment: Sequence[float], control: Sequence[float], seed: int
) -> dict[str, Any]:
    delta = np.asarray(treatment, dtype=float) - np.asarray(control, dtype=float)
    rng = np.random.default_rng(seed)
    draws = delta[
        rng.integers(0, len(delta), size=(20_000, len(delta)))
    ].mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "positive_tapes": int((delta > 0).sum()),
        "n_tapes": int(len(delta)),
        "deltas": delta.tolist(),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "full"), default="preflight")
    parser.add_argument("--families", nargs="+", default=["R1r", "R2r"])
    parser.add_argument("--tapes", type=int)
    parser.add_argument("--scenarios", type=int)
    parser.add_argument("--seed-start", type=int, default=1_410_001)
    parser.add_argument("--horizon-weeks", type=int, default=52)
    parser.add_argument("--epoch-weeks", type=int, default=4)
    parser.add_argument("--metric", default="ret_excel")
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--skip-dynamic", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/expanded_contract_comparators_v2"),
    )
    args = parser.parse_args()
    tapes_n = args.tapes or (1 if args.phase == "preflight" else 12)
    scenarios_n = args.scenarios or (1 if args.phase == "preflight" else 5)
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    epoch_hours = float(args.epoch_weeks * HOURS_PER_WEEK)
    output = args.output_dir / args.phase
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()
    controller_path = (
        script_path.parent.parent
        / "supply_chain"
        / "expanded_contract_controllers_v2.py"
    )
    opened = {
        "schema": SCHEMA,
        "phase": args.phase,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "runner_sha256": file_sha(script_path),
        "controller_sha256": file_sha(controller_path),
        "families": args.families,
        "tapes": tapes_n,
        "scenarios": scenarios_n,
        "seed_start": args.seed_start,
        "horizon_weeks": args.horizon_weeks,
        "epoch_weeks": args.epoch_weeks,
        "metric": args.metric,
        "static_posture_count": len(ALL_POSTURES),
        "confirmation_roots_opened": False,
        "claim_status": "DEVELOPMENT_INSTRUMENT",
    }
    write_json(output / "opening_receipt.json", opened)

    started = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    all_traces: dict[str, Any] = {}
    family_results: dict[str, Any] = {}

    for family_index, family in enumerate(args.families):
        roots = [
            args.seed_start + family_index * 100_000 + index
            for index in range(tapes_n)
        ]
        actual_tapes = [materialize_tape(root, horizon, family) for root in roots]
        future_by_root = {
            root: [
                materialize_tape(root + (scenario + 1) * 10_000, horizon, family)
                for scenario in range(scenarios_n)
            ]
            for root in roots
        }
        write_json(output / f"{family}_actual_tapes.json", actual_tapes)

        static_rows: list[dict[str, Any]] = []
        for posture_index, posture in enumerate(ALL_POSTURES):
            for tape in actual_tapes:
                static_rows.append(
                    run_static(posture, tape, horizon, family, epoch_hours)
                )
            if posture_index % 24 == 0:
                print(
                    f"{family} static {posture_index + 1}/{len(ALL_POSTURES)}",
                    flush=True,
                )
        all_rows.extend(static_rows)
        posture_means: list[tuple[Posture, float]] = []
        for posture in ALL_POSTURES:
            values = [
                row[args.metric]
                for row in static_rows
                if tuple(row["posture"]) == posture
            ]
            posture_means.append((posture, float(np.mean(values))))
        posture_means.sort(key=lambda pair: (-pair[1], pair[0]))
        incumbent = posture_means[0][0]

        if args.candidate_limit > 0:
            candidates = tuple(
                posture for posture, _ in posture_means[: args.candidate_limit]
            )
            if incumbent not in candidates:
                candidates = (incumbent, *candidates)
        else:
            candidates = ALL_POSTURES

        ddmrp_rows: list[dict[str, Any]] = []
        mpc_rows: list[dict[str, Any]] = []
        pi_rows: list[dict[str, Any]] = []
        for tape in actual_tapes:
            ddmrp, ddmrp_trace = run_ddmrp(
                tape, horizon, family, epoch_hours
            )
            ddmrp_rows.append(ddmrp)
            all_traces[f"{family}:{tape['seed']}:ddmrp"] = ddmrp_trace
            if args.skip_dynamic:
                continue
            mpc, mpc_trace = run_replay_controller(
                actual_tape=tape,
                scenario_tapes=future_by_root[int(tape["seed"])],
                candidates=candidates,
                horizon=horizon,
                family=family,
                epoch_hours=epoch_hours,
                metric=args.metric,
                perfect_information=False,
            )
            pi, pi_trace = run_replay_controller(
                actual_tape=tape,
                scenario_tapes=(),
                candidates=candidates,
                horizon=horizon,
                family=family,
                epoch_hours=epoch_hours,
                metric=args.metric,
                perfect_information=True,
            )
            mpc_rows.append(mpc)
            pi_rows.append(pi)
            all_traces[f"{family}:{tape['seed']}:mpc"] = mpc_trace
            all_traces[f"{family}:{tape['seed']}:pi"] = pi_trace
            print(f"{family} dynamic tape {tape['seed']} complete", flush=True)
        all_rows.extend(ddmrp_rows + mpc_rows + pi_rows)

        incumbent_values = [
            row[args.metric]
            for row in static_rows
            if tuple(row["posture"]) == incumbent
        ]
        comparisons: dict[str, Any] = {}
        for arm_rows in (ddmrp_rows, mpc_rows, pi_rows):
            if not arm_rows:
                continue
            arm = str(arm_rows[0]["arm"])
            comparisons[arm] = paired_interval(
                [row[args.metric] for row in arm_rows],
                incumbent_values,
                seed=args.seed_start + len(comparisons),
            )

        pi_best_actions = {
            tuple(trace_row["selected_posture"])
            for key, trace in all_traces.items()
            if key.startswith(f"{family}:") and key.endswith(":pi")
            for trace_row in trace
        }
        family_results[family] = {
            "roots": roots,
            "incumbent_posture": list(incumbent),
            "incumbent_name": posture_name(incumbent),
            "incumbent_mean": posture_means[0][1],
            "runner_up_posture": list(posture_means[1][0]),
            "runner_up_mean": posture_means[1][1],
            "candidate_count": len(candidates),
            "comparisons": comparisons,
            "pi_action_ranking_reversal": len(pi_best_actions) >= 2,
            "pi_distinct_best_actions": [list(row) for row in sorted(pi_best_actions)],
        }

    write_json(output / "rows.json", all_rows)
    write_json(output / "traces.json", all_traces)
    result = {
        **opened,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "family_results": family_results,
        "row_count": len(all_rows),
        "trace_count": len(all_traces),
        "all_prefix_state_hashes_match": all(
            scenario["state_hash_match"]
            for trace in all_traces.values()
            for epoch in trace
            for candidate in epoch.get("candidate_rows", [])
            for scenario in candidate.get("scenario_values", [])
        ),
        "dynamic_headroom_label": "GREEDY_PI_BEST_FOUND_NOT_EXACT_CEILING",
        "confirmation_roots_opened": False,
    }
    result["self_sha256"] = canonical_sha(result)
    write_json(output / "result.json", result)
    completion = {
        "result_sha256": file_sha(output / "result.json"),
        "rows_sha256": file_sha(output / "rows.json"),
        "traces_sha256": file_sha(output / "traces.json"),
        "status": "COMPLETE",
        "confirmation_roots_opened": False,
    }
    write_json(output / "completion_receipt.json", completion)
    print(json.dumps(family_results, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

