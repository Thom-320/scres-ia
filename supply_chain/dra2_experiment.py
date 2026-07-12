"""Shared fail-closed experiment utilities for DRA-2."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

from .config import HOURS_PER_DAY, HOURS_PER_WEEK, SIMULATION_HORIZON
from .dra2_convoy import ConvoyThresholdPolicy
from .episode_metrics import compute_episode_metrics
from .supply_chain import MFSCSimulation


CONTRACT_ID = "op7_op8_finite_convoy_v1"
PROXY = Path(__file__).resolve().parent / "data" / "garrido_proxy_v1_freeze_2026-07-10.json"
FAMILIES = ("routine", "assembly_scarcity", "op8_interruption", "r24_mixed")


def digest(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def proxy_kwargs() -> dict[str, Any]:
    payload = json.loads(PROXY.read_text(encoding="utf-8"))
    kwargs = dict(payload["sim_kwargs"])
    kwargs.pop("risk_level", None)
    kwargs.pop("seed_stream_mode", None)
    return kwargs


def advance_including(sim: MFSCSimulation, target: float) -> None:
    """Advance while processing every event scheduled exactly at target."""
    target = float(target)
    while sim.env.peek() <= target:
        sim.env.step()
    if float(sim.env.now) < target:
        sim.env.run(until=target)


def materialize_tape(seed: int, family: str, horizon_weeks: int, split: str) -> dict[str, Any]:
    if family not in FAMILIES:
        raise ValueError(family)
    horizon = max(float(SIMULATION_HORIZON), 8_000 + horizon_weeks * HOURS_PER_WEEK)
    sim = MFSCSimulation(
        seed=seed, horizon=horizon, risks_enabled=False,
        strict_exogenous_crn=True, **proxy_kwargs(),
    )
    sim._start_processes()
    while not sim.warmup_complete:
        advance_including(sim, min(sim.env.now + 1.0, sim.horizon))
    start = float(sim.env.now)
    processes: list[Any] = []
    if family == "assembly_scarcity":
        processes = [sim._risk_R11, sim._risk_R21]
    elif family == "op8_interruption":
        processes = [sim._risk_R22]
    elif family == "r24_mixed":
        processes = [sim._risk_R21, sim._risk_R22, sim._risk_R24]
    for process in processes:
        sim.env.process(process())
    end = start + horizon_weeks * HOURS_PER_WEEK
    advance_including(sim, end)

    events: list[dict[str, Any]] = []
    for event in sim.risk_events:
        if event.start_time < start or event.start_time >= end:
            continue
        affected_ops = [int(op) for op in event.affected_ops]
        if event.risk_id == "R22":
            # Preserve occurrence/duration while targeting the DRA-2 Op8 link.
            affected_ops = [8]
        events.append(
            {
                "risk_id": str(event.risk_id),
                "start_time": float(event.start_time - start),
                "end_time": float(min(event.end_time, end) - start),
                "duration": float(min(event.end_time, end) - event.start_time),
                "affected_ops": affected_ops,
                "description": str(event.description),
                "magnitude": float(event.magnitude),
                "unit": str(event.unit),
            }
        )
    events.sort(key=lambda row: (row["start_time"], row["risk_id"]))
    tape = {
        "contract_id": CONTRACT_ID,
        "tape_id": f"dra2-{split}-{family}-{seed}",
        "split": split,
        "family": family,
        "seed": int(seed),
        "horizon_weeks": int(horizon_weeks),
        "risk_events": events,
        "face_validation": "PENDING",
    }
    tape["sha256"] = digest(tape)
    return tape


def make_sim(tape: dict[str, Any]) -> tuple[MFSCSimulation, float]:
    horizon = max(
        float(SIMULATION_HORIZON),
        8_000 + int(tape["horizon_weeks"]) * HOURS_PER_WEEK,
    )
    sim = MFSCSimulation(
        seed=int(tape["seed"]), horizon=horizon, risks_enabled=False,
        strict_exogenous_crn=True, op8_dispatch_mode="finite_convoy_v1",
        **proxy_kwargs(),
    )
    sim._start_processes()
    while not sim.warmup_complete:
        advance_including(sim, min(sim.env.now + 1.0, sim.horizon))
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


def resource_snapshot(sim: MFSCSimulation) -> dict[str, float]:
    return dict(sim.op8_convoy_metrics())


def resource_delta(sim: MFSCSimulation, before: dict[str, float]) -> dict[str, float]:
    current = sim.op8_convoy_metrics()
    result = {
        key: float(value) - float(before.get(key, 0.0))
        for key, value in current.items()
        if key not in {"op8_convoy_available", "op8_convoy_load_factor", "op8_convoy_resource_residual"}
    }
    capacity = max(result["op8_convoy_capacity_committed"], 1.0)
    result["op8_convoy_load_factor"] = result["op8_convoy_dispatched_rations"] / capacity
    result["op8_convoy_resource_residual"] = current["op8_convoy_resource_residual"]
    return result


def exogenous_hashes(sim: MFSCSimulation, start: float) -> dict[str, str]:
    risks = [
        {"id": e.risk_id, "start": round(float(e.start_time - start), 9),
         "duration": round(float(e.duration), 9), "ops": list(map(int, e.affected_ops)),
         "magnitude": round(float(e.magnitude), 9)}
        for e in sim.risk_events if e.start_time >= start - 1e-9
    ]
    demand = [
        (round(float(t - start), 9), round(float(q), 9))
        for t, q in sim.daily_demand if t >= start - 1e-9
    ]
    return {"risk_sha256": digest(risks), "demand_sha256": digest(demand)}


def state_hash(sim: MFSCSimulation) -> str:
    payload = {
        "time": round(float(sim.env.now), 9),
        "observation": sim.get_op8_convoy_observation(),
        "inventory": sim._inventory_detail(),
        "convoy": sim.op8_convoy_metrics(),
        "queue": [
            {"j": o.j, "remaining": o.remaining_qty, "in_flight": o.in_flight_qty,
             "contingent": o.contingent, "lost": o.lost}
            for o in sim.pending_backorders
        ],
        "op8_down": sim.op_down_count[8],
    }
    return digest(payload)


def run_static_policy(tape: dict[str, Any], policy: ConvoyThresholdPolicy) -> dict[str, Any]:
    sim, start = make_sim(tape)
    baseline_resource = resource_snapshot(sim)
    end = start + int(tape["horizon_weeks"]) * HOURS_PER_WEEK
    backlog_auc = 0.0
    live_epochs = 0
    decision_epochs = 0
    while sim.env.now < end - 1e-9:
        observation = sim.get_op8_convoy_observation()
        if sim.op8_convoy_dispatch_feasible():
            live_epochs += 1
        action = policy.action(observation)
        sim.apply_op8_convoy_action(action, source=policy.policy_id)
        decision_epochs += 1
        target = min(end, float(sim.env.now) + HOURS_PER_DAY)
        advance_including(sim, target)
        backlog_auc += float(sim.pending_backorder_qty) * HOURS_PER_DAY
    metrics = compute_episode_metrics(sim, treatment_start=start)
    ledger = sim.flow_ledger()
    metrics.update(resource_delta(sim, baseline_resource))
    metrics.update(exogenous_hashes(sim, start))
    metrics.update(
        {
            "backlog_auc": float(backlog_auc),
            "live_epochs": int(live_epochs),
            "decision_epochs": int(decision_epochs),
            "live_fraction": float(live_epochs / max(decision_epochs, 1)),
            "mass_residual": max(
                abs(float(ledger["raw_residual"])),
                abs(float(ledger["ration_residual"])),
            ),
        }
    )
    return metrics


def run_actions_to(
    sim: MFSCSimulation,
    start: float,
    end: float,
    policy: ConvoyThresholdPolicy,
    *,
    forced_actions: Iterable[str] = (),
) -> None:
    forced = iter(forced_actions)
    exhausted = False
    while sim.env.now < end - 1e-9:
        if not exhausted:
            try:
                action = next(forced)
            except StopIteration:
                exhausted = True
                action = policy.action(sim.get_op8_convoy_observation())
        else:
            action = policy.action(sim.get_op8_convoy_observation())
        sim.apply_op8_convoy_action(action, source="branch_or_continuation")
        advance_including(sim, min(end, float(sim.env.now) + HOURS_PER_DAY))


def episode_metrics_from(sim: MFSCSimulation, treatment_start: float) -> dict[str, Any]:
    metrics = compute_episode_metrics(sim, treatment_start=treatment_start)
    ledger = sim.flow_ledger()
    metrics["mass_residual"] = max(
        abs(float(ledger["raw_residual"])), abs(float(ledger["ration_residual"]))
    )
    return metrics
