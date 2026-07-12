"""Program F: fixed-budget risk-conditioned mitigation portfolio.

This lane is deliberately separate from historical environments.  Threats and
signals are materialized once; policy actions only transform realized damage or
consume finite reserve stock.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .config import HOURS_PER_WEEK, SIMULATION_HORIZON
from .episode_metrics import compute_episode_metrics
from .supply_chain import MFSCSimulation, RiskEvent


CONTRACT_ID = "mfsc_risk_mitigation_portfolio_v1"
EVENT_MODEL_ID = "program_f_event_model_v1"
CONTEXTS = ("equipment_pressure", "interdiction_campaign", "mission_surge")
ACTIONS = ((2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1))
INITIAL_ACTION = (1, 1, 0)
PROXY = Path(__file__).resolve().parent / "data" / "garrido_proxy_v1_freeze_2026-07-10.json"


def digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def proxy_kwargs() -> dict[str, Any]:
    payload = json.loads(PROXY.read_text(encoding="utf-8"))
    kwargs = dict(payload["sim_kwargs"])
    kwargs.pop("risk_level", None)
    kwargs.pop("seed_stream_mode", None)
    return kwargs


def advance_including(sim: MFSCSimulation, target: float) -> None:
    target = float(target)
    while sim.env.peek() <= target:
        sim.env.step()
    if float(sim.env.now) < target:
        sim.env.run(until=target)


def _context_schedule(rng: np.random.Generator, first: str, weeks: int) -> list[str]:
    result: list[str] = []
    current = first
    while len(result) < weeks:
        dwell = int(rng.integers(4, 9))
        result.extend([current] * min(dwell, weeks - len(result)))
        current = str(rng.choice([value for value in CONTEXTS if value != current]))
    return result


def materialize_tape(seed: int, first_context: str, split: str, weeks: int = 32) -> dict[str, Any]:
    if first_context not in CONTEXTS:
        raise ValueError(first_context)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xF20260712]))
    contexts = _context_schedule(rng, first_context, weeks)
    signals = []
    for week in range(weeks):
        target = contexts[min(week + 1, weeks - 1)]
        if float(rng.random()) < 0.75:
            predicted = target
        else:
            predicted = str(rng.choice([value for value in CONTEXTS if value != target]))
        scores = {context: (0.8 if context == predicted else 0.1) for context in CONTEXTS}
        signals.append({"week": week, "target_week": min(week + 1, weeks - 1), "scores": scores})

    dominant = {
        "R11": "equipment_pressure", "R22": "interdiction_campaign",
        "R23": "interdiction_campaign", "R24": "mission_surge",
    }
    rates = {
        "R11": (4.0, 1.0), "R22": (0.125, 1 / 24),
        "R23": (0.125, 1 / 48), "R24": (0.5, 0.25),
    }
    events: list[dict[str, Any]] = []
    event_id = 0
    for week, context in enumerate(contexts):
        for risk_id in ("R11", "R22", "R23", "R24"):
            rate = rates[risk_id][0 if context == dominant[risk_id] else 1]
            for _ in range(int(rng.poisson(rate))):
                event_id += 1
                onset = week * HOURS_PER_WEEK + float(rng.uniform(1.0, HOURS_PER_WEEK - 1.0))
                row: dict[str, Any] = {
                    "event_id": f"F{event_id:05d}", "risk_id": risk_id,
                    "onset_hours": onset, "context_at_onset": context,
                }
                if risk_id == "R11":
                    row.update({
                        "base_duration_hours": float(rng.exponential(5.0 if context == dominant[risk_id] else 2.0)),
                        "affected_ops": [int(rng.choice((5, 6)))], "magnitude": 1.0,
                    })
                elif risk_id == "R22":
                    row.update({
                        "base_duration_hours": float(rng.exponential(24.0)),
                        "affected_ops": [int(rng.choice((4, 8, 10, 12)))], "magnitude": 1.0,
                    })
                elif risk_id == "R23":
                    row.update({"base_duration_hours": float(rng.exponential(120.0)), "affected_ops": [11], "magnitude": 1.0})
                else:
                    row.update({"base_duration_hours": 0.0, "affected_ops": [13], "magnitude": int(rng.integers(2400, 2601))})
                events.append(row)
    events.sort(key=lambda row: (row["onset_hours"], row["event_id"]))
    tape = {
        "contract_id": CONTRACT_ID, "event_model_id": EVENT_MODEL_ID,
        "tape_id": f"program-f-{split}-{first_context}-{seed}", "split": split,
        "seed": int(seed), "weeks": int(weeks), "first_context": first_context,
        "context_schedule": contexts, "signals": signals, "base_events": events,
    }
    tape["threat_sha256"] = digest({
        "context_schedule": contexts, "signals": signals, "base_events": events,
    })
    return tape


@dataclass(frozen=True)
class ConstantPortfolio:
    action: tuple[int, int, int]

    @property
    def policy_id(self) -> str:
        return "constant_" + "".join(map(str, self.action))

    def __call__(self, observation: dict[str, float]) -> tuple[int, int, int]:
        return self.action


class ProgramFController:
    def __init__(self, sim: MFSCSimulation, tape: dict[str, Any], start: float):
        self.sim, self.tape, self.start = sim, tape, float(start)
        self.active_action = INITIAL_ACTION
        self.pending_action = INITIAL_ACTION
        self.condition = 0.25
        self.current_week = 0
        self.action_events: list[dict[str, Any]] = []
        self.damage_events: list[dict[str, Any]] = []
        self.maintenance_downtime_hours = 0.0
        self.token_hours = {"M": 0.0, "T": 0.0, "R": 0.0}

    def observation(self) -> dict[str, float]:
        week = min(self.current_week, int(self.tape["weeks"]) - 1)
        scores = self.tape["signals"][week]["scores"]
        now = float(self.sim.env.now)
        recent = [row for row in self.damage_events if row["start_time"] >= now - 4 * HOURS_PER_WEEK]
        return {
            "equipment_condition_score": float(scores["equipment_pressure"]),
            "route_threat_score": float(scores["interdiction_campaign"]),
            "mission_tempo_score": float(scores["mission_surge"]),
            "equipment_condition": float(self.condition),
            "recent_r11_count": float(sum(row["risk_id"] == "R11" for row in recent)),
            "recent_transport_attack_count": float(sum(row["risk_id"] in {"R22", "R23"} for row in recent)),
            "recent_r24_count": float(sum(row["risk_id"] == "R24" for row in recent)),
            "recent_damage_hours": float(sum(row["realized_duration_hours"] for row in recent)),
            "sb_inventory": float(self.sim.rations_sb.level),
            "reserve_inventory": float(self.sim.emergency_theatre_reserve.level),
            "backlog_qty": float(self.sim.pending_backorder_qty),
            "backlog_count": float(len(self.sim.pending_backorders)),
            "active_m": float(self.active_action[0]), "active_t": float(self.active_action[1]),
            "active_r": float(self.active_action[2]), "pending_m": float(self.pending_action[0]),
            "pending_t": float(self.pending_action[1]), "pending_r": float(self.pending_action[2]),
            "week_phase": float(week / max(int(self.tape["weeks"]) - 1, 1)),
        }

    def request(self, action: Iterable[int]) -> None:
        value = tuple(int(item) for item in action)
        if value not in ACTIONS:
            raise ValueError(f"Invalid Program F allocation: {value}")
        self.pending_action = value

    def activate_week(self, week: int) -> None:
        self.current_week = int(week)
        self.active_action = self.pending_action
        if sum(self.active_action) != 2:
            raise AssertionError("Program F budget invariant violated")
        context = self.tape["context_schedule"][week]
        wear = 0.12 if context == "equipment_pressure" else 0.05
        self.condition = float(np.clip(
            self.condition + wear - 0.20 * self.active_action[0], 0.0, 1.0
        ))
        maintenance_hours = (0.0, 12.0, 24.0)[self.active_action[0]]
        if maintenance_hours > 0.0:
            self.sim.env.process(self._planned_maintenance(maintenance_hours))
        for key, tokens in zip(("M", "T", "R"), self.active_action):
            self.token_hours[key] += float(tokens) * HOURS_PER_WEEK
        self.sim.request_emergency_reserve_target(10_000.0)
        self.action_events.append({
            "week": week, "time": float(self.sim.env.now), "action": list(self.active_action),
            "condition": self.condition, "context_hidden_from_policy": context,
        })

    def _planned_maintenance(self, hours: float):
        for op_id in (5, 6, 7):
            self.sim._take_down(op_id)
        self.maintenance_downtime_hours += float(hours)
        yield self.sim.env.timeout(float(hours))
        for op_id in (5, 6, 7):
            self.sim._bring_up(op_id)

    def schedule_threats(self) -> None:
        for event in self.tape["base_events"]:
            self.sim.env.process(self._threat_process(dict(event)))

    def _threat_process(self, event: dict[str, Any]):
        target = self.start + float(event["onset_hours"])
        if target > self.sim.env.now:
            yield self.sim.env.timeout(target - self.sim.env.now)
        risk_id = str(event["risk_id"])
        base = float(event["base_duration_hours"])
        realized = base
        if risk_id == "R11":
            factor = (1.0, 0.75, 0.50)[self.active_action[0]]
            realized = max(1.0, base * (1.0 + self.condition) * factor)
        elif risk_id in {"R22", "R23"}:
            factor = (1.0, 0.65, 0.40)[self.active_action[1]]
            realized = max(1.0, base * factor)
        start = float(self.sim.env.now)
        if risk_id == "R24":
            surge = float(event["magnitude"])
            quota = (0.0, 2500.0, 5000.0)[self.active_action[2]]
            self.sim.set_program_f_r24_issue_quota(min(quota, surge))
            self.sim._contingent_demand_pending = min(
                self.sim._contingent_demand_pending + surge, 5 * 2600
            )
            replayed = RiskEvent("R24", start, start, 0.0, [13], "Program F exogenous surge", surge, "rations")
            self.sim.risk_events.append(replayed)
            self.sim._add_ret_quantity_risk(replayed)
        else:
            for op_id in event["affected_ops"]:
                self.sim._take_down(int(op_id))
            yield self.sim.env.timeout(realized)
            for op_id in event["affected_ops"]:
                self.sim._bring_up(int(op_id))
            self.sim.risk_events.append(RiskEvent(
                risk_id, start, float(self.sim.env.now), realized,
                list(map(int, event["affected_ops"])), "Program F realized damage",
            ))
        self.damage_events.append({
            "event_id": event["event_id"], "risk_id": risk_id, "start_time": start,
            "base_duration_hours": base, "realized_duration_hours": float(realized),
            "active_action": list(self.active_action),
        })


def make_sim(tape: dict[str, Any]) -> tuple[MFSCSimulation, ProgramFController, float]:
    horizon = max(float(SIMULATION_HORIZON), 8_000 + int(tape["weeks"]) * HOURS_PER_WEEK)
    sim = MFSCSimulation(
        seed=int(tape["seed"]), horizon=horizon, risks_enabled=False,
        strict_exogenous_crn=True, **proxy_kwargs(),
    )
    sim._start_processes()
    while not sim.warmup_complete:
        advance_including(sim, min(float(sim.env.now) + 1.0, sim.horizon))
    start = float(sim.env.now)
    sim.configure_emergency_theatre_reserve(
        capacity=10_000.0, initial_stock=10_000.0, target=10_000.0,
        replenishment_lead_time=168.0, issue_delay=24.0,
        route_ops=(10, 11, 12), transport_mode="fixed_lead",
    )
    sim.enable_program_f_reserve()
    controller = ProgramFController(sim, tape, start)
    controller.schedule_threats()
    return sim, controller, start


def run_policy(tape: dict[str, Any], policy: Callable[[dict[str, float]], Iterable[int]]) -> dict[str, Any]:
    sim, controller, start = make_sim(tape)
    end = start + int(tape["weeks"]) * HOURS_PER_WEEK
    for week in range(int(tape["weeks"])):
        controller.activate_week(week)
        controller.request(policy(controller.observation()))
        advance_including(sim, min(end, start + (week + 1) * HOURS_PER_WEEK))
    metrics = compute_episode_metrics(sim, treatment_start=start)
    ledger = sim.flow_ledger()
    metrics.update({
        "threat_sha256": tape["threat_sha256"],
        "mass_residual": max(abs(float(ledger["raw_residual"])), abs(float(ledger["ration_residual"]))),
        "maintenance_downtime_hours": controller.maintenance_downtime_hours,
        "reserve_units_issued": float(sim.program_f_reserve_fragments_issued),
        "reserve_units_replenished": float(sim.emergency_reserve_units_replenished),
        "token_hours_m": controller.token_hours["M"], "token_hours_t": controller.token_hours["T"],
        "token_hours_r": controller.token_hours["R"], "total_token_hours": sum(controller.token_hours.values()),
        "action_events": controller.action_events, "damage_events": controller.damage_events,
    })
    return metrics
