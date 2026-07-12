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
OBSERVATION_KEYS = (
    "equipment_condition_score", "route_threat_score", "mission_tempo_score",
    "equipment_condition", "recent_r11_count", "recent_transport_attack_count",
    "recent_r24_count", "recent_damage_hours", "sb_inventory", "reserve_inventory",
    "backlog_qty", "backlog_count", "active_m", "active_t", "active_r",
    "pending_m", "pending_t", "pending_r", "week_phase",
)
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


def actions_for_budget(tokens: int) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (m, t, r) for m in range(3) for t in range(3) for r in range(3)
        if m + t + r == int(tokens)
    )


def default_profile() -> dict[str, Any]:
    return {
        "cell_id": "program_f_primary", "efficacy_level": "base",
        "condition_reduction_per_token": 0.20,
        "r11_factors": [1.0, 0.75, 0.50],
        "transport_factors": [1.0, 0.65, 0.40],
        "reserve_issue": [0.0, 2500.0, 5000.0],
        "signal_accuracy": 0.75, "dwell_weeks": [4, 8],
        "budget_tokens": 2,
        "risk_amplitude": "dominant_increased_background_current",
        "minimum_commitment_weeks": 1,
    }


def profile_for_cell(cell: dict[str, Any]) -> dict[str, Any]:
    idx = {"low": 0, "base": 1, "high": 2}[cell["efficacy_level"]]
    return {
        "cell_id": cell["cell_id"], "efficacy_level": cell["efficacy_level"],
        "condition_reduction_per_token": [0.15, 0.20, 0.25][idx],
        "r11_factors": [[1.0, 0.85, 0.65], [1.0, 0.75, 0.50], [1.0, 0.65, 0.40]][idx],
        "transport_factors": [[1.0, 0.75, 0.55], [1.0, 0.65, 0.40], [1.0, 0.55, 0.30]][idx],
        "reserve_issue": [[0.0, 2000.0, 4000.0], [0.0, 2500.0, 5000.0], [0.0, 3000.0, 6000.0]][idx],
        "signal_accuracy": float(cell["signal_accuracy"]),
        "dwell_weeks": list(map(int, cell["context_dwell_weeks"])),
        "budget_tokens": int(cell["budget_tokens"]),
        "risk_amplitude": str(cell["risk_amplitude"]),
        "minimum_commitment_weeks": int(cell["minimum_commitment_weeks"]),
    }


def _context_schedule(
    rng: np.random.Generator, first: str, weeks: int, dwell_weeks: tuple[int, int]
) -> list[str]:
    result: list[str] = []
    current = first
    while len(result) < weeks:
        dwell = int(rng.integers(int(dwell_weeks[0]), int(dwell_weeks[1]) + 1))
        result.extend([current] * min(dwell, weeks - len(result)))
        current = str(rng.choice([value for value in CONTEXTS if value != current]))
    return result


def materialize_tape(
    seed: int, first_context: str, split: str, weeks: int = 32,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if first_context not in CONTEXTS:
        raise ValueError(first_context)
    profile = dict(default_profile() if profile is None else profile)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xF20260712]))
    contexts = _context_schedule(rng, first_context, weeks, tuple(profile["dwell_weeks"]))
    signals = []
    for week in range(weeks):
        target = contexts[min(week + 1, weeks - 1)]
        if float(rng.random()) < float(profile["signal_accuracy"]):
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
            use_dominant = (
                profile["risk_amplitude"] == "dominant_increased_background_current"
                and context == dominant[risk_id]
            )
            rate = rates[risk_id][0 if use_dominant else 1]
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
        "profile": profile,
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
        self.profile = dict(tape.get("profile") or default_profile())
        budget = int(self.profile["budget_tokens"])
        initial = {1: (1, 0, 0), 2: INITIAL_ACTION, 3: (1, 1, 1)}[budget]
        self.active_action = initial
        self.pending_action = initial
        self.condition = 0.25
        self.current_week = 0
        self.action_events: list[dict[str, Any]] = []
        self.damage_events: list[dict[str, Any]] = []
        self.consumed_base_events: list[dict[str, Any]] = []
        self.maintenance_downtime_hours = 0.0
        self.token_hours = {"M": 0.0, "T": 0.0, "R": 0.0}
        self.last_switch_week = -int(self.profile["minimum_commitment_weeks"])
        self.rejected_switches = 0

    def observation(self) -> dict[str, float]:
        week = min(self.current_week, int(self.tape["weeks"]) - 1)
        scores = self.tape["signals"][week]["scores"]
        now = float(self.sim.env.now)
        recent = [row for row in self.damage_events if row["start_time"] >= now - 4 * HOURS_PER_WEEK]
        observation = {
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
        if tuple(observation) != OBSERVATION_KEYS:
            raise AssertionError("Program F observation schema drift")
        return observation

    def request(self, action: Iterable[int]) -> None:
        value = tuple(int(item) for item in action)
        if value not in actions_for_budget(int(self.profile["budget_tokens"])):
            raise ValueError(f"Invalid Program F allocation: {value}")
        if (
            value != self.active_action
            and self.current_week - self.last_switch_week
            < int(self.profile["minimum_commitment_weeks"])
        ):
            self.rejected_switches += 1
            return
        self.pending_action = value

    def activate_week(self, week: int) -> None:
        self.current_week = int(week)
        previous = self.active_action
        self.active_action = self.pending_action
        if self.active_action != previous:
            self.last_switch_week = int(week)
        if sum(self.active_action) != int(self.profile["budget_tokens"]):
            raise AssertionError("Program F budget invariant violated")
        context = self.tape["context_schedule"][week]
        wear = 0.12 if context == "equipment_pressure" else 0.05
        self.condition = float(np.clip(
            self.condition + wear
            - float(self.profile["condition_reduction_per_token"]) * self.active_action[0],
            0.0, 1.0
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
        start = float(self.sim.env.now)
        risk_id = str(event["risk_id"])
        base = float(event["base_duration_hours"])
        self.consumed_base_events.append({
            "event_id": str(event["event_id"]), "risk_id": risk_id,
            "onset_hours": float(start - self.start),
            "base_duration_hours": base,
            "affected_ops": list(map(int, event["affected_ops"])),
            "magnitude": float(event["magnitude"]),
            "context_at_onset": str(event["context_at_onset"]),
        })
        realized = base
        if risk_id == "R11":
            factor = self.profile["r11_factors"][self.active_action[0]]
            realized = max(1.0, base * (1.0 + self.condition) * factor)
        elif risk_id in {"R22", "R23"}:
            factor = self.profile["transport_factors"][self.active_action[1]]
            realized = max(1.0, base * factor)
        if risk_id == "R24":
            surge = float(event["magnitude"])
            quota = self.profile["reserve_issue"][self.active_action[2]]
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


def runtime_exogenous_artifacts(
    sim: MFSCSimulation, controller: ProgramFController, start: float
) -> dict[str, Any]:
    demand_rows = [
        {
            "j": int(order.j),
            "time_hours": round(float(order.OPTj) - float(start), 9),
            "quantity": round(float(order.quantity), 9),
            "contingent": bool(order.contingent),
            "destination": order.cssu_destination,
        }
        for order in sim.orders if float(order.OPTj) >= float(start) - 1e-9
    ]
    threat_rows = sorted(
        controller.consumed_base_events,
        key=lambda row: (row["onset_hours"], row["event_id"]),
    )
    return {
        "consumed_base_threat_sha256": digest(threat_rows),
        "realized_demand_sha256": digest(demand_rows),
        "consumed_base_threat_rows": threat_rows,
        "realized_demand_rows": demand_rows,
    }


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
    metrics.update(runtime_exogenous_artifacts(sim, controller, start))
    return metrics


def branch_from_week(
    tape: dict[str, Any], *, prefix_action: tuple[int, int, int],
    state_week: int, branch_action: tuple[int, int, int], horizon_weeks: int,
) -> dict[str, Any]:
    """Exact replay branch: common prefix, then hold one allocation for H weeks."""
    sim, controller, start = make_sim(tape)
    state_week = int(state_week)
    for week in range(state_week):
        controller.activate_week(week)
        controller.request(prefix_action)
        advance_including(sim, start + (week + 1) * HOURS_PER_WEEK)
    controller.activate_week(state_week)
    observation = controller.observation()
    treatment_start = float(sim.env.now)
    for offset in range(int(horizon_weeks)):
        week = state_week + offset
        if offset > 0:
            controller.activate_week(week)
        controller.request(branch_action)
        advance_including(sim, treatment_start + (offset + 1) * HOURS_PER_WEEK)
    metrics = compute_episode_metrics(sim, treatment_start=treatment_start)
    ledger = sim.flow_ledger()
    exogenous = runtime_exogenous_artifacts(sim, controller, start)
    return {
        "observation": observation,
        "ret": float(metrics["ret_excel"]),
        "service": float(metrics["service_loss_auc_ration_hours"]),
        "lost": float(metrics["n_lost"]),
        "mass_residual": max(abs(float(ledger["raw_residual"])), abs(float(ledger["ration_residual"]))),
        "threat_sha256": tape["threat_sha256"],
        "state_week": state_week, "horizon_weeks": int(horizon_weeks),
        "prefix_action": list(prefix_action), "branch_action": list(branch_action),
        **exogenous,
    }
