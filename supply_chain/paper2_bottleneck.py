"""Paper 2 bottleneck-migration controller on the full event-driven MFSC DES."""
from __future__ import annotations

from typing import Callable, Iterable, Any

from .config import HOURS_PER_WEEK, SIMULATION_HORIZON
from .episode_metrics import compute_episode_metrics
from .program_f import (
    CONTEXTS, ProgramFController, advance_including, materialize_tape as materialize_f_tape,
    proxy_kwargs, runtime_exogenous_artifacts,
)
from .supply_chain import MFSCSimulation

ACTIONS = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
ACTION_NAMES = {(1, 0, 0): "M", (0, 1, 0): "T", (0, 0, 1): "R"}


def profile() -> dict[str, Any]:
    return {
        "cell_id": "paper2_bottleneck_high_authority_v1",
        "efficacy_level": "declared_high_authority",
        "condition_reduction_per_token": 0.0,
        "r11_factors": [1.0, 0.20],
        "transport_factors": [1.0, 0.20],
        "reserve_issue": [0.0, 5000.0],
        "signal_accuracy": 0.85,
        "dwell_weeks": [4, 8],
        "budget_tokens": 1,
        "risk_amplitude": "dominant_increased_background_current",
        "minimum_commitment_weeks": 1,
    }


def materialize_tape(seed: int, first_context: str, split: str, weeks: int = 24):
    return materialize_f_tape(seed, first_context, split, weeks=weeks, profile=profile())


class BottleneckController(ProgramFController):
    """One response team; unlike Program F maintenance, M is recovery capacity, not downtime."""

    def activate_week(self, week: int) -> None:
        self.current_week = int(week)
        previous = self.active_action
        self.active_action = self.pending_action
        if self.active_action != previous:
            self.last_switch_week = int(week)
        if self.active_action not in ACTIONS:
            raise AssertionError("Paper 2 one-team invariant violated")
        for key, tokens in zip(("M", "T", "R"), self.active_action):
            self.token_hours[key] += float(tokens) * HOURS_PER_WEEK
        self.sim.request_emergency_reserve_target(10_000.0)
        self.action_events.append({
            "week": int(week), "time": float(self.sim.env.now),
            "action": list(self.active_action),
            "context_hidden_from_policy": self.tape["context_schedule"][week],
        })


def make_sim(tape):
    horizon = max(float(SIMULATION_HORIZON), 8_000 + int(tape["weeks"]) * HOURS_PER_WEEK)
    kwargs = proxy_kwargs(); kwargs["assembly_flow_mode"] = "serial_wip"
    sim = MFSCSimulation(seed=int(tape["seed"]), horizon=horizon, risks_enabled=False,
                         strict_exogenous_crn=True, **kwargs)
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
    controller = BottleneckController(sim, tape, start)
    controller.schedule_threats()
    return sim, controller, start


def signal_policy(observation: dict[str, float]):
    scores = (observation["equipment_condition_score"],
              observation["route_threat_score"], observation["mission_tempo_score"])
    return ACTIONS[max(range(3), key=lambda i: (scores[i], -i))]


def run_policy(tape, policy: Callable[[dict[str, float]], Iterable[int]]):
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
        "mass_residual": max(abs(float(ledger["raw_residual"])),
                             abs(float(ledger["ration_residual"]))),
        "token_hours_m": controller.token_hours["M"],
        "token_hours_t": controller.token_hours["T"],
        "token_hours_r": controller.token_hours["R"],
        "total_token_hours": sum(controller.token_hours.values()),
        "action_events": controller.action_events,
        "damage_events": controller.damage_events,
        "reserve_units_issued": float(sim.program_f_reserve_fragments_issued),
    })
    metrics.update(runtime_exogenous_artifacts(sim, controller, start))
    return metrics
