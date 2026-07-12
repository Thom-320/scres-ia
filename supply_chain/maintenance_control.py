"""Paper 2 finite-maintenance control layered on the full MFSC DES.

The tape contains policy-invariant wear, sensor, R11-candidate and R14 innovation
streams.  Actions transform vulnerability and consume one shared 24-hour crew
slot; they never rewrite the exogenous tape.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Callable
from itertools import product

import numpy as np
import simpy

from .config import HOURS_PER_DAY, HOURS_PER_WEEK, RISKS_CURRENT
from .episode_metrics import compute_episode_metrics
from .program_f import advance_including, proxy_kwargs
from .supply_chain import MFSCSimulation, RATIONS_PER_HOUR, RiskEvent

ACTIONS = ("PM5", "PM6", "PM7")
ACTION_TO_OP = {name: int(name[-1]) for name in ACTIONS}
OBSERVATION_KEYS = (
    "condition_signal_op5", "condition_signal_op6", "condition_signal_op7",
    "utilization_op5", "utilization_op6", "utilization_op7",
    "recent_failures_op5", "recent_failures_op6", "recent_failures_op7",
    "recent_downtime_op5", "recent_downtime_op6", "recent_downtime_op7",
    "wip_op5_op6", "wip_op6_op7", "backlog_qty", "backlog_count",
    "crew_busy", "crew_eta_hours", "previous_action", "weeks_since_pm_op5",
    "weeks_since_pm_op6", "weeks_since_pm_op7", "week_phase",
)
FORBIDDEN_OBSERVATIONS = (
    "true_condition", "next_failure", "future_wear", "future_repair_duration",
    "future_r14", "oracle_action", "oracle_value", "future_outcome",
)


def digest(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def default_cell() -> dict[str, Any]:
    return {
        "sensor_balanced_accuracy": 0.75,
        "pm_restore_fraction": 0.50,
        "wip_capacity_days": 2,
        "wear_heterogeneity": "high",
        "repair_profile": "current",
    }


def materialize_tape(seed: int, *, weeks: int = 24) -> dict[str, Any]:
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xC0B202]))
    hours = int(weeks * HOURS_PER_WEEK)
    initial = rng.uniform(0.15, 0.45, size=3)
    hourly_wear = rng.random((hours, 3)).tolist()
    hourly_sensor = rng.normal(0.0, 1.0, size=(hours, 3)).tolist()
    candidates: list[dict[str, Any]] = []
    t = float(rng.integers(1, 169))
    sequence = 0
    while t < hours:
        target = int(rng.choice((5, 6)))
        candidates.append({
            "event_id": f"R11-{sequence}", "onset_hours": float(t),
            "target": target, "realization_u": float(rng.random()),
            "repair_u": float(max(1e-12, rng.random())),
        })
        sequence += 1
        t += float(rng.integers(1, 169))
    r14_seeds = [int(x) for x in rng.integers(0, 2**31 - 1, size=weeks * 7)]
    payload = {
        "seed": int(seed), "weeks": int(weeks), "initial_condition": initial.tolist(),
        "hourly_wear": hourly_wear, "hourly_sensor_noise": hourly_sensor,
        "r11_candidates": candidates, "r14_daily_seeds": r14_seeds,
    }
    payload["base_exogenous_sha256"] = digest(payload)
    return payload


def truncate_tape(tape: dict[str, Any], weeks: int) -> dict[str, Any]:
    """Return an exact-prefix tape without resampling any innovation."""
    hours = int(weeks * HOURS_PER_WEEK)
    result = {
        "seed": int(tape["seed"]), "weeks": int(weeks),
        "initial_condition": list(tape["initial_condition"]),
        "hourly_wear": list(tape["hourly_wear"][:hours]),
        "hourly_sensor_noise": list(tape["hourly_sensor_noise"][:hours]),
        "r11_candidates": [
            dict(row) for row in tape["r11_candidates"]
            if float(row["onset_hours"]) < hours
        ],
        "r14_daily_seeds": list(tape["r14_daily_seeds"][: weeks * 7]),
    }
    result["base_exogenous_sha256"] = digest(result)
    return result


@dataclass
class MaintenanceRecord:
    kind: str
    op_id: int
    requested_at: float
    started_at: float
    ended_at: float
    hours: float


class MaintenanceController:
    def __init__(
        self, sim: MFSCSimulation, tape: dict[str, Any], start: float,
        cell: dict[str, Any] | None = None,
    ) -> None:
        self.sim, self.tape, self.start = sim, tape, float(start)
        self.cell = dict(default_cell() if cell is None else cell)
        self.crew = simpy.Resource(sim.env, capacity=1)
        self.condition = np.asarray(tape["initial_condition"], dtype=float)
        self.condition_signal = self.condition.copy()
        self.utilization_hours = np.zeros(3)
        self.recent_failures = np.zeros(3)
        self.recent_downtime = np.zeros(3)
        self.weeks_since_pm = np.zeros(3)
        self.previous_action = "PM5"
        self.crew_busy_until = float(start)
        self.records: list[MaintenanceRecord] = []
        self.action_events: list[dict[str, Any]] = []
        self.consumed_candidates: list[dict[str, Any]] = []
        self.consumed_wear: list[dict[str, Any]] = []
        self.scheduled_pm_hours = 0.0
        self.executed_pm_hours = 0.0
        self.corrective_hours = 0.0
        self.blocked_hours = {5: 0.0, 6: 0.0}
        self.starved_hours = {6: 0.0, 7: 0.0}
        self.sim.env.process(self._condition_process())
        self.sim.env.process(self._quality_process())
        for row in tape["r11_candidates"]:
            self.sim.env.process(self._candidate_process(dict(row)))

    def _idx(self, op_id: int) -> int:
        return int(op_id) - 5

    def observation(self) -> dict[str, float]:
        now = float(self.sim.env.now)
        result = {
            **{f"condition_signal_op{op}": float(self.condition_signal[self._idx(op)]) for op in (5, 6, 7)},
            **{f"utilization_op{op}": float(self.utilization_hours[self._idx(op)] / HOURS_PER_WEEK) for op in (5, 6, 7)},
            **{f"recent_failures_op{op}": float(self.recent_failures[self._idx(op)]) for op in (5, 6, 7)},
            **{f"recent_downtime_op{op}": float(self.recent_downtime[self._idx(op)] / HOURS_PER_WEEK) for op in (5, 6, 7)},
            "wip_op5_op6": float(self.sim.wip_op5_op6.level),
            "wip_op6_op7": float(self.sim.wip_op6_op7.level),
            "backlog_qty": float(self.sim.pending_backorder_qty),
            "backlog_count": float(len(self.sim.pending_backorders)),
            "crew_busy": float(now < self.crew_busy_until),
            "crew_eta_hours": float(max(0.0, self.crew_busy_until - now)),
            "previous_action": float(ACTIONS.index(self.previous_action)),
            **{f"weeks_since_pm_op{op}": float(self.weeks_since_pm[self._idx(op)]) for op in (5, 6, 7)},
            "week_phase": float(((now - self.start) % HOURS_PER_WEEK) / HOURS_PER_WEEK),
        }
        if tuple(result) != OBSERVATION_KEYS:
            raise AssertionError("Maintenance observation schema drifted")
        return result

    def request(self, action: str, week: int) -> None:
        if action not in ACTIONS:
            raise ValueError(action)
        op_id = ACTION_TO_OP[action]
        self.previous_action = action
        self.weeks_since_pm += 1.0
        self.recent_failures[:] = 0.0
        self.recent_downtime[:] = 0.0
        self.utilization_hours[:] = 0.0
        self.scheduled_pm_hours += 24.0
        self.action_events.append({
            "week": int(week), "requested_at": float(self.sim.env.now),
            "action": action, "op_id": op_id,
        })
        self.sim.env.process(self._maintenance_process(op_id))

    def _maintenance_process(self, op_id: int):
        requested = float(self.sim.env.now)
        with self.crew.request(priority=1) if isinstance(self.crew, simpy.PriorityResource) else self.crew.request() as req:
            yield req
            start = float(self.sim.env.now)
            self.crew_busy_until = start + 24.0
            self.sim._take_down(op_id)
            yield self.sim.env.timeout(24.0)
            self.sim._bring_up(op_id)
            idx = self._idx(op_id)
            restore = float(self.cell["pm_restore_fraction"])
            self.condition[idx] *= 1.0 - restore
            self.weeks_since_pm[idx] = 0.0
            self.executed_pm_hours += 24.0
            self.records.append(MaintenanceRecord(
                "preventive", op_id, requested, start, float(self.sim.env.now), 24.0
            ))

    def _condition_process(self):
        hetero = np.array([0.8, 1.0, 1.2]) if self.cell["wear_heterogeneity"] == "high" else np.ones(3)
        q = float(self.cell["sensor_balanced_accuracy"])
        noise_scale = max(0.02, (1.0 - q) * 0.55)
        for h, (wear_row, noise_row) in enumerate(zip(self.tape["hourly_wear"], self.tape["hourly_sensor_noise"])):
            yield self.sim.env.timeout(1.0)
            active = np.array([not self.sim._is_down(op) for op in (5, 6, 7)], dtype=float)
            start56, start67 = float(self.sim.wip_op5_op6.level), float(self.sim.wip_op6_op7.level)
            capable = np.array([
                active[0] and self.sim.raw_material_al.level > 0 and start56 < self.sim.wip_op5_op6.capacity,
                active[1] and start56 > 0 and start67 < self.sim.wip_op6_op7.capacity,
                active[2] and start67 > 0,
            ], dtype=float)
            self.utilization_hours += capable
            base = 0.00045
            innovations = 0.5 + np.asarray(wear_row, dtype=float)
            delta = base * hetero * innovations * (0.25 + 0.75 * capable)
            self.condition = np.clip(self.condition + delta, 0.0, 1.0)
            self.condition_signal = np.clip(
                self.condition + noise_scale * np.asarray(noise_row), 0.0, 1.0
            )
            if start56 >= self.sim.wip_op5_op6.capacity - 1e-9:
                self.blocked_hours[5] += 1.0
            if start67 >= self.sim.wip_op6_op7.capacity - 1e-9:
                self.blocked_hours[6] += 1.0
            if start56 <= 1e-9:
                self.starved_hours[6] += 1.0
            if start67 <= 1e-9:
                self.starved_hours[7] += 1.0
            self.consumed_wear.append({"hour": h, "wear": wear_row})

    def _candidate_process(self, row: dict[str, Any]):
        target = self.start + float(row["onset_hours"])
        if target > self.sim.env.now:
            yield self.sim.env.timeout(target - self.sim.env.now)
        idx = self._idx(int(row["target"]))
        hazard = 0.04 + 0.86 * float(self.condition[idx]) ** 2
        realized = float(row["realization_u"]) < hazard
        consumed = dict(row)
        consumed["realized"] = bool(realized)
        self.consumed_candidates.append(consumed)
        if not realized:
            return
        requested = float(self.sim.env.now)
        with self.crew.request() as req:
            yield req
            start = float(self.sim.env.now)
            mean = 2.0 if self.cell["repair_profile"] == "current" else 5.0
            repair = max(1.0, -np.log(float(row["repair_u"])) * mean * (1.0 + self.condition[idx]))
            self.crew_busy_until = start + repair
            self.sim._take_down(int(row["target"]))
            yield self.sim.env.timeout(repair)
            self.sim._bring_up(int(row["target"]))
            self.recent_failures[idx] += 1.0
            self.recent_downtime[idx] += repair
            self.corrective_hours += repair
            self.condition[idx] = max(0.0, self.condition[idx] - 0.15)
            self.records.append(MaintenanceRecord(
                "corrective", int(row["target"]), requested, start,
                float(self.sim.env.now), float(repair)
            ))
            self.sim.risk_events.append(RiskEvent(
                "R11", start, float(self.sim.env.now), repair,
                [int(row["target"])], "condition-mediated R11",
            ))

    def _quality_process(self):
        day = 0
        base_p = float(RISKS_CURRENT["R14"]["occurrence"]["p"])
        while day < len(self.tape["r14_daily_seeds"]):
            yield self.sim.env.timeout(HOURS_PER_DAY)
            produced = max(0.0, float(self.sim._today_produced))
            self.sim._today_produced = 0.0
            p = min(0.30, base_p * (0.5 + 1.5 * float(self.condition[2])))
            defects = int(np.random.default_rng(self.tape["r14_daily_seeds"][day]).binomial(int(produced), p))
            day += 1
            if defects <= 0:
                continue
            defects = min(defects, int(self.sim._pending_batch))
            if defects <= 0:
                continue
            self.sim._pending_batch -= defects
            self.sim.total_produced -= defects
            yield self.sim.rework_op6.put(defects)
            event = RiskEvent("R14", float(self.sim.env.now), float(self.sim.env.now), 0.0, [7], "condition-mediated defects", float(defects), "defective_products")
            self.sim.risk_events.append(event)
            self.sim._add_ret_quantity_risk(event)

    def exogenous_artifacts(self) -> dict[str, Any]:
        return {
            "base_exogenous_sha256": self.tape["base_exogenous_sha256"],
            "consumed_wear_sha256": digest(self.consumed_wear),
            "consumed_r11_candidates_sha256": digest(self.consumed_candidates),
        }


def make_sim(tape: dict[str, Any], cell: dict[str, Any] | None = None):
    cell = dict(default_cell() if cell is None else cell)
    horizon = 8_000.0 + float(tape["weeks"]) * HOURS_PER_WEEK + 336.0
    kwargs = proxy_kwargs()
    kwargs.update({
        "assembly_flow_mode": "serial_wip", "r14_defect_mode": "thesis_strict_op6",
        "serial_wip_capacity_rations": (
            float(cell["wip_capacity_days"]) * 8.0 * RATIONS_PER_HOUR,
            float(cell["wip_capacity_days"]) * 8.0 * RATIONS_PER_HOUR,
        ),
    })
    sim = MFSCSimulation(
        seed=int(tape["seed"]), horizon=horizon, risks_enabled=False,
        strict_exogenous_crn=True, **kwargs,
    )
    sim._start_processes()
    while not sim.warmup_complete:
        advance_including(sim, min(float(sim.env.now) + 1.0, sim.horizon))
        if sim.env.now >= sim.horizon:
            raise RuntimeError("maintenance lane did not warm up")
    start = float(sim.env.now)
    return sim, MaintenanceController(sim, tape, start, cell), start


def run_policy(
    tape: dict[str, Any], policy: Callable[[dict[str, float]], str],
    *, cell: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sim, controller, start = make_sim(tape, cell)
    end = start + int(tape["weeks"]) * HOURS_PER_WEEK
    for week in range(int(tape["weeks"])):
        controller.request(str(policy(controller.observation())), week)
        advance_including(sim, min(end, start + (week + 1) * HOURS_PER_WEEK))
    metrics = compute_episode_metrics(sim, treatment_start=start)
    ledger = sim.flow_ledger()
    metrics.update(controller.exogenous_artifacts())
    metrics.update({
        "scheduled_pm_hours": controller.scheduled_pm_hours,
        "executed_pm_hours": controller.executed_pm_hours,
        "corrective_hours": controller.corrective_hours,
        "mass_residual": max(abs(float(ledger["raw_residual"])), abs(float(ledger["ration_residual"]))),
        "blocked_hours": dict(controller.blocked_hours),
        "starved_hours": dict(controller.starved_hours),
        "action_events": controller.action_events,
        "maintenance_records": [record.__dict__ for record in controller.records],
    })
    return metrics


def periodic_policy(sequence: tuple[str, ...]) -> Callable[[dict[str, float]], str]:
    counter = {"i": 0}
    def policy(_: dict[str, float]) -> str:
        value = sequence[counter["i"] % len(sequence)]
        counter["i"] += 1
        return value
    return policy


def periodic_sequences(max_period: int = 6) -> tuple[tuple[str, ...], ...]:
    """All distinct minimal-period calendars through ``max_period``."""
    rows: list[tuple[str, ...]] = []
    for period in range(1, int(max_period) + 1):
        for sequence in product(ACTIONS, repeat=period):
            if any(
                period % divisor == 0
                and tuple(sequence) == tuple(sequence[:divisor]) * (period // divisor)
                for divisor in range(1, period)
            ):
                continue
            rows.append(tuple(sequence))
    return tuple(rows)


def worst_condition_policy(obs: dict[str, float]) -> str:
    return max(ACTIONS, key=lambda action: obs[f"condition_signal_op{ACTION_TO_OP[action]}"])


def wip_bottleneck_policy(obs: dict[str, float]) -> str:
    w56, w67 = obs["wip_op5_op6"], obs["wip_op6_op7"]
    if w56 <= 0.25 * max(w67, 1.0):
        return "PM7"
    if w67 <= 0.25 * max(w56, 1.0):
        return "PM5"
    return "PM6"
