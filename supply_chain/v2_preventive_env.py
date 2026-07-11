"""Program-v2 preventive-control environment.

This lane is a disclosed change of decision class, not a repair or extension of
the thesis-native Program L contract.  A finite reserve is positioned behind
the threatened Op10--Op12 corridor.  The controller receives an explicitly
imperfect warning and selects a daily order-up-to target.  All replenishment is
stock-conserving, lead-time constrained, and route aware.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import HOURS_PER_WEEK
from .env_experimental_shifts import MFSCGymEnvShifts
from .episode_metrics import compute_episode_metrics
from .l_program_env import CampaignTape, PROXY_CONTRACT_PATH


CONTRACT_ID = "preventive_reserve_v2"
STEP_HOURS = 24.0
WARNING_LEAD_HOURS = 336.0
RESERVE_TARGETS = (0.0, 15_000.0, 30_000.0)
ELIGIBLE_RISKS = {"R22", "R23"}

OBSERVATION_FIELDS = (
    "reserve_level_fraction",
    "reserve_target_fraction",
    "reserve_in_transit_fraction",
    "warning_active",
    "warning_lead_remaining_fraction",
    "rations_sb_fraction",
    "pending_backlog_qty_fraction",
    "pending_backlog_count_fraction",
    "oldest_backlog_age_fraction",
    "rolling_fill_rate",
    "downstream_corridor_down",
    "previous_day_demand_fraction",
    "previous_day_delivered_fraction",
    "time_fraction",
)


@dataclass(frozen=True)
class WarningInterval:
    start_time: float
    expiry_time: float
    source: str  # true | false
    event_key: str

    def payload(self) -> dict[str, Any]:
        return {
            "start_time": float(self.start_time),
            "expiry_time": float(self.expiry_time),
            "source": self.source,
            "event_key": self.event_key,
        }


@dataclass(frozen=True)
class WarningSchedule:
    tape_sha256: str
    mode: str
    seed: int
    intervals: tuple[WarningInterval, ...]
    false_negative_probability: float = 0.20
    false_positive_opportunity_probability: float = 0.20

    def payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "contract_id": CONTRACT_ID,
            "tape_sha256": self.tape_sha256,
            "mode": self.mode,
            "seed": int(self.seed),
            "false_negative_probability": self.false_negative_probability,
            "false_positive_opportunity_probability": (
                self.false_positive_opportunity_probability
            ),
            "intervals": [row.payload() for row in self.intervals],
        }
        if include_hash:
            raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            payload["sha256"] = sha256(raw.encode("utf-8")).hexdigest()
        return payload


def build_warning_schedule(
    tape: CampaignTape,
    *,
    seed: int,
    mode: str = "imperfect",
) -> WarningSchedule:
    """Build warnings without changing the realized risk tape.

    ``imperfect`` applies a 20% false-negative draw to each eligible event and
    an independent 20% false-warning opportunity per eligible event.  A false
    warning is placed on an event-free day.  ``shuffled_placebo`` preserves the
    imperfect schedule's alert count and durations but shifts every interval by
    a deterministic half-horizon permutation, destroying temporal usefulness.
    """
    if not tape.calendar_materialized:
        raise ValueError("Program-v2 warnings require a materialized CampaignTape.")
    if mode not in {"perfect", "imperfect", "shuffled_placebo", "none"}:
        raise ValueError(f"Unsupported warning mode {mode!r}.")
    rng = np.random.default_rng(int(seed))
    horizon = float(tape.horizon_weeks) * HOURS_PER_WEEK
    eligible = [
        row
        for row in tape.risk_events
        if str(row.get("risk_id")) in ELIGIBLE_RISKS
        and bool({10, 11, 12}.intersection(int(op) for op in row.get("affected_ops", ())))
    ]
    intervals: list[WarningInterval] = []
    if mode != "none":
        for idx, event in enumerate(eligible):
            detected = mode == "perfect" or rng.random() >= 0.20
            if detected:
                onset = float(event["start_time"])
                start = max(0.0, onset - WARNING_LEAD_HOURS)
                intervals.append(
                    WarningInterval(
                        start_time=start,
                        expiry_time=min(horizon, start + WARNING_LEAD_HOURS),
                        source="true",
                        event_key=f"{event['risk_id']}:{idx}:{onset:.6f}",
                    )
                )
            if mode != "perfect" and rng.random() < 0.20:
                # Candidate false-warning days are drawn away from all eligible
                # onsets by at least the warning lead.  This makes the placebo
                # falsifiable rather than a near-miss alert in disguise.
                for _ in range(1_000):
                    start = float(rng.integers(0, max(1, int(horizon // 24)))) * 24.0
                    if all(
                        abs(start + WARNING_LEAD_HOURS - float(e["start_time"]))
                        >= WARNING_LEAD_HOURS
                        for e in eligible
                    ):
                        intervals.append(
                            WarningInterval(
                                start_time=start,
                                expiry_time=min(horizon, start + WARNING_LEAD_HOURS),
                                source="false",
                                event_key=f"false:{idx}:{start:.6f}",
                            )
                        )
                        break

    intervals.sort(key=lambda row: (row.start_time, row.event_key))
    if mode == "shuffled_placebo" and intervals:
        shifted: list[WarningInterval] = []
        offset = max(24.0, horizon / 2.0)
        for row in intervals:
            start = (row.start_time + offset) % max(24.0, horizon)
            shifted.append(
                WarningInterval(
                    start_time=start,
                    expiry_time=min(horizon, start + WARNING_LEAD_HOURS),
                    source=row.source,
                    event_key=f"placebo:{row.event_key}",
                )
            )
        intervals = sorted(shifted, key=lambda row: (row.start_time, row.event_key))

    return WarningSchedule(
        tape_sha256=tape.digest(),
        mode=mode,
        seed=int(seed),
        intervals=tuple(intervals),
    )


class PreventiveReserveV2Env(gym.Env[np.ndarray, int]):
    """Daily finite-reserve controller with an imperfect observable warning."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        horizon_weeks: int = 104,
        reserve_capacity: float = 30_000.0,
        holding_cost_per_unit_day: float = 1.0,
        target_change_cost: float = 500.0,
        proxy_contract_path: str | Path = PROXY_CONTRACT_PATH,
        contract_id: str = CONTRACT_ID,
        replenishment_transport_mode: str = "fixed_lead",
        replenishment_lead_hours: float = WARNING_LEAD_HOURS,
    ) -> None:
        super().__init__()
        self.horizon_weeks = int(horizon_weeks)
        self.max_steps = self.horizon_weeks * 7
        self.reserve_capacity = float(reserve_capacity)
        self.contract_id = str(contract_id)
        self.replenishment_transport_mode = str(replenishment_transport_mode)
        self.replenishment_lead_hours = max(0.0, float(replenishment_lead_hours))
        if self.reserve_capacity < max(RESERVE_TARGETS):
            raise ValueError("reserve_capacity must support every discrete target.")
        self.holding_cost_per_unit_day = max(0.0, float(holding_cost_per_unit_day))
        self.target_change_cost = max(0.0, float(target_change_cost))
        proxy_path = Path(proxy_contract_path)
        proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
        sim = dict(proxy["sim_kwargs"])
        self.proxy_sha256 = sha256(proxy_path.read_bytes()).hexdigest()
        self._base = MFSCGymEnvShifts(
            step_size_hours=STEP_HOURS,
            max_steps=self.max_steps,
            risk_level=str(sim["risk_level"]),
            reward_mode="control_v1",
            observation_version="v1",
            action_contract="track_a_v1",
            action_mode="full",
            year_basis=str(sim["year_basis"]),
            stochastic_pt=False,
            warmup_hours_override=0.0,
            warmup_trigger=str(sim["warmup_trigger"]),
            raw_material_flow_mode=str(sim["raw_material_flow_mode"]),
            raw_material_order_up_to_multiplier=float(
                sim["raw_material_order_up_to_multiplier"]
            ),
            demand_on_hand_fulfillment_delay=float(
                sim["demand_on_hand_fulfillment_delay"]
            ),
            risk_occurrence_mode=str(sim["risk_occurrence_mode"]),
            risk_attribution_source=str(sim["risk_attribution_source"]),
            ret_recovery_period_mode=str(sim["ret_recovery_period_mode"]),
            replenishment_route_aware=bool(sim["replenishment_route_aware"]),
            procurement_contract_mode=str(sim["procurement_contract_mode"]),
            order_fulfillment_mode=str(sim["order_fulfillment_mode"]),
            op9_dispatch_policy=str(sim["op9_dispatch_policy"]),
            downstream_transport_capacity_mode=str(
                sim["downstream_transport_capacity_mode"]
            ),
            op9_freight_offset_hours=float(sim["op9_freight_offset_hours"]),
            r24_attribution_window_hours=float(sim["r24_attribution_window_hours"]),
            demand_start_after_warmup=bool(sim["demand_start_after_warmup"]),
            priming_enabled=False,
            clear_backlog_after_priming=False,
            risks_enabled=False,
        )
        self.action_space = spaces.Discrete(len(RESERVE_TARGETS))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(OBSERVATION_FIELDS),),
            dtype=np.float32,
        )
        self.tape: CampaignTape | None = None
        self.warning_schedule: WarningSchedule | None = None
        self.treatment_start = 0.0
        self.previous_target = 0.0
        self.previous_info: dict[str, Any] = {}
        self.cumulative_cost = 0.0

    @property
    def sim(self) -> Any:
        return self._base.sim

    def _active_warning(self) -> tuple[float, float]:
        if self.warning_schedule is None:
            return 0.0, 0.0
        relative = float(self.sim.env.now) - self.treatment_start
        active = [
            row
            for row in self.warning_schedule.intervals
            if row.start_time <= relative < row.expiry_time
        ]
        if not active:
            return 0.0, 0.0
        remaining = max(row.expiry_time - relative for row in active)
        return 1.0, float(np.clip(remaining / WARNING_LEAD_HOURS, 0.0, 1.0))

    def _oldest_backlog_age(self) -> float:
        if not self.sim.pending_backorders:
            return 0.0
        oldest = min(float(order.OPTj) for order in self.sim.pending_backorders)
        return max(0.0, float(self.sim.env.now) - oldest)

    def _observation(self) -> np.ndarray:
        warning, lead = self._active_warning()
        reserve = self.sim.emergency_reserve_metrics()
        horizon = self.horizon_weeks * HOURS_PER_WEEK
        return np.asarray(
            [
                reserve["emergency_reserve_level"] / self.reserve_capacity,
                reserve["emergency_reserve_target"] / self.reserve_capacity,
                reserve["emergency_reserve_in_transit"] / self.reserve_capacity,
                warning,
                lead,
                float(self.sim.rations_sb.level) / 100_000.0,
                float(self.sim.pending_backorder_qty) / 100_000.0,
                float(len(self.sim.pending_backorders)) / 60.0,
                self._oldest_backlog_age() / 672.0,
                float(self.sim.get_observation_v7_extra()[4]),
                float(self.sim._emergency_corridor_down()),
                float(self.previous_info.get("new_demanded", 0.0)) / 2_500.0,
                float(self.previous_info.get("new_delivered", 0.0)) / 2_500.0,
                np.clip(
                    (float(self.sim.env.now) - self.treatment_start) / horizon,
                    0.0,
                    1.0,
                ),
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        options = dict(options or {})
        tape_value = options.pop("campaign_tape", None)
        if isinstance(tape_value, CampaignTape):
            tape = tape_value
        elif isinstance(tape_value, Mapping):
            tape = CampaignTape.from_mapping(tape_value)
        else:
            raise ValueError("reset requires a materialized campaign_tape.")
        if not tape.calendar_materialized:
            raise ValueError("Program-v2 requires a materialized campaign tape.")
        if int(tape.horizon_weeks) != self.horizon_weeks:
            raise ValueError("CampaignTape horizon differs from environment horizon.")
        warning_mode = str(options.pop("warning_mode", "imperfect"))
        warning_seed = int(options.pop("warning_seed", (seed or tape.base_seed) + 9001))
        initial_target_index = int(options.pop("initial_target_index", 0))
        if options:
            raise ValueError(f"Unknown reset options: {sorted(options)}")
        initial_target = RESERVE_TARGETS[initial_target_index]

        self._base.enabled_risks = set()
        self._base.risk_event_tape = None
        _, base_info = self._base.reset(
            seed=int(seed if seed is not None else tape.base_seed),
            options={"initial_buffers": {}, "initial_shifts": 2},
        )
        self.sim.configure_emergency_theatre_reserve(
            capacity=self.reserve_capacity,
            initial_stock=initial_target,
            target=initial_target,
            replenishment_lead_time=self.replenishment_lead_hours,
            issue_delay=24.0,
            route_ops=(10, 11, 12),
            transport_mode=self.replenishment_transport_mode,
        )
        self.treatment_start = float(self.sim.env.now)
        absolute_events = []
        for row in tape.risk_events:
            shifted = dict(row)
            shifted["start_time"] = self.treatment_start + float(row["start_time"])
            shifted["end_time"] = self.treatment_start + float(
                row.get("end_time", row["start_time"])
            )
            absolute_events.append(shifted)
        self.sim.risk_event_tape = self.sim._normalize_risk_event_tape(absolute_events)
        self.sim.env.process(self.sim._risk_event_tape_replay())
        self.tape = tape
        self.warning_schedule = build_warning_schedule(
            tape, seed=warning_seed, mode=warning_mode
        )
        self.previous_target = initial_target
        self.previous_info = {}
        self.cumulative_cost = 0.0
        return self._observation(), {
            **base_info,
            "contract_id": self.contract_id,
            "proxy_sha256": self.proxy_sha256,
            "campaign_tape": tape.payload(include_hash=True),
            "warning_schedule": self.warning_schedule.payload(),
            "reserve_targets": list(RESERVE_TARGETS),
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Expected target action 0..2, got {action!r}.")
        target = RESERVE_TARGETS[int(action)]
        changed = float(target != self.previous_target)
        before = self.sim.emergency_reserve_metrics()
        self.sim.request_emergency_reserve_target(target)
        _, _, terminated, truncated, info = self._base.step({"assembly_shifts": 2})
        after = self.sim.emergency_reserve_metrics()
        inventory_time_delta = max(
            0.0,
            after["emergency_reserve_inventory_time"]
            - before["emergency_reserve_inventory_time"],
        )
        holding_cost = (
            inventory_time_delta / 24.0 * self.holding_cost_per_unit_day
        )
        change_cost = changed * self.target_change_cost
        service_loss = max(
            0.0,
            float(info.get("new_demanded", 0.0))
            - float(info.get("new_order_fulfilled", 0.0)),
        )
        reward = -(service_loss + holding_cost + change_cost)
        self.cumulative_cost += holding_cost + change_cost
        self.previous_target = target
        self.previous_info = dict(info)
        enriched = {
            **info,
            "contract_id": self.contract_id,
            "warning_active": self._active_warning()[0],
            "reserve_target": target,
            "reserve_target_changed": changed,
            "reserve_inventory_time_step": inventory_time_delta,
            "reserve_holding_cost_step": holding_cost,
            "reserve_target_change_cost_step": change_cost,
            "service_loss_step": service_loss,
            "reserve_metrics": after,
            "v2_reward": reward,
        }
        return self._observation(), float(reward), terminated, truncated, enriched

    def terminal_metrics(self) -> dict[str, float]:
        metrics = compute_episode_metrics(self.sim, treatment_start=self.treatment_start)
        metrics.update(self.sim.emergency_reserve_metrics())
        metrics["emergency_reserve_cost"] = float(self.cumulative_cost)
        return metrics

    def close(self) -> None:
        self._base.close()
