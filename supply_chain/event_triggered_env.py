"""Daily multiscale intervention wrapper for the thesis-faithful Track-A surface.

The wrapper adds a genuine HOLD decision.  HOLD advances the physical DES by one
day while preserving the last requested posture; it does not reissue inventory
top-ups.  INTERVENE may change the requested shift immediately (subject to the
existing daily ramp and surge-hour budget) and may commit one common strategic
buffer target with a seven-day physical lead.  Pending buffer commitments cannot
be cancelled or overwritten.

This module is an oracle/comparator surface.  It does not alter the canonical ReT
and it contains no knowledge of future risk events.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np

from .config import CAPACITY_BY_SHIFTS, HOURS_PER_DAY, HOURS_PER_WEEK, INVENTORY_BUFFERS, OPERATIONS
from .external_env_interface import make_thesis_aligned_training_env


_BUFFER_KEYS = ("op3_rm", "op5_rm", "op9_rations")
_I1344 = INVENTORY_BUFFERS[1344]


class EventTriggeredTrackAEnv(gym.Wrapper):
    """Track-A posture with daily ``HOLD | INTERVENE`` opportunities.

    Action is ``[gate, common_buffer_fraction, shift_signal]``.  ``gate <= 0``
    means HOLD.  A positive gate records an intervention; shift signals map to
    S1/S2/S3 using the frozen Track-A bands.  Buffer changes activate after
    ``buffer_lead_hours`` and are rate-limited to one commitment per week.
    """

    action_contract = "track_a_event_triggered_multiscale_v1"

    def __init__(
        self,
        env: gym.Env,
        *,
        init_frac: float = 0.0,
        init_shifts: int = 1,
        review_hours: float = HOURS_PER_DAY,
        buffer_lead_hours: float = HOURS_PER_WEEK,
        buffer_commit_cooldown_hours: float = HOURS_PER_WEEK,
        replenishment_period_hours: float = HOURS_PER_WEEK,
    ) -> None:
        super().__init__(env)
        if abs(float(getattr(env.unwrapped, "step_size", review_hours)) - review_hours) > 1e-9:
            raise ValueError("base environment step_size must equal review_hours")
        if min(review_hours, buffer_lead_hours, buffer_commit_cooldown_hours) <= 0.0:
            raise ValueError("review, lead, and cooldown hours must be positive")
        if int(init_shifts) not in (1, 2, 3):
            raise ValueError("init_shifts must be one of 1, 2, 3")

        self.review_hours = float(review_hours)
        self.buffer_lead_hours = float(buffer_lead_hours)
        self.buffer_commit_cooldown_hours = float(buffer_commit_cooldown_hours)
        self.replenishment_period_hours = float(replenishment_period_hours)
        self.init_frac = float(np.clip(init_frac, 0.0, 1.0))
        self.init_shifts = int(init_shifts)
        self.action_space = gym.spaces.Box(
            low=np.asarray([-1.0, 0.0, -1.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        )
        # control-state tail: current target, requested/effective shift, pending
        # commitment, cooldown, time since intervention, surge budget fraction.
        low = np.concatenate(
            [env.observation_space.low, np.zeros(7, dtype=np.float32)]
        )
        high = np.concatenate(
            [env.observation_space.high, np.ones(7, dtype=np.float32)]
        )
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self._reset_control_state()

    def _reset_control_state(self) -> None:
        self.current_frac = self.init_frac
        self.requested_shift = self.init_shifts
        self.pending_buffer_commitment: dict[str, Any] | None = None
        self.last_buffer_commit_time = float("-inf")
        self.last_intervention_time = float("-inf")
        self.intervention_count = 0
        self.buffer_commitment_count = 0
        self.rejected_buffer_commitment_count = 0
        self.action_trajectory: list[dict[str, Any]] = []

    @staticmethod
    def _shift_from_signal(signal: float) -> int:
        if signal < -0.33:
            return 1
        if signal < 0.33:
            return 2
        return 3

    @staticmethod
    def _validate_action(action: Any) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.shape != (3,):
            raise ValueError(f"event-triggered action must have shape (3,), got {arr.shape}")
        return np.clip(
            arr,
            np.asarray([-1.0, 0.0, -1.0], dtype=np.float32),
            np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        )

    @staticmethod
    def _normalized_targets(sim: Any, frac: float) -> dict[str, float]:
        if frac <= 1e-9:
            return {}
        targets = {key: float(frac) * float(_I1344[key]) for key in _BUFFER_KEYS}
        if hasattr(sim, "_normalize_inventory_buffer_targets"):
            return dict(sim._normalize_inventory_buffer_targets(targets))
        return targets

    def _activate_buffer_commitment(
        self, *, frac: float, targets: Mapping[str, float], delay_hours: float
    ):
        """SimPy process: activate exactly one immutable commitment after its lead."""
        sim = self.unwrapped.sim
        yield sim.env.timeout(float(delay_hours))
        sim.inventory_buffer_targets = dict(targets)
        sim.inventory_replenishment_period = (
            self.replenishment_period_hours if targets else None
        )
        for key, target in targets.items():
            sim._top_up_inventory_buffer(key, float(target))
        self.current_frac = float(frac)
        self.pending_buffer_commitment = None

    def _commit_buffer(self, frac: float, now: float) -> tuple[bool, str]:
        if self.pending_buffer_commitment is not None:
            self.rejected_buffer_commitment_count += 1
            return False, "pending_commitment"
        if now - self.last_buffer_commit_time < self.buffer_commit_cooldown_hours - 1e-9:
            self.rejected_buffer_commitment_count += 1
            return False, "weekly_cooldown"
        if abs(frac - self.current_frac) <= 1e-9:
            return False, "unchanged"

        sim = self.unwrapped.sim
        targets = self._normalized_targets(sim, frac)
        due = now + self.buffer_lead_hours
        self.pending_buffer_commitment = {
            "committed_at": now,
            "activates_at": due,
            "fraction": float(frac),
            "targets": dict(targets),
        }
        self.last_buffer_commit_time = now
        self.buffer_commitment_count += 1
        sim.env.process(
            self._activate_buffer_commitment(
                frac=float(frac),
                targets=targets,
                # SimPy's run(until=t) stops before normal events at t.  Put the
                # activation at the immediately preceding float so the stated
                # convention is: commitments due at a review boundary activate
                # before that boundary's observation.
                delay_hours=max(0.0, self.buffer_lead_hours - 1e-6),
            )
        )
        return True, "accepted"

    def _posture_action(self) -> dict[str, float | int]:
        sim = self.unwrapped.sim
        shifts = int(self.requested_shift)
        cap = CAPACITY_BY_SHIFTS[shifts]
        return {
            "assembly_shifts": shifts,
            "op3_q": float(cap["op3_q"]),
            "op3_rop": float(OPERATIONS[3]["rop"]),
            "op9_q_min": float(sim.params.get("op9_q_min", OPERATIONS[9]["q"][0])),
            "op9_q_max": float(sim.params.get("op9_q_max", OPERATIONS[9]["q"][1])),
            "op9_rop": float(OPERATIONS[9]["rop"]),
            "batch_size": float(cap["op7_q"]),
        }

    def _augment(self, obs: Any) -> np.ndarray:
        now = float(self.unwrapped.sim.env.now)
        pending = self.pending_buffer_commitment
        cooldown_remaining = max(
            0.0,
            self.buffer_commit_cooldown_hours - (now - self.last_buffer_commit_time),
        )
        since_intervention = (
            self.buffer_commit_cooldown_hours
            if not np.isfinite(self.last_intervention_time)
            else max(0.0, now - self.last_intervention_time)
        )
        budget_total = float(getattr(self.unwrapped, "surge_budget_hours", 0.0))
        budget_left = float(getattr(self.unwrapped, "_surge_budget_remaining", 0.0))
        budget_fraction = 1.0 if not np.isfinite(budget_total) else (
            max(0.0, min(1.0, budget_left / max(budget_total, 1.0)))
        )
        tail = np.asarray(
            [
                self.current_frac,
                (self.requested_shift - 1) / 2.0,
                (int(getattr(self.unwrapped, "_effective_shift", 1)) - 1) / 2.0,
                1.0 if pending is not None else 0.0,
                cooldown_remaining / self.buffer_commit_cooldown_hours,
                min(1.0, since_intervention / self.buffer_commit_cooldown_hours),
                budget_fraction,
            ],
            dtype=np.float32,
        )
        return np.concatenate([np.asarray(obs, dtype=np.float32), tail])

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._reset_control_state()
        reset_options = dict(options or {})
        reset_options.setdefault("initial_shifts", self.init_shifts)
        obs, info = self.env.reset(seed=seed, options=reset_options)
        sim = self.unwrapped.sim
        targets = self._normalized_targets(sim, self.init_frac)
        sim.inventory_buffer_targets = dict(targets)
        sim.inventory_replenishment_period = (
            self.replenishment_period_hours if targets else None
        )
        info = dict(info)
        info.update(self._control_info(phase="reset", buffer_status="initial"))
        return self._augment(obs), info

    def _control_info(self, *, phase: str, buffer_status: str) -> dict[str, Any]:
        return {
            "action_contract": self.action_contract,
            "action_phase": phase,
            "requested_shift": int(self.requested_shift),
            "effective_shift": int(getattr(self.unwrapped, "_effective_shift", 1)),
            "current_buffer_fraction": float(self.current_frac),
            "pending_buffer_commitment": (
                dict(self.pending_buffer_commitment)
                if self.pending_buffer_commitment is not None
                else None
            ),
            "buffer_commitment_status": buffer_status,
            "intervention_count": int(self.intervention_count),
            "buffer_commitment_count": int(self.buffer_commitment_count),
            "rejected_buffer_commitment_count": int(
                self.rejected_buffer_commitment_count
            ),
        }

    def step(self, action: Any):
        arr = self._validate_action(action)
        intervene = bool(arr[0] > 0.0)
        now = float(self.unwrapped.sim.env.now)
        buffer_status = "hold"
        if intervene:
            self.intervention_count += 1
            self.last_intervention_time = now
            self.requested_shift = self._shift_from_signal(float(arr[2]))
            accepted, buffer_status = self._commit_buffer(float(arr[1]), now)
            if accepted:
                buffer_status = "accepted"

        obs, reward, terminated, truncated, info = self.env.step(self._posture_action())
        row = {
            "decision_time": now,
            "advance_hours": self.review_hours,
            "decision": "INTERVENE" if intervene else "HOLD",
            "requested_shift": int(self.requested_shift),
            "effective_shift_after": int(getattr(self.unwrapped, "_effective_shift", 1)),
            "current_buffer_fraction_after": float(self.current_frac),
            "buffer_commitment_status": buffer_status,
        }
        self.action_trajectory.append(row)
        info = dict(info)
        info.update(
            self._control_info(
                phase="daily_intervention" if intervene else "daily_hold",
                buffer_status=buffer_status,
            )
        )
        info["action_trajectory_last"] = dict(row)
        return self._augment(obs), float(reward), bool(terminated), bool(truncated), info


def make_event_triggered_track_a_env(**overrides: Any) -> EventTriggeredTrackAEnv:
    """Build the daily multiscale Track-A oracle/comparator environment."""
    init_frac = float(overrides.pop("init_frac", 0.0))
    init_shifts = int(overrides.pop("init_shifts", 1))
    review_hours = float(overrides.pop("review_hours", HOURS_PER_DAY))
    buffer_lead_hours = float(overrides.pop("buffer_lead_hours", HOURS_PER_WEEK))
    cooldown = float(
        overrides.pop("buffer_commit_cooldown_hours", HOURS_PER_WEEK)
    )
    replenishment = float(
        overrides.pop("replenishment_period_hours", HOURS_PER_WEEK)
    )
    overrides["step_size_hours"] = review_hours
    overrides.setdefault("observation_version", "v10")
    overrides.setdefault("surge_inertia", True)
    overrides.setdefault("surge_ramp_per_step", 1)
    overrides.setdefault("initial_shifts", init_shifts)
    if "initial_buffers" not in overrides and init_frac > 1e-9:
        overrides["initial_buffers"] = {
            key: init_frac * float(_I1344[key]) for key in _BUFFER_KEYS
        }
    # Buffer timing is owned by this wrapper; disabling the simulator's generic
    # delayed top-up prevents duplicate commitments.
    overrides["inventory_replenishment_lead_time"] = 0.0
    base = make_thesis_aligned_training_env(**overrides)
    return EventTriggeredTrackAEnv(
        base,
        init_frac=init_frac,
        init_shifts=init_shifts,
        review_hours=review_hours,
        buffer_lead_hours=buffer_lead_hours,
        buffer_commit_cooldown_hours=cooldown,
        replenishment_period_hours=replenishment,
    )
