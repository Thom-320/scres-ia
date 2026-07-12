"""Track B-P: preventive action contract composing track_b_v1 with lagged buffers.

`track_bp_v1` extends the canonical 8D `track_b_v1` contract with three
strategic-buffer target fractions (Op3 raw material, Op5 raw material,
Op9 rations) drawn from Garrido's I_{t,S} decision family (Table 6.16).
Raising a target only takes effect `inventory_replenishment_lead_time`
hours later (`MFSCSimulation._delayed_buffer_top_up`), so — unlike every
dim of `track_b_v1` — the buffer lever carries temporal commitment.
NOTE the commitment is a lag, not a hard ex-ante constraint: in regimes
whose outages outlast the lead time (e.g. R21 at impact multipliers,
multi-week recoveries), a post-onset target raise still lands mid-outage
and `_top_up_inventory_buffer` injects stock regardless of route status.
Value extracted through this lever therefore mixes ex-ante positioning
with lagged reaction; separating the two requires the within-checkpoint
blocking controls (see scripts/audit_track_bp_timing_within.py), not the
contract alone.

The instant track_b dims are kept intact on purpose: the preventive
question is whether the lagged lever adds value ON TOP of the best
reactive contract, not against a handicapped one.

Buffer holding is intentionally not priced into the training reward here
(mirrors dispatch, which is priced via the post-hoc cost sensitivity);
Gate 0/1 score physics on local exposed-order ReT, not reward.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from .config import INVENTORY_BUFFERS
from .external_env_interface import make_track_b_env

BUFFER_KEYS: tuple[str, ...] = ("op3_rm", "op5_rm", "op9_rations")
# Full-scale reference: the largest thesis buffer level (I_1344, Table 6.16).
BUFFER_FULL_SCALE: dict[str, float] = {
    key: float(INVENTORY_BUFFERS[1344][key]) for key in BUFFER_KEYS
}

TRACK_BP_ACTION_CONTRACT = "track_bp_v1"
TRACK_BP_ACTION_DIM = 11
TRACK_B_FIXED_BUFFER_ACTION_CONTRACT = "track_b_fixed_buffers_v1"


class TrackBPreventiveEnv(gym.Wrapper):
    """11D wrapper over a `track_b_v1` base env.

    dims 0-7: passed through verbatim to the base `track_b_v1` decode
              (op3/op9 qty+ROP, op5 qty, shift, op10/op12 dispatch).
    dims 8-10: buffer target fractions in [0, 1] for op3_rm / op5_rm /
              op9_rations, scaled by I_1344. Each decision step the
              targets are (re)emitted and an order-up-to top-up is
              scheduled after the sim's replenishment lead time —
              a weekly review with lead, Garrido's I_168 cadence.
    """

    action_contract = TRACK_BP_ACTION_CONTRACT

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        base_space = env.action_space
        if base_space.shape != (8,):
            raise ValueError(
                "TrackBPreventiveEnv requires a track_b_v1 (8D) base env, "
                f"got action space shape {base_space.shape}."
            )
        self.action_space = gym.spaces.Box(
            low=np.concatenate(
                [base_space.low, np.zeros(3, dtype=np.float32)]
            ).astype(np.float32),
            high=np.concatenate(
                [base_space.high, np.ones(3, dtype=np.float32)]
            ).astype(np.float32),
            dtype=np.float32,
        )
        self._last_fracs: dict[str, float] = {key: 0.0 for key in BUFFER_KEYS}

    # ------------------------------------------------------------------ decode
    def _validate_action(self, action: Any) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.shape != (TRACK_BP_ACTION_DIM,):
            raise ValueError(
                f"track_bp_v1 action must have shape ({TRACK_BP_ACTION_DIM},), "
                f"got {arr.shape}."
            )
        return np.clip(arr, self.action_space.low, self.action_space.high)

    def _apply_buffer_targets(self, fracs: dict[str, float]) -> dict[str, float]:
        """Write buffer targets on the sim and schedule the lagged top-up.

        Mirrors PerOpBufferTrackAEnv._set_targets_by_fracs
        (continuous_its_env.py) but leaves `inventory_replenishment_period`
        alone: the weekly action re-emission IS the review cadence, and the
        sim-level periodic loop only runs when the env was constructed with
        initial buffers.
        """
        sim = getattr(self.env.unwrapped, "sim", None)
        if sim is None:
            return {}
        targets = {
            key: float(fracs[key]) * BUFFER_FULL_SCALE[key]
            for key in BUFFER_KEYS
            if fracs[key] > 1e-6
        }
        if not targets:
            sim.inventory_buffer_targets = {}
            return {}
        if hasattr(sim, "_normalize_inventory_buffer_targets"):
            internal = sim._normalize_inventory_buffer_targets(targets)
        else:
            internal = dict(targets)
        sim.inventory_buffer_targets = dict(internal)
        lead = float(getattr(sim, "inventory_replenishment_lead_time", 0.0) or 0.0)
        if lead > 0.0:
            sim.env.process(sim._delayed_buffer_top_up(lead))
        else:
            for key, target in internal.items():
                sim._top_up_inventory_buffer(key, float(target))
        return targets

    # ------------------------------------------------------------------ gym API
    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        self._last_fracs = {key: 0.0 for key in BUFFER_KEYS}
        return obs, info

    def step(self, action: Any):
        arr = self._validate_action(action)
        base_action = arr[:8]
        fracs = {
            key: float(np.clip(arr[8 + i], 0.0, 1.0))
            for i, key in enumerate(BUFFER_KEYS)
        }
        # Schedule the (lagged) top-up before advancing the sim so the lead
        # clock starts at the current decision epoch.
        targets = self._apply_buffer_targets(fracs)
        self._last_fracs = dict(fracs)
        obs, reward, terminated, truncated, info = self.env.step(base_action)
        info = dict(info)
        info["track_bp_buffer_fracs"] = dict(fracs)
        info["track_bp_buffer_targets"] = dict(targets)
        return obs, reward, terminated, truncated, info


def make_track_bp_env(
    *,
    inventory_replenishment_lead_time: float = 168.0,
    **overrides: Any,
) -> TrackBPreventiveEnv:
    """Build the Track B-P preventive env (track_b_v1 base + lagged buffers).

    All `make_track_b_env` overrides pass through; the lead time defaults to
    one review period (168h) and can be raised (e.g. 336h) to harden the
    commitment. `initial_buffers`/`inventory_replenishment_period` overrides
    additionally start the sim-level periodic review loop.
    """
    base = make_track_b_env(
        inventory_replenishment_lead_time=float(inventory_replenishment_lead_time),
        **overrides,
    )
    return TrackBPreventiveEnv(base)


class FixedBufferTrackBEnv(gym.Wrapper):
    """Expose the canonical 8D contract while holding three buffers fixed.

    This is the strong mechanism baseline for Track B-P: the adaptive policy
    controls exactly the original eight Track-B dimensions, while the
    strategic-buffer posture is frozen before training and re-emitted every
    week through the same lead-time machinery as ``track_bp_v1``.
    """

    action_contract = TRACK_B_FIXED_BUFFER_ACTION_CONTRACT

    def __init__(self, env: TrackBPreventiveEnv, fixed_fracs: tuple[float, float, float]):
        super().__init__(env)
        fracs = np.asarray(fixed_fracs, dtype=np.float32).reshape(-1)
        if fracs.shape != (3,):
            raise ValueError(f"fixed_fracs must have shape (3,), got {fracs.shape}")
        self.fixed_fracs = np.clip(fracs, 0.0, 1.0)
        self.action_space = gym.spaces.Box(
            low=np.asarray(env.action_space.low[:8], dtype=np.float32),
            high=np.asarray(env.action_space.high[:8], dtype=np.float32),
            dtype=np.float32,
        )

    def step(self, action: Any):
        base_action = np.asarray(action, dtype=np.float32).reshape(-1)
        if base_action.shape != (8,):
            raise ValueError(f"fixed-buffer Track B action must have shape (8,), got {base_action.shape}")
        full_action = np.concatenate([base_action, self.fixed_fracs]).astype(np.float32)
        obs, reward, terminated, truncated, info = self.env.step(full_action)
        info = dict(info)
        info["track_bp_fixed_buffer_fracs"] = {
            key: float(self.fixed_fracs[i]) for i, key in enumerate(BUFFER_KEYS)
        }
        return obs, reward, terminated, truncated, info


def make_track_b_fixed_buffer_env(
    *,
    fixed_fracs: tuple[float, float, float],
    inventory_replenishment_lead_time: float = 168.0,
    **overrides: Any,
) -> FixedBufferTrackBEnv:
    """Build an 8D Track-B learner with a frozen strategic-buffer posture."""
    preventive = make_track_bp_env(
        inventory_replenishment_lead_time=inventory_replenishment_lead_time,
        **overrides,
    )
    return FixedBufferTrackBEnv(preventive, fixed_fracs)
