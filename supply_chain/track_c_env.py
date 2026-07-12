"""Track C: campaign-regime environment with priced economics (2026-07-10).

`make_track_c_env` composes the track_bp_v1 11D contract (8D track_b_v1 +
3 lagged strategic-buffer fractions) with:
  - the `campaign_v1` calm/campaign regime process (CRN-safe exogenous
    schedule; exact thinning on R21/R22/R23/R24 frequencies; impact
    multipliers applied at event-fire time);
  - route-aware replenishment (a buffer top-up cannot arrive while its
    node/inbound LOC is down; it queues until the route reopens);
  - `risk_level='current'` (the campaign process replaces the adaptive
    Markov benchmark; multiplier knobs work under 'current').

Economics are computed by `TrackCEconomicsWrapper` from ACTUAL time-weighted
container stock (not target fractions), plus dispatch expediting excess and
shift level. The J_v3 objective is assembled by the gate runners from these
per-episode aggregates with lambdas frozen from Gate C0 baseline statistics
(see docs/TRACK_C_PREREGISTRATION_2026-07-10.md).

Design doc: docs/TRACK_C_FROM_ZERO_REDESIGN_2026-07-10.md.
Note: dim 4 (op5_q) of the base contract remains a disclosed no-op (the env
is built without an initial op5_rm buffer); op5 authority flows through
dim 9 (op5 buffer-target fraction) to avoid the op5_q/target write conflict.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from .config import CAMPAIGN_V1_CONFIG
from .track_bp_env import (
    BUFFER_FULL_SCALE,
    BUFFER_KEYS,
    TrackBPreventiveEnv,
    make_track_bp_env,
)

TRACK_C_ACTION_CONTRACT = "track_c_v1"
TRACK_C_ACTION_DIM = 11

# Containers charged for holding, and the full-scale capacity each level is
# normalized by (I_1344 in container units; op3/op5 targets are multiplied by
# NUM_RAW_MATERIALS by the sim's normalizer under bom_total_units flow modes,
# so capacities are computed with the same normalizer at wrapper init).
HOLDING_CONTAINER_BY_KEY = {
    "op3_rm": "raw_material_wdc",
    "op5_rm": "raw_material_al",
    "op9_rations": "rations_sb",
}


class TrackCEconomicsWrapper(gym.Wrapper):
    """Accumulate actual-inventory holding, dispatch excess, and shift usage.

    Exposes per-episode aggregates in `info['track_c_econ']` at every step:
      holding_frac_mean  — mean over steps of mean over the three charged
                           containers of level / I_1344 capacity (same
                           normalization as the sim's buffer targets). The
                           operating-stock floor is included on purpose: every
                           policy pays for the stock it actually holds, and
                           the floor cancels in paired contrasts.
      dispatch_excess_mean — mean over steps of (m10-1)+ + (m12-1)+ decoded
                           from the executed action.
      shift_excess_mean  — mean over steps of (S-1).
      campaign_frac      — fraction of steps spent in the campaign state.
    """

    def __init__(self, env: TrackBPreventiveEnv):
        super().__init__(env)
        sim = getattr(env.unwrapped, "sim", None)
        self._caps = self._capacities(sim)
        self._reset_accumulators()

    def _capacities(self, sim: Any) -> dict[str, float]:
        raw = {key: float(BUFFER_FULL_SCALE[key]) for key in BUFFER_KEYS}
        if sim is not None and hasattr(sim, "_normalize_inventory_buffer_targets"):
            return {
                k: max(1.0, float(v))
                for k, v in sim._normalize_inventory_buffer_targets(raw).items()
            }
        return raw

    def _reset_accumulators(self) -> None:
        self._n_steps = 0
        self._holding_sum = 0.0
        self._dispatch_sum = 0.0
        self._shift_sum = 0.0
        self._campaign_steps = 0

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        # Capacities depend on the freshly constructed sim's flow mode.
        self._caps = self._capacities(getattr(self.env.unwrapped, "sim", None))
        self._reset_accumulators()
        return obs, info

    def step(self, action: Any):
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, terminated, truncated, info = self.env.step(arr)
        info = dict(info)
        detail = info.get("inventory_detail") or {}
        fracs = []
        for key, container in HOLDING_CONTAINER_BY_KEY.items():
            cap = self._caps.get(key, float(BUFFER_FULL_SCALE[key]))
            fracs.append(float(detail.get(container, 0.0)) / cap)
        holding = float(np.mean(fracs)) if fracs else 0.0
        # track_b_v1 decode: dispatch multiplier = 1.25 + 0.75 * signal.
        m10 = 1.25 + 0.75 * float(np.clip(arr[6], -1.0, 1.0))
        m12 = 1.25 + 0.75 * float(np.clip(arr[7], -1.0, 1.0))
        dispatch_excess = max(0.0, m10 - 1.0) + max(0.0, m12 - 1.0)
        shift_signal = float(np.clip(arr[5], -1.0, 1.0))
        shifts = 1 if shift_signal < -0.33 else (2 if shift_signal < 0.33 else 3)

        self._n_steps += 1
        self._holding_sum += holding
        self._dispatch_sum += dispatch_excess
        self._shift_sum += float(shifts - 1)
        if info.get("campaign_state") == "campaign":
            self._campaign_steps += 1

        n = max(1, self._n_steps)
        info["track_c_econ"] = {
            "holding_frac_mean": self._holding_sum / n,
            "dispatch_excess_mean": self._dispatch_sum / n,
            "shift_excess_mean": self._shift_sum / n,
            "campaign_frac": self._campaign_steps / n,
            "n_steps": self._n_steps,
        }
        return obs, reward, terminated, truncated, info


def make_track_c_env(
    *,
    campaign_config: dict[str, Any] | None = None,
    replenishment_route_aware: bool = True,
    inventory_replenishment_lead_time: float = 168.0,
    **overrides: Any,
) -> TrackCEconomicsWrapper:
    """Build the Track C env: track_bp 11D + campaign_v1 + priced physics."""
    cfg = dict(CAMPAIGN_V1_CONFIG if campaign_config is None else campaign_config)
    defaults: dict[str, Any] = {
        "reward_mode": "control_v1",
        "observation_version": "v10",
        "risk_level": "current",
        "step_size_hours": 168.0,
        "max_steps": 104,
        "surge_inertia": True,
    }
    defaults.update(overrides)
    base = make_track_bp_env(
        inventory_replenishment_lead_time=float(inventory_replenishment_lead_time),
        campaign_config=cfg,
        replenishment_route_aware=bool(replenishment_route_aware),
        **defaults,
    )
    return TrackCEconomicsWrapper(base)


def j_v3(
    ret_excel: float,
    econ: dict[str, float],
    lambdas: dict[str, float],
) -> float:
    """Cost-adjusted Excel ReT (the single Track C objective).

    lambdas carry keys lam_h / lam_d / lam_s, frozen from Gate C0 baseline
    statistics BEFORE any optimization (see the pre-registration doc).
    """
    return (
        float(ret_excel)
        - float(lambdas["lam_h"]) * float(econ["holding_frac_mean"])
        - float(lambdas["lam_d"]) * float(econ["dispatch_excess_mean"])
        - float(lambdas["lam_s"]) * float(econ["shift_excess_mean"])
    )
