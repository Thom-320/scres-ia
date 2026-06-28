"""Continuous I_{t,S} Track-A action contract (additive port of `continuous_it_s`).

Recovers the continuous-buffer idea (originally on branch `codex/garrido-postfix-reruns`,
commit feab7ac) as a standalone wrapper that does NOT edit the Codex-owned env files.

Keeps Garrido-Rios' SAME two decision variables, de-discretized:
  action[0] in [0, 1]  -> fraction of the I1344 buffer applied commonly at Op3/Op5/Op9
  action[1] in [-1, 1] -> continuous shift signal mapped to S1/S2/S3 (tri-level bands)

This is a continuous relaxation of the thesis buffer-size variable. It is NOT Track B
(no ROP control, no per-node buffers, no downstream Op10/Op12). Mirrors the buffer-target
and per-step action machinery of `ThesisFactorizedTrackAEnv`.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from .external_env_interface import make_thesis_aligned_training_env
from .config import INVENTORY_BUFFERS, OPERATIONS, CAPACITY_BY_SHIFTS

_I1344 = INVENTORY_BUFFERS[1344]  # {op3_rm, op5_rm, op9_rations}
_BUFFER_KEYS = ("op3_rm", "op5_rm", "op9_rations")

# v8 realized-risk observation block: pass the CAUSE (which risk is active / just occurred),
# not just the consequences (down flags). Non-oracular: only active or recently-elapsed risks.
RISK_IDS = ("R11", "R12", "R13", "R14", "R21", "R22", "R23", "R24", "R3")
RISK_FAMILIES = ("R1", "R2", "R3")
# Hazard block = history-derived risk expectation ("predict without an oracle"): how long since
# each family last hit (overdue -> build buffer) + an EWMA of the realized risk rate.
HAZARD_FEATURE_NAMES = (
    [f"weeks_since_last_{fam}" for fam in RISK_FAMILIES] + ["ewma_risk_rate"]
)
RISK_FEATURE_NAMES = (
    [f"active_{r}" for r in RISK_IDS]
    + [f"recent_{r}" for r in RISK_IDS]
    + ["active_risk_duration_norm", "n_active_risks_norm"]
    + HAZARD_FEATURE_NAMES
)


class ContinuousItsTrackAEnv(gym.Wrapper):
    """Box([0,-1],[1,1]) continuous relaxation of the thesis (I_{t,S}, S) decision."""

    action_contract = "track_a_continuous_its_v1"
    action_space_mode = "continuous_it_s"

    def __init__(self, env: gym.Env, *, init_frac: float | None = None,
                 risk_obs: bool = False, risk_recent_window: float = 336.0,
                 step_window: float = 168.0, base_field_names: list | None = None,
                 holding_cost: float = 0.0, shift_cost: float = 0.0,
                 ewma_decay: float = 0.85, replenishment_period: float = 168.0) -> None:
        super().__init__(env)
        self.init_frac = init_frac  # fixed pre-warmup prepositioning (fraction of I1344)
        self.risk_obs = bool(risk_obs)
        self.risk_recent_window = float(risk_recent_window)
        self.step_window = float(step_window)
        # balanced cost so prevention is a real decision (you can't hold max buffer for free)
        self.holding_cost = float(holding_cost)
        self.shift_cost = float(shift_cost)
        self.ewma_decay = float(ewma_decay)
        # strategic-buffer replenishment cadence (decoupled from decision cadence): shorter -> faster
        # recovery -> higher ReT, without paying 7x decision steps. Default = weekly (thesis).
        self.replenishment_period = float(replenishment_period)
        self._ewma_rate = 0.0
        self._max_buffer_units = float(sum(_I1344.values()))
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )
        self._base_obs_dim = int(env.observation_space.shape[0])
        if self.risk_obs:
            extra = len(RISK_FEATURE_NAMES)
            low = np.concatenate([env.observation_space.low, np.zeros(extra, dtype=np.float32)])
            high = np.concatenate([env.observation_space.high, np.ones(extra, dtype=np.float32)])
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
            self.obs_field_names = list(base_field_names or []) + list(RISK_FEATURE_NAMES)
        else:
            self.observation_space = env.observation_space
            self.obs_field_names = list(base_field_names or [])

    def _risk_features(self, update_ewma: bool = False) -> np.ndarray:
        """Realized-risk block (weekly cadence): risks are hourly but decisions are weekly, so
        'active_R*' = the risk's window overlapped the LAST decision step (step_window), and
        'recent_R*' = overlapped the last risk_recent_window. Non-oracular (no future risks)."""
        sim = getattr(self.unwrapped, "sim", None)
        feats = np.zeros(len(RISK_FEATURE_NAMES), dtype=np.float32)
        if sim is None or not getattr(sim, "risk_events", None):
            return feats
        now = float(sim.env.now)
        step_start = now - self.step_window
        recent_start = now - self.risk_recent_window
        n_step = 0
        step_dur = 0.0
        for ev in sim.risk_events:
            st, en, rid = float(ev.start_time), float(ev.end_time), str(ev.risk_id)
            if rid not in RISK_IDS:
                rid = rid[:2] if rid[:2] in RISK_IDS else ("R3" if rid == "R3" else None)
            if rid is None:
                continue
            ri = RISK_IDS.index(rid)
            if en >= step_start and st <= now:  # overlapped the last decision step
                feats[ri] = 1.0
                n_step += 1
                step_dur += max(0.0, min(en, now) - max(st, step_start))
            if en >= recent_start and st <= now:  # overlapped the recent window
                feats[len(RISK_IDS) + ri] = 1.0
        n_feat = len(RISK_FEATURE_NAMES)
        n_haz = len(HAZARD_FEATURE_NAMES)
        feats[n_feat - n_haz - 2] = float(np.clip(step_dur / max(self.step_window, 1.0), 0.0, 1.0))
        feats[n_feat - n_haz - 1] = float(np.clip(n_step / 10.0, 0.0, 1.0))
        # hazard block: weeks-since-last per family (overdue -> higher) + EWMA realized rate
        now_w = now
        for fi, fam in enumerate(RISK_FAMILIES):
            last_end = 0.0
            for ev in sim.risk_events:
                rid = str(ev.risk_id)
                famk = "R3" if rid == "R3" else rid[:2]
                if famk == fam and float(ev.end_time) <= now_w:
                    last_end = max(last_end, float(ev.end_time))
            weeks_since = (now_w - last_end) / 168.0 if last_end > 0 else now_w / 168.0
            feats[n_feat - n_haz + fi] = float(np.clip(weeks_since / 52.0, 0.0, 1.0))
        feats[n_feat - 1] = float(np.clip(self._ewma_rate / 10.0, 0.0, 1.0))
        if update_ewma:
            self._ewma_rate = self.ewma_decay * self._ewma_rate + (1.0 - self.ewma_decay) * n_step
        return feats

    def _augment(self, obs, update_ewma=False):
        if not self.risk_obs:
            return obs
        return np.concatenate(
            [np.asarray(obs, dtype=np.float32), self._risk_features(update_ewma=update_ewma)])

    def resource_composite(self, frac: float, shifts: int) -> float:
        """Per-step resource use, normalized [0,1]: buffer fraction + extra shifts. Charged to BOTH
        static and dynamic in evaluation so a profligate constant-S3/high-buffer static is not free."""
        return 0.5 * float(np.clip(frac, 0.0, 1.0)) + 0.5 * (float(int(shifts) - 1) / 2.0)

    @staticmethod
    def _validate_action(action: Any) -> np.ndarray:
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.shape != (2,):
            raise ValueError(
                f"Continuous I_t,S action must have shape (2,), got {action_array.shape}."
            )
        return np.clip(
            action_array,
            np.array([0.0, -1.0], dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32),
        )

    @staticmethod
    def _shift_from_signal(x: float) -> int:
        x = float(x)
        if x < -0.33:
            return 1
        if x < 0.33:
            return 2
        return 3

    def _set_targets(self, frac: float) -> dict[str, float]:
        sim = getattr(self.unwrapped, "sim", None)
        if sim is None:
            return {}
        frac = float(np.clip(frac, 0.0, 1.0))
        if frac <= 1e-6:
            sim.inventory_buffer_targets = {}
            sim.inventory_replenishment_period = None
            return {}
        targets = {k: frac * float(_I1344[k]) for k in _BUFFER_KEYS}
        if hasattr(sim, "_normalize_inventory_buffer_targets"):
            internal = sim._normalize_inventory_buffer_targets(targets)
        else:
            internal = dict(targets)
        sim.inventory_buffer_targets = dict(internal)
        sim.inventory_replenishment_period = self.replenishment_period
        for key, target in internal.items():
            sim._top_up_inventory_buffer(key, float(target))
        return targets

    def _action_dict(self, shifts: int) -> dict[str, float | int]:
        sim = getattr(self.unwrapped, "sim", None)
        cap = CAPACITY_BY_SHIFTS[int(shifts)]
        op9_min = float(OPERATIONS[9]["q"][0])
        op9_max = float(OPERATIONS[9]["q"][1])
        if sim is not None:
            op9_min = float(sim.params.get("op9_q_min", op9_min))
            op9_max = float(sim.params.get("op9_q_max", op9_max))
        return {
            "assembly_shifts": int(shifts),
            "op3_q": float(cap["op3_q"]),
            "op3_rop": float(OPERATIONS[3]["rop"]),
            "op9_q_min": op9_min,
            "op9_q_max": op9_max,
            "op9_rop": float(OPERATIONS[9]["rop"]),
            "batch_size": float(cap["op7_q"]),
        }

    def _decision_payload(
        self, *, frac: float, shifts: int, targets: dict[str, float], phase: str
    ) -> dict[str, Any]:
        return {
            "action_contract": self.action_contract,
            "action_space_mode": self.action_space_mode,
            "action_phase": phase,
            "continuous_inventory_buffer_fraction": float(np.clip(frac, 0.0, 1.0)),
            "assembly_shift_signal_level": int(shifts),
            "inventory_buffer_targets": dict(targets),
            "continuous_its_frac": float(np.clip(frac, 0.0, 1.0)),
            "continuous_its_shift": int(shifts),
            "continuous_its_buffer_units": (
                float(sum(targets.values())) if targets else 0.0
            ),
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._ewma_rate = 0.0
        obs, info = self.env.reset(seed=seed, options=options)
        frac0 = self.init_frac
        if options and "init_frac" in options:
            frac0 = options["init_frac"]
        info = dict(info)
        if frac0 is not None:
            targets = self._set_targets(float(frac0))  # preposition before warmup
            info.update(
                self._decision_payload(
                    frac=float(frac0), shifts=1, targets=targets, phase="reset"
                )
            )
            info["initial_decision"] = {
                "continuous_inventory_buffer_fraction": float(
                    np.clip(float(frac0), 0.0, 1.0)
                ),
                "inventory_buffer_targets": dict(targets),
                "applied_before_warmup": True,
            }
        else:
            info.update(
                self._decision_payload(frac=0.0, shifts=1, targets={}, phase="reset")
            )
        return self._augment(obs), info

    def step(self, action: Any):
        a = self._validate_action(action)
        frac = float(a[0])
        shifts = self._shift_from_signal(a[1])
        targets = self._set_targets(frac)
        obs, reward, terminated, truncated, info = self.env.step(self._action_dict(shifts))
        buf_frac = float(np.clip(frac, 0.0, 1.0))
        reward = (float(reward)
                  - self.holding_cost * buf_frac
                  - self.shift_cost * (float(shifts - 1) / 2.0))
        info = dict(info)
        info.update(
            self._decision_payload(
                frac=frac, shifts=shifts, targets=targets, phase="weekly_decision"
            )
        )
        info["resource_composite"] = self.resource_composite(buf_frac, shifts)
        return self._augment(obs, update_ewma=True), float(reward), bool(terminated), bool(truncated), info


def make_continuous_its_track_a_env(**overrides: Any) -> ContinuousItsTrackAEnv:
    """Build the continuous I_{t,S} Track-A env (faithful base + continuous buffer wrapper)."""
    init_frac = overrides.pop("init_frac", None)
    risk_obs = bool(overrides.pop("risk_obs", False))
    risk_recent_window = float(overrides.pop("risk_recent_window", 336.0))
    holding_cost = float(overrides.pop("holding_cost", 0.0))
    shift_cost = float(overrides.pop("shift_cost", 0.0))
    replenishment_period = float(overrides.pop("replenishment_period", 168.0))
    overrides.pop("action_space_mode", None)
    overrides.pop("learn_initial_decision", None)  # not supported in this v1 wrapper
    obs_v = overrides.get("observation_version", "v6")
    try:
        from .external_env_interface import get_observation_fields
        base_field_names = list(get_observation_fields(obs_v))
    except Exception:
        base_field_names = []
    step_window = float(overrides.get("step_size_hours", 168.0))
    base = make_thesis_aligned_training_env(**overrides)
    return ContinuousItsTrackAEnv(base, init_frac=init_frac, risk_obs=risk_obs,
                                  risk_recent_window=risk_recent_window, step_window=step_window,
                                  base_field_names=base_field_names,
                                  holding_cost=holding_cost, shift_cost=shift_cost,
                                  replenishment_period=replenishment_period)
