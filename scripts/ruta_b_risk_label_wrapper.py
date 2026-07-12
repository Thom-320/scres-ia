"""Ruta B: online future-risk label wrapper for joint auxiliary-loss training.

Ruta A (pretrain a belief encoder, transplant the trunk into PPO before RL
fine-tuning) demonstrated a specific failure mode this session
(`docs/TRACK_B_PREVENTIVE_LEARNING_ROOT_CAUSE_AND_ALTERNATIVES_2026-07-04.md`):
the pretrained representation gets *repurposed* during PPO fine-tuning into a
generic risk-averse posture change (near-elimination of shift S1) rather than
staying "prediction-shaped". Nothing during RL training forces the trunk to
keep predicting.

Ruta B keeps the auxiliary prediction task alive through the entire training
process: an auxiliary BCE loss on "does a target risk (R22/R24) start within
the next K weeks" is added to the PPO loss at every gradient step, not just at
initialization.

This wrapper supplies the label. Because ``strict_exogenous_crn=True`` makes
the R22/R24 event calendar action-independent for a fixed seed (verified
repeatedly this session, e.g. the oracle prevention-ceiling gate), the true
future-risk label for a training episode can be obtained with a throwaway
"discovery" rollout under the same seed before the real training rollout
begins -- exactly the technique already used in
``scripts/audit_track_b_oracle_prevention_ceiling.py``. The label is appended
as one extra observation dimension; the paired feature extractor
(``scripts/ruta_b_aux_extractor.py``) splits it back off before the policy
sees the real observation, using it only as a training target for its
auxiliary head.

This doubles per-episode DES simulation cost (one discovery pass + one real
pass) -- an accepted, understood tradeoff, not a bug.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RutaBRiskLabelWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        target_risks: tuple[str, ...] = ("R22", "R24"),
        lead_weeks: int = 2,
        step_size_hours: float = 168.0,
        label_mode: str = "true",
    ) -> None:
        super().__init__(env)
        if label_mode not in ("true", "permuted", "constant"):
            raise ValueError(f"Unknown label_mode: {label_mode!r}")
        self._target_risks = set(target_risks)
        self._lead_weeks = int(lead_weeks)
        self._step_size_hours = float(step_size_hours)
        self._label_mode = label_mode
        self._future_label_by_step: dict[int, float] = {}
        self._step_idx = 0
        self._max_steps = int(getattr(env.unwrapped, "max_steps", 104))

        base_space = env.observation_space
        if len(base_space.shape) != 1:
            raise ValueError("RutaBRiskLabelWrapper expects a flat vector observation.")
        low = np.concatenate([base_space.low, np.array([0.0], dtype=np.float32)])
        high = np.concatenate([base_space.high, np.array([1.0], dtype=np.float32)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _discover_calendar(self, seed: int) -> None:
        """Throwaway pass: run to completion with a neutral action, capture
        the real risk-event calendar for this seed. Action-independent for
        every risk except R14 (not a target here)."""
        discovery_env = self.env
        obs, _info = discovery_env.reset(seed=seed)
        neutral_action = np.zeros(discovery_env.action_space.shape, dtype=np.float32)
        terminated = truncated = False
        while not (terminated or truncated):
            obs, _reward, terminated, truncated, _info = discovery_env.step(neutral_action)

        events = list(discovery_env.unwrapped.sim.risk_events)
        target_steps: list[int] = []
        for ev in events:
            if str(ev.risk_id) not in self._target_risks:
                continue
            start_step = int(float(ev.start_time) // self._step_size_hours)
            target_steps.append(start_step)

        label_by_step: dict[int, float] = {}
        for step in range(self._max_steps):
            is_upcoming = any(
                step < target_step <= step + self._lead_weeks for target_step in target_steps
            )
            label_by_step[step] = 1.0 if is_upcoming else 0.0
        self._future_label_by_step = label_by_step

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        label = self._future_label_by_step.get(self._step_idx, 0.0)
        return np.concatenate([np.asarray(obs, dtype=np.float32), np.array([label], dtype=np.float32)])

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._discover_calendar(seed)
            if self._label_mode == "permuted":
                # Negative control: destroy the temporal alignment of the true
                # label sequence while preserving its exact base rate. If the
                # preventive counterfactual signal survives this, it was never
                # about predictive content -- only about the extra head acting
                # as a regularizer.
                rng = np.random.default_rng(seed)
                steps = sorted(self._future_label_by_step)
                values = np.array([self._future_label_by_step[s] for s in steps])
                rng.shuffle(values)
                self._future_label_by_step = {s: float(v) for s, v in zip(steps, values)}
            elif self._label_mode == "constant":
                # Efficiency-ladder control: every step gets the episode base
                # rate. The BCE gradient becomes trivial (no per-step signal),
                # isolating "any aux gradient at all" from temporal content.
                steps = sorted(self._future_label_by_step)
                base_rate = (
                    float(np.mean([self._future_label_by_step[s] for s in steps])) if steps else 0.0
                )
                self._future_label_by_step = {s: base_rate for s in steps}
        else:
            self._future_label_by_step = {step: 0.0 for step in range(self._max_steps)}
        obs, info = self.env.reset(seed=seed, options=options)
        self._step_idx = 0
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_idx += 1
        return self._augment(obs), reward, terminated, truncated, info


class ConstantLabelPadWrapper(gym.Wrapper):
    """Cheap eval-time counterpart to ``RutaBRiskLabelWrapper``.

    Appends a constant 0.0 instead of the true future-risk label. Correctness
    note: the paired feature extractor only ever reads the real observation
    columns (``obs[:, :-1]``) to compute the actor/critic features -- the
    label column feeds the auxiliary head only, which is a training-time
    concept with no effect on ``predict()``/action selection. So the constant
    pad is behaviorally exact for evaluation, and skips the (expensive)
    discovery pass entirely.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        base_space = env.observation_space
        low = np.concatenate([base_space.low, np.array([0.0], dtype=np.float32)])
        high = np.concatenate([base_space.high, np.array([1.0], dtype=np.float32)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(obs, dtype=np.float32), np.array([0.0], dtype=np.float32)])

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs), reward, terminated, truncated, info
