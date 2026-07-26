"""Matched-rights meta-episode environment for the v2 retained learning curve.

Contract: contracts/oracle_retained_learning_curve_v2.json (frozen 2026-07-26).

The 2026-07-24 pilot compared a learner that could not carry knowledge across campaigns
against a model-predictive controller explicitly initialized with a carried posterior. This
module removes that asymmetry so the comparison isolates architecture rather than
information rights.

Parity construction
-------------------
`retained_prior_path` derives the carried prior of each campaign from the realized order
products of the PRECEDING campaigns only. It is therefore policy-independent, which is what
makes exact parity possible: the identical scalar can be handed to the model-predictive
controller (as the prior of its belief) and to the learner (as an observation feature),
without either arm receiving information the other lacks.

A meta-episode is one history of `campaigns` consecutive campaigns. Physical state is rebuilt
per campaign, so only knowledge crosses the boundary. Two arms are defined:

    retained : observation carries prior_i, recurrent state persists across campaigns
    reset    : observation carries 0.5,     recurrent state is cleared at each campaign

They are the SAME environment class with the SAME observation space; `assert_arm_parity`
checks that two instances differ in nothing but the retention flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from supply_chain.program_o_ret_env import OBSERVATION_DIM, ProgramORetOnlyEnv
from supply_chain.retained_context_discovery import (
    build_campaign_history,
    retained_prior_path,
)

META_OBSERVATION_DIM = OBSERVATION_DIM + 1  # + the carried prior
UNINFORMED_PRIOR = 0.5
REGIME_PERSISTENCE = 0.90
DOMINANT_SHARE = 0.90


@dataclass(frozen=True)
class HistorySpec:
    """One meta-episode: a root, a persistence cell, and its campaign count."""

    history_root: int
    kappa: float
    campaigns: int = 12

    @property
    def cell_index(self) -> int:
        return 0 if self.kappa < 0.85 else 2


def load_history(spec: HistorySpec, scheduler: Mapping[str, Sequence[str]]):
    """Campaigns plus their policy-independent carried priors."""
    campaigns = build_campaign_history(
        history_root=spec.history_root, campaigns=spec.campaigns, kappa=spec.kappa,
        scheduler=scheduler, regime_persistence=REGIME_PERSISTENCE,
        dominant_share=DOMINANT_SHARE,
    )
    priors = retained_prior_path(
        campaigns, regime_persistence=REGIME_PERSISTENCE, dominant_share=DOMINANT_SHARE)
    if len(priors) != len(campaigns):
        raise AssertionError("prior path length must match the campaign count")
    return campaigns, priors


class MetaCampaignEnv(gym.Env):
    """Consecutive campaigns of one history; retention is the ONLY arm difference."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        scheduler: Mapping[str, Sequence[str]],
        histories: Sequence[HistorySpec],
        retained: bool,
        objective_fn,
        rng_seed: int = 0,
    ) -> None:
        super().__init__()
        if not histories:
            raise ValueError("at least one history is required")
        self.scheduler = {str(k): tuple(v) for k, v in scheduler.items()}
        self.histories = tuple(histories)
        self.retained = bool(retained)
        self._objective_fn = objective_fn
        self._rng = np.random.default_rng(int(rng_seed))
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(META_OBSERVATION_DIM,), dtype=np.float32)
        self._inner = ProgramORetOnlyEnv(
            scheduler=self.scheduler, tape_seed_start=1, tape_seed_end=1)
        self._cache: dict[HistorySpec, tuple] = {}
        self._history: HistorySpec | None = None
        self._campaigns: tuple = ()
        self._priors: tuple = ()
        self._index = 0
        self.last_calendar: list[int] = []
        self.last_objective: float | None = None

    # -- arm identity -----------------------------------------------------
    def configuration(self) -> dict[str, Any]:
        """Everything that defines this env EXCEPT the retention flag."""
        return {
            "class": type(self).__name__,
            "histories": [(h.history_root, h.kappa, h.campaigns) for h in self.histories],
            "observation_dim": int(self.observation_space.shape[0]),
            "action_n": int(self.action_space.n),
            "regime_persistence": REGIME_PERSISTENCE,
            "dominant_share": DOMINANT_SHARE,
            "objective_fn": getattr(self._objective_fn, "__name__", "?"),
        }

    # -- episode plumbing -------------------------------------------------
    def _load(self, spec: HistorySpec):
        if spec not in self._cache:
            self._cache[spec] = load_history(spec, self.scheduler)
        return self._cache[spec]

    def _prior(self) -> float:
        return float(self._priors[self._index]) if self.retained else UNINFORMED_PRIOR

    def _observe(self) -> np.ndarray:
        inner, _ = self._inner._current_observation()
        return np.clip(np.append(inner, np.float32(self._prior())), 0.0, 1.0).astype(
            np.float32)

    def _start_campaign(self) -> np.ndarray:
        campaign = self._campaigns[self._index]
        self._inner.reset(options={"skeleton": campaign.skeleton,
                                   "cell_index": self._history.cell_index,
                                   "tape_seed": self._inner.tape_seed_start})
        self.last_calendar = []
        return self._observe()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        spec = options.get("history")
        if spec is None:
            spec = self.histories[int(self._rng.integers(len(self.histories)))]
        self._history = spec
        self._campaigns, self._priors = self._load(spec)
        self._index = 0
        return self._start_campaign(), {"history_root": spec.history_root,
                                        "campaign_index": 0}

    def step(self, action: int):
        obs, _reward, terminated, _trunc, info = self._inner.step(int(action))
        self.last_calendar.append(int(action))
        if not terminated:
            return self._observe(), 0.0, False, False, {}

        calendar = list(info["calendar"])
        objective = float(self._objective_fn(self._campaigns[self._index].skeleton,
                                             calendar))
        self.last_objective = objective
        payload = {
            "campaign_index": self._campaigns[self._index].campaign_index,
            "history_root": self._history.history_root,
            "calendar": calendar,
            "objective": objective,
            "carried_prior": self._prior(),
            "campaign_boundary": True,
        }
        self._index += 1
        if self._index >= len(self._campaigns):
            return (np.zeros(META_OBSERVATION_DIM, dtype=np.float32), objective, True,
                    False, payload)
        return self._start_campaign(), objective, False, False, payload


def assert_arm_parity(retained_env: MetaCampaignEnv, reset_env: MetaCampaignEnv) -> None:
    """Fail closed unless the two arms differ in nothing but the retention flag."""
    if retained_env.retained == reset_env.retained:
        raise AssertionError("parity check needs one retained and one reset arm")
    a, b = retained_env.configuration(), reset_env.configuration()
    if a != b:
        diff = {k: (a[k], b[k]) for k in a if a.get(k) != b.get(k)}
        raise AssertionError(f"arms differ beyond the retention flag: {diff}")
    if retained_env.observation_space != reset_env.observation_space:
        raise AssertionError("observation spaces differ between arms")
    if retained_env.action_space.n != reset_env.action_space.n:
        raise AssertionError("action spaces differ between arms")
