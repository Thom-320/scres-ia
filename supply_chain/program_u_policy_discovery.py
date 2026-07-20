"""Program U policy-search environments.

The two environments in this module deliberately separate two questions that
are often conflated:

``StaticCalendarDiscoveryEnv`` asks whether a learning algorithm can discover a
single open-loop calendar using a limited number of simulator calls.  The
physical tape is never exposed to the learner and the DES is evaluated only
after the complete calendar has been proposed.

``EndogenousReviewProgramORetEnv`` asks whether a state-feedback learner can
allocate a finite number of managerial review rights over an episode.  At each
review it selects both the product-mix action and the number of weeks until the
next review.  This is a semi-Markov decision contract; it is not a claim that
the underlying DES can be interrupted at arbitrary SimPy events.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from supply_chain.program_o_ret_env import ProgramORetOnlyEnv


StaticEvaluator = Callable[[tuple[int, ...], int], Mapping[str, float]]


class StaticCalendarDiscoveryEnv(gym.Env[np.ndarray, int]):
    """Construct one tape-independent calendar and score it once.

    The construction steps are an algorithmic device, not physical recourse.
    The observation contains only the calendar prefix and construction phase;
    it contains no tape, demand, risk, inventory, or DES state.  Consequently a
    trained deterministic policy represents one open-loop calendar.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        evaluator: StaticEvaluator,
        tape_ids: Sequence[int],
        horizon: int = 8,
        action_count: int = 4,
        primary_metric: str = "ret_visible",
    ) -> None:
        super().__init__()
        if horizon <= 0 or action_count <= 1:
            raise ValueError("horizon must be positive and action_count > 1")
        if not tape_ids:
            raise ValueError("at least one development tape is required")
        self.evaluator = evaluator
        self.tape_ids = tuple(map(int, tape_ids))
        self.horizon = int(horizon)
        self.action_count = int(action_count)
        self.primary_metric = str(primary_metric)
        self.action_space = spaces.Discrete(self.action_count)
        # phase + one-hot prefix.  Unchosen positions remain all zero.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1 + self.horizon * self.action_count,),
            dtype=np.float32,
        )
        self._episode = 0
        self._calendar: list[int] = []
        self._tape_id = self.tape_ids[0]

    def _observation(self) -> np.ndarray:
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[0] = len(self._calendar) / self.horizon
        for period, action in enumerate(self._calendar):
            obs[1 + period * self.action_count + int(action)] = 1.0
        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        self._calendar = []
        self._tape_id = int(
            options.get("tape_id", self.tape_ids[self._episode % len(self.tape_ids)])
        )
        if self._tape_id not in self.tape_ids:
            raise ValueError("tape_id is outside the declared development set")
        self._episode += 1
        # tape_id is audit metadata only and is intentionally absent from obs.
        return self._observation(), {"physical_evaluations": 0}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        value = int(action)
        if not self.action_space.contains(value):
            raise ValueError("calendar action outside the declared alphabet")
        if len(self._calendar) >= self.horizon:
            raise RuntimeError("step() called after terminal calendar")
        self._calendar.append(value)
        terminated = len(self._calendar) == self.horizon
        if not terminated:
            return self._observation(), 0.0, False, False, {"physical_evaluations": 0}

        metrics = {k: float(v) for k, v in self.evaluator(tuple(self._calendar), self._tape_id).items()}
        if self.primary_metric not in metrics:
            raise KeyError(f"evaluator omitted primary metric {self.primary_metric!r}")
        info = {
            "calendar": list(self._calendar),
            "metrics": metrics,
            "physical_evaluations": 1,
            "policy_class": "single_open_loop_calendar",
        }
        return self._observation(), metrics[self.primary_metric], True, False, info


class EndogenousReviewProgramORetEnv(ProgramORetOnlyEnv):
    """Program O-R with a finite budget of endogenously timed reviews.

    Action ``a = mix * len(dwell_options) + dwell_index`` selects product mix
    ``mix in {0,1,2,3}`` and the number of weekly commitments before the next
    observation.  A finite review budget prevents the shortest dwell from
    weakly dominating every other choice.  Once the budget is exhausted, the
    selected mix is held for the remainder of the episode.
    """

    def __init__(
        self,
        *,
        dwell_options: Sequence[int] = (1, 2, 4),
        review_budget: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        dwell = tuple(sorted(set(map(int, dwell_options))))
        if not dwell or dwell[0] <= 0:
            raise ValueError("dwell options must be positive integers")
        if review_budget <= 0:
            raise ValueError("review_budget must be positive")
        self.dwell_options = dwell
        self.review_budget = int(review_budget)
        self.action_space = spaces.Discrete(4 * len(self.dwell_options))
        self._review_count = 0
        self._review_trajectory: list[dict[str, int]] = []

    def reset(self, **kwargs: Any):
        obs, info = super().reset(**kwargs)
        self._review_count = 0
        self._review_trajectory = []
        info = dict(info)
        info.update(review_budget=self.review_budget, reviews_used=0)
        return obs, info

    def decode_action(self, action: int) -> tuple[int, int]:
        value = int(action)
        if not self.action_space.contains(value):
            raise ValueError("endogenous-review action outside action space")
        width = len(self.dwell_options)
        return value // width, self.dwell_options[value % width]

    def step(self, action: int):
        mix, requested_dwell = self.decode_action(action)
        if self._review_count >= self.review_budget:
            raise RuntimeError("review budget exhausted after a non-terminal transition")
        self._review_count += 1
        remaining = int(self._skeleton.decision_weeks) - len(self._actions)  # type: ignore[union-attr]
        if self._review_count == self.review_budget:
            dwell = remaining
        else:
            dwell = min(requested_dwell, remaining)
        start_week = len(self._actions)
        final = None
        for _ in range(dwell):
            final = super().step(mix)
            if final[2] or final[3]:
                break
        if final is None:
            raise AssertionError("endogenous review made no physical commitment")
        row = {
            "review": self._review_count,
            "start_week": start_week,
            "mix": mix,
            "requested_dwell": requested_dwell,
            "executed_dwell": len(self._actions) - start_week,
        }
        self._review_trajectory.append(row)
        obs, reward, terminated, truncated, info = final
        info = dict(info)
        info.update(
            reviews_used=self._review_count,
            review_budget=self.review_budget,
            review_trajectory=list(self._review_trajectory),
            realized_calendar=list(self._actions),
        )
        return obs, reward, terminated, truncated, info

