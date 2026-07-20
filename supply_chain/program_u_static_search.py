"""Budget-matched static-calendar search for Program U0.

Every method receives the same panel objective and can query it only through
``BudgetedPanelObjective``.  The exhaustive 4^8 answer key is intentionally not
part of this module; callers may attach it only after a candidate and trace are
frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Sequence

import numpy as np


Calendar = tuple[int, ...]
PanelEvaluator = Callable[[Calendar, int], float]


class BudgetExhausted(RuntimeError):
    """Raised before a query would exceed the frozen simulator-call budget."""


@dataclass(frozen=True)
class TracePoint:
    calendar_tape_calls: int
    best_score: float
    best_calendar: Calendar


@dataclass(frozen=True)
class SearchResult:
    method: str
    best_calendar: Calendar
    best_score: float
    calendar_tape_calls: int
    unique_calendars: int
    trace: tuple[TracePoint, ...]


class BudgetedPanelObjective:
    """Mean panel score with a hard calendar×tape call budget and caching."""

    def __init__(
        self,
        *,
        evaluator: PanelEvaluator,
        tape_ids: Sequence[int],
        horizon: int = 8,
        action_count: int = 4,
        call_budget: int,
    ) -> None:
        if not tape_ids:
            raise ValueError("at least one tape is required")
        if horizon <= 0 or action_count <= 1:
            raise ValueError("invalid calendar geometry")
        if call_budget < len(tape_ids):
            raise ValueError("budget must permit at least one panel evaluation")
        self.evaluator = evaluator
        self.tape_ids = tuple(map(int, tape_ids))
        self.horizon = int(horizon)
        self.action_count = int(action_count)
        self.call_budget = int(call_budget)
        self.calls = 0
        self.cache: dict[Calendar, float] = {}
        self.trace: list[TracePoint] = []
        self.best_calendar: Calendar | None = None
        self.best_score = -math.inf

    @property
    def panel_cost(self) -> int:
        return len(self.tape_ids)

    @property
    def remaining_calls(self) -> int:
        return self.call_budget - self.calls

    def _validate(self, calendar: Iterable[int]) -> Calendar:
        value = tuple(map(int, calendar))
        if len(value) != self.horizon:
            raise ValueError("calendar has the wrong horizon")
        if any(action < 0 or action >= self.action_count for action in value):
            raise ValueError("calendar contains an invalid action")
        return value

    def evaluate(self, calendar: Iterable[int]) -> float:
        value = self._validate(calendar)
        if value in self.cache:
            return self.cache[value]
        if self.remaining_calls < self.panel_cost:
            raise BudgetExhausted("calendar panel would exceed the frozen budget")
        scores = [float(self.evaluator(value, tape)) for tape in self.tape_ids]
        if not np.all(np.isfinite(scores)):
            raise ValueError("simulator returned a non-finite score")
        score = float(np.mean(scores))
        self.calls += self.panel_cost
        self.cache[value] = score
        if score > self.best_score:
            self.best_score = score
            self.best_calendar = value
        if self.best_calendar is None:
            raise AssertionError("best calendar was not initialized")
        self.trace.append(TracePoint(self.calls, self.best_score, self.best_calendar))
        return score

    def result(self, method: str) -> SearchResult:
        if self.best_calendar is None:
            raise RuntimeError("search made no physical query")
        return SearchResult(
            method=method,
            best_calendar=self.best_calendar,
            best_score=self.best_score,
            calendar_tape_calls=self.calls,
            unique_calendars=len(self.cache),
            trace=tuple(self.trace),
        )


def _random_calendar(objective: BudgetedPanelObjective, rng: np.random.Generator) -> Calendar:
    return tuple(map(int, rng.integers(0, objective.action_count, size=objective.horizon)))


def random_search(objective: BudgetedPanelObjective, *, seed: int) -> SearchResult:
    rng = np.random.default_rng(seed)
    while objective.remaining_calls >= objective.panel_cost:
        objective.evaluate(_random_calendar(objective, rng))
    return objective.result("random_search")


def cross_entropy_search(
    objective: BudgetedPanelObjective,
    *,
    seed: int,
    population: int = 24,
    elite_fraction: float = 0.25,
    smoothing: float = 0.35,
) -> SearchResult:
    rng = np.random.default_rng(seed)
    probs = np.full((objective.horizon, objective.action_count), 1 / objective.action_count)
    elite_n = max(2, int(math.ceil(population * elite_fraction)))
    while objective.remaining_calls >= objective.panel_cost:
        rows: list[tuple[float, Calendar]] = []
        for _ in range(population):
            calendar = tuple(
                int(rng.choice(objective.action_count, p=probs[t]))
                for t in range(objective.horizon)
            )
            try:
                rows.append((objective.evaluate(calendar), calendar))
            except BudgetExhausted:
                break
        if not rows:
            break
        rows.sort(reverse=True)
        elites = [row[1] for row in rows[: min(elite_n, len(rows))]]
        frequencies = np.full_like(probs, 1e-3)
        for calendar in elites:
            frequencies[np.arange(objective.horizon), calendar] += 1.0
        frequencies /= frequencies.sum(axis=1, keepdims=True)
        probs = smoothing * probs + (1.0 - smoothing) * frequencies
    return objective.result("cross_entropy")


def autoregressive_policy_gradient_search(
    objective: BudgetedPanelObjective,
    *,
    seed: int,
    learning_rate: float = 0.12,
    entropy: float = 0.01,
) -> SearchResult:
    """REINFORCE over position-wise categorical logits with a running baseline."""
    rng = np.random.default_rng(seed)
    logits = np.zeros((objective.horizon, objective.action_count), dtype=float)
    baseline = 0.0
    count = 0
    while objective.remaining_calls >= objective.panel_cost:
        shifted = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(shifted)
        probs /= probs.sum(axis=1, keepdims=True)
        calendar = tuple(
            int(rng.choice(objective.action_count, p=probs[t]))
            for t in range(objective.horizon)
        )
        reward = objective.evaluate(calendar)
        count += 1
        baseline += (reward - baseline) / count
        advantage = reward - baseline
        grad = -probs
        grad[np.arange(objective.horizon), calendar] += 1.0
        entropy_grad = -(np.log(np.clip(probs, 1e-12, 1.0)) + 1.0)
        logits += learning_rate * (advantage * grad + entropy * entropy_grad)
    return objective.result("autoregressive_policy_gradient")


def cma_es_search(
    objective: BudgetedPanelObjective,
    *,
    seed: int,
    population: int = 18,
) -> SearchResult:
    """Small full-covariance CMA-ES on a rounded continuous calendar embedding."""
    rng = np.random.default_rng(seed)
    n = objective.horizon
    mean = np.full(n, (objective.action_count - 1) / 2.0)
    covariance = np.eye(n)
    sigma = 1.1
    mu = max(2, population // 2)
    raw_weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = raw_weights / raw_weights.sum()
    while objective.remaining_calls >= objective.panel_cost:
        samples: list[tuple[float, np.ndarray]] = []
        for _ in range(population):
            vector = rng.multivariate_normal(mean, sigma * sigma * covariance)
            calendar = tuple(
                map(int, np.clip(np.rint(vector), 0, objective.action_count - 1))
            )
            try:
                samples.append((objective.evaluate(calendar), vector))
            except BudgetExhausted:
                break
        if not samples:
            break
        samples.sort(key=lambda row: row[0], reverse=True)
        selected = samples[: min(mu, len(samples))]
        local_weights = weights[: len(selected)]
        local_weights /= local_weights.sum()
        old_mean = mean.copy()
        mean = sum(w * row[1] for w, row in zip(local_weights, selected, strict=True))
        centered = [row[1] - old_mean for row in selected]
        update = sum(
            w * np.outer(delta, delta)
            for w, delta in zip(local_weights, centered, strict=True)
        )
        covariance = 0.8 * covariance + 0.2 * update / max(sigma * sigma, 1e-9)
        covariance += np.eye(n) * 1e-6
        sigma *= 0.98
    return objective.result("cma_es_discrete_embedding")


def bayesian_optimization_search(
    objective: BudgetedPanelObjective,
    *,
    seed: int,
    initial_points: int = 12,
    candidate_pool: int = 512,
) -> SearchResult:
    """Gaussian-process expected-improvement search over one-hot calendars."""
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel

    rng = np.random.default_rng(seed)

    def features(calendar: Calendar) -> np.ndarray:
        matrix = np.zeros((objective.horizon, objective.action_count), dtype=float)
        matrix[np.arange(objective.horizon), calendar] = 1.0
        return matrix.ravel()

    while len(objective.cache) < initial_points and objective.remaining_calls >= objective.panel_cost:
        objective.evaluate(_random_calendar(objective, rng))
    while objective.remaining_calls >= objective.panel_cost:
        calendars = list(objective.cache)
        x = np.asarray([features(calendar) for calendar in calendars])
        y = np.asarray([objective.cache[calendar] for calendar in calendars])
        kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-6)
        model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed)
        model.fit(x, y)
        pool: list[Calendar] = []
        seen = set(calendars)
        while len(pool) < candidate_pool:
            candidate = _random_calendar(objective, rng)
            if candidate not in seen:
                seen.add(candidate)
                pool.append(candidate)
        xp = np.asarray([features(calendar) for calendar in pool])
        mean, std = model.predict(xp, return_std=True)
        improvement = mean - float(np.max(y))
        z = np.divide(improvement, std, out=np.zeros_like(improvement), where=std > 0)
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        objective.evaluate(pool[int(np.argmax(ei))])
    return objective.result("bayesian_optimization")


def ppo_calendar_search(
    objective: BudgetedPanelObjective,
    *,
    seed: int,
    learning_rate: float = 3e-4,
    entropy: float = 0.02,
) -> SearchResult:
    """PPO search whose terminal reward is one complete physical tape panel."""
    from stable_baselines3 import PPO

    from supply_chain.program_u_policy_discovery import StaticCalendarDiscoveryEnv

    def panel_evaluator(calendar: Calendar, _dummy_tape: int) -> dict[str, float]:
        return {"ret_visible": objective.evaluate(calendar)}

    env = StaticCalendarDiscoveryEnv(
        evaluator=panel_evaluator,
        tape_ids=(0,),
        horizon=objective.horizon,
        action_count=objective.action_count,
    )
    model = PPO(
        "MlpPolicy",
        env,
        seed=int(seed),
        learning_rate=float(learning_rate),
        n_steps=objective.horizon,
        batch_size=objective.horizon,
        gamma=1.0,
        gae_lambda=1.0,
        ent_coef=float(entropy),
        verbose=0,
        device="cpu",
    )
    # Each learn call proposes exactly one complete calendar. Cached repeats do
    # not consume fictitious DES calls. The cap fails closed if PPO collapses
    # to duplicate proposals: underspending is allowed; overspending is not.
    duplicate_cap = 50 * max(1, objective.call_budget // objective.panel_cost)
    episodes = 0
    while objective.remaining_calls >= objective.panel_cost and episodes < duplicate_cap:
        model.learn(
            total_timesteps=objective.horizon,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        episodes += 1
    return objective.result("ppo_static_calendar")
