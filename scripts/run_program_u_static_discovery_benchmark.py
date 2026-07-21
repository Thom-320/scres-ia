#!/usr/bin/env python3
"""Burned-data benchmark for simulator-budgeted static-calendar discovery.

Every optimizer sees only the scores of calendars it explicitly proposes.  The
complete 65,536-calendar matrix is constructed only after all candidates have
been frozen and is used solely to calculate exact rank and regret.

This is an optimization benchmark, not evidence of state-feedback adaptation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO  # noqa: E402

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    extract_full_des_skeleton,
    full_action_calendars,
    simulate_full_des_frontier,
)
from supply_chain.program_u_policy_discovery import StaticCalendarDiscoveryEnv  # noqa: E402


TAPES = (94800001, 94800002, 94800003)
HORIZON = 8
ACTIONS = 4


@dataclass
class SearchResult:
    algorithm: str
    seed: int
    budget: int
    calendar: tuple[int, ...]
    score: float
    trace: list[float]
    proposed: int
    elapsed_seconds: float = 0.0
    frozen_calendar: tuple[int, ...] | None = None
    frozen_mean_entropy: float | None = None


class CalendarOracle:
    """Score explicit proposals without exposing unqueried calendar values."""

    def __init__(self, skeletons, sched) -> None:
        self.skeletons = skeletons
        self.scheduler = sched
        self.proposals = 0

    def score(self, calendar: tuple[int, ...]) -> float:
        return float(self.score_many([calendar])[0])

    def score_many(self, calendars: list[tuple[int, ...]]) -> np.ndarray:
        """Vectorized direct-simulator queries; no unproposed row is evaluated."""
        if not calendars:
            return np.empty(0, dtype=float)
        for calendar in calendars:
            if len(calendar) != HORIZON or any(a not in range(ACTIONS) for a in calendar):
                raise ValueError("invalid static calendar")
        self.proposals += len(calendars)
        candidates = np.asarray(calendars, dtype=np.uint8)
        return np.mean(np.column_stack([
            simulate_full_des_frontier(
                skeleton=self.skeletons[tape], scheduler=self.scheduler, calendars=candidates
            )["ret_visible"]
            for tape in TAPES
        ]), axis=1)


def _update(best_calendar, best_score, trace, calendar, score):
    if best_calendar is None or score > best_score + 1e-15 or (
        abs(score - best_score) <= 1e-15 and calendar < best_calendar
    ):
        best_calendar, best_score = calendar, float(score)
    trace.append(float(best_score))
    return best_calendar, best_score


def random_search(oracle: CalendarOracle, *, budget: int, seed: int) -> SearchResult:
    rng = np.random.default_rng(seed)
    best = None; best_score = -np.inf; trace: list[float] = []
    calendars = [tuple(map(int, row)) for row in rng.integers(0, ACTIONS, size=(budget, HORIZON))]
    for calendar, score in zip(calendars, oracle.score_many(calendars)):
        best, best_score = _update(best, best_score, trace, calendar, score)
    assert best is not None
    return SearchResult("random", seed, budget, best, best_score, trace, budget)


def cem_search(oracle: CalendarOracle, *, budget: int, seed: int) -> SearchResult:
    rng = np.random.default_rng(seed)
    probabilities = np.full((HORIZON, ACTIONS), 1.0 / ACTIONS)
    population = min(64, max(16, budget // 4))
    elite_count = max(4, int(np.ceil(0.15 * population)))
    best = None; best_score = -np.inf; trace: list[float] = []; used = 0
    while used < budget:
        size = min(population, budget - used)
        calendars = [tuple(int(rng.choice(ACTIONS, p=probabilities[t])) for t in range(HORIZON)) for _ in range(size)]
        scores = oracle.score_many(calendars)
        for calendar, score in zip(calendars, scores):
            best, best_score = _update(best, best_score, trace, calendar, float(score))
        elite = np.argsort(scores)[-min(elite_count, size):]
        counts = np.full((HORIZON, ACTIONS), 0.5)
        for index in elite:
            for period, action in enumerate(calendars[int(index)]):
                counts[period, action] += 1.0
        update = counts / counts.sum(axis=1, keepdims=True)
        probabilities = 0.35 * probabilities + 0.65 * update
        used += size
    assert best is not None
    return SearchResult("cem", seed, budget, best, best_score, trace, used)


def diagonal_es_search(oracle: CalendarOracle, *, budget: int, seed: int) -> SearchResult:
    """Honest diagonal evolution strategy; not labeled as full CMA-ES."""
    rng = np.random.default_rng(seed)
    mean = np.full(HORIZON, 1.5); sigma = np.full(HORIZON, 1.2)
    population = min(48, max(12, budget // 4)); elite_count = max(3, population // 5)
    best = None; best_score = -np.inf; trace: list[float] = []; used = 0
    while used < budget:
        size = min(population, budget - used)
        latent = rng.normal(mean, sigma, size=(size, HORIZON))
        integer = np.rint(latent).clip(0, ACTIONS - 1).astype(int)
        calendars = [tuple(map(int, row)) for row in integer]
        scores = oracle.score_many(calendars)
        for calendar, score in zip(calendars, scores):
            best, best_score = _update(best, best_score, trace, calendar, float(score))
        elite = latent[np.argsort(scores)[-min(elite_count, size):]]
        mean = 0.3 * mean + 0.7 * elite.mean(axis=0)
        sigma = np.maximum(0.20, 0.3 * sigma + 0.7 * elite.std(axis=0))
        used += size
    assert best is not None
    return SearchResult("diagonal_es", seed, budget, best, best_score, trace, used)


def rf_ucb_search(oracle: CalendarOracle, *, budget: int, seed: int) -> SearchResult:
    """SMAC-style random-forest UCB search over the finite calendar space."""
    rng = np.random.default_rng(seed)
    seen: dict[tuple[int, ...], float] = {}
    best = None; best_score = -np.inf; trace: list[float] = []
    initial = min(32, budget)
    calendars = [tuple(map(int, row)) for row in rng.integers(0, ACTIONS, size=(initial, HORIZON))]
    for calendar, score in zip(calendars, oracle.score_many(calendars)):
        seen[calendar] = float(score)
        best, best_score = _update(best, best_score, trace, calendar, score)
    batch_size = 16
    while len(trace) < budget:
        x = np.asarray(list(seen), dtype=float); y = np.asarray(list(seen.values()))
        model = ExtraTreesRegressor(n_estimators=64, min_samples_leaf=2, max_features=0.75, random_state=seed + len(trace), n_jobs=1).fit(x, y)
        pool = rng.integers(0, ACTIONS, size=(min(4096, 64 * budget), HORIZON))
        pool = np.unique(pool, axis=0)
        pool = np.asarray([row for row in pool if tuple(map(int, row)) not in seen], dtype=float)
        if len(pool) == 0:
            calendars = [tuple(map(int, rng.integers(0, ACTIONS, HORIZON)))]
        else:
            tree_predictions = np.vstack([tree.predict(pool) for tree in model.estimators_])
            acquisition = tree_predictions.mean(axis=0) + 0.75 * tree_predictions.std(axis=0)
            take = min(batch_size, budget - len(trace), len(pool))
            selected = np.argpartition(acquisition, -take)[-take:]
            calendars = [tuple(map(int, pool[index])) for index in selected]
        scores = oracle.score_many(calendars)
        for calendar, score in zip(calendars, scores):
            seen[calendar] = float(score)
            best, best_score = _update(best, best_score, trace, calendar, score)
    assert best is not None
    return SearchResult("rf_ucb", seed, budget, best, best_score, trace, budget)


def ppo_search(
    oracle: CalendarOracle,
    *,
    budget: int,
    seed: int,
    learning_rate: float = 1e-3,
    n_steps: int | None = None,
    n_epochs: int = 10,
    clip_range: float = 0.2,
    ent_coef: float = 0.0,
    gae_lambda: float = 1.0,
    net_arch: tuple[int, ...] = (64, 64),
) -> SearchResult:
    # One terminal episode proposes one calendar and therefore consumes one
    # candidate budget. The tape id is deliberately a dummy and absent from obs.
    trace: list[float] = []; best = None; best_score = -np.inf
    def evaluator(calendar: tuple[int, ...], _tape_id: int):
        nonlocal best, best_score
        score = oracle.score(calendar)
        best, best_score = _update(best, best_score, trace, calendar, score)
        return {"ret_visible": score}
    env = StaticCalendarDiscoveryEnv(evaluator=evaluator, tape_ids=(0,), horizon=HORIZON, action_count=ACTIONS)
    rollout = int(n_steps or min(256, max(32, budget * HORIZON)))
    rollout = min(rollout, budget * HORIZON)
    batch_size = min(64, rollout)
    while rollout % batch_size != 0 and batch_size > 1:
        batch_size -= 1
    model = PPO("MlpPolicy", env, seed=seed, learning_rate=learning_rate, n_steps=rollout,
                batch_size=batch_size, gamma=1.0, gae_lambda=gae_lambda,
                n_epochs=n_epochs, clip_range=clip_range, ent_coef=ent_coef,
                policy_kwargs={"net_arch": list(net_arch)}, verbose=0, device="cpu")
    model.learn(total_timesteps=budget * HORIZON, progress_bar=False)
    # SB3 rounds to complete rollouts. Truncate for equal-budget reporting.
    trace = trace[:budget]
    if not trace:
        raise AssertionError("PPO proposed no calendars")
    assert best is not None

    # Freeze what the trained deterministic policy actually emits.  Do not
    # evaluate its terminal step here: the exact score matrix remains
    # inaccessible until every optimizer has frozen its candidate.
    def forbidden_evaluator(_calendar: tuple[int, ...], _tape_id: int):
        raise AssertionError("frozen-policy extraction must not query the answer key")

    freeze_env = StaticCalendarDiscoveryEnv(
        evaluator=forbidden_evaluator,
        tape_ids=(0,),
        horizon=HORIZON,
        action_count=ACTIONS,
    )
    obs, _ = freeze_env.reset()
    frozen: list[int] = []
    entropies: list[float] = []
    for period in range(HORIZON):
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        distribution = model.policy.get_distribution(obs_tensor)
        entropy = distribution.entropy()
        entropies.append(float(entropy.detach().cpu().numpy().reshape(-1)[0]))
        action, _ = model.predict(obs, deterministic=True)
        value = int(np.asarray(action).reshape(-1)[0])
        frozen.append(value)
        if period < HORIZON - 1:
            obs, _, terminated, truncated, _ = freeze_env.step(value)
            if terminated or truncated:
                raise AssertionError("freeze environment terminated before the calendar was complete")

    return SearchResult(
        "ppo_autoregressive",
        seed,
        budget,
        best,
        float(max(trace)),
        trace,
        len(trace),
        frozen_calendar=tuple(frozen),
        frozen_mean_entropy=float(np.mean(entropies)),
    )


SEARCHERS = {
    "random": random_search,
    "cem": cem_search,
    "diagonal_es": diagonal_es_search,
    "rf_ucb": rf_ucb_search,
    "ppo_autoregressive": ppo_search,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", default="128,512,2048")
    parser.add_argument("--optimizer-seeds", default="20260731,20260732,20260733,20260734,20260735")
    parser.add_argument("--algorithms", default=",".join(SEARCHERS))
    parser.add_argument("--output", type=Path, default=ROOT / "results/program_u/static_discovery_benchmark_v1/result.json")
    args = parser.parse_args()
    budgets = tuple(map(int, args.budgets.split(","))); seeds = tuple(map(int, args.optimizer_seeds.split(",")))
    algorithms = tuple(args.algorithms.split(","))
    if any(name not in SEARCHERS for name in algorithms):
        raise ValueError("unknown algorithm")
    sched = scheduler(); skeletons = {}
    for tape in TAPES:
        skeletons[tape], _ = extract_full_des_skeleton(seed=tape, scheduler=sched, regime_persistence=0.75, dominant_share=0.90, downstream_freight_physics_mode="fixed_clock_physical_v1")
    frozen: list[SearchResult] = []
    for budget in budgets:
        for seed in seeds:
            for name in algorithms:
                oracle = CalendarOracle(skeletons, sched)
                started = time.perf_counter()
                result = SEARCHERS[name](oracle, budget=budget, seed=seed)
                result.elapsed_seconds = float(time.perf_counter() - started)
                frozen.append(result)

    # The answer key becomes accessible only here, after every search result is frozen.
    calendars = full_action_calendars()
    score_matrix = np.column_stack([
        simulate_full_des_frontier(skeleton=skeletons[tape], scheduler=sched, calendars=calendars)["ret_visible"]
        for tape in TAPES
    ])
    mean_scores = score_matrix.mean(axis=1); optimum_index = int(np.argmax(mean_scores)); optimum = float(mean_scores[optimum_index])
    rows = []
    for result in frozen:
        best_seen_index = int(sum(action * (ACTIONS ** (HORIZON - 1 - period)) for period, action in enumerate(result.calendar)))
        best_seen_score = float(mean_scores[best_seen_index])
        rank = int(1 + np.sum(mean_scores > best_seen_score + 1e-15))
        regret = float(optimum - best_seen_score)
        frozen_calendar = result.frozen_calendar or result.calendar
        frozen_index = int(sum(action * (ACTIONS ** (HORIZON - 1 - period)) for period, action in enumerate(frozen_calendar)))
        frozen_score = float(mean_scores[frozen_index])
        frozen_rank = int(1 + np.sum(mean_scores > frozen_score + 1e-15))
        frozen_regret = float(optimum - frozen_score)
        rows.append({
            "algorithm": result.algorithm, "optimizer_seed": result.seed, "candidate_budget": result.budget,
            # Legacy aliases retain compatibility with the first exploratory
            # result while the explicit fields prevent best-seen/final-policy
            # conflation.
            "calendar": list(result.calendar), "mean_ret": best_seen_score, "exact_rank": rank,
            "simple_regret": regret, "exact_optimum_recovered": rank == 1,
            "best_seen_calendar": list(result.calendar),
            "best_seen_mean_ret": best_seen_score,
            "best_seen_exact_rank": rank,
            "best_seen_simple_regret": regret,
            "best_seen_optimum_recovered": rank == 1,
            "frozen_policy_calendar": list(frozen_calendar),
            "frozen_policy_mean_ret": frozen_score,
            "frozen_policy_exact_rank": frozen_rank,
            "frozen_policy_simple_regret": frozen_regret,
            "frozen_policy_optimum_recovered": frozen_rank == 1,
            "frozen_policy_mean_entropy": result.frozen_mean_entropy,
            "best_so_far_auc": float(np.mean(result.trace)), "proposals": result.proposed,
            "elapsed_seconds": result.elapsed_seconds,
        })
    payload = {
        "schema_version": "program_u_static_discovery_benchmark_v1_1",
        "created_at": datetime.now(timezone.utc).isoformat(), "claim_status": "BURNED_DEVELOPMENT_NO_CLAIM",
        "tapes": list(TAPES), "answer_key_read_during_search": False,
        "search_space_size": int(len(calendars)), "optimal_calendar": list(map(int, calendars[optimum_index])),
        "optimal_mean_ret": optimum, "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**{k: v for k, v in payload.items() if k != "rows"}, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
