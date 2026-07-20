"""Vectorized burned-data T0 planner for the Program O/Q full-DES contract.

The planner sees only ``StateRichObservation`` fields.  Its particle stream is
keyed by the observation digest, never by tape/seed or realized future demand.
Candidate policies are scored in a compact, resource-conserving planning model;
the selected calendar is always evaluated by the certified full-DES transducer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from itertools import product
import time
from typing import Mapping, Sequence

import numpy as np

from supply_chain.program_o_state_rich import (
    StateRichConfiguration,
    StateRichObservation,
    state_rich_calendar,
)


@dataclass(frozen=True)
class FullDEST0Config:
    horizon: int
    mode: str
    particles: int = 32
    regime_persistence: float = 0.75
    dominant_share: float = 0.90
    worst_product_floor: float = 0.70

    def __post_init__(self) -> None:
        if self.horizon not in (1, 3, 4, 6, 8):
            raise ValueError("horizon must be one of 1,3,4,6,8")
        if self.mode not in {"nominal", "scenario", "robust", "constraint_aware"}:
            raise ValueError("unknown T0 mode")
        if self.particles <= 0:
            raise ValueError("particles must be positive")

    @property
    def config_id(self) -> str:
        return f"ret_proxy_{self.mode}_h{self.horizon}_p{self.particles}"


@lru_cache(maxsize=None)
def _sequences(horizon: int) -> np.ndarray:
    return np.asarray(tuple(product(range(4), repeat=int(horizon))), dtype=np.uint8)


def _scheduler_counts(scheduler: Mapping[str, Sequence[str]]) -> np.ndarray:
    rows = []
    for action in range(4):
        labels = tuple(scheduler[str(action)])
        rows.append((labels.count("P_C"), labels.count("P_H")))
    out = np.asarray(rows, dtype=float)
    if out.shape != (4, 2) or not np.all(out.sum(axis=1) == 3):
        raise ValueError("scheduler must allocate exactly three weekly batches")
    return out


def _particle_demand(observation: StateRichObservation, config: FullDEST0Config) -> np.ndarray:
    horizon = min(config.horizon, observation.remaining_decisions)
    if config.mode == "nominal":
        belief = float(observation.belief_c)
        rows = []
        for _ in range(horizon):
            share_c = belief * config.dominant_share + (1.0 - belief) * (1.0 - config.dominant_share)
            rows.append((15000.0 * share_c, 15000.0 * (1.0 - share_c)))
            belief = config.regime_persistence * belief + (1.0 - config.regime_persistence) * (1.0 - belief)
        return np.asarray([rows], dtype=float)
    seed = int.from_bytes(
        hashlib.sha256((observation.observation_sha256 + "::T0_PARTICLES_V1").encode()).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(seed)
    demand = np.empty((config.particles, horizon, 2), dtype=float)
    regime_c = rng.random(config.particles) < float(observation.belief_c)
    for step in range(horizon):
        probability_c = np.where(regime_c, config.dominant_share, 1.0 - config.dominant_share)
        count_c = rng.binomial(6, probability_c)
        demand[:, step, 0] = 2500.0 * count_c
        demand[:, step, 1] = 2500.0 * (6 - count_c)
        keep = rng.random(config.particles) < config.regime_persistence
        regime_c = np.where(keep, regime_c, ~regime_c)
    return demand


def choose_t0_action(
    observation: StateRichObservation,
    *,
    scheduler: Mapping[str, Sequence[str]],
    config: FullDEST0Config,
    chunk_size: int = 4096,
) -> tuple[int, dict[str, float]]:
    horizon = min(config.horizon, observation.remaining_decisions)
    sequences = _sequences(horizon)
    counts = _scheduler_counts(scheduler)
    demand = _particle_demand(observation, config)
    initial = (
        np.asarray(observation.on_hand, dtype=float)
        + np.asarray(observation.locked_pipeline, dtype=float)
        - np.asarray(observation.backlog_quantity, dtype=float)
    )
    best_key: tuple[float, ...] | None = None
    best_action = 0
    evaluated = 0
    for start in range(0, len(sequences), chunk_size):
        block = sequences[start : start + chunk_size]
        net = np.broadcast_to(initial, (len(block), len(demand), 2)).copy()
        backlog_area = np.zeros((len(block), len(demand)), dtype=float)
        total_demand = np.zeros((len(demand), 2), dtype=float)
        for step in range(horizon):
            net += 5000.0 * counts[block[:, step]][:, None, :]
            net -= demand[None, :, step, :]
            backlog_area += np.maximum(0.0, -net).sum(axis=2)
            total_demand += demand[:, step, :]
        shortage = np.maximum(0.0, -net)
        scale = np.maximum(total_demand.sum(axis=1), 1.0)
        ret_proxy = 1.0 - (backlog_area + shortage.sum(axis=2)) / (scale[None, :] * (horizon + 1.0))
        product_fill = 1.0 - shortage / np.maximum(total_demand[None, :, :], 1.0)
        worst_fill = np.min(product_fill, axis=2)
        mean_ret = ret_proxy.mean(axis=1)
        tail_count = max(1, int(np.ceil(0.10 * ret_proxy.shape[1])))
        tail_ret = np.partition(ret_proxy, tail_count - 1, axis=1)[:, :tail_count].mean(axis=1)
        min_fill = worst_fill.min(axis=1)
        if config.mode == "nominal" or config.mode == "scenario":
            primary = mean_ret
            feasible = np.ones(len(block), dtype=float)
        elif config.mode == "robust":
            primary = tail_ret
            feasible = np.ones(len(block), dtype=float)
        else:
            primary = mean_ret
            feasible = (min_fill >= config.worst_product_floor).astype(float)
        for offset in range(len(block)):
            sequence = tuple(map(int, block[offset]))
            key = (
                float(feasible[offset]),
                float(primary[offset]),
                float(tail_ret[offset]),
                float(min_fill[offset]),
                *tuple(-value for value in sequence),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_action = sequence[0]
        evaluated += len(block)
    assert best_key is not None
    return int(best_action), {
        "candidate_sequences": float(evaluated),
        "planning_objective": float(best_key[1]),
        "planning_tail": float(best_key[2]),
        "planning_worst_fill": float(best_key[3]),
        "planning_feasible": float(best_key[0]),
    }


def t0_calendar(
    *,
    skeleton: Mapping[str, object],
    scheduler: Mapping[str, Sequence[str]],
    config: FullDEST0Config,
) -> tuple[tuple[int, ...], dict[str, object]]:
    weeks = int(skeleton["decision_weeks"])
    prefix: list[int] = []
    decisions = []
    started = time.perf_counter()
    for week in range(weeks):
        probe = tuple(prefix + [0] * (weeks - len(prefix)))
        _calendar, rows = state_rich_calendar(
            skeleton=skeleton,
            scheduler=scheduler,
            config=StateRichConfiguration("belief_mpc", 1),
            regime_persistence=config.regime_persistence,
            dominant_share=config.dominant_share,
            action_overrides=probe,
        )
        observation = rows[week].observation
        action, diagnostics = choose_t0_action(observation, scheduler=scheduler, config=config)
        prefix.append(action)
        decisions.append({"week": week, "action": action, **diagnostics})
    return tuple(prefix), {
        "config_id": config.config_id,
        "online_ms": (time.perf_counter() - started) * 1000.0,
        "decisions": decisions,
    }


def t0_grid(*, particles: int = 32) -> tuple[FullDEST0Config, ...]:
    return tuple(
        FullDEST0Config(horizon=h, mode=mode, particles=(1 if mode == "nominal" else particles))
        for h in (1, 3, 4, 6, 8)
        for mode in ("nominal", "scenario", "robust", "constraint_aware")
    )
