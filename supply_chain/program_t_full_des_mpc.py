"""Vectorized burned-data T0 planner for the Program O/Q full-DES contract.

The planner sees only ``StateRichObservation`` fields.  Its particle stream is
keyed by the observation digest, never by tape/seed or realized future demand.
Candidate policies are scored in a compact, resource-conserving planning model;
the selected calendar is always evaluated by the certified full-DES transducer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
from itertools import product
import math
import time
from typing import Mapping, Sequence

import numpy as np

from supply_chain.program_o_full_des_transducer import (
    FullDESSkeleton,
    simulate_full_des_frontier,
)
from supply_chain.program_o_state_rich import (
    StateRichConfiguration,
    StateRichObservation,
    state_rich_calendar,
)
from supply_chain.program_t_joint_belief import THETA_GRID, ExactJointBelief, weekly_product_counts


@dataclass(frozen=True)
class FullDEST0Config:
    horizon: int
    mode: str
    particles: int = 32
    regime_persistence: float = 0.75
    dominant_share: float = 0.90
    worst_product_floor: float = 0.70
    belief_integration: str = "mc"

    def __post_init__(self) -> None:
        if self.horizon not in (1, 3, 4, 6, 8):
            raise ValueError("horizon must be one of 1,3,4,6,8")
        if self.mode not in {"nominal", "scenario", "robust", "constraint_aware"}:
            raise ValueError("unknown T0 mode")
        if self.particles <= 0:
            raise ValueError("particles must be positive")
        if self.belief_integration not in {"mc", "stratified"}:
            raise ValueError("belief_integration must be mc or stratified")

    @property
    def config_id(self) -> str:
        suffix = "" if self.belief_integration == "mc" else "_stratified"
        return f"ret_proxy_{self.mode}_h{self.horizon}_p{self.particles}{suffix}"


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
) -> tuple[int, dict[str, object]]:
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


def _synthetic_skeletons(
    base: FullDESSkeleton,
    observation: StateRichObservation,
    config: FullDEST0Config,
) -> tuple[FullDESSkeleton, ...]:
    """Replace every unobserved demand row with a belief-generated row."""
    count = 1 if config.mode == "nominal" else config.particles
    seed = int.from_bytes(
        hashlib.sha256((observation.observation_sha256 + "::T0_RET_SCENARIOS_V1").encode()).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(seed)
    past = [i for i, t in enumerate(base.order_times) if t < observation.decision_time - 1e-12]
    future_weeks = range(observation.week, base.decision_weeks)
    out = []
    for scenario_index in range(count):
        times = [base.order_times[i] for i in past]
        quantities = [base.order_quantities[i] for i in past]
        products = [base.order_products[i] for i in past]
        contingent = [False for _ in past]
        regime_c = (
            observation.belief_c >= 0.5
            if config.mode == "nominal"
            else bool(rng.random() < observation.belief_c)
        )
        for week in future_weeks:
            if config.mode == "nominal":
                share_c = (
                    config.dominant_share if regime_c else 1.0 - config.dominant_share
                )
                count_c = int(round(6 * share_c))
                labels = ["P_C"] * count_c + ["P_H"] * (6 - count_c)
                quantities_week = [2500.0] * 6
            else:
                probability_c = config.dominant_share if regime_c else 1.0 - config.dominant_share
                labels = ["P_C" if rng.random() < probability_c else "P_H" for _ in range(6)]
                quantities_week = list(map(float, rng.integers(2400, 2601, size=6)))
            for offset, product_id, quantity in zip((30, 54, 78, 102, 126, 150), labels, quantities_week):
                times.append(base.decision_start + 168.0 * week + float(offset))
                quantities.append(quantity)
                products.append(product_id)
                contingent.append(False)
            if config.mode == "nominal":
                belief_next = config.regime_persistence * float(regime_c) + (1.0 - config.regime_persistence) * float(not regime_c)
                regime_c = belief_next >= 0.5
            elif rng.random() > config.regime_persistence:
                regime_c = not regime_c
        order = np.argsort(np.asarray(times), kind="stable")
        out.append(
            replace(
                base,
                seed=-(scenario_index + 1),
                order_times=tuple(float(times[i]) for i in order),
                order_quantities=tuple(float(quantities[i]) for i in order),
                order_products=tuple(str(products[i]) for i in order),
                order_contingent=tuple(bool(contingent[i]) for i in order),
                tape_sha256="belief_generated_no_realized_future",
                skeleton_sha256=hashlib.sha256(
                    f"{observation.observation_sha256}:{scenario_index}".encode()
                ).hexdigest(),
            )
        )
    return tuple(out)


def _joint_belief_skeletons(
    base: FullDESSkeleton,
    observation: StateRichObservation,
    config: FullDEST0Config,
    belief: ExactJointBelief,
) -> tuple[tuple[FullDESSkeleton, ...], np.ndarray]:
    """Generate weighted scenarios from the complete theta/regime mixture.

    The original ``mc`` path remains available for historical reproduction.
    ``stratified`` represents every positive joint state when the particle
    budget permits and assigns its conditional paths exactly that state's
    posterior mass. Demand paths remain conditional samples, so p16/p64
    convergence remains an explicit gate rather than a hidden exactness claim.
    """
    seed = int.from_bytes(
        hashlib.sha256((observation.observation_sha256 + "::T0_JOINT_BELIEF_V1").encode()).digest()[:8],
        "big",
    )
    count = 1 if config.mode == "nominal" else config.particles
    if config.belief_integration == "mc":
        sampled = tuple(
            (*state, 1.0 / count, -1, count)
            for state in belief.sample_states(count=count, seed=seed)
        )
    else:
        joint = belief.probability.reshape(-1)
        positive = np.flatnonzero(joint > 0.0)
        if count < len(positive):
            positive = positive[np.argsort(joint[positive])[::-1][:count]]
        allocations = {int(index): 1 for index in positive}
        remaining = count - len(allocations)
        if remaining > 0:
            expected = joint[positive] / joint[positive].sum() * remaining
            floors = np.floor(expected).astype(int)
            for index, extra in zip(positive, floors):
                allocations[int(index)] += int(extra)
            remainder = remaining - int(floors.sum())
            order_by_fraction = np.argsort(expected - floors)[::-1]
            for offset in order_by_fraction[:remainder]:
                allocations[int(positive[offset])] += 1
        represented_mass = float(sum(joint[index] for index in allocations))
        sampled_rows = []
        for flat_index in sorted(allocations):
            theta_index, regime_index = divmod(flat_index, 2)
            rho, share = THETA_GRID[theta_index]
            weight = (
                float(joint[flat_index] / represented_mass)
                / allocations[flat_index]
            )
            sampled_rows.extend(
                (
                    rho,
                    share,
                    bool(regime_index),
                    weight,
                    conditional_index,
                    allocations[flat_index],
                )
                for conditional_index in range(allocations[flat_index])
            )
        sampled = tuple(sampled_rows)
    past = [i for i, t in enumerate(base.order_times) if t < observation.decision_time - 1e-12]
    future_weeks = range(observation.week, base.decision_weeks)
    out = []
    weights = []
    for scenario_index, (
        rho,
        share,
        initial_regime_c,
        scenario_weight,
        conditional_index,
        conditional_count,
    ) in enumerate(sampled):
        scenario_seed = seed ^ ((scenario_index + 1) * 0x9E3779B97F4A7C15)
        rng = np.random.default_rng(scenario_seed & ((1 << 64) - 1))
        times = [base.order_times[i] for i in past]
        quantities = [base.order_quantities[i] for i in past]
        products = [base.order_products[i] for i in past]
        contingent = [False for _ in past]
        regime_c = bool(initial_regime_c)
        for week in future_weeks:
            probability_c = share if regime_c else 1.0 - share
            if config.belief_integration == "stratified":
                # Cranley-Patterson rotations of an evenly spaced conditional
                # grid provide deterministic, nested low-variance coverage.
                quantile = (
                    (conditional_index + 0.5) / conditional_count
                    + (week + 1) * 0.6180339887498949
                ) % 1.0
                cumulative = 0.0
                count_c = 6
                for candidate in range(7):
                    cumulative += (
                        math.comb(6, candidate)
                        * probability_c**candidate
                        * (1.0 - probability_c) ** (6 - candidate)
                    )
                    if quantile <= cumulative + 1e-15:
                        count_c = candidate
                        break
                labels = ["P_C"] * count_c + ["P_H"] * (6 - count_c)
                rotation = (conditional_index + week) % 6
                labels = labels[rotation:] + labels[:rotation]
                quantities_week = [2500.0] * 6
            else:
                labels = ["P_C" if rng.random() < probability_c else "P_H" for _ in range(6)]
                quantities_week = list(map(float, rng.integers(2400, 2601, size=6)))
            for offset, product_id, quantity in zip((30, 54, 78, 102, 126, 150), labels, quantities_week):
                times.append(base.decision_start + 168.0 * week + float(offset))
                quantities.append(quantity)
                products.append(product_id)
                contingent.append(False)
            if config.belief_integration == "stratified":
                transition_quantile = (
                    (conditional_index + 0.5) / conditional_count
                    + (week + 1) * 0.4142135623730950
                ) % 1.0
                if transition_quantile > rho:
                    regime_c = not regime_c
            elif rng.random() > rho:
                regime_c = not regime_c
        order = np.argsort(np.asarray(times), kind="stable")
        out.append(
            replace(
                base,
                seed=-(scenario_index + 1),
                order_times=tuple(float(times[i]) for i in order),
                order_quantities=tuple(float(quantities[i]) for i in order),
                order_products=tuple(str(products[i]) for i in order),
                order_contingent=tuple(bool(contingent[i]) for i in order),
                tape_sha256="joint_belief_generated_no_realized_future",
                skeleton_sha256=hashlib.sha256(
                    f"{observation.observation_sha256}:joint:{scenario_index}".encode()
                ).hexdigest(),
            )
        )
        weights.append(float(scenario_weight))
    normalized = np.asarray(weights, dtype=float)
    normalized /= normalized.sum()
    return tuple(out), normalized


def _weighted_lower_tail_mean(
    values: np.ndarray, weights: np.ndarray, alpha: float
) -> np.ndarray:
    """Return the weighted lower-tail mean for every candidate row."""
    if values.ndim != 2 or weights.shape != (values.shape[1],):
        raise ValueError("weighted tail inputs have incompatible shapes")
    order = np.argsort(values, axis=1)
    sorted_values = np.take_along_axis(values, order, axis=1)
    sorted_weights = weights[order]
    cumulative_before = np.cumsum(sorted_weights, axis=1) - sorted_weights
    included = np.clip(float(alpha) - cumulative_before, 0.0, sorted_weights)
    denominator = included.sum(axis=1)
    return np.divide(
        (sorted_values * included).sum(axis=1),
        denominator,
        out=np.zeros(values.shape[0], dtype=float),
        where=denominator > 0.0,
    )


def choose_ret_transducer_action(
    observation: StateRichObservation,
    *,
    base_skeleton: FullDESSkeleton,
    prefix: Sequence[int],
    scheduler: Mapping[str, Sequence[str]],
    config: FullDEST0Config,
    joint_belief: ExactJointBelief | None = None,
) -> tuple[int, dict[str, object]]:
    """Score candidate action sequences with canonical ReT transducer rollouts."""
    horizon = min(config.horizon, observation.remaining_decisions)
    sequences = _sequences(horizon)
    tail = base_skeleton.decision_weeks - len(prefix) - horizon
    calendars = np.empty((len(sequences), base_skeleton.decision_weeks), dtype=np.uint8)
    if prefix:
        calendars[:, : len(prefix)] = np.asarray(prefix, dtype=np.uint8)
    calendars[:, len(prefix) : len(prefix) + horizon] = sequences
    if tail:
        calendars[:, len(prefix) + horizon :] = sequences[:, -1, None]
    if joint_belief is None:
        scenarios = _synthetic_skeletons(base_skeleton, observation, config)
        scenario_weights = np.full(len(scenarios), 1.0 / len(scenarios))
    else:
        scenarios, scenario_weights = _joint_belief_skeletons(
            base_skeleton, observation, config, joint_belief
        )
    ret = np.empty((len(sequences), len(scenarios)), dtype=float)
    worst = np.empty_like(ret)
    lost = np.empty_like(ret)
    for index, scenario in enumerate(scenarios):
        panel = simulate_full_des_frontier(
            skeleton=scenario, scheduler=scheduler, calendars=calendars
        )
        ret[:, index] = panel["ret_visible"]
        worst[:, index] = panel["worst_product_fill"]
        lost[:, index] = panel["lost_orders"]
    mean_ret = ret @ scenario_weights
    tail_ret = _weighted_lower_tail_mean(ret, scenario_weights, 0.10)
    min_fill = worst.min(axis=1)
    feasible = (min_fill >= config.worst_product_floor) & (lost.max(axis=1) <= 1e-12)
    primary = tail_ret if config.mode == "robust" else mean_ret
    if config.mode != "constraint_aware":
        feasible[:] = True
    fallback_used = bool(config.mode == "constraint_aware" and not np.any(feasible))
    if fallback_used:
        best = max(
            range(len(sequences)),
            key=lambda i: (
                float(tail_ret[i]),
                float(min_fill[i]),
                float(mean_ret[i]),
                *tuple(-int(x) for x in sequences[i]),
            ),
        )
    else:
        best = max(
            range(len(sequences)),
            key=lambda i: (
                int(feasible[i]),
                float(primary[i]),
                float(tail_ret[i]),
                float(min_fill[i]),
                *tuple(-int(x) for x in sequences[i]),
            ),
        )
    return int(sequences[best, 0]), {
        "candidate_sequences": float(len(sequences)),
        "scenario_count": float(len(scenarios)),
        "belief_integration": config.belief_integration,
        "planning_ret": float(mean_ret[best]),
        "planning_tail": float(tail_ret[best]),
        "planning_worst_fill": float(min_fill[best]),
        "planning_feasible": float(feasible[best]),
        "fallback_used": float(fallback_used),
        "fallback_policy": "robust_tail" if fallback_used else "none",
    }


def ret_transducer_t0_calendar(
    *,
    skeleton: FullDESSkeleton,
    scheduler: Mapping[str, Sequence[str]],
    config: FullDEST0Config,
) -> tuple[tuple[int, ...], dict[str, object]]:
    prefix: list[int] = []
    decisions = []
    started = time.perf_counter()
    for week in range(skeleton.decision_weeks):
        probe = tuple(prefix + [0] * (skeleton.decision_weeks - len(prefix)))
        _calendar, rows = state_rich_calendar(
            skeleton=skeleton.as_dict(),
            scheduler=scheduler,
            config=StateRichConfiguration("belief_mpc", 1),
            regime_persistence=config.regime_persistence,
            dominant_share=config.dominant_share,
            action_overrides=probe,
        )
        observation = rows[week].observation
        action, diagnostics = choose_ret_transducer_action(
            observation,
            base_skeleton=skeleton,
            prefix=prefix,
            scheduler=scheduler,
            config=config,
        )
        prefix.append(action)
        decisions.append({"week": week, "action": action, **diagnostics})
    return tuple(prefix), {
        "config_id": "ret_transducer_" + config.config_id,
        "online_ms": (time.perf_counter() - started) * 1000.0,
        "decisions": decisions,
    }


def joint_belief_ret_transducer_calendar(
    *,
    skeleton: FullDESSkeleton,
    scheduler: Mapping[str, Sequence[str]],
    config: FullDEST0Config,
    belief: ExactJointBelief,
    history_transform: str = "real",
) -> tuple[tuple[int, ...], dict[str, object]]:
    """Plan with a causal joint posterior; no true current regime is consumed."""
    if history_transform not in {"real", "wrong_product", "shuffled"}:
        raise ValueError("unknown history transform")
    counts = list(
        weekly_product_counts(
            order_times=skeleton.order_times,
            order_products=skeleton.order_products,
            decision_start=skeleton.decision_start,
            weeks=skeleton.decision_weeks,
        )
    )
    if history_transform == "wrong_product":
        counts = [6 - count for count in counts]
    elif history_transform == "shuffled":
        counts = list(reversed(counts))
    posterior = belief.copy()
    prefix: list[int] = []
    decisions = []
    started = time.perf_counter()
    for week in range(skeleton.decision_weeks):
        if week:
            posterior.observe_previous_week(counts[week - 1])
        probe = tuple(prefix + [0] * (skeleton.decision_weeks - len(prefix)))
        _calendar, rows = state_rich_calendar(
            skeleton=skeleton.as_dict(),
            scheduler=scheduler,
            config=StateRichConfiguration("belief_mpc", 1),
            regime_persistence=config.regime_persistence,
            dominant_share=config.dominant_share,
            action_overrides=probe,
        )
        observation = rows[week].observation
        action, diagnostics = choose_ret_transducer_action(
            observation,
            base_skeleton=skeleton,
            prefix=prefix,
            scheduler=scheduler,
            config=config,
            joint_belief=posterior,
        )
        prefix.append(action)
        decisions.append(
            {
                "week": week,
                "action": action,
                "posterior": posterior.as_dict(),
                **diagnostics,
            }
        )
    return tuple(prefix), {
        "config_id": "ret_transducer_joint_" + config.config_id,
        "history_transform": history_transform,
        "online_ms": (time.perf_counter() - started) * 1000.0,
        "decisions": decisions,
    }
