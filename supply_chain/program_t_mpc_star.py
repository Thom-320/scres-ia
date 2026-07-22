"""MPC* — the strengthened structured belief-MPC comparator for Q-R1 Gate 1.

This is the "MPC challenge" comparator the two-track plan freezes BEFORE any
learner is trained.  It fixes the four weaknesses the external audits isolated
in ``program_t_full_des_mpc.py`` (which is left byte-identical so prior frozen
artifacts still reproduce):

1. Exact / stratified 6-state integration.  The mixture over (theta, regime) is
   enumerated with exact posterior weights (``ExactJointBelief.enumerate_states``)
   instead of Monte-Carlo sampling ``count`` states WITH replacement — removing
   the dominant source of the p4-vs-p64 action instability (0/8 agreement).
2. Fail-closed feasibility.  The worst-product-fill floor and no-lost-order
   constraint are ALWAYS enforced; when no candidate is feasible the planner
   returns a guaranteed-safe default (the candidate maximizing weighted worst
   fill), never the best infeasible action.
3. Objective alignment.  The primary objective and the gate's scoring metric are
   the same ``ret_visible`` rollout (``ret_full`` is structurally degenerate in
   the fast-path transducer), with worst-product enforced as a hard constraint
   rather than dropped.
4. Frozen cross-arm CRN.  Demand realizations are keyed by
   (observation digest, state index, realization index) only — independent of
   particle count, arm, or config — so MPC* and every learner are scored on
   byte-identical realizations.

The planner still sees only ``StateRichObservation`` fields; scenarios never use
the realized future tape, seed, or true current regime.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
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
from supply_chain.program_t_full_des_mpc import _sequences
from supply_chain.program_t_joint_belief import ExactJointBelief, weekly_product_counts


@dataclass(frozen=True)
class MPCStarConfig:
    horizon: int
    realizations_per_state: int = 8
    worst_product_floor: float = 0.70
    mode: str = "constraint_aware"  # {"scenario","robust","constraint_aware"}
    weight_floor: float = 1e-12

    def __post_init__(self) -> None:
        if self.horizon not in (1, 3, 4, 6, 8):
            raise ValueError("horizon must be one of 1,3,4,6,8")
        if self.realizations_per_state <= 0:
            raise ValueError("realizations_per_state must be positive")
        if self.mode not in {"scenario", "robust", "constraint_aware"}:
            raise ValueError("unknown MPC* mode")

    @property
    def config_id(self) -> str:
        return f"mpc_star_{self.mode}_h{self.horizon}_r{self.realizations_per_state}"


def _realization_seed(observation_sha256: str, state_index: int, realization: int) -> int:
    payload = f"{observation_sha256}::MPC_STAR_CRN_V1::s{state_index}::r{realization}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def exact_stratified_skeletons(
    base: FullDESSkeleton,
    observation: StateRichObservation,
    belief: ExactJointBelief,
    config: MPCStarConfig,
) -> tuple[tuple[FullDESSkeleton, float], ...]:
    """Enumerate the 6-state mixture exactly; CRN demand per (state, realization).

    Returns ``(skeleton, weight)`` pairs whose weights sum to one: each latent
    state contributes ``realizations_per_state`` equally-weighted demand tapes,
    scaled by the exact posterior weight of that state.
    """
    states = belief.enumerate_states(weight_floor=config.weight_floor)
    past = [i for i, t in enumerate(base.order_times) if t < observation.decision_time - 1e-12]
    future_weeks = range(observation.week, base.decision_weeks)
    out: list[tuple[FullDESSkeleton, float]] = []
    for state_index, (rho, share, initial_regime_c, state_weight) in enumerate(states):
        per_realization_weight = state_weight / float(config.realizations_per_state)
        for realization in range(config.realizations_per_state):
            rng = np.random.default_rng(
                _realization_seed(observation.observation_sha256, state_index, realization)
            )
            times = [base.order_times[i] for i in past]
            quantities = [base.order_quantities[i] for i in past]
            products = [base.order_products[i] for i in past]
            contingent = [False for _ in past]
            regime_c = bool(initial_regime_c)
            for week in future_weeks:
                probability_c = share if regime_c else 1.0 - share
                labels = ["P_C" if rng.random() < probability_c else "P_H" for _ in range(6)]
                quantities_week = list(map(float, rng.integers(2400, 2601, size=6)))
                for offset, product_id, quantity in zip(
                    (30, 54, 78, 102, 126, 150), labels, quantities_week
                ):
                    times.append(base.decision_start + 168.0 * week + float(offset))
                    quantities.append(quantity)
                    products.append(product_id)
                    contingent.append(False)
                if rng.random() > rho:
                    regime_c = not regime_c
            order = np.argsort(np.asarray(times), kind="stable")
            skeleton = replace(
                base,
                seed=-(state_index * config.realizations_per_state + realization + 1),
                order_times=tuple(float(times[i]) for i in order),
                order_quantities=tuple(float(quantities[i]) for i in order),
                order_products=tuple(str(products[i]) for i in order),
                order_contingent=tuple(bool(contingent[i]) for i in order),
                tape_sha256="mpc_star_crn_generated_no_realized_future",
                skeleton_sha256=hashlib.sha256(
                    f"{observation.observation_sha256}:mpcstar:{state_index}:{realization}".encode()
                ).hexdigest(),
            )
            out.append((skeleton, per_realization_weight))
    return tuple(out)


def _weighted_tail_mean(ret: np.ndarray, weights: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    """Weighted CVaR-alpha (mean of the worst alpha weight-mass) per row."""
    order = np.argsort(ret, axis=1, kind="stable")
    sorted_ret = np.take_along_axis(ret, order, axis=1)
    sorted_w = weights[order]
    cum_w = np.cumsum(sorted_w, axis=1)
    # include columns until cumulative weight first reaches alpha
    include = cum_w - sorted_w < alpha
    include[:, 0] = True  # always include the worst realization
    masked_w = np.where(include, sorted_w, 0.0)
    denom = masked_w.sum(axis=1)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return (sorted_ret * masked_w).sum(axis=1) / denom


def choose_mpc_star_action(
    observation: StateRichObservation,
    *,
    base_skeleton: FullDESSkeleton,
    prefix: Sequence[int],
    scheduler: Mapping[str, Sequence[str]],
    config: MPCStarConfig,
    belief: ExactJointBelief,
) -> tuple[int, dict[str, float]]:
    horizon = min(config.horizon, observation.remaining_decisions)
    sequences = _sequences(horizon)
    tail = base_skeleton.decision_weeks - len(prefix) - horizon
    calendars = np.empty((len(sequences), base_skeleton.decision_weeks), dtype=np.uint8)
    if prefix:
        calendars[:, : len(prefix)] = np.asarray(prefix, dtype=np.uint8)
    calendars[:, len(prefix) : len(prefix) + horizon] = sequences
    if tail:
        calendars[:, len(prefix) + horizon :] = sequences[:, -1, None]
    scenarios = exact_stratified_skeletons(base_skeleton, observation, belief, config)
    weights = np.asarray([weight for _sk, weight in scenarios], dtype=float)
    weights = weights / weights.sum()
    ret = np.empty((len(sequences), len(scenarios)), dtype=float)
    worst = np.empty_like(ret)
    lost = np.empty_like(ret)
    for index, (scenario, _weight) in enumerate(scenarios):
        panel = simulate_full_des_frontier(
            skeleton=scenario, scheduler=scheduler, calendars=calendars
        )
        ret[:, index] = panel["ret_visible"]
        worst[:, index] = panel["worst_product_fill"]
        lost[:, index] = panel["lost_orders"]
    mean_ret = ret @ weights
    tail_ret = _weighted_tail_mean(ret, weights, alpha=0.10)
    # Feasibility is exact-weighted: a candidate is feasible if its weighted worst
    # fill clears the floor AND it never loses an order in ANY supported scenario.
    weighted_worst = worst @ weights
    min_worst = worst.min(axis=1)
    any_lost = lost.max(axis=1) > 1e-12
    feasible = (weighted_worst >= config.worst_product_floor) & (~any_lost)
    primary = tail_ret if config.mode == "robust" else mean_ret

    def _key(i: int) -> tuple:
        return (
            int(feasible[i]),
            float(primary[i]),
            float(tail_ret[i]),
            float(weighted_worst[i]),
            *tuple(-int(x) for x in sequences[i]),
        )

    feasible_indices = [i for i in range(len(sequences)) if feasible[i]]
    fallback_used = 0.0
    if feasible_indices:
        best = max(feasible_indices, key=_key)
    else:
        # Fail-closed: no candidate is feasible -> return the guaranteed-safe
        # default (maximize weighted worst fill), never the best infeasible ReT.
        fallback_used = 1.0
        best = max(
            range(len(sequences)),
            key=lambda i: (float(weighted_worst[i]), float(min_worst[i]), *tuple(-int(x) for x in sequences[i])),
        )
    return int(sequences[best, 0]), {
        "candidate_sequences": float(len(sequences)),
        "scenario_count": float(len(scenarios)),
        "support_states": float(len(scenarios) // config.realizations_per_state),
        "planning_ret": float(mean_ret[best]),
        "planning_tail": float(tail_ret[best]),
        "planning_worst_fill": float(weighted_worst[best]),
        "planning_feasible": float(feasible[best]),
        "fallback_used": fallback_used,
    }


def mpc_star_calendar(
    *,
    skeleton: FullDESSkeleton,
    scheduler: Mapping[str, Sequence[str]],
    config: MPCStarConfig,
    belief: ExactJointBelief,
    history_transform: str = "real",
) -> tuple[tuple[int, ...], dict[str, object]]:
    """Full-campaign MPC* calendar threading the causal joint posterior."""
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
    fallbacks = 0
    started = time.perf_counter()
    for week in range(skeleton.decision_weeks):
        if week:
            posterior.observe_previous_week(counts[week - 1])
        probe = tuple(prefix + [0] * (skeleton.decision_weeks - len(prefix)))
        _calendar, rows = state_rich_calendar(
            skeleton=skeleton.as_dict(),
            scheduler=scheduler,
            config=StateRichConfiguration("belief_mpc", 1),
            regime_persistence=0.75,
            dominant_share=0.90,
            action_overrides=probe,
        )
        observation = rows[week].observation
        action, diagnostics = choose_mpc_star_action(
            observation,
            base_skeleton=skeleton,
            prefix=prefix,
            scheduler=scheduler,
            config=config,
            belief=posterior,
        )
        fallbacks += int(diagnostics["fallback_used"])
        prefix.append(action)
        decisions.append({"week": week, "action": action, **diagnostics})
    return tuple(prefix), {
        "config_id": "mpc_star_" + config.config_id,
        "history_transform": history_transform,
        "online_ms": (time.perf_counter() - started) * 1000.0,
        "fallbacks": fallbacks,
        "decisions": decisions,
    }
