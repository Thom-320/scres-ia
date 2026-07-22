"""Deployable comparator challenge v2 for Q-R1 burned development.

This module is intentionally separate from the frozen historical planner.  It
uses a planning bank keyed only by history root, campaign, week, latent state,
and conditional path.  Beliefs affect scenario weights, never the simulated
conditional paths, so retained and reset arms receive genuine common random
numbers even after their actions and observations diverge.

The comparator optimizes the non-censorable early ReT cohort.  Service and
ledger restrictions are explicit.  If no candidate is feasible it raises an
abstention instead of labelling an infeasible action as safe.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
from itertools import product
import math
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
from supply_chain.program_t_joint_belief import ExactJointBelief, weekly_product_counts
from supply_chain.q_r1_retained_learning import PhysicalCampaignState


@dataclass(frozen=True)
class PlanningKey:
    history_root: int
    campaign_index: int
    week: int

    def token(self) -> str:
        return f"{self.history_root}:{self.campaign_index}:{self.week}"


@dataclass(frozen=True)
class ComparatorV2Config:
    horizon: int
    conditional_paths: int
    mode: str = "constraint_aware"
    worst_product_floor: float = 0.70
    max_unresolved_orders: float | None = None
    tail_alpha: float = 0.10
    service_statistic: str = "expected"
    value_indifference_tolerance: float = 0.0
    tie_breaker: str = "legacy"

    def __post_init__(self) -> None:
        if self.horizon not in (1, 3, 4, 6, 8):
            raise ValueError("horizon must be one of 1,3,4,6,8")
        if self.conditional_paths <= 0:
            raise ValueError("conditional_paths must be positive")
        if self.mode not in {"scenario", "robust", "constraint_aware"}:
            raise ValueError("unknown comparator v2 mode")
        if not 0.0 <= self.worst_product_floor <= 1.0:
            raise ValueError("worst_product_floor must be in [0,1]")
        if self.max_unresolved_orders is not None and self.max_unresolved_orders < 0:
            raise ValueError("max_unresolved_orders must be non-negative")
        if not 0.0 < self.tail_alpha <= 1.0:
            raise ValueError("tail_alpha must be in (0,1]")
        if self.service_statistic not in {"expected", "worst_case"}:
            raise ValueError("service_statistic must be expected or worst_case")
        if self.value_indifference_tolerance < 0.0:
            raise ValueError("value_indifference_tolerance must be non-negative")
        if self.tie_breaker not in {"legacy", "service"}:
            raise ValueError("tie_breaker must be legacy or service")

    @property
    def config_id(self) -> str:
        unresolved = (
            "none"
            if self.max_unresolved_orders is None
            else str(int(self.max_unresolved_orders))
        )
        return (
            f"qr1_v2_{self.mode}_h{self.horizon}_c{self.conditional_paths}"
            f"_wf{self.worst_product_floor:.2f}_u{unresolved}"
            f"_{self.service_statistic}_tol{self.value_indifference_tolerance:.4f}"
            f"_{self.tie_breaker}"
        )


class NoFeasibleStructuredAction(RuntimeError):
    """The planning bank contains no action satisfying frozen restrictions."""


@lru_cache(maxsize=None)
def _sequences(horizon: int) -> np.ndarray:
    return np.asarray(tuple(product(range(4), repeat=int(horizon))), dtype=np.uint8)


def _van_der_corput(index: int, base: int) -> float:
    value = 0.0
    denominator = 1.0
    current = int(index)
    while current:
        current, remainder = divmod(current, int(base))
        denominator *= float(base)
        value += float(remainder) / denominator
    return value


def _fixed_shift(key: PlanningKey, *, theta_index: int, regime_index: int, stream: str) -> float:
    token = f"{key.token()}:{theta_index}:{regime_index}:{stream}"
    raw = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
    return raw / float(1 << 64)


def _binomial_inverse_six(probability: float, quantile: float) -> int:
    cumulative = 0.0
    for count in range(7):
        cumulative += (
            math.comb(6, count)
            * probability**count
            * (1.0 - probability) ** (6 - count)
        )
        if quantile <= cumulative + 1e-15:
            return count
    return 6


def _weighted_lower_tail_mean(values: np.ndarray, weights: np.ndarray, alpha: float) -> np.ndarray:
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


def conditional_scenario_bank(
    *,
    base: FullDESSkeleton,
    observation: StateRichObservation,
    belief: ExactJointBelief,
    key: PlanningKey,
    conditional_paths: int,
) -> tuple[tuple[FullDESSkeleton, ...], np.ndarray, str]:
    """Build nested conditional paths and apply exact posterior weights."""
    if key.week != int(observation.week):
        raise ValueError("planning key week does not match observation week")
    past = [i for i, time_value in enumerate(base.order_times) if time_value < observation.decision_time - 1e-12]
    future_weeks = range(int(observation.week), int(base.decision_weeks))
    scenarios: list[FullDESSkeleton] = []
    weights: list[float] = []
    scenario_tokens: list[str] = []
    states = belief.enumerate_states()
    for state_index, (rho, share, initial_regime_c, state_weight) in enumerate(states):
        theta_index, regime_index = divmod(state_index, 2)
        count_shift = _fixed_shift(
            key, theta_index=theta_index, regime_index=regime_index, stream="count"
        )
        transition_shift = _fixed_shift(
            key, theta_index=theta_index, regime_index=regime_index, stream="transition"
        )
        for conditional_index in range(int(conditional_paths)):
            times = [base.order_times[i] for i in past]
            quantities = [base.order_quantities[i] for i in past]
            products = [base.order_products[i] for i in past]
            contingent = [False for _ in past]
            regime_c = bool(initial_regime_c)
            for relative_week, week in enumerate(future_weeks):
                probability_c = float(share) if regime_c else 1.0 - float(share)
                count_quantile = (
                    _van_der_corput(conditional_index + 1, 2)
                    + count_shift
                    + _van_der_corput(relative_week + 1, 5)
                ) % 1.0
                count_c = _binomial_inverse_six(probability_c, count_quantile)
                labels = ["P_C"] * count_c + ["P_H"] * (6 - count_c)
                rotation = (
                    conditional_index + relative_week + theta_index + regime_index
                ) % 6
                labels = labels[rotation:] + labels[:rotation]
                for offset, product_id in zip((30, 54, 78, 102, 126, 150), labels):
                    times.append(base.decision_start + 168.0 * week + float(offset))
                    quantities.append(2500.0)
                    products.append(product_id)
                    contingent.append(False)
                transition_quantile = (
                    _van_der_corput(conditional_index + 1, 3)
                    + transition_shift
                    + _van_der_corput(relative_week + 1, 7)
                ) % 1.0
                if transition_quantile > float(rho):
                    regime_c = not regime_c
            order = np.argsort(np.asarray(times), kind="stable")
            scenario_token = (
                f"{key.token()}:{theta_index}:{regime_index}:{conditional_index}"
            )
            scenarios.append(
                replace(
                    base,
                    seed=-(state_index * 1_000_000 + conditional_index + 1),
                    order_times=tuple(float(times[i]) for i in order),
                    order_quantities=tuple(float(quantities[i]) for i in order),
                    order_products=tuple(str(products[i]) for i in order),
                    order_contingent=tuple(bool(contingent[i]) for i in order),
                    tape_sha256="qr1_v2_conditional_bank_no_realized_future",
                    skeleton_sha256=hashlib.sha256(scenario_token.encode()).hexdigest(),
                )
            )
            weights.append(float(state_weight) / float(conditional_paths))
            scenario_tokens.append(scenario_token)
    normalized = np.asarray(weights, dtype=float)
    if not np.isclose(normalized.sum(), 1.0, atol=1e-12):
        raise AssertionError("scenario weights do not sum to one")
    bank_sha = hashlib.sha256("|".join(scenario_tokens).encode()).hexdigest()
    return tuple(scenarios), normalized, bank_sha


def choose_comparator_v2_action(
    observation: StateRichObservation,
    *,
    base_skeleton: FullDESSkeleton,
    prefix: Sequence[int],
    scheduler: Mapping[str, Sequence[str]],
    belief: ExactJointBelief,
    planning_key: PlanningKey,
    config: ComparatorV2Config,
) -> tuple[int, dict[str, object]]:
    horizon = min(int(config.horizon), int(observation.remaining_decisions))
    sequences = _sequences(horizon)
    tail = int(base_skeleton.decision_weeks) - len(prefix) - horizon
    calendars = np.empty((len(sequences), base_skeleton.decision_weeks), dtype=np.uint8)
    if prefix:
        calendars[:, : len(prefix)] = np.asarray(prefix, dtype=np.uint8)
    calendars[:, len(prefix) : len(prefix) + horizon] = sequences
    if tail:
        calendars[:, len(prefix) + horizon :] = sequences[:, -1, None]

    scenarios, weights, bank_sha = conditional_scenario_bank(
        base=base_skeleton,
        observation=observation,
        belief=belief,
        key=planning_key,
        conditional_paths=config.conditional_paths,
    )
    shape = (len(sequences), len(scenarios))
    early_complete = np.empty(shape, dtype=float)
    ret_visible = np.empty(shape, dtype=float)
    ret_full = np.empty(shape, dtype=float)
    worst_fill = np.empty(shape, dtype=float)
    unresolved = np.empty(shape, dtype=float)
    lost = np.empty(shape, dtype=float)
    for scenario_index, scenario in enumerate(scenarios):
        panel = simulate_full_des_frontier(
            skeleton=scenario,
            scheduler=scheduler,
            calendars=calendars,
            include_q_r1_metrics=True,
        )
        early_complete[:, scenario_index] = panel["early_ret_complete_cohort"]
        ret_visible[:, scenario_index] = panel["ret_visible"]
        ret_full[:, scenario_index] = panel["ret_full"]
        worst_fill[:, scenario_index] = panel["worst_product_fill"]
        unresolved[:, scenario_index] = panel["unresolved_orders"]
        lost[:, scenario_index] = panel["lost_orders"]

    mean_early = early_complete @ weights
    tail_early = _weighted_lower_tail_mean(
        early_complete, weights, config.tail_alpha
    )
    mean_visible = ret_visible @ weights
    mean_full = ret_full @ weights
    minimum_fill = worst_fill.min(axis=1)
    expected_fill = worst_fill @ weights
    maximum_unresolved = unresolved.max(axis=1)
    expected_unresolved = unresolved @ weights
    maximum_lost = lost.max(axis=1)
    service_fill = (
        expected_fill
        if config.service_statistic == "expected"
        else minimum_fill
    )
    service_unresolved = (
        expected_unresolved
        if config.service_statistic == "expected"
        else maximum_unresolved
    )
    feasible = (service_fill >= config.worst_product_floor) & (maximum_lost <= 1e-12)
    if config.max_unresolved_orders is not None:
        feasible &= service_unresolved <= config.max_unresolved_orders + 1e-12
    if config.mode in {"scenario", "robust"}:
        feasible[:] = maximum_lost <= 1e-12
    if not np.any(feasible):
        raise NoFeasibleStructuredAction(
            f"no sequence satisfies {config.config_id} at {planning_key.token()}"
        )
    primary = tail_early if config.mode == "robust" else mean_early
    eligible = np.flatnonzero(feasible)
    if config.tie_breaker == "service":
        best_primary = float(primary[eligible].max())
        eligible = eligible[
            primary[eligible]
            >= best_primary - float(config.value_indifference_tolerance) - 1e-15
        ]
        best = max(
            map(int, eligible),
            key=lambda index: (
                float(service_fill[index]),
                -float(service_unresolved[index]),
                float(mean_full[index]),
                float(mean_visible[index]),
                *tuple(-int(value) for value in sequences[index]),
            ),
        )
    else:
        best = max(
            map(int, eligible),
            key=lambda index: (
                float(primary[index]),
                float(mean_early[index]),
                float(tail_early[index]),
                float(service_fill[index]),
                float(mean_full[index]),
                *tuple(-int(value) for value in sequences[index]),
            ),
        )
    return int(sequences[best, 0]), {
        "config_id": config.config_id,
        "planning_key": planning_key.token(),
        "scenario_bank_sha256": bank_sha,
        "scenario_count": len(scenarios),
        "latent_state_count": 6,
        "exact_posterior_weights": True,
        "conditional_demand_exact": False,
        "planning_early_ret_complete_cohort": float(mean_early[best]),
        "planning_early_tail": float(tail_early[best]),
        "planning_ret_visible": float(mean_visible[best]),
        "planning_ret_full": float(mean_full[best]),
        "planning_worst_product_fill": float(minimum_fill[best]),
        "planning_expected_worst_product_fill": float(expected_fill[best]),
        "planning_max_unresolved_orders": float(maximum_unresolved[best]),
        "planning_expected_unresolved_orders": float(expected_unresolved[best]),
        "planning_max_lost_orders": float(maximum_lost[best]),
        "planning_feasible": True,
        "value_indifference_tolerance": config.value_indifference_tolerance,
        "tie_breaker": config.tie_breaker,
        "near_optimal_sequence_count": int(len(eligible)),
        "fallback_used": False,
    }


def comparator_v2_calendar(
    *,
    campaign: PhysicalCampaignState,
    belief: ExactJointBelief,
    scheduler: Mapping[str, Sequence[str]],
    config: ComparatorV2Config,
) -> tuple[tuple[int, ...], dict[str, object]]:
    counts = weekly_product_counts(
        order_times=campaign.skeleton.order_times,
        order_products=campaign.skeleton.order_products,
        decision_start=campaign.skeleton.decision_start,
        weeks=campaign.skeleton.decision_weeks,
    )
    posterior = belief.copy()
    prefix: list[int] = []
    diagnostics: list[dict[str, object]] = []
    for week in range(campaign.skeleton.decision_weeks):
        if week:
            posterior.observe_previous_week(counts[week - 1])
        probe = tuple(prefix + [0] * (campaign.skeleton.decision_weeks - len(prefix)))
        _calendar, rows = state_rich_calendar(
            skeleton=campaign.skeleton.as_dict(),
            scheduler=scheduler,
            config=StateRichConfiguration("belief_mpc", 1),
            regime_persistence=0.90,
            dominant_share=0.90,
            action_overrides=probe,
        )
        observation = rows[week].observation
        action, detail = choose_comparator_v2_action(
            observation,
            base_skeleton=campaign.skeleton,
            prefix=prefix,
            scheduler=scheduler,
            belief=posterior,
            planning_key=PlanningKey(
                history_root=campaign.history_root,
                campaign_index=campaign.campaign_index,
                week=week,
            ),
            config=config,
        )
        prefix.append(action)
        diagnostics.append({"week": week, "action": action, **detail})
    return tuple(prefix), {
        "config_id": config.config_id,
        "decisions": diagnostics,
        "abstained": False,
    }
