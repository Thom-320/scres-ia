"""Finite state-rich classical controllers for Program O.

This module is development-only.  It maps a strictly non-anticipative
operational state to the same four weekly production actions used by the
complete 4^8 open-loop frontier.  It never reads a tape identifier, latent
regime, future demand label/quantity, oracle calendar, or score-time metric.

The policies are deliberately finite and parameter-free beyond the small
enumerated family returned by :func:`finite_state_rich_configurations`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
from itertools import product
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from supply_chain.program_o_full_des import PRODUCTS
from supply_chain.program_o_hobs import (
    observable_calendar,
    posterior_after_week,
    predicted_request_share_c,
    transition_belief,
)


WEEKLY_ORDERS = 6
EXPECTED_ORDER_QUANTITY = 2500.0
EXPECTED_WEEKLY_DEMAND = WEEKLY_ORDERS * EXPECTED_ORDER_QUANTITY
BATCH_QUANTITY = 5000.0


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class StateRichConfiguration:
    policy_id: str
    parameter: int

    @property
    def config_id(self) -> str:
        return f"{self.policy_id}__{self.parameter}"


@dataclass(frozen=True)
class StateRichObservation:
    week: int
    decision_time: float
    on_hand: tuple[float, float]
    locked_pipeline: tuple[float, float]
    backlog_quantity: tuple[float, float]
    backlog_orders: tuple[int, int]
    max_backlog_age: tuple[float, float]
    in_flight_quantity: tuple[float, float]
    belief_c: float
    predicted_share_c: float
    previous_action: int | None
    remaining_decisions: int
    observation_sha256: str


@dataclass(frozen=True)
class StateRichDecision:
    observation: StateRichObservation
    action: int
    objective: tuple[float, ...]
    tied_actions: tuple[int, ...]


def _observation_payload(observation: StateRichObservation) -> dict[str, Any]:
    payload = asdict(observation)
    payload.pop("observation_sha256", None)
    return payload


def swap_product_channels(
    observation: StateRichObservation,
) -> StateRichObservation:
    """Return the frozen C/H-label counterfactual for one observation."""
    payload = _observation_payload(observation)
    for key in (
        "on_hand",
        "locked_pipeline",
        "backlog_quantity",
        "backlog_orders",
        "max_backlog_age",
        "in_flight_quantity",
    ):
        payload[key] = tuple(reversed(payload[key]))
    payload["belief_c"] = 1.0 - float(observation.belief_c)
    payload["predicted_share_c"] = 1.0 - float(observation.predicted_share_c)
    payload["previous_action"] = (
        None
        if observation.previous_action is None
        else 3 - int(observation.previous_action)
    )
    return replace(
        observation,
        **payload,
        observation_sha256=_digest(payload),
    )


def add_minority_backlog(
    observation: StateRichObservation,
    *,
    action: int,
    scheduler: Mapping[str, Sequence[str]],
) -> StateRichObservation:
    """Add one batch to the product receiving fewer slots under ``action``."""
    counts = _scheduler_array(scheduler)[int(action)]
    minority = 0 if counts[0] < counts[1] else 1
    backlog = list(map(float, observation.backlog_quantity))
    orders = list(map(int, observation.backlog_orders))
    backlog[minority] += BATCH_QUANTITY
    orders[minority] += 1
    payload = _observation_payload(observation)
    payload["backlog_quantity"] = tuple(backlog)
    payload["backlog_orders"] = tuple(orders)
    return replace(
        observation,
        backlog_quantity=tuple(backlog),
        backlog_orders=tuple(orders),
        observation_sha256=_digest(payload),
    )


def finite_state_rich_configurations() -> tuple[StateRichConfiguration, ...]:
    """Return the complete preregistered ten-controller family."""
    return (
        StateRichConfiguration("base_stock", 1),
        StateRichConfiguration("base_stock", 2),
        StateRichConfiguration("max_pressure", 0),
        StateRichConfiguration("max_pressure", 5000),
        StateRichConfiguration("min_cost_flow", 1),
        StateRichConfiguration("min_cost_flow", 2),
        StateRichConfiguration("belief_mpc", 3),
        StateRichConfiguration("belief_mpc", 4),
        StateRichConfiguration("belief_dp", 3),
        StateRichConfiguration("belief_dp", 4),
    )


def _scheduler_array(scheduler: Mapping[str, Sequence[str]]) -> tuple[tuple[int, int], ...]:
    output = []
    for action in range(4):
        labels = tuple(scheduler[str(action)])
        if len(labels) != 3 or any(label not in PRODUCTS for label in labels):
            raise ValueError("scheduler must map each action to three products")
        count_c = sum(label == "P_C" for label in labels)
        output.append((count_c, 3 - count_c))
    return tuple(output)


def _choose_lexicographic(
    rows: Sequence[tuple[int, tuple[float, ...]]]
) -> tuple[int, tuple[float, ...], tuple[int, ...]]:
    best = min(objective for _action, objective in rows)
    tied = tuple(action for action, objective in rows if objective == best)
    return min(tied), best, tied


def _forecast_shares(
    belief_c: float,
    *,
    regime_persistence: float,
    dominant_share: float,
    horizon: int,
) -> tuple[float, ...]:
    belief = float(belief_c)
    shares = []
    for step in range(int(horizon)):
        if step:
            belief = transition_belief(
                belief, regime_persistence=float(regime_persistence)
            )
        shares.append(
            predicted_request_share_c(
                belief, dominant_share=float(dominant_share)
            )
        )
    return tuple(shares)


def _inventory_position(observation: StateRichObservation) -> np.ndarray:
    return (
        np.asarray(observation.on_hand, dtype=float)
        + np.asarray(observation.locked_pipeline, dtype=float)
        - np.asarray(observation.backlog_quantity, dtype=float)
    )


def _target_allocation_action(
    observation: StateRichObservation,
    *,
    target: Sequence[float],
    scheduler_counts: Sequence[tuple[int, int]],
) -> tuple[int, tuple[float, ...], tuple[int, ...]]:
    position = _inventory_position(observation)
    target_array = np.asarray(target, dtype=float)
    rows = []
    for action, counts in enumerate(scheduler_counts):
        post = position + BATCH_QUANTITY * np.asarray(counts, dtype=float)
        shortage = np.maximum(0.0, target_array - post)
        excess = np.maximum(0.0, post - target_array)
        objective = (
            float(shortage.sum()),
            float(shortage.max()),
            float(excess.sum()),
            float(
                abs(
                    action
                    - (
                        observation.previous_action
                        if observation.previous_action is not None
                        else 1
                    )
                )
            ),
        )
        rows.append((action, objective))
    return _choose_lexicographic(rows)


def _deterministic_mpc_action(
    observation: StateRichObservation,
    *,
    horizon: int,
    scheduler_counts: Sequence[tuple[int, int]],
    regime_persistence: float,
    dominant_share: float,
) -> tuple[int, tuple[float, ...], tuple[int, ...]]:
    effective_horizon = min(int(horizon), int(observation.remaining_decisions))
    shares = _forecast_shares(
        observation.belief_c,
        regime_persistence=float(regime_persistence),
        dominant_share=float(dominant_share),
        horizon=effective_horizon,
    )
    initial = _inventory_position(observation)
    rows = []
    for sequence in product(range(4), repeat=effective_horizon):
        net = initial.copy()
        backlog_area = 0.0
        worst_backlog = 0.0
        switches = 0
        previous = observation.previous_action
        for step, action in enumerate(sequence):
            net += BATCH_QUANTITY * np.asarray(
                scheduler_counts[int(action)], dtype=float
            )
            demand = EXPECTED_WEEKLY_DEMAND * np.asarray(
                (shares[step], 1.0 - shares[step]), dtype=float
            )
            net -= demand
            shortage = np.maximum(0.0, -net)
            backlog_area += float(shortage.sum())
            worst_backlog = max(worst_backlog, float(shortage.max()))
            if previous is not None and int(action) != int(previous):
                switches += 1
            previous = int(action)
        terminal_shortage = np.maximum(0.0, -net)
        objective = (
            backlog_area,
            worst_backlog,
            float(terminal_shortage.sum()),
            float(terminal_shortage.max()),
            float(switches),
        )
        rows.append((int(sequence[0]), objective))
    best_objective = min(objective for _action, objective in rows)
    tied = tuple(sorted({action for action, objective in rows if objective == best_objective}))
    return min(tied), best_objective, tied


def _binomial_probability(n: int, k: int, probability: float) -> float:
    return math.comb(int(n), int(k)) * float(probability) ** int(k) * (
        1.0 - float(probability)
    ) ** (int(n) - int(k))


def _belief_dp_action(
    observation: StateRichObservation,
    *,
    horizon: int,
    scheduler_counts: Sequence[tuple[int, int]],
    regime_persistence: float,
    dominant_share: float,
) -> tuple[int, tuple[float, ...], tuple[int, ...]]:
    """Approximate belief DP over weekly product-count observations.

    The planning model uses the thesis mean order quantity (2,500) and the
    exact seven-outcome binomial mixture for the six weekly product labels.
    It is a strong classical comparator, not a certified bound on the full
    continuous-state POMDP.
    """
    effective_horizon = min(int(horizon), int(observation.remaining_decisions))
    initial = tuple(float(value) for value in _inventory_position(observation))
    rho = float(regime_persistence)
    share = float(dominant_share)

    @lru_cache(maxsize=None)
    def value(
        net_c: float,
        net_h: float,
        belief_key: float,
        depth: int,
        previous_action: int,
    ) -> tuple[float, float, int]:
        if depth <= 0:
            terminal = (max(0.0, -net_c), max(0.0, -net_h))
            return float(sum(terminal)), float(max(terminal)), -1
        belief = float(belief_key)
        action_rows: list[tuple[int, tuple[float, float]]] = []
        for action, counts in enumerate(scheduler_counts):
            supplied_c = net_c + BATCH_QUANTITY * float(counts[0])
            supplied_h = net_h + BATCH_QUANTITY * float(counts[1])
            expected_area = 0.0
            expected_worst = 0.0
            for count_c in range(WEEKLY_ORDERS + 1):
                probability = belief * _binomial_probability(
                    WEEKLY_ORDERS, count_c, share
                ) + (1.0 - belief) * _binomial_probability(
                    WEEKLY_ORDERS, count_c, 1.0 - share
                )
                if probability <= 0.0:
                    continue
                next_c = supplied_c - count_c * EXPECTED_ORDER_QUANTITY
                next_h = supplied_h - (
                    WEEKLY_ORDERS - count_c
                ) * EXPECTED_ORDER_QUANTITY
                labels = ("P_C",) * count_c + ("P_H",) * (
                    WEEKLY_ORDERS - count_c
                )
                posterior = posterior_after_week(
                    belief, labels, dominant_share=share
                )
                next_belief = transition_belief(
                    posterior, regime_persistence=rho
                )
                future_area, future_worst, _ = value(
                    round(next_c, 8),
                    round(next_h, 8),
                    round(next_belief, 12),
                    depth - 1,
                    int(action),
                )
                immediate = max(0.0, -next_c) + max(0.0, -next_h)
                immediate_worst = max(max(0.0, -next_c), max(0.0, -next_h))
                expected_area += probability * (immediate + future_area)
                expected_worst += probability * max(immediate_worst, future_worst)
            action_rows.append(
                (
                    int(action),
                    (
                        float(expected_area),
                        float(expected_worst),
                        float(abs(int(action) - int(previous_action))),
                    ),
                )
            )
        selected, objective, _tied = _choose_lexicographic(action_rows)
        return float(objective[0]), float(objective[1]), int(selected)

    rows = []
    previous = (
        int(observation.previous_action)
        if observation.previous_action is not None
        else 1
    )
    # Evaluate the root actions explicitly so the complete tie set is retained.
    for root_action, counts in enumerate(scheduler_counts):
        supplied_c = initial[0] + BATCH_QUANTITY * float(counts[0])
        supplied_h = initial[1] + BATCH_QUANTITY * float(counts[1])
        expected_area = 0.0
        expected_worst = 0.0
        for count_c in range(WEEKLY_ORDERS + 1):
            probability = observation.belief_c * _binomial_probability(
                WEEKLY_ORDERS, count_c, share
            ) + (1.0 - observation.belief_c) * _binomial_probability(
                WEEKLY_ORDERS, count_c, 1.0 - share
            )
            next_c = supplied_c - count_c * EXPECTED_ORDER_QUANTITY
            next_h = supplied_h - (WEEKLY_ORDERS - count_c) * EXPECTED_ORDER_QUANTITY
            labels = ("P_C",) * count_c + ("P_H",) * (WEEKLY_ORDERS - count_c)
            posterior = posterior_after_week(
                observation.belief_c, labels, dominant_share=share
            )
            next_belief = transition_belief(posterior, regime_persistence=rho)
            future_area, future_worst, _ = value(
                round(next_c, 8),
                round(next_h, 8),
                round(next_belief, 12),
                effective_horizon - 1,
                int(root_action),
            )
            immediate = max(0.0, -next_c) + max(0.0, -next_h)
            immediate_worst = max(max(0.0, -next_c), max(0.0, -next_h))
            expected_area += probability * (immediate + future_area)
            expected_worst += probability * max(immediate_worst, future_worst)
        rows.append(
            (
                int(root_action),
                (
                    float(expected_area),
                    float(expected_worst),
                    float(abs(int(root_action) - previous)),
                ),
            )
        )
    return _choose_lexicographic(rows)


def choose_state_rich_action(
    observation: StateRichObservation,
    config: StateRichConfiguration,
    *,
    scheduler: Mapping[str, Sequence[str]],
    regime_persistence: float,
    dominant_share: float,
) -> tuple[int, tuple[float, ...], tuple[int, ...]]:
    counts = _scheduler_array(scheduler)
    policy = str(config.policy_id)
    parameter = int(config.parameter)
    if policy == "base_stock":
        target = (EXPECTED_WEEKLY_DEMAND * parameter / 2.0,) * 2
        return _target_allocation_action(
            observation, target=target, scheduler_counts=counts
        )
    if policy == "max_pressure":
        position = _inventory_position(observation)
        demand = EXPECTED_WEEKLY_DEMAND * np.asarray(
            (observation.predicted_share_c, 1.0 - observation.predicted_share_c)
        )
        pressure = demand - position
        difference = float(pressure[0] - pressure[1])
        if (
            observation.previous_action is not None
            and abs(difference) <= float(parameter)
        ):
            action = int(observation.previous_action)
            return action, (abs(difference),), (action,)
        positive = np.maximum(pressure, 0.0)
        share_c = (
            float(positive[0] / positive.sum())
            if float(positive.sum()) > 1e-12
            else 0.5
        )
        rows = [
            (action, (abs(float(counts[action][0]) / 3.0 - share_c),))
            for action in range(4)
        ]
        return _choose_lexicographic(rows)
    if policy == "min_cost_flow":
        shares = _forecast_shares(
            observation.belief_c,
            regime_persistence=float(regime_persistence),
            dominant_share=float(dominant_share),
            horizon=parameter,
        )
        target_c = EXPECTED_WEEKLY_DEMAND * sum(shares)
        target = (target_c, EXPECTED_WEEKLY_DEMAND * parameter - target_c)
        return _target_allocation_action(
            observation, target=target, scheduler_counts=counts
        )
    if policy == "belief_mpc":
        return _deterministic_mpc_action(
            observation,
            horizon=parameter,
            scheduler_counts=counts,
            regime_persistence=float(regime_persistence),
            dominant_share=float(dominant_share),
        )
    if policy == "belief_dp":
        return _belief_dp_action(
            observation,
            horizon=parameter,
            scheduler_counts=counts,
            regime_persistence=float(regime_persistence),
            dominant_share=float(dominant_share),
        )
    raise ValueError(f"unknown state-rich policy: {policy}")


def state_rich_calendar(
    *,
    skeleton: Mapping[str, Any],
    scheduler: Mapping[str, Sequence[str]],
    config: StateRichConfiguration,
    regime_persistence: float,
    dominant_share: float,
    observation_mode: str = "real",
    action_overrides: Sequence[int] | None = None,
    action_prefix: Sequence[int] | None = None,
    initial_belief_c: float = 0.5,
) -> tuple[tuple[int, ...], tuple[StateRichDecision, ...]]:
    """Replay state and produce a policy or execution-supplied calendar."""
    weeks = int(skeleton["decision_weeks"])
    if action_overrides is not None:
        action_overrides = tuple(int(value) for value in action_overrides)
        if len(action_overrides) != weeks or any(
            value not in range(4) for value in action_overrides
        ):
            raise ValueError(
                f"action_overrides must contain {weeks} actions in {{0,1,2,3}}"
            )
    if action_prefix is not None:
        action_prefix = tuple(int(value) for value in action_prefix)
        if not 0 < len(action_prefix) < weeks or any(
            value not in range(4) for value in action_prefix
        ):
            raise ValueError(
                f"action_prefix must contain 1..{weeks - 1} actions in {{0,1,2,3}}"
            )
    if action_overrides is not None and action_prefix is not None:
        raise ValueError("action_overrides and action_prefix are mutually exclusive")
    start = float(skeleton["decision_start"])
    score_time = float(skeleton["score_time"])
    order_times = np.asarray(skeleton["order_times"], dtype=float)
    quantities = np.asarray(skeleton["order_quantities"], dtype=float)
    products = tuple(str(value) for value in skeleton["order_products"])
    product_index = {product_id: index for index, product_id in enumerate(PRODUCTS)}
    requested_product = np.asarray(
        [product_index[value] for value in products], dtype=np.int8
    )
    inventory = np.asarray(skeleton["opening_inventory"], dtype=float).copy()
    pending = np.zeros(len(order_times), dtype=bool)
    created = np.zeros(len(order_times), dtype=bool)
    released = np.zeros(len(order_times), dtype=bool)
    oat = np.full(len(order_times), np.inf, dtype=float)
    actions: list[int] = []
    decisions: list[StateRichDecision] = []
    real_observation_history: list[StateRichObservation] = []

    # Reuse the audited label-only HMM solely to compute the strictly historical
    # belief state.  Its selected action is ignored.
    _unused, belief_rows = observable_calendar(
        request_times=order_times,
        request_products=products,
        decision_start=start,
        decision_weeks=weeks,
        policy_id="belief_extreme_v1",
        initial_action=1,
        regime_persistence=float(regime_persistence),
        dominant_share=float(dominant_share),
        initial_belief_c=float(initial_belief_c),
    )

    events: list[tuple[float, int, str, Any]] = []
    for week in range(weeks):
        # Strict half-open observation: same-time physical events are not seen.
        events.append((start + 168.0 * week, -1, "decision", week))
    release_slots = tuple(map(float, skeleton["release_slots"]))
    release_completion_slots = tuple(
        map(
            float,
            skeleton.get("release_completion_slots")
            or [time + 48.0 for time in release_slots],
        )
    )
    release_available = tuple(
        map(bool, skeleton.get("release_available") or [True] * len(release_slots))
    )
    if not (
        len(release_slots) == len(release_completion_slots) == len(release_available)
    ):
        raise ValueError("release vectors must have equal length")
    for release_index, time in enumerate(release_slots):
        events.append((float(time), 0, "release", int(release_index)))
    batch_arrivals = [
        (float(time), int(week), int(position))
        for time, week, position in skeleton["batch_arrivals"]
    ]
    for time, week, position in batch_arrivals:
        events.append((time, 1, "batch", (week, position)))
    for order_index, time in enumerate(order_times):
        events.append((float(time), 2, "demand", int(order_index)))

    contingent = np.asarray(
        skeleton.get("order_contingent") or [False] * len(order_times), dtype=bool
    )
    priority_order = np.lexsort(
        (
            np.arange(len(order_times), dtype=np.int64),
            order_times,
            quantities,
            1 - contingent.astype(np.uint8),
        )
    )
    for now, _priority, kind, payload in sorted(events):
        if kind == "decision":
            week = int(payload)
            locked = np.zeros(2, dtype=float)
            for arrival_time, target_week, position in batch_arrivals:
                if target_week >= week or arrival_time < float(now) - 1e-12:
                    continue
                label = tuple(scheduler[str(actions[target_week])])[int(position)]
                locked[product_index[label]] += BATCH_QUANTITY
            backlog = np.zeros(2, dtype=float)
            backlog_orders = np.zeros(2, dtype=int)
            max_age = np.zeros(2, dtype=float)
            in_flight = np.zeros(2, dtype=float)
            for order_index in range(len(order_times)):
                product_id = int(requested_product[order_index])
                if pending[order_index]:
                    backlog[product_id] += quantities[order_index]
                    backlog_orders[product_id] += 1
                    max_age[product_id] = max(
                        max_age[product_id], float(now) - order_times[order_index]
                    )
                elif released[order_index] and oat[order_index] >= float(now) - 1e-12:
                    in_flight[product_id] += quantities[order_index]
            belief_row = belief_rows[week]
            raw_observation = {
                "week": week,
                "decision_time": float(now),
                "on_hand": inventory.tolist(),
                "locked_pipeline": locked.tolist(),
                "backlog_quantity": backlog.tolist(),
                "backlog_orders": backlog_orders.tolist(),
                "max_backlog_age": max_age.tolist(),
                "in_flight_quantity": in_flight.tolist(),
                "belief_c": float(belief_row.belief_c),
                "predicted_share_c": float(belief_row.predicted_share_c),
                "previous_action": None if not actions else int(actions[-1]),
                "remaining_decisions": weeks - week,
            }
            observation = StateRichObservation(
                week=int(raw_observation["week"]),
                decision_time=float(raw_observation["decision_time"]),
                on_hand=tuple(raw_observation["on_hand"]),
                locked_pipeline=tuple(raw_observation["locked_pipeline"]),
                backlog_quantity=tuple(raw_observation["backlog_quantity"]),
                backlog_orders=tuple(raw_observation["backlog_orders"]),
                max_backlog_age=tuple(raw_observation["max_backlog_age"]),
                in_flight_quantity=tuple(raw_observation["in_flight_quantity"]),
                belief_c=float(raw_observation["belief_c"]),
                predicted_share_c=float(raw_observation["predicted_share_c"]),
                previous_action=raw_observation["previous_action"],
                remaining_decisions=int(raw_observation["remaining_decisions"]),
                observation_sha256=_digest(raw_observation),
            )
            real_observation_history.append(observation)
            if observation_mode == "stale_t2":
                donor = real_observation_history[max(0, week - 2)]
                stale_payload = _observation_payload(observation)
                for key in (
                    "on_hand",
                    "locked_pipeline",
                    "backlog_quantity",
                    "backlog_orders",
                    "max_backlog_age",
                    "in_flight_quantity",
                    "belief_c",
                    "predicted_share_c",
                ):
                    stale_payload[key] = getattr(donor, key)
                observation = replace(
                    observation,
                    **stale_payload,
                    observation_sha256=_digest(stale_payload),
                )
            elif observation_mode == "stale_operational_current_belief":
                donor = real_observation_history[max(0, week - 2)]
                stale_payload = _observation_payload(observation)
                for key in (
                    "on_hand",
                    "locked_pipeline",
                    "backlog_quantity",
                    "backlog_orders",
                    "max_backlog_age",
                    "in_flight_quantity",
                ):
                    stale_payload[key] = getattr(donor, key)
                observation = replace(
                    observation,
                    **stale_payload,
                    observation_sha256=_digest(stale_payload),
                )
            elif observation_mode == "current_operational_stale_belief":
                donor = real_observation_history[max(0, week - 2)]
                stale_payload = _observation_payload(observation)
                stale_payload["belief_c"] = donor.belief_c
                stale_payload["predicted_share_c"] = donor.predicted_share_c
                observation = replace(
                    observation,
                    **stale_payload,
                    observation_sha256=_digest(stale_payload),
                )
            elif observation_mode == "no_state":
                no_state_payload = _observation_payload(observation)
                for key in (
                    "on_hand",
                    "locked_pipeline",
                    "backlog_quantity",
                    "max_backlog_age",
                    "in_flight_quantity",
                ):
                    no_state_payload[key] = (0.0, 0.0)
                no_state_payload["backlog_orders"] = (0, 0)
                no_state_payload["belief_c"] = 0.5
                no_state_payload["predicted_share_c"] = 0.5
                observation = replace(
                    observation,
                    **no_state_payload,
                    observation_sha256=_digest(no_state_payload),
                )
            elif observation_mode == "belief_only":
                belief_only_payload = _observation_payload(observation)
                for key in (
                    "on_hand",
                    "locked_pipeline",
                    "backlog_quantity",
                    "max_backlog_age",
                    "in_flight_quantity",
                ):
                    belief_only_payload[key] = (0.0, 0.0)
                belief_only_payload["backlog_orders"] = (0, 0)
                observation = replace(
                    observation,
                    **belief_only_payload,
                    observation_sha256=_digest(belief_only_payload),
                )
            elif observation_mode == "operational_only":
                operational_only_payload = _observation_payload(observation)
                operational_only_payload["belief_c"] = 0.5
                operational_only_payload["predicted_share_c"] = 0.5
                observation = replace(
                    observation,
                    **operational_only_payload,
                    observation_sha256=_digest(operational_only_payload),
                )
            elif observation_mode == "swapped_state":
                previous_action = observation.previous_action
                observation = replace(
                    swap_product_channels(observation),
                    previous_action=previous_action,
                )
                swapped_payload = _observation_payload(observation)
                observation = replace(
                    observation,
                    observation_sha256=_digest(swapped_payload),
                )
            elif observation_mode == "swapped_operational_current_belief":
                belief_c = observation.belief_c
                predicted_share_c = observation.predicted_share_c
                previous_action = observation.previous_action
                observation = replace(
                    swap_product_channels(observation),
                    belief_c=belief_c,
                    predicted_share_c=predicted_share_c,
                    previous_action=previous_action,
                )
                swapped_payload = _observation_payload(observation)
                observation = replace(
                    observation,
                    observation_sha256=_digest(swapped_payload),
                )
            elif observation_mode != "real":
                raise ValueError(f"unknown observation mode: {observation_mode}")
            if action_prefix is not None and week < len(action_prefix):
                action = int(action_prefix[week])
                objective = ()
                tied = (action,)
            elif action_overrides is None:
                action, objective, tied = choose_state_rich_action(
                    observation,
                    config,
                    scheduler=scheduler,
                    regime_persistence=float(regime_persistence),
                    dominant_share=float(dominant_share),
                )
            else:
                action = int(action_overrides[week])
                objective = ()
                tied = (action,)
            actions.append(int(action))
            decisions.append(
                StateRichDecision(
                    observation=observation,
                    action=int(action),
                    objective=tuple(float(value) for value in objective),
                    tied_actions=tuple(int(value) for value in tied),
                )
            )
            continue
        if kind == "batch":
            week, position = payload
            label = tuple(scheduler[str(actions[int(week)])])[int(position)]
            inventory[product_index[label]] += BATCH_QUANTITY
            continue
        if kind == "demand":
            order_index = int(payload)
            pending[order_index] = True
            created[order_index] = True
            continue
        release_index = int(payload)
        if not release_available[release_index]:
            continue
        chosen = None
        for order_index in priority_order:
            if not created[order_index] or not pending[order_index]:
                continue
            product_id = int(requested_product[order_index])
            if inventory[product_id] + 1e-9 >= quantities[order_index]:
                chosen = int(order_index)
                break
        if chosen is not None:
            product_id = int(requested_product[chosen])
            inventory[product_id] -= quantities[chosen]
            pending[chosen] = False
            released[chosen] = True
            oat[chosen] = float(release_completion_slots[release_index])

    if len(actions) != weeks or any(action not in range(4) for action in actions):
        raise AssertionError("state-rich controller did not emit a full calendar")
    if any(float(decision.observation.decision_time) > score_time for decision in decisions):
        raise AssertionError("decision after score horizon")
    return tuple(actions), tuple(decisions)


def decision_rows(decisions: Sequence[StateRichDecision]) -> list[dict[str, Any]]:
    return [
        {
            "observation": asdict(decision.observation),
            "action": int(decision.action),
            "objective": list(decision.objective),
            "tied_actions": list(decision.tied_actions),
        }
        for decision in decisions
    ]
