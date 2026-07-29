"""Observable policies for Program O's pre-learner H_obs gate.

The primary policy in this module deliberately consumes a very small
information set: product labels of requests that occurred strictly before the
weekly decision timestamp.  It cannot access the latent regime, the tape seed,
future requests, oracle calendars, or score-time summaries.

This module only maps an observation history to an eight-action calendar.  The
calendar must still be rolled through the certified full-DES transducer and
direct replay before it can contribute scientific evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence


PRODUCT_C = "P_C"
PRODUCT_H = "P_H"
PRODUCTS = (PRODUCT_C, PRODUCT_H)

POLICY_IDS = (
    "belief_extreme_v1",
    "belief_quota_v1",
    "previous_week_majority_v1",
    "no_history_v1",
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ObservableDecision:
    week: int
    decision_time: float
    prior_request_products: tuple[str, ...]
    prior_request_times: tuple[float, ...]
    prior_actions: tuple[int, ...]
    belief_c: float
    predicted_share_c: float
    action: int
    tie_state: bool
    observation_sha256: str


def posterior_after_week(
    prior_c: float,
    labels: Sequence[str],
    *,
    dominant_share: float,
) -> float:
    """Exact two-state posterior after one week's observed request labels."""
    if not 0.0 < float(prior_c) < 1.0:
        raise ValueError("prior_c must be strictly between zero and one")
    if not 0.5 < float(dominant_share) < 1.0:
        raise ValueError("dominant_share must be in (0.5, 1)")
    if any(label not in PRODUCTS for label in labels):
        raise ValueError("unknown product label")
    count_c = sum(label == PRODUCT_C for label in labels)
    count_h = len(labels) - count_c
    log_odds = math.log(float(prior_c) / (1.0 - float(prior_c)))
    log_odds += (count_c - count_h) * math.log(
        float(dominant_share) / (1.0 - float(dominant_share))
    )
    if log_odds >= 0.0:
        inverse = math.exp(-log_odds)
        return 1.0 / (1.0 + inverse)
    odds = math.exp(log_odds)
    return odds / (1.0 + odds)


def transition_belief(prior_c: float, *, regime_persistence: float) -> float:
    """Predict the next regime after the symmetric two-state transition."""
    rho = float(regime_persistence)
    if not 0.5 <= rho < 1.0:
        raise ValueError("regime_persistence must be in [0.5, 1)")
    return rho * float(prior_c) + (1.0 - rho) * (1.0 - float(prior_c))


def predicted_request_share_c(belief_c: float, *, dominant_share: float) -> float:
    share = float(dominant_share)
    return float(belief_c) * share + (1.0 - float(belief_c)) * (1.0 - share)


def _nearest_action(value: float) -> tuple[int, bool]:
    distances = [abs(float(value) - action) for action in range(4)]
    best = min(distances)
    ties = [action for action, distance in enumerate(distances) if abs(distance-best) <= 1e-12]
    return min(ties), len(ties) > 1


def choose_action(
    policy_id: str,
    *,
    predicted_share_c: float,
    previous_week_labels: Sequence[str],
    initial_action: int,
    week: int,
) -> tuple[int, bool]:
    """Choose one of four production-right allocations without future data."""
    if policy_id not in POLICY_IDS:
        raise ValueError(f"unknown policy_id: {policy_id}")
    if int(initial_action) not in range(4):
        raise ValueError("initial_action must be in {0,1,2,3}")
    if int(week) == 0:
        return int(initial_action), False
    if policy_id == "belief_extreme_v1":
        if abs(float(predicted_share_c) - 0.5) <= 1e-12:
            return 0, True
        return (3 if float(predicted_share_c) > 0.5 else 0), False
    if policy_id == "belief_quota_v1":
        return _nearest_action(3.0 * float(predicted_share_c))
    if policy_id == "previous_week_majority_v1":
        count_c = sum(label == PRODUCT_C for label in previous_week_labels)
        count_h = len(previous_week_labels) - count_c
        if count_c == count_h:
            return 1, True
        return (2 if count_c > count_h else 1), False
    return int(initial_action), False


def observable_calendar(
    *,
    request_times: Sequence[float],
    request_products: Sequence[str],
    decision_start: float,
    decision_weeks: int,
    policy_id: str,
    initial_action: int,
    regime_persistence: float = 0.75,
    dominant_share: float = 0.90,
    history_delay_weeks: int = 0,
    swap_observed_labels: bool = False,
    ignore_history: bool = False,
) -> tuple[tuple[int, ...], tuple[ObservableDecision, ...]]:
    """Build an action calendar from events strictly before each decision.

    ``history_delay_weeks=1`` is the frozen extra-week-delay placebo.  Missing
    weeks propagate the Markov prior without a likelihood update.
    """
    weeks = int(decision_weeks)
    if weeks <= 0 or weeks > 8:
        raise ValueError("decision_weeks must be in 1..8")
    if len(request_times) != len(request_products):
        raise ValueError("request time/product vectors must have equal length")
    if int(history_delay_weeks) < 0:
        raise ValueError("history_delay_weeks must be non-negative")
    events = sorted(
        (float(time), str(product))
        for time, product in zip(request_times, request_products, strict=True)
    )
    if any(product not in PRODUCTS for _, product in events):
        raise ValueError("unknown product label")

    action_history: list[int] = []
    decisions: list[ObservableDecision] = []
    for week in range(weeks):
        decision_time = float(decision_start) + 168.0 * week
        latest_observed_week = week - 1 - int(history_delay_weeks)
        cutoff = float(decision_start) + 168.0 * (latest_observed_week + 1)
        visible = [
            (time, product)
            for time, product in events
            if time < decision_time - 1e-12 and time < cutoff - 1e-12
        ]
        labels = [product for _, product in visible]
        if swap_observed_labels:
            labels = [PRODUCT_H if label == PRODUCT_C else PRODUCT_C for label in labels]
        if ignore_history:
            labels = []

        belief = 0.5
        previous_labels: tuple[str, ...] = ()
        for prior_week in range(week):
            week_start = float(decision_start) + 168.0 * prior_week
            week_end = week_start + 168.0
            observed_this_week = (
                not ignore_history and prior_week <= latest_observed_week
            )
            week_labels = tuple(
                product
                for time, product in zip(
                    [row[0] for row in visible], labels, strict=True
                )
                if week_start <= time < week_end
            ) if observed_this_week else ()
            if week_labels:
                belief = posterior_after_week(
                    belief, week_labels, dominant_share=float(dominant_share)
                )
            belief = transition_belief(
                belief, regime_persistence=float(regime_persistence)
            )
            if prior_week == week - 1:
                previous_labels = week_labels

        share_c = predicted_request_share_c(
            belief, dominant_share=float(dominant_share)
        )
        action, tied = choose_action(
            policy_id,
            predicted_share_c=share_c,
            previous_week_labels=previous_labels,
            initial_action=int(initial_action),
            week=week,
        )
        observation_payload = {
            "week": week,
            "decision_time": decision_time,
            "prior_request_times": [time for time, _ in visible],
            "prior_request_products": labels,
            "prior_actions": action_history,
            "belief_c": belief,
            "predicted_share_c": share_c,
        }
        decisions.append(
            ObservableDecision(
                week=week,
                decision_time=decision_time,
                prior_request_products=tuple(labels),
                prior_request_times=tuple(time for time, _ in visible),
                prior_actions=tuple(action_history),
                belief_c=float(belief),
                predicted_share_c=float(share_c),
                action=int(action),
                tie_state=bool(tied),
                observation_sha256=_digest(observation_payload),
            )
        )
        action_history.append(int(action))
    return tuple(action_history), tuple(decisions)


def calendar_index(calendar: Iterable[int]) -> int:
    values = tuple(int(value) for value in calendar)
    if not values or any(value not in range(4) for value in values):
        raise ValueError("calendar must contain actions in {0,1,2,3}")
    return sum(value * 4 ** (len(values) - position - 1) for position, value in enumerate(values))


def decision_audit_rows(decisions: Sequence[ObservableDecision]) -> list[dict[str, Any]]:
    return [asdict(decision) for decision in decisions]
