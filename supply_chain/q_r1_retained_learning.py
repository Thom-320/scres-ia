"""Q-R1 burned discovery primitives for causal cross-campaign retention.

Physical campaign state is immutable and shared across arms.  Knowledge state
is the only treatment.  Three estimands are kept separate:

``prefix_natural_replanning``
    The retained/reset treatment chooses two actions, after which the same
    reset-belief MPC policy replans on the physical state actually reached by
    each prefix.
``sustained_control``
    Each retained/reset controller acts for the full campaign.
``historical_splice``
    The frozen legacy construction concatenates an arm prefix with a reset
    calendar planned on another trajectory.  It is diagnostic only.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import time
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from supply_chain.program_o_full_des import PRODUCTS
from supply_chain.program_o_full_des_transducer import (
    FullDESSkeleton,
    extract_full_des_skeleton,
    simulate_full_des_frontier,
)
from supply_chain.program_t_full_des_mpc import (
    FullDEST0Config,
    choose_ret_transducer_action,
    joint_belief_ret_transducer_calendar,
)
from supply_chain.program_t_joint_belief import (
    THETA_GRID,
    ExactJointBelief,
    weekly_product_counts,
)
from supply_chain.ret_thesis import (
    compute_order_level_ret_excel_request_snapshot_ledger,
)
from supply_chain.program_o_state_rich import StateRichConfiguration, state_rich_calendar


PERSISTENCE_MODES: Mapping[str, float] = {
    "iid": 1.0 / len(THETA_GRID),
    "persistent_0p75": 0.75,
    "persistent_0p90": 0.90,
}
ARMS = (
    "reset_exact_bayes_mpc",
    "retained_exact_bayes_mpc",
    "oracle_parameters_mpc",
    "shuffled_posterior_mpc",
    "wrong_posterior_mpc",
)
RESOURCE_KEYS = (
    "gross_policy_batch_slots",
    "gross_production_quantity",
    "charged_daily_dispatch_slots",
    "charged_downstream_vehicle_hours",
)
ACTUAL_RESOURCE_KEYS = (
    "actual_loaded_departures",
    "actual_payload",
    "actual_downstream_vehicle_hours",
)
ESTIMANDS = (
    "prefix_natural_replanning",
    "sustained_control",
    "historical_splice",
)


class BeliefProvider(Protocol):
    """Deployable cross-campaign belief interface."""

    def reset(self) -> None: ...
    def observe(self, observation: Any) -> None: ...
    def between_campaign_transition(self, persistence: float) -> Any: ...
    def predictive_distribution(self) -> Any: ...


class CampaignController(Protocol):
    """Common interface for structured and future learned controllers."""

    def reset(self, *, retain_knowledge: bool) -> None: ...
    def act(self, observation: Any) -> int: ...
    def observe(self, observation: Any) -> None: ...
    def export_knowledge(self) -> Any: ...


@dataclass(frozen=True)
class PhysicalCampaignState:
    history_root: int
    campaign_index: int
    persistence_mode: str
    theta: tuple[float, float]
    initial_regime: str
    skeleton: FullDESSkeleton

    @property
    def tape_seed(self) -> int:
        return int(self.history_root) * 100 + int(self.campaign_index)

    def reset_physical(self) -> "PhysicalCampaignState":
        """Return the same immutable physical state for a paired arm."""
        return self


@dataclass
class KnowledgeState:
    belief: ExactJointBelief
    source: str

    @classmethod
    def reset_knowledge(cls) -> "KnowledgeState":
        return cls(ExactJointBelief.uniform(), "reset")

    def retain_knowledge(self, persistence: float) -> "KnowledgeState":
        return KnowledgeState(
            self.belief.between_campaign_transition(float(persistence)),
            "retained",
        )


def transition_theta(
    theta_index: int, *, persistence: float, rng: np.random.Generator
) -> int:
    if rng.random() < float(persistence):
        return int(theta_index)
    alternatives = [index for index in range(len(THETA_GRID)) if index != theta_index]
    return int(alternatives[int(rng.integers(0, len(alternatives)))])


def build_parameter_history(
    *,
    history_root: int,
    campaigns: int,
    persistence_mode: str,
    scheduler: Mapping[str, Sequence[str]],
) -> tuple[PhysicalCampaignState, ...]:
    if persistence_mode not in PERSISTENCE_MODES:
        raise ValueError(f"unknown persistence mode: {persistence_mode}")
    if campaigns < 2:
        raise ValueError("at least two campaigns are required")
    persistence = PERSISTENCE_MODES[persistence_mode]
    rng = np.random.default_rng(int(history_root) ^ 0x51A7D00D)
    theta_index = int(rng.integers(0, len(THETA_GRID)))
    output: list[PhysicalCampaignState] = []
    for campaign_index in range(int(campaigns)):
        theta = THETA_GRID[theta_index]
        initial_regime = PRODUCTS[int(rng.integers(0, 2))]
        skeleton, _sim = extract_full_des_skeleton(
            seed=int(history_root) * 100 + campaign_index,
            scheduler=scheduler,
            regime_persistence=theta[0],
            dominant_share=theta[1],
            downstream_freight_physics_mode="fixed_clock_physical_v1",
            initial_regime=initial_regime,
        )
        output.append(
            PhysicalCampaignState(
                history_root=int(history_root),
                campaign_index=campaign_index,
                persistence_mode=persistence_mode,
                theta=theta,
                initial_regime=initial_regime,
                skeleton=skeleton,
            )
        )
        theta_index = transition_theta(
            theta_index, persistence=persistence, rng=rng
        )
    return tuple(output)


def retained_belief_path(
    campaigns: Sequence[PhysicalCampaignState],
) -> tuple[ExactJointBelief, ...]:
    if not campaigns:
        return ()
    persistence = PERSISTENCE_MODES[campaigns[0].persistence_mode]
    current = ExactJointBelief.uniform()
    output: list[ExactJointBelief] = []
    for campaign in campaigns:
        output.append(current.copy())
        counts = weekly_product_counts(
            order_times=campaign.skeleton.order_times,
            order_products=campaign.skeleton.order_products,
            decision_start=campaign.skeleton.decision_start,
            weeks=campaign.skeleton.decision_weeks,
        )
        posterior = current.copy()
        posterior.observe_campaign(counts)
        current = posterior.between_campaign_transition(persistence)
    return tuple(output)


def wrong_belief(belief: ExactJointBelief) -> ExactJointBelief:
    marginal = tuple(reversed(belief.theta_marginal))
    return ExactJointBelief.from_theta_marginal(marginal)


def common_continuation_calendar(
    arm_calendar: Sequence[int], reset_calendar: Sequence[int], *, decision_count: int = 2
) -> tuple[int, ...]:
    if len(arm_calendar) != len(reset_calendar):
        raise ValueError("calendar lengths differ")
    if not 0 < decision_count <= len(arm_calendar):
        raise ValueError("invalid cold-start decision count")
    return tuple(map(int, arm_calendar[:decision_count])) + tuple(
        map(int, reset_calendar[decision_count:])
    )


def controller_calendar_from_prefix(
    *,
    campaign: PhysicalCampaignState,
    treatment_prefix: Sequence[int],
    continuation_belief: ExactJointBelief,
    scheduler: Mapping[str, Sequence[str]],
    config: FullDEST0Config,
    history_transform: str = "real",
) -> tuple[tuple[int, ...], dict[str, object]]:
    """Apply one controller policy after a fixed prefix on the reached state.

    This is a policy continuation, not a calendar splice.  At every week the
    state-rich observation is recomputed from ``prefix`` and therefore reflects
    the inventory, backlog, and pipeline produced by the arm's actual prior
    actions.  The continuation belief is updated only with causal demand
    observations, which are common across paired arms.
    """
    decisions = tuple(map(int, treatment_prefix))
    if not 0 < len(decisions) < campaign.skeleton.decision_weeks:
        raise ValueError("treatment prefix must be shorter than the campaign")
    if any(action not in range(4) for action in decisions):
        raise ValueError("prefix contains an unavailable action")
    if history_transform not in {"real", "wrong_product", "shuffled"}:
        raise ValueError("unknown history transform")
    counts = list(
        weekly_product_counts(
            order_times=campaign.skeleton.order_times,
            order_products=campaign.skeleton.order_products,
            decision_start=campaign.skeleton.decision_start,
            weeks=campaign.skeleton.decision_weeks,
        )
    )
    if history_transform == "wrong_product":
        counts = [6 - count for count in counts]
    elif history_transform == "shuffled":
        counts = list(reversed(counts))

    posterior = continuation_belief.copy()
    prefix: list[int] = []
    diagnostics: list[dict[str, object]] = []
    started = time.perf_counter()
    for week in range(campaign.skeleton.decision_weeks):
        if week:
            posterior.observe_previous_week(counts[week - 1])
        if week < len(decisions):
            action = decisions[week]
            detail: dict[str, object] = {"treatment_action": True}
        else:
            probe = tuple(
                prefix
                + [0] * (campaign.skeleton.decision_weeks - len(prefix))
            )
            _calendar, rows = state_rich_calendar(
                skeleton=campaign.skeleton.as_dict(),
                scheduler=scheduler,
                config=StateRichConfiguration("belief_mpc", 1),
                regime_persistence=config.regime_persistence,
                dominant_share=config.dominant_share,
                action_overrides=probe,
            )
            observation = rows[week].observation
            action, action_detail = choose_ret_transducer_action(
                observation,
                base_skeleton=campaign.skeleton,
                prefix=prefix,
                scheduler=scheduler,
                config=config,
                joint_belief=posterior,
            )
            detail = {"treatment_action": False, **action_detail}
        prefix.append(int(action))
        diagnostics.append(
            {
                "week": week,
                "action": int(action),
                "posterior": posterior.as_dict(),
                **detail,
            }
        )
    return tuple(prefix), {
        "config_id": "natural_replanning_" + config.config_id,
        "treatment_decisions": len(decisions),
        "history_transform": history_transform,
        "online_ms": (time.perf_counter() - started) * 1000.0,
        "decisions": diagnostics,
    }


def _order_namespaces(trace: Mapping[str, Any]) -> list[SimpleNamespace]:
    rows = []
    for raw in trace["orders"]:
        row = dict(raw)
        row.setdefault("LTj", 48.0)
        row.setdefault("lost_time", None)
        row.setdefault("metrics_excluded", False)
        rows.append(SimpleNamespace(**row))
    return rows


def early_cohort_metrics(
    *, trace: Mapping[str, Any], decision_start: float, score_time: float
) -> dict[str, float]:
    return early_cohort_metrics_from_orders(
        orders=_order_namespaces(trace),
        decision_start=decision_start,
        score_time=score_time,
    )


def early_cohort_metrics_from_orders(
    *, orders: Sequence[Any], decision_start: float, score_time: float
) -> dict[str, float]:
    """Score visible and non-censorable versions of the two-week cohort.

    ``early_ret_visible`` is the unmodified workbook-visible ReT and therefore
    excludes lost or unresolved rows. ``early_ret_complete_cohort`` keeps the
    same generated cohort in the denominator and assigns zero to every omitted
    row.  Both are reported; neither is silently substituted for the other.
    """
    cutoff = float(decision_start) + 2.0 * 168.0
    cohort = [order for order in orders if float(order.OPTj) < cutoff - 1e-12]
    cohort_ids = {int(order.j) for order in cohort}
    ledger = compute_order_level_ret_excel_request_snapshot_ledger(
        orders, emit_order_ids=cohort_ids
    )
    completed = [order for order in cohort if order.OATj is not None and not order.lost]
    unresolved = [order for order in cohort if order.OATj is None and not order.lost]
    lost = [order for order in cohort if order.lost]
    service_loss = sum(
        float(order.quantity)
        * max(
            0.0,
            min(float(score_time), float(order.OATj) if order.OATj is not None else float(score_time))
            - (float(order.OPTj) + 48.0),
        )
        for order in cohort
    )
    fill: dict[str, float] = {}
    for product_id in PRODUCTS:
        demanded = sum(float(order.quantity) for order in cohort if order.requested_product_id == product_id)
        delivered = sum(float(order.quantity) for order in completed if order.requested_product_id == product_id)
        fill[product_id] = 1.0 if demanded <= 0.0 else delivered / demanded
    visible_values = np.asarray(ledger["ret_values"], dtype=float)
    complete_values = np.concatenate(
        (
            visible_values,
            np.zeros(len(cohort) - len(visible_values), dtype=float),
        )
    )
    visible_q10 = (
        float(np.quantile(visible_values, 0.10)) if len(visible_values) else 1.0
    )
    complete_q10 = (
        float(np.quantile(complete_values, 0.10)) if len(complete_values) else 1.0
    )
    tail_count = max(1, int(np.ceil(0.10 * len(complete_values)))) if len(complete_values) else 1
    complete_cvar10 = (
        float(np.sort(complete_values)[:tail_count].mean())
        if len(complete_values)
        else 1.0
    )
    visible_ret = float(ledger["mean_ret_excel"])
    complete_ret = (
        float(visible_values.sum() / len(cohort)) if cohort else 1.0
    )
    unresolved_quantity = sum(float(order.quantity) for order in unresolved)
    lost_quantity = sum(float(order.quantity) for order in lost)
    visible_quantity = sum(float(order.quantity) for order in completed)
    return {
        "early_ret_2w": visible_ret,
        "early_ret_visible": visible_ret,
        "early_ret_complete_cohort": complete_ret,
        "early_ret_visible_q10": visible_q10,
        "early_ret_complete_cohort_q10": complete_q10,
        "early_ret_complete_cohort_cvar10": complete_cvar10,
        "early_visible_rows": float(ledger["n_visible_rows"]),
        "early_omitted_rows": float(len(cohort) - ledger["n_visible_rows"]),
        "early_generated_orders": float(len(cohort)),
        "early_unresolved_orders": float(len(unresolved)),
        "early_unresolved_quantity": float(unresolved_quantity),
        "early_lost_orders": float(len(lost)),
        "early_lost_quantity": float(lost_quantity),
        "early_visible_quantity": float(visible_quantity),
        "early_service_loss_to_score": float(service_loss),
        "early_worst_product_fill": float(min(fill.values())),
        "early_fill_P_C": float(fill["P_C"]),
        "early_fill_P_H": float(fill["P_H"]),
    }


def evaluate_calendar(
    *,
    campaign: PhysicalCampaignState,
    calendar: Sequence[int],
    scheduler: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    trace: dict[str, Any] = {}
    panel = simulate_full_des_frontier(
        skeleton=campaign.skeleton,
        scheduler=scheduler,
        calendars=np.asarray([calendar], dtype=np.uint8),
        trace_out=trace,
    )
    row = {key: float(value[0]) for key, value in panel.items()}
    return {
        **early_cohort_metrics(
            trace=trace,
            decision_start=campaign.skeleton.decision_start,
            score_time=campaign.skeleton.score_time,
        ),
        "ret_visible": row["ret_visible"],
        "ret_full": row["ret_full"],
        "ret_visible_cvar10": row["ret_visible_cvar10"],
        "worst_product_fill": row["worst_product_fill"],
        "lost_orders": row["lost_orders"],
        "unresolved_orders": row["unresolved_orders"],
        "unresolved_quantity": row["unresolved_quantity"],
        "lost_quantity": row["lost_quantity"],
        "service_loss_auc": row["service_loss_auc"],
        "fill_P_C": row["fill_P_C"],
        "fill_P_H": row["fill_P_H"],
        "mass_residual": row["mass_residual"],
        "partition_residual": row["partition_residual"],
        **{key: row[key] for key in RESOURCE_KEYS},
        **{key: row[key] for key in ACTUAL_RESOURCE_KEYS},
    }


def controller_calendar(
    *,
    campaign: PhysicalCampaignState,
    belief: ExactJointBelief,
    scheduler: Mapping[str, Sequence[str]],
    config: FullDEST0Config,
    history_transform: str = "real",
) -> tuple[tuple[int, ...], dict[str, object]]:
    return joint_belief_ret_transducer_calendar(
        skeleton=campaign.skeleton,
        scheduler=scheduler,
        config=config,
        belief=belief,
        history_transform=history_transform,
    )


def controller_prefix(
    *,
    campaign: PhysicalCampaignState,
    belief: ExactJointBelief,
    scheduler: Mapping[str, Sequence[str]],
    config: FullDEST0Config,
    decisions: int = 2,
    history_transform: str = "real",
) -> tuple[tuple[int, ...], dict[str, object]]:
    """Plan only the treatment prefix needed by the cold-start estimand."""
    if not 0 < int(decisions) <= campaign.skeleton.decision_weeks:
        raise ValueError("invalid prefix decision count")
    if history_transform not in {"real", "wrong_product", "shuffled"}:
        raise ValueError("unknown history transform")
    counts = list(
        weekly_product_counts(
            order_times=campaign.skeleton.order_times,
            order_products=campaign.skeleton.order_products,
            decision_start=campaign.skeleton.decision_start,
            weeks=campaign.skeleton.decision_weeks,
        )
    )
    if history_transform == "wrong_product":
        counts = [6 - count for count in counts]
    elif history_transform == "shuffled":
        counts = list(reversed(counts))
    posterior = belief.copy()
    prefix: list[int] = []
    diagnostics = []
    for week in range(int(decisions)):
        if week:
            posterior.observe_previous_week(counts[week - 1])
        probe = tuple(
            prefix
            + [0] * (campaign.skeleton.decision_weeks - len(prefix))
        )
        _calendar, rows = state_rich_calendar(
            skeleton=campaign.skeleton.as_dict(),
            scheduler=scheduler,
            config=StateRichConfiguration("belief_mpc", 1),
            regime_persistence=config.regime_persistence,
            dominant_share=config.dominant_share,
            action_overrides=probe,
        )
        observation = rows[week].observation
        action, detail = choose_ret_transducer_action(
            observation,
            base_skeleton=campaign.skeleton,
            prefix=prefix,
            scheduler=scheduler,
            config=config,
            joint_belief=posterior,
        )
        prefix.append(action)
        diagnostics.append(
            {
                "week": week,
                "action": action,
                "posterior": posterior.as_dict(),
                **detail,
            }
        )
    return tuple(prefix), {"decisions": diagnostics}
