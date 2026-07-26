"""Executable primitives for the Q-R1 matched-retention factorial v4.

This module contains only contract-level construction and evaluation helpers.
It does not open development roots, train a learner, select a checkpoint, or
write a freeze/opening receipt.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import time
from typing import Any

from scripts.run_q_r1_successor_abc import fixed_theta_belief
from supply_chain.program_t_joint_belief import ExactJointBelief
from supply_chain.q_r1_comparator_v2 import (
    ComparatorV2Config,
    comparator_v2_calendar,
)
from supply_chain.q_r1_retained_learning import (
    PhysicalCampaignState,
    evaluate_calendar,
)
from supply_chain.retained_context_discovery import (
    CampaignSpec,
    retained_prior_path,
)


REQUIRED_SERVICE_FIELDS = (
    "worst_product_fill",
    "unresolved_orders",
    "unresolved_quantity",
    "lost_orders",
    "lost_quantity",
    "service_loss_auc",
)
REQUIRED_PRIMARY_FIELDS = (
    "early_ret_complete_cohort",
    "early_ret_visible",
    "ret_visible",
    "ret_full",
)


def campaign_as_structured_state(campaign: CampaignSpec) -> PhysicalCampaignState:
    """Give the structured comparator the identical immutable physical campaign."""
    persistence_mode = {
        0.5: "iid",
        0.75: "persistent_0p75",
        0.9: "persistent_0p90",
    }.get(float(campaign.kappa))
    if persistence_mode is None:
        raise ValueError("factorial kappa must be one of 0.50, 0.75, 0.90")
    return PhysicalCampaignState(
        history_root=int(campaign.history_root),
        campaign_index=int(campaign.campaign_index),
        persistence_mode=persistence_mode,
        theta=(0.90, 0.90),
        initial_regime=str(campaign.initial_regime),
        skeleton=campaign.skeleton,
    )


def matched_prior_paths(
    histories: Sequence[Sequence[CampaignSpec]],
    *,
    regime_persistence: float = 0.90,
    dominant_share: float = 0.90,
) -> tuple[tuple[float, ...], ...]:
    """Compute the one policy-independent prior path shared by learner and MPC."""
    return tuple(
        retained_prior_path(
            history,
            regime_persistence=float(regime_persistence),
            dominant_share=float(dominant_share),
        )
        for history in histories
    )


def structured_pair_rows(
    *,
    histories: Sequence[Sequence[CampaignSpec]],
    prior_paths: Sequence[Sequence[float]],
    scheduler: Mapping[str, Sequence[str]],
    config: ComparatorV2Config,
    calendar_builder: Callable[..., tuple[tuple[int, ...], dict[str, object]]] = (
        comparator_v2_calendar
    ),
    calendar_evaluator: Callable[..., dict[str, Any]] = evaluate_calendar,
    cache: dict[
        tuple[str, str, str],
        tuple[tuple[int, ...], dict[str, object], dict[str, Any]],
    ]
    | None = None,
    per_calendar_hard_cap_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """Evaluate retained/reset structured arms on matched physical histories."""
    if len(histories) != len(prior_paths):
        raise ValueError("histories and prior_paths must have equal length")
    rows: list[dict[str, Any]] = []
    for history, priors in zip(histories, prior_paths, strict=True):
        if len(history) != len(priors):
            raise ValueError("each prior path must match its history")
        for campaign, retained_prior in zip(history, priors, strict=True):
            structured_campaign = campaign_as_structured_state(campaign)
            for arm, prior in (
                ("structured_reset", 0.5),
                ("structured_retained", float(retained_prior)),
            ):
                key = (
                    campaign.skeleton.skeleton_sha256,
                    float(prior).hex(),
                    config.config_id,
                )
                cached = None if cache is None else cache.get(key)
                if cached is None:
                    started = time.perf_counter()
                    belief: ExactJointBelief = fixed_theta_belief(prior)
                    calendar, diagnostics = calendar_builder(
                        campaign=structured_campaign,
                        belief=belief,
                        scheduler=scheduler,
                        config=config,
                    )
                    metrics = calendar_evaluator(
                        campaign=structured_campaign,
                        calendar=calendar,
                        scheduler=scheduler,
                    )
                    planning_elapsed = time.perf_counter() - started
                    if (
                        per_calendar_hard_cap_seconds is not None
                        and planning_elapsed > float(per_calendar_hard_cap_seconds)
                    ):
                        raise RuntimeError("STOP_COMPUTE_BUDGET_PREDECLARED")
                    if cache is not None:
                        cache[key] = (calendar, diagnostics, metrics)
                    cache_hit = False
                else:
                    calendar, diagnostics, metrics = cached
                    planning_elapsed = 0.0
                    cache_hit = True
                missing = (
                    set(REQUIRED_PRIMARY_FIELDS)
                    | set(REQUIRED_SERVICE_FIELDS)
                ) - set(metrics)
                if missing:
                    raise RuntimeError(
                        f"structured comparator omitted mandatory fields: {sorted(missing)}"
                    )
                rows.append(
                    {
                        "history_root": int(campaign.history_root),
                        "campaign_index": int(campaign.campaign_index),
                        "kappa": float(campaign.kappa),
                        "arm": arm,
                        "explicit_prior": float(prior),
                        "calendar": list(map(int, calendar)),
                        "skeleton_sha256": campaign.skeleton.skeleton_sha256,
                        "prefix_state_hash": campaign.skeleton.prefix_state_hash,
                        "comparator_config_id": config.config_id,
                        "comparator_diagnostics": diagnostics,
                        "structured_compute_seconds": planning_elapsed,
                        "structured_cache_hit": cache_hit,
                        **metrics,
                    }
                )
    return rows
