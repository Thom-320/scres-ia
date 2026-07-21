"""Burned-only retained-context cold-start discovery for Program Q physics.

The module separates physical state from decision knowledge.  Each campaign
gets a fresh full-DES skeleton; all arms replay that same skeleton and differ
only in the initial belief supplied to the identical belief-MPC controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from supply_chain.program_o_full_des import PRODUCTS
from supply_chain.program_o_full_des_transducer import (
    FullDESSkeleton,
    extract_full_des_skeleton,
    simulate_full_des_frontier,
)
from supply_chain.program_o_hobs import posterior_after_week, transition_belief
from supply_chain.program_o_state_rich import StateRichConfiguration, state_rich_calendar


ARMS = (
    "oracle_initial_context",
    "retained_posterior",
    "reset_posterior_0p5",
    "shuffled_posterior",
    "wrong_posterior",
)
RESOURCE_KEYS = (
    "gross_policy_batch_slots",
    "gross_production_quantity",
    "charged_daily_dispatch_slots",
    "charged_downstream_vehicle_hours",
)


@dataclass(frozen=True)
class CampaignSpec:
    history_root: int
    campaign_index: int
    kappa: float
    initial_regime: str
    last_regime: str
    skeleton: FullDESSkeleton

    @property
    def tape_seed(self) -> int:
        return int(self.history_root) * 100 + int(self.campaign_index)


def cross_campaign_transition(belief_c: float, kappa: float) -> float:
    if not 0.5 <= float(kappa) <= 1.0:
        raise ValueError("kappa must be in [0.5, 1]")
    return float(kappa) * float(belief_c) + (1.0 - float(kappa)) * (
        1.0 - float(belief_c)
    )


def campaign_final_posterior(
    *,
    initial_belief_c: float,
    order_products: Sequence[str],
    regime_persistence: float,
    dominant_share: float,
) -> float:
    labels = tuple(map(str, order_products))
    if len(labels) % 6:
        raise ValueError("campaign must contain six request labels per week")
    belief = float(initial_belief_c)
    weeks = len(labels) // 6
    for week in range(weeks):
        start = 6 * week
        belief = posterior_after_week(
            belief,
            labels[start : start + 6],
            dominant_share=float(dominant_share),
        )
        if week + 1 < weeks:
            belief = transition_belief(
                belief, regime_persistence=float(regime_persistence)
            )
    return float(belief)


def build_campaign_history(
    *,
    history_root: int,
    campaigns: int,
    kappa: float,
    scheduler: Mapping[str, Sequence[str]],
    regime_persistence: float,
    dominant_share: float,
) -> tuple[CampaignSpec, ...]:
    if int(campaigns) < 2:
        raise ValueError("at least two campaigns are required")
    rng = np.random.default_rng(int(history_root) ^ 0x5A17C0DE)
    initial = PRODUCTS[int(rng.integers(0, 2))]
    output: list[CampaignSpec] = []
    for campaign_index in range(int(campaigns)):
        tape_seed = int(history_root) * 100 + campaign_index
        skeleton, sim = extract_full_des_skeleton(
            seed=tape_seed,
            scheduler=scheduler,
            regime_persistence=float(regime_persistence),
            dominant_share=float(dominant_share),
            downstream_freight_physics_mode="fixed_clock_physical_v1",
            initial_regime=initial,
        )
        regimes = tuple(map(str, sim.program_o_tape["regimes"]))
        output.append(
            CampaignSpec(
                history_root=int(history_root),
                campaign_index=campaign_index,
                kappa=float(kappa),
                initial_regime=str(initial),
                last_regime=regimes[-1],
                skeleton=skeleton,
            )
        )
        stays = bool(rng.random() < float(kappa))
        initial = regimes[-1] if stays else PRODUCTS[1 - PRODUCTS.index(regimes[-1])]
    return tuple(output)


def retained_prior_path(
    campaigns: Sequence[CampaignSpec],
    *,
    regime_persistence: float,
    dominant_share: float,
) -> tuple[float, ...]:
    priors: list[float] = []
    prior = 0.5
    for campaign in campaigns:
        priors.append(float(prior))
        posterior = campaign_final_posterior(
            initial_belief_c=prior,
            order_products=campaign.skeleton.order_products,
            regime_persistence=float(regime_persistence),
            dominant_share=float(dominant_share),
        )
        prior = cross_campaign_transition(posterior, campaign.kappa)
    return tuple(priors)


def _prefix_service_loss(trace: Mapping[str, Any], cutoff: float) -> float:
    total = 0.0
    for order in trace["orders"]:
        opt = float(order["OPTj"])
        if opt >= float(cutoff) - 1e-12:
            continue
        oat = order.get("OATj")
        end = float(cutoff) if oat is None else min(float(cutoff), float(oat))
        total += float(order["quantity"]) * max(0.0, end - (opt + 48.0))
    return float(total)


def evaluate_campaign_prior(
    *,
    campaign: CampaignSpec,
    arm: str,
    initial_belief_c: float,
    scheduler: Mapping[str, Sequence[str]],
    regime_persistence: float,
    dominant_share: float,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    prior = min(1.0 - 1e-6, max(1e-6, float(initial_belief_c)))
    calendar, decisions = state_rich_calendar(
        skeleton=campaign.skeleton.as_dict(),
        scheduler=scheduler,
        config=StateRichConfiguration("belief_mpc", 3),
        regime_persistence=float(regime_persistence),
        dominant_share=float(dominant_share),
        initial_belief_c=prior,
    )
    trace: dict[str, Any] = {}
    panel = simulate_full_des_frontier(
        skeleton=campaign.skeleton,
        scheduler=scheduler,
        calendars=np.asarray([calendar], dtype=np.uint8),
        trace_out=trace,
    )
    row = {key: float(values[0]) for key, values in panel.items()}
    cutoff = float(campaign.skeleton.decision_start) + 336.0
    return {
        "history_root": campaign.history_root,
        "campaign_index": campaign.campaign_index,
        "tape_seed": campaign.tape_seed,
        "kappa": campaign.kappa,
        "arm": arm,
        "initial_regime": campaign.initial_regime,
        "last_regime": campaign.last_regime,
        "initial_belief_c": prior,
        "calendar": list(map(int, calendar)),
        "initial_action": int(calendar[0]),
        "initial_observation_sha256": decisions[0].observation.observation_sha256,
        "skeleton_sha256": campaign.skeleton.skeleton_sha256,
        "prefix_state_hash": campaign.skeleton.prefix_state_hash,
        "ret_visible": row["ret_visible"],
        "service_loss_auc_first_two_weeks": _prefix_service_loss(trace, cutoff),
        "service_loss_auc": row["service_loss_auc"],
        "worst_product_fill": row["worst_product_fill"],
        **{key: row[key] for key in RESOURCE_KEYS},
    }


def arm_priors(
    *,
    histories: Sequence[Sequence[CampaignSpec]],
    regime_persistence: float,
    dominant_share: float,
) -> dict[str, list[tuple[float, ...]]]:
    retained = [
        retained_prior_path(
            history,
            regime_persistence=float(regime_persistence),
            dominant_share=float(dominant_share),
        )
        for history in histories
    ]
    output: dict[str, list[tuple[float, ...]]] = {arm: [] for arm in ARMS}
    for history_index, history in enumerate(histories):
        output["retained_posterior"].append(retained[history_index])
        output["reset_posterior_0p5"].append(tuple(0.5 for _ in history))
        output["shuffled_posterior"].append(retained[(history_index + 1) % len(histories)])
        oracle = tuple(
            1.0 - 1e-6 if row.initial_regime == "P_C" else 1e-6 for row in history
        )
        output["oracle_initial_context"].append(oracle)
        output["wrong_posterior"].append(tuple(1.0 - value for value in oracle))
    return output
