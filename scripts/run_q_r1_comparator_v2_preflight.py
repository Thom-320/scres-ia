#!/usr/bin/env python3
"""Burned-only convergence and ReT--service preflight for comparator v2."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from scripts.run_q_r1_successor_abc import fixed_theta_belief  # noqa: E402
from supply_chain.program_o_state_rich import (  # noqa: E402
    StateRichConfiguration,
    state_rich_calendar,
)
from supply_chain.q_r1_comparator_v2 import (  # noqa: E402
    ComparatorV2Config,
    NoFeasibleStructuredAction,
    PlanningKey,
    choose_comparator_v2_action,
    comparator_v2_calendar,
)
from supply_chain.q_r1_retained_learning import (  # noqa: E402
    PhysicalCampaignState,
    evaluate_calendar,
)
from supply_chain.retained_context_discovery import (  # noqa: E402
    arm_priors,
    build_campaign_history,
)


BURNED_LOW = 7_570_801
BURNED_HIGH = 7_570_824


def development_states(args: argparse.Namespace) -> list[tuple[PhysicalCampaignState, float]]:
    sched = scheduler()
    rows: list[tuple[PhysicalCampaignState, float]] = []
    for kappa in (0.75, 0.90):
        histories = [
            build_campaign_history(
                history_root=args.seed_start + offset,
                campaigns=args.campaigns,
                kappa=kappa,
                scheduler=sched,
                regime_persistence=0.90,
                dominant_share=0.90,
            )
            for offset in range(args.histories)
        ]
        priors = arm_priors(
            histories=histories,
            regime_persistence=0.90,
            dominant_share=0.90,
        )["retained_posterior"]
        for history_index, history in enumerate(histories):
            for campaign in history[1:]:
                rows.append(
                    (
                        PhysicalCampaignState(
                            history_root=campaign.history_root,
                            campaign_index=campaign.campaign_index,
                            persistence_mode=f"binary_{kappa}",
                            theta=(0.90, 0.90),
                            initial_regime=campaign.initial_regime,
                            skeleton=campaign.skeleton,
                        ),
                        float(priors[history_index][campaign.campaign_index]),
                    )
                )
    if len(rows) <= args.states:
        return rows
    indices = np.linspace(0, len(rows) - 1, args.states, dtype=int)
    return [rows[int(index)] for index in indices]


def family(args: argparse.Namespace, conditional_paths: int) -> list[ComparatorV2Config]:
    configs: list[ComparatorV2Config] = []
    for horizon in args.horizons:
        configs.append(
            ComparatorV2Config(
                horizon=horizon,
                conditional_paths=conditional_paths,
                mode="scenario",
                worst_product_floor=0.0,
                value_indifference_tolerance=args.value_indifference_tolerance,
                tie_breaker=args.tie_breaker,
            )
        )
        for floor in (() if args.scenario_only else args.service_floors):
            configs.append(
                ComparatorV2Config(
                    horizon=horizon,
                    conditional_paths=conditional_paths,
                    mode="constraint_aware",
                    worst_product_floor=floor,
                    service_statistic="expected",
                    value_indifference_tolerance=args.value_indifference_tolerance,
                    tie_breaker=args.tie_breaker,
                )
            )
    return configs


def first_observation(campaign: PhysicalCampaignState):
    _calendar, rows = state_rich_calendar(
        skeleton=campaign.skeleton.as_dict(),
        scheduler=scheduler(),
        config=StateRichConfiguration("belief_mpc", 1),
        regime_persistence=0.90,
        dominant_share=0.90,
        action_overrides=(0,) * campaign.skeleton.decision_weeks,
    )
    return rows[0].observation


def _signature_tuple(row: dict[str, object]) -> tuple[int, str, float, str]:
    signature = row["signature"]
    if not isinstance(signature, list) or len(signature) != 4:
        raise ValueError("invalid convergence signature")
    return (
        int(signature[0]),
        str(signature[1]),
        float(signature[2]),
        str(signature[3]),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    if not (BURNED_LOW <= args.seed_start <= BURNED_HIGH):
        raise ValueError("preflight must start inside the burned Q-R1 namespace")
    if args.seed_start + args.histories - 1 > BURNED_HIGH:
        raise ValueError("preflight would leave the burned Q-R1 namespace")
    states = development_states(args)
    sched = scheduler()
    started = time.perf_counter()
    low_configs = family(args, args.low_paths)
    high_by_signature = {
        (
            config.horizon,
            config.mode,
            config.worst_product_floor,
            config.service_statistic,
        ): config
        for config in family(args, args.high_paths)
    }
    convergence: list[dict[str, object]] = []
    convergence_pairs: list[dict[str, object]] = []
    converged_signatures: list[tuple[int, str, float, str]] = []
    if args.convergence_receipt is not None:
        receipt = json.loads(args.convergence_receipt.read_text())
        if receipt.get("claim_status") != "BURNED_DEVELOPMENT_NO_CLAIM":
            raise ValueError("convergence receipt has incompatible claim status")
        if receipt.get("history_roots") != [
            args.seed_start,
            args.seed_start + args.histories - 1,
        ]:
            raise ValueError("convergence receipt history roots do not match")
        if int(receipt.get("states", -1)) != len(states):
            raise ValueError("convergence receipt state count does not match")
        if receipt.get("conditional_path_budgets") != [
            args.low_paths,
            args.high_paths,
        ]:
            raise ValueError("convergence receipt path budgets do not match")
        if float(receipt.get("value_indifference_tolerance", 0.0)) != float(
            args.value_indifference_tolerance
        ):
            raise ValueError("convergence receipt indifference tolerance does not match")
        if str(receipt.get("tie_breaker", "legacy")) != args.tie_breaker:
            raise ValueError("convergence receipt tie breaker does not match")
        convergence = list(receipt["convergence"])
        convergence_pairs = list(receipt.get("convergence_pairs", []))
        converged_signatures = [
            _signature_tuple(row)
            for row in convergence
            if bool(row.get("convergence_pass"))
        ]
    else:
        for low in low_configs:
            signature = (
                low.horizon,
                low.mode,
                low.worst_product_floor,
                low.service_statistic,
            )
            high = high_by_signature[signature]
            low_actions: list[int] = []
            high_actions: list[int] = []
            errors: list[float] = []
            low_abstentions = 0
            high_abstentions = 0
            for campaign, retained_prior in states:
                observation = first_observation(campaign)
                for prior_label, prior in (
                    ("retained", retained_prior),
                    ("reset", 0.5),
                ):
                    outputs = []
                    for config in (low, high):
                        if time.perf_counter() - started > args.hard_cap_seconds:
                            raise TimeoutError("comparator v2 preflight hard cap exceeded")
                        try:
                            action, detail = choose_comparator_v2_action(
                                observation,
                                base_skeleton=campaign.skeleton,
                                prefix=(),
                                scheduler=sched,
                                belief=fixed_theta_belief(prior),
                                planning_key=PlanningKey(
                                    campaign.history_root,
                                    campaign.campaign_index,
                                    0,
                                ),
                                config=config,
                            )
                            outputs.append((action, detail))
                        except NoFeasibleStructuredAction:
                            outputs.append(None)
                    if outputs[0] is None:
                        low_abstentions += 1
                    if outputs[1] is None:
                        high_abstentions += 1
                    if outputs[0] is not None and outputs[1] is not None:
                        low_actions.append(int(outputs[0][0]))
                        high_actions.append(int(outputs[1][0]))
                        error = abs(
                            float(outputs[0][1]["planning_early_ret_complete_cohort"])
                            - float(outputs[1][1]["planning_early_ret_complete_cohort"])
                        )
                        errors.append(error)
                        convergence_pairs.append(
                            {
                                "signature": list(signature),
                                "history_root": campaign.history_root,
                                "campaign_index": campaign.campaign_index,
                                "persistence_mode": campaign.persistence_mode,
                                "prior_arm": prior_label,
                                "retained_prior": retained_prior,
                                "low_action": int(outputs[0][0]),
                                "high_action": int(outputs[1][0]),
                                "absolute_planning_value_error": error,
                            }
                        )
            comparable = len(errors)
            agreement = (
                float(np.mean(np.asarray(low_actions) == np.asarray(high_actions)))
                if comparable
                else 0.0
            )
            mean_error = float(np.mean(errors)) if errors else float("inf")
            q95_error = float(np.quantile(errors, 0.95)) if errors else float("inf")
            passed = (
                low_abstentions == 0
                and high_abstentions == 0
                and agreement >= 0.95
                and mean_error < 0.005
                and q95_error < 0.01
            )
            convergence.append(
                {
                    "signature": list(signature),
                    "low_config": low.config_id,
                    "high_config": high.config_id,
                    "first_action_agreement": agreement,
                    "mean_abs_planning_value_error": mean_error,
                    "q95_abs_planning_value_error": q95_error,
                    "low_abstentions": low_abstentions,
                    "high_abstentions": high_abstentions,
                    "comparable_arm_states": comparable,
                    "convergence_pass": passed,
                }
            )
            if passed:
                converged_signatures.append(signature)

    if args.convergence_only:
        return {
            "schema_version": "q_r1_comparator_v2_preflight_v1",
            "claim_status": "BURNED_DEVELOPMENT_NO_CLAIM",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "history_roots": [args.seed_start, args.seed_start + args.histories - 1],
            "states": len(states),
            "conditional_path_budgets": [args.low_paths, args.high_paths],
            "value_indifference_tolerance": args.value_indifference_tolerance,
            "tie_breaker": args.tie_breaker,
            "selection_performed": False,
            "learner_return_used": False,
            "retained_minus_reset_used_for_selection": False,
            "phase": "convergence_only",
            "convergence": convergence,
            "convergence_pairs": convergence_pairs,
            "pareto": [],
            "pareto_pairs": [],
            "elapsed_seconds": time.perf_counter() - started,
        }

    pareto = []
    pareto_pairs: list[dict[str, object]] = []
    for signature in converged_signatures:
        config = next(
            row
            for row in low_configs
            if (
                row.horizon,
                row.mode,
                row.worst_product_floor,
                row.service_statistic,
            )
            == signature
        )
        retained_metrics = []
        reset_metrics = []
        abstentions = 0
        for campaign, retained_prior in states:
            pair = []
            for prior in (retained_prior, 0.5):
                if time.perf_counter() - started > args.hard_cap_seconds:
                    raise TimeoutError("comparator v2 preflight hard cap exceeded")
                try:
                    calendar, _detail = comparator_v2_calendar(
                        campaign=campaign,
                        belief=fixed_theta_belief(prior),
                        scheduler=sched,
                        config=config,
                    )
                except NoFeasibleStructuredAction:
                    abstentions += 1
                    pair = []
                    break
                pair.append(
                    evaluate_calendar(
                        campaign=campaign,
                        calendar=calendar,
                        scheduler=sched,
                    )
                )
            if len(pair) == 2:
                retained_metrics.append(pair[0])
                reset_metrics.append(pair[1])
                pareto_pairs.append(
                    {
                        "config_id": config.config_id,
                        "history_root": campaign.history_root,
                        "campaign_index": campaign.campaign_index,
                        "persistence_mode": campaign.persistence_mode,
                        "retained_prior": retained_prior,
                        "retained": pair[0],
                        "reset": pair[1],
                    }
                )
        if retained_metrics:
            def delta(key: str) -> float:
                return float(
                    np.mean(
                        [
                            float(retained[key]) - float(reset[key])
                            for retained, reset in zip(retained_metrics, reset_metrics)
                        ]
                    )
                )

            pareto.append(
                {
                    "config_id": config.config_id,
                    "pairs": len(retained_metrics),
                    "abstentions": abstentions,
                    "mean_absolute_early_ret_complete_cohort": float(
                        np.mean(
                            [
                                row["early_ret_complete_cohort"]
                                for pair in (retained_metrics, reset_metrics)
                                for row in pair
                            ]
                        )
                    ),
                    "retained_minus_reset_early_ret_complete_cohort": delta(
                        "early_ret_complete_cohort"
                    ),
                    "retained_minus_reset_worst_product_fill": delta(
                        "worst_product_fill"
                    ),
                    "retained_minus_reset_unresolved_orders": delta(
                        "unresolved_orders"
                    ),
                    "retained_minus_reset_lost_orders": delta("lost_orders"),
                    "max_mass_residual": float(
                        max(
                            abs(float(row["mass_residual"]))
                            for pair in (retained_metrics, reset_metrics)
                            for row in pair
                        )
                    ),
                }
            )

    return {
        "schema_version": "q_r1_comparator_v2_preflight_v1",
        "claim_status": "BURNED_DEVELOPMENT_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "history_roots": [args.seed_start, args.seed_start + args.histories - 1],
        "states": len(states),
        "conditional_path_budgets": [args.low_paths, args.high_paths],
        "value_indifference_tolerance": args.value_indifference_tolerance,
        "tie_breaker": args.tie_breaker,
        "selection_performed": False,
        "learner_return_used": False,
        "retained_minus_reset_used_for_selection": False,
        "phase": "pareto_from_receipt" if args.convergence_receipt else "all",
        "convergence": convergence,
        "convergence_pairs": convergence_pairs,
        "pareto": pareto,
        "pareto_pairs": pareto_pairs,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=BURNED_LOW)
    parser.add_argument("--histories", type=int, default=4)
    parser.add_argument("--campaigns", type=int, default=12)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--horizons", nargs="+", type=int, default=[3, 4])
    parser.add_argument(
        "--service-floors", nargs="+", type=float, default=[0.70, 0.75, 0.80]
    )
    parser.add_argument("--low-paths", type=int, default=4)
    parser.add_argument("--high-paths", type=int, default=16)
    parser.add_argument("--hard-cap-seconds", type=float, default=7200.0)
    parser.add_argument("--value-indifference-tolerance", type=float, default=0.0)
    parser.add_argument("--tie-breaker", choices=("legacy", "service"), default="legacy")
    parser.add_argument(
        "--scenario-only",
        action="store_true",
        help="Evaluate only the ReT-primary scenario controller.",
    )
    parser.add_argument("--convergence-only", action="store_true")
    parser.add_argument("--convergence-receipt", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
