#!/usr/bin/env python3
"""Burned-data convergence and universal-selection audit for retained MPC.

The script never opens confirmation roots. It first compares stratified p16
and p64 first actions/values, then evaluates only converged p16 configurations
on the same burned development states. The universal configuration maximizes
absolute pooled control quality, never the retained-minus-reset treatment
effect or a learner return.
"""

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
from supply_chain.program_t_full_des_mpc import FullDEST0Config  # noqa: E402
from supply_chain.q_r1_retained_learning import (  # noqa: E402
    PhysicalCampaignState,
    controller_calendar,
    controller_prefix,
    evaluate_calendar,
)
from supply_chain.retained_context_discovery import (  # noqa: E402
    arm_priors,
    build_campaign_history,
)


def config_family(
    horizons: tuple[int, ...], modes: tuple[str, ...]
) -> tuple[tuple[int, str], ...]:
    return tuple(
        (horizon, mode)
        for horizon in horizons
        for mode in modes
    )


def development_states(args: argparse.Namespace) -> list[tuple[PhysicalCampaignState, float]]:
    sched = scheduler()
    output = []
    for kappa in (0.75, 0.9):
        histories = [
            build_campaign_history(
                history_root=args.seed_start + index,
                campaigns=args.campaigns,
                kappa=kappa,
                scheduler=sched,
                regime_persistence=0.90,
                dominant_share=0.90,
            )
            for index in range(args.histories)
        ]
        priors = arm_priors(
            histories=histories,
            regime_persistence=0.90,
            dominant_share=0.90,
        )["retained_posterior"]
        for history_index, history in enumerate(histories):
            for campaign in history[1:]:
                output.append(
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
    # Deterministic spread over roots/campaign indices, not outcome selection.
    if len(output) <= args.states:
        return output
    indices = np.linspace(0, len(output) - 1, args.states, dtype=int)
    return [output[int(index)] for index in indices]


def planning_value(detail: dict[str, object]) -> float:
    decision = detail["decisions"][0]
    return float(decision["planning_ret"])


def run(args: argparse.Namespace) -> dict[str, object]:
    sched = scheduler()
    states = development_states(args)
    started = time.perf_counter()
    convergence_rows = []
    converged = []
    for horizon, mode in config_family(tuple(args.horizons), tuple(args.modes)):
        actions_16 = []
        actions_64 = []
        value_errors = []
        times_16 = []
        times_64 = []
        for campaign, prior in states:
            if time.perf_counter() - started > args.hard_cap_seconds:
                raise TimeoutError("retained MPC calibration hard cap exceeded")
            belief = fixed_theta_belief(prior)
            details = {}
            actions = {}
            for particles in (16, 64):
                config = FullDEST0Config(
                    horizon=horizon,
                    mode=mode,
                    particles=particles,
                    worst_product_floor=0.70,
                    belief_integration="stratified",
                )
                tick = time.perf_counter()
                prefix, detail = controller_prefix(
                    campaign=campaign,
                    belief=belief,
                    scheduler=sched,
                    config=config,
                    decisions=1,
                )
                elapsed = (time.perf_counter() - tick) * 1000.0
                actions[particles] = int(prefix[0])
                details[particles] = detail
                (times_16 if particles == 16 else times_64).append(elapsed)
            actions_16.append(actions[16])
            actions_64.append(actions[64])
            value_errors.append(abs(planning_value(details[16]) - planning_value(details[64])))
        agreement = float(np.mean(np.asarray(actions_16) == np.asarray(actions_64)))
        max_value_error = float(np.max(value_errors))
        passed = agreement >= 0.95 and max_value_error < 0.005
        row = {
            "horizon": horizon,
            "mode": mode,
            "p16_p64_first_action_agreement": agreement,
            "p16_p64_max_planning_value_error": max_value_error,
            "p16_mean_online_ms_first_action": float(np.mean(times_16)),
            "p64_mean_online_ms_first_action": float(np.mean(times_64)),
            "states": len(states),
            "convergence_pass": passed,
        }
        convergence_rows.append(row)
        if passed:
            converged.append((horizon, mode))

    performance_rows = []
    for horizon, mode in converged:
        config = FullDEST0Config(
            horizon=horizon,
            mode=mode,
            particles=16,
            worst_product_floor=0.70,
            belief_integration="stratified",
        )
        absolute_complete = []
        absolute_visible = []
        absolute_full = []
        worst_fill = []
        lost = []
        online = []
        for campaign, prior in states:
            for belief in (fixed_theta_belief(prior), fixed_theta_belief(0.5)):
                calendar, detail = controller_calendar(
                    campaign=campaign,
                    belief=belief,
                    scheduler=sched,
                    config=config,
                )
                metrics = evaluate_calendar(
                    campaign=campaign, calendar=calendar, scheduler=sched
                )
                absolute_complete.append(metrics["early_ret_complete_cohort"])
                absolute_visible.append(metrics["early_ret_visible"])
                absolute_full.append(metrics["ret_full"])
                worst_fill.append(metrics["worst_product_fill"])
                lost.append(metrics["lost_orders"])
                online.append(detail["online_ms"])
        performance_rows.append(
            {
                "config_id": config.config_id,
                "horizon": horizon,
                "mode": mode,
                "mean_early_ret_complete_cohort": float(np.mean(absolute_complete)),
                "mean_early_ret_visible": float(np.mean(absolute_visible)),
                "mean_ret_full": float(np.mean(absolute_full)),
                "mean_worst_product_fill": float(np.mean(worst_fill)),
                "max_lost_orders": float(np.max(lost)),
                "mean_online_ms_campaign": float(np.mean(online)),
            }
        )
    eligible = [row for row in performance_rows if row["max_lost_orders"] <= 0.0]
    selected = (
        max(
            eligible,
            key=lambda row: (
                row["mean_early_ret_complete_cohort"],
                row["mean_ret_full"],
                row["mean_worst_product_fill"],
                -row["mean_online_ms_campaign"],
                row["config_id"],
            ),
        )
        if eligible
        else None
    )
    return {
        "schema_version": "q_r1_retained_mpc_calibration_v1",
        "claim_status": "BURNED_DEVELOPMENT_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "history_roots": [args.seed_start, args.seed_start + args.histories - 1],
        "state_selection": "deterministic linspace over roots/campaigns; no outcome used",
        "selection_objective": "absolute pooled early complete-cohort ReT, then ret_full and worst fill; never retained-minus-reset or learner return",
        "convergence_gate": {
            "first_action_agreement_min": 0.95,
            "max_planning_value_error": 0.005,
        },
        "h8_boundary": (
            "included after explicit CLI opt-in"
            if 8 in args.horizons
            else "not run; requires cached/DP preflight and explicit CLI opt-in"
        ),
        "convergence": convergence_rows,
        "performance": performance_rows,
        "selected_universal_config": selected,
        "confirmation_authorized": selected is not None,
        "learner_training_authorized": False,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=7_570_801)
    parser.add_argument("--histories", type=int, default=4)
    parser.add_argument("--campaigns", type=int, default=12)
    parser.add_argument("--states", type=int, default=20)
    parser.add_argument(
        "--horizons", nargs="+", type=int, choices=(1, 3, 4, 6, 8),
        default=[1, 3, 4, 6],
    )
    parser.add_argument(
        "--modes", nargs="+", choices=("scenario", "robust", "constraint_aware"),
        default=["scenario", "robust", "constraint_aware"],
    )
    parser.add_argument("--hard-cap-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
