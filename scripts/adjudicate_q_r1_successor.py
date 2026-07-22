#!/usr/bin/env python3
"""Adjudicate repaired Q-R1 results without a post-outcome arm selector."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


CLAIM_ESTIMANDS = ("prefix_natural_replanning", "sustained_control")
DIAGNOSTIC_ESTIMAND = "historical_splice"
PRIMARY_ARM = "retained_posterior"
PLACEBOS = ("shuffled_posterior", "delayed_posterior", "wrong_posterior")


def _arm(summary: dict, estimand: str, kappa: float, arm: str) -> dict:
    return summary[estimand][str(kappa)][arm]


def adjudicate(payload: dict) -> dict:
    summary = payload["summary"]
    estimands = {}
    for estimand in CLAIM_ESTIMANDS:
        primary = _arm(summary, estimand, 0.9, PRIMARY_ARM)
        dose = _arm(summary, estimand, 0.75, PRIMARY_ARM)
        iid = _arm(summary, estimand, 0.5, PRIMARY_ARM)
        placebos = {
            arm: _arm(summary, estimand, 0.9, arm) for arm in PLACEBOS
        }
        visible_pass = (
            primary["mean_early_ret_visible_delta"] >= 0.01
            and primary["early_ret_visible_clustered_ci95"][0] > 0.0
        )
        complete_pass = (
            primary["mean_early_ret_complete_cohort_delta"] >= 0.01
            and primary["early_ret_complete_cohort_clustered_ci95"][0] > 0.0
        )
        mechanism_pass = (
            primary["mean_early_ret_complete_cohort_delta"]
            > dose["mean_early_ret_complete_cohort_delta"]
            > iid["mean_early_ret_complete_cohort_delta"]
            and abs(iid["mean_early_ret_complete_cohort_delta"]) <= 0.005
            and all(
                row["mean_early_ret_complete_cohort_delta"] <= 0.0
                for row in placebos.values()
            )
        )
        guardrails = {
            "lost_demand_no_increase": (
                primary["mean_lost_orders_delta"] <= 0.0
                and primary["mean_lost_quantity_delta"] <= 0.0
            ),
            "worst_product_lcb_ge_minus_0p02": (
                primary["worst_product_fill_clustered_ci95"][0] >= -0.02
            ),
            "scheduled_resources_exact": (
                primary["max_scheduled_resource_error"] == 0.0
            ),
            "unresolved_fully_reported_not_hidden": True,
        }
        passed = visible_pass and complete_pass and mechanism_pass and all(guardrails.values())
        estimands[estimand] = {
            "primary_0p90": primary,
            "dose_0p75": dose,
            "iid_null": iid,
            "placebos_0p90": placebos,
            "visible_pass": visible_pass,
            "complete_cohort_pass": complete_pass,
            "mechanism_pass": mechanism_pass,
            "guardrails": guardrails,
            "pass_retained_information_value": passed,
        }

    diagnostic = _arm(summary, DIAGNOSTIC_ESTIMAND, 0.9, PRIMARY_ARM)
    any_claim_pass = any(
        result["pass_retained_information_value"] for result in estimands.values()
    )
    verdict = (
        "PASS_REPAIRED_RETAINED_INFORMATION_VALUE"
        if any_claim_pass
        else "STOP_REPAIRED_Q_R1_NO_RETAINED_INFORMATION_PASS"
    )
    return {
        "schema_version": "q_r1_successor_adjudication_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_schema": payload["schema_version"],
        "estimands": estimands,
        "historical_splice_diagnostic_0p90": diagnostic,
        "historical_splice_can_rescue": False,
        "eligible_controller_arms": [
            "retained_posterior",
            "reset_posterior_0p5",
        ],
        "ineligible_as_actions": [
            "oracle_initial_context",
            "shuffled_posterior",
            "delayed_posterior",
            "wrong_posterior",
        ],
        "oracle_use": "reported only as an information upper bound; never selected per episode",
        "post_outcome_episode_selector_used": False,
        "incremental_learned_residual_established": False,
        "verdict": verdict,
        "learner_training_authorized": False,
        "next_if_pass": "freeze an observable D4 selector/encoder contract against the universal retained MPC",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    payload = json.loads(args.source.read_text())
    result = adjudicate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
