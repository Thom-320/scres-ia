#!/usr/bin/env python3
"""Run burned-only R0 retained-context discovery and emit one JSON verdict."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.retained_context_discovery import (  # noqa: E402
    ARMS,
    RESOURCE_KEYS,
    arm_priors,
    build_campaign_history,
    evaluate_campaign_prior,
)


CELLS = {
    "rho90_share90": (0.90, 0.90),
    "rho75_share90": (0.75, 0.90),
}


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(row["kappa"], row["history_root"], row["campaign_index"], row["arm"]): row for row in rows}
    contrasts: dict[str, dict[str, Any]] = {}
    for kappa in sorted({float(row["kappa"]) for row in rows}):
        arm_delta: dict[str, list[float]] = defaultdict(list)
        fill_delta: dict[str, list[float]] = defaultdict(list)
        early_loss_delta: dict[str, list[float]] = defaultdict(list)
        resource_error = 0.0
        roots = sorted({int(row["history_root"]) for row in rows if row["kappa"] == kappa})
        campaigns = sorted({int(row["campaign_index"]) for row in rows if row["kappa"] == kappa and int(row["campaign_index"]) > 0})
        for root in roots:
            for campaign in campaigns:
                reset = by_key[(kappa, root, campaign, "reset_posterior_0p5")]
                for arm in ARMS:
                    row = by_key[(kappa, root, campaign, arm)]
                    arm_delta[arm].append(float(row["ret_visible"]) - float(reset["ret_visible"]))
                    fill_delta[arm].append(float(row["worst_product_fill"]) - float(reset["worst_product_fill"]))
                    early_loss_delta[arm].append(float(reset["service_loss_auc_first_two_weeks"]) - float(row["service_loss_auc_first_two_weeks"]))
                    for key in RESOURCE_KEYS:
                        resource_error = max(resource_error, abs(float(row[key]) - float(reset[key])))
        contrasts[str(kappa)] = {
            arm: {
                "mean_ret_delta_vs_reset": _mean(arm_delta[arm]),
                "favorable_fraction": _mean([float(value > 0.0) for value in arm_delta[arm]]),
                "mean_worst_product_fill_delta": _mean(fill_delta[arm]),
                "mean_early_service_loss_reduction": _mean(early_loss_delta[arm]),
                "n_pairs": len(arm_delta[arm]),
            }
            for arm in ARMS
        }
        contrasts[str(kappa)]["max_programmed_resource_error"] = resource_error

    oracle_headroom = float(contrasts["0.9"]["oracle_initial_context"]["mean_ret_delta_vs_reset"])
    if oracle_headroom < 0.03:
        verdict = "STOP_NO_RETAINED_CONTEXT_HEADROOM"
    else:
        d05 = float(contrasts["0.5"]["retained_posterior"]["mean_ret_delta_vs_reset"])
        d075 = float(contrasts["0.75"]["retained_posterior"]["mean_ret_delta_vs_reset"])
        d09 = float(contrasts["0.9"]["retained_posterior"]["mean_ret_delta_vs_reset"])
        pass_checks = {
            "retained_effect_0p9_ge_0p03": d09 >= 0.03,
            "favorable_fraction_0p9_ge_0p70": float(contrasts["0.9"]["retained_posterior"]["favorable_fraction"]) >= 0.70,
            "monotone_kappa": d09 >= d075 > 0.0,
            "iid_null_equivalent": abs(d05) <= 0.01,
            "shuffled_no_advantage": float(contrasts["0.9"]["shuffled_posterior"]["mean_ret_delta_vs_reset"]) <= 0.01,
            "wrong_no_advantage": float(contrasts["0.9"]["wrong_posterior"]["mean_ret_delta_vs_reset"]) <= 0.01,
            "resources_exact": all(float(contrasts[str(k)]["max_programmed_resource_error"]) == 0.0 for k in (0.5, 0.75, 0.9)),
            "worst_product_noninferior": float(contrasts["0.9"]["retained_posterior"]["mean_worst_product_fill_delta"]) >= -0.02,
        }
        verdict = "PASS_RETAINED_CONTEXT_EXPLORATORY_SIGNAL" if all(pass_checks.values()) else "STOP_RETAINED_CONTEXT_PATTERN_FAILED"
        contrasts["promotion_checks"] = pass_checks
    return {"contrasts": contrasts, "verdict": verdict, "oracle_headroom_0p9": oracle_headroom}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.cell not in CELLS:
        raise ValueError(f"unknown cell: {args.cell}")
    rho, share = CELLS[args.cell]
    sched = scheduler()
    rows: list[dict[str, Any]] = []
    physical_identity_failures = 0
    for kappa in (0.50, 0.75, 0.90):
        histories = [
            build_campaign_history(
                history_root=args.seed_start + index,
                campaigns=args.campaigns,
                kappa=kappa,
                scheduler=sched,
                regime_persistence=rho,
                dominant_share=share,
            )
            for index in range(args.histories)
        ]
        priors = arm_priors(
            histories=histories,
            regime_persistence=rho,
            dominant_share=share,
        )
        for history_index, history in enumerate(histories):
            for campaign_index, campaign in enumerate(history):
                campaign_rows = []
                for arm in ARMS:
                    row = evaluate_campaign_prior(
                        campaign=campaign,
                        arm=arm,
                        initial_belief_c=priors[arm][history_index][campaign_index],
                        scheduler=sched,
                        regime_persistence=rho,
                        dominant_share=share,
                    )
                    campaign_rows.append(row)
                    rows.append(row)
                hashes = {(row["skeleton_sha256"], row["prefix_state_hash"]) for row in campaign_rows}
                if len(hashes) != 1:
                    physical_identity_failures += 1
    summary = summarize(rows)
    payload = {
        "schema_version": "retained_context_discovery_result_v1",
        "status": "EXPLORATORY_NO_CLAIM",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cell": args.cell,
        "history_roots": [args.seed_start, args.seed_start + args.histories - 1],
        "campaign_seed_derivation": "history_root * 100 + campaign_index",
        "histories_per_kappa": args.histories,
        "campaigns_per_history": args.campaigns,
        "kappas": [0.5, 0.75, 0.9],
        "arms": list(ARMS),
        "sealed_seed_namespaces_opened": False,
        "physical_identity_failures": physical_identity_failures,
        **summary,
    }
    if args.include_rows:
        payload["rows"] = rows
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", default="rho90_share90", choices=sorted(CELLS))
    parser.add_argument("--seed-start", type=int, default=7_570_001)
    parser.add_argument("--histories", type=int, default=24)
    parser.add_argument("--campaigns", type=int, default=12)
    parser.add_argument("--include-rows", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results/retained_context/r0_primary_v1/result.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("status", "verdict", "oracle_headroom_0p9")}, indent=2))


if __name__ == "__main__":
    main()
