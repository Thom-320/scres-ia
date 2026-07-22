#!/usr/bin/env python3
"""Burned c256/c1024 convergence on c64/c256 action-disagreement states."""

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
from supply_chain.program_o_state_rich import StateRichConfiguration, state_rich_calendar  # noqa: E402
from supply_chain.q_r1_comparator_v2 import (  # noqa: E402
    ComparatorV2Config,
    PlanningKey,
    choose_comparator_v2_action,
)
from supply_chain.q_r1_retained_learning import PhysicalCampaignState  # noqa: E402
from supply_chain.retained_context_discovery import arm_priors, build_campaign_history  # noqa: E402


def disagreement_targets(payload: dict[str, object]) -> list[dict[str, object]]:
    rows = [
        row
        for row in payload["convergence_pairs"]
        if row["signature"][1] == "scenario"
        and int(row["low_action"]) != int(row["high_action"])
    ]
    identities = [
        (
            int(row["history_root"]),
            int(row["campaign_index"]),
            str(row["persistence_mode"]),
            str(row["prior_arm"]),
        )
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate disagreement target")
    return sorted(rows, key=lambda row: identities[rows.index(row)])


def rebuild_target(row: dict[str, object]) -> tuple[PhysicalCampaignState, float]:
    mode = str(row["persistence_mode"])
    kappa = float(mode.removeprefix("binary_"))
    root = int(row["history_root"])
    campaign_index = int(row["campaign_index"])
    histories = [
        build_campaign_history(
            history_root=root,
            campaigns=12,
            kappa=kappa,
            scheduler=scheduler(),
            regime_persistence=0.90,
            dominant_share=0.90,
        )
    ]
    retained = arm_priors(
        histories=histories,
        regime_persistence=0.90,
        dominant_share=0.90,
    )["retained_posterior"][0][campaign_index]
    campaign = histories[0][campaign_index]
    state = PhysicalCampaignState(
        history_root=root,
        campaign_index=campaign_index,
        persistence_mode=mode,
        theta=(0.90, 0.90),
        initial_regime=campaign.initial_regime,
        skeleton=campaign.skeleton,
    )
    prior = float(retained) if row["prior_arm"] == "retained" else 0.5
    return state, prior


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--low-paths", type=int, default=256)
    parser.add_argument("--high-paths", type=int, default=1024)
    parser.add_argument("--hard-cap-seconds", type=float, default=10_800.0)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if not 0 <= args.shard_index < args.shards:
        raise ValueError("invalid shard index")
    payload = json.loads(args.input.read_text())
    if payload.get("claim_status") != "BURNED_DEVELOPMENT_NO_CLAIM":
        raise ValueError("target source must be burned development")
    all_targets = disagreement_targets(payload)
    targets = all_targets[args.shard_index :: args.shards]
    configs = [
        ComparatorV2Config(horizon=4, conditional_paths=paths, mode="scenario", worst_product_floor=0.0)
        for paths in (args.low_paths, args.high_paths)
    ]
    started = time.perf_counter()
    rows = []
    for target in targets:
        campaign, prior = rebuild_target(target)
        observation = first_observation(campaign)
        outputs = []
        for config in configs:
            if time.perf_counter() - started > args.hard_cap_seconds:
                raise TimeoutError("targeted convergence hard cap exceeded")
            action, detail = choose_comparator_v2_action(
                observation,
                base_skeleton=campaign.skeleton,
                prefix=(),
                scheduler=scheduler(),
                belief=fixed_theta_belief(prior),
                planning_key=PlanningKey(
                    campaign.history_root, campaign.campaign_index, 0
                ),
                config=config,
            )
            outputs.append((action, detail))
        rows.append(
            {
                "history_root": campaign.history_root,
                "campaign_index": campaign.campaign_index,
                "persistence_mode": campaign.persistence_mode,
                "prior_arm": target["prior_arm"],
                "prior": prior,
                "previous_c64_action": int(target["low_action"]),
                "previous_c256_action": int(target["high_action"]),
                "c256_action": int(outputs[0][0]),
                "c1024_action": int(outputs[1][0]),
                "absolute_planning_value_error": abs(
                    float(outputs[0][1]["planning_early_ret_complete_cohort"])
                    - float(outputs[1][1]["planning_early_ret_complete_cohort"])
                ),
            }
        )
    errors = np.asarray([row["absolute_planning_value_error"] for row in rows], dtype=float)
    agreement = np.asarray([row["c256_action"] == row["c1024_action"] for row in rows], dtype=bool)
    result = {
        "schema_version": "q_r1_comparator_v2_targeted_convergence_v1",
        "claim_status": "BURNED_INSTRUMENT_CALIBRATION_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.input),
        "target_rule": "scenario c64 action differs from c256 action",
        "all_target_count": len(all_targets),
        "shards": args.shards,
        "shard_index": args.shard_index,
        "path_budgets": [args.low_paths, args.high_paths],
        "rows": rows,
        "agreement": float(agreement.mean()) if len(rows) else 1.0,
        "mean_abs_planning_value_error": float(errors.mean()) if len(rows) else 0.0,
        "q95_abs_planning_value_error": float(np.quantile(errors, 0.95)) if len(rows) else 0.0,
        "elapsed_seconds": time.perf_counter() - started,
        "selection_performed": False,
        "learner_return_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

