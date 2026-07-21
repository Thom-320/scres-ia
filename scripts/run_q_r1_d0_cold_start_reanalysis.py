#!/usr/bin/env python3
"""Reproduce burned R0 with paired rows and the Q-R1 cold-start estimand."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.program_o_state_rich import StateRichConfiguration, state_rich_calendar  # noqa: E402
from supply_chain.q_r1_retained_learning import (  # noqa: E402
    PhysicalCampaignState,
    common_continuation_calendar,
    evaluate_calendar,
)
from supply_chain.retained_context_discovery import (  # noqa: E402
    ARMS,
    arm_priors,
    build_campaign_history,
)


CELLS = {"rho90_share90": (0.90, 0.90)}


def cluster_lcb(values: dict[int, list[float]], *, seed: int = 20260721) -> float:
    roots = sorted(values)
    means = np.asarray([np.mean(values[root]) for root in roots], dtype=float)
    if len(means) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(means), size=(5000, len(means)))
    return float(np.quantile(means[draws].mean(axis=1), 0.025))


def summarize(rows: list[dict]) -> dict:
    by_key = {
        (row["kappa"], row["history_root"], row["campaign_index"], row["arm"]): row
        for row in rows
    }
    contrasts = {}
    for kappa in (0.5, 0.75, 0.9):
        roots = sorted({row["history_root"] for row in rows if row["kappa"] == kappa})
        arm_out = {}
        for arm in ARMS:
            early: dict[int, list[float]] = defaultdict(list)
            total: dict[int, list[float]] = defaultdict(list)
            favorable = []
            action_divergence = []
            fill = []
            for root in roots:
                campaigns = sorted(
                    row["campaign_index"]
                    for row in rows
                    if row["kappa"] == kappa
                    and row["history_root"] == root
                    and row["arm"] == arm
                    and row["campaign_index"] > 0
                )
                for campaign in campaigns:
                    target = by_key[(kappa, root, campaign, arm)]
                    reset = by_key[(kappa, root, campaign, "reset_posterior_0p5")]
                    delta = target["early_ret_2w"] - reset["early_ret_2w"]
                    early[root].append(delta)
                    total[root].append(target["ret_visible"] - reset["ret_visible"])
                    favorable.append(delta > 0.0)
                    action_divergence.append(target["calendar"][:2] != reset["calendar"][:2])
                    fill.append(target["worst_product_fill"] - reset["worst_product_fill"])
            values = [value for root in roots for value in early[root]]
            total_values = [value for root in roots for value in total[root]]
            arm_out[arm] = {
                "mean_early_ret_delta": float(np.mean(values)),
                "early_ret_lcb95_history_clustered": cluster_lcb(early),
                "mean_total_ret_delta": float(np.mean(total_values)),
                "favorable_fraction": float(np.mean(favorable)),
                "first_two_action_divergence": float(np.mean(action_divergence)),
                "mean_worst_product_delta": float(np.mean(fill)),
                "n_pairs": len(values),
            }
        contrasts[str(kappa)] = arm_out
    return contrasts


def run(args: argparse.Namespace) -> dict:
    rho, share = CELLS[args.cell]
    sched = scheduler()
    rows: list[dict] = []
    for kappa in (0.5, 0.75, 0.9):
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
                calendars = {}
                for arm in ARMS:
                    calendar, _decisions = state_rich_calendar(
                        skeleton=campaign.skeleton.as_dict(),
                        scheduler=sched,
                        config=StateRichConfiguration("belief_mpc", 3),
                        regime_persistence=rho,
                        dominant_share=share,
                        initial_belief_c=priors[arm][history_index][campaign_index],
                    )
                    calendars[arm] = tuple(calendar)
                continuation = calendars["reset_posterior_0p5"]
                physical = PhysicalCampaignState(
                    history_root=campaign.history_root,
                    campaign_index=campaign.campaign_index,
                    persistence_mode="iid" if kappa == 0.5 else f"binary_{kappa}",
                    theta=(rho, share),
                    initial_regime=campaign.initial_regime,
                    skeleton=campaign.skeleton,
                )
                for arm in ARMS:
                    calendar = common_continuation_calendar(calendars[arm], continuation)
                    metrics = evaluate_calendar(
                        campaign=physical, calendar=calendar, scheduler=sched
                    )
                    rows.append(
                        {
                            "kappa": kappa,
                            "history_root": campaign.history_root,
                            "campaign_index": campaign.campaign_index,
                            "tape_seed": campaign.tape_seed,
                            "arm": arm,
                            "skeleton_sha256": campaign.skeleton.skeleton_sha256,
                            "prefix_state_hash": campaign.skeleton.prefix_state_hash,
                            "initial_belief_c": priors[arm][history_index][campaign_index],
                            "calendar": list(calendar),
                            **metrics,
                        }
                    )
    contrasts = summarize(rows)
    return {
        "schema_version": "q_r1_d0_cold_start_reanalysis_v1",
        "claim_status": "EXPLORATORY_NO_CLAIM",
        "historical_r0_verdict_preserved": "STOP_NO_RETAINED_CONTEXT_HEADROOM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cell": args.cell,
        "history_roots": [args.seed_start, args.seed_start + args.histories - 1],
        "campaigns_per_history": args.campaigns,
        "cold_start_decisions": 2,
        "common_continuation": "reset_posterior_0p5 weeks 3-8",
        "contrasts": contrasts,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", default="rho90_share90", choices=CELLS)
    parser.add_argument("--seed-start", type=int, default=7_570_001)
    parser.add_argument("--histories", type=int, default=24)
    parser.add_argument("--campaigns", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/q_r1/d0_cold_start_reanalysis_v1/result.json",
    )
    args = parser.parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["contrasts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
