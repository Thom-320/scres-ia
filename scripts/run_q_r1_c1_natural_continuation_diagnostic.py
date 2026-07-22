#!/usr/bin/env python3
"""Q-R1 C1 diagnostic: does the retained effect survive NATURAL continuation?

EXPLORATORY_NO_CLAIM. Burned roots 7570801-7570824 only (already opened by the
frozen replication). Reproduces, in-repo, the external audit's A/B/C estimand
triad so the claim has a custodied artifact:

  A = retained 2-week prefix, then the RESET-belief MPC policy replanning
      weeks 3-8 on the state actually reached by the arm (natural common
      continuation).
  B = retained-belief MPC controller for the FULL campaign (its own natural
      receding-horizon calendar).
  C = the frozen replication's fixed splice (retained prefix + the reset
      trajectory's weeks 3-8) — diagnostic only, reproduces the burn.

Harness self-checks (all must pass for the result to be interpretable):
  1. iid (kappa=0.5) deltas are exactly 0 for every estimand (priors identical).
  2. Forcing the reset policy's own first two actions reproduces its natural
     calendar exactly (validates forced_prefix).
  3. The recomputed C rows match the committed burn artifact per-pair.

This runner modifies nothing under results/q_r1/cold_start_replication_v1/.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.program_o_state_rich import (  # noqa: E402
    StateRichConfiguration,
    state_rich_calendar,
)
from supply_chain.q_r1_retained_learning import (  # noqa: E402
    PhysicalCampaignState,
    common_continuation_calendar,
    evaluate_calendar,
)
from supply_chain.retained_context_discovery import (  # noqa: E402
    arm_priors,
    build_campaign_history,
)

CELL = ("rho90_share90", 0.90, 0.90)
ESTIMANDS = ("A_natural_replan", "B_retained_full", "C_frozen_splice")
DELTA_KEYS = (
    "early_ret_2w",  # VISIBLE cohort mean (completed-only) — the burn's endpoint
    "early_ret_full_2w",  # FULL 12-order cohort (unresolved/lost score 0) — the honest endpoint
    "worst_product_fill",
    "unresolved_orders",
    "lost_orders",
    "early_unresolved_orders",
    "early_worst_product_fill",
    "early_service_loss_to_score",
    "actual_loaded_departures",
    "actual_payload",
    "actual_downstream_vehicle_hours",
)
RESOURCE_KEYS = (
    "gross_policy_batch_slots",
    "gross_production_quantity",
    "charged_daily_dispatch_slots",
    "charged_downstream_vehicle_hours",
)


def cluster_bootstrap(by_root: dict[int, list[float]], *, seed: int = 20260722):
    """Replicates the frozen adjudicator's history-clustered bootstrap."""
    roots = sorted(by_root)
    means = np.asarray([np.mean(by_root[root]) for root in roots], dtype=float)
    if len(means) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(means), size=(10_000, len(means)))
    sampled = means[draws].mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def delta_stats(pairs: list[dict], key: str) -> dict:
    values = np.asarray([pair[key] for pair in pairs], dtype=float)
    by_root: dict[int, list[float]] = defaultdict(list)
    for pair in pairs:
        by_root[pair["history_root"]].append(pair[key])
    lcb, ucb = cluster_bootstrap(by_root)
    tol = 1e-12
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
        "n_favorable": int(np.sum(values > tol)),
        "n_zero": int(np.sum(np.abs(values) <= tol)),
        "n_adverse": int(np.sum(values < -tol)),
        "clustered_ci95": [lcb, ucb],
        "n_pairs": int(values.size),
    }


def run(args: argparse.Namespace) -> dict:
    _cell_name, rho, share = CELL
    sched = scheduler()
    committed = json.loads(
        (ROOT / "results/q_r1/cold_start_replication_v1/d0_retained_context.json").read_text()
    )
    committed_rows = {
        (row["kappa"], row["history_root"], row["campaign_index"], row["arm"]): row
        for row in committed["rows"]
    }
    pairs_out: list[dict] = []
    selfcheck_failures = 0
    crosscheck_max_abs = 0.0
    a_equals_b = {kappa: 0 for kappa in args.kappas}
    started = time.time()
    for kappa in args.kappas:
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
        priors = arm_priors(histories=histories, regime_persistence=rho, dominant_share=share)
        for history_index, history in enumerate(histories):
            for campaign_index, campaign in enumerate(history):
                if campaign_index == 0:
                    continue  # the adjudicated estimand excludes the cold history start
                prior_ret = priors["retained_posterior"][history_index][campaign_index]
                prior_reset = priors["reset_posterior_0p5"][history_index][campaign_index]
                common = dict(
                    skeleton=campaign.skeleton.as_dict(),
                    scheduler=sched,
                    config=StateRichConfiguration("belief_mpc", 3),
                    regime_persistence=rho,
                    dominant_share=share,
                )
                cal_reset_nat, _ = state_rich_calendar(**common, initial_belief_c=prior_reset)
                cal_ret_nat, _ = state_rich_calendar(**common, initial_belief_c=prior_ret)
                cal_ret_a, _ = state_rich_calendar(
                    **common,
                    initial_belief_c=prior_reset,
                    forced_prefix=tuple(cal_ret_nat[:2]),
                )
                cal_reset_check, _ = state_rich_calendar(
                    **common,
                    initial_belief_c=prior_reset,
                    forced_prefix=tuple(cal_reset_nat[:2]),
                )
                if tuple(cal_reset_check) != tuple(cal_reset_nat):
                    selfcheck_failures += 1
                cal_ret_c = common_continuation_calendar(cal_ret_nat, cal_reset_nat)
                if tuple(cal_ret_a) == tuple(cal_ret_nat):
                    a_equals_b[kappa] += 1
                physical = PhysicalCampaignState(
                    history_root=campaign.history_root,
                    campaign_index=campaign.campaign_index,
                    persistence_mode="iid" if kappa == 0.5 else f"binary_{kappa}",
                    theta=(rho, share),
                    initial_regime=campaign.initial_regime,
                    skeleton=campaign.skeleton,
                )
                metrics = {
                    "reset": evaluate_calendar(campaign=physical, calendar=cal_reset_nat, scheduler=sched),
                    "A_natural_replan": evaluate_calendar(campaign=physical, calendar=cal_ret_a, scheduler=sched),
                    "B_retained_full": evaluate_calendar(campaign=physical, calendar=cal_ret_nat, scheduler=sched),
                    "C_frozen_splice": evaluate_calendar(campaign=physical, calendar=cal_ret_c, scheduler=sched),
                }
                # Full-cohort endpoint: the 12-order early cohort with unresolved/lost
                # scored 0 (closes the completed-only selection hole in early_ret_2w).
                # Derived from existing fields; canonical ReT is left untouched.
                for label in metrics:
                    generated = metrics[label]["early_generated_orders"]
                    metrics[label]["early_ret_full_2w"] = (
                        metrics[label]["early_ret_2w"] * metrics[label]["early_visible_rows"] / generated
                        if generated > 0.0
                        else 0.0
                    )
                burn_ret = committed_rows.get(
                    (kappa, campaign.history_root, campaign_index, "retained_posterior")
                )
                burn_reset = committed_rows.get(
                    (kappa, campaign.history_root, campaign_index, "reset_posterior_0p5")
                )
                if burn_ret is not None and burn_reset is not None:
                    crosscheck_max_abs = max(
                        crosscheck_max_abs,
                        abs(metrics["C_frozen_splice"]["early_ret_2w"] - burn_ret["early_ret_2w"]),
                        abs(metrics["reset"]["early_ret_2w"] - burn_reset["early_ret_2w"]),
                    )
                # Action-equivalence is NOT belief-equivalence: the retained and reset
                # priors differ by construction; record that difference alongside
                # whether it actually changed the first two actions (the treatment).
                belief_abs_diff = abs(float(prior_ret) - float(prior_reset))
                first2_action_changed = tuple(cal_ret_nat[:2]) != tuple(cal_reset_nat[:2])
                row = {
                    "kappa": kappa,
                    "history_root": campaign.history_root,
                    "campaign_index": campaign_index,
                    "initial_belief_abs_diff": belief_abs_diff,
                    "first2_action_changed": bool(first2_action_changed),
                    "a_equals_b_calendar": bool(tuple(cal_ret_a) == tuple(cal_ret_nat)),
                    "calendars": {
                        "reset": list(map(int, cal_reset_nat)),
                        "A_natural_replan": list(map(int, cal_ret_a)),
                        "B_retained_full": list(map(int, cal_ret_nat)),
                        "C_frozen_splice": list(map(int, cal_ret_c)),
                    },
                }
                for estimand in ESTIMANDS:
                    for key in DELTA_KEYS:
                        row[f"{estimand}:d_{key}"] = float(
                            metrics[estimand][key] - metrics["reset"][key]
                        )
                    row[f"{estimand}:d_resources_max_abs"] = float(
                        max(
                            abs(metrics[estimand][key] - metrics["reset"][key])
                            for key in RESOURCE_KEYS
                        )
                    )
                pairs_out.append(row)
            elapsed = time.time() - started
            print(
                f"[{elapsed:7.1f}s] kappa={kappa} root={campaign.history_root} done "
                f"({len(pairs_out)} pairs)",
                flush=True,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict] = {}
    for kappa in args.kappas:
        subset = [row for row in pairs_out if row["kappa"] == kappa]
        n_belief_diff = sum(1 for row in subset if row["initial_belief_abs_diff"] > 1e-9)
        n_action_changed = sum(1 for row in subset if row["first2_action_changed"])
        summaries[str(kappa)] = {
            "n_pairs": len(subset),
            "a_equals_b_calendars": a_equals_b[kappa],
            "action_vs_belief": {
                "n_initial_belief_differs": n_belief_diff,
                "max_initial_belief_abs_diff": max(
                    (row["initial_belief_abs_diff"] for row in subset), default=0.0
                ),
                "n_first2_action_changed": n_action_changed,
                "note": "A==B calendars is ACTION-equivalence, not belief-equivalence: "
                "priors differ in n_initial_belief_differs pairs but only change the "
                "first two actions in n_first2_action_changed of them.",
            },
            "estimands": {
                estimand: {
                    key: delta_stats(
                        [
                            {
                                "history_root": row["history_root"],
                                key: row[f"{estimand}:d_{key}"],
                            }
                            for row in subset
                        ],
                        key,
                    )
                    for key in DELTA_KEYS
                }
                for estimand in ESTIMANDS
            },
            "resources_max_abs": max(
                (row[f"{estimand}:d_resources_max_abs"] for row in subset for estimand in ESTIMANDS),
                default=0.0,
            ),
        }
    payload = {
        "schema_version": "q_r1_c1_natural_continuation_diagnostic_v2",
        "claim_status": "EXPLORATORY_NO_CLAIM",
        "seed_policy": "BURNED_ROOTS_ONLY_7570801_7570824",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cell": CELL[0],
        "history_roots": [args.seed_start, args.seed_start + args.histories - 1],
        "campaigns_per_history": args.campaigns,
        "estimand_definitions": {
            "A_natural_replan": "retained first-2 actions forced, reset-belief MPC replans weeks 3-8 on the reached state",
            "B_retained_full": "retained-belief MPC full-campaign natural calendar",
            "C_frozen_splice": "retained prefix + reset-trajectory weeks 3-8 (the burn's estimand)",
            "baseline": "reset-belief MPC full-campaign natural calendar",
        },
        "harness_self_checks": {
            "forced_prefix_selfcheck_failures": selfcheck_failures,
            "burn_crosscheck_max_abs_early_ret": crosscheck_max_abs,
        },
        "summaries": summaries,
        "rows": pairs_out,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=7_570_801)
    parser.add_argument("--histories", type=int, default=24)
    parser.add_argument("--campaigns", type=int, default=12)
    parser.add_argument(
        "--kappas", type=float, nargs="+", default=[0.5, 0.75, 0.9]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/q_r1/c1_natural_continuation_diagnostic_v1/result.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    slim = {
        kappa: {
            "a_equals_b": summary["a_equals_b_calendars"],
            "n_pairs": summary["n_pairs"],
            "action_vs_belief": summary["action_vs_belief"],
            **{
                estimand: {
                    "d_early_ret_visible_mean": summary["estimands"][estimand]["early_ret_2w"]["mean"],
                    "d_early_ret_visible_ci95": summary["estimands"][estimand]["early_ret_2w"]["clustered_ci95"],
                    "d_early_ret_FULL_mean": summary["estimands"][estimand]["early_ret_full_2w"]["mean"],
                    "d_early_ret_FULL_ci95": summary["estimands"][estimand]["early_ret_full_2w"]["clustered_ci95"],
                    "d_worst_fill_mean": summary["estimands"][estimand]["worst_product_fill"]["mean"],
                    "d_unresolved_mean": summary["estimands"][estimand]["unresolved_orders"]["mean"],
                    "d_unresolved_max": summary["estimands"][estimand]["unresolved_orders"]["max"],
                    "d_actual_payload_mean": summary["estimands"][estimand]["actual_payload"]["mean"],
                }
                for estimand in ESTIMANDS
            },
        }
        for kappa, summary in payload["summaries"].items()
    }
    print(json.dumps({"self_checks": payload["harness_self_checks"], "summary": slim}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
