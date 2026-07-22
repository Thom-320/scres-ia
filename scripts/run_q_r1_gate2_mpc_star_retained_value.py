#!/usr/bin/env python3
"""Q-R1 Gate 2 — retained parametric value against the frozen MPC*.

EXPLORATORY_NO_CLAIM. Burned roots only. Uses the frozen MPC*
(contracts/mpc_star_frozen_v1.json) as the controller on BOTH arms and asks:
does retained parametric (theta) knowledge still improve full-cohort cold-start
ReT when the strongest structured controller consumes it?

  retained arm : MPC* with the between-campaign-retained joint posterior
  reset arm    : MPC* with the uniform (physical-reset) joint posterior

Both arms share the immutable skeleton and reset the latent regime to 0.5 (the
joint carrier retains only theta); the ONLY difference is the theta marginal, so
the delta isolates retained parametric knowledge captured by the strongest MPC.
Primary endpoint is the honest full-cohort early ReT (Gate 0). iid persistence is
the algebraic null.
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
from supply_chain.program_t_joint_belief import ExactJointBelief  # noqa: E402
from supply_chain.program_t_mpc_star import MPCStarConfig, mpc_star_calendar  # noqa: E402
from supply_chain.q_r1_retained_learning import (  # noqa: E402
    PhysicalCampaignState,
    build_parameter_history,
    evaluate_calendar,
    retained_belief_path,
)

CELL = ("rho90_share90", 0.90, 0.90)
FROZEN_MPC_STAR = MPCStarConfig(horizon=3, realizations_per_state=8, mode="constraint_aware")


def cluster_bootstrap(by_root: dict[int, list[float]], *, seed: int = 20260722):
    roots = sorted(by_root)
    means = np.asarray([np.mean(by_root[root]) for root in roots], dtype=float)
    if len(means) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(means), size=(10_000, len(means)))
    sampled = means[draws].mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def _full_cohort(metrics: dict) -> float:
    generated = metrics["early_generated_orders"]
    if generated <= 0.0:
        return 0.0
    return metrics["early_ret_2w"] * metrics["early_visible_rows"] / generated


def run(args: argparse.Namespace) -> dict:
    _name, rho, share = CELL
    sched = scheduler()
    modes = {"persistent_0p90": "persistent_0p90", "iid": "iid"}
    rows: list[dict] = []
    started = time.time()
    for mode_label, persistence_mode in modes.items():
        for index in range(args.histories):
            history = build_parameter_history(
                history_root=args.seed_start + index,
                campaigns=12,
                persistence_mode=persistence_mode,
                scheduler=sched,
            )
            retained = retained_belief_path(history)
            for campaign_index in args.campaigns:
                campaign = history[campaign_index]
                physical = PhysicalCampaignState(
                    history_root=campaign.history_root,
                    campaign_index=campaign.campaign_index,
                    persistence_mode=persistence_mode,
                    theta=campaign.theta,
                    initial_regime=campaign.initial_regime,
                    skeleton=campaign.skeleton,
                )
                cal_ret, diag_ret = mpc_star_calendar(
                    skeleton=campaign.skeleton, scheduler=sched,
                    config=FROZEN_MPC_STAR, belief=retained[campaign_index].copy(),
                )
                cal_reset, diag_reset = mpc_star_calendar(
                    skeleton=campaign.skeleton, scheduler=sched,
                    config=FROZEN_MPC_STAR, belief=ExactJointBelief.uniform(),
                )
                m_ret = evaluate_calendar(campaign=physical, calendar=cal_ret, scheduler=sched)
                m_reset = evaluate_calendar(campaign=physical, calendar=cal_reset, scheduler=sched)
                rows.append({
                    "mode": mode_label,
                    "history_root": campaign.history_root,
                    "campaign_index": campaign_index,
                    "first2_changed": tuple(cal_ret[:2]) != tuple(cal_reset[:2]),
                    "d_early_ret_full": _full_cohort(m_ret) - _full_cohort(m_reset),
                    "d_early_ret_visible": m_ret["early_ret_2w"] - m_reset["early_ret_2w"],
                    "d_worst_product_fill": m_ret["worst_product_fill"] - m_reset["worst_product_fill"],
                    "d_unresolved_orders": m_ret["unresolved_orders"] - m_reset["unresolved_orders"],
                    "d_lost_orders": m_ret["lost_orders"] - m_reset["lost_orders"],
                    "retained_fallbacks": int(diag_ret["fallbacks"]),
                    "reset_fallbacks": int(diag_reset["fallbacks"]),
                    "resources_max_abs": max(
                        abs(m_ret[k] - m_reset[k]) for k in (
                            "gross_policy_batch_slots", "gross_production_quantity",
                            "charged_daily_dispatch_slots", "charged_downstream_vehicle_hours",
                        )
                    ),
                })
            print(f"[{time.time() - started:7.1f}s] {mode_label} root={campaign.history_root} "
                  f"({len(rows)} rows)", flush=True)

    def summarize(subset: list[dict], key: str) -> dict:
        values = np.asarray([r[key] for r in subset], dtype=float)
        by_root: dict[int, list[float]] = defaultdict(list)
        for r in subset:
            by_root[r["history_root"]].append(r[key])
        lcb, ucb = cluster_bootstrap(by_root)
        tol = 1e-12
        return {
            "mean": float(np.mean(values)), "median": float(np.median(values)),
            "clustered_ci95": [lcb, ucb],
            "n_favorable": int(np.sum(values > tol)), "n_zero": int(np.sum(np.abs(values) <= tol)),
            "n_adverse": int(np.sum(values < -tol)), "n": int(values.size),
        }

    summaries = {}
    for mode_label in modes:
        subset = [r for r in rows if r["mode"] == mode_label]
        summaries[mode_label] = {
            "n_pairs": len(subset),
            "n_first2_changed": sum(1 for r in subset if r["first2_changed"]),
            "d_early_ret_full": summarize(subset, "d_early_ret_full"),
            "d_early_ret_visible": summarize(subset, "d_early_ret_visible"),
            "d_worst_product_fill": summarize(subset, "d_worst_product_fill"),
            "d_unresolved_orders": summarize(subset, "d_unresolved_orders"),
            "resources_max_abs": max((r["resources_max_abs"] for r in subset), default=0.0),
            "total_fallbacks": sum(r["retained_fallbacks"] + r["reset_fallbacks"] for r in subset),
        }
    return {
        "schema_version": "q_r1_gate2_mpc_star_retained_value_v1",
        "claim_status": "EXPLORATORY_NO_CLAIM",
        "seed_policy": "BURNED_ROOTS_ONLY",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cell": CELL[0],
        "controller": "frozen MPC* (contracts/mpc_star_frozen_v1.json): constraint_aware exact-stratified h3 r8",
        "carrier": "parametric_theta_joint_posterior (regime reset to 0.5 both arms)",
        "primary_endpoint": "d_early_ret_full (full 12-order cohort, unresolved/lost=0)",
        "campaign_indices": args.campaigns,
        "histories": args.histories,
        "summaries": summaries,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=7_570_801)
    parser.add_argument("--histories", type=int, default=24)
    parser.add_argument("--campaigns", type=int, nargs="+", default=[6, 11])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/q_r1/gate2_mpc_star_retained_value_v1/result.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    slim = {
        mode: {
            "n_pairs": s["n_pairs"], "n_first2_changed": s["n_first2_changed"],
            "d_full_mean": round(s["d_early_ret_full"]["mean"], 6),
            "d_full_ci95": [round(x, 5) for x in s["d_early_ret_full"]["clustered_ci95"]],
            "fav/zero/adv": f"{s['d_early_ret_full']['n_favorable']}/{s['d_early_ret_full']['n_zero']}/{s['d_early_ret_full']['n_adverse']}",
            "d_unresolved_mean": round(s["d_unresolved_orders"]["mean"], 4),
            "resources_max_abs": s["resources_max_abs"], "fallbacks": s["total_fallbacks"],
        }
        for mode, s in payload["summaries"].items()
    }
    print(json.dumps(slim, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
