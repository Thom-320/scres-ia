#!/usr/bin/env python3
"""Q-R1 Gate 1 — MPC* calibration and action-stability certificate.

EXPLORATORY_NO_CLAIM. Burned roots only. Demonstrates that exact/stratified
6-state integration (MPC*) is action-stable where the legacy Monte-Carlo MPC was
not (the audit measured p4-vs-p64 first-action agreement of 0/8), then freezes
the MPC* configuration used as the strongest-structured comparator downstream.

For a subset of burned campaigns, at the campaign-start (uniform/reset) belief,
it computes the first-two treatment-window actions under:
  - MPC* exact integration: {h3 r8, h3 r16, h4 r8}  (constraint_aware, fail-closed)
  - legacy MC MPC: {h3 scenario p16, h3 scenario p64}
and reports pairwise first-action and first-two-action agreement, fallback usage,
and online latency. MPC* self-agreement across r is the convergence check; the
MPC*-vs-legacy gap is the strengthening evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.program_t_full_des_mpc import (  # noqa: E402
    FullDEST0Config,
    ret_transducer_t0_calendar,
    joint_belief_ret_transducer_calendar,
)
from supply_chain.program_t_joint_belief import ExactJointBelief  # noqa: E402
from supply_chain.program_t_mpc_star import MPCStarConfig, mpc_star_calendar  # noqa: E402
from supply_chain.retained_context_discovery import build_campaign_history  # noqa: E402

CELL = ("rho90_share90", 0.90, 0.90)


def run(args: argparse.Namespace) -> dict:
    _name, rho, share = CELL
    sched = scheduler()
    mpc_star_configs = {
        "mpc_star_ca_h3_r8": MPCStarConfig(horizon=3, realizations_per_state=8, mode="constraint_aware"),
        "mpc_star_ca_h3_r16": MPCStarConfig(horizon=3, realizations_per_state=16, mode="constraint_aware"),
        "mpc_star_ca_h4_r8": MPCStarConfig(horizon=4, realizations_per_state=8, mode="constraint_aware"),
    }
    legacy_configs = {
        "legacy_scenario_h3_p16": FullDEST0Config(horizon=3, mode="scenario", particles=16),
        "legacy_scenario_h3_p64": FullDEST0Config(horizon=3, mode="scenario", particles=64),
    }
    rows: list[dict] = []
    started = time.time()
    for index in range(args.campaigns):
        history = build_campaign_history(
            history_root=args.seed_start + index,
            campaigns=3,
            kappa=0.9,
            scheduler=sched,
            regime_persistence=rho,
            dominant_share=share,
        )
        campaign = history[1]  # a non-initial campaign
        belief = ExactJointBelief.uniform()
        row: dict = {"history_root": campaign.history_root, "campaign_index": 1}
        for name, cfg in mpc_star_configs.items():
            calendar, diag = mpc_star_calendar(
                skeleton=campaign.skeleton, scheduler=sched, config=cfg, belief=belief.copy()
            )
            row[name] = {
                "first2": list(map(int, calendar[:2])),
                "calendar": list(map(int, calendar)),
                "fallbacks": int(diag["fallbacks"]),
                "online_ms": float(diag["online_ms"]),
            }
        for name, cfg in legacy_configs.items():
            calendar, diag = joint_belief_ret_transducer_calendar(
                skeleton=campaign.skeleton, scheduler=sched, config=cfg, belief=belief.copy()
            )
            row[name] = {
                "first2": list(map(int, calendar[:2])),
                "calendar": list(map(int, calendar)),
                "online_ms": float(diag["online_ms"]),
            }
        rows.append(row)
        print(
            f"[{time.time() - started:7.1f}s] root={campaign.history_root} "
            f"({index + 1}/{args.campaigns}) done",
            flush=True,
        )

    names = list(mpc_star_configs) + list(legacy_configs)
    agreement: dict[str, dict[str, dict]] = {}
    for a in names:
        agreement[a] = {}
        for b in names:
            same_first = sum(1 for r in rows if r[a]["first2"][0] == r[b]["first2"][0])
            same_first2 = sum(1 for r in rows if r[a]["first2"] == r[b]["first2"])
            agreement[a][b] = {
                "first_action_agree": f"{same_first}/{len(rows)}",
                "first2_agree": f"{same_first2}/{len(rows)}",
            }
    fallbacks = {
        name: sum(r[name]["fallbacks"] for r in rows)
        for name in mpc_star_configs
    }
    median_ms = {
        name: sorted(r[name]["online_ms"] for r in rows)[len(rows) // 2] for name in names
    }
    return {
        "schema_version": "q_r1_gate1_mpc_star_calibration_v1",
        "claim_status": "EXPLORATORY_NO_CLAIM",
        "seed_policy": "BURNED_ROOTS_ONLY",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cell": CELL[0],
        "belief": "uniform_reset_campaign_start",
        "n_campaigns": len(rows),
        "configs": {
            "mpc_star": {k: v.config_id for k, v in mpc_star_configs.items()},
            "legacy": {k: v.config_id for k, v in legacy_configs.items()},
        },
        "first2_action_agreement_matrix": agreement,
        "mpc_star_fallback_totals": fallbacks,
        "median_online_ms": median_ms,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=7_570_801)
    parser.add_argument("--campaigns", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/q_r1/gate1_mpc_star_calibration_v1/result.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "agreement": payload["first2_action_agreement_matrix"],
                "fallbacks": payload["mpc_star_fallback_totals"],
                "median_online_ms": payload["median_online_ms"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
