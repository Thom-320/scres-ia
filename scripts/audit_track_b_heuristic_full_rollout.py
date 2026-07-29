#!/usr/bin/env python3
"""Evaluate Track B heuristic references as FULL, independent rollouts
(no mid-episode splice) under the same protocol used for the trained
policies and the risk-event ledger, so they can be compared honestly.

This exists because ``R_full - R_reset(w)`` (splicing a reference action
into a frozen policy's trajectory and continuing) was found to be invalid
for this simulator: ``_adaptive_regime_controller`` draws from the SHARED
``self.rng`` stream (also consumed by processing-time noise and, for
Track B, literally the SAME object as ``risk_rng``/``demand_rng`` since
Track B's env never requests ``strict_exogenous_crn=True``). Substituting
an action changes ``self.rng`` consumption immediately, which changes the
realized risk calendar for the rest of the episode -- contaminating any
post-window ReT delta with real risk-exposure differences, not just
behavioral ones. See
docs/TRACK_B_COUNTERFACTUAL_RNG_ENTANGLEMENT_FINDING_2026-07-03.md.

The valid alternative: run each heuristic as its own full policy, fresh
reset per episode, same eval-seed protocol as the trained-policy ledger
run. This is exactly the CRN-paired methodology already used everywhere
else in this project (E1-E6, no-forecast confirm) -- no splicing, no
future-event anchoring, no invalid assumption about shared risk calendars
across conditions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_track_b_prevention_mechanism import (  # noqa: E402
    EVAL_EPISODE_SEED_OFFSET,
    env_kwargs,
    finalize_episode,
    mean,
    row_from_step,
    save_csv,
    window_label,
)
from scripts.audit_track_b_risk_event_ledger import (  # noqa: E402
    RISK_CATEGORY,
    event_aligned_action_study,
)
from scripts.track_b_heuristics import make_heuristic_defaults  # noqa: E402
from supply_chain.external_env_interface import make_track_b_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/track_b_heuristic_full_rollout_2026-07-03"),
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=[
            "heur_forecast_threshold",
            "heur_downstream_reactive",
            "heur_s1_max_downstream",
        ],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=104)
    parser.add_argument("--reward-mode", default="control_v1")
    parser.add_argument("--risk-level", default="adaptive_benchmark_v2")
    parser.add_argument("--observation-version", default="v7")
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    return parser.parse_args()


def run_heuristic_episode(
    *,
    heuristic_name: str,
    heuristic: Any,
    args: argparse.Namespace,
    seed: int,
    episode: int,
    eval_seed: int,
) -> dict[str, Any]:
    env = make_track_b_env(**env_kwargs(args))
    heuristic.reset()
    obs, info = env.reset(seed=eval_seed)
    terminated = False
    truncated = False
    step = 0
    rows: list[dict[str, Any]] = []
    shifts: list[int] = []

    while not (terminated or truncated):
        obs_before = np.asarray(obs, dtype=np.float32).copy()
        action = heuristic(obs_before, info)
        obs, reward, terminated, truncated, info = env.step(action)
        shifts.append(int(info.get("shifts_active", 1)))
        rows.append(
            row_from_step(
                policy=heuristic_name,
                seed=seed,
                episode=episode,
                eval_seed=eval_seed,
                condition="full",
                step=step,
                obs_before=obs_before,
                reward=float(reward),
                info=info,
            )
        )
        step += 1

    ret_excel, fill_rate, service_loss_auc, cost_index = finalize_episode(env, shifts)
    risk_events = list(env.unwrapped.sim.risk_events)
    risk_rows = [
        {
            "policy": heuristic_name,
            "seed": seed,
            "episode": episode,
            "eval_seed": eval_seed,
            "risk_id": ev.risk_id,
            "category": RISK_CATEGORY.get(ev.risk_id, "sin_categoria"),
            "start_time_hours": float(ev.start_time),
            "end_time_hours": float(ev.end_time),
            "duration_hours": float(ev.duration),
            "start_step": int(ev.start_time // float(args.step_size_hours)),
            "affected_ops": ",".join(str(op) for op in ev.affected_ops),
            "magnitude": float(ev.magnitude),
        }
        for ev in risk_events
    ]
    env.close()
    for row in rows:
        row["step_absolute_hours"] = row["step"] * float(args.step_size_hours)

    return {
        "step_rows": rows,
        "risk_rows": risk_rows,
        "episode_summary": {
            "policy": heuristic_name,
            "seed": seed,
            "episode": episode,
            "eval_seed": eval_seed,
            "order_ret_excel": ret_excel,
            "fill_rate": fill_rate,
            "service_loss_auc": service_loss_auc,
            "cost_index": cost_index,
            "n_steps": step,
            "n_risk_events": len(risk_rows),
        },
    }


def summarize_policy(episode_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for row in episode_summaries:
        by_policy.setdefault(row["policy"], []).append(row)
    out = []
    for policy, rows in sorted(by_policy.items()):
        ret_vals = [float(r["order_ret_excel"]) for r in rows]
        n = len(ret_vals)
        m = mean(ret_vals)
        std = float(np.std(ret_vals, ddof=1)) if n > 1 else 0.0
        sem = std / (n ** 0.5) if n > 1 else 0.0
        ci95 = 1.96 * sem
        out.append(
            {
                "policy": policy,
                "n_episodes": n,
                "order_ret_excel_mean": m,
                "order_ret_excel_std": std,
                "order_ret_excel_ci95_low": m - ci95,
                "order_ret_excel_ci95_high": m + ci95,
                "fill_rate_mean": mean([float(r["fill_rate"]) for r in rows]),
                "cost_index_mean": mean([float(r["cost_index"]) for r in rows]),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    heuristics = make_heuristic_defaults()
    unknown = [p for p in args.policies if p not in heuristics]
    if unknown:
        raise SystemExit(f"Unknown heuristic policies: {unknown}. Available: {list(heuristics)}")

    all_step_rows: list[dict[str, Any]] = []
    all_risk_rows: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []

    for policy in args.policies:
        heuristic = heuristics[policy]
        for seed in args.seeds:
            for episode in range(1, int(args.eval_episodes) + 1):
                eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + (episode - 1)
                result = run_heuristic_episode(
                    heuristic_name=policy,
                    heuristic=heuristic,
                    args=args,
                    seed=seed,
                    episode=episode,
                    eval_seed=eval_seed,
                )
                all_step_rows.extend(result["step_rows"])
                all_risk_rows.extend(result["risk_rows"])
                episode_summaries.append(result["episode_summary"])

    save_csv(out / "episode_summary.csv", episode_summaries)
    policy_summary = summarize_policy(episode_summaries)
    save_csv(out / "policy_summary.csv", policy_summary)
    save_csv(out / "risk_event_ledger.csv", all_risk_rows)
    save_csv(out / "step_ledger_full.csv", all_step_rows)

    event_study_rows = event_aligned_action_study(all_step_rows, all_risk_rows)
    save_csv(out / "risk_event_aligned_action_study.csv", event_study_rows)
    event_study_by_risk_rows = event_aligned_action_study(
        all_step_rows, all_risk_rows, by_risk_id=True
    )
    save_csv(out / "risk_event_aligned_by_risk_study.csv", event_study_by_risk_rows)

    print(f"Wrote heuristic full-rollout bundle to {out}")
    for row in policy_summary:
        print(
            f"{row['policy']:>28s}: order_ret_excel_mean={row['order_ret_excel_mean']:.6f} "
            f"CI95=[{row['order_ret_excel_ci95_low']:.6f},{row['order_ret_excel_ci95_high']:.6f}] "
            f"fill_rate={row['fill_rate_mean']:.4f} cost_index={row['cost_index_mean']:.4f} "
            f"n={row['n_episodes']}"
        )


if __name__ == "__main__":
    main()
