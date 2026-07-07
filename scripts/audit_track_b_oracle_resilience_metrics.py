#!/usr/bin/env python3
"""Resilience-metric follow-up to the oracle prevention-ceiling gate.

``audit_track_b_oracle_prevention_ceiling.py`` found the best oracle
configuration per target risk set cannot move mean ReT Excel by more than
+0.000021 -- far below the +0.0004 signal margin. But ReT Excel is a single
cost-weighted composite averaged over every order in the episode; a real
resilience gain concentrated in the handful of orders touched by R22/R24
could be invisible in that aggregate.

This script reruns only the already-identified best configuration per target
set (plus the B=0 baseline) and reports the resilience-specific fields that
``supply_chain/episode_metrics.py::compute_episode_metrics`` already computes
but the prevention/oracle scripts have been discarding: ``ttr_mean``,
``ttr_p95`` (time-to-recovery), ``ret_excel_cvar05`` (tail mean of the worst
5% of orders), ``ret_excel_rolling_4w_min`` (worst 4-week window), and
``backlog_age_mean/max``. If none of these move either, the "no ceiling"
verdict holds at the mechanism level, not just the aggregate-scalar level.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_track_b_oracle_prevention_ceiling import (  # noqa: E402
    EVAL_EPISODE_SEED_OFFSET,
    RiskEvent,
    build_boost_vector,
    env_kwargs,
    load_runtime,
    make_track_b_env,
    mean,
    parse_target_set,
    predict_action,
    prep_window_steps,
)
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("outputs/experiments/track_b_oracle_resilience_metrics_2026-07-04")
DEFAULT_PPO_BUNDLES = (
    Path("outputs/experiments/track_b_fixed_rng_confirm_5seed_60k_2026-07-03"),
)
RESILIENCE_FIELDS = (
    "ret_excel",
    "ret_excel_cvar05",
    "ret_excel_rolling_4w_min",
    "service_loss_auc_per_order",
    "ttr_mean",
    "ttr_p95",
    "backlog_age_mean",
    "backlog_age_max",
    "fill_rate",
)


@dataclass
class Condition:
    name: str
    target_ids: set[str]
    lead_weeks: int
    boost: float


DEFAULT_CONDITIONS = (
    Condition(name="baseline_B0", target_ids=set(), lead_weeks=0, boost=0.0),
    Condition(name="oracle_R22_L8_B0.75", target_ids={"R22"}, lead_weeks=8, boost=0.75),
    Condition(name="oracle_R24_L8_B1.0", target_ids={"R24"}, lead_weeks=8, boost=1.0),
    Condition(name="oracle_R22+R24_L8_B1.0", target_ids={"R22", "R24"}, lead_weeks=8, boost=1.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=104)
    parser.add_argument("--reward-mode", default="control_v1")
    parser.add_argument("--risk-level", default="adaptive_benchmark_v2")
    parser.add_argument("--observation-version", default="v7")
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument("--ppo-bundles", nargs="+", type=Path, default=list(DEFAULT_PPO_BUNDLES))
    parser.add_argument("--boost-dims", nargs="+", default=["shift", "op10_q", "op12_q"])
    return parser.parse_args()


def run_episode_full_metrics(
    *,
    runtime: Any,
    args: argparse.Namespace,
    eval_seed: int,
    boost_steps: set[int] | None,
    boost_vector: np.ndarray | None,
) -> tuple[dict[str, float], int, list[RiskEvent]]:
    env = make_track_b_env(**env_kwargs(args))
    obs, _info = env.reset(seed=eval_seed)
    terminated = False
    truncated = False
    step = 0
    while not (terminated or truncated):
        obs_before = np.asarray(obs, dtype=np.float32).copy()
        action = predict_action(runtime, obs_before)
        if boost_steps is not None and boost_vector is not None and step in boost_steps:
            action = np.clip(action + boost_vector, -1.0, 1.0).astype(np.float32)
        obs, _reward, terminated, truncated, _info = env.step(action)
        step += 1
    metrics = compute_episode_metrics(env.unwrapped.sim)
    risk_events = [
        RiskEvent(
            risk_id=str(ev.risk_id),
            start_step=int(float(ev.start_time) // float(args.step_size_hours)),
        )
        for ev in env.unwrapped.sim.risk_events
    ]
    env.close()
    return metrics, step, risk_events


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    runtimes = {seed: load_runtime("ppo_mlp", seed, args) for seed in args.seeds}

    # First pass: discover the risk-event calendar per (seed, episode) using the
    # unmodified policy -- this is also the baseline_B0 condition itself.
    calendars: dict[tuple[int, int], tuple[list[RiskEvent], int]] = {}
    per_episode_rows: list[dict[str, Any]] = []

    for cond in DEFAULT_CONDITIONS:
        for seed in args.seeds:
            runtime = runtimes[seed]
            for episode in range(1, int(args.eval_episodes) + 1):
                eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + (episode - 1)
                if cond.name == "baseline_B0":
                    metrics, steps, risk_events = run_episode_full_metrics(
                        runtime=runtime, args=args, eval_seed=eval_seed,
                        boost_steps=None, boost_vector=None,
                    )
                    calendars[(seed, episode)] = (risk_events, steps)
                else:
                    risk_events, steps = calendars[(seed, episode)]
                    boost_steps = prep_window_steps(
                        risk_events, target_risk_ids=cond.target_ids,
                        lead_weeks=cond.lead_weeks, max_steps=steps,
                    )
                    boost_vector = build_boost_vector(args.boost_dims, cond.boost)
                    metrics, steps, _events = run_episode_full_metrics(
                        runtime=runtime, args=args, eval_seed=eval_seed,
                        boost_steps=boost_steps, boost_vector=boost_vector,
                    )
                row = {"condition": cond.name, "seed": seed, "episode": episode, "eval_seed": eval_seed}
                for field in RESILIENCE_FIELDS:
                    row[field] = float(metrics.get(field, 0.0))
                per_episode_rows.append(row)
        print(f"[done] condition={cond.name}")

    save_csv(out / "resilience_per_episode.csv", per_episode_rows)

    summary_rows: list[dict[str, Any]] = []
    for cond in DEFAULT_CONDITIONS:
        rows = [r for r in per_episode_rows if r["condition"] == cond.name]
        agg = {"condition": cond.name, "n_episodes": len(rows)}
        for field in RESILIENCE_FIELDS:
            agg[f"{field}_mean"] = mean([r[field] for r in rows])
        summary_rows.append(agg)

    save_csv(out / "resilience_summary.csv", summary_rows)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "conditions": summary_rows,
                "config": vars(args) | {"ppo_bundles": [str(p) for p in args.ppo_bundles], "output_dir": str(out)},
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    baseline = next(r for r in summary_rows if r["condition"] == "baseline_B0")
    print("\n=== Resilience metrics: baseline vs. best oracle per target set ===")
    header = ["condition"] + [f"{f}_mean" for f in RESILIENCE_FIELDS]
    print(" | ".join(header))
    for row in summary_rows:
        print(" | ".join(f"{row[h]:.4f}" if isinstance(row[h], float) else str(row[h]) for h in header))
    print(f"\nWrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
