#!/usr/bin/env python3
"""Build a risk-event ledger and a per-risk-category event-study for frozen
Track B policies (PPO+MLP, Real-KAN).

This is the "cheap logging step" agreed before touching H4 or reward design:
the existing prevention audit (``audit_track_b_prevention_mechanism.py``)
anchors on a *proxy* (first regime transition or forecast crossing 0.20 in the
whole episode) which is indirect and gives only one anchor per episode. Here
we anchor directly on ``sim.risk_events`` (real risk_id + start_time from the
DES itself), stratified by empirical frequency:

- frequent/learnable: R11, R13, R24 (many occurrences per 104-week episode)
- frequent/rate-like: R14 (many defect events; analyse separately because it can
  numerically dominate discrete event counts)
- intermediate: R12, R22, R23
- rare/black-swan: R21, R3 (near-zero occurrences per episode; stress-test only)

For frequent risks this gives dozens to hundreds of anchors per episode
instead of one — far more statistical power than the old regime-proxy anchor.
No retraining: this only replays already-trained frozen policies.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_track_b_prevention_mechanism import (  # noqa: E402
    ACTION_DIMS,
    EVAL_EPISODE_SEED_OFFSET,
    PolicyRuntime,
    env_kwargs,
    load_runtime,
    mean,
    predict_action,
    row_from_step,
    save_csv,
)
from supply_chain.external_env_interface import make_track_b_env  # noqa: E402

RISK_CATEGORY = {
    "R11": "frecuente",
    "R13": "frecuente",
    "R14": "frecuente_tasa",
    "R24": "frecuente",
    "R12": "intermedio",
    "R22": "intermedio",
    "R23": "intermedio",
    "R21": "raro",
    "R3": "raro",
}

# Expected events/year at the "current" (R1) risk level, from the occurrence
# distributions in supply_chain/config.py (RISKS_CURRENT), mean inter-arrival
# time converted to an annual rate. Recorded here for the sanity check against
# what the ledger actually observes.
EXPECTED_EVENTS_PER_YEAR_CURRENT = {
    "R11": 8760 / ((1 + 168) / 2),
    "R12": 2 * (12 * (1 / 11)),
    "R13": 12 * (12 * (1 / 10)),
    "R21": 8760 / ((1 + 16_128) / 2),
    "R22": 8760 / ((1 + 4_032) / 2),
    "R23": 8760 / ((1 + 8_064) / 2),
    "R24": 8760 / ((1 + 672) / 2),
    "R3": 8760 / ((1 + 161_280) / 2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/track_b_risk_event_ledger_2026-07-03"),
    )
    parser.add_argument("--policies", nargs="+", default=["ppo_mlp", "real_kan"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=104)
    parser.add_argument("--reward-mode", default="control_v1")
    parser.add_argument("--risk-level", default="adaptive_benchmark_v2")
    parser.add_argument("--observation-version", default="v7")
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument(
        "--ppo-bundles",
        nargs="+",
        type=Path,
        default=[
            Path("outputs/experiments/track_b_ablation_8d_final_2026-07-01/joint"),
            Path(
                "outputs/experiments/track_b_gain_2026-06-30/"
                "top_tier_confirm_v3_output/track_b_top_tier_confirm_5seed_60k_h104"
            ),
            Path(
                "outputs/experiments/track_b_seed_expansion_2026-07-02/"
                "track_b_seed_expansion_6_10_claude"
            ),
        ],
    )
    parser.add_argument(
        "--real-kan-bundles",
        nargs="+",
        type=Path,
        default=[
            Path("outputs/experiments/track_b_real_kan_sidecar_2026-07-03/confirm_5seed_60k_h104"),
            Path(
                "outputs/experiments/track_b_real_kan_sidecar_2026-07-03/"
                "confirm_10seed_extension_6_10_60k_h104"
            ),
        ],
    )
    return parser.parse_args()


def _env_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return env_kwargs(args)


def run_episode_with_risk_ledger(
    *,
    runtime: PolicyRuntime,
    args: argparse.Namespace,
    seed: int,
    episode: int,
    eval_seed: int,
) -> dict[str, Any]:
    env = make_track_b_env(**_env_kwargs(args))
    obs, info = env.reset(seed=eval_seed)
    terminated = False
    truncated = False
    step = 0
    rows: list[dict[str, Any]] = []
    final_info = info

    while not (terminated or truncated):
        obs_before = np.asarray(obs, dtype=np.float32).copy()
        action = predict_action(runtime, obs_before)
        obs, reward, terminated, truncated, final_info = env.step(action)
        rows.append(
            row_from_step(
                policy=runtime.name,
                seed=seed,
                episode=episode,
                eval_seed=eval_seed,
                condition="full",
                step=step,
                obs_before=obs_before,
                reward=float(reward),
                info=final_info,
            )
        )
        step += 1

    risk_events = list(env.unwrapped.sim.risk_events)
    risk_rows = [
        {
            "policy": runtime.name,
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
    return {"step_rows": rows, "risk_rows": risk_rows}


def event_aligned_action_study(
    step_rows: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
    *,
    window: tuple[int, int] = (-4, 8),
    by_risk_id: bool = False,
) -> list[dict[str, Any]]:
    """Align action_intensity to EVERY real risk-event onset (not one proxy
    anchor per episode), grouped by policy x category x relative week."""
    by_episode_steps: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in step_rows:
        key = (row["policy"], row["seed"], row["episode"])
        by_episode_steps.setdefault(key, []).append(row)
    for key in by_episode_steps:
        by_episode_steps[key].sort(key=lambda r: r["step"])

    buckets: dict[tuple[str, str, str, int], list[float]] = {}
    for ev in risk_rows:
        key = (ev["policy"], ev["seed"], ev["episode"])
        steps = by_episode_steps.get(key)
        if not steps:
            continue
        anchor = ev["start_step"]
        category = ev["category"]
        for row in steps:
            rel = int(row["step"]) - anchor
            if window[0] <= rel <= window[1]:
                risk_id = str(ev["risk_id"]) if by_risk_id else "ALL"
                bucket_key = (ev["policy"], risk_id, category, rel)
                buckets.setdefault(bucket_key, []).append(float(row["action_intensity"]))

    out = []
    for (policy, risk_id, category, rel), values in sorted(buckets.items()):
        out.append(
            {
                "policy": policy,
                "risk_id": risk_id,
                "category": category,
                "relative_week": rel,
                "n_event_observations": len(values),
                "action_intensity_mean": mean(values),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    all_step_rows: list[dict[str, Any]] = []
    all_risk_rows: list[dict[str, Any]] = []

    for policy in args.policies:
        for seed in args.seeds:
            runtime = load_runtime(policy, seed, args)
            for episode in range(1, int(args.eval_episodes) + 1):
                eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + (episode - 1)
                result = run_episode_with_risk_ledger(
                    runtime=runtime,
                    args=args,
                    seed=seed,
                    episode=episode,
                    eval_seed=eval_seed,
                )
                all_step_rows.extend(result["step_rows"])
                all_risk_rows.extend(result["risk_rows"])

    save_csv(out / "risk_event_ledger.csv", all_risk_rows)

    # Frequency sanity check: observed events/year vs. expected at the
    # "current" (R1) base rate. adaptive_benchmark_v2 multiplies these by a
    # time-varying regime intensity (0.90x-1.85x) plus, for R22/R23/R24, an
    # extra fixed multiplier -- so observed rates should run a bit ABOVE the
    # R1 baseline, not match it exactly.
    n_episode_years = len(args.seeds) * args.eval_episodes * len(args.policies) * (
        int(args.max_steps) * float(args.step_size_hours) / 8760.0
    )
    freq_summary = []
    for risk_id, category in RISK_CATEGORY.items():
        n_obs = sum(1 for r in all_risk_rows if r["risk_id"] == risk_id)
        observed_per_year = n_obs / n_episode_years if n_episode_years > 0 else 0.0
        freq_summary.append(
            {
                "risk_id": risk_id,
                "category": category,
                "n_observed": n_obs,
                "observed_events_per_year": observed_per_year,
                "expected_events_per_year_current_R1": EXPECTED_EVENTS_PER_YEAR_CURRENT.get(risk_id, ""),
            }
        )
    save_csv(out / "risk_frequency_check.csv", freq_summary)

    event_study_rows = event_aligned_action_study(all_step_rows, all_risk_rows)
    save_csv(out / "risk_event_aligned_action_study.csv", event_study_rows)
    event_study_by_risk_rows = event_aligned_action_study(
        all_step_rows, all_risk_rows, by_risk_id=True
    )
    save_csv(out / "risk_event_aligned_by_risk_study.csv", event_study_by_risk_rows)

    print(f"Wrote risk event ledger bundle to {out}")
    print(f"Total risk events logged: {len(all_risk_rows)}")
    for row in freq_summary:
        expected = row["expected_events_per_year_current_R1"]
        expected_str = f"{expected:.2f}/yr" if isinstance(expected, (float, int)) else "n/a"
        print(
            f"{row['risk_id']:>4s} ({row['category']:>14s}): observed {row['observed_events_per_year']:.2f}/yr "
            f"vs expected(R1) {expected_str}, n={row['n_observed']}"
        )


if __name__ == "__main__":
    main()
