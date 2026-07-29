#!/usr/bin/env python3
"""Oracle prevention-ceiling gate for Track B (fixed-RNG).

Question: even with PERFECT foreknowledge of exactly when a target risk
(R22/R24) will occur -- the real DES risk calendar for a fixed eval seed,
which is action-independent under ``strict_exogenous_crn=True`` (verified
this session for every discrete risk except R14) -- can boosting the
canonical PPO+MLP v7 fixed-RNG policy's {shift, op10_q, op12_q} posture
ahead of a known event buy any Garrido/Excel ReT beyond what that policy
already achieves reactively?

This is the cheapest possible test before committing to RL-constrained /
two-level-policy / Ruta-B end-to-end auxiliary-loss alternatives (see
``docs/TRACK_B_PREVENTIVE_LEARNING_ROOT_CAUSE_AND_ALTERNATIVES_2026-07-04.md``):
it touches neither the environment nor the training pipeline, only replays
an already-trained checkpoint with a privileged action override.

Design, per (seed, episode):
  1. Discovery/baseline pass: run the canonical policy unmodified (no
     boost), capture the real ``sim.risk_events`` calendar. This IS the
     B=0 reference point (the already-known canonical ReT Excel).
  2. Build the union of "prep window" steps: L weeks strictly before every
     target-risk event start (steps ``[anchor-L, anchor-1]``).
  3. Oracle pass: same eval seed (same DES event calendar), same policy
     predictions recomputed from the (possibly-diverging) live observation
     at every step, EXCEPT: in prep-window steps, add a boost B to the
     {shift, op10_q, op12_q} action dims before clipping to [-1, 1].
  4. Sweep lead L and boost B; report the best oracle ReT Excel against
     the unmodified baseline.

If the best oracle configuration cannot beat the baseline by a margin
comparable to already-detected effects in this project (~+0.0004), there is
no ReT ceiling that any prevention-focused technique could realistically
capture under this reward/environment -- the next real move is changing the
environment's reactive-vs-preventive economics, not more reward/architecture
engineering on top of the current one.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_track_b_prevention_mechanism import (  # noqa: E402
    ACTION_DIMS,
    EVAL_EPISODE_SEED_OFFSET,
    PolicyRuntime,
    env_kwargs,
    finalize_episode,
    load_runtime,
    mean,
)
from scripts.audit_track_b_risk_event_ledger import RISK_CATEGORY  # noqa: E402
from supply_chain.external_env_interface import make_track_b_env  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("outputs/experiments/track_b_oracle_prevention_ceiling_2026-07-04")
DEFAULT_PPO_BUNDLES = (
    Path("outputs/experiments/track_b_fixed_rng_confirm_5seed_60k_2026-07-03"),
)
DEFAULT_TARGET_SETS = ("R22", "R24", "R22+R24")
DEFAULT_LEAD_WEEKS = (1, 2, 4, 8)
DEFAULT_BOOSTS = (0.25, 0.5, 0.75, 1.0)
DEFAULT_BOOST_DIMS = ("shift", "op10_q", "op12_q")


@dataclass
class RiskEvent:
    risk_id: str
    start_step: int


@dataclass
class BaselineEpisode:
    seed: int
    episode: int
    eval_seed: int
    ret_excel: float
    fill_rate: float
    cost_index: float
    steps: int
    risk_events: list[RiskEvent]


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
    parser.add_argument("--target-sets", nargs="+", default=list(DEFAULT_TARGET_SETS))
    parser.add_argument("--lead-weeks", nargs="+", type=int, default=list(DEFAULT_LEAD_WEEKS))
    parser.add_argument("--boosts", nargs="+", type=float, default=list(DEFAULT_BOOSTS))
    parser.add_argument("--boost-dims", nargs="+", default=list(DEFAULT_BOOST_DIMS))
    parser.add_argument(
        "--signal-margin",
        type=float,
        default=0.0004,
        help="Minimum oracle-minus-baseline delta to call the ceiling real.",
    )
    return parser.parse_args()


def predict_action(runtime: PolicyRuntime, obs: np.ndarray) -> np.ndarray:
    obs_norm = runtime.vec_norm.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])
    action, _ = runtime.model.predict(obs_norm, deterministic=True)
    return np.asarray(action[0], dtype=np.float32)


def run_episode(
    *,
    runtime: PolicyRuntime,
    args: argparse.Namespace,
    eval_seed: int,
    boost_steps: set[int] | None = None,
    boost_vector: np.ndarray | None = None,
) -> tuple[float, float, float, int, list[RiskEvent]]:
    env = make_track_b_env(**env_kwargs(args))
    obs, _info = env.reset(seed=eval_seed)
    terminated = False
    truncated = False
    step = 0
    shifts: list[int] = []
    while not (terminated or truncated):
        obs_before = np.asarray(obs, dtype=np.float32).copy()
        action = predict_action(runtime, obs_before)
        if boost_steps is not None and boost_vector is not None and step in boost_steps:
            action = np.clip(action + boost_vector, -1.0, 1.0).astype(np.float32)
        obs, _reward, terminated, truncated, info = env.step(action)
        shifts.append(int(info.get("shifts_active", 1)))
        step += 1
    ret_excel, fill_rate, _service_loss_auc, cost_index = finalize_episode(env, shifts)
    risk_events = [
        RiskEvent(
            risk_id=str(ev.risk_id),
            start_step=int(float(ev.start_time) // float(args.step_size_hours)),
        )
        for ev in env.unwrapped.sim.risk_events
    ]
    env.close()
    return ret_excel, fill_rate, cost_index, step, risk_events


def prep_window_steps(
    risk_events: list[RiskEvent],
    *,
    target_risk_ids: set[str],
    lead_weeks: int,
    max_steps: int,
) -> set[int]:
    steps: set[int] = set()
    for ev in risk_events:
        if ev.risk_id not in target_risk_ids:
            continue
        for step in range(ev.start_step - lead_weeks, ev.start_step):
            if 0 <= step < max_steps:
                steps.add(step)
    return steps


def build_boost_vector(boost_dims: list[str], boost: float) -> np.ndarray:
    vec = np.zeros(len(ACTION_DIMS), dtype=np.float32)
    for dim in boost_dims:
        vec[ACTION_DIMS.index(dim)] = boost
    return vec


def parse_target_set(spec: str) -> set[str]:
    return {token.strip() for token in spec.split("+") if token.strip()}


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

    baselines: list[BaselineEpisode] = []
    baseline_rows: list[dict[str, Any]] = []
    runtimes: dict[int, PolicyRuntime] = {}
    for seed in args.seeds:
        runtime = load_runtime("ppo_mlp", seed, args)
        runtimes[seed] = runtime
        for episode in range(1, int(args.eval_episodes) + 1):
            eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + (episode - 1)
            ret_excel, fill_rate, cost_index, steps, risk_events = run_episode(
                runtime=runtime, args=args, eval_seed=eval_seed
            )
            baselines.append(
                BaselineEpisode(
                    seed=seed,
                    episode=episode,
                    eval_seed=eval_seed,
                    ret_excel=ret_excel,
                    fill_rate=fill_rate,
                    cost_index=cost_index,
                    steps=steps,
                    risk_events=risk_events,
                )
            )
            baseline_rows.append(
                {
                    "seed": seed,
                    "episode": episode,
                    "eval_seed": eval_seed,
                    "ret_excel": ret_excel,
                    "fill_rate": fill_rate,
                    "cost_index": cost_index,
                    "n_risk_events": len(risk_events),
                }
            )
            print(
                f"[baseline] seed={seed} ep={episode} ret_excel={ret_excel:.6f} "
                f"n_events={len(risk_events)}"
            )

    baseline_mean = mean([b.ret_excel for b in baselines])
    save_csv(out / "baseline_episodes.csv", baseline_rows)

    grid_rows: list[dict[str, Any]] = []
    per_episode_rows: list[dict[str, Any]] = []
    for target_spec in args.target_sets:
        target_ids = parse_target_set(target_spec)
        for lead in args.lead_weeks:
            for boost in args.boosts:
                boost_vector = build_boost_vector(args.boost_dims, boost)
                ret_values: list[float] = []
                fill_values: list[float] = []
                cost_values: list[float] = []
                for b in baselines:
                    runtime = runtimes[b.seed]
                    boost_steps = prep_window_steps(
                        b.risk_events,
                        target_risk_ids=target_ids,
                        lead_weeks=lead,
                        max_steps=b.steps,
                    )
                    ret_excel, fill_rate, cost_index, _steps, _events = run_episode(
                        runtime=runtime,
                        args=args,
                        eval_seed=b.eval_seed,
                        boost_steps=boost_steps,
                        boost_vector=boost_vector,
                    )
                    ret_values.append(ret_excel)
                    fill_values.append(fill_rate)
                    cost_values.append(cost_index)
                    per_episode_rows.append(
                        {
                            "target_set": target_spec,
                            "lead_weeks": lead,
                            "boost": boost,
                            "seed": b.seed,
                            "episode": b.episode,
                            "eval_seed": b.eval_seed,
                            "ret_excel": ret_excel,
                            "baseline_ret_excel": b.ret_excel,
                            "delta": ret_excel - b.ret_excel,
                            "n_boost_steps": len(boost_steps),
                        }
                    )
                grid_ret_mean = mean(ret_values)
                grid_rows.append(
                    {
                        "target_set": target_spec,
                        "lead_weeks": lead,
                        "boost": boost,
                        "ret_excel_mean": grid_ret_mean,
                        "fill_rate_mean": mean(fill_values),
                        "cost_index_mean": mean(cost_values),
                        "delta_vs_baseline": grid_ret_mean - baseline_mean,
                        "n_episodes": len(ret_values),
                    }
                )
                print(
                    f"[oracle] target={target_spec} L={lead} B={boost:.2f} "
                    f"ret_excel_mean={grid_ret_mean:.6f} "
                    f"delta={grid_ret_mean - baseline_mean:+.6f}"
                )

    save_csv(out / "oracle_grid.csv", grid_rows)
    save_csv(out / "oracle_per_episode.csv", per_episode_rows)

    best_per_target: dict[str, dict[str, Any]] = {}
    for row in grid_rows:
        key = str(row["target_set"])
        if key not in best_per_target or row["ret_excel_mean"] > best_per_target[key]["ret_excel_mean"]:
            best_per_target[key] = row

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_ret_excel_mean": baseline_mean,
        "n_baseline_episodes": len(baselines),
        "signal_margin": args.signal_margin,
        "best_per_target_set": best_per_target,
        "config": {
            "seeds": args.seeds,
            "eval_episodes": args.eval_episodes,
            "max_steps": args.max_steps,
            "observation_version": args.observation_version,
            "risk_level": args.risk_level,
            "ppo_bundles": [str(p) for p in args.ppo_bundles],
            "target_sets": args.target_sets,
            "lead_weeks": args.lead_weeks,
            "boosts": args.boosts,
            "boost_dims": args.boost_dims,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nBaseline (B=0, canonical v7 fixed-RNG): ret_excel_mean={baseline_mean:.6f}")
    for key, row in best_per_target.items():
        verdict = "SIGNAL" if row["delta_vs_baseline"] >= args.signal_margin else "no signal"
        print(
            f"Best oracle for {key}: L={row['lead_weeks']} B={row['boost']:.2f} "
            f"ret_excel_mean={row['ret_excel_mean']:.6f} "
            f"delta={row['delta_vs_baseline']:+.6f} -> {verdict}"
        )
    print(f"\nWrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
