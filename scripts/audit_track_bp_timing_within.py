#!/usr/bin/env python3
"""Track B-P within-checkpoint timing controls (supersedes the graft audit).

The graft audit (audit_track_bp_timing.py) compared two DIFFERENT trained
policies and could not attribute the residual to timing (confounds: per-op
levels, dispatch-buffer coordination, training differences). These controls
operate on the SAME 11D checkpoint, manipulating ONLY its three buffer
outputs under the same exogenous event tape (strict CRN):

  self        - checkpoint as-is (closed loop).
  clamp_perop - buffer dims replaced by the checkpoint's OWN per-op mean
                fracs from a reference pass of the same episode (matched
                levels, all timing destroyed).
  replay      - buffer dims replayed open-loop from the reference pass in
                original step order (control for open-loop application).
  permuted    - reference schedule cyclically shifted by half the horizon
                (same marginals and smoothness, alignment with the event
                tape broken). Timing value = replay - permuted.
  block_pre   - buffers clamped to per-op means ONLY in the 3 weeks before
                each R21 event start (kills ex-ante adjustment; policy free
                elsewhere).
  block_post  - buffers clamped to per-op means ONLY from event start to
                one week after recovery (kills lagged reactive top-ups).

Estimands (per training seed, CRN-paired per episode; inference = t-CI over
the per-seed mean deltas, NOT pooled episodes):
  schedule_value   = self - clamp_perop
  alignment_value  = replay - permuted
  exante_component = self - block_pre
  reactive_component = self - block_post
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402
from scipy import stats as scipy_stats  # noqa: E402

from scripts.run_track_b_smoke import build_env_kwargs, build_parser as smoke_build_parser  # noqa: E402
from scripts.run_track_b_observation_ablation import OBS_ABLATION_CONFIGS  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.track_bp_env import make_track_bp_env  # noqa: E402

EVAL_EPISODE_SEED_OFFSET = 50_000
ARMS = ("self", "clamp_perop", "replay", "permuted", "block_pre", "block_post")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--run-11d", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--eval-episodes", type=int, default=24)
    p.add_argument("--pre-window-weeks", type=int, default=3)
    p.add_argument("--post-window-weeks", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=104)
    p.add_argument("--obs-config", default="v10_no_regime_forecast")
    p.add_argument("--enabled-risks", default="R21")
    p.add_argument("--risk-frequency-by-id", default="R21=8")
    p.add_argument("--risk-impact-by-id", default="R21=4")
    p.add_argument("--replenishment-lead-time", type=float, default=168.0)
    p.add_argument("--target-risk", default="R21")
    return p


def build_args(cli):
    args = smoke_build_parser().parse_args([])
    args.risk_level = "current"
    args.faithful = True
    args.reward_mode = "control_v1"
    args.max_steps = cli.max_steps
    args.enabled_risks = cli.enabled_risks
    args.risk_frequency_by_id = cli.risk_frequency_by_id or None
    args.risk_impact_by_id = cli.risk_impact_by_id or None
    args.observation_version = OBS_ABLATION_CONFIGS[str(cli.obs_config)].observation_version
    args.inventory_replenishment_lead_time = float(cli.replenishment_lead_time)
    return args


def make_env(args, cli):
    kwargs = build_env_kwargs(args)
    lead = kwargs.pop("inventory_replenishment_lead_time", 168.0)
    env = make_track_bp_env(inventory_replenishment_lead_time=lead, **kwargs)
    wrapper = OBS_ABLATION_CONFIGS[str(cli.obs_config)].wrapper
    if wrapper is not None:
        env = wrapper(env)
    return env


def load(run_dir: Path, seed: int, sample_env):
    seed_dir = run_dir / "models" / f"seed{seed}"
    model = PPO.load(str(seed_dir / "ppo_model.zip"), device="cpu")
    vec_norm = VecNormalize.load(str(seed_dir / "vec_normalize.pkl"),
                                 DummyVecEnv([lambda: sample_env]))
    vec_norm.training = False
    return model, vec_norm


def rollout(model, vec_norm, args, cli, eval_seed: int, *,
            buffer_override=None) -> dict[str, Any]:
    """Roll the checkpoint; buffer_override(step, policy_frac_vec) -> frac vec."""
    env = make_env(args, cli)
    obs, _ = env.reset(seed=eval_seed)
    terminated = truncated = False
    step = 0
    frac_trace: list[list[float]] = []
    while not (terminated or truncated):
        obs_n = vec_norm.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])
        action, _ = model.predict(obs_n, deterministic=True)
        action = np.asarray(action[0], dtype=np.float32)
        fracs = np.clip(action[8:], 0.0, 1.0)
        if buffer_override is not None:
            fracs = np.asarray(buffer_override(step, fracs), dtype=np.float32)
            action = np.concatenate([action[:8], fracs])
        frac_trace.append([float(x) for x in fracs])
        obs, _r, terminated, truncated, _i = env.step(action)
        step += 1
    m = compute_episode_metrics(env.unwrapped.sim)
    events = [
        (float(e.start_time) / 168.0, float(e.end_time) / 168.0)
        for e in env.unwrapped.sim.risk_events
        if str(e.risk_id) == str(cli.target_risk)
    ]
    env.close()
    return {
        "ret_excel": float(m["ret_excel"]),
        "frac_trace": frac_trace,
        "events": events,
    }


def main() -> None:
    cli = build_parser().parse_args()
    out = cli.output_dir
    out.mkdir(parents=True, exist_ok=True)
    args = build_args(cli)
    eval_seeds = [1 + EVAL_EPISODE_SEED_OFFSET + i for i in range(cli.eval_episodes)]
    sample = make_env(args, cli)

    rows: list[dict[str, Any]] = []
    for seed in cli.seeds:
        model, vec_norm = load(cli.run_11d, seed, sample)
        for es in eval_seeds:
            ref = rollout(model, vec_norm, args, cli, es)
            trace = np.array(ref["frac_trace"], dtype=np.float32)
            n_steps = trace.shape[0]
            perop_mean = trace.mean(axis=0)
            shift = n_steps // 2
            permuted_trace = np.roll(trace, shift, axis=0)
            pre_steps: set[int] = set()
            post_steps: set[int] = set()
            for start, end in ref["events"]:
                s0 = int(np.floor(start))
                e0 = int(np.ceil(end))
                pre_steps.update(range(max(0, s0 - cli.pre_window_weeks), max(0, s0)))
                post_steps.update(range(max(0, s0), min(n_steps, e0 + cli.post_window_weeks + 1)))

            overrides = {
                "self": None,
                "clamp_perop": lambda t, f: perop_mean,
                "replay": lambda t, f: trace[min(t, n_steps - 1)],
                "permuted": lambda t, f: permuted_trace[min(t, n_steps - 1)],
                "block_pre": lambda t, f: perop_mean if t in pre_steps else f,
                "block_post": lambda t, f: perop_mean if t in post_steps else f,
            }
            arm_rets = {"self": ref["ret_excel"]}
            for arm in ARMS[1:]:
                arm_rets[arm] = rollout(model, vec_norm, args, cli, es,
                                        buffer_override=overrides[arm])["ret_excel"]
            for arm in ARMS:
                rows.append({
                    "train_seed": seed, "eval_seed": es, "arm": arm,
                    "ret_excel": arm_rets[arm],
                    "n_events": len(ref["events"]),
                })
        print(f"seed {seed} done", flush=True)

    with (out / "timing_within_rows.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by = {}
    for r in rows:
        by[(r["train_seed"], r["eval_seed"], r["arm"])] = r["ret_excel"]

    def estimand(a: str, b: str) -> dict[str, Any]:
        per_seed = []
        for seed in cli.seeds:
            ds = [by[(seed, es, a)] - by[(seed, es, b)] for es in eval_seeds]
            per_seed.append(float(np.mean(ds)))
        ps = np.array(per_seed)
        if len(ps) > 1:
            ci = scipy_stats.t.interval(0.95, len(ps) - 1, loc=ps.mean(),
                                        scale=scipy_stats.sem(ps))
        else:
            ci = (float("nan"), float("nan"))
        return {"per_seed": [round(x, 6) for x in per_seed],
                "mean": float(ps.mean()),
                "seed_clustered_ci95": [float(ci[0]), float(ci[1])],
                "seeds_positive": int((ps > 0).sum())}

    summary = {
        "config": {k: str(v) for k, v in vars(cli).items()},
        "resolved_env_kwargs": {k: repr(v) for k, v in sorted(build_env_kwargs(args).items())},
        "arm_means": {
            arm: float(np.mean([by[(s, es, arm)] for s in cli.seeds for es in eval_seeds]))
            for arm in ARMS
        },
        "schedule_value__self_minus_clamp": estimand("self", "clamp_perop"),
        "openloop_control__replay_minus_self": estimand("replay", "self"),
        "alignment_value__replay_minus_permuted": estimand("replay", "permuted"),
        "exante_component__self_minus_blockpre": estimand("self", "block_pre"),
        "reactive_component__self_minus_blockpost": estimand("self", "block_post"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
