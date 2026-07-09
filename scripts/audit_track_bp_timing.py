#!/usr/bin/env python3
"""Track B-P timing audit: is the learned policy preventive-static or anticipatory?

Decisive hybrid test: replay the trained 8D (adaptive-only) PPO checkpoints on
the 11D track_bp env, grafting a CONSTANT buffer fraction onto dims 8-10. If
some constant fraction matches PPO_11D's episode ReT, the learned policy is
"optimal static buffering + adaptation" (preventive, not anticipatory). If
PPO_11D beats every constant graft, the buffer usage is state/time-dependent
(anticipatory timing has value).

Also logs PPO_11D's per-step buffer fractions against the R21 event calendar
for a lead-lag read (does holding rise with time-since-last-event hazard?).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from scripts.run_track_b_smoke import build_env_kwargs, build_parser as smoke_build_parser  # noqa: E402
from scripts.run_track_b_observation_ablation import OBS_ABLATION_CONFIGS  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.track_bp_env import make_track_bp_env  # noqa: E402

EVAL_EPISODE_SEED_OFFSET = 50_000


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--run-8d", type=Path, required=True,
                   help="track_bp confirm 8D run dir (models/seedN/...)")
    p.add_argument("--run-11d", type=Path, required=True,
                   help="track_bp confirm 11D run dir (models/seedN/...)")
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--eval-episodes", type=int, default=24)
    p.add_argument("--constant-fracs", nargs="+", type=float,
                   default=[0.1, 0.15, 0.2, 0.3, 0.5, 0.75])
    p.add_argument("--max-steps", type=int, default=104)
    p.add_argument("--obs-config", default="v10_no_regime_forecast")
    p.add_argument("--enabled-risks", default="R21")
    p.add_argument("--risk-frequency-by-id", default="R21=8")
    p.add_argument("--risk-impact-by-id", default="R21=4")
    p.add_argument("--replenishment-lead-time", type=float, default=168.0)
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


def run_episode(model, vec_norm, args, cli, eval_seed: int, *,
                graft_frac: float | None = None,
                log_steps: bool = False) -> dict[str, Any]:
    env = make_env(args, cli)
    obs, _ = env.reset(seed=eval_seed)
    terminated = truncated = False
    fracs: list[float] = []
    step_log: list[dict[str, float]] = []
    step = 0
    while not (terminated or truncated):
        obs_n = vec_norm.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])
        action, _ = model.predict(obs_n, deterministic=True)
        action = np.asarray(action[0], dtype=np.float32)
        if graft_frac is not None:
            # 8D policy on the 11D env: append the constant buffer fraction.
            action = np.concatenate([action[:8], np.full(3, graft_frac, np.float32)])
        frac = float(np.mean(np.clip(action[8:], 0.0, 1.0)))
        fracs.append(frac)
        if log_steps:
            step_log.append({"step": step, "buffer_frac": frac})
        obs, _r, terminated, truncated, _i = env.step(action)
        step += 1
    m = compute_episode_metrics(env.unwrapped.sim)
    out: dict[str, Any] = {
        "ret_excel": float(m["ret_excel"]),
        "mean_buffer_frac": float(np.mean(fracs)),
    }
    if log_steps:
        events = [
            {"start_step": float(e.start_time) / 168.0,
             "end_step": float(e.end_time) / 168.0}
            for e in env.unwrapped.sim.risk_events if str(e.risk_id) == "R21"
        ]
        out["step_log"] = step_log
        out["r21_events"] = events
    env.close()
    return out


def main() -> None:
    cli = build_parser().parse_args()
    out = cli.output_dir
    out.mkdir(parents=True, exist_ok=True)
    args = build_args(cli)
    eval_seeds = [1 + EVAL_EPISODE_SEED_OFFSET + i for i in range(cli.eval_episodes)]

    sample = make_env(args, cli)
    results: dict[str, Any] = {"config": {k: str(v) for k, v in vars(cli).items()}}

    # Arm 1: hybrid grafts — 8D policy + constant buffer fraction.
    grafts: dict[str, list[float]] = {}
    for frac in cli.constant_fracs:
        rets = []
        for seed in cli.seeds:
            model, vec_norm = load(cli.run_8d, seed, sample)
            rets.extend(
                run_episode(model, vec_norm, args, cli, es, graft_frac=frac)["ret_excel"]
                for es in eval_seeds
            )
        grafts[f"{frac:.2f}"] = rets
        print(f"graft {frac:.2f}: mean {np.mean(rets):.4f}", flush=True)
    results["graft_means"] = {k: float(np.mean(v)) for k, v in grafts.items()}

    # Arm 2: the 11D policy itself (re-evaluated here for exact pairing).
    ppo11: list[float] = []
    for seed in cli.seeds:
        model, vec_norm = load(cli.run_11d, seed, sample)
        ppo11.extend(
            run_episode(model, vec_norm, args, cli, es)["ret_excel"]
            for es in eval_seeds
        )
    results["ppo11_mean"] = float(np.mean(ppo11))
    print(f"ppo11: mean {np.mean(ppo11):.4f}", flush=True)

    # Timing delta: PPO_11D vs the BEST constant graft, paired per (seed, episode).
    best_key = max(grafts, key=lambda k: np.mean(grafts[k]))
    best = np.array(grafts[best_key])
    d = np.array(ppo11) - best
    rng = np.random.default_rng(0)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    results["best_graft"] = best_key
    results["timing_delta_vs_best_graft"] = {
        "mean": float(d.mean()), "ci95": [float(lo), float(hi)],
        "n": int(len(d)), "positive": int((d > 0).sum()),
    }

    # Arm 3: behavioral lead-lag log for seed 1, first 4 episodes.
    model, vec_norm = load(cli.run_11d, cli.seeds[0], sample)
    logs = [run_episode(model, vec_norm, args, cli, es, log_steps=True)
            for es in eval_seeds[:4]]
    results["behavior_logs"] = [
        {"step_log": l["step_log"], "r21_events": l["r21_events"]} for l in logs
    ]

    (out / "timing_audit.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items()
                      if k not in ("config", "behavior_logs")}, indent=2))


if __name__ == "__main__":
    main()
