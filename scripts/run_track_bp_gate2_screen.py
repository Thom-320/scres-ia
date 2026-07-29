#!/usr/bin/env python3
"""Track B-P Gate 2: PPO screen on the preventive contract, oracle-anchored.

Trains PPO (canonical hyperparameters) on `track_bp_v1` in a gate-positive
cell, then evaluates CRN-paired against the three Gate-1 clock policies on the
SAME eval seeds. Reports the conversion fraction of the static preventive
oracle headroom:

    conversion = (PPO - never_prepared) / (always_prepared - never_prepared)

and the anticipation check: PPO vs always_prepared at what mean buffer
holding (matching `always` at materially lower holding = timed buffering;
matching it at ~1.0 holding = static buffering learned, no anticipation).

Pre-registered in docs/TRACK_BP_PREREGISTRATION_2026-07-08.md (Gate 2 runs
only after Gate 0/1 positive — see docs/TRACK_BP_GATES_0_1_VERDICT_2026-07-09.md).
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
from scripts.run_track_bp_gate1_oracle import buffer_schedule  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.track_bp_env import (  # noqa: E402
    TRACK_BP_ACTION_DIM,
    make_track_b_fixed_buffer_env,
    make_track_bp_env,
)

EVAL_EPISODE_SEED_OFFSET = 50_000
CLOCK_POLICIES = ("never_prepared", "always_prepared", "calendar_prepared")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--train-timesteps", type=int, default=30_000)
    p.add_argument("--eval-episodes", type=int, default=24)
    p.add_argument("--max-steps", type=int, default=104)
    p.add_argument("--obs-config", default="v10_no_regime_forecast",
                   choices=list(OBS_ABLATION_CONFIGS.keys()))
    p.add_argument("--enabled-risks", default="R21")
    p.add_argument("--risk-frequency-by-id", default="R21=8")
    p.add_argument("--risk-impact-by-id", default="R21=4")
    p.add_argument("--risk-level", default="current")
    p.add_argument("--replenishment-lead-time", type=float, default=168.0)
    p.add_argument("--prep-window-weeks", type=int, default=4)
    p.add_argument("--calendar-cycle-weeks", type=int, default=24)
    p.add_argument("--step-size-hours", type=float, default=168.0)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument(
        "--contract",
        choices=["track_bp", "track_b", "track_b_fixed_buffers"],
        default="track_bp",
                   help="track_b (8D, no buffer dims) is the contract-ablation arm that "
                        "isolates the preventive increment: PPO_11D - PPO_8D on the same "
                        "cell/seeds. Clock-policy arms are skipped for track_b (buffers "
                        "do not exist on that contract; reuse the track_bp run's rows). "
                        "track_b_fixed_buffers keeps the 8D learner but re-emits a frozen "
                        "three-buffer posture through the Track B-P lead-time physics.")
    p.add_argument(
        "--fixed-buffer-fracs",
        nargs=3,
        type=float,
        metavar=("OP3", "OP5", "OP9"),
        default=[0.0, 0.0, 0.0],
        help="Frozen Op3/Op5/Op9 fractions used by track_b_fixed_buffers.",
    )
    p.add_argument("--features-extractor", choices=["mlp", "real_kan"], default="mlp",
                   help="real_kan plugs the official pykan extractor (Garrido's literal "
                        "KAN suggestion) as the efficiency/interpretability sidecar; "
                        "same protocol otherwise.")
    p.add_argument("--kan-features-dim", type=int, default=32)
    p.add_argument("--kan-hidden-width", type=int, default=32)
    return p


def build_args(cli: argparse.Namespace) -> argparse.Namespace:
    args = smoke_build_parser().parse_args([])
    args.risk_level = cli.risk_level
    args.faithful = True
    args.reward_mode = "control_v1"
    args.max_steps = cli.max_steps
    args.enabled_risks = cli.enabled_risks
    args.risk_frequency_by_id = cli.risk_frequency_by_id or None
    args.risk_impact_by_id = cli.risk_impact_by_id or None
    args.observation_version = OBS_ABLATION_CONFIGS[str(cli.obs_config)].observation_version
    args.inventory_replenishment_lead_time = float(cli.replenishment_lead_time)
    return args


def make_env(args: argparse.Namespace, cli: argparse.Namespace):
    kwargs = build_env_kwargs(args)
    lead = kwargs.pop("inventory_replenishment_lead_time", 168.0)
    contract = getattr(cli, "contract", "track_bp")
    if contract == "track_b":
        from supply_chain.external_env_interface import make_track_b_env
        env = make_track_b_env(**kwargs)
    elif contract == "track_b_fixed_buffers":
        env = make_track_b_fixed_buffer_env(
            fixed_fracs=tuple(float(x) for x in cli.fixed_buffer_fracs),
            inventory_replenishment_lead_time=lead,
            **kwargs,
        )
    else:
        env = make_track_bp_env(inventory_replenishment_lead_time=lead, **kwargs)
    wrapper = OBS_ABLATION_CONFIGS[str(cli.obs_config)].wrapper
    if wrapper is not None:
        env = wrapper(env)
    return env


def run_clock_episode(policy: str, args, cli, eval_seed: int) -> dict[str, Any]:
    env = make_env(args, cli)
    env.reset(seed=eval_seed)
    terminated = truncated = False
    step = 0
    fracs: list[float] = []
    while not (terminated or truncated):
        action = np.zeros(TRACK_BP_ACTION_DIM, dtype=np.float32)
        frac = buffer_schedule(policy, step, cli)
        action[8:] = frac
        fracs.append(frac)
        _o, _r, terminated, truncated, _i = env.step(action)
        step += 1
    m = compute_episode_metrics(env.unwrapped.sim)
    env.close()
    return {"ret_excel": float(m["ret_excel"]),
            "mean_buffer_frac": float(np.mean(fracs))}


def run_ppo_episode(model, vec_norm, args, cli, eval_seed: int) -> dict[str, Any]:
    env = make_env(args, cli)
    obs, _ = env.reset(seed=eval_seed)
    terminated = truncated = False
    fracs: list[float] = []
    while not (terminated or truncated):
        obs_n = vec_norm.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])
        action, _ = model.predict(obs_n, deterministic=True)
        action = np.asarray(action[0], dtype=np.float32)
        if action.shape[0] > 8:
            fracs.append(float(np.mean(np.clip(action[8:], 0.0, 1.0))))
        elif cli.contract == "track_b_fixed_buffers":
            fracs.append(float(np.mean(np.clip(cli.fixed_buffer_fracs, 0.0, 1.0))))
        else:
            fracs.append(0.0)
        obs, _r, terminated, truncated, _i = env.step(action)
    m = compute_episode_metrics(env.unwrapped.sim)
    env.close()
    return {"ret_excel": float(m["ret_excel"]),
            "mean_buffer_frac": float(np.mean(fracs))}


def main() -> None:
    cli = build_parser().parse_args()
    out = cli.output_dir
    out.mkdir(parents=True, exist_ok=True)
    args = build_args(cli)
    eval_seeds = [1 + EVAL_EPISODE_SEED_OFFSET + i for i in range(cli.eval_episodes)]

    # Clock-policy oracle on the exact eval seeds (self-contained comparators).
    # Skipped for the track_b contract ablation (no buffer dims to schedule);
    # compare its PPO rows against the track_bp run's clock/PPO rows offline.
    clock: dict[str, list[dict[str, Any]]] = {p: [] for p in CLOCK_POLICIES}
    if cli.contract == "track_bp":
        for policy in CLOCK_POLICIES:
            for es in eval_seeds:
                clock[policy].append(run_clock_episode(policy, args, cli, es))
        print("clock oracle done", flush=True)

    ppo_rows: dict[int, list[dict[str, Any]]] = {}
    for seed in cli.seeds:
        def _make():
            return make_env(args, cli)
        venv = DummyVecEnv([_make])
        vec_norm = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
        policy_kwargs: dict[str, Any] = {"net_arch": [64, 64]}
        if cli.features_extractor == "real_kan":
            from scripts.real_kan_extractor import RealKANFeaturesExtractor
            policy_kwargs = {
                "features_extractor_class": RealKANFeaturesExtractor,
                "features_extractor_kwargs": {
                    "features_dim": int(cli.kan_features_dim),
                    "hidden_width": int(cli.kan_hidden_width),
                    "seed": int(seed),
                },
                "net_arch": [64, 64],
            }
        model = PPO(
            "MlpPolicy", vec_norm,
            learning_rate=cli.learning_rate, n_steps=1024, batch_size=256,
            n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            policy_kwargs=policy_kwargs, seed=seed, verbose=0, device="cpu",
        )
        model.learn(total_timesteps=cli.train_timesteps, progress_bar=False)
        vec_norm.training = False
        seed_dir = out / "models" / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(seed_dir / "ppo_model.zip"))
        vec_norm.save(str(seed_dir / "vec_normalize.pkl"))
        ppo_rows[seed] = [run_ppo_episode(model, vec_norm, args, cli, es) for es in eval_seeds]
        print(f"seed {seed} trained + evaluated", flush=True)

    def mean_ret(rows: list[dict[str, Any]]) -> float:
        return float(np.mean([r["ret_excel"] for r in rows]))

    have_clock = bool(clock["never_prepared"])
    never = np.array([r["ret_excel"] for r in clock["never_prepared"]]) if have_clock else None
    always = np.array([r["ret_excel"] for r in clock["always_prepared"]]) if have_clock else None
    oracle = (always - never) if have_clock else None

    per_seed: dict[str, Any] = {}
    conversions: list[float] = []
    for seed in cli.seeds:
        ppo = np.array([r["ret_excel"] for r in ppo_rows[seed]])
        entry: dict[str, Any] = {
            "ppo_ret_mean": float(np.mean(ppo)),
            "mean_buffer_frac": float(np.mean([r["mean_buffer_frac"] for r in ppo_rows[seed]])),
        }
        if have_clock:
            d_never = ppo - never
            d_always = ppo - always
            conv = float(np.mean(d_never) / np.mean(oracle)) if np.mean(oracle) > 0 else float("nan")
            conversions.append(conv)
            entry.update({
                "delta_vs_never_mean": float(np.mean(d_never)),
                "delta_vs_always_mean": float(np.mean(d_always)),
                "conversion_of_oracle": conv,
            })
        per_seed[str(seed)] = entry

    summary = {
        "config": {k: str(v) for k, v in vars(cli).items()},
        "resolved_env_kwargs": {k: repr(v) for k, v in sorted(build_env_kwargs(args).items())},
        "per_seed": per_seed,
    }
    if have_clock:
        summary["clock_policy_means"] = {
            p: {"ret_excel": mean_ret(clock[p]),
                "mean_buffer_frac": float(np.mean([r["mean_buffer_frac"] for r in clock[p]]))}
            for p in CLOCK_POLICIES
        }
        summary["oracle_always_minus_never_mean"] = float(np.mean(oracle))
        summary["conversion_mean"] = float(np.mean(conversions)) if conversions else None
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows_flat = []
    for policy in CLOCK_POLICIES:
        for es, r in zip(eval_seeds, clock[policy]):
            rows_flat.append({"arm": policy, "eval_seed": es, **r})
    for seed in cli.seeds:
        for es, r in zip(eval_seeds, ppo_rows[seed]):
            rows_flat.append({"arm": f"ppo_seed{seed}", "eval_seed": es, **r})
    import csv
    with (out / "gate2_rows.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_flat[0].keys()))
        w.writeheader()
        w.writerows(rows_flat)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
