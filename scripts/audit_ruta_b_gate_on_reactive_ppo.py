#!/usr/bin/env python3
"""Apples-to-apples causal gate: run Ruta B's EXACT inlined counterfactual on
the frozen reactive Case C PPO checkpoints.

Motivation: the Ruta B confirmatory result (74.1% positive pairs) used the
counterfactual loop inlined in scripts/run_track_b_ruta_b_sidecar.py, while
the reactive baseline's null result came from the separate
scripts/audit_track_b_risk_event_counterfactual.py. Same conceptual design
(reset pre-risk actions to the policy's own median calm action, window
[-4,-1] weeks), but not literally the same code. This script removes that
asymmetry: it imports the sidecar's own logic and runs it, unchanged except
for the policy-loading path, against the reactive PPO checkpoints
(v7_no_forecast, plain PPO, no label channel). If the reactive baseline
suddenly also shows a high positive-pair rate under THIS code, the Ruta B
signal is a gate artifact; if it stays null, the comparison is clean.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_b_smoke import build_env_kwargs, save_csv  # noqa: E402
from scripts.run_track_b_observation_ablation import OBS_ABLATION_CONFIGS  # noqa: E402
from scripts.run_track_b_ruta_b_sidecar import (  # noqa: E402
    EVAL_EPISODE_SEED_OFFSET,
    build_args,
)
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.external_env_interface import make_track_b_env  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help="Dir containing models/seed{N}/{ppo_model.zip,vec_normalize.pkl}",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--eval-episodes", type=int, default=12)
    p.add_argument("--max-steps", type=int, default=104)
    p.add_argument("--obs-config", default="v7_no_forecast", choices=list(OBS_ABLATION_CONFIGS.keys()))
    p.add_argument("--target-risks", nargs="+", default=["R22", "R24"])
    p.add_argument("--max-events-per-risk-episode", type=int, default=8)
    return p


def make_env(args: argparse.Namespace, obs_config: str):
    env = make_track_b_env(**build_env_kwargs(args))
    wrapper = OBS_ABLATION_CONFIGS[str(obs_config)].wrapper
    if wrapper is not None:
        env = wrapper(env)
    return env


def load_policy(checkpoint_root: Path, seed: int, sample_env) -> tuple[PPO, VecNormalize]:
    seed_dir = checkpoint_root / "models" / f"seed{seed}"
    model = PPO.load(str(seed_dir / "ppo_model.zip"), device="cpu")
    vec_norm = VecNormalize.load(
        str(seed_dir / "vec_normalize.pkl"), DummyVecEnv([lambda: sample_env])
    )
    vec_norm.training = False
    return model, vec_norm


def predict_action(model: PPO, vec_norm: VecNormalize, obs: np.ndarray) -> np.ndarray:
    obs_norm = vec_norm.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])
    action, _ = model.predict(obs_norm, deterministic=True)
    return np.asarray(action[0], dtype=np.float32)


def run_counterfactual_for_seed(
    model: PPO,
    vec_norm: VecNormalize,
    args: argparse.Namespace,
    cli: argparse.Namespace,
    seed: int,
) -> list[dict[str, Any]]:
    # Identical structure to run_track_b_ruta_b_sidecar.run_counterfactual:
    # full pass -> median calm action -> per-anchor reset of weeks [-4,-1].
    rows: list[dict[str, Any]] = []
    target_risks = set(cli.target_risks)
    for episode_idx in range(cli.eval_episodes):
        eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + episode_idx

        env = make_env(args, cli.obs_config)
        obs, _info = env.reset(seed=eval_seed)
        terminated = truncated = False
        steps_full: list[np.ndarray] = []
        while not (terminated or truncated):
            action = predict_action(model, vec_norm, obs)
            steps_full.append(action)
            obs, _reward, terminated, truncated, _info = env.step(action)
        metrics_full = compute_episode_metrics(env.unwrapped.sim)
        risk_events = list(env.unwrapped.sim.risk_events)
        env.close()
        ret_full = float(metrics_full["ret_excel"])

        calm_action = np.median(np.stack(steps_full), axis=0) if steps_full else None

        anchors = [
            int(float(ev.start_time) // float(args.step_size_hours))
            for ev in risk_events
            if str(ev.risk_id) in target_risks
        ]
        anchors = anchors[: cli.max_events_per_risk_episode]

        for anchor_step in anchors:
            reset_steps = {s for s in range(anchor_step - 4, anchor_step) if 0 <= s < args.max_steps}
            if not reset_steps or calm_action is None:
                continue
            env = make_env(args, cli.obs_config)
            obs, _info = env.reset(seed=eval_seed)
            terminated = truncated = False
            step = 0
            while not (terminated or truncated):
                if step in reset_steps:
                    action = calm_action
                else:
                    action = predict_action(model, vec_norm, obs)
                obs, _reward, terminated, truncated, _info = env.step(action)
                step += 1
            metrics_reset = compute_episode_metrics(env.unwrapped.sim)
            env.close()
            ret_reset = float(metrics_reset["ret_excel"])
            rows.append(
                {
                    "seed": seed,
                    "episode": episode_idx + 1,
                    "eval_seed": eval_seed,
                    "anchor_step": anchor_step,
                    "ret_full": ret_full,
                    "ret_reset": ret_reset,
                    "delta_ret_excel": ret_full - ret_reset,
                }
            )
    return rows


def main() -> None:
    cli = build_parser().parse_args()
    out = cli.output_dir
    out.mkdir(parents=True, exist_ok=True)

    obs_config = OBS_ABLATION_CONFIGS[str(cli.obs_config)]
    args = build_args(
        {
            "max_steps": cli.max_steps,
            "eval_episodes": cli.eval_episodes,
            "observation_version": obs_config.observation_version,
        }
    )

    all_rows: list[dict[str, Any]] = []
    for seed in cli.seeds:
        print(f"=== seed {seed}: counterfactual on reactive PPO checkpoint ===", flush=True)
        sample_env = make_env(args, cli.obs_config)
        model, vec_norm = load_policy(cli.checkpoint_root, seed, sample_env)
        all_rows.extend(run_counterfactual_for_seed(model, vec_norm, args, cli, seed))

    save_csv(out / "counterfactual_rows.csv", all_rows)

    n_pairs = len(all_rows)
    n_positive = sum(1 for r in all_rows if r["delta_ret_excel"] > 0)
    mean_delta = float(np.mean([r["delta_ret_excel"] for r in all_rows])) if n_pairs else 0.0
    by_seed: dict[int, list[float]] = {}
    for r in all_rows:
        by_seed.setdefault(int(r["seed"]), []).append(float(r["delta_ret_excel"]))
    per_seed = {
        str(s): {
            "n_pairs": len(d),
            "n_positive": sum(1 for x in d if x > 0),
            "positive_rate": sum(1 for x in d if x > 0) / len(d),
            "mean_delta": float(np.mean(d)),
        }
        for s, d in sorted(by_seed.items())
    }
    summary = {
        "checkpoint_root": str(cli.checkpoint_root),
        "obs_config": str(cli.obs_config),
        "target_risks": list(cli.target_risks),
        "counterfactual_n_pairs": n_pairs,
        "counterfactual_n_positive": n_positive,
        "counterfactual_positive_rate": (n_positive / n_pairs) if n_pairs else 0.0,
        "counterfactual_mean_delta_ret_excel": mean_delta,
        "per_seed": per_seed,
        "config": vars(cli),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
