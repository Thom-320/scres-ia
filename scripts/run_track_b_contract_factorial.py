#!/usr/bin/env python3
"""Paper 1 Blocker #2: corrected decision-contract factorial.

The published E4 ablation is misdescribed (verified 2026-07-09): its
"shift_only" arm freezes ONLY dims 6-7 (the policy still controls all upstream
dims 0-5 including shift) and its "downstream_only" arm freezes ONLY dim 5
(the policy still controls upstream dims 0-4 plus dispatch), and each arm used
a different comparator. It therefore cannot identify downstream dispatch
access as the mechanism.

This factorial fixes both defects. Arms (all with the SAME observation,
reward, training budget, canonical lr 3e-4, and the SAME common held-out tape
battery and common static comparator):

  joint          - full 8D track_b_v1 (no freeze);
  upstream_shift - dims 6-7 frozen at 0.0 (neutral 1.25x dispatch): upstream
                   inventory + shift authority, NO dispatch access;
  dispatch_only  - dims 0-5 frozen at 0.0 (neutral multipliers, S2): dispatch
                   authority ONLY;
  (baseline)     - the common static comparator evaluated on the same tapes.

Primary pre-registered nested contrast: joint - upstream_shift (the causal
value of ADDING Op10/Op12 dispatch authority to an otherwise identical
contract). Promote the "decision-contract alignment" mechanism iff its
two-way (seed x tape) CI is wholly positive with consistent seed direction;
otherwise the paper downgrades to the fallback title and descriptive claim.
Secondary: dispatch_only - static (is dispatch authority alone sufficient?).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402
from scipy import stats as scipy_stats  # noqa: E402

from scripts.run_track_b_crossed_eval import (  # noqa: E402
    CANONICAL_ENV_KWARGS,
    episode_metrics_row,
    run_static_episode,
    static_action,
    two_way_bootstrap,
)
from supply_chain.external_env_interface import make_track_b_env  # noqa: E402

ARM_FROZEN_VALUES: dict[str, dict[int, float]] = {
    "joint": {},
    "upstream_shift": {6: 0.0, 7: 0.0},
    "dispatch_only": {d: 0.0 for d in range(6)},
    # Strong fixed-dispatch anchor: Op10=2.0x and Op12=1.5x under the
    # track_b_v1 decoder (1.25 + 0.75*x).  This is the reviewer-critical
    # replacement for the neutral 1.25x/1.25x no-dispatch arm.
    "upstream_shift_best_dispatch": {6: 1.0, 7: 1.0 / 3.0},
}


class FreezeDimsWrapper(gym.ActionWrapper):
    """Force selected action dimensions to specified values every step."""

    def __init__(self, env: gym.Env, frozen_values: dict[int, float]):
        super().__init__(env)
        self.frozen_values = {int(d): float(v) for d, v in frozen_values.items()}

    def action(self, action):
        if not self.frozen_values:
            return action
        modified = np.array(action, dtype=np.float32, copy=True)
        for d, value in self.frozen_values.items():
            modified[d] = value
        return modified


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--arms", nargs="+", choices=list(ARM_FROZEN_VALUES.keys()),
                   default=["joint", "upstream_shift", "dispatch_only"])
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--train-timesteps", type=int, default=60_000)
    p.add_argument("--eval-seed-base", type=int, default=200_001)
    p.add_argument("--eval-episodes", type=int, default=24)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--static-shift", type=int, default=2)
    p.add_argument("--static-op10-mult", type=float, default=2.0)
    p.add_argument("--static-op12-mult", type=float, default=1.5)
    p.add_argument("--skip-static", action="store_true",
                   help="Reuse static rows from a crossed-eval run on the same tapes.")
    return p


def make_arm_env(arm: str):
    env = make_track_b_env(**CANONICAL_ENV_KWARGS)
    frozen = ARM_FROZEN_VALUES[arm]
    if frozen:
        env = FreezeDimsWrapper(env, frozen)
    return env


def run_ppo_episode(model, vec_norm, arm: str, eval_seed: int) -> dict[str, float]:
    env = make_arm_env(arm)
    obs, _ = env.reset(seed=eval_seed)
    terminated = truncated = False
    while not (terminated or truncated):
        obs_n = vec_norm.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])
        action, _ = model.predict(obs_n, deterministic=True)
        obs, _r, terminated, truncated, _i = env.step(np.asarray(action[0], dtype=np.float32))
    row = episode_metrics_row(env.unwrapped.sim)
    env.close()
    return row


def main() -> None:
    cli = build_parser().parse_args()
    out = cli.output_dir
    out.mkdir(parents=True, exist_ok=True)
    tapes = [cli.eval_seed_base + i for i in range(cli.eval_episodes)]

    rows: list[dict[str, Any]] = []
    static_by_tape: dict[int, dict[str, float]] = {}
    if not cli.skip_static:
        for t in tapes:
            r = run_static_episode(cli, t)
            static_by_tape[t] = r
            rows.append({"arm": "static", "train_seed": 0, "eval_seed": t, **r})
        print("static baseline done", flush=True)

    results: dict[tuple[str, int, int], dict[str, float]] = {}
    for arm in cli.arms:
        for seed in cli.seeds:
            venv = DummyVecEnv([lambda a=arm: make_arm_env(a)])
            vec_norm = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
            model = PPO(
                "MlpPolicy", vec_norm,
                learning_rate=cli.learning_rate, n_steps=1024, batch_size=256,
                n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                policy_kwargs={"net_arch": [64, 64]}, seed=seed, verbose=0, device="cpu",
            )
            model.learn(total_timesteps=cli.train_timesteps, progress_bar=False)
            vec_norm.training = False
            seed_dir = out / "models" / arm / f"seed{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(seed_dir / "ppo_model.zip"))
            vec_norm.save(str(seed_dir / "vec_normalize.pkl"))
            for t in tapes:
                r = run_ppo_episode(model, vec_norm, arm, t)
                results[(arm, seed, t)] = r
                rows.append({"arm": arm, "train_seed": seed, "eval_seed": t, **r})
            print(f"{arm} seed {seed} trained + evaluated", flush=True)

    with (out / "factorial_rows.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary: dict[str, Any] = {
        "config": {k: str(v) for k, v in vars(cli).items()},
        "env_kwargs": CANONICAL_ENV_KWARGS,
        "arm_frozen_values": {
            a: {str(d): v for d, v in ARM_FROZEN_VALUES[a].items()}
            for a in cli.arms
        },
    }

    def contrast(arm_a: str, arm_b: str, key: str) -> dict[str, Any]:
        delta = np.array([[results[(arm_a, s, t)][key] - results[(arm_b, s, t)][key]
                           for t in tapes] for s in cli.seeds])
        per_seed = delta.mean(axis=1)
        lo2, hi2 = two_way_bootstrap(delta)
        tci = scipy_stats.t.interval(0.95, len(per_seed) - 1,
                                     loc=per_seed.mean(), scale=scipy_stats.sem(per_seed))
        return {
            "mean": float(delta.mean()),
            "two_way_ci95": [lo2, hi2],
            "seed_t_ci95": [float(tci[0]), float(tci[1])],
            "per_seed": [float(v) for v in per_seed],
            "seeds_positive": int((per_seed > 0).sum()),
        }

    def arm_vs_static(arm: str, key: str) -> dict[str, Any]:
        if not static_by_tape:
            return {}
        delta = np.array([[results[(arm, s, t)][key] - static_by_tape[t][key]
                           for t in tapes] for s in cli.seeds])
        lo2, hi2 = two_way_bootstrap(delta)
        return {"mean": float(delta.mean()), "two_way_ci95": [lo2, hi2]}

    for key in ("ret_excel", "ret_excel_cvar05"):
        block: dict[str, Any] = {
            "arm_means": {a: float(np.mean([[results[(a, s, t)][key] for t in tapes]
                                            for s in cli.seeds])) for a in cli.arms},
        }
        if "joint" in cli.arms and "upstream_shift" in cli.arms:
            block["PRIMARY_joint_minus_upstream_shift"] = contrast("joint", "upstream_shift", key)
        if "joint" in cli.arms and "dispatch_only" in cli.arms:
            block["joint_minus_dispatch_only"] = contrast("joint", "dispatch_only", key)
        for a in cli.arms:
            block[f"{a}_minus_static"] = arm_vs_static(a, key)
        summary[key] = block

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
