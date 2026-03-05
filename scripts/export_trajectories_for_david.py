#!/usr/bin/env python3
"""
Export DES observation trajectories for external model training (e.g., DKANA).

Generates numpy arrays from the shift_control environment using random policy.
David can load these directly into his PyTorch pipeline.

Usage:
    python scripts/export_trajectories_for_david.py
    python scripts/export_trajectories_for_david.py --episodes 200 --output-dir data_for_dkana
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from supply_chain.external_env_interface import make_shift_control_env, get_shift_control_env_spec, spec_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trajectories for external models.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to collect.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/data_export"))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--risk-level", default="current", choices=["current", "increased", "severe"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_obs = []
    all_actions = []
    all_rewards = []
    all_episode_ids = []
    episode_lengths = []

    for ep in range(args.episodes):
        env = make_shift_control_env(risk_level=args.risk_level)
        obs, _ = env.reset(seed=args.seed_start + ep)
        ep_obs, ep_actions, ep_rewards = [obs.copy()], [], []
        done, truncated = False, False

        while not (done or truncated):
            action = env.action_space.sample()
            obs, reward, done, truncated, _ = env.step(action)
            ep_obs.append(obs.copy())
            ep_actions.append(action.copy())
            ep_rewards.append(reward)

        T = len(ep_actions)
        episode_lengths.append(T)
        all_obs.extend(ep_obs[:T])  # align with actions
        all_actions.extend(ep_actions)
        all_rewards.extend(ep_rewards)
        all_episode_ids.extend([ep] * T)

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{args.episodes} done (T={T} steps)")

    # Save as numpy
    np.save(args.output_dir / "observations.npy", np.array(all_obs, dtype=np.float32))
    np.save(args.output_dir / "actions.npy", np.array(all_actions, dtype=np.float32))
    np.save(args.output_dir / "rewards.npy", np.array(all_rewards, dtype=np.float32))
    np.save(args.output_dir / "episode_ids.npy", np.array(all_episode_ids, dtype=np.int32))

    # Save env spec as JSON
    spec = get_shift_control_env_spec()
    with (args.output_dir / "env_spec.json").open("w") as f:
        json.dump(spec_to_dict(spec), f, indent=2)

    # Save metadata
    meta = {
        "episodes": args.episodes,
        "total_steps": len(all_rewards),
        "episode_lengths": episode_lengths,
        "risk_level": args.risk_level,
        "obs_shape": [len(all_obs), 15],
        "action_shape": [len(all_actions), 5],
        "policy": "random",
        "note": "Collected with random policy. Use for offline training or as baseline data.",
    }
    with (args.output_dir / "metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nExported {len(all_rewards):,} steps from {args.episodes} episodes")
    print(f"  observations.npy  shape=({len(all_obs)}, 15)")
    print(f"  actions.npy       shape=({len(all_actions)}, 5)")
    print(f"  rewards.npy       shape=({len(all_rewards)},)")
    print(f"  episode_ids.npy   shape=({len(all_episode_ids)},)")
    print(f"  env_spec.json     (environment contract)")
    print(f"  metadata.json     (collection metadata)")
    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
