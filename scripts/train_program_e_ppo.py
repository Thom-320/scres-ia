#!/usr/bin/env python3
"""Train the single preregistered Program E MaskablePPO configuration."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch.nn as nn

from supply_chain.dra2_policy_env import ProgramEConvoyEnv


DEFAULT_SEEDS = tuple(range(9301, 9311))


def train_one(payload: tuple[int, int, str, str, str]) -> dict:
    seed, timesteps, tapes_path, normalizers_path, output_dir = payload
    tapes = json.loads(Path(tapes_path).read_text())
    normalizers = json.loads(Path(normalizers_path).read_text())

    def make_env():
        return ProgramEConvoyEnv(tapes, normalizers, episode_days=56, random_tapes=True)

    env = DummyVecEnv([make_env])
    model = MaskablePPO(
        "MlpPolicy", env,
        learning_rate=3e-4, n_steps=2048, batch_size=64,
        gamma=.99, gae_lambda=.95, clip_range=.2, ent_coef=.01,
        policy_kwargs={"net_arch": [64, 64], "activation_fn": nn.Tanh},
        seed=seed, verbose=0, device="cpu",
    )
    started = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    path = Path(output_dir) / f"maskable_ppo_seed_{seed}"
    model.save(path)
    elapsed = time.time() - started
    env.close()
    return {
        "learner_seed": seed, "timesteps": timesteps,
        "model_path": str(path.with_suffix(".zip")),
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tapes", type=Path, default=Path("results/program_e/data/training_tapes.json"))
    parser.add_argument("--normalizers", type=Path, default=Path("results/program_e/data/normalizers.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/program_e/ppo"))
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    payloads = [
        (seed, args.timesteps, str(args.tapes), str(args.normalizers), str(args.output_dir))
        for seed in seeds
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(seeds))) as pool:
        futures = {pool.submit(train_one, payload): payload[0] for payload in payloads}
        for future in as_completed(futures):
            row = future.result(); rows.append(row)
            print(f"[program-e-ppo] seed={row['learner_seed']} complete", flush=True)
    rows.sort(key=lambda row: row["learner_seed"])
    verdict = {
        "algorithm": "MaskablePPO", "seeds": list(seeds),
        "timesteps_per_seed": args.timesteps, "runs_complete": len(rows),
        "validation_tapes_opened": 0, "virgin_tapes_opened": 0,
        "interpretation": (
            "PROGRAM_E_PPO_TRAINING_COMPLETE"
            if args.timesteps == 200_000 and seeds == DEFAULT_SEEDS
            else "PROGRAM_E_PPO_TECHNICAL_SMOKE"
        ),
        "runs": rows,
    }
    (args.output_dir / "training_verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
