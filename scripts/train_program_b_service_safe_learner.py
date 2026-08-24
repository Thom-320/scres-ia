#!/usr/bin/env python3
"""Train one development learner under the frozen service-safe Program B reward."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sb3_contrib import RecurrentPPO  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from supply_chain.program_o_ret_env import ProgramORetOnlyEnv  # noqa: E402

CONTRACT = ROOT / "contracts/program_b_service_safe_learner_v1.json"
PARENT = ROOT / "contracts/program_o_full_des_hpi_translation_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scheduler() -> dict[str, list[str]]:
    parent = json.loads(PARENT.read_text())
    key = parent["action"]["primary_scheduler"]
    return parent["action"]["within_week_schedulers"][key]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("PPO_MLP", "RecurrentPPO_MLP"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timesteps", type=int)
    args = parser.parse_args()
    contract = json.loads(CONTRACT.read_text())
    if args.seed not in set(map(int, contract["learner"]["seeds"])):
        raise SystemExit("seed is outside the frozen development learner set")
    timesteps = int(args.timesteps or contract["learner"]["timesteps_per_seed"])
    training_start, training_end = map(int, contract["training_tapes"]["range"])
    env = ProgramORetOnlyEnv(
        scheduler=scheduler(),
        tape_seed_start=training_start,
        tape_seed_end=training_end,
        reward_mode="service_safe",
    )
    hp = contract["learner"]["hyperparameters"]
    if args.architecture == "PPO_MLP":
        model = PPO(
            "MlpPolicy",
            env,
            seed=args.seed,
            policy_kwargs={"net_arch": list(map(int, contract["learner"]["policy_kwargs"]["net_arch"]))},
            device=contract["learner"]["device"],
            verbose=0,
            learning_rate=float(hp["learning_rate"]),
            n_steps=int(hp["n_steps"]),
            batch_size=int(hp["batch_size"]),
            gamma=float(hp["gamma"]),
            gae_lambda=float(hp["gae_lambda"]),
            clip_range=float(hp["clip_range"]),
            ent_coef=float(hp["ent_coef"]),
        )
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            seed=args.seed,
            policy_kwargs={
                "net_arch": list(map(int, contract["learner"]["policy_kwargs"]["net_arch"])),
                "lstm_hidden_size": int(contract["learner"]["policy_kwargs"]["lstm_hidden_size"]),
            },
            device=contract["learner"]["device"],
            verbose=0,
            learning_rate=float(hp["learning_rate"]),
            n_steps=int(hp["n_steps"]),
            batch_size=int(hp["batch_size"]),
            gamma=float(hp["gamma"]),
            gae_lambda=float(hp["gae_lambda"]),
            clip_range=float(hp["clip_range"]),
            ent_coef=float(hp["ent_coef"]),
        )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.architecture.lower()}_seed_{args.seed}"
    model_path = args.output / stem
    model.save(model_path)
    zip_path = model_path.with_suffix(".zip")
    manifest = {
        "schema_version": "program_b_service_safe_training_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": args.architecture,
        "seed": args.seed,
        "timesteps_requested": timesteps,
        "timesteps_executed": int(model.num_timesteps),
        "training_tapes": [training_start, training_end],
        "reward_mode": "service_safe",
        "contract_sha256": sha256(CONTRACT),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "model": str(zip_path),
        "model_sha256": sha256(zip_path),
    }
    (args.output / f"{stem}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
