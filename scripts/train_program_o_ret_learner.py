#!/usr/bin/env python3
"""Train exactly one frozen Program O-R RecurrentPPO seed."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sb3_contrib import RecurrentPPO  # noqa: E402

from supply_chain.program_o_ret_env import ProgramORetOnlyEnv  # noqa: E402
from supply_chain.program_o_ret_freeze import verify_execution_freeze  # noqa: E402


CONTRACT = ROOT / "contracts/program_o_ret_only_learner_v1.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scheduler() -> dict[str, list[str]]:
    parent = json.loads(
        (ROOT / "contracts/program_o_full_des_hpi_translation_v1.json").read_text()
    )
    key = parent["action"]["primary_scheduler"]
    return parent["action"]["within_week_schedulers"][key]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learner-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--timesteps", type=int)
    args = parser.parse_args()

    contract = json.loads(CONTRACT.read_text())
    seeds = list(map(int, contract["learner"]["learner_seeds"]))
    if args.learner_index not in range(len(seeds)):
        raise SystemExit("learner-index outside frozen 0..9 range")
    learner_seed = seeds[args.learner_index]
    if args.smoke:
        start = int(contract["tape_custody"]["nonscientific_smoke"])
        end = start + 64
        requested_timesteps = int(args.timesteps or 16)
    else:
        if contract["status"] != "FROZEN_BEFORE_748_SCIENTIFIC_SEEDS":
            raise SystemExit("scientific training blocked until source/execution freeze")
        verify_execution_freeze(ROOT, CONTRACT)
        training_start, _training_end = map(int, contract["tape_custody"]["training"])
        start = training_start + args.learner_index * 25_025
        end = start + 25_024
        requested_timesteps = int(contract["learner"]["requested_timesteps_per_seed"])

    env = ProgramORetOnlyEnv(
        scheduler=scheduler(), tape_seed_start=start, tape_seed_end=end
    )
    hp = contract["learner"]["hyperparameters"]
    policy_kwargs = contract["learner"]["policy_kwargs"]
    model = RecurrentPPO(
        contract["learner"]["policy"],
        env,
        seed=learner_seed,
        learning_rate=float(hp["learning_rate"]),
        n_steps=int(hp["n_steps"]),
        batch_size=int(hp["batch_size"]),
        gamma=float(hp["gamma"]),
        gae_lambda=float(hp["gae_lambda"]),
        clip_range=float(hp["clip_range"]),
        ent_coef=float(hp["ent_coef"]),
        policy_kwargs={
            "lstm_hidden_size": int(policy_kwargs["lstm_hidden_size"]),
            "net_arch": list(map(int, policy_kwargs["net_arch"])),
        },
        verbose=1,
        device="cpu",
    )
    model.learn(total_timesteps=requested_timesteps, progress_bar=False)
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / f"recurrent_ppo_seed_{learner_seed}"
    model.save(model_path)
    zip_path = model_path.with_suffix(".zip")
    manifest = {
        "schema_version": "program_o_ret_learner_training_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "smoke": bool(args.smoke),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "contract_sha256": digest(CONTRACT),
        "learner_seed": learner_seed,
        "learner_index": args.learner_index,
        "tape_range": [start, end],
        "requested_timesteps": requested_timesteps,
        "executed_timesteps": int(model.num_timesteps),
        "model": str(zip_path.relative_to(ROOT) if zip_path.is_relative_to(ROOT) else zip_path),
        "model_sha256": digest(zip_path),
    }
    (args.output / f"training_manifest_seed_{learner_seed}.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
