#!/usr/bin/env python3
"""Train one scientifically motivated Q2 learner on burned development data.

Supported candidates are the frozen RecurrentPPO incumbent, distributional
QR-DQN for the four-action discrete contract, and frame-stacked PPO as a memory
ablation.  This runner does not adjudicate against MPC by itself.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sb3_contrib import QRDQN, RecurrentPPO  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS, ProgramORetOnlyEnv  # noqa: E402
from supply_chain.program_q2_history import CausalFrameStack  # noqa: E402
from supply_chain.program_q2_reward import (  # noqa: E402
    ProgramQ2RewardWrapper,
    TerminalRewardCalibration,
)


def _potential(path: Path | None):
    if path is None:
        return None
    payload = json.loads(path.read_text())
    weights = np.asarray(payload["weights"], dtype=float)
    bias = float(payload.get("bias", 0.0))
    if weights.shape != (21,):
        raise ValueError("PBRS potential must contain exactly 21 frozen weights")
    return lambda observation: float(bias + weights @ np.asarray(observation, dtype=float)[-21:])


def make_env(args, *, raw_evaluation: bool = False):
    env = ProgramORetOnlyEnv(
        scheduler=scheduler(),
        tape_seed_start=args.training_seed_start,
        tape_seed_end=args.training_seed_end,
    )
    if not raw_evaluation:
        calibration = (
            TerminalRewardCalibration(args.reward_mean, args.reward_standard_deviation)
            if args.reward_mode == "standardized_terminal"
            else None
        )
        potential = _potential(args.potential_json)
        if args.reward_mode == "pbrs_terminal" and potential is None:
            raise ValueError("PBRS requires --potential-json frozen before training")
        env = ProgramQ2RewardWrapper(
            env,
            mode=args.reward_mode,
            calibration=calibration,
            potential=potential or (lambda _observation: 0.0),
        )
    if args.algorithm in ("qrdqn", "ppo_frame_stack"):
        env = CausalFrameStack(env, frames=8)
    return env


def train(args):
    env = make_env(args)
    common = dict(seed=args.optimizer_seed, gamma=args.gamma, verbose=0, device="cpu")
    if args.algorithm == "recurrent_ppo":
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.entropy,
            policy_kwargs={"lstm_hidden_size": args.hidden_size, "net_arch": [args.hidden_size, args.hidden_size]},
            **common,
        )
    elif args.algorithm == "qrdqn":
        model = QRDQN(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            buffer_size=min(args.replay_buffer_size, args.total_timesteps),
            learning_starts=min(args.learning_starts, max(1, args.total_timesteps // 10)),
            batch_size=args.batch_size,
            train_freq=4,
            gradient_steps=1,
            exploration_fraction=0.20,
            exploration_final_eps=0.02,
            policy_kwargs={"net_arch": [args.hidden_size, args.hidden_size], "n_quantiles": args.n_quantiles},
            **common,
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.entropy,
            policy_kwargs={"net_arch": [args.hidden_size, args.hidden_size]},
            **common,
        )
    log_dir = args.output / "training_log"
    model.set_logger(configure(str(log_dir), ["csv", "json"]))
    started = time.perf_counter()
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
    elapsed = float(time.perf_counter() - started)
    model.save(args.output / "model")
    return model, elapsed


def evaluate(model, args) -> list[dict]:
    env = make_env(args, raw_evaluation=True)
    rows = []
    for cell_index, cell in enumerate(CONFIRMED_RET_CELLS):
        for tape in args.evaluation_tapes:
            observation, _ = env.reset(options={"cell_index": cell_index, "tape_seed": tape})
            state = None
            episode_start = np.ones((1,), dtype=bool)
            calendar = []
            while True:
                if args.algorithm == "recurrent_ppo":
                    action, state = model.predict(
                        observation,
                        state=state,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                    episode_start[:] = False
                else:
                    action, _ = model.predict(observation, deterministic=True)
                value = int(np.asarray(action).reshape(-1)[0])
                calendar.append(value)
                observation, _, terminated, truncated, info = env.step(value)
                if terminated or truncated:
                    break
            metrics = info["metrics"]
            rows.append(
                {
                    "cell": cell.cell_id,
                    "tape": tape,
                    "calendar": calendar,
                    **{key: float(metrics[key]) for key in (
                        "ret_visible",
                        "worst_product_fill",
                        "lost_orders",
                        "gross_policy_batch_slots",
                        "gross_production_quantity",
                        "charged_downstream_vehicle_hours",
                    )},
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=("recurrent_ppo", "qrdqn", "ppo_frame_stack"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-timesteps", type=int, default=60_000)
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--training-seed-start", type=int, required=True)
    parser.add_argument("--training-seed-end", type=int, required=True)
    parser.add_argument("--evaluation-tapes", type=lambda value: tuple(map(int, value.split(","))), default=(94800001, 94800002, 94800003))
    parser.add_argument("--reward-mode", choices=("raw_terminal", "standardized_terminal", "pbrs_terminal"), default="raw_terminal")
    parser.add_argument("--reward-mean", type=float, default=0.8)
    parser.add_argument("--reward-standard-deviation", type=float, default=0.1)
    parser.add_argument("--potential-json", type=Path)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--hidden-size", type=int, choices=(64, 128), default=64)
    parser.add_argument("--n-quantiles", type=int, default=200)
    parser.add_argument("--replay-buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    required_episodes = int(np.ceil(args.total_timesteps / 8)) + 2
    available = args.training_seed_end - args.training_seed_start + 1
    if available < required_episodes:
        raise ValueError(f"training range has {available} seeds; at least {required_episodes} are required")
    model, elapsed = train(args)
    rows = evaluate(model, args)
    payload = {
        "schema_version": "program_q2_minimal_learner_run_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_status": "BURNED_DEVELOPMENT_NO_CLAIM",
        "algorithm": args.algorithm,
        "reward_mode": args.reward_mode,
        "optimizer_seed": args.optimizer_seed,
        "total_timesteps": args.total_timesteps,
        "training_seed_range": [args.training_seed_start, args.training_seed_end],
        "evaluation_tapes": list(args.evaluation_tapes),
        "elapsed_seconds": elapsed,
        "rows": rows,
        "summary": {
            "mean_ret_visible": float(np.mean([row["ret_visible"] for row in rows])),
            "mean_worst_product_fill": float(np.mean([row["worst_product_fill"] for row in rows])),
            "lost_orders_max": float(max(row["lost_orders"] for row in rows)),
            "resources_exact": all(row["gross_policy_batch_slots"] == 24.0 for row in rows),
        },
        "adjudication": "NOT_RUN_REQUIRES_FROZEN_BEST_STRUCTURED_AND_MULTI_SEED_PANEL",
    }
    (args.output / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
