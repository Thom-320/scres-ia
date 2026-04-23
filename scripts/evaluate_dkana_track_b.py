#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable

_CACHE_ROOT = Path(tempfile.gettempdir()) / "mfsc_runtime_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_b_smoke import STATIC_POLICY_SPECS, build_static_policy_action
from supply_chain.dkana import DKANAOnlinePolicyAdapter, DKANAPolicy
from supply_chain.external_env_interface import (
    STATE_CONSTRAINT_FIELDS,
    get_episode_terminal_metrics,
    make_track_b_env,
)

PolicyFn = Callable[[np.ndarray, dict[str, Any]], np.ndarray | dict[str, float | int]]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DKANA checkpoint on the Track B lane."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--risk-level", default="adaptive_benchmark_v2")
    parser.add_argument("--reward-mode", default="ReT_seq_v1")
    parser.add_argument("--observation-version", default="v7")
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument("--ppo-model-path", type=Path, default=None)
    parser.add_argument("--ppo-vec-normalize-path", type=Path, default=None)
    return parser


def extract_downstream_multipliers(info: dict[str, Any]) -> tuple[float, float]:
    clipped_action = info.get("clipped_action")
    if (
        isinstance(clipped_action, (list, tuple, np.ndarray))
        and len(clipped_action) >= 7
    ):
        return (
            float(1.25 + 0.75 * float(clipped_action[5])),
            float(1.25 + 0.75 * float(clipped_action[6])),
        )
    return 1.0, 1.0


def load_dkana_policy(checkpoint_path: Path) -> tuple[DKANAPolicy, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = dict(checkpoint["model_config"])
    model = DKANAPolicy(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def make_ppo_policy(args: argparse.Namespace) -> PolicyFn:
    if args.ppo_model_path is None:
        raise ValueError("--ppo-model-path is required.")
    model = PPO.load(str(args.ppo_model_path), device="cpu")
    vec_norm: VecNormalize | None = None
    if args.ppo_vec_normalize_path is not None:
        dummy_vec = DummyVecEnv(
            [
                lambda: make_track_b_env(
                    reward_mode=args.reward_mode,
                    risk_level=args.risk_level,
                    observation_version=args.observation_version,
                    step_size_hours=args.step_size_hours,
                    max_steps=args.max_steps,
                )
            ]
        )
        vec_norm = VecNormalize.load(str(args.ppo_vec_normalize_path), dummy_vec)
        vec_norm.training = False
        vec_norm.norm_reward = False

    def policy(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        del info
        obs_batch = np.asarray(obs, dtype=np.float32)[None, :]
        if vec_norm is not None:
            obs_batch = vec_norm.normalize_obs(obs_batch)
        action, _ = model.predict(obs_batch, deterministic=True)
        return np.asarray(action[0], dtype=np.float32)

    return policy


def run_policy(
    policy_name: str,
    policy_fn: PolicyFn,
    *,
    args: argparse.Namespace,
    reset_fn: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode_idx in range(args.episodes):
        if reset_fn is not None:
            reset_fn()
        env = make_track_b_env(
            reward_mode=args.reward_mode,
            risk_level=args.risk_level,
            observation_version=args.observation_version,
            step_size_hours=args.step_size_hours,
            max_steps=args.max_steps,
        )
        obs, info = env.reset(seed=args.seed + episode_idx)
        terminated = False
        truncated = False
        reward_total = 0.0
        demanded_total = 0.0
        backorder_qty_total = 0.0
        shift_counts = {1: 0, 2: 0, 3: 0}
        op10_multipliers: list[float] = []
        op12_multipliers: list[float] = []
        steps = 0
        final_info = info

        while not (terminated or truncated):
            action = policy_fn(np.asarray(obs, dtype=np.float32), final_info)
            obs, reward, terminated, truncated, final_info = env.step(action)
            reward_total += float(reward)
            demanded_total += float(final_info.get("new_demanded", 0.0))
            backorder_qty_total += float(final_info.get("new_backorder_qty", 0.0))
            shift_counts[int(final_info.get("shifts_active", 1))] += 1
            op10_mult, op12_mult = extract_downstream_multipliers(final_info)
            op10_multipliers.append(op10_mult)
            op12_multipliers.append(op12_mult)
            steps += 1

        terminal_metrics = get_episode_terminal_metrics(env)
        total_steps = max(1, steps)
        flow_backorder_rate = (
            backorder_qty_total / demanded_total if demanded_total > 0.0 else 0.0
        )
        row = {
            "policy": policy_name,
            "seed": args.seed + episode_idx,
            "episode": episode_idx + 1,
            "steps": steps,
            "reward_total": reward_total,
            "fill_rate": float(terminal_metrics["fill_rate_order_level"]),
            "backorder_rate": float(terminal_metrics["backorder_rate_order_level"]),
            "order_level_ret_mean": float(terminal_metrics["order_level_ret_mean"]),
            "flow_fill_rate": 1.0 - flow_backorder_rate,
            "flow_backorder_rate": flow_backorder_rate,
            "pct_steps_S1": 100.0 * shift_counts[1] / total_steps,
            "pct_steps_S2": 100.0 * shift_counts[2] / total_steps,
            "pct_steps_S3": 100.0 * shift_counts[3] / total_steps,
            "op10_multiplier_step_mean": float(np.mean(op10_multipliers or [1.0])),
            "op12_multiplier_step_mean": float(np.mean(op12_multipliers or [1.0])),
        }
        rows.append(row)
        env.close()
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    policies = sorted({str(row["policy"]) for row in rows})
    summary: list[dict[str, Any]] = []
    metrics = ("reward_total", "fill_rate", "backorder_rate", "order_level_ret_mean")
    for policy in policies:
        policy_rows = [row for row in rows if row["policy"] == policy]
        out: dict[str, Any] = {"policy": policy, "episodes": len(policy_rows)}
        for metric in metrics:
            values = np.asarray(
                [float(row[metric]) for row in policy_rows], dtype=np.float64
            )
            out[f"{metric}_mean"] = float(values.mean())
            out[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(out)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_dkana_policy(args.checkpoint)
    dataset_metadata = checkpoint["dataset_metadata"]
    model_config = checkpoint["model_config"]
    observation_fields = tuple(dataset_metadata["env_spec"]["observation_fields"])
    relation_mode = str(dataset_metadata.get("relation_mode", "equality"))
    window_size = int(dataset_metadata["window_size"])
    action_dim = int(model_config["action_dim"])
    dkana_adapter = DKANAOnlinePolicyAdapter(
        model,
        window_size=window_size,
        observation_fields=observation_fields,
        state_constraint_fields=STATE_CONSTRAINT_FIELDS,
        action_dim=action_dim,
        relation_mode=relation_mode,
    )

    all_rows = run_policy(
        "dkana",
        dkana_adapter,
        args=args,
        reset_fn=dkana_adapter.reset,
    )
    for spec in STATIC_POLICY_SPECS:
        action_payload = build_static_policy_action(spec)
        all_rows.extend(
            run_policy(
                spec.label,
                lambda obs, info, payload=action_payload: dict(payload),
                args=args,
            )
        )
    if args.ppo_model_path is not None:
        all_rows.extend(run_policy("ppo", make_ppo_policy(args), args=args))

    summary = summarize(all_rows)
    write_csv(args.output_dir / "episode_metrics.csv", all_rows)
    write_csv(args.output_dir / "summary_metrics.csv", summary)
    with (args.output_dir / "evaluation_metadata.json").open(
        "w", encoding="utf-8"
    ) as file_obj:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "episodes": args.episodes,
                "seed": args.seed,
                "risk_level": args.risk_level,
                "reward_mode": args.reward_mode,
                "observation_version": args.observation_version,
                "relation_mode": relation_mode,
                "window_size": window_size,
                "action_dim": action_dim,
            },
            file_obj,
            indent=2,
        )
    print(f"Wrote DKANA Track B evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
