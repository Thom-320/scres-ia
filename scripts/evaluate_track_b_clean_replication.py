#!/usr/bin/env python3
"""Held-out evaluation and stop-rule verdict for the Track B clean-joint replication.

Tests whether the factorial ``joint`` PPO bundle (5 seeds, 52-dim v7
throughout, ``control_v1``, full 8-D contract) retains its post-hoc ret_excel
advantage over the frozen best full-contract static policy on completely
fresh tapes (500001-500060) that neither the checkpoints nor the human have
ever inspected.

See ``docs/TRACK_B_CLEAN_REPLICATION_PROTOCOL_2026-07-10.md`` for the frozen
protocol and stop rule.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_b_contract_factorial import make_arm_env  # noqa: E402
from scripts.run_track_b_crossed_eval import (  # noqa: E402
    CANONICAL_ENV_KWARGS,
    episode_metrics_row,
    two_way_bootstrap,
)
from supply_chain.external_env_interface import make_track_b_env  # noqa: E402

CONFIRM_MIN = 500_061
CONFIRM_MAX = 500_120


def run_constant_episode(action: list[float], tape: int) -> dict[str, float]:
    """Roll out the frozen static comparator (constant 8-signal vector)."""
    env = make_track_b_env(**CANONICAL_ENV_KWARGS)
    env.reset(seed=tape)
    terminated = truncated = False
    fixed = np.asarray(action, dtype=np.float32)
    while not (terminated or truncated):
        _o, _r, terminated, truncated, _i = env.step(fixed)
    metrics = episode_metrics_row(env.unwrapped.sim)
    env.close()
    return metrics


def joint_worker(task: tuple[int, str, list[int]]) -> list[dict[str, Any]]:
    """Evaluate one joint checkpoint across all confirmatory tapes."""
    seed, seed_dir_raw, tapes = task
    seed_dir = Path(seed_dir_raw)
    model = PPO.load(str(seed_dir / "ppo_model.zip"), device="cpu")
    vec_norm = VecNormalize.load(
        str(seed_dir / "vec_normalize.pkl"), DummyVecEnv([lambda: make_arm_env("joint")])
    )
    vec_norm.training = False
    rows: list[dict[str, Any]] = []
    for tape in tapes:
        env = make_arm_env("joint")
        obs, _ = env.reset(seed=tape)
        terminated = truncated = False
        while not (terminated or truncated):
            obs_n = vec_norm.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])
            action, _ = model.predict(obs_n, deterministic=True)
            obs, _r, terminated, truncated, _i = env.step(
                np.asarray(action[0], dtype=np.float32)
            )
        rows.append(
            {
                "family": "joint",
                "train_seed": seed,
                "eval_seed": tape,
                **episode_metrics_row(env.unwrapped.sim),
            }
        )
        env.close()
    return rows


def summarize(delta: np.ndarray, seeds: list[int], tapes: list[int]) -> dict[str, Any]:
    per_seed = delta.mean(axis=1)
    per_tape = delta.mean(axis=0)
    lo, hi = two_way_bootstrap(delta)
    return {
        "mean_delta": float(delta.mean()),
        "two_way_ci95": [lo, hi],
        "per_seed_mean": {str(s): float(v) for s, v in zip(seeds, per_seed)},
        "seeds_positive": int((per_seed > 0).sum()),
        "tapes_positive": int((per_tape > 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-static", type=Path, required=True)
    parser.add_argument(
        "--joint-root",
        type=Path,
        required=True,
        help="Root containing models/joint/seed{N}/ppo_model.zip",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--test-seed-base", type=int, default=CONFIRM_MIN)
    parser.add_argument("--test-tapes", type=int, default=60)
    args = parser.parse_args()
    tapes = list(range(args.test_seed_base, args.test_seed_base + args.test_tapes))
    if min(tapes) < CONFIRM_MIN or max(tapes) > CONFIRM_MAX:
        raise SystemExit(f"confirmatory tapes must remain in {CONFIRM_MIN}..{CONFIRM_MAX}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Frozen static comparator: calibration-only, unchanged from the challenge.
    frozen = json.loads(args.frozen_static.read_text(encoding="utf-8"))
    action = [float(v) for v in frozen["candidate"]["signals"]]
    rows: list[dict[str, Any]] = []
    static_by_tape: dict[int, dict[str, float]] = {}
    for tape in tapes:
        metrics = run_constant_episode(action, tape)
        static_by_tape[tape] = metrics
        rows.append(
            {"family": "best_full_static", "train_seed": 0, "eval_seed": tape, **metrics}
        )
    print("held-out static complete", flush=True)

    # Joint bundle: 5 seeds, identical frozen checkpoints.
    joint_tasks = [
        (seed, str(args.joint_root / "models" / "joint" / f"seed{seed}"), tapes)
        for seed in range(1, 6)
    ]
    joint_by_key: dict[tuple[int, int], dict[str, float]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for block in pool.map(joint_worker, joint_tasks):
            rows.extend(block)
    for r in rows:
        if r["family"] == "joint":
            joint_by_key[(int(r["train_seed"]), int(r["eval_seed"]))] = r
    print("joint checkpoints complete", flush=True)

    with (args.output_dir / "replication_rows.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    seeds = list(range(1, 6))
    result: dict[str, Any] = {
        "protocol": "docs/TRACK_B_CLEAN_REPLICATION_PROTOCOL_2026-07-10.md",
        "test_tapes": tapes,
        "frozen_static_policy": frozen,
    }
    for key in ("ret_excel", "ret_excel_cvar05"):
        delta = np.array(
            [
                [joint_by_key[(seed, tape)][key] - static_by_tape[tape][key] for tape in tapes]
                for seed in seeds
            ]
        )
        summary_block = summarize(delta, seeds, tapes)
        result[key] = {
            "joint_minus_best_full_static": summary_block,
            "means": {
                "best_full_static": float(
                    np.mean([r[key] for r in rows if r["family"] == "best_full_static"])
                ),
                "joint": float(
                    np.mean([r[key] for r in rows if r["family"] == "joint"])
                ),
            },
        }

    primary = result["ret_excel"]["joint_minus_best_full_static"]
    passed = (
        primary["two_way_ci95"][0] > 0
        and primary["seeds_positive"] >= 4
        and primary["tapes_positive"] >= 50
    )
    result["stop_rule"] = {
        "passed": bool(passed),
        "verdict": (
            "GAP_REPLICATES_OPEN_ADAPTIVE_CASE"
            if passed
            else "GAP_DOES_NOT_REPLICATE_PROCEED_PASO_1"
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
