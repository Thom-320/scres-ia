#!/usr/bin/env python3
"""Held-out evaluation and stop-rule verdict for the Track B contract challenge."""
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
    load_policy,
    run_ppo_episode,
    two_way_bootstrap,
)
from supply_chain.external_env_interface import make_track_b_env  # noqa: E402

TEST_MIN = 400_001
TEST_MAX = 400_060


def run_constant_episode(action: list[float], tape: int) -> dict[str, float]:
    env = make_track_b_env(**CANONICAL_ENV_KWARGS)
    env.reset(seed=tape)
    terminated = truncated = False
    fixed = np.asarray(action, dtype=np.float32)
    while not (terminated or truncated):
        _o, _r, terminated, truncated, _i = env.step(fixed)
    metrics = episode_metrics_row(env.unwrapped.sim)
    env.close()
    return metrics


def canonical_worker(task: tuple[int, str, list[int]]) -> list[dict[str, Any]]:
    seed, seed_dir_raw, tapes = task
    model, vec_norm, n_obs = load_policy(Path(seed_dir_raw))
    return [
        {"family": "canonical_joint", "train_seed": seed, "eval_seed": tape,
         **run_ppo_episode(model, vec_norm, tape, n_obs)}
        for tape in tapes
    ]


def factorial_worker(task: tuple[str, int, str, list[int]]) -> list[dict[str, Any]]:
    arm, seed, seed_dir_raw, tapes = task
    seed_dir = Path(seed_dir_raw)
    model = PPO.load(str(seed_dir / "ppo_model.zip"), device="cpu")
    vec_norm = VecNormalize.load(
        str(seed_dir / "vec_normalize.pkl"), DummyVecEnv([lambda: make_arm_env(arm)])
    )
    vec_norm.training = False
    rows = []
    for tape in tapes:
        env = make_arm_env(arm)
        obs, _ = env.reset(seed=tape)
        terminated = truncated = False
        while not (terminated or truncated):
            obs_n = vec_norm.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])
            action, _ = model.predict(obs_n, deterministic=True)
            obs, _r, terminated, truncated, _i = env.step(np.asarray(action[0], dtype=np.float32))
        rows.append({"family": arm, "train_seed": seed, "eval_seed": tape,
                     **episode_metrics_row(env.unwrapped.sim)})
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
    parser.add_argument("--canonical-root-a", type=Path, required=True)
    parser.add_argument("--canonical-root-b", type=Path, required=True)
    parser.add_argument("--factorial-joint-root", type=Path, required=True)
    parser.add_argument("--anchored-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--test-seed-base", type=int, default=TEST_MIN)
    parser.add_argument("--test-tapes", type=int, default=60)
    args = parser.parse_args()
    tapes = list(range(args.test_seed_base, args.test_seed_base + args.test_tapes))
    if min(tapes) < TEST_MIN or max(tapes) > TEST_MAX:
        raise SystemExit(f"final tapes must remain in {TEST_MIN}..{TEST_MAX}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frozen = json.loads(args.frozen_static.read_text(encoding="utf-8"))
    action = [float(v) for v in frozen["candidate"]["signals"]]
    rows: list[dict[str, Any]] = []
    static_by_tape = {}
    for tape in tapes:
        metrics = run_constant_episode(action, tape)
        static_by_tape[tape] = metrics
        rows.append({"family": "best_full_static", "train_seed": 0,
                     "eval_seed": tape, **metrics})
    print("held-out static complete", flush=True)

    canonical_tasks = []
    for seed in range(1, 11):
        root = args.canonical_root_a if seed <= 5 else args.canonical_root_b
        canonical_tasks.append((seed, str(root / "models" / f"seed{seed}"), tapes))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for block in pool.map(canonical_worker, canonical_tasks):
            rows.extend(block)
    print("canonical checkpoints complete", flush=True)

    factorial_tasks = []
    for seed in range(1, 6):
        factorial_tasks.append(("joint", seed, str(args.factorial_joint_root / "models" / "joint" / f"seed{seed}"), tapes))
        factorial_tasks.append(("upstream_shift_best_dispatch", seed,
                                str(args.anchored_root / "models" / "upstream_shift_best_dispatch" / f"seed{seed}"), tapes))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for block in pool.map(factorial_worker, factorial_tasks):
            rows.extend(block)
    print("factorial joint and anchored checkpoints complete", flush=True)

    with (args.output_dir / "challenge_rows.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by = {(r["family"], int(r["train_seed"]), int(r["eval_seed"])): r for r in rows}
    result: dict[str, Any] = {
        "protocol": "docs/TRACK_B_SAME_CONTRACT_CHALLENGE_PROTOCOL_2026-07-10.md",
        "test_tapes": tapes,
        "frozen_static_policy": frozen,
    }
    for key in ("ret_excel", "ret_excel_cvar05"):
        static_delta = np.array([
            [by[("canonical_joint", seed, tape)][key] - static_by_tape[tape][key] for tape in tapes]
            for seed in range(1, 11)
        ])
        anchored_delta = np.array([
            [by[("joint", seed, tape)][key] - by[("upstream_shift_best_dispatch", seed, tape)][key]
             for tape in tapes]
            for seed in range(1, 6)
        ])
        result[key] = {
            "canonical_joint_minus_best_full_static": summarize(static_delta, list(range(1, 11)), tapes),
            "factorial_joint_minus_upstream_shift_best_dispatch": summarize(anchored_delta, list(range(1, 6)), tapes),
            "means": {
                family: float(np.mean([r[key] for r in rows if r["family"] == family]))
                for family in ("best_full_static", "canonical_joint", "joint", "upstream_shift_best_dispatch")
            },
        }
    primary = result["ret_excel"]
    c1 = primary["canonical_joint_minus_best_full_static"]
    c2 = primary["factorial_joint_minus_upstream_shift_best_dispatch"]
    passed = c1["two_way_ci95"][0] > 0 and c2["two_way_ci95"][0] > 0 and \
        c1["seeds_positive"] == 10 and c2["seeds_positive"] == 5
    result["stop_rule"] = {
        "passed": bool(passed),
        "verdict": (
            "CLOSE_PAPER_1_EXPERIMENTS_IJPR_CIE_SMPT"
            if passed else "RETIRE_BOTTLENECK_ADAPTIVE_ADVANTAGE_PIVOT_CIE_SMPT"
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
