#!/usr/bin/env python3
"""Paper 1 Blocker #1: fully crossed held-out evaluation of the Track B headline.

Motivation (independent assessment 2026-07-09, verified): the canonical 10-seed
bundle evaluates each checkpoint on eval tapes `seed+50000+ep`, so the 120 rows
contain only 21 unique tapes and a training-seed-only CI understates tape
dependence. This script evaluates ALL 10 frozen checkpoints and the ex ante
selected static comparator (S2, Op10x2.00, Op12x1.50) on the SAME fresh
held-out tape battery (default eval seeds 200001..200060 — never used by any
prior run), then reports dependence-aware inference:

- two-way cluster bootstrap (resample checkpoints AND tapes independently);
- per-checkpoint mean deltas with t-CI (10 units);
- per-tape mean deltas;
- leave-one-out sensitivity over checkpoints and tapes (dominance check).

Promotion rule (pre-registered): the headline survives iff the two-way CI is
wholly positive and no single checkpoint or tape flips the sign on
leave-one-out. Environment protocol is the canonical one: control_v1 reward,
adaptive_benchmark_v2, v7 observation, h104, thesis year basis, stochastic PT.
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

from supply_chain.config import OPERATIONS  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.external_env_interface import make_track_b_env  # noqa: E402

CANONICAL_ENV_KWARGS: dict[str, Any] = {
    "reward_mode": "control_v1",
    "observation_version": "v7",
    "risk_level": "adaptive_benchmark_v2",
    "step_size_hours": 168.0,
    "max_steps": 104,
}

# Keys as emitted by supply_chain.episode_metrics.compute_episode_metrics.
# `ret_excel` is the per-episode order-mean Garrido/Excel ReT (the primary).
METRIC_KEYS = ("ret_excel", "ret_excel_cvar05", "fill_rate", "flow_fill_rate",
               "ctj_p99")
INFERENCE_KEYS = ("ret_excel", "ret_excel_cvar05")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ckpt-root-a", type=Path,
                   default=Path("outputs/experiments/track_b_gain_2026-06-30/"
                                "top_tier_confirm_v3_output/track_b_top_tier_confirm_5seed_60k_h104"))
    p.add_argument("--seeds-a", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--ckpt-root-b", type=Path,
                   default=Path("outputs/experiments/track_b_seed_expansion_2026-07-02/"
                                "track_b_seed_expansion_6_10_claude"))
    p.add_argument("--seeds-b", nargs="+", type=int, default=[6, 7, 8, 9, 10])
    p.add_argument("--eval-seed-base", type=int, default=200_001)
    p.add_argument("--eval-episodes", type=int, default=60)
    p.add_argument("--static-shift", type=int, default=2)
    p.add_argument("--static-op10-mult", type=float, default=2.0)
    p.add_argument("--static-op12-mult", type=float, default=1.5)
    return p


def static_action(cli) -> dict[str, float | int]:
    return {
        "op3_q": float(OPERATIONS[3]["q"]),
        "op3_rop": float(OPERATIONS[3]["rop"]),
        "op9_q_min": float(OPERATIONS[9]["q"][0]),
        "op9_q_max": float(OPERATIONS[9]["q"][1]),
        "op9_rop": float(OPERATIONS[9]["rop"]),
        "op10_q_min": float(OPERATIONS[10]["q"][0]) * cli.static_op10_mult,
        "op10_q_max": float(OPERATIONS[10]["q"][1]) * cli.static_op10_mult,
        "op12_q_min": float(OPERATIONS[12]["q"][0]) * cli.static_op12_mult,
        "op12_q_max": float(OPERATIONS[12]["q"][1]) * cli.static_op12_mult,
        "assembly_shifts": int(cli.static_shift),
    }


class ObsSliceWrapper(gym.ObservationWrapper):
    """Truncate the observation to the first n dims.

    PROVENANCE NOTE (2026-07-09): seeds 1-5 of the canonical bundle were
    trained on the 48-dim v7 (before commit a3d9ea9 appended
    weeks_since_last_R22/R23/R24 + ewma_downstream_risk_rate at the END of
    v7_extra); seeds 6-10 were trained on the current 52-dim v7. Field order
    for dims 0-47 is unchanged, so obs[:48] reproduces the older training
    observation exactly. This heterogeneity must be disclosed in the paper.
    """

    def __init__(self, env: gym.Env, n_obs: int):
        super().__init__(env)
        self.n_obs = int(n_obs)
        low = env.observation_space.low[: self.n_obs]
        high = env.observation_space.high[: self.n_obs]
        self.observation_space = gym.spaces.Box(low=low, high=high,
                                                dtype=env.observation_space.dtype)

    def observation(self, observation):
        return np.asarray(observation, dtype=np.float32)[: self.n_obs]


def expected_obs_dim(seed_dir: Path) -> int:
    import pickle
    with (seed_dir / "vec_normalize.pkl").open("rb") as fh:
        vn = pickle.load(fh)
    return int(vn.observation_space.shape[0])


def load_policy(seed_dir: Path):
    n_obs = expected_obs_dim(seed_dir)
    model = PPO.load(str(seed_dir / "ppo_model.zip"), device="cpu")

    def _make():
        return ObsSliceWrapper(make_track_b_env(**CANONICAL_ENV_KWARGS), n_obs)

    vec_norm = VecNormalize.load(str(seed_dir / "vec_normalize.pkl"),
                                 DummyVecEnv([_make]))
    vec_norm.training = False
    return model, vec_norm, n_obs


def episode_metrics_row(sim) -> dict[str, float]:
    m = compute_episode_metrics(sim)
    return {k: float(m.get(k, 0.0)) for k in METRIC_KEYS}


def run_static_episode(cli, eval_seed: int) -> dict[str, float]:
    env = make_track_b_env(**CANONICAL_ENV_KWARGS)
    env.reset(seed=eval_seed)
    action = static_action(cli)
    terminated = truncated = False
    while not (terminated or truncated):
        _o, _r, terminated, truncated, _i = env.step(action)
    row = episode_metrics_row(env.unwrapped.sim)
    env.close()
    return row


def run_ppo_episode(model, vec_norm, eval_seed: int, n_obs: int) -> dict[str, float]:
    env = make_track_b_env(**CANONICAL_ENV_KWARGS)
    obs, _ = env.reset(seed=eval_seed)
    terminated = truncated = False
    while not (terminated or truncated):
        sliced = np.asarray(obs, dtype=np.float32)[:n_obs]
        obs_n = vec_norm.normalize_obs(sliced[None, :])
        action, _ = model.predict(obs_n, deterministic=True)
        obs, _r, terminated, truncated, _i = env.step(np.asarray(action[0], dtype=np.float32))
    row = episode_metrics_row(env.unwrapped.sim)
    env.close()
    return row


def two_way_bootstrap(delta: np.ndarray, iters: int = 10_000, seed: int = 0):
    """delta: (n_ckpt, n_tape) matrix. Resample both axes independently."""
    rng = np.random.default_rng(seed)
    n_c, n_t = delta.shape
    means = np.empty(iters)
    for i in range(iters):
        ci = rng.integers(0, n_c, n_c)
        ti = rng.integers(0, n_t, n_t)
        means[i] = float(delta[np.ix_(ci, ti)].mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    cli = build_parser().parse_args()
    out = cli.output_dir
    out.mkdir(parents=True, exist_ok=True)
    tapes = [cli.eval_seed_base + i for i in range(cli.eval_episodes)]
    ckpts = [(s, cli.ckpt_root_a / "models" / f"seed{s}") for s in cli.seeds_a] + \
            [(s, cli.ckpt_root_b / "models" / f"seed{s}") for s in cli.seeds_b]
    for s, d in ckpts:
        if not (d / "ppo_model.zip").exists():
            raise SystemExit(f"missing checkpoint for seed {s}: {d}")

    rows: list[dict[str, Any]] = []
    # Static comparator once per tape.
    static_by_tape: dict[int, dict[str, float]] = {}
    for t in tapes:
        r = run_static_episode(cli, t)
        static_by_tape[t] = r
        rows.append({"arm": "static", "train_seed": 0, "eval_seed": t,
                     "obs_dim": 0, **r})
    print(f"static done on {len(tapes)} tapes", flush=True)

    ppo_by = {}
    obs_dims: dict[int, int] = {}
    for seed, seed_dir in ckpts:
        model, vec_norm, n_obs = load_policy(seed_dir)
        obs_dims[seed] = n_obs
        for t in tapes:
            r = run_ppo_episode(model, vec_norm, t, n_obs)
            ppo_by[(seed, t)] = r
            rows.append({"arm": "ppo", "train_seed": seed, "eval_seed": t,
                         "obs_dim": n_obs, **r})
        print(f"checkpoint seed {seed} done (obs_dim={n_obs})", flush=True)

    with (out / "crossed_rows.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    seeds = [s for s, _ in ckpts]
    summary: dict[str, Any] = {
        "config": {k: str(v) for k, v in vars(cli).items()},
        "env_kwargs": CANONICAL_ENV_KWARGS,
        "n_checkpoints": len(seeds),
        "n_tapes": len(tapes),
        "obs_dims_by_seed": {str(s): obs_dims[s] for s in seeds},
        "obs_heterogeneity_note": (
            "Seeds 1-5 trained on 48-dim v7 (pre-a3d9ea9), seeds 6-10 on 52-dim "
            "v7; the four appended tail fields are sliced off for the older "
            "checkpoints, reproducing their exact training observation."
        ),
    }
    for key in INFERENCE_KEYS:
        delta = np.array([[ppo_by[(s, t)][key] - static_by_tape[t][key]
                           for t in tapes] for s in seeds])
        per_ckpt = delta.mean(axis=1)
        per_tape = delta.mean(axis=0)
        lo2, hi2 = two_way_bootstrap(delta)
        tci = scipy_stats.t.interval(0.95, len(per_ckpt) - 1,
                                     loc=per_ckpt.mean(), scale=scipy_stats.sem(per_ckpt))
        # Leave-one-out dominance checks.
        loo_ckpt = [float(np.delete(delta, i, axis=0).mean()) for i in range(len(seeds))]
        loo_tape = [float(np.delete(delta, j, axis=1).mean()) for j in range(len(tapes))]
        summary[key] = {
            "mean_delta": float(delta.mean()),
            "two_way_ci95": [lo2, hi2],
            "ckpt_t_ci95": [float(tci[0]), float(tci[1])],
            "per_ckpt_mean": {str(s): float(v) for s, v in zip(seeds, per_ckpt)},
            "ckpts_positive": int((per_ckpt > 0).sum()),
            "tapes_positive": int((per_tape > 0).sum()),
            "min_loo_ckpt_mean": float(min(loo_ckpt)),
            "min_loo_tape_mean": float(min(loo_tape)),
            "ppo_mean": float(np.mean([[ppo_by[(s, t)][key] for t in tapes] for s in seeds])),
            "static_mean": float(np.mean([static_by_tape[t][key] for t in tapes])),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
