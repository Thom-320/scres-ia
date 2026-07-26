#!/usr/bin/env python3
"""Stage 2 pilot: training progress on the clairvoyant-headroom diagnostic.

Trains a learner on campaigns built from a FRESH root block (7650001+) and, at fixed
training-time checkpoints, rolls the deterministic policy out on the 48 burned campaigns
whose exact clairvoyant ceiling is known by enumeration. Each checkpoint yields the
fraction of the clairvoyant headroom captured, so the curve answers "what percentage of
the theoretical maximum does the model reach, and when does it pass the best static
policy" without any estimated ceiling.

Train and evaluation campaigns are disjoint by construction (different history roots) and
are built by the same `rebuild_campaign` path, so the distributions match.

The learner is rewarded on `early_ret_complete_cohort` -- the SAME scalar the ceiling and
the MPC arms are graded on. The frozen env rewards `ret_visible`, so the objective is
recomputed by a thin wrapper rather than by editing the frozen environment.

Usage:
    .venv/bin/python scripts/run_oracle_learning_curve.py --arch recurrent_ppo --seeds 5
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.c6_perbatch_ceiling import OBJECTIVE, SCHED_PATTERN, rebuild_campaign  # noqa: E402
from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.oracle_capture import (  # noqa: E402
    BOOT_SEED,
    best_static_calendar,
    load_campaigns,
    pooled_capture,
)
from supply_chain.program_o_full_des_transducer import simulate_full_des_frontier  # noqa: E402
from supply_chain.program_o_ret_env import ProgramORetOnlyEnv  # noqa: E402

FRONTIERS = ROOT / "results/fig5_surrogate_v1/frontiers"
OUT = ROOT / "results/oracle_capture_v1"
TRAIN_ROOT_START, TRAIN_ROOT_END = 7_650_001, 7_650_040  # verified unused block
EVAL_CELLS = ((0.75, range(1, 6)), (0.90, range(6, 12)))  # as in the burned eval set


class OracleObjectiveReward(gym.Wrapper):
    """Reward the graded objective instead of the frozen env's ret_visible."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            skeleton = self.env.unwrapped._skeleton
            metrics = simulate_full_des_frontier(
                skeleton=skeleton, scheduler=SCHED_PATTERN,
                calendars=np.asarray([info["calendar"]], dtype=np.uint8),
                include_q_r1_metrics=True)
            reward = float(np.asarray(metrics[OBJECTIVE])[0])
            info["objective"] = reward
        return obs, reward, terminated, truncated, info


def training_skeletons() -> list:
    """Campaigns from the fresh root block, same (kappa, index) shape as the eval set."""
    out = []
    for root in range(TRAIN_ROOT_START, TRAIN_ROOT_END + 1):
        for kappa, indices in EVAL_CELLS:
            for index in indices:
                out.append((root, kappa, index))
    return out


class TrainingEnv(ProgramORetOnlyEnv):
    """Cycles deterministically through the training campaign list."""

    def __init__(self, *, specs, rng_seed: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._specs = list(specs)
        self._rng = np.random.default_rng(rng_seed)
        self._cache: dict[tuple, object] = {}

    def reset(self, *, seed=None, options=None):
        spec = self._specs[int(self._rng.integers(len(self._specs)))]
        if spec not in self._cache:
            self._cache[spec] = rebuild_campaign(*spec).skeleton
        options = dict(options or {})
        options["skeleton"] = self._cache[spec]
        # the skeleton is injected, so tape_seed only has to be a legal value: pass one
        # explicitly or the parent's namespace counter aborts after the first episode
        options.setdefault("tape_seed", self.tape_seed_start)
        options.setdefault("cell_index", 0 if spec[1] == 0.75 else 2)
        return super().reset(seed=seed, options=options)


def rollout_calendar(model, env, skeleton, cell_index: int) -> list[int]:
    """Deterministic greedy rollout on one campaign -> its 8-week calendar."""
    obs, _ = env.reset(options={"skeleton": skeleton, "cell_index": cell_index,
                                "tape_seed": env.tape_seed_start})
    state, done = None, False
    calendar: list[int] = []
    while not done:
        action, state = model.predict(obs, state=state, episode_start=np.array([not calendar]),
                                     deterministic=True)
        obs, _r, done, _t, info = env.step(int(action))
        calendar = info.get("calendar", calendar) if done else calendar + [int(action)]
    return list(calendar)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("recurrent_ppo", "ppo_mlp"), default="recurrent_ppo")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=7_650_101)
    parser.add_argument("--total-timesteps", type=int, default=48_000)
    parser.add_argument("--eval-every", type=int, default=3_000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out_path = args.output or OUT / f"learning_curve_{args.arch}.json"
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite {out_path}")

    from sb3_contrib import RecurrentPPO  # noqa: PLC0415
    from stable_baselines3 import PPO  # noqa: PLC0415

    campaigns = load_campaigns(FRONTIERS)
    static_row, _ = best_static_calendar(campaigns)
    bar = {c.key: float(c.labels[static_row]) for c in campaigns}
    ceil_mean = float(np.mean([c.ceiling for c in campaigns]))

    sched = scheduler()
    eval_env = ProgramORetOnlyEnv(scheduler=sched, tape_seed_start=1, tape_seed_end=1)
    cell_of = {"binary_0.75": 0, "binary_0.9": 2}

    eval_skeletons = {c.key: rebuild_campaign(
        c.history_root, c.kappa, c.campaign_index).skeleton for c in campaigns}

    def grade(model) -> dict:
        calendars = {}
        for c in campaigns:
            calendars[c.key] = rollout_calendar(
                model, eval_env, eval_skeletons[c.key], cell_of[c.persistence_mode])
        pooled = pooled_capture(campaigns, calendars, bar, np.random.default_rng(BOOT_SEED))
        values = [c.value_of(calendars[c.key]) for c in campaigns]
        return {
            **pooled,
            "mean_label": float(np.mean(values)),
            "exact_optimum_hits": int(sum(
                1 for c in campaigns
                if c.ceiling - c.value_of(calendars[c.key]) <= 1e-9)),
            "distinct_calendars": len({tuple(v) for v in calendars.values()}),
        }

    specs = training_skeletons()
    curves = []
    started = time.perf_counter()
    for offset in range(args.seeds):
        seed = args.seed_start + offset
        env = OracleObjectiveReward(TrainingEnv(
            specs=specs, rng_seed=seed, scheduler=sched,
            tape_seed_start=1, tape_seed_end=1))
        if args.arch == "recurrent_ppo":
            model = RecurrentPPO("MlpLstmPolicy", env, seed=seed, n_steps=64,
                                 batch_size=64, verbose=0)
        else:
            model = PPO("MlpPolicy", env, seed=seed, n_steps=64, batch_size=64, verbose=0)

        points = [{"timesteps": 0, **grade(model)}]
        print(f"[{args.arch} seed {seed}] t=0 capture "
              f"{points[0]['pooled_ratio']:+.4f} hits {points[0]['exact_optimum_hits']}",
              flush=True)
        done_steps = 0
        while done_steps < args.total_timesteps:
            chunk = min(args.eval_every, args.total_timesteps - done_steps)
            model.learn(total_timesteps=chunk, reset_num_timesteps=False,
                        progress_bar=False)
            done_steps += chunk
            point = {"timesteps": done_steps, **grade(model)}
            points.append(point)
            print(f"[{args.arch} seed {seed}] t={done_steps} capture "
                  f"{point['pooled_ratio']:+.4f} (LCB {point['lcb95']:+.4f}) "
                  f"hits {point['exact_optimum_hits']} "
                  f"cal {point['distinct_calendars']} "
                  f"({time.perf_counter()-started:.0f}s)", flush=True)
        curves.append({"seed": seed, "points": points})

    payload = {
        "schema": "oracle_learning_curve_v1",
        "display_name": "training_progress_pilot_unmatched_retention_rights",
        "claim_status": "BURNED_DEVELOPMENT_NO_CLAIM_METHODOLOGICAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": args.arch,
        "objective": OBJECTIVE,
        "metric": "pooled capture = sum(V - best_static) / sum(ceiling - best_static) "
                  "over the 48 burned campaigns; ceiling exact by 4^8 enumeration",
        "interpretation_boundary": {
            "purpose": "instrument and training-progress pilot only",
            "retention_rights_matched_to_retained_mpc": False,
            "learner_initial_belief_c": 0.5,
            "recurrent_state_crosses_campaign_boundaries": False,
            "architecture_ranking_authorized": False,
            "confirmatory_learning_claim_authorized": False,
        },
        "training": {
            "root_block": [TRAIN_ROOT_START, TRAIN_ROOT_END],
            "n_training_campaigns": len(specs),
            "disjoint_from_evaluation": True,
            "total_timesteps": args.total_timesteps,
            "eval_every": args.eval_every,
            "seeds": [args.seed_start + i for i in range(args.seeds)],
        },
        "reference_levels": {
            "ceiling_mean_label": ceil_mean,
            "best_static_capture": 0.0,
            "best_static_calendar": [(static_row // 4 ** (7 - w)) % 4 for w in range(8)],
        },
        "curves": curves,
        "elapsed_seconds": time.perf_counter() - started,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
