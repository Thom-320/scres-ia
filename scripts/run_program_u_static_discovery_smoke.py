#!/usr/bin/env python3
"""Burned-seed smoke: can PPO discover a strong static calendar?

Training queries one calendar/tape pair per episode through the Program O
transducer.  It never reads the complete calendar matrix.  Only after training
is complete do we enumerate the 65,536 calendars to obtain a small-case answer
key and report optimization regret.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO  # noqa: E402

from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    extract_full_des_skeleton,
    full_action_calendars,
    simulate_full_des_frontier,
)
from supply_chain.program_u_policy_discovery import StaticCalendarDiscoveryEnv  # noqa: E402


SMOKE_TAPES = (94800001, 94800002, 94800003)


def scheduler() -> dict[str, list[str]]:
    contract = json.loads(
        (ROOT / "contracts/program_o_full_des_hpi_translation_v1.json").read_text()
    )
    key = contract["action"]["primary_scheduler"]
    return contract["action"]["within_week_schedulers"][key]


def deterministic_calendar(model: PPO, env: StaticCalendarDiscoveryEnv) -> tuple[int, ...]:
    observation, _ = env.reset(options={"tape_id": SMOKE_TAPES[0]})
    calendar: list[int] = []
    for _ in range(env.horizon):
        action, _state = model.predict(observation, deterministic=True)
        observation, _reward, terminated, _truncated, _info = env.step(int(action))
        calendar.append(int(action))
        if terminated:
            break
    return tuple(calendar)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--optimizer-seed", type=int, default=20260720)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/program_u/static_discovery_smoke_v1/result.json",
    )
    args = parser.parse_args()
    sched = scheduler()
    skeletons = {}
    for tape in SMOKE_TAPES:
        skeletons[tape], _sim = extract_full_des_skeleton(
            seed=tape,
            scheduler=sched,
            regime_persistence=0.75,
            dominant_share=0.90,
            downstream_freight_physics_mode="fixed_clock_physical_v1",
        )
    training_calls = 0

    def evaluator(calendar: tuple[int, ...], tape_id: int):
        nonlocal training_calls
        training_calls += 1
        panel = simulate_full_des_frontier(
            skeleton=skeletons[int(tape_id)],
            scheduler=sched,
            calendars=np.asarray([calendar], dtype=np.uint8),
        )
        return {key: float(value[0]) for key, value in panel.items()}

    env = StaticCalendarDiscoveryEnv(
        evaluator=evaluator,
        tape_ids=SMOKE_TAPES,
        horizon=8,
        action_count=4,
    )
    model = PPO(
        "MlpPolicy",
        env,
        seed=int(args.optimizer_seed),
        learning_rate=float(args.learning_rate),
        n_steps=256,
        batch_size=64,
        gamma=1.0,
        gae_lambda=1.0,
        ent_coef=float(args.ent_coef),
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=int(args.timesteps), progress_bar=False)
    calls_before_answer_key = int(training_calls)
    learned = deterministic_calendar(model, env)
    # deterministic_calendar performs one final physical evaluation.  It is a
    # candidate evaluation, not part of the answer-key enumeration.
    candidate_scores = []
    for tape in SMOKE_TAPES:
        candidate_scores.append(float(evaluator(learned, tape)["ret_visible"]))
    candidate_mean = float(np.mean(candidate_scores))

    # Answer key is deliberately computed after learner freeze.
    calendars = full_action_calendars()
    score_matrix = np.column_stack(
        [
            simulate_full_des_frontier(
                skeleton=skeletons[tape], scheduler=sched, calendars=calendars
            )["ret_visible"]
            for tape in SMOKE_TAPES
        ]
    )
    mean_scores = score_matrix.mean(axis=1)
    optimum_index = int(np.argmax(mean_scores))
    optimum_calendar = tuple(map(int, calendars[optimum_index]))
    optimum_mean = float(mean_scores[optimum_index])
    rank = int(1 + np.sum(mean_scores > candidate_mean + 1e-15))
    payload = {
        "schema_version": "program_u_static_discovery_smoke_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_status": "BURNED_SMOKE_NOT_SCIENTIFIC_EVIDENCE",
        "tapes": list(SMOKE_TAPES),
        "optimizer_seed": int(args.optimizer_seed),
        "learning_rate": float(args.learning_rate),
        "ent_coef": float(args.ent_coef),
        "timesteps": int(args.timesteps),
        "training_calendar_tape_evaluations_before_answer_key": calls_before_answer_key,
        "search_space_size": int(len(calendars)),
        "learned_calendar": list(learned),
        "learned_mean_ret": candidate_mean,
        "learned_rank_of_65536": rank,
        "optimal_calendar": list(optimum_calendar),
        "optimal_mean_ret": optimum_mean,
        "simple_regret": optimum_mean - candidate_mean,
        "exact_optimum_recovered": learned == optimum_calendar,
        "answer_key_read_during_training": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
