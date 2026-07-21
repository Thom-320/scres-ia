#!/usr/bin/env python3
"""Burned dynamic warm-start probe for Program Q2.

The physical environment and canonical terminal ReT remain unchanged.  The
three arms differ only in actor initialization: scratch, behavior cloning of
the best calibration-selected static calendar, or behavior cloning of the
calibration-selected structured controller.  Checkpoints expose retention,
drift, and learned deviations on held-out burned tapes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    extract_full_des_skeleton,
    simulate_full_des_frontier,
)
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS, ProgramORetOnlyEnv  # noqa: E402
from supply_chain.program_o_state_rich import (  # noqa: E402
    finite_state_rich_configurations,
    state_rich_calendar,
)


CALIBRATION = ROOT / "results/program_o/ret_only_learner_v1/calibration_run/result.json"
ARMS = ("scratch", "static_bc", "structured_bc")


def _teacher_episodes(kind: str, payload: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    if kind not in ("static_bc", "structured_bc"):
        return []
    tapes = tuple(range(payload["seed_range"][0], payload["seed_range"][1] + 1))
    env = ProgramORetOnlyEnv(scheduler=scheduler(), tape_seed_start=tapes[0], tape_seed_end=tapes[-1])
    episodes = []
    for cell_index, cell in enumerate(CONFIRMED_RET_CELLS):
        summary = payload["cell_summaries"][cell.cell_id]
        for tape_index, tape in enumerate(tapes):
            calendar = (
                summary["best_open_loop_calendar"]
                if kind == "static_bc"
                else summary["best_classical_calendars"][tape_index]
            )
            observation, _ = env.reset(options={"cell_index": cell_index, "tape_seed": tape})
            observations = []
            actions = []
            for action in calendar:
                observations.append(np.asarray(observation, dtype=np.float32))
                actions.append(int(action))
                observation, _, terminated, truncated, _ = env.step(int(action))
            if not (terminated or truncated):
                raise AssertionError("teacher episode did not terminate")
            episodes.append((np.asarray(observations), np.asarray(actions, dtype=np.int64)))
    return episodes


def behavior_clone(
    model: RecurrentPPO,
    episodes: list[tuple[np.ndarray, np.ndarray]],
    *,
    epochs: int,
    seed: int,
    learning_rate: float = 1e-3,
) -> list[dict]:
    if not episodes:
        return []
    policy = model.policy
    rng = np.random.default_rng(seed)
    history = []
    original_rates = [float(group["lr"]) for group in policy.optimizer.param_groups]
    for group in policy.optimizer.param_groups:
        group["lr"] = float(learning_rate)
    try:
        for epoch in range(epochs):
            losses = []
            correct = 0
            count = 0
            for index in rng.permutation(len(episodes)):
                observations, actions = episodes[int(index)]
                obs = torch.as_tensor(observations, dtype=torch.float32, device=policy.device)
                target = torch.as_tensor(actions, dtype=torch.long, device=policy.device)
                shape = (policy.lstm_actor.num_layers, 1, policy.lstm_actor.hidden_size)
                zero = torch.zeros(shape, device=policy.device)
                states = RNNStates(pi=(zero.clone(), zero.clone()), vf=(zero.clone(), zero.clone()))
                episode_starts = torch.zeros(len(actions), dtype=torch.float32, device=policy.device)
                episode_starts[0] = 1.0
                _, log_prob, _ = policy.evaluate_actions(obs, target, states, episode_starts)
                loss = -log_prob.mean()
                policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                policy.optimizer.step()
                losses.append(float(loss.detach().cpu()))

                with torch.no_grad():
                    actions_hat, _, _, _ = policy.forward(obs, states, episode_starts, deterministic=True)
                correct += int((actions_hat.reshape(-1).long() == target).sum().cpu())
                count += len(actions)
            accuracy = correct / count
            history.append({"epoch": epoch + 1, "mean_nll": float(np.mean(losses)), "accuracy": float(accuracy)})
            if accuracy >= 0.995:
                break
    finally:
        for group, rate in zip(policy.optimizer.param_groups, original_rates):
            group["lr"] = rate
    return history


def _model(env: ProgramORetOnlyEnv, seed: int) -> RecurrentPPO:
    return RecurrentPPO(
        "MlpLstmPolicy",
        env,
        seed=seed,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        normalize_advantage=True,
        policy_kwargs={"lstm_hidden_size": 64, "net_arch": [64, 64]},
        verbose=0,
        device="cpu",
    )


def _selected_structured_calendar(skeleton, cell_id: str, calibration: dict) -> tuple[int, ...]:
    config_id = calibration["cell_summaries"][cell_id]["best_classical_config"]
    configs = {row.config_id: row for row in finite_state_rich_configurations()}
    calendar, _ = state_rich_calendar(
        skeleton=skeleton.as_dict(),
        scheduler=scheduler(),
        config=configs[config_id],
        regime_persistence=0.75,
        dominant_share=0.90,
    )
    return tuple(map(int, calendar))


def evaluate(model: RecurrentPPO, tapes: tuple[int, ...], calibration: dict) -> list[dict]:
    sched = scheduler()
    rows = []
    for cell_index, cell in enumerate(CONFIRMED_RET_CELLS):
        static_calendar = tuple(map(int, calibration["cell_summaries"][cell.cell_id]["best_open_loop_calendar"]))
        for tape in tapes:
            skeleton, _ = extract_full_des_skeleton(
                seed=tape,
                scheduler=sched,
                regime_persistence=cell.regime_persistence,
                dominant_share=cell.dominant_share,
                downstream_freight_physics_mode="fixed_clock_physical_v1",
            )
            env = ProgramORetOnlyEnv(scheduler=sched, tape_seed_start=tape, tape_seed_end=tape)
            observation, _ = env.reset(options={"cell_index": cell_index, "tape_seed": tape, "skeleton": skeleton})
            state = None
            episode_start = np.ones((1,), dtype=bool)
            calendar = []
            while True:
                action, state = model.predict(observation, state=state, episode_start=episode_start, deterministic=True)
                value = int(np.asarray(action).reshape(-1)[0])
                calendar.append(value)
                observation, _, terminated, truncated, info = env.step(value)
                episode_start[:] = False
                if terminated or truncated:
                    break
            structured_calendar = _selected_structured_calendar(skeleton, cell.cell_id, calibration)
            comparison = simulate_full_des_frontier(
                skeleton=skeleton,
                scheduler=sched,
                calendars=np.asarray([calendar, static_calendar, structured_calendar], dtype=np.uint8),
            )
            rows.append({
                "cell": cell.cell_id,
                "tape": tape,
                "calendar": calendar,
                "static_calendar": list(static_calendar),
                "structured_calendar": list(structured_calendar),
                "ret_visible": float(comparison["ret_visible"][0]),
                "delta_vs_static": float(comparison["ret_visible"][0] - comparison["ret_visible"][1]),
                "delta_vs_structured": float(comparison["ret_visible"][0] - comparison["ret_visible"][2]),
                "worst_product_delta_vs_structured": float(comparison["worst_product_fill"][0] - comparison["worst_product_fill"][2]),
                "lost_orders": float(comparison["lost_orders"][0]),
                "resource_spread": float(max(comparison["gross_policy_batch_slots"]) - min(comparison["gross_policy_batch_slots"])),
            })
    return rows


def _run(payload: tuple[str, int, int, int, tuple[int, ...], Path, int, tuple[int, ...]]) -> dict:
    arm, optimizer_seed, train_low, train_high, evaluation_tapes, calibration_path, bc_epochs, checkpoints = payload
    torch.set_num_threads(1)
    calibration = json.loads(calibration_path.read_text())
    env = ProgramORetOnlyEnv(scheduler=scheduler(), tape_seed_start=train_low, tape_seed_end=train_high)
    model = _model(env, optimizer_seed)
    bc_history = behavior_clone(model, _teacher_episodes(arm, calibration), epochs=bc_epochs, seed=optimizer_seed)
    checkpoint_rows = []
    previous = 0
    for checkpoint in checkpoints:
        if checkpoint > previous:
            model.learn(total_timesteps=checkpoint - previous, reset_num_timesteps=False, progress_bar=False)
        rows = evaluate(model, evaluation_tapes, calibration)
        checkpoint_rows.append({
            "timesteps": checkpoint,
            "mean_delta_vs_static": float(np.mean([row["delta_vs_static"] for row in rows])),
            "mean_delta_vs_structured": float(np.mean([row["delta_vs_structured"] for row in rows])),
            "favorable_vs_structured": float(np.mean([row["delta_vs_structured"] > 0.0 for row in rows])),
            "worst_product_mean_delta": float(np.mean([row["worst_product_delta_vs_structured"] for row in rows])),
            "lost_orders_max": float(max(row["lost_orders"] for row in rows)),
            "resource_spread_max": float(max(row["resource_spread"] for row in rows)),
            "rows": rows,
        })
        previous = checkpoint
    return {"arm": arm, "optimizer_seed": optimizer_seed, "training_seed_range": [train_low, train_high], "bc_history": bc_history, "checkpoints": checkpoint_rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--optimizer-seeds", default="20261101,20261102,20261103")
    parser.add_argument("--training-seed-base", type=int, default=757100001)
    parser.add_argument("--evaluation-tapes", default="7490001-7490024")
    parser.add_argument("--bc-epochs", type=int, default=100)
    parser.add_argument("--checkpoints", default="0,10000,30000,60000")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    optimizer_seeds = tuple(map(int, args.optimizer_seeds.split(",")))
    start, end = map(int, args.evaluation_tapes.split("-"))
    evaluation_tapes = tuple(range(start, end + 1))
    checkpoints = tuple(map(int, args.checkpoints.split(",")))
    if not checkpoints or checkpoints[0] != 0 or tuple(sorted(set(checkpoints))) != checkpoints:
        raise ValueError("checkpoints must be sorted, unique, and start at zero")
    episodes_per_seed = int(np.ceil(max(checkpoints) / 8)) + 100
    tasks = []
    for seed_index, optimizer_seed in enumerate(optimizer_seeds):
        low = args.training_seed_base + seed_index * 10_000
        high = low + episodes_per_seed - 1
        for arm in ARMS:
            tasks.append((arm, optimizer_seed, low, high, evaluation_tapes, args.calibration, args.bc_epochs, checkpoints))
    started = time.perf_counter()
    if args.jobs == 1:
        runs = [_run(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(tasks))) as executor:
            runs = list(executor.map(_run, tasks))
    summaries = []
    for arm in ARMS:
        for checkpoint in checkpoints:
            members = [next(row for row in run["checkpoints"] if row["timesteps"] == checkpoint) for run in runs if run["arm"] == arm]
            summaries.append({
                "arm": arm,
                "timesteps": checkpoint,
                "mean_delta_vs_static": float(np.mean([row["mean_delta_vs_static"] for row in members])),
                "mean_delta_vs_structured": float(np.mean([row["mean_delta_vs_structured"] for row in members])),
                "positive_seed_fraction_vs_structured": float(np.mean([row["mean_delta_vs_structured"] > 0.0 for row in members])),
                "favorable_vs_structured": float(np.mean([row["favorable_vs_structured"] for row in members])),
                "worst_product_mean_delta": float(np.mean([row["worst_product_mean_delta"] for row in members])),
            })
    payload = {
        "schema_version": "program_q2_dynamic_warmstart_probe_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_status": "BURNED_DEVELOPMENT_NO_CLAIM",
        "reward": "raw_terminal_ret_visible",
        "normalize_advantage": True,
        "arms": list(ARMS),
        "optimizer_seeds": list(optimizer_seeds),
        "evaluation_tapes": list(evaluation_tapes),
        "runs": runs,
        "summaries": summaries,
        "elapsed_seconds": float(time.perf_counter() - started),
        "verdict": "EXPLORATORY_WARMSTART_COMPLETE",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "runs"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
