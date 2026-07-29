#!/usr/bin/env python3
"""Frozen K3 PPO training and one-shot learner test."""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

from supply_chain.program_g import ret_order_metrics
from supply_chain.replenish import D0, central_cell, materialize_tape
from supply_chain.replenish_ret import (
    BUDGET_D0, LEVELS, WEEKLY_CAP_D0, WEEKS, _budget_feasible_quantity,
    paced_policy, rollout_policy, sS_policy,
)
from supply_chain.supply_chain import OrderRecord

TRAIN = tuple(range(6710001, 6710121))
TEST = tuple(range(6900001, 6900121))
SS = (2.0, 3.0)
MPC = (1.25, 0.0, 1.5)


def obs_array(onhand, pipeline, pending, spent, week, tape):
    backlog = sum(float(order.remaining_qty) for order in pending)
    return np.asarray([
        onhand / D0, sum(pipeline) / D0, backlog / D0,
        (BUDGET_D0 - spent) / BUDGET_D0, (WEEKS - week) / WEEKS,
        float(tape.signal[week] / D0) if week < WEEKS else 0.0,
    ], dtype=np.float32)


class K3Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seeds):
        super().__init__()
        self.seeds = tuple(seeds)
        self.cursor = 0
        self.action_space = spaces.Discrete(len(LEVELS))
        self.observation_space = spaces.Box(0.0, 10.0, shape=(6,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        tape_seed = self.seeds[self.cursor % len(self.seeds)]
        self.cursor += 1
        self.tape = materialize_tape(tape_seed, central_cell(), WEEKS)
        self.week = 0
        self.onhand = D0
        self.pipeline = [0.0]
        self.pending = []
        self.orders = []
        self.spent = 0.0
        return obs_array(self.onhand, self.pipeline, self.pending, self.spent, 0, self.tape), {}

    def step(self, action):
        self.onhand += self.pipeline.pop(0)
        q = _budget_feasible_quantity(
            LEVELS[int(action)], self.spent, self.week, exact_budget=True
        )
        self.spent += q
        self.pipeline.append(q * D0)
        hour = float(self.week * 168)
        demand = float(self.tape.demand[self.week])
        order = OrderRecord(j=self.week + 1, OPTj=hour, quantity=demand, remaining_qty=demand, LTj=168.0)
        self.pending.append(order)
        self.orders.append(order)
        before = sum(float(row.remaining_qty) for row in self.pending)
        for row in self.pending:
            take = min(self.onhand, float(row.remaining_qty))
            self.onhand -= take
            row.remaining_qty -= take
            if row.remaining_qty <= 1e-9 and row.OATj is None:
                row.OATj = hour
                row.CTj = hour - row.OPTj
                row.backorder = row.CTj > row.LTj
        self.pending = [row for row in self.pending if row.remaining_qty > 1e-9]
        after = sum(float(row.remaining_qty) for row in self.pending)
        served = before - after
        reward = served / D0 - 0.25 * after / D0
        self.week += 1
        terminated = self.week >= WEEKS
        if terminated:
            for row in self.pending:
                row.lost = True
                row.lost_time = float(WEEKS * 168)
            reward += float(ret_order_metrics(self.orders)["ret_order"])
            observation = np.zeros(6, dtype=np.float32)
        else:
            observation = obs_array(self.onhand, self.pipeline, self.pending, self.spent, self.week, self.tape)
        return observation, float(reward), terminated, False, {}


def model_policy(model):
    def policy(obs):
        array = np.asarray([
            obs["on_hand_D0"], obs["pipeline_D0"], obs["backlog_D0"],
            obs["remaining_budget_D0"] / BUDGET_D0,
            obs["weeks_remaining"] / WEEKS, obs["forecast_D0"],
        ], dtype=np.float32)
        action, _ = model.predict(array, deterministic=True)
        return LEVELS[int(action)]
    return policy


def ci(values, seed=20260715):
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = rng.choice(array, size=(5000, len(array)), replace=True).mean(axis=1)
    return [float(array.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def main() -> None:
    tapes = [materialize_tape(seed, central_cell(), WEEKS) for seed in TEST]
    ss = [rollout_policy(tape, sS_policy(*SS), exact_budget=True) for tape in tapes]
    mpc = [rollout_policy(tape, paced_policy(*MPC), exact_budget=True) for tape in tapes]
    seed_rows = []
    for learner_seed in range(6):
        model = PPO(
            "MlpPolicy", K3Env(TRAIN), seed=learner_seed, verbose=0,
            learning_rate=3e-4, n_steps=1024, batch_size=256, gamma=0.99,
            policy_kwargs={"net_arch": [64, 64], "activation_fn": __import__("torch").nn.Tanh},
        )
        model.learn(total_timesteps=120_000)
        learned = [
            rollout_policy(tape, model_policy(model), exact_budget=True)
            for tape in tapes
        ]
        vs_ss = [a.ret_order - b.ret_order for a, b in zip(learned, ss)]
        vs_mpc = [a.ret_order - b.ret_order for a, b in zip(learned, mpc)]
        seed_rows.append({
            "seed": learner_seed, "ret_mean": float(np.mean([row.ret_order for row in learned])),
            "vs_sS_ret": ci(vs_ss), "vs_mpc_ret": ci(vs_mpc),
            "vs_mpc_ret_quantity": ci([a.ret_quantity - b.ret_quantity for a, b in zip(learned, mpc)]),
            "vs_mpc_lost": ci([a.lost - b.lost for a, b in zip(learned, mpc)]),
            "vs_mpc_ordered_D0": ci([a.ordered_D0 - b.ordered_D0 for a, b in zip(learned, mpc)]),
            "beats_sS": ci(vs_ss)[1] > 0.0,
            "beats_mpc": ci(vs_mpc)[1] > 0.0,
        })
        print("seed", learner_seed, seed_rows[-1]["vs_sS_ret"], seed_rows[-1]["vs_mpc_ret"], flush=True)
    mpc_wins = sum(row["beats_mpc"] for row in seed_rows)
    output = {
        "contract_id": "program_k3_ret_budgeted_replenishment_v1",
        "stage": "frozen_ppo_virgin_test", "test_seeds": [min(TEST), max(TEST)],
        "ppo_config": {"steps": 120000, "seeds": 6, "net": [64, 64], "gamma": 0.99},
        "comparators": {"sS": SS, "mpc": MPC}, "learner_seeds": seed_rows,
        "seeds_beating_sS": sum(row["beats_sS"] for row in seed_rows),
        "seeds_beating_mpc": mpc_wins,
        "verdict": "CONFIRM_K3_NEURAL_INCREMENTAL_VALUE" if mpc_wins >= 4 else "K3_ADAPTIVE_MPC_CONFIRMED_NO_NEURAL_INCREMENT",
        "paper2_adaptive_confirmed": True,
        "paper3_neural_retention_authorized": bool(mpc_wins >= 4),
    }
    path = Path("results/k3/ppo_virgin.json")
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("seeds_beating_sS", "seeds_beating_mpc", "verdict")}, indent=2))


if __name__ == "__main__":
    main()
