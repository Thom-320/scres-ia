#!/usr/bin/env python3
"""Corrective audit: compare K3 adaptive claims with PPO's fixed 8-week schedule."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from supply_chain.replenish import central_cell, materialize_tape
from supply_chain.replenish_ret import (
    WEEKS, paced_policy, rollout_actions, rollout_policy, sS_policy,
)

FIXED = (1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.0, 0.0)
SS = (2.0, 3.0)
MPC = (1.25, 0.0, 1.5)


def ci(values, seed=20260716):
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = rng.choice(array, size=(10000, len(array)), replace=True).mean(axis=1)
    return [float(array.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def panel(start, end):
    tapes = [materialize_tape(seed, central_cell(), WEEKS) for seed in range(start, end + 1)]
    fixed = [rollout_actions(tape, FIXED, exact_budget=True) for tape in tapes]
    ss = [rollout_policy(tape, sS_policy(*SS), exact_budget=True) for tape in tapes]
    mpc = [rollout_policy(tape, paced_policy(*MPC), exact_budget=True) for tape in tapes]
    return {
        "fixed_ret_mean": float(np.mean([row.ret_order for row in fixed])),
        "sS_ret_mean": float(np.mean([row.ret_order for row in ss])),
        "mpc_ret_mean": float(np.mean([row.ret_order for row in mpc])),
        "fixed_minus_sS_ret": ci([a.ret_order - b.ret_order for a, b in zip(fixed, ss)]),
        "fixed_minus_mpc_ret": ci([a.ret_order - b.ret_order for a, b in zip(fixed, mpc)]),
        "fixed_minus_mpc_ret_quantity": ci([a.ret_quantity - b.ret_quantity for a, b in zip(fixed, mpc)]),
        "fixed_minus_mpc_lost": ci([a.lost - b.lost for a, b in zip(fixed, mpc)]),
        "fixed_minus_mpc_ordered_D0": ci([a.ordered_D0 - b.ordered_D0 for a, b in zip(fixed, mpc)]),
    }


def main() -> None:
    output = {
        "contract_id": "program_k3_ret_budgeted_replenishment_v1",
        "audit": "open_loop_period8_confound",
        "ppo_seed0_unique_test_sequences": 1,
        "ppo_seed0_modal_sequence": FIXED,
        "ppo_seed0_minus_fixed_ret": [0.0, 0.0, 0.0],
        "confirmation_6800001_6800120": panel(6800001, 6800120),
        "learner_test_6900001_6900120": panel(6900001, 6900120),
        "verdict": "RETRACT_K3_ADAPTIVE_AND_NEURAL_CLAIMS_STATIC_PERIOD8_CONFOUND",
        "paper2_adaptive_confirmed": False,
        "paper3_neural_retention_authorized": False,
        "interpretation": "PPO learned one fixed eight-week open-loop schedule. The original static frontier stopped at period four. The fixed schedule matches PPO exactly and beats the tested MPC under identical resources; therefore neither neural incremental value nor observable adaptive headroom is established.",
    }
    path = Path("results/k3/open_loop_confound_audit.json")
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
