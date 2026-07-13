#!/usr/bin/env python3
"""Single-use K3 confirmation with fully frozen policies."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from supply_chain.replenish import central_cell, materialize_tape
from supply_chain.replenish_ret import WEEKS, paced_policy, rollout_policy, sS_policy

SEEDS = range(6800001, 6800121)
SS = (2.0, 3.0)
MPC = (1.25, 0.0, 1.5)


def ci(values, seed=20260714):
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = rng.choice(array, size=(10000, len(array)), replace=True).mean(axis=1)
    return [float(array.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def main() -> None:
    cell = central_cell()
    tapes = [materialize_tape(seed, cell, WEEKS) for seed in SEEDS]
    baseline = [
        rollout_policy(tape, sS_policy(*SS), exact_budget=True) for tape in tapes
    ]
    candidate = [
        rollout_policy(tape, paced_policy(*MPC), exact_budget=True) for tape in tapes
    ]
    deltas = {
        "ret_order": [a.ret_order - b.ret_order for a, b in zip(candidate, baseline)],
        "ret_quantity": [a.ret_quantity - b.ret_quantity for a, b in zip(candidate, baseline)],
        "lost": [a.lost - b.lost for a, b in zip(candidate, baseline)],
        "remaining_qty": [a.remaining_qty - b.remaining_qty for a, b in zip(candidate, baseline)],
        "ordered_D0": [a.ordered_D0 - b.ordered_D0 for a, b in zip(candidate, baseline)],
    }
    cis = {key: ci(values) for key, values in deltas.items()}
    nonnegative = float(np.mean(np.asarray(deltas["ret_order"]) >= 0.0))
    gates = {
        "ret_material": cis["ret_order"][0] >= 0.01,
        "ret_lcb_positive": cis["ret_order"][1] > 0.0,
        "ret_quantity_noninferior": cis["ret_quantity"][1] >= 0.0,
        "lost_noninferior": cis["lost"][2] <= 0.0,
        "remaining_qty_noninferior": cis["remaining_qty"][2] <= 0.0,
        "resource_exact": max(abs(value) for value in deltas["ordered_D0"]) <= 1e-9,
        "nonnegative_tapes": nonnegative >= 0.70,
    }
    output = {
        "contract_id": "program_k3_ret_budgeted_replenishment_v1",
        "stage": "locked_confirmation",
        "seeds": [min(SEEDS), max(SEEDS)],
        "tape_hashes": [tape.sha for tape in tapes],
        "frozen_policies": {"sS": SS, "mpc": MPC},
        "candidate_minus_sS": cis,
        "positive_tape_fraction": float(np.mean(np.asarray(deltas["ret_order"]) > 0.0)),
        "nonnegative_tape_fraction": nonnegative,
        "gates": gates,
        "verdict": "CONFIRM_K3_OBSERVABLE_RET_HEADROOM" if all(gates.values()) else "STOP_K3_CONFIRMATION_FAILED",
        "learner_authorized": bool(all(gates.values())),
        "learner_seeds_opened": False,
    }
    path = Path("results/k3/confirmation.json")
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("candidate_minus_sS", "nonnegative_tape_fraction", "gates", "verdict")}, indent=2))


if __name__ == "__main__":
    main()
