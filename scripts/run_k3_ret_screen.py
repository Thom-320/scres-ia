#!/usr/bin/env python3
"""K3 development screen: canonical ReT, equal budget, no learner."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import numpy as np

from supply_chain.replenish import central_cell, materialize_tape
from supply_chain.replenish_ret import (
    WEEKS, paced_policy, periodic_calendars, rollout_actions, rollout_policy,
    sS_policy,
)

CAL = range(6700001, 6700061)
TEST = range(6700101, 6700221)


def ci(values, seed=20260712):
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = rng.choice(a, size=(4000, len(a)), replace=True).mean(axis=1)
    return [float(a.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def main() -> None:
    cell = central_cell()
    cal = [materialize_tape(seed, cell, WEEKS) for seed in CAL]
    test = [materialize_tape(seed, cell, WEEKS) for seed in TEST]
    calendars = periodic_calendars(4)

    static_means = [mean(rollout_actions(tape, seq).ret_order for tape in cal) for seq in calendars]
    best_static = calendars[int(np.argmax(static_means))]

    sS_grid = [
        (s, S) for s in np.arange(0.0, 2.01, 0.25)
        for S in np.arange(0.5, 3.01, 0.25) if S > s
    ]
    sS_means = [mean(rollout_policy(tape, sS_policy(s, S)).ret_order for tape in cal) for s, S in sS_grid]
    best_sS = sS_grid[int(np.argmax(sS_means))]

    inventory_grid = [(0.0, beta) for beta in np.arange(0.0, 1.51, 0.25)]
    inventory_means = [
        mean(rollout_policy(tape, paced_policy(alpha, beta)).ret_order for tape in cal)
        for alpha, beta in inventory_grid
    ]
    best_inventory = inventory_grid[int(np.argmax(inventory_means))]

    mpc_grid = [
        (alpha, beta) for alpha in np.arange(0.25, 1.51, 0.25)
        for beta in np.arange(0.0, 1.51, 0.25)
    ]
    mpc_means = [
        mean(rollout_policy(tape, paced_policy(alpha, beta)).ret_order for tape in cal)
        for alpha, beta in mpc_grid
    ]
    best_mpc = mpc_grid[int(np.argmax(mpc_means))]

    policies = {
        "periodic_static": lambda tape: rollout_actions(tape, best_static),
        "budgeted_sS": lambda tape: rollout_policy(tape, sS_policy(*best_sS)),
        "inventory_paced": lambda tape: rollout_policy(tape, paced_policy(*best_inventory)),
        "signal_inventory_mpc": lambda tape: rollout_policy(tape, paced_policy(*best_mpc)),
    }
    rows = {name: [fn(tape) for tape in test] for name, fn in policies.items()}
    classical_names = ("periodic_static", "budgeted_sS", "inventory_paced")
    best_classical = max(classical_names, key=lambda name: mean(row.ret_order for row in rows[name]))
    baseline = rows[best_classical]
    candidate = rows["signal_inventory_mpc"]
    ret_delta = [a.ret_order - b.ret_order for a, b in zip(candidate, baseline)]
    qty_delta = [a.ret_quantity - b.ret_quantity for a, b in zip(candidate, baseline)]
    lost_delta = [a.lost - b.lost for a, b in zip(candidate, baseline)]
    resource_delta = [a.ordered_D0 - b.ordered_D0 for a, b in zip(candidate, baseline)]

    # Hindsight over the full periodic frontier is a restricted PI diagnostic,
    # not an upper bound over every possible open-loop sequence.
    restricted_pi = []
    for index, tape in enumerate(test):
        static_value = rows["periodic_static"][index].ret_order
        best_tape = max(rollout_actions(tape, seq).ret_order for seq in calendars)
        restricted_pi.append(best_tape - static_value)

    summary = {}
    for name, policy_rows in rows.items():
        summary[name] = {
            "ret_order_mean": mean(row.ret_order for row in policy_rows),
            "ret_quantity_mean": mean(row.ret_quantity for row in policy_rows),
            "lost_mean": mean(row.lost for row in policy_rows),
            "remaining_qty_mean": mean(row.remaining_qty for row in policy_rows),
            "ordered_D0_mean": mean(row.ordered_D0 for row in policy_rows),
        }
    gates = {
        "ret_delta_min": ci(ret_delta)[0] >= 0.01,
        "ret_lcb_positive": ci(ret_delta)[1] > 0.0,
        "ret_quantity_noninferior": ci(qty_delta)[1] >= 0.0,
        "lost_noninferior": ci(lost_delta)[2] <= 0.0,
        "resource_noninferior": ci(resource_delta)[2] <= 1e-9,
    }
    output = {
        "contract_id": "program_k3_ret_budgeted_replenishment_v1",
        "stage": "development_screen_no_learner",
        "cell": cell,
        "seeds": {"calibration": [min(CAL), max(CAL)], "test": [min(TEST), max(TEST)]},
        "tape_hashes": {"calibration": [tape.sha for tape in cal], "test": [tape.sha for tape in test]},
        "n_periodic_calendars": len(calendars),
        "selected": {"static": best_static, "sS": best_sS, "inventory": best_inventory, "mpc": best_mpc},
        "best_classical": best_classical,
        "policies": summary,
        "restricted_pi_ret_delta": ci(restricted_pi),
        "candidate_minus_best_classical": {
            "ret_order": ci(ret_delta), "ret_quantity": ci(qty_delta),
            "lost_orders": ci(lost_delta), "ordered_D0": ci(resource_delta),
            "positive_tape_fraction": float(np.mean(np.asarray(ret_delta) > 0.0)),
            "nonnegative_tape_fraction": float(np.mean(np.asarray(ret_delta) >= 0.0)),
        },
        "gates": gates,
        "verdict": "PROMOTE_K3_TO_FRESH_CONFIRMATION" if all(gates.values()) else "STOP_K3_NO_RESOURCE_EQUAL_RET_HEADROOM",
        "learner_authorized": False,
        "confirmation_seeds_opened": False,
    }
    path = Path("results/k3/development_screen.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("n_periodic_calendars", "best_classical", "selected", "candidate_minus_best_classical", "gates", "verdict")}, indent=2))


if __name__ == "__main__":
    main()
