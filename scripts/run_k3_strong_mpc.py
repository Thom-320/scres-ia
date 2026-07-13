#!/usr/bin/env python3
"""Terminal pre-learner K3 gate with exact resource equality and strong controls."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import numpy as np

from supply_chain.replenish import central_cell, materialize_tape
from supply_chain.replenish_ret import (
    BUDGET_D0, WEEKS, paced_policy, periodic_calendars, rollout_actions,
    rollout_policy, sS_policy,
)

CAL = range(6710001, 6710121)
TEST = range(6720001, 6720301)


def ci(values, seed=20260713):
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = rng.choice(array, size=(5000, len(array)), replace=True).mean(axis=1)
    return [
        float(array.mean()), float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    ]


def evaluate_policy(tapes, policy):
    return [rollout_policy(tape, policy, exact_budget=True) for tape in tapes]


def main() -> None:
    cell = central_cell()
    cal = [materialize_tape(seed, cell, WEEKS) for seed in CAL]
    test = [materialize_tape(seed, cell, WEEKS) for seed in TEST]
    calendars = periodic_calendars(4)

    static_cal = [
        mean(
            rollout_actions(tape, sequence, exact_budget=True).ret_order
            for tape in cal
        )
        for sequence in calendars
    ]
    best_static = calendars[int(np.argmax(static_cal))]

    ss_grid = [
        (float(s), float(S))
        for s in np.arange(0.0, 2.01, 0.25)
        for S in np.arange(0.5, 3.01, 0.25)
        if S > s
    ]
    ss_cal = [
        mean(row.ret_order for row in evaluate_policy(cal, sS_policy(s, S)))
        for s, S in ss_grid
    ]
    best_ss = ss_grid[int(np.argmax(ss_cal))]

    inventory_grid = [
        (0.0, float(beta), float(gamma))
        for beta in np.arange(0.0, 1.51, 0.25)
        for gamma in np.arange(0.0, 2.01, 0.5)
    ]
    inventory_cal = [
        mean(row.ret_order for row in evaluate_policy(cal, paced_policy(*params)))
        for params in inventory_grid
    ]
    best_inventory = inventory_grid[int(np.argmax(inventory_cal))]

    mpc_grid = [
        (float(alpha), float(beta), float(gamma))
        for alpha in np.arange(0.0, 1.51, 0.25)
        for beta in np.arange(0.0, 1.51, 0.25)
        for gamma in np.arange(0.0, 2.01, 0.5)
    ]
    mpc_cal = [
        mean(row.ret_order for row in evaluate_policy(cal, paced_policy(*params)))
        for params in mpc_grid
    ]
    best_mpc = mpc_grid[int(np.argmax(mpc_cal))]

    policies = {
        "periodic_static": [
            rollout_actions(tape, best_static, exact_budget=True) for tape in test
        ],
        "budgeted_sS": evaluate_policy(test, sS_policy(*best_ss)),
        "inventory_paced": evaluate_policy(test, paced_policy(*best_inventory)),
        "strong_mpc": evaluate_policy(test, paced_policy(*best_mpc)),
    }
    classical = ("periodic_static", "budgeted_sS", "inventory_paced")
    best_classical = max(
        classical, key=lambda name: mean(row.ret_order for row in policies[name])
    )
    baseline = policies[best_classical]
    candidate = policies["strong_mpc"]
    deltas = {
        "ret_order": [a.ret_order - b.ret_order for a, b in zip(candidate, baseline)],
        "ret_quantity": [a.ret_quantity - b.ret_quantity for a, b in zip(candidate, baseline)],
        "lost": [a.lost - b.lost for a, b in zip(candidate, baseline)],
        "ordered_D0": [a.ordered_D0 - b.ordered_D0 for a, b in zip(candidate, baseline)],
        "remaining_qty": [a.remaining_qty - b.remaining_qty for a, b in zip(candidate, baseline)],
    }
    cis = {key: ci(value) for key, value in deltas.items()}
    gates = {
        "ret_material": cis["ret_order"][0] >= 0.01,
        "ret_lcb_positive": cis["ret_order"][1] > 0.0,
        "ret_quantity_noninferior": cis["ret_quantity"][1] >= 0.0,
        "lost_noninferior": cis["lost"][2] <= 0.0,
        "resource_exact": max(abs(value) for value in deltas["ordered_D0"]) <= 1e-9,
        "nonnegative_tapes": float(np.mean(np.asarray(deltas["ret_order"]) >= 0.0)) >= 0.70,
    }
    summary = {
        name: {
            "ret_order": mean(row.ret_order for row in rows),
            "ret_quantity": mean(row.ret_quantity for row in rows),
            "lost": mean(row.lost for row in rows),
            "remaining_qty": mean(row.remaining_qty for row in rows),
            "ordered_D0": mean(row.ordered_D0 for row in rows),
        }
        for name, rows in policies.items()
    }
    output = {
        "contract_id": "program_k3_ret_budgeted_replenishment_v1",
        "stage": "terminal_prelearner_strong_mpc",
        "seeds": {"calibration": [min(CAL), max(CAL)], "test": [min(TEST), max(TEST)]},
        "selected": {
            "static": best_static, "sS": best_ss,
            "inventory": best_inventory, "mpc": best_mpc,
        },
        "n_periodic_calendars": len(calendars),
        "best_classical": best_classical,
        "policies": summary,
        "candidate_minus_best_classical": cis,
        "positive_tape_fraction": float(np.mean(np.asarray(deltas["ret_order"]) > 0.0)),
        "nonnegative_tape_fraction": float(np.mean(np.asarray(deltas["ret_order"]) >= 0.0)),
        "gates": gates,
        "verdict": "PROMOTE_K3_TO_CONFIRMATION" if all(gates.values()) else "STOP_K3_TERMINAL_NO_RESOURCE_EQUAL_RET_HEADROOM",
        "learner_authorized": False,
        "confirmation_opened": False,
    }
    path = Path("results/k3/strong_mpc_terminal.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected": output["selected"], "best_classical": best_classical,
        "candidate_minus_best_classical": cis, "gates": gates,
        "verdict": output["verdict"],
    }, indent=2))


if __name__ == "__main__":
    main()
