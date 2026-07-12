#!/usr/bin/env python3
"""Fail-closed Paper 2 physics/CRN/observation preflight."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from supply_chain.maintenance_control import (
    ACTIONS, FORBIDDEN_OBSERVATIONS, OBSERVATION_KEYS, make_sim,
    materialize_tape, periodic_policy, run_policy,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1200001)
    parser.add_argument("--weeks", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    tape = materialize_tape(args.seed, weeks=args.weeks)
    sim, controller, _ = make_sim(tape)
    obs = controller.observation()
    rows = {action: run_policy(tape, periodic_policy((action,))) for action in ACTIONS}
    verdict = {
        "contract_id": "paper2_maintenance_control_v1",
        "seed": args.seed,
        "checks": {
            "observation_whitelist": tuple(obs) == OBSERVATION_KEYS,
            "no_privileged_observations": not set(obs).intersection(FORBIDDEN_OBSERVATIONS),
            "mass_conservation": max(row["mass_residual"] for row in rows.values()) <= 1e-5,
            "equal_scheduled_pm_hours": len({row["scheduled_pm_hours"] for row in rows.values()}) == 1,
            "base_exogenous_crn": len({row["base_exogenous_sha256"] for row in rows.values()}) == 1,
            "consumed_wear_crn": len({row["consumed_wear_sha256"] for row in rows.values()}) == 1,
            "all_actions_recorded": all(row["action_events"] for row in rows.values()),
            "finite_wip_live": any(sum(row["blocked_hours"].values()) + sum(row["starved_hours"].values()) > 0 for row in rows.values()),
        },
        "rows": rows,
    }
    verdict["verdict"] = "PASS_PAPER2_MAINTENANCE_PREFLIGHT" if all(verdict["checks"].values()) else "FAIL_PAPER2_MAINTENANCE_PREFLIGHT"
    encoded = json.dumps(verdict, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if verdict["verdict"].startswith("FAIL"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
