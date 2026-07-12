#!/usr/bin/env python3
"""Execute Program I prefix-balanced exact-replay branching (no learner)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from supply_chain.config import HOURS_PER_WEEK, INVENTORY_BUFFERS, THESIS_FAITHFUL_PROTOCOL as P
from supply_chain.decision_right_discovery import adaptive_headroom_verdict
from supply_chain.episode_metrics import compute_episode_metrics
from supply_chain.supply_chain import MFSCSimulation


def load_contract(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def new_sim(seed: int, horizon_weeks: int) -> MFSCSimulation:
    return MFSCSimulation(
        shifts=1, initial_buffers=dict(INVENTORY_BUFFERS[168]), seed=seed,
        horizon=horizon_weeks * HOURS_PER_WEEK, risks_enabled=True,
        risk_level="current", strict_exogenous_crn=True,
        year_basis=P["year_basis"], warmup_trigger=P["warmup_trigger"],
        r14_defect_mode=P["r14_defect_mode"],
    )


def replay(
    *, seed: int, decision_week: int, horizon_weeks: int,
    prefix: dict[str, float], branch: dict[str, float], family: str,
) -> dict[str, Any]:
    sim = new_sim(seed, 52)
    _obs, _reward, _done, _info = sim.step(prefix, decision_week * HOURS_PER_WEEK)
    before_obs = sim.get_observation().astype(float).tolist()
    before_flow = sim.flow_ledger()
    before_orders = len(sim.orders)
    _obs, _reward, _done, _info = sim.step(branch, horizon_weeks * HOURS_PER_WEEK)
    metrics = compute_episode_metrics(sim)
    if family == "production":
        resource = (int(branch["assembly_shifts"]) - 1) * horizon_weeks * HOURS_PER_WEEK
    elif family == "dispatch":
        resource = horizon_weeks * HOURS_PER_WEEK / float(branch["op9_rop"])
    else:
        resource = horizon_weeks * HOURS_PER_WEEK * (
            1.0 / float(branch["op10_rop"]) + 1.0 / float(branch["op12_rop"])
        )
    return {
        "before_observation": before_obs, "before_flow": before_flow,
        "before_orders": before_orders, "ret_excel": float(metrics["ret_excel"]),
        "service_loss": float(metrics["service_loss_auc_ration_hours"]),
        "lost_orders": float(metrics["lost_orders"]), "resource": float(resource),
        "mass_residual": max(abs(float(sim.flow_ledger()["raw_residual"])), abs(float(sim.flow_ledger()["ration_residual"]))),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    serial = []
    for row in rows:
        serial.append({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serial[0]))
        writer.writeheader()
        writer.writerows(serial)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path("contracts/program_i_branching_v1.json"))
    parser.add_argument("--tapes", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/program_i/branching"))
    args = parser.parse_args()
    contract = load_contract(args.contract)
    n_tapes = args.tapes or int(contract["tapes"]["count"])
    if n_tapes > int(contract["tapes"]["count"]):
        raise SystemExit("Cannot exceed preregistered tape count")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for family, spec in contract["families"].items():
        actions = list(spec["actions"].items())
        for tape_index in range(n_tapes):
            seed = int(contract["tapes"]["seed_start"]) + tape_index
            for state_index, decision_week in enumerate(contract["states"]["decision_weeks"]):
                prefix_name, prefix = actions[(tape_index + state_index) % len(actions)]
                expected_obs = None
                expected_flow = None
                expected_orders = None
                state_id = f"{family}|{seed}|w{decision_week}|p{prefix_name}"
                for horizon_weeks in contract["horizons_weeks"]:
                    for action_name, action in actions:
                        result = replay(seed=seed, decision_week=decision_week, horizon_weeks=horizon_weeks, prefix=prefix, branch=action, family=family)
                        if expected_obs is None:
                            expected_obs, expected_flow, expected_orders = result["before_observation"], result["before_flow"], result["before_orders"]
                        else:
                            if result["before_observation"] != expected_obs or result["before_flow"] != expected_flow or result["before_orders"] != expected_orders:
                                raise AssertionError(f"Prefix replay identity failed for {state_id}")
                        rows.append({"family":family,"state_id":state_id,"tape_id":str(seed),"decision_week":decision_week,"prefix":prefix_name,"horizon_weeks":horizon_weeks,"action":action_name,**{k:v for k,v in result.items() if not k.startswith("before_")}})
            print(f"[program-i-branch] {family} tape {tape_index + 1}/{n_tapes}", flush=True)

    verdicts = {}
    for family, spec in contract["families"].items():
        family_rows = [row for row in rows if row["family"] == family]
        horizon_verdicts = {}
        winners = {}
        for horizon in contract["horizons_weeks"]:
            selected = [row for row in family_rows if row["horizon_weeks"] == horizon]
            values = {state: {row["action"]: row["ret_excel"] for row in selected if row["state_id"] == state} for state in sorted({row["state_id"] for row in selected})}
            tapes = {row["state_id"]: row["tape_id"] for row in selected}
            by_state_action = {(row["state_id"], row["action"]): row for row in selected}
            action_names = sorted({row["action"] for row in selected})
            constant_means = {
                action: float(np.mean([values[state][action] for state in values]))
                for action in action_names
            }
            best_constant = max(action_names, key=lambda action: (constant_means[action], action))
            service_base = 0.0
            service_oracle = 0.0
            lost_delta = []
            for state, action_values in values.items():
                oracle_action = max(action_names, key=lambda action: (action_values[action], action))
                base_row = by_state_action[(state, best_constant)]
                oracle_row = by_state_action[(state, oracle_action)]
                service_base += float(base_row["service_loss"])
                service_oracle += float(oracle_row["service_loss"])
                lost_delta.append(float(oracle_row["lost_orders"]) - float(base_row["lost_orders"]))
            service_reduction = (service_base - service_oracle) / max(abs(service_base), 1.0)
            resource_equal = bool(spec["resource_equal_by_construction"]) and all(len({round(row["resource"], 9) for row in selected if row["state_id"] == state}) == 1 for state in values)
            local = adaptive_headroom_verdict(values, tapes, observable_deltas=None, service_reduction=service_reduction, resource_equivalent=resource_equal, thresholds=contract["gates"])
            local["service_loss_reduction"] = float(service_reduction)
            local["lost_orders_delta_mean"] = float(np.mean(lost_delta))
            local["gates"]["lost_orders_nonincrease"] = local["lost_orders_delta_mean"] <= 0.0
            local["promote_to_rl"] = False
            local["max_mass_residual"] = max(row["mass_residual"] for row in selected)
            horizon_verdicts[str(horizon)] = local
            winners[horizon] = {state: max(values[state], key=lambda action: (values[state][action], action)) for state in values}
        common = sorted(set(winners[4]) & set(winners[8]))
        agreement = float(np.mean([winners[4][state] == winners[8][state] for state in common]))
        verdicts[family] = {
            "horizons": horizon_verdicts, "horizon_winner_agreement": agreement,
            "horizon_stable": agreement >= float(contract["gates"]["horizon_ranking_agreement_min"]),
            "promote_to_observable_policy": False,
            "interpretation": "LOCAL_HEADROOM_REQUIRES_OBSERVABLE_ROLLOUT" if horizon_verdicts["8"]["gates"]["ranking_diversity"] and horizon_verdicts["8"]["gates"]["oracle_headroom"] and horizon_verdicts["8"]["gates"]["resource_equivalence"] else "STOP_NO_RESOURCE_ADJUSTED_LOCAL_HEADROOM"
        }
    final = {"contract_id":contract["contract_id"],"n_tapes":n_tapes,"n_rows":len(rows),"families":verdicts,"confirmation_opened":False,"rl_trained":False}
    write_rows(args.output_dir / "branch_rows.csv", rows)
    (args.output_dir / "verdict.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
