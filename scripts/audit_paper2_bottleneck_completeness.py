#!/usr/bin/env python3
"""Retrospective completeness audit of the already-burned bottleneck tapes.

This is a corrective audit, not a preregistered confirmatory analysis.  It
quantifies missing comparator/resource/trajectory work without opening the
unopened learner seed range.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.paper2_bottleneck import (  # noqa: E402
    ACTIONS,
    ACTION_NAMES,
    CONTEXTS,
    materialize_tape,
    run_policy,
    signal_policy,
)


ROOT = Path(__file__).resolve().parent.parent
CAL_START, N_CAL = 1_100_001, 60
LOCKED_START, N_LOCKED = 1_110_001, 120
WEEKS = 24
ACTION_FOR_CONTEXT = dict(zip(CONTEXTS, ACTIONS))
METRIC_KEYS = (
    "ret_excel",
    "ration_ret_excel",
    "ret_excel_cvar05",
    "service_loss_auc_ration_hours",
    "n_lost",
    "consumed_base_threat_sha256",
    "realized_demand_sha256",
)


def constant(action):
    return lambda observation: action


def tapes(start: int, n: int, split: str):
    return [
        materialize_tape(start + i, CONTEXTS[i % len(CONTEXTS)], split, weeks=WEEKS)
        for i in range(n)
    ]


def week_from_observation(observation: dict[str, float]) -> int:
    return int(round(float(observation["week_phase"]) * (WEEKS - 1)))


def active_sequence(row: dict) -> tuple[str, ...]:
    return tuple(ACTION_NAMES[tuple(event["action"])] for event in row["action_events"])


def calendar_policy(active_actions: tuple[tuple[int, int, int], ...]):
    if len(active_actions) != WEEKS or active_actions[0] != ACTIONS[0]:
        raise ValueError("A fixed active calendar must have 24 weeks and start at M")

    def policy(observation):
        week = week_from_observation(observation)
        return active_actions[min(week + 1, WEEKS - 1)]

    return policy


def context_oracle_policy(tape: dict):
    def policy(observation):
        week = week_from_observation(observation)
        target = tape["context_schedule"][min(week + 1, WEEKS - 1)]
        return ACTION_FOR_CONTEXT[target]

    return policy


def ci95(values, seed: int = 20260713, n_boot: int = 4000):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.asarray([
        rng.choice(x, len(x), replace=True).mean() for _ in range(n_boot)
    ])
    return [float(x.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def requested_calendar_from_sequence(names: tuple[str, ...]):
    actions_by_name = {name: action for action, name in ACTION_NAMES.items()}
    return tuple(actions_by_name[name] for name in names)


def phase_calendar(cal_rows: list[dict]):
    sequences = [active_sequence(row) for row in cal_rows]
    phase_actions = []
    for phase in range(3):
        start, end = phase * 8, (phase + 1) * 8
        counts = Counter(name for sequence in sequences for name in sequence[start:end])
        phase_actions.append(counts.most_common(1)[0][0])
    desired = [phase_actions[min(week // 8, 2)] for week in range(WEEKS)]
    desired[0] = "M"
    return tuple(desired), phase_actions


def per_week_calendar(cal_rows: list[dict]):
    sequences = [active_sequence(row) for row in cal_rows]
    desired = []
    for week in range(WEEKS):
        counts = Counter(sequence[week] for sequence in sequences)
        desired.append(counts.most_common(1)[0][0])
    desired[0] = "M"
    return tuple(desired)


def metric_digest(row: dict) -> str:
    payload = {key: row[key] for key in METRIC_KEYS}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    started = time.perf_counter()
    calibration = tapes(CAL_START, N_CAL, "calibration")
    locked = tapes(LOCKED_START, N_LOCKED, "locked")

    cal_signal = [run_policy(tape, signal_policy) for tape in calibration]
    cal_sequences = [active_sequence(row) for row in cal_signal]
    modal_sequence, modal_frequency = Counter(cal_sequences).most_common(1)[0]
    per_week_names = per_week_calendar(cal_signal)
    phase_names, phase_actions = phase_calendar(cal_signal)

    replacement_policies: dict[str, Callable] = {
        "constant_M": constant(ACTIONS[0]),
        "signal_adaptive": signal_policy,
        "modal_complete_calibration_sequence": calendar_policy(
            requested_calendar_from_sequence(modal_sequence)
        ),
        "calibration_per_week_calendar": calendar_policy(
            requested_calendar_from_sequence(per_week_names)
        ),
        "calibration_phase_only": calendar_policy(
            requested_calendar_from_sequence(phase_names)
        ),
    }
    locked_rows = {
        name: [run_policy(tape, policy) for tape in locked]
        for name, policy in replacement_policies.items()
    }
    oracle_rows = [run_policy(tape, context_oracle_policy(tape)) for tape in locked]
    locked_rows["clairvoyant_next_context_feasible"] = oracle_rows

    baseline = locked_rows["constant_M"]
    signal = locked_rows["signal_adaptive"]
    comparisons = {}
    for name, rows in locked_rows.items():
        if name == "constant_M":
            continue
        comparisons[name] = {
            "ret_minus_constant_M_ci95": ci95([
                row["ret_excel"] - base["ret_excel"]
                for row, base in zip(rows, baseline)
            ]),
            "ret_minus_signal_ci95": ci95([
                row["ret_excel"] - candidate["ret_excel"]
                for row, candidate in zip(rows, signal)
            ]),
            "service_reduction_vs_constant_M_ci95": ci95([
                (base["service_loss_auc_ration_hours"] - row["service_loss_auc_ration_hours"])
                / max(abs(base["service_loss_auc_ration_hours"]), 1.0)
                for row, base in zip(rows, baseline)
            ]),
            "lost_minus_constant_M_ci95": ci95([
                row["n_lost"] - base["n_lost"] for row, base in zip(rows, baseline)
            ]),
        }

    signal_sequences = [active_sequence(row) for row in signal]
    component_resources = {}
    for name, rows in locked_rows.items():
        component_resources[name] = {
            "total_team_hours_unique": sorted({row["total_token_hours"] for row in rows}),
            "mean_m_hours": float(np.mean([row["token_hours_m"] for row in rows])),
            "mean_t_hours": float(np.mean([row["token_hours_t"] for row in rows])),
            "mean_r_hours": float(np.mean([row["token_hours_r"] for row in rows])),
            "mean_reserve_units_issued": float(np.mean([row["reserve_units_issued"] for row in rows])),
            "max_reserve_units_issued": float(np.max([row["reserve_units_issued"] for row in rows])),
        }

    null_hashes = []
    null_equal = True
    for tape in tapes(1_099_001, 3, "disposable_null"):
        null_tape = deepcopy(tape)
        null_tape["profile"]["r11_factors"] = [1.0, 1.0]
        null_tape["profile"]["transport_factors"] = [1.0, 1.0]
        null_tape["profile"]["reserve_issue"] = [0.0, 0.0]
        rows = [run_policy(null_tape, constant(action)) for action in ACTIONS]
        rows.append(run_policy(null_tape, signal_policy))
        digests = [metric_digest(row) for row in rows]
        null_hashes.append({"seed": null_tape["seed"], "metric_hashes": digests})
        null_equal &= len(set(digests)) == 1

    calendar_count = sum(
        math.comb(WEEKS - switches, switches) * (len(ACTIONS) - 1) ** switches
        for switches in range(WEEKS // 2 + 1)
    )
    elapsed = time.perf_counter() - started
    n_runs = N_CAL + N_LOCKED * len(locked_rows) + 3 * 4
    seconds_per_run = elapsed / n_runs
    frontier_runs = calendar_count * N_CAL
    pi_runs = calendar_count * N_LOCKED

    result = {
        "schema_version": "paper2_bottleneck_corrective_completeness_v1",
        "status": "ACTIVE_FOR_BOUND_NOT_TERMINAL",
        "scientific_use": "retrospective corrective audit on already-burned calibration/locked tapes; not a confirmatory claim",
        "learner_range_1120001_opened": False,
        "canonical_metric": "ret_excel_visible_v1",
        "calendar_frontier": {
            "weeks": WEEKS,
            "actions": [ACTION_NAMES[action] for action in ACTIONS],
            "effective_full_horizon_calendar_count": calendar_count,
            "constants_previously_tested": len(ACTIONS),
            "calibration_frontier_des_runs": frontier_runs,
            "per_tape_pi_des_runs": pi_runs,
            "measured_seconds_per_run": seconds_per_run,
            "estimated_serial_cpu_days_frontier": frontier_runs * seconds_per_run / 86400.0,
            "estimated_serial_cpu_days_pi": pi_runs * seconds_per_run / 86400.0,
            "exact_bound_available": False,
            "blocking_implementation_fact": "The SimPy MFSC has no branchable state snapshot and terminal canonical ReT is path-dependent/non-additive."
        },
        "trajectory_audit": {
            "locked_unique_signal_sequences": len(set(signal_sequences)),
            "locked_n_tapes": N_LOCKED,
            "calibration_modal_complete_sequence": "".join(modal_sequence),
            "calibration_modal_frequency": modal_frequency,
            "calibration_n_tapes": N_CAL,
            "calibration_per_week_calendar": "".join(per_week_names),
            "calibration_phase_actions": phase_actions,
            "replacement_comparisons": comparisons,
        },
        "component_resource_audit": {
            "total_team_hours_equal": all(
                values["total_team_hours_unique"] == [WEEKS * 168.0]
                for values in component_resources.values()
            ),
            "frozen_contract_resource": "one_response_team_week_equal_by_construction",
            "allocation_destination_hours_equal": False,
            "allocation_destination_hours_are_separate_contract_resources": False,
            "reserve_units_issued_equal": False,
            "reserve_resource_semantics_resolved": False,
            "reason": (
                "The frozen contract defines one team-week, so unequal M/T/R destination hours "
                "are the decision trajectory rather than separate resource purchases. Reserve "
                "ration issue/replenishment differs and must be frozen as an outcome or separately "
                "matched resource before promotion. The real team's cross-target fungibility is "
                "also unvalidated."
            ),
            "by_policy": component_resources,
        },
        "clairvoyant_diagnostic": {
            "kind": "feasible_policy_lower_bound_not_H_PI_ceiling",
            **comparisons["clairvoyant_next_context_feasible"],
        },
        "null_physics": {
            "definition": "r11 and transport efficacy factors set to 1; reserve issue set to 0",
            "exact_selected_metric_equivalence": null_equal,
            "selected_metric_keys": list(METRIC_KEYS),
            "n_retrospective_tapes": 3,
            "scope_warning": "Does not hash complete actions, internal state, flow trajectories or full order ledgers.",
            "rows": null_hashes,
            "preregistered": False,
        },
        "conclusion": "The tested observable signal policy is a null, but the 24-week integrated-resource family lacks a complete open-loop frontier and resource-restricted PI ceiling. It cannot support either Paper 2 or a family-level boundary certificate yet.",
    }
    output = ROOT / "results" / "paper2_bottleneck" / "corrective_completeness_audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if null_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
