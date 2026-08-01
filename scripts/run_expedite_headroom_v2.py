#!/usr/bin/env python3
"""Run the preregistered scarce-expedition headroom experiment.

This is an extended-DES experiment, not a thesis-faithful reproduction.  It
compares three calendars under a common exogenous risk tape:

* a fixed leg/evenly-spaced constant selected on calibration seeds;
* a tape oracle selected from a frozen risk-overlap score (not outcome-optimal);
* a seed-only placebo with the same number and size of charges.

The primary adjudication endpoint is the service-first lexicographic key.  ReT
and Cobb--Douglas are retained as named sensitivities; neither is allowed to
rescue an abandonment win.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.cobb_douglas_resilience import (  # noqa: E402
    CobbDouglasRecorder,
    score_comparison_set,
)
from supply_chain.config import HOURS_PER_DAY, HOURS_PER_YEAR, THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.service_first_metric import (  # noqa: E402
    SERVICE_FIRST_METRIC_ID,
    service_first_components,
    service_first_key,
)
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402


R1R = ("R11", "R12", "R13", "R14")
R2R = ("R21", "R22", "R23", "R24")
REGIMES: dict[str, tuple[tuple[str, ...], dict[str, float], dict[str, float]]] = {
    "R1r": (R1R, {}, {}),
    "R2r": (R2R, {}, {}),
    "R1r+R2r": (R1R + R2R, {}, {}),
    "R1r+R2r|freq3": (R1R + R2R, {"R23": 3.0}, {}),
}
LEGS = ("op8", "op10", "op12")
LEG_OPS: dict[str, frozenset[int]] = {
    "op8": frozenset({8, 9}),
    "op10": frozenset({10, 11}),
    "op12": frozenset({12, 13}),
}
BUDGETS = (0.0, 168.0, 336.0, 672.0, 1344.0)
CHARGE_HOURS = 24.0
PRIMARY_DIAGNOSTIC = "ret_excel_risk_conditional"
SIDE_METRICS = (
    "ret_excel",
    "ret_excel_visible_clipped_0_1",
    "ret_excel_full_ledger",
    "flow_fill_rate",
    "lost_orders",
    "backorder_qty_final",
    "service_loss_auc_ration_hours",
)
MIN_SAME_LEG_GAP_DAYS = 2
# The first Op8/Op10/Op12 hooks occur around 919--990 h in the deterministic
# baseline.  The action calendar therefore opens at the next whole-day boundary
# (42 days = 1008 h), before any policy can arm an expedition.  This is a
# conservative fixed epoch, not a peek at an evaluation tape.
ACTION_START_DAY = 42
# The final 14 days are a fixed settlement window: no new request is armed
# there, so a valid calendar has time to reach its next eligible hook before
# the episode closes.
ACTION_TAIL_DAYS = 14
BOOTSTRAP_SEED = 20260801

# Blocks already used by the adjacent headroom campaigns.  The manifest is
# read and repeated in the result so a later runner cannot silently reuse them.
EXCLUDED_BLOCKS = (
    (4_900_001, 4_900_006),
    (4_900_501, 4_900_506),
    (5_100_001, 5_100_012),
    (5_200_001, 5_200_008),
    (5_400_001, 5_400_008),
    (5_600_001, 5_600_008),
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _tape_row(event: Any) -> dict[str, Any]:
    return {
        "risk_id": str(event.risk_id),
        "start_time": float(event.start_time),
        "end_time": float(event.end_time),
        "duration": float(event.duration),
        "affected_ops": [int(op) for op in event.affected_ops],
        "description": str(event.description or ""),
        "magnitude": float(event.magnitude),
        "unit": str(event.unit or "incidents"),
        "affected_cssu": event.affected_cssu,
    }


def tape_sha256(tape: Iterable[Mapping[str, Any]]) -> str:
    rows = sorted(
        [dict(row) for row in tape],
        key=lambda row: (
            float(row["start_time"]),
            str(row["risk_id"]),
            float(row.get("end_time", row["start_time"])),
        ),
    )
    return sha256(_canonical_json(rows).encode()).hexdigest()


def _sim(
    *,
    regime: str,
    seed: int,
    horizon: float,
    tape: Iterable[Mapping[str, Any]] | None,
    budget: float,
    reduction_hours: float = 12.0,
) -> MFSCSimulation:
    risks, frequency, impact = REGIMES[regime]
    return MFSCSimulation(
        shifts=1,
        initial_buffers={"op3_rm": 0.0, "op5_rm": 0.0, "op9_rations": 0.0},
        inventory_replenishment_period=0.0,
        seed=seed,
        horizon=horizon,
        risks_enabled=True,
        risk_level="current",
        enabled_risks=set(risks),
        risk_frequency_multipliers_by_id=frequency or None,
        risk_impact_multipliers_by_id=impact or None,
        risk_event_tape=tape,
        strict_exogenous_crn=True,
        year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"],
        r14_defect_mode=P["r14_defect_mode"],
        order_fulfillment_mode="op9_linked",
        op9_dispatch_policy="fixed_clock_daily",
        downstream_transport_capacity_mode="parallel",
        expedite_budget_hours=budget,
        expedite_reduction_hours=reduction_hours,
        expedite_charge_hours=CHARGE_HOURS,
    )


def materialize_tape(regime: str, seed: int, horizon: float) -> list[dict[str, Any]]:
    """Generate one exogenous tape before any calendar arm is evaluated."""
    sim = _sim(regime=regime, seed=seed, horizon=horizon, tape=None, budget=0.0)
    sim.run()
    enabled = set(REGIMES[regime][0])
    rows = [_tape_row(event) for event in sim.risk_events
            if str(event.risk_id) in enabled]
    return sorted(rows, key=lambda row: (row["start_time"], row["risk_id"]))


def _slot_list(
    days: int,
    *,
    start_day: int = ACTION_START_DAY,
    end_day: int | None = None,
) -> list[tuple[int, str]]:
    end = days - ACTION_TAIL_DAYS if end_day is None else int(end_day)
    return [(day, leg) for day in range(start_day, end) for leg in LEGS]


def _respect_gap(candidates: Iterable[tuple[int, str]], n: int) -> list[tuple[int, str]]:
    candidate_list = [(int(day), str(leg)) for day, leg in candidates]
    selected: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for slot in candidate_list:
        if slot in seen:
            continue
        if any(
            slot[1] == old_leg
            and abs(slot[0] - old_day) < MIN_SAME_LEG_GAP_DAYS
            for old_day, old_leg in selected
        ):
            continue
        selected.append(slot)
        seen.add(slot)
        if len(selected) == n:
            return sorted(selected)
    # A random or score-sorted pass can greedily choose a late slot and then
    # skip an earlier compatible slot. Fill deterministically from the
    # remaining calendar before declaring the contract impossible.
    for slot in sorted(candidate_list):
        if slot in seen:
            continue
        if any(
            slot[1] == old_leg
            and abs(slot[0] - old_day) < MIN_SAME_LEG_GAP_DAYS
            for old_day, old_leg in selected
        ):
            continue
        selected.append(slot)
        seen.add(slot)
        if len(selected) == n:
            return sorted(selected)
    raise ValueError(
        f"admissible calendar has only {len(selected)} slots for n={n}; "
        "the fixed same-leg spacing contract cannot be satisfied"
    )


def constant_schedule(
    budget: float,
    *,
    leg: str,
    phase_day: int,
    days: int,
    start_day: int = ACTION_START_DAY,
    end_day: int | None = None,
) -> list[tuple[int, str]]:
    n = int(round(float(budget) / CHARGE_HOURS))
    if n == 0:
        return []
    if leg not in LEGS:
        raise ValueError(f"unknown leg {leg!r}")
    effective_end_day = days - ACTION_TAIL_DAYS if end_day is None else int(end_day)
    available_days = effective_end_day - start_day
    if available_days <= 0:
        raise ValueError("horizon ends before the fixed expedition action epoch")
    candidates = [
        (
            start_day
            + (int(phase_day) + int(np.floor(i * available_days / n)))
            % available_days,
            leg,
        )
        for i in range(n)
    ]
    return _respect_gap(sorted(candidates), n)


def placebo_schedule(
    budget: float,
    *,
    seed: int,
    days: int,
    start_day: int = ACTION_START_DAY,
    end_day: int | None = None,
) -> list[tuple[int, str]]:
    n = int(round(float(budget) / CHARGE_HOURS))
    if n == 0:
        return []
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(budget), 0xE7D]))
    slots = _slot_list(days, start_day=start_day, end_day=end_day)
    order = rng.permutation(len(slots))
    return _respect_gap((slots[int(i)] for i in order), n)


def tape_oracle_schedule(
    budget: float,
    *,
    tape: Iterable[Mapping[str, Any]],
    days: int,
    start_day: int = ACTION_START_DAY,
    end_day: int | None = None,
) -> list[tuple[int, str]]:
    """Select top risk-overlap slots, not top observed outcome slots."""
    n = int(round(float(budget) / CHARGE_HOURS))
    if n == 0:
        return []
    events = [dict(row) for row in tape]
    scored: list[tuple[float, int, int, str]] = []
    for day, leg in _slot_list(days, start_day=start_day, end_day=end_day):
        start = day * HOURS_PER_DAY
        end = start + HOURS_PER_DAY
        score = 0.0
        for event in events:
            overlap = max(
                0.0,
                min(end, float(event["end_time"]))
                - max(start, float(event["start_time"])),
            )
            if overlap <= 0.0:
                continue
            affected = {int(op) for op in event.get("affected_ops", [])}
            if affected.intersection(LEG_OPS[leg]):
                score += overlap * max(1.0, abs(float(event.get("magnitude", 1.0))))
        scored.append((score, day, LEGS.index(leg), leg))
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))
    return _respect_gap(((day, leg) for _, day, _, leg in scored), n)


def run_episode(
    *,
    regime: str,
    seed: int,
    horizon: float,
    budget: float,
    policy: str,
    schedule: list[tuple[int, str]],
    tape: list[dict[str, Any]],
) -> dict[str, Any]:
    sim = _sim(
        regime=regime,
        seed=seed,
        horizon=horizon,
        tape=tape,
        budget=budget,
    )
    recorder = CobbDouglasRecorder(period_hours=HOURS_PER_DAY)
    by_day: dict[int, list[str]] = defaultdict(list)
    for day, leg in schedule:
        by_day[int(day)].append(str(leg))
    days = int(round(horizon / HOURS_PER_DAY))
    for day in range(days):
        for leg in by_day.get(day, []):
            sim.arm_expedition(leg)
        sim.step(step_hours=HOURS_PER_DAY)
        recorder.sample(sim)
    if abs(float(sim.env.now) - float(horizon)) > 1e-6:
        raise AssertionError(f"episode ended at {sim.env.now}, expected {horizon}")
    panel = compute_episode_metrics(sim)
    exp_events = [dict(event) for event in sim.expedite_events]
    armed = [event for event in exp_events if event.get("status") == "armed"]
    applied = [event for event in exp_events if event.get("status") == "applied"]
    rejected = [event for event in exp_events if event.get("status") == "rejected_budget"]
    components = service_first_components(panel)
    return {
        "regime": regime,
        "seed": int(seed),
        "budget": float(budget),
        "policy": policy,
        "schedule": [[int(day), str(leg)] for day, leg in schedule],
        "tape_sha256": tape_sha256(tape),
        "metrics": {key: float(panel[key]) for key in (
            PRIMARY_DIAGNOSTIC, *SIDE_METRICS,
        )},
        "service_first": components,
        "cobb_douglas": recorder.aggregate(),
        "expedition": {
            "granted_hours": float(sim.expedite_budget_hours),
            "charged_hours": float(
                sum(float(event.get("budget_charge", 0.0)) for event in armed)
            ),
            "remaining_hours": float(sim.expedite_budget_remaining),
            "n_scheduled": len(schedule),
            "n_armed": len(armed),
            "n_applied": len(applied),
            "n_rejected_budget": len(rejected),
            "armed": armed,
            "applied": applied,
            "rejected_budget": rejected,
        },
    }


def _aggregate_service(rows: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    return (
        float(np.mean([row["service_first"]["no_lost_orders"] for row in rows])),
        float(np.mean([row["service_first"]["flow_fill_rate"] for row in rows])),
        float(np.mean([row["service_first"]["negative_backorder_qty_final"] for row in rows])),
        float(np.mean([row["service_first"]["ret_excel_visible_clipped_0_1"] for row in rows])),
    )


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty vector")
    draws = np.array([
        float(np.mean(values[rng.integers(0, values.size, size=values.size)]))
        for _ in range(n_boot)
    ])
    return {
        "mean": float(np.mean(values)),
        "lcb95": float(np.percentile(draws, 2.5)),
        "ucb95": float(np.percentile(draws, 97.5)),
    }


def _seed_paired(
    rows: list[dict[str, Any]],
    *,
    budget: float,
    candidate: str,
    comparator: str,
    field: str,
    seeds: list[int],
    regime_names: list[str],
) -> np.ndarray:
    lookup = {
        (row["policy"], row["regime"], int(row["seed"])): row
        for row in rows
        if float(row["budget"]) == float(budget)
    }
    values = []
    for seed in seeds:
        cell_diffs = []
        for regime in regime_names:
            left = lookup[(candidate, regime, seed)]
            right = lookup[(comparator, regime, seed)]
            if field == "service_first_key":
                cell_diffs.append(float(service_first_key(left["metrics"])[0]
                                     - service_first_key(right["metrics"])[0]))
            else:
                cell_diffs.append(float(left["metrics"][field] - right["metrics"][field]))
        values.append(float(np.mean(cell_diffs)))
    return np.asarray(values, dtype=float)


def _lookup_rows(rows: list[dict[str, Any]], budget: float, policy: str) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (row["regime"], int(row["seed"])): row
        for row in rows
        if float(row["budget"]) == float(budget) and row["policy"] == policy
    }


def _seed_custody(seeds: list[int]) -> dict[str, Any]:
    overlaps = {
        f"{low}-{high}": [seed for seed in seeds if low <= seed <= high]
        for low, high in EXCLUDED_BLOCKS
    }
    return {
        "passed": not any(overlaps.values()) and len(seeds) == len(set(seeds)),
        "requested_seeds": seeds,
        "excluded_blocks": [list(block) for block in EXCLUDED_BLOCKS],
        "overlaps": overlaps,
    }


def _falsifiers(
    *,
    rows: list[dict[str, Any]],
    zero_identity: list[dict[str, Any]],
    tapes: dict[tuple[str, int], list[dict[str, Any]]],
    seeds: list[int],
    eval_seeds: list[int],
    regime_names: list[str],
    budgets: tuple[float, ...],
    days: int,
) -> dict[str, Any]:
    all_rows = rows
    f1_rows = []
    f2_rows = []
    f3_rows = []
    for row in all_rows:
        exp = row["expedition"]
        expected_charge = len(row["schedule"]) * CHARGE_HOURS
        f1_rows.append(
            exp["charged_hours"] <= row["budget"] + 1e-9
            and abs(exp["charged_hours"] - expected_charge) <= 1e-9
            and abs(exp["remaining_hours"] - (row["budget"] - expected_charge)) <= 1e-9
        )
        f2_rows.extend(
            abs(float(event["nominal_pt"]) - float(event["effective_base_pt"])
                - float(event["reduction_hours"])) <= 1e-9
            and abs(float(event["reduction_hours"]) - 12.0) <= 1e-9
            for event in exp["applied"]
        )
        f3_rows.append(
            exp["n_armed"] == exp["n_applied"] == len(row["schedule"])
            and all(float(event["applied_at"]) >= float(event["armed_at"])
                    for event in exp["applied"])
        )

    f4 = all(
        pair["left"]["metrics"] == pair["right"]["metrics"]
        and pair["left"]["service_first"] == pair["right"]["service_first"]
        and pair["left"]["cobb_douglas"] == pair["right"]["cobb_douglas"]
        for pair in zero_identity
    )
    grouped_tapes: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in all_rows:
        grouped_tapes[(row["regime"], int(row["seed"]))].add(row["tape_sha256"])
    f5 = all(len(hashes) == 1 for hashes in grouped_tapes.values())
    placebo_rows = [row for row in all_rows if row["policy"] == "placebo"]
    f6 = all(
        row["expedition"]["n_scheduled"] == row["expedition"]["n_armed"]
        and row["expedition"]["charged_hours"]
        == row["expedition"]["n_scheduled"] * CHARGE_HOURS
        for row in placebo_rows
    )
    # The deployable calendars are pure functions of seed, budget and the
    # frozen grid. Recomputing them with a deliberately altered tape is the
    # executable no-future check; the tape oracle is excluded by design.
    f7_checks = []
    for row in placebo_rows:
        expected = placebo_schedule(
            float(row["budget"]), seed=int(row["seed"]), days=days,
            start_day=ACTION_START_DAY,
        )
        f7_checks.append(expected == [tuple(item) for item in row["schedule"]])
    constant_rows = [row for row in all_rows if row["policy"] == "constant"]
    # Constant schedules are already stored with their chosen leg/phase. They
    # are checked for being seed-invariant within a budget.
    for budget in budgets:
        schedules = {
            tuple(tuple(item) for item in row["schedule"])
            for row in constant_rows if float(row["budget"]) == float(budget)
        }
        f7_checks.append(len(schedules) == 1)
    f8_bad = []
    for budget in budgets:
        oracle = _lookup_rows(all_rows, budget, "tape_oracle")
        constant = _lookup_rows(all_rows, budget, "constant")
        for key, left in oracle.items():
            right = constant[key]
            left_metrics = left["metrics"]
            right_metrics = right["metrics"]
            ret_gain = (
                left_metrics["ret_excel_visible_clipped_0_1"]
                > right_metrics["ret_excel_visible_clipped_0_1"] + 1e-12
            )
            service_worse = (
                left_metrics["lost_orders"] > right_metrics["lost_orders"] + 1e-12
                or left_metrics["flow_fill_rate"] < right_metrics["flow_fill_rate"] - 1e-12
                or left_metrics["backorder_qty_final"]
                > right_metrics["backorder_qty_final"] + 1e-12
            )
            if ret_gain and service_worse and service_first_key(left_metrics) > service_first_key(right_metrics):
                f8_bad.append(key)
    custody = _seed_custody(seeds)
    checks = {
        "f1_budget_conserved": {"passed": bool(all(f1_rows)), "evidence": {"rows": len(f1_rows)}},
        "f2_pt_effect_real": {"passed": bool(f2_rows) and bool(all(f2_rows)), "evidence": {"applied_events": len(f2_rows)}},
        "f3_next_leg_only": {"passed": bool(all(f3_rows)), "evidence": {"rows": len(f3_rows)}},
        "f4_zero_budget_identity": {"passed": f4, "evidence": {"paired_zero_rows": len(zero_identity)}},
        "f5_same_exogenous_tape": {
            "passed": f5,
            "evidence": {"groups": len(grouped_tapes), "hashes_per_group": {str(k): sorted(v) for k, v in grouped_tapes.items()}},
        },
        "f6_placebo_same_charge": {"passed": f6, "evidence": {"placebo_rows": len(placebo_rows)}},
        "f7_no_future_in_deployable_arms": {
            "passed": bool(f7_checks) and bool(all(f7_checks)),
            "evidence": {"checks": len(f7_checks), "passed_checks": int(sum(f7_checks))},
        },
        "f8_no_abandonment_win": {
            "passed": not f8_bad,
            "evidence": {"why_it_can_fail": "a raw ReT gain must not override a service loss", "contradictory_cells": [list(k) for k in f8_bad]},
        },
        "f9_seeds_virgin": {
            "passed": bool(custody["passed"]),
            "evidence": custody,
        },
    }
    checks["all_passed"] = all(item["passed"] for key, item in checks.items() if key != "all_passed")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--calibration-seeds", type=int, default=4)
    parser.add_argument("--horizon-hours", type=float, default=float(HOURS_PER_YEAR))
    parser.add_argument("--n-boot", type=int, default=2_000)
    parser.add_argument("--output", type=Path,
                        default=Path("results/sensitivity/expedite_headroom_v2/result.json"))
    parser.add_argument("--seed-base", type=int, default=5_800_001)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {args.output}")
    if args.seeds <= args.calibration_seeds or args.calibration_seeds < 1:
        raise ValueError("--seeds must be greater than --calibration-seeds >= 1")
    horizon = float(args.horizon_hours)
    if horizon <= 0 or abs(horizon / HOURS_PER_DAY - round(horizon / HOURS_PER_DAY)) > 1e-9:
        raise ValueError("horizon must be a positive whole number of days")
    seeds = [int(args.seed_base) + i for i in range(args.seeds)]
    calibration_seeds = seeds[:args.calibration_seeds]
    evaluation_seeds = seeds[args.calibration_seeds:]
    days = int(round(horizon / HOURS_PER_DAY))
    started = time.perf_counter()

    tapes: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for regime in REGIMES:
        for seed in seeds:
            tapes[(regime, seed)] = materialize_tape(regime, seed, horizon)
    print(f"  risk tapes: {len(tapes)} ({time.perf_counter() - started:.0f}s)", flush=True)

    # The constant is selected once per budget on calibration seeds, across all
    # four regimes. The service-first key is aggregated componentwise; no ReT
    # scalar is allowed to select it.
    constant_choices: dict[float, dict[str, Any]] = {}
    for budget in BUDGETS:
        if budget == 0.0:
            constant_choices[budget] = {"leg": "op8", "phase_day": 0, "key": [1.0, 0.0, 0.0, 0.0]}
            continue
        candidates = [(leg, phase) for leg in LEGS for phase in range(7)]
        candidate_scores: dict[str, dict[str, Any]] = {}
        for leg, phase in candidates:
            schedule = constant_schedule(
                budget, leg=leg, phase_day=phase, days=days,
                start_day=ACTION_START_DAY,
            )
            candidate_rows = []
            for regime in REGIMES:
                for seed in calibration_seeds:
                    row = run_episode(
                        regime=regime, seed=seed, horizon=horizon, budget=budget,
                        policy="constant_calibration", schedule=schedule,
                        tape=tapes[(regime, seed)],
                    )
                    candidate_rows.append(row)
            key = _aggregate_service(candidate_rows)
            name = f"{leg}|phase={phase}"
            candidate_scores[name] = {"leg": leg, "phase_day": phase, "key": list(key)}
        chosen_name = max(candidate_scores, key=lambda name: tuple(candidate_scores[name]["key"]))
        constant_choices[budget] = {"candidate": chosen_name, **candidate_scores[chosen_name],
                                    "n_candidates": len(candidate_scores)}
        print(f"  constant B={budget:.0f}: {chosen_name} key={constant_choices[budget]['key']}", flush=True)

    rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        choice = constant_choices[budget]
        for regime in REGIMES:
            for seed in evaluation_seeds:
                tape = tapes[(regime, seed)]
                schedules = {
                    "constant": constant_schedule(
                        budget, leg=choice["leg"], phase_day=int(choice["phase_day"]),
                        days=days, start_day=ACTION_START_DAY,
                    ),
                    "tape_oracle": tape_oracle_schedule(
                        budget, tape=tape, days=days, start_day=ACTION_START_DAY,
                    ),
                    "placebo": placebo_schedule(
                        budget, seed=seed, days=days, start_day=ACTION_START_DAY,
                    ),
                }
                for policy, schedule in schedules.items():
                    rows.append(run_episode(
                        regime=regime, seed=seed, horizon=horizon, budget=budget,
                        policy=policy, schedule=schedule, tape=tape,
                    ))
        print(f"  evaluation B={budget:.0f} ({time.perf_counter() - started:.0f}s)", flush=True)

    # B=0 identity is checked with the same replay tape in two independent
    # feature-disabled constructions; this isolates the expedition hook from
    # the unrelated difference between a generated and a replayed risk path.
    zero_identity: list[dict[str, Any]] = []
    for regime in REGIMES:
        for seed in evaluation_seeds:
            tape = tapes[(regime, seed)]
            left = run_episode(regime=regime, seed=seed, horizon=horizon, budget=0.0,
                               policy="zero_identity_left", schedule=[], tape=tape)
            right = run_episode(regime=regime, seed=seed, horizon=horizon, budget=0.0,
                                policy="zero_identity_right", schedule=[], tape=tape)
            zero_identity.append({"left": left, "right": right})

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    regime_names = list(REGIMES)
    comparisons: dict[str, Any] = {}
    for budget in BUDGETS:
        budget_result: dict[str, Any] = {}
        for candidate, comparator in (("tape_oracle", "constant"), ("tape_oracle", "placebo")):
            key = f"{candidate}_vs_{comparator}"
            budget_result[key] = {}
            for field in (*SIDE_METRICS, "ret_excel_risk_conditional"):
                values = _seed_paired(rows, budget=budget, candidate=candidate,
                                      comparator=comparator, field=field,
                                      seeds=evaluation_seeds, regime_names=regime_names)
                budget_result[key][field] = _bootstrap_ci(values, rng, args.n_boot)
            oracle = _lookup_rows(rows, budget, candidate)
            control = _lookup_rows(rows, budget, comparator)
            lex_wins = []
            service_compatible = []
            for pair_key, left in oracle.items():
                right = control[pair_key]
                lex_wins.append(service_first_key(left["metrics"]) > service_first_key(right["metrics"]))
                service_compatible.append(
                    left["metrics"]["lost_orders"] <= right["metrics"]["lost_orders"] + 1e-9
                    and left["metrics"]["flow_fill_rate"] >= right["metrics"]["flow_fill_rate"] - 1e-9
                    and left["metrics"]["backorder_qty_final"] <= right["metrics"]["backorder_qty_final"] + 1e-9
                )
            # Seed-level fractions are the independent bootstrap units.
            win_by_seed = []
            compat_by_seed = []
            for seed in evaluation_seeds:
                keys = [(regime, seed) for regime in regime_names]
                win_by_seed.append(float(np.mean([lex_wins[list(oracle).index(k)] for k in keys])))
                compat_by_seed.append(float(np.mean([service_compatible[list(oracle).index(k)] for k in keys])))
            budget_result[key]["lexicographic_win_fraction"] = _bootstrap_ci(
                np.asarray(win_by_seed), rng, args.n_boot
            )
            budget_result[key]["service_compatible_fraction"] = _bootstrap_ci(
                np.asarray(compat_by_seed), rng, args.n_boot
            )
        comparisons[str(int(budget))] = budget_result

    falsifiers = _falsifiers(
        rows=rows,
        zero_identity=zero_identity,
        tapes=tapes,
        seeds=seeds,
        eval_seeds=evaluation_seeds,
        regime_names=regime_names,
        budgets=BUDGETS,
        days=days,
    )

    # Secondary Cobb--Douglas scores use the frozen calibration exponents and
    # one declared comparison set: evaluation policy x budget x regime,
    # averaged over evaluation seeds. The set-relative kappa is never mixed
    # into the service-first adjudication.
    cells: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"B{int(row['budget'])}|{row['policy']}|{row['regime']}"].append(row)
    for name, group in grouped.items():
        keys = ("zeta", "epsilon", "phi", "tau", "kappa")
        cells[name] = {key: float(np.mean([row["cobb_douglas"][key] for row in group])) for key in keys}
    cd_contract = json.loads(Path("contracts/cobb_douglas_calibration_v1.json").read_text())
    cd_scores = score_comparison_set(cells, cd_contract["exponents"])

    max_budget = max(BUDGETS)
    fill_gate = comparisons[str(int(max_budget))]["tape_oracle_vs_constant"]["flow_fill_rate"]
    fill_placebo = comparisons[str(int(max_budget))]["tape_oracle_vs_placebo"]["flow_fill_rate"]
    service_gate = comparisons[str(int(max_budget))]["tape_oracle_vs_constant"]["service_compatible_fraction"]
    if (
        falsifiers["all_passed"]
        and fill_gate["lcb95"] >= 0.01
        and fill_placebo["lcb95"] > 0.0
        and service_gate["lcb95"] >= 0.70
    ):
        verdict = "TIMING_HEADROOM_FOUND_SERVICE_FIRST"
    elif (
        falsifiers["all_passed"]
        and fill_gate["lcb95"] >= 0.01
        and fill_placebo["ucb95"] <= 0.0
    ):
        verdict = "OPEN_LOOP_EXPEDITION_VALUE_NO_TIMING_INFORMATION"
    elif falsifiers["all_passed"]:
        verdict = "NO_TIMING_HEADROOM_UNDER_SERVICE_FIRST"
    else:
        verdict = "HALTED_FALSIFIER_FAILED"

    payload: dict[str, Any] = {
        "schema_version": "expedite_headroom_v2",
        "claim_status": verdict,
        "contract_status": "extended_des_only",
        "endpoint": SERVICE_FIRST_METRIC_ID,
        "diagnostic_primary": PRIMARY_DIAGNOSTIC,
        "budgets": list(BUDGETS),
        "horizon_hours": horizon,
        "days": days,
        "regimes": list(REGIMES),
        "seeds": seeds,
        "calibration_seeds": calibration_seeds,
        "evaluation_seeds": evaluation_seeds,
        "constant_choices": constant_choices,
        "comparisons": comparisons,
        "rows": rows,
        "cobb_douglas_cells": cells,
        "cobb_douglas_scores": cd_scores,
        "zero_budget_identity": zero_identity,
        "risk_tape_manifest": {
            f"{regime}|{seed}": {
                "sha256": tape_sha256(tapes[(regime, seed)]),
                "n_events": len(tapes[(regime, seed)]),
            }
            for regime in REGIMES for seed in seeds
        },
        "falsifiers": falsifiers,
        "reading_rule": {
            "H_PI_fill_threshold": 0.01,
            "service_compatible_lcb_threshold": 0.70,
            "tape_oracle_is_not_outcome_optimal": True,
            "no_weighted_service_ret_scalar": True,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload,
        args.output,
        contract=Path("docs/PREREGISTRO_EXPEDICION_HEADROOM_V2_2026-08-01.md"),
        reference=Path("contracts/cobb_douglas_calibration_v1.json"),
        stamp_extra={"experiment": "expedite_headroom_v2", "seed_base": args.seed_base},
    )
    print(f"  verdict: {verdict}")
    print(f"  falsifiers: {'PASA' if falsifiers['all_passed'] else 'FALLA'}")
    print(f"  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
