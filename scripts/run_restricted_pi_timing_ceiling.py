#!/usr/bin/env python3
"""Execute the frozen restricted timing ceiling after the risk gate passes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.paper2_exhaustive_search.restricted_timing_oracle import (
    ScheduleSpec,
    evaluate_schedule,
    paired_promotion_summary,
    periodic_binary_calendars,
    placebo_signal_series,
    posture_from_label,
    safe_against,
    select_frozen_postures,
)
from scripts.run_garrido_risk_headroom_sensitivity import build_profiles
from scripts.run_track_a_headroom_search import continuous_candidates


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def event_objects(rows: list[dict[str, Any]]) -> list[SimpleNamespace]:
    return [SimpleNamespace(**row) for row in rows]


def risk_signal_series(
    events: list[dict[str, Any]], *, start: float, n_steps: int
) -> list[float]:
    return [
        float(
            sum(
                row["risk_id"] in {"R22", "R24"}
                and now - 168.0 <= float(row["start_time"]) <= now
                for row in events
            )
        )
        for now in (start + 24.0 * index for index in range(n_steps))
    ]


def schedule_specs() -> list[ScheduleSpec]:
    specs = periodic_binary_calendars()
    specs.extend(
        ScheduleSpec("restricted_privileged", f"restricted_offset_{offset:+g}", float(offset))
        for offset in (-168, -72, -24, 0, 24, 72)
    )
    specs.extend(
        ScheduleSpec("weekly_privileged", f"weekly_offset_{offset:+g}", float(offset))
        for offset in (-168, -72, -24, 0, 24, 72)
    )
    specs.extend(
        ScheduleSpec(f"observable_{family}", f"observable_{family}", family)
        for family in ("real", "stale_168h", "shuffled_within_tape", "cross_tape_shift17")
    )
    return specs


def evaluate_task(task: dict[str, Any]) -> dict[str, Any]:
    seed = int(task["seed"])
    family = task["family"]
    if family == "constant":
        posture = posture_from_label(task["posture_label"])
        low = high = posture
        spec = ScheduleSpec("constant", task["schedule_id"], False)
    else:
        low = posture_from_label(task["low_label"])
        high = posture_from_label(task["high_label"])
        spec = ScheduleSpec(family, task["schedule_id"], task["payload"])
    metrics = evaluate_schedule(
        seed=seed,
        risk_overrides=task["risk_overrides"],
        low=low,
        high=high,
        spec=spec,
        max_daily_steps=int(task["max_daily_steps"]),
        known_risk_events=event_objects(task.get("known_risk_events", [])),
        observed_signal_series=task.get("observed_signal_series"),
    )
    return {
        "seed": seed,
        "family": family,
        "schedule_id": task["schedule_id"],
        "metrics": metrics,
    }


def best_fixed_schedule(
    rows: list[dict[str, Any]], families: set[str], low_rows: dict[int, dict[str, Any]]
) -> str:
    identifiers = sorted(
        {row["schedule_id"] for row in rows if row["family"] in families}
    )
    admissible = []
    for identifier in identifiers:
        selected = [row for row in rows if row["schedule_id"] == identifier]
        if len(selected) == len(low_rows) and all(
            safe_against(row["metrics"], low_rows[int(row["seed"])]) for row in selected
        ):
            admissible.append(identifier)
    if not admissible:
        return "constant_low"
    return max(
        admissible,
        key=lambda identifier: (
            float(np.mean([
                row["metrics"]["ret_excel"]
                for row in rows if row["schedule_id"] == identifier
            ])),
            identifier,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default="contracts/restricted_pi_timing_ceiling_v1.json")
    parser.add_argument("--risk-result", required=True)
    parser.add_argument("--risk-raw-rows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke-seeds", default="")
    args = parser.parse_args()

    contract_path = Path(args.contract)
    contract = json.loads(contract_path.read_text())
    risk_result_path = Path(args.risk_result)
    risk_raw_path = Path(args.risk_raw_rows)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if not risk_result_path.is_file() or not risk_raw_path.is_file():
        write_json_atomic(output / "result.json", {
            "status": "STOP_BEFORE_TIMING_RISK_EVIDENCE_MISSING",
            "timing_seeds_opened": False,
        })
        return 2
    risk_result = json.loads(risk_result_path.read_text())
    if risk_result.get("status") != "DEVELOPMENT_DOOR_FOUND":
        write_json_atomic(output / "result.json", {
            "status": "STOP_BEFORE_TIMING_NO_RISK_DOOR",
            "source_risk_status": risk_result.get("status"),
            "source_risk_result_sha256": sha256(risk_result_path),
            "timing_seeds_opened": False,
        })
        return 0

    r2_summaries = risk_result.get("group_budget_summaries", {}).get(
        "R2_frequency", []
    )
    frozen_cap = float(contract["posture_frontier"]["resource_budget_cap"])
    r2_summary = next(
        (
            row for row in r2_summaries
            if abs(float(row.get("budget_cap", -1.0)) - frozen_cap) <= 1e-12
            and bool(row.get("door_pass", False))
        ),
        None,
    )
    if r2_summary is None:
        write_json_atomic(output / "result.json", {
            "status": "STOP_BEFORE_TIMING_NO_R2_DOOR_AT_FROZEN_CAP",
            "frozen_resource_cap": frozen_cap,
            "source_risk_result_sha256": sha256(risk_result_path),
            "timing_seeds_opened": False,
        })
        return 0
    selection = select_frozen_postures(
        risk_raw_path,
        budget_cap=frozen_cap,
        robust_label=str(r2_summary["best_robust_constant"]),
    )
    if selection is None:
        write_json_atomic(output / "result.json", {
            "status": "STOP_BEFORE_TIMING_NO_SAFE_PROFILE_POSTURES",
            "source_risk_result_sha256": sha256(risk_result_path),
            "timing_seeds_opened": False,
        })
        return 0
    low, high, selected_profile = selection
    profiles = {row["id"]: row for row in build_profiles()}
    profile = profiles[selected_profile]

    seed_range = contract["development"]["seeds"]
    seeds = list(range(int(seed_range[0]), int(seed_range[1]) + 1))
    if args.smoke_seeds:
        seeds = [int(value) for value in args.smoke_seeds.split(",") if value.strip()]
    n_steps = int(contract["development"]["max_daily_steps"])

    # Probe the exogenous event tape with the low constant. Per-risk CRN makes
    # the same tape mandatory for every later action trajectory.
    probe_path = output / "risk_tapes.json"
    if probe_path.is_file():
        probes = json.loads(probe_path.read_text())
    else:
        probes = {}
        for seed in seeds:
            row = evaluate_task({
                "seed": seed,
                "family": "constant",
                "schedule_id": "constant_low",
                "posture_label": low.label,
                "risk_overrides": profile["overrides"],
                "max_daily_steps": n_steps,
            })
            probes[str(seed)] = row
            write_json_atomic(probe_path, probes)

    signals = {
        seed: risk_signal_series(
            probes[str(seed)]["metrics"]["realized_risk_events"],
            start=float(probes[str(seed)]["metrics"]["treatment_start"]),
            n_steps=n_steps,
        )
        for seed in seeds
    }
    shifted = {seed: seeds[(index + 17) % len(seeds)] for index, seed in enumerate(seeds)}

    candidate_constants = continuous_candidates(
        contract["posture_frontier"]["buffer_fractions"],
        contract["posture_frontier"]["shift_levels"],
    )
    tasks: list[dict[str, Any]] = []
    for seed in seeds:
        common = {
            "seed": seed,
            "risk_overrides": profile["overrides"],
            "max_daily_steps": n_steps,
            "known_risk_events": probes[str(seed)]["metrics"]["realized_risk_events"],
        }
        for candidate in candidate_constants:
            if candidate.label == low.label:
                continue  # already retained from the tape probe
            tasks.append({
                **common,
                "family": "constant",
                "schedule_id": f"constant_{candidate.label}",
                "posture_label": candidate.label,
            })
        for spec in schedule_specs():
            signal = None
            if spec.family.startswith("observable_"):
                placebo = str(spec.payload)
                signal = placebo_signal_series(
                    signals[seed],
                    family=placebo,
                    seed=seed,
                    cross_tape=signals[shifted[seed]],
                )
            tasks.append({
                **common,
                "family": spec.family,
                "schedule_id": spec.schedule_id,
                "payload": spec.payload,
                "low_label": low.label,
                "high_label": high.label,
                "observed_signal_series": signal,
            })

    partial_path = output / "raw_rows.partial.jsonl"
    rows = []
    if partial_path.is_file():
        rows = [json.loads(line) for line in partial_path.read_text().splitlines() if line]
    completed = {(int(row["seed"]), row["schedule_id"]) for row in rows}
    tasks = [task for task in tasks if (int(task["seed"]), task["schedule_id"]) not in completed]
    total = len(seeds) * (18 + len(schedule_specs()))

    def retain(row: dict[str, Any]) -> None:
        rows.append(row)
        with partial_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        write_json_atomic(output / "progress.json", {
            "completed": len(rows) + len(probes),
            "total": total,
            "fraction": (len(rows) + len(probes)) / total,
        })

    if int(args.workers) <= 1:
        for task in tasks:
            retain(evaluate_task(task))
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = [executor.submit(evaluate_task, task) for task in tasks]
            for future in as_completed(futures):
                retain(future.result())

    all_rows = list(rows)
    all_rows.extend(probes[str(seed)] for seed in seeds)
    # Fail closed if action trajectories perturbed the exogenous tape.
    tape_hashes = {}
    for seed in seeds:
        seed_rows = [row for row in all_rows if int(row["seed"]) == seed]
        hashes = {
            hashlib.sha256(json.dumps(
                row["metrics"]["realized_risk_events"], sort_keys=True
            ).encode()).hexdigest()
            for row in seed_rows
        }
        tape_hashes[str(seed)] = sorted(hashes)
    if any(len(values) != 1 for values in tape_hashes.values()):
        write_json_atomic(output / "result.json", {
            "status": "STOP_EXOGENOUS_CRN_MISMATCH",
            "risk_tape_hashes": tape_hashes,
            "timing_seeds_opened": True,
        })
        return 3

    low_rows = {int(probes[str(seed)]["seed"]): probes[str(seed)]["metrics"] for seed in seeds}
    best_constant = best_fixed_schedule(all_rows, {"constant"}, low_rows)
    best_open_loop = best_fixed_schedule(all_rows, {"open_loop_8week_periodic"}, low_rows)
    fixed_ids = {best_constant, best_open_loop, "observable_real"}
    comparators: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    nonobservable_comparators: list[dict[str, Any]] = []
    privileged_against_nonobservable: list[dict[str, Any]] = []
    observable_against_nonobservable: list[dict[str, Any]] = []
    selected_offsets: dict[str, str] = {}
    for seed in seeds:
        seed_rows = [row for row in all_rows if int(row["seed"]) == seed]
        base = low_rows[seed]
        comparator_pool = [
            row for row in seed_rows
            if row["schedule_id"] in fixed_ids or row["family"] == "weekly_privileged"
        ]
        comparator_pool = [row for row in comparator_pool if safe_against(row["metrics"], base)]
        comparator = max(comparator_pool, key=lambda row: row["metrics"]["ret_excel"])
        nonobservable_pool = [
            row for row in comparator_pool if row["schedule_id"] != "observable_real"
        ]
        nonobservable = max(
            nonobservable_pool, key=lambda row: row["metrics"]["ret_excel"]
        )
        candidate_pool = [
            row for row in seed_rows
            if row["family"] == "restricted_privileged"
            and safe_against(row["metrics"], comparator["metrics"])
        ]
        candidate = (
            max(candidate_pool, key=lambda row: row["metrics"]["ret_excel"])
            if candidate_pool else comparator
        )
        comparators.append(comparator["metrics"])
        candidates.append(candidate["metrics"])
        nonobservable_comparators.append(nonobservable["metrics"])
        privileged_pool = [
            row for row in seed_rows
            if row["family"] == "restricted_privileged"
            and safe_against(row["metrics"], nonobservable["metrics"])
        ]
        privileged_against_nonobservable.append(
            max(privileged_pool, key=lambda row: row["metrics"]["ret_excel"])["metrics"]
            if privileged_pool else nonobservable["metrics"]
        )
        observable_row = next(
            row for row in seed_rows if row["schedule_id"] == "observable_real"
        )
        observable_against_nonobservable.append(
            observable_row["metrics"]
            if safe_against(observable_row["metrics"], nonobservable["metrics"])
            else nonobservable["metrics"]
        )
        selected_offsets[str(seed)] = candidate["schedule_id"]

    promotion = paired_promotion_summary(candidates, comparators)
    ceiling_deltas = np.asarray([
        float(candidate["ret_excel"]) - float(comparator["ret_excel"])
        for candidate, comparator in zip(
            privileged_against_nonobservable, nonobservable_comparators
        )
    ])
    observable_deltas = np.asarray([
        float(candidate["ret_excel"]) - float(comparator["ret_excel"])
        for candidate, comparator in zip(
            observable_against_nonobservable, nonobservable_comparators
        )
    ])
    mean_ceiling = float(np.mean(ceiling_deltas))
    mean_observable = float(np.mean(observable_deltas))
    eta_timing = (
        mean_observable / mean_ceiling if mean_ceiling > 1e-12 else None
    )
    observable_conversion = {
        "development_only": True,
        "confirmed_H_obs": False,
        "restricted_ceiling_increment": mean_ceiling,
        "observable_ewma_increment": mean_observable,
        "eta_timing": eta_timing,
        "favorable_observable_tapes": int(np.sum(observable_deltas > 0.0)),
    }
    placebo_results = {}
    real = {
        int(row["seed"]): row["metrics"]
        for row in all_rows if row["schedule_id"] == "observable_real"
    }
    for family in ("stale_168h", "shuffled_within_tape", "cross_tape_shift17"):
        placebo = {
            int(row["seed"]): row["metrics"]
            for row in all_rows if row["schedule_id"] == f"observable_{family}"
        }
        placebo_results[family] = paired_promotion_summary(
            [real[seed] for seed in seeds],
            [placebo[seed] for seed in seeds],
        )

    observable_hashes = sorted({
        real[seed]["action_trajectory_sha256"] for seed in seeds
    })
    periodic_hashes = sorted({
        row["metrics"]["action_trajectory_sha256"]
        for row in all_rows if row["family"] == "open_loop_8week_periodic"
    })
    feedback_trajectory_audit = {
        "observable_unique_action_trajectory_hashes": len(observable_hashes),
        "observable_hashes": observable_hashes,
        "matches_any_open_loop_hash": any(
            value in set(periodic_hashes) for value in observable_hashes
        ),
        "state_dependent_development_signal": len(observable_hashes) >= 2,
        "claim_limit": "burned-tape diagnostic only; OOS feedback certification remains required"
    }

    result = {
        "schema_version": "restricted_pi_timing_ceiling_result_v1",
        "status": (
            "PASS_RESTRICTED_PI_TIMING_CEILING"
            if promotion["promotion_pass"] else "STOP_RESTRICTED_PI_TIMING_CEILING"
        ),
        "contract_sha256": sha256(contract_path),
        "source_risk_result_sha256": sha256(risk_result_path),
        "source_risk_raw_rows_sha256": sha256(risk_raw_path),
        "selected_profile": selected_profile,
        "low_posture": low.label,
        "high_posture": high.label,
        "best_constant": best_constant,
        "best_open_loop": best_open_loop,
        "promotion": promotion,
        "observable_conversion": observable_conversion,
        "placebo_results": placebo_results,
        "feedback_trajectory_audit": feedback_trajectory_audit,
        "selected_privileged_schedule_by_seed": selected_offsets,
        "risk_tape_hashes": tape_hashes,
        "timing_seeds_opened": True,
        "claim_boundary": {
            "restricted_PI_timing_ceiling_established": promotion["promotion_pass"],
            "H_PI_established": False,
            "H_obs_established": False,
            "learner_authorized": False,
            "paper2_confirmed": False,
            "paper3_authorized": False
        }
    }
    write_json_atomic(output / "result.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
