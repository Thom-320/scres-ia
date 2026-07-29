#!/usr/bin/env python3
"""Program L Gate 2: deterministic prefix replay and observable branching.

The script consumes a completed Gate-1 calibration package.  It never clones a
live SimPy environment.  Each branch reconstructs the episode, replays the same
action prefix, asserts raw-state identity, and then evaluates S1/S2/S3 under the
same exogenous seed/tape.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.l_program_env import (  # noqa: E402
    BUFFER_LEVELS,
    OBSERVATION_FIELDS,
    CampaignTape,
    GarridoLearningEnv,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def best_static_shift_by_buffer(summary: list[dict[str, str]]) -> dict[int, int]:
    result: dict[int, int] = {}
    for buffer_level in BUFFER_LEVELS:
        eligible = [
            row
            for row in summary
            if str(row["policy"]).startswith("static_")
            and int(row["buffer_level"]) == buffer_level
        ]
        if len(eligible) != 3:
            raise ValueError(f"Expected three static shifts for buffer {buffer_level}.")
        winner = max(eligible, key=lambda row: float(row["ret_excel_mean"]))
        result[buffer_level] = int(winner.get("policy_shift", winner["initial_shift"]))
    return result


def replay_branch(
    *,
    tape: CampaignTape,
    buffer_level: int,
    reference_shift: int,
    decision_week: int,
    branch_action: int,
    branch_weeks: int,
) -> tuple[list[float], dict[str, Any], dict[str, Any], float]:
    env = GarridoLearningEnv(
        max_steps=tape.horizon_weeks,
        buffer_level=buffer_level,
    )
    try:
        _obs, info = env.reset(
            seed=tape.base_seed,
            options={
                "campaign_tape": tape,
                "buffer_level": buffer_level,
                "initial_state_seed": tape.base_seed,
                "initial_shift": 1,
            },
        )
        for _ in range(decision_week):
            _obs, _reward, term, trunc, info = env.step(reference_shift - 1)
            if term or trunc:
                raise RuntimeError("Decision week lies after episode termination.")
        before = env.audit_state()
        final_ret = float(info.get("ret_excel", 0.0))
        for local_week in range(branch_weeks):
            action = branch_action if local_week == 0 else reference_shift - 1
            _obs, _reward, term, trunc, info = env.step(action)
            final_ret = float(info["ret_excel"])
            if term or trunc:
                break
        after = env.audit_state()
        return list(before["raw_observation"]), before, after, final_ret
    finally:
        env.close()


def branch_rows(
    *,
    tapes: list[CampaignTape],
    static_shifts: dict[int, int],
    states_per_buffer: int,
    branch_weeks: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_tape = max(1, math.ceil(states_per_buffer / len(tapes)))
    for buffer_level in BUFFER_LEVELS:
        collected = 0
        reference_shift = static_shifts[buffer_level]
        for tape in tapes:
            if collected >= states_per_buffer:
                break
            last_start = tape.horizon_weeks - branch_weeks
            if last_start <= 1:
                raise ValueError("Branching requires horizon_weeks > branch_weeks + 1.")
            weeks = np.linspace(1, last_start - 1, per_tape, dtype=int)
            for decision_week in sorted(set(int(v) for v in weeks)):
                if collected >= states_per_buffer:
                    break
                expected_raw: np.ndarray | None = None
                state_id = f"I{buffer_level}:{tape.campaign_id}:w{decision_week}"
                for branch_action in (0, 1, 2):
                    raw, before, after, final_ret = replay_branch(
                        tape=tape,
                        buffer_level=buffer_level,
                        reference_shift=reference_shift,
                        decision_week=decision_week,
                        branch_action=branch_action,
                        branch_weeks=branch_weeks,
                    )
                    raw_arr = np.asarray(raw, dtype=np.float64)
                    if expected_raw is None:
                        expected_raw = raw_arr
                    else:
                        np.testing.assert_array_equal(expected_raw, raw_arr)
                    row = {
                        "state_id": state_id,
                        "campaign_id": tape.campaign_id,
                        "campaign_sha256": tape.digest(),
                        "family": tape.family,
                        "risk_level": tape.risk_level,
                        "buffer_level": buffer_level,
                        "reference_shift": reference_shift,
                        "decision_week": decision_week,
                        "branch_action": branch_action,
                        "branch_shift": branch_action + 1,
                        "late_backlog_hours_increment": float(
                            after["reward_totals"]["late_backlog_hours"]
                            - before["reward_totals"]["late_backlog_hours"]
                        ),
                        "total_backlog_hours_increment": float(
                            after["reward_totals"]["total_backlog_hours"]
                            - before["reward_totals"]["total_backlog_hours"]
                        ),
                        "shift_hours_increment": float(
                            after["resource_totals"]["shift_hours"]
                            - before["resource_totals"]["shift_hours"]
                        ),
                        "extra_shift_hours_increment": float(
                            after["resource_totals"]["extra_shift_hours"]
                            - before["resource_totals"]["extra_shift_hours"]
                        ),
                        "ret_excel_final": final_ret,
                    }
                    row.update(
                        {
                            f"obs_{field}": float(value)
                            for field, value in zip(
                                OBSERVATION_FIELDS, raw, strict=True
                            )
                        }
                    )
                    rows.append(row)
                collected += 1
        if collected != states_per_buffer:
            raise RuntimeError(
                f"Collected {collected}, expected {states_per_buffer}, for I{buffer_level}."
            )
        print(
            f"[gate2] completed I{buffer_level}: {collected} states / "
            f"{collected * 3} branches",
            flush=True,
        )
    return rows


def label_states(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, float]]:
    by_state: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_state.setdefault(str(row["state_id"]), []).append(row)
    labelled = []
    lambdas: dict[int, float] = {}
    for buffer_level in BUFFER_LEVELS:
        states = [
            branches
            for branches in by_state.values()
            if int(branches[0]["buffer_level"]) == buffer_level
        ]
        loss_values = [
            float(row["late_backlog_hours_increment"])
            for branches in states
            for row in branches
        ]
        shift_values = [
            float(row["shift_hours_increment"])
            for branches in states
            for row in branches
        ]
        positive_loss = [value for value in loss_values if value > 0.0]
        scale = (
            float(np.median(positive_loss)) / max(float(np.median(shift_values)), 1.0)
            if positive_loss else 1.0
        )
        candidates = np.concatenate(
            ([0.0], np.geomspace(max(scale * 1e-4, 1e-12), scale * 1e4, 161))
        )
        target_shift = float(
            np.mean(
                [
                    float(
                        next(
                            row
                            for row in branches
                            if int(row["branch_action"])
                            == int(row["reference_shift"]) - 1
                        )["shift_hours_increment"]
                    )
                    for branches in states
                ]
            )
        )

        def winners_for(lambda_shift: float) -> list[dict[str, Any]]:
            return [
                min(
                    branches,
                    key=lambda row: (
                        float(row["late_backlog_hours_increment"])
                        + lambda_shift * float(row["shift_hours_increment"]),
                        float(row["total_backlog_hours_increment"]),
                        int(row["branch_action"]),
                    ),
                )
                for branches in states
            ]

        scored = []
        for candidate in candidates:
            winners = winners_for(float(candidate))
            mean_shift = float(
                np.mean([float(row["shift_hours_increment"]) for row in winners])
            )
            mean_loss = float(
                np.mean([float(row["late_backlog_hours_increment"]) for row in winners])
            )
            scored.append((abs(mean_shift - target_shift), mean_loss, float(candidate), winners))
        _gap, _loss, chosen_lambda, chosen_winners = min(
            scored, key=lambda item: (item[0], item[1], item[2])
        )
        lambdas[buffer_level] = chosen_lambda
        for branches, winner in zip(states, chosen_winners, strict=True):
            base = branches[0]
            labelled.append(
                {
                    "state_id": base["state_id"],
                    "campaign_id": base["campaign_id"],
                    "buffer_level": buffer_level,
                    "reference_action": int(base["reference_shift"]) - 1,
                    "optimal_action": int(winner["branch_action"]),
                    "budget_lambda": chosen_lambda,
                    **{
                        f"obs_{field}": float(base[f"obs_{field}"])
                        for field in OBSERVATION_FIELDS
                    },
                }
            )
    return labelled, lambdas


def clustered_ci(values: dict[str, float], *, seed: int = 0) -> tuple[float, float]:
    keys = sorted(values)
    arr = np.asarray([values[key] for key in keys], dtype=np.float64)
    if arr.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(5000, dtype=np.float64)
    for index in range(len(boots)):
        sample = rng.integers(0, arr.size, arr.size)
        boots[index] = float(arr[sample].mean())
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def crossfit(
    rows: list[dict[str, Any]], labelled: list[dict[str, Any]], folds: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    branch_lookup = {
        (str(row["state_id"]), int(row["branch_action"])): row for row in rows
    }
    predictions: list[dict[str, Any]] = []
    verdicts: dict[str, Any] = {}
    for buffer_level in BUFFER_LEVELS:
        data = [row for row in labelled if int(row["buffer_level"]) == buffer_level]
        x = np.asarray(
            [[row[f"obs_{field}"] for field in OBSERVATION_FIELDS] for row in data],
            dtype=np.float64,
        )
        y = np.asarray([row["optimal_action"] for row in data], dtype=np.int64)
        groups = np.asarray([row["campaign_id"] for row in data])
        n_splits = min(int(folds), len(set(groups)))
        if n_splits < 2:
            raise ValueError("Cross-fitting requires at least two distinct tapes.")
        predicted = np.empty_like(y)
        for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(x, y, groups):
            tree = DecisionTreeClassifier(max_depth=4, random_state=0)
            tree.fit(x[train_idx], y[train_idx])
            predicted[test_idx] = tree.predict(x[test_idx])

        tape_values: dict[str, dict[str, float]] = {}
        optimal_counts = {action: int((y == action).sum()) for action in (0, 1, 2)}
        for item, prediction, truth in zip(data, predicted, y, strict=True):
            state_id = str(item["state_id"])
            reference = int(item["reference_action"])
            chosen = branch_lookup[(state_id, int(prediction))]
            baseline = branch_lookup[(state_id, reference)]
            campaign_id = str(item["campaign_id"])
            acc = tape_values.setdefault(
                campaign_id,
                {"pred_loss": 0.0, "base_loss": 0.0, "pred_shift": 0.0,
                 "base_shift": 0.0, "ret_relative": 0.0, "n": 0.0},
            )
            acc["pred_loss"] += float(chosen["late_backlog_hours_increment"])
            acc["base_loss"] += float(baseline["late_backlog_hours_increment"])
            acc["pred_shift"] += float(chosen["shift_hours_increment"])
            acc["base_shift"] += float(baseline["shift_hours_increment"])
            baseline_ret = float(baseline["ret_excel_final"])
            acc["ret_relative"] += (
                float(chosen["ret_excel_final"]) / baseline_ret - 1.0
                if baseline_ret > 0.0 else 0.0
            )
            acc["n"] += 1.0
            predictions.append(
                {
                    "state_id": state_id,
                    "campaign_id": campaign_id,
                    "buffer_level": buffer_level,
                    "optimal_action": int(truth),
                    "predicted_action": int(prediction),
                    "reference_action": reference,
                }
            )
        reductions: dict[str, float] = {}
        resource_deltas: dict[str, float] = {}
        ret_relative: dict[str, float] = {}
        for campaign_id, acc in tape_values.items():
            reductions[campaign_id] = (
                1.0 - acc["pred_loss"] / acc["base_loss"]
                if acc["base_loss"] > 0.0 else 0.0
            )
            resource_deltas[campaign_id] = (
                acc["pred_shift"] / acc["base_shift"] - 1.0
                if acc["base_shift"] > 0.0 else float("inf")
            )
            ret_relative[campaign_id] = acc["ret_relative"] / max(acc["n"], 1.0)
        reduction_mean = float(np.mean(list(reductions.values())))
        reduction_ci = clustered_ci(reductions)
        resource_mean = float(np.mean(list(resource_deltas.values())))
        ret_mean = float(np.mean(list(ret_relative.values())))
        action_fractions = {
            f"S{action + 1}": optimal_counts[action] / max(len(data), 1)
            for action in (0, 1, 2)
        }
        varied = sum(fraction >= 0.10 for fraction in action_fractions.values()) >= 2
        passed = bool(
            buffer_level not in {0, 1344}
            and reduction_mean >= 0.05
            and reduction_ci[0] >= 0.05
            and abs(resource_mean) <= 0.02
            and ret_mean >= -0.01
            and varied
        )
        verdicts[str(buffer_level)] = {
            "service_loss_reduction_mean": reduction_mean,
            "service_loss_reduction_ci95": list(reduction_ci),
            "shift_hours_relative_delta": resource_mean,
            "ret_excel_relative_delta": ret_mean,
            "optimal_action_fractions": action_fractions,
            "two_actions_at_least_10pct": varied,
            "passed": passed,
        }
    promoted = sum(bool(row["passed"]) for row in verdicts.values()) >= 2
    return predictions, {
        "buffers": verdicts,
        "promotion_rule": "at least two non-extreme buffers pass all criteria",
        "promoted_to_ppo": promoted,
        "verdict": "PROMOTE_TO_PPO" if promoted else "STOP_NO_DEPLOYABLE_ADAPTIVE_HEADROOM",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--states-per-buffer", type=int, default=100)
    parser.add_argument("--branch-weeks", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.gate1_dir / "manifest.json").read_text())
    if manifest.get("split") != "calibration":
        raise ValueError("Gate 2 accepts calibration tapes only.")
    tapes = [CampaignTape.from_mapping(row) for row in manifest["tapes"]]
    summary = read_csv(args.gate1_dir / "static_18_policy_summary.csv")
    best_ret_static_shifts = best_static_shift_by_buffer(summary)
    # Continue every branch with the calibration-selected best fixed shift for
    # that buffer.  This is the deployable comparator specified by Gate 2; the
    # perfect-information labels remain diagnostic and are cross-fitted by tape.
    budget_reference_shifts = dict(best_ret_static_shifts)
    rows = branch_rows(
        tapes=tapes,
        static_shifts=budget_reference_shifts,
        states_per_buffer=args.states_per_buffer,
        branch_weeks=args.branch_weeks,
    )
    labelled, budget_lambdas = label_states(rows)
    predictions, verdict = crossfit(rows, labelled, args.folds)
    write_csv(args.output_dir / "branch_rows.csv", rows)
    write_csv(args.output_dir / "branch_states.csv", labelled)
    write_csv(args.output_dir / "crossfit_predictions.csv", predictions)
    payload = {
        "kind": "l_program_gate2_branching",
        "contract_id": "garrido_learning_v1",
        "gate1_manifest": str(args.gate1_dir / "manifest.json"),
        "states_per_buffer": args.states_per_buffer,
        "branch_weeks": args.branch_weeks,
        "tree_max_depth": 4,
        "fold_group": "campaign_id",
        "best_ret_static_shift_by_buffer": best_ret_static_shifts,
        "budget_reference_shift_by_buffer": budget_reference_shifts,
        "budget_label_lambda_by_buffer": budget_lambdas,
        **verdict,
    }
    (args.output_dir / "verdict.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
