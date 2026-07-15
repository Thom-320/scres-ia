#!/usr/bin/env python3
"""Post-STOP causal diagnostic for Program O's rho90/share90 validation cell."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.screen_program_o_hobs_fit import primary_scheduler  # noqa: E402
from scripts.screen_program_o_state_rich_fit import HIGHER_KEYS, LOWER_KEYS  # noqa: E402
from supply_chain.program_o_full_des_transducer import MATRIX_KEYS  # noqa: E402
from supply_chain.program_o_state_rich import (  # noqa: E402
    StateRichConfiguration,
    state_rich_calendar,
)


DEFAULT_CONTRACT = ROOT / "contracts/program_o_rho90_share90_causal_diagnostic_v1.json"
PARENT_FULL = ROOT / "contracts/program_o_full_des_hpi_translation_v1.json"
PARENT_STATE = ROOT / "contracts/program_o_state_rich_comparator_fit_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_calendar(index: int, weeks: int = 8) -> tuple[int, ...]:
    actions = [0] * weeks
    value = int(index)
    for position in range(weeks - 1, -1, -1):
        actions[position] = value % 4
        value //= 4
    if value:
        raise ValueError("calendar index outside frozen horizon")
    return tuple(actions)


def load_panel(validation_dir: Path, cell_id: str, seeds: list[int]) -> dict[str, np.ndarray]:
    columns: dict[str, list[np.ndarray]] = {key: [] for key in MATRIX_KEYS}
    for seed in seeds:
        path = validation_dir / "raw_calendar_matrix" / cell_id / f"tape_{seed}.npz"
        with np.load(path) as shard:
            for key in MATRIX_KEYS:
                columns[key].append(np.asarray(shard[key], dtype=float))
    return {key: np.stack(values) for key, values in columns.items()}


def load_skeleton(validation_dir: Path, cell_id: str, seed: int) -> dict[str, Any]:
    return json.loads((validation_dir / "skeletons" / cell_id / f"tape_{seed}.json").read_text())


def weekly_demand(skeleton: Mapping[str, Any]) -> list[dict[str, float]]:
    start = float(skeleton["decision_start"])
    rows = [dict(c_qty=0.0, h_qty=0.0, c_orders=0.0, h_orders=0.0) for _ in range(8)]
    for time, quantity, product in zip(
        skeleton["order_times"], skeleton["order_quantities"], skeleton["order_products"]
    ):
        week = int(np.floor((float(time) - start) / 168.0))
        week = min(7, max(0, week))
        prefix = "c" if product == "P_C" else "h"
        rows[week][f"{prefix}_qty"] += float(quantity)
        rows[week][f"{prefix}_orders"] += 1.0
    for row in rows:
        total = row["c_qty"] + row["h_qty"]
        row["c_share"] = row["c_qty"] / total if total else 0.5
    return rows


def action_c_share(action: int, scheduler: Mapping[str, Any]) -> float:
    labels = tuple(scheduler[str(int(action))])
    return sum(label == "P_C" for label in labels) / float(len(labels))


def corrected_max_t(
    *,
    panels: Mapping[str, Mapping[str, np.ndarray]],
    result: Mapping[str, Any],
    resamples: int,
) -> dict[str, Any]:
    """Studentized max-t audit; descriptive only and never used for promotion."""
    cells = list(result["cells"])
    n_tapes = 48
    seed = int.from_bytes(
        hashlib.sha256(b"program-o-rho90-share90-causal-diagnostic-v1").digest()[:8],
        "big",
    )
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, n_tapes, size=(resamples, n_tapes))
    counts = np.zeros((resamples, n_tapes), dtype=float)
    for position, sample in enumerate(samples):
        counts[position] = np.bincount(sample, minlength=n_tapes) / float(n_tapes)

    names: list[str] = []
    kinds: list[str] = []
    points: list[float] = []
    boots: list[np.ndarray] = []
    for cell_id in cells:
        panel = panels[cell_id]
        row = result["cells"][cell_id]
        policy_indices = np.asarray(row["calendar_indices"], dtype=np.int64)
        tape_rows = np.arange(n_tapes, dtype=np.int64)
        policy = {key: panel[key][tape_rows, policy_indices] for key in MATRIX_KEYS}
        selected = np.empty(resamples, dtype=np.int64)
        for start in range(0, resamples, 25):
            stop = min(resamples, start + 25)
            selected[start:stop] = np.argmax(counts[start:stop] @ panel["ret_visible"], axis=1)
        static_point = int(np.argmax(panel["ret_visible"].mean(axis=0)))
        signed = [("ret_visible", 1.0, "primary")]
        signed.extend((key, 1.0, "guardrail") for key in HIGHER_KEYS)
        signed.extend((key, -1.0, "guardrail") for key in LOWER_KEYS)
        for key, sign, kind in signed:
            comparator = panel[key][:, static_point]
            point = float(sign * (policy[key] - comparator).mean())
            policy_boot = counts @ policy[key]
            static_boot = np.asarray(
                [np.dot(counts[b], panel[key][:, selected[b]]) for b in range(resamples)]
            )
            names.append(f"{cell_id}::{kind}::{key}")
            kinds.append(kind)
            points.append(point)
            boots.append(sign * (policy_boot - static_boot))
        real = panel["ret_visible"][tape_rows, policy_indices]
        for family, block in result["placebos"][cell_id].items():
            for mode, placebo in block["placebos"].items():
                indices = np.asarray(placebo["calendar_indices"], dtype=np.int64)
                contrast = real - panel["ret_visible"][tape_rows, indices]
                names.append(f"{cell_id}::placebo::{family}::{mode}")
                kinds.append("placebo")
                points.append(float(contrast.mean()))
                boots.append(counts @ contrast)

    point = np.asarray(points)
    boot = np.column_stack(boots)
    standard_error = boot.std(axis=0, ddof=1)
    active = standard_error > 1e-15
    max_t = np.zeros(resamples, dtype=float)
    if np.any(active):
        standardized = (point[None, active] - boot[:, active]) / standard_error[active]
        max_t = np.max(standardized, axis=1)
    critical = float(np.quantile(max_t, 0.95))
    lcb = point.copy()
    lcb[active] = point[active] - critical * standard_error[active]
    estimates = {
        name: {
            "kind": kind,
            "estimate": float(point[index]),
            "bootstrap_se": float(standard_error[index]),
            "studentized_simultaneous_lcb95": float(lcb[index]),
        }
        for index, (name, kind) in enumerate(zip(names, kinds))
    }
    return {
        "method": "studentized_one_sided_max_t",
        "descriptive_only": True,
        "resamples": int(resamples),
        "estimand_count": len(names),
        "critical": critical,
        "estimates": estimates,
        "primary_all_lcb_ge_0_01": all(
            row["studentized_simultaneous_lcb95"] >= 0.01
            for row in estimates.values()
            if row["kind"] == "primary"
        ),
        "placebo_all_lcb_gt_zero": all(
            row["studentized_simultaneous_lcb95"] > 0.0
            for row in estimates.values()
            if row["kind"] == "placebo"
        ),
        "guardrail_all_lcb_ge_zero": all(
            row["studentized_simultaneous_lcb95"] >= 0.0
            for row in estimates.values()
            if row["kind"] == "guardrail"
        ),
    }


def mean_by_group(rows: list[dict[str, Any]], key: str, group: str) -> float | None:
    values = [float(row[key]) for row in rows if row["group"] == group]
    return float(np.mean(values)) if values else None


def run(contract_path: Path, validation_dir: Path, output: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text())
    result_path = validation_dir / "result.json"
    if sha256(result_path) != contract["source_result"]["sha256"]:
        raise RuntimeError("source result identity mismatch")
    result = json.loads(result_path.read_text())
    if result["status"] != contract["source_result"]["terminal_status"]:
        raise RuntimeError("source STOP status mismatch")
    seeds = list(range(7420049, 7420097))
    if result["seeds"] != seeds:
        raise RuntimeError("burned seed block mismatch")

    all_panels = {cell: load_panel(validation_dir, cell, seeds) for cell in result["cells"]}
    cell_id = contract["scope"]["cell"]
    panel = all_panels[cell_id]
    cell = result["cells"][cell_id]
    policy_indices = np.asarray(cell["calendar_indices"], dtype=np.int64)
    static_index = int(cell["static_index"])
    static_calendar = decode_calendar(static_index)
    scheduler = primary_scheduler(json.loads(PARENT_FULL.read_text()))
    model = json.loads(PARENT_STATE.read_text())["observation_contract"]["model_parameters"]
    config = StateRichConfiguration("belief_mpc", 3)

    tape_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    exact_mismatches: list[dict[str, Any]] = []
    for position, seed in enumerate(seeds):
        skeleton = load_skeleton(validation_dir, cell_id, seed)
        calendar, decisions = state_rich_calendar(
            skeleton=skeleton,
            scheduler=scheduler,
            config=config,
            regime_persistence=float(model["regime_persistence"]),
            dominant_share=float(model["dominant_product_share"]),
        )
        expected = tuple(cell["calendars"][position])
        if calendar != expected:
            exact_mismatches.append(
                {
                    "seed": seed,
                    "kind": "calendar",
                    "expected": list(expected),
                    "actual": list(calendar),
                }
            )
        demand = weekly_demand(skeleton)
        policy_index = int(policy_indices[position])
        policy_ret = float(panel["ret_visible"][position, policy_index])
        static_ret = float(panel["ret_visible"][position, static_index])
        delta = policy_ret - static_ret
        stored_delta = float(cell["per_tape_ret_delta"][position])
        if abs(delta - stored_delta) > 1e-12:
            exact_mismatches.append(
                {"seed": seed, "kind": "stored_delta", "expected": stored_delta, "actual": delta}
            )
        oracle_index = int(np.argmax(panel["ret_visible"][position]))
        oracle_ret = float(panel["ret_visible"][position, oracle_index])
        static_percentile = float(np.mean(panel["ret_visible"][position] <= static_ret))
        current_mismatch = []
        next_mismatch = []
        positions = []
        wrong_positions = []
        belief_errors = []
        for week, decision in enumerate(decisions):
            observation = decision.observation
            action_share = action_c_share(decision.action, scheduler)
            current_share = float(demand[week]["c_share"])
            next_share = float(demand[min(7, week + 1)]["c_share"])
            position_vector = (
                np.asarray(observation.on_hand)
                + np.asarray(observation.locked_pipeline)
                - np.asarray(observation.backlog_quantity)
            )
            predicted_majority = 0 if observation.predicted_share_c >= 0.5 else 1
            wrong_product = 1 - predicted_majority
            current_mismatch.append(abs(action_share - current_share))
            next_mismatch.append(abs(action_share - next_share))
            positions.append(float(position_vector[0] - position_vector[1]))
            wrong_positions.append(float(max(0.0, position_vector[wrong_product])))
            belief_errors.append(abs(float(observation.predicted_share_c) - current_share))
            decision_rows.append(
                {
                    "seed": seed,
                    "group": "favorable" if delta > 0 else "non_favorable",
                    "week": week,
                    "action": int(decision.action),
                    "action_c_share": action_share,
                    "objective": list(decision.objective),
                    "tied_actions": list(decision.tied_actions),
                    "demand_c_share_current_week": current_share,
                    "demand_c_share_next_week": next_share,
                    "current_week_mismatch": current_mismatch[-1],
                    "next_week_mismatch": next_mismatch[-1],
                    "inventory_position_c_minus_h": positions[-1],
                    "wrong_product_positive_position": wrong_positions[-1],
                    "belief_absolute_error_current_week": belief_errors[-1],
                    "observation": asdict(observation),
                }
            )
        weekly_majorities = [row["c_share"] >= 0.5 for row in demand]
        transitions = sum(a != b for a, b in zip(weekly_majorities, weekly_majorities[1:]))
        policy_quantity_ret = float(panel["quantity_ret_full"][position, policy_index])
        static_quantity_ret = float(panel["quantity_ret_full"][position, static_index])
        policy_worst_fill = float(panel["worst_product_fill"][position, policy_index])
        static_worst_fill = float(panel["worst_product_fill"][position, static_index])
        tape_rows.append(
            {
                "seed": seed,
                "group": "favorable" if delta > 0 else "non_favorable",
                "ret_delta": delta,
                "policy_ret": policy_ret,
                "static_ret": static_ret,
                "oracle_ret": oracle_ret,
                "policy_to_oracle_gap": oracle_ret - policy_ret,
                "static_calendar_percentile": static_percentile,
                "oracle_calendar_index": oracle_index,
                "policy_calendar_index": policy_index,
                "quantity_ret_delta": policy_quantity_ret - static_quantity_ret,
                "worst_product_fill_delta": policy_worst_fill - static_worst_fill,
                "omitted_quantity_delta": float(
                    panel["omitted_quantity"][position, policy_index]
                    - panel["omitted_quantity"][position, static_index]
                ),
                "max_backlog_age_delta": float(
                    panel["max_backlog_age"][position, policy_index]
                    - panel["max_backlog_age"][position, static_index]
                ),
                "ending_inventory_c_delta": float(
                    panel["ending_inventory_P_C"][position, policy_index]
                    - panel["ending_inventory_P_C"][position, static_index]
                ),
                "ending_inventory_h_delta": float(
                    panel["ending_inventory_P_H"][position, policy_index]
                    - panel["ending_inventory_P_H"][position, static_index]
                ),
                "regime_transitions": transitions,
                "first_half_c_share": float(np.mean([row["c_share"] for row in demand[:4]])),
                "second_half_c_share": float(np.mean([row["c_share"] for row in demand[4:]])),
                "phase_shift_absolute": abs(
                    float(np.mean([row["c_share"] for row in demand[:4]]))
                    - float(np.mean([row["c_share"] for row in demand[4:]]))
                ),
                "mean_current_week_action_mismatch": float(np.mean(current_mismatch)),
                "mean_next_week_action_mismatch": float(np.mean(next_mismatch)),
                "next_minus_current_mismatch": float(
                    np.mean(next_mismatch) - np.mean(current_mismatch)
                ),
                "mean_wrong_product_positive_position": float(np.mean(wrong_positions)),
                "max_wrong_product_positive_position": float(np.max(wrong_positions)),
                "mean_belief_absolute_error": float(np.mean(belief_errors)),
                "calendar": list(calendar),
                "static_calendar": list(static_calendar),
            }
        )

    stored_static = int(np.argmax(panel["ret_visible"].mean(axis=0)))
    if stored_static != static_index:
        exact_mismatches.append(
            {"kind": "static_index", "expected": static_index, "actual": stored_static}
        )
    favorable = [row for row in tape_rows if row["group"] == "favorable"]
    non_favorable = [row for row in tape_rows if row["group"] == "non_favorable"]
    feature_keys = [
        "ret_delta",
        "policy_to_oracle_gap",
        "static_calendar_percentile",
        "regime_transitions",
        "first_half_c_share",
        "second_half_c_share",
        "phase_shift_absolute",
        "mean_current_week_action_mismatch",
        "mean_next_week_action_mismatch",
        "next_minus_current_mismatch",
        "mean_wrong_product_positive_position",
        "max_wrong_product_positive_position",
        "mean_belief_absolute_error",
        "quantity_ret_delta",
        "worst_product_fill_delta",
        "omitted_quantity_delta",
        "max_backlog_age_delta",
    ]
    group_comparison = {
        key: {
            "favorable_mean": mean_by_group(tape_rows, key, "favorable"),
            "non_favorable_mean": mean_by_group(tape_rows, key, "non_favorable"),
            "non_favorable_minus_favorable": (
                mean_by_group(tape_rows, key, "non_favorable")
                - mean_by_group(tape_rows, key, "favorable")
            ),
        }
        for key in feature_keys
    }
    delta_vector = np.asarray([row["ret_delta"] for row in tape_rows])
    correlations = {}
    for key in feature_keys[1:]:
        values = np.asarray([row[key] for row in tape_rows], dtype=float)
        correlations[key] = (
            None if np.std(values) <= 1e-15 else float(np.corrcoef(delta_vector, values)[0, 1])
        )
    sorted_delta = np.sort(delta_vector)
    gaps = np.diff(sorted_delta)
    largest_gap_index = int(np.argmax(gaps)) if len(gaps) else 0
    corrected = corrected_max_t(
        panels=all_panels,
        result=result,
        resamples=int(contract["corrective_inference"]["bootstrap_resamples"]),
    )

    metric_sign_reversals = {
        "quantity_ret": int(
            sum(
                bool(
                    np.sign(row["ret_delta"]) != np.sign(row["quantity_ret_delta"])
                    and abs(row["quantity_ret_delta"]) > 1e-12
                )
                for row in tape_rows
            )
        ),
        "worst_product_fill": int(
            sum(
                bool(
                    np.sign(row["ret_delta"]) != np.sign(row["worst_product_fill_delta"])
                    and abs(row["worst_product_fill_delta"]) > 1e-12
                )
                for row in tape_rows
            )
        ),
    }
    technical_invalidation = bool(exact_mismatches)
    status = (
        "TECHNICAL_INVALIDATION_REQUIRES_CORRECTIVE_REPEAT"
        if technical_invalidation
        else "GENUINE_DECISIONAL_INSTABILITY_PROGRAM_O_STOP_STANDS"
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "tape_map.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tape_rows[0]))
        writer.writeheader()
        writer.writerows(tape_rows)
    with (output / "decision_map.jsonl").open("w") as handle:
        for row in decision_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    diagnostic = {
        "schema_version": "program_o_rho90_share90_causal_diagnostic_result_v1",
        "status": status,
        "source_result_sha256": sha256(result_path),
        "contract_sha256": sha256(contract_path),
        "stop_relabelled": False,
        "new_tapes_opened": False,
        "counts": {
            "favorable": len(favorable),
            "non_favorable": len(non_favorable),
            "non_favorable_seeds": [row["seed"] for row in non_favorable],
        },
        "implementation_and_metric_audit": {
            "exact_mismatches": exact_mismatches,
            "calendar_reconstruction_pass": not any(
                row.get("kind") == "calendar" for row in exact_mismatches
            ),
            "stored_delta_reconstruction_pass": not any(
                row.get("kind") == "stored_delta" for row in exact_mismatches
            ),
            "static_index_reconstruction_pass": stored_static == static_index,
            "physical_replay_failures_in_source": len(result["physical_replay"]["failures"]),
            "unique_scheduled_resource_vectors": result["physical_replay"][
                "unique_scheduled_resource_vectors"
            ],
            "metric_sign_reversals": metric_sign_reversals,
            "original_unstandardized_critical": result["inference"]["simultaneous_critical"],
        },
        "distribution": {
            "mean": float(np.mean(delta_vector)),
            "median": float(np.median(delta_vector)),
            "quantiles": {
                str(q): float(np.quantile(delta_vector, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
            },
            "largest_internal_gap": float(gaps[largest_gap_index]) if len(gaps) else 0.0,
            "largest_gap_bounds": (
                [float(sorted_delta[largest_gap_index]), float(sorted_delta[largest_gap_index + 1])]
                if len(gaps)
                else []
            ),
        },
        "group_comparison": group_comparison,
        "correlations_with_ret_delta": correlations,
        "corrected_inference_descriptive_only": corrected,
        "decision_rule": {
            "technical_invalidation": technical_invalidation,
            "corrective_repeat_licensed": technical_invalidation,
            "program_o_stop_stands": not technical_invalidation,
            "learner_authorized": False,
            "paper2_confirmed": False,
        },
    }
    (output / "result.json").write_text(json.dumps(diagnostic, indent=2, sort_keys=True) + "\n")
    return diagnostic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.contract, args.validation_dir, args.output)
    print(json.dumps({"status": result["status"], "counts": result["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
