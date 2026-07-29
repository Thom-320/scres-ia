#!/usr/bin/env python3
"""Evaluate Program O's frozen finite state-rich family on burned fit tapes.

The script is deliberately unable to generate tapes.  It consumes only the
retrieved `7420001--7420048` skeletons and complete 4^8 raw matrices from the
custody-verified label-only fit run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supply_chain.program_o_full_des_transducer import MATRIX_KEYS  # noqa: E402
from supply_chain.program_o_hobs import calendar_index  # noqa: E402
from supply_chain.program_o_state_rich import (  # noqa: E402
    StateRichConfiguration,
    StateRichDecision,
    add_minority_backlog,
    choose_state_rich_action,
    decision_rows,
    finite_state_rich_configurations,
    state_rich_calendar,
    swap_product_channels,
)


DEFAULT_CONTRACT = ROOT / "contracts/program_o_state_rich_comparator_fit_v1.json"
DEFAULT_PARENT_RUN = (
    ROOT / "outputs/program_o_runs/program-o-hobs-fit-v1-20260715/artifacts/fit"
)
PARENT_HOBS_CONTRACT = ROOT / "contracts/program_o_hobs_prelearner_v1.json"
PARENT_FULL_DES_CONTRACT = ROOT / "contracts/program_o_full_des_hpi_translation_v1.json"

HIGHER_KEYS = (
    "ration_ret_visible",
    "ret_full",
    "quantity_ret_full",
    "ret_visible_cvar10",
    "worst_product_fill",
)
LOWER_KEYS = (
    "omitted_rows",
    "omitted_quantity",
    "lost_orders",
    "lost_quantity",
    "unresolved_orders",
    "unresolved_quantity",
    "max_backlog_age",
    "service_loss_auc",
)
RESERVED_KEYS = (
    "gross_policy_batch_slots",
    "gross_production_quantity",
    "charged_daily_dispatch_slots",
    "charged_downstream_vehicle_hours",
)
ACTUAL_RESOURCE_KEYS = (
    "actual_loaded_departures",
    "actual_payload",
    "actual_downstream_vehicle_hours",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def primary_scheduler() -> Mapping[str, Sequence[str]]:
    parent = json.loads(PARENT_FULL_DES_CONTRACT.read_text())
    scheduler_id = str(parent["action"]["primary_scheduler"])
    return parent["action"]["within_week_schedulers"][scheduler_id]


def load_panel(parent_run: Path, cell_id: str, seeds: Sequence[int]) -> dict[str, np.ndarray]:
    values = {key: [] for key in MATRIX_KEYS}
    for seed in seeds:
        path = parent_run / "raw_calendar_matrix" / cell_id / f"tape_{seed}.npz"
        with np.load(path) as shard:
            if tuple(shard.files) != MATRIX_KEYS:
                raise AssertionError(f"matrix schema drift: {cell_id}/{seed}")
            for key in MATRIX_KEYS:
                values[key].append(np.asarray(shard[key]))
    return {key: np.stack(rows) for key, rows in values.items()}


def action_audit(calendars: Sequence[Sequence[int]]) -> dict[str, Any]:
    sequences = Counter(tuple(map(int, row)) for row in calendars)
    counts = Counter(int(action) for row in calendars for action in row)
    weeks = len(calendars[0])
    varying = sum(len({int(row[week]) for row in calendars}) > 1 for week in range(weeks))
    total_actions = len(calendars) * weeks
    material = sum(count >= 0.10 * total_actions for count in counts.values())
    passed = (
        len(sequences) >= 8
        and max(sequences.values()) / len(calendars) <= 0.50
        and varying >= 2
        and material >= 2
    )
    return {
        "unique_sequences": len(sequences),
        "modal_sequence_fraction": max(sequences.values()) / len(calendars),
        "varying_week_indices": varying,
        "action_counts": {str(key): int(value) for key, value in sorted(counts.items())},
        "material_action_levels_at_10pct": material,
        "passed": bool(passed),
    }


def state_counterfactual_audit(
    *,
    decisions_by_tape: Sequence[Sequence[StateRichDecision]],
    config: StateRichConfiguration,
    scheduler: Mapping[str, Sequence[str]],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    """Certify direct state dependence with frozen label and backlog interventions."""
    swap_tested = 0
    swap_passed = 0
    minority_tested = 0
    minority_passed = 0
    failures: list[dict[str, Any]] = []
    scheduler_counts = {
        int(action): sum(label == "P_C" for label in labels)
        for action, labels in scheduler.items()
    }
    for tape_index, decisions in enumerate(decisions_by_tape):
        for decision in decisions:
            if len(decision.tied_actions) != 1:
                continue
            original_action = int(decision.action)
            swapped = swap_product_channels(decision.observation)
            swapped_action, _objective, swapped_ties = choose_state_rich_action(
                swapped,
                config,
                scheduler=scheduler,
                regime_persistence=float(model["regime_persistence"]),
                dominant_share=float(model["dominant_product_share"]),
            )
            swap_tested += 1
            expected_swap = 3 - original_action
            if int(swapped_action) == expected_swap and len(swapped_ties) == 1:
                swap_passed += 1
            elif len(failures) < 20:
                failures.append(
                    {
                        "kind": "product_channel_swap",
                        "tape_index": tape_index,
                        "week": int(decision.observation.week),
                        "original_action": original_action,
                        "counterfactual_action": int(swapped_action),
                        "expected_action": expected_swap,
                        "counterfactual_ties": list(map(int, swapped_ties)),
                    }
                )

            pressured = add_minority_backlog(
                decision.observation,
                action=original_action,
                scheduler=scheduler,
            )
            pressured_action, _objective, _ties = choose_state_rich_action(
                pressured,
                config,
                scheduler=scheduler,
                regime_persistence=float(model["regime_persistence"]),
                dominant_share=float(model["dominant_product_share"]),
            )
            original_c = scheduler_counts[original_action]
            pressured_c = scheduler_counts[int(pressured_action)]
            minority_tested += 1
            monotone = (
                pressured_c >= original_c
                if original_c < 1.5
                else pressured_c <= original_c
            )
            if monotone:
                minority_passed += 1
            elif len(failures) < 20:
                failures.append(
                    {
                        "kind": "minority_backlog_monotonicity",
                        "tape_index": tape_index,
                        "week": int(decision.observation.week),
                        "original_action": original_action,
                        "counterfactual_action": int(pressured_action),
                    }
                )
    passed = (
        swap_tested > 0
        and swap_passed == swap_tested
        and minority_tested > 0
        and minority_passed == minority_tested
    )
    return {
        "product_channel_swap": {
            "tested": swap_tested,
            "passed_count": swap_passed,
        },
        "minority_backlog_monotonicity": {
            "tested": minority_tested,
            "passed_count": minority_passed,
        },
        "failures_truncated": failures,
        "passed": bool(passed),
    }


def paired_bootstrap_lcb95(values: np.ndarray, *, identity: str) -> float:
    """Deterministic one-sided paired bootstrap lower confidence bound."""
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not len(vector):
        raise ValueError("paired bootstrap requires a non-empty vector")
    seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(vector), size=(10_000, len(vector)))
    return float(np.quantile(vector[indices].mean(axis=1), 0.05))


def evaluate_information_placebos(
    *,
    config: StateRichConfiguration,
    cell_id: str,
    skeletons: Sequence[Mapping[str, Any]],
    panel: Mapping[str, np.ndarray],
    scheduler: Mapping[str, Sequence[str]],
    model: Mapping[str, Any],
    real_calendar_indices: Sequence[int],
) -> dict[str, Any]:
    """Evaluate the four frozen state-information placebos on paired tapes."""
    tape_rows = np.arange(len(skeletons), dtype=np.int64)
    real = panel["ret_visible"][
        tape_rows, np.asarray(real_calendar_indices, dtype=np.int64)
    ]
    modes = (
        "stale_t2",
        "no_state",
        "swapped_state",
        "cross_tape_shift17",
    )
    rows: dict[str, Any] = {}
    for mode in modes:
        calendars = []
        for position, skeleton in enumerate(skeletons):
            source = (
                skeletons[(position + 17) % len(skeletons)]
                if mode == "cross_tape_shift17"
                else skeleton
            )
            calendar, _decisions = state_rich_calendar(
                skeleton=source,
                scheduler=scheduler,
                config=config,
                regime_persistence=float(model["regime_persistence"]),
                dominant_share=float(model["dominant_product_share"]),
                observation_mode=("real" if mode == "cross_tape_shift17" else mode),
            )
            calendars.append(list(calendar))
        indices = np.asarray([calendar_index(row) for row in calendars], dtype=np.int64)
        placebo = panel["ret_visible"][tape_rows, indices]
        contrast = real - placebo
        lcb = paired_bootstrap_lcb95(
            contrast,
            identity=(
                f"program-o-state-rich-placebo-v1:{config.config_id}:"
                f"{cell_id}:{mode}"
            ),
        )
        rows[mode] = {
            "mean_real_minus_placebo": float(contrast.mean()),
            "paired_lcb95": lcb,
            "favorable_tapes": int((contrast > 0.0).sum()),
            "unique_placebo_sequences": len({tuple(row) for row in calendars}),
            "calendar_indices": indices.tolist(),
            "passed": bool(lcb > 0.0),
        }
    return {
        "bootstrap_resamples": 10_000,
        "comparison": "real-state ReT minus placebo-state ReT on paired burned fit tapes",
        "placebos": rows,
        "passed": all(row["passed"] for row in rows.values()),
    }


def matched_resource_frontier(
    *,
    panel: Mapping[str, np.ndarray],
    policy_metrics: Mapping[str, np.ndarray],
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Select the best single static that resource-dominates on every tape."""
    eligible = np.ones(panel["ret_visible"].shape[1], dtype=bool)
    for key in ACTUAL_RESOURCE_KEYS:
        eligible &= np.all(
            panel[key] + float(tolerance) >= policy_metrics[key][:, None], axis=0
        )
    eligible_indices = np.flatnonzero(eligible)
    if not len(eligible_indices):
        return {
            "calendar_index": None,
            "eligible_calendar_count": 0,
            "mean_ret_visible": None,
            "policy_delta": None,
            "frontier_resource_means": None,
            "minimum_per_tape_resource_slack": None,
            "passed": False,
        }
    local = int(np.argmax(panel["ret_visible"].mean(axis=0)[eligible_indices]))
    index = int(eligible_indices[local])
    delta = float(
        policy_metrics["ret_visible"].mean()
        - panel["ret_visible"][:, index].mean()
    )
    return {
        "calendar_index": index,
        "eligible_calendar_count": int(len(eligible_indices)),
        "mean_ret_visible": float(panel["ret_visible"][:, index].mean()),
        "policy_delta": delta,
        "frontier_resource_means": {
            key: float(panel[key][:, index].mean()) for key in ACTUAL_RESOURCE_KEYS
        },
        "minimum_per_tape_resource_slack": {
            key: float(np.min(panel[key][:, index] - policy_metrics[key]))
            for key in ACTUAL_RESOURCE_KEYS
        },
        "passed": bool(delta >= 0.015),
    }


def evaluate_configuration(
    *,
    config: StateRichConfiguration,
    cell: Mapping[str, Any],
    seeds: Sequence[int],
    skeletons: Sequence[Mapping[str, Any]],
    panel: Mapping[str, np.ndarray],
    scheduler: Mapping[str, Sequence[str]],
    model: Mapping[str, Any],
) -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
    calendars = []
    audits = []
    decisions_by_tape = []
    for skeleton in skeletons:
        calendar, decisions = state_rich_calendar(
            skeleton=skeleton,
            scheduler=scheduler,
            config=config,
            regime_persistence=float(model["regime_persistence"]),
            dominant_share=float(model["dominant_product_share"]),
        )
        calendars.append(list(calendar))
        audits.append(decision_rows(decisions))
        decisions_by_tape.append(decisions)
    indices = np.asarray([calendar_index(row) for row in calendars], dtype=np.int64)
    tape_rows = np.arange(len(seeds), dtype=np.int64)
    static_index = int(np.argmax(panel["ret_visible"].mean(axis=0)))
    static_metrics = {key: panel[key][:, static_index] for key in MATRIX_KEYS}
    policy_metrics = {key: panel[key][tape_rows, indices] for key in MATRIX_KEYS}
    deltas = {key: policy_metrics[key] - static_metrics[key] for key in MATRIX_KEYS}
    means = {key: float(value.mean()) for key, value in deltas.items()}
    tolerance = 1e-12
    metric_guardrails = all(means[key] >= -tolerance for key in HIGHER_KEYS) and all(
        means[key] <= tolerance for key in LOWER_KEYS
    )
    reserved_equal = all(abs(means[key]) <= tolerance for key in RESERVED_KEYS)
    strict_actual_use = all(means[key] <= tolerance for key in ACTUAL_RESOURCE_KEYS)

    candidate_resource_mean = {
        key: float(policy_metrics[key].mean()) for key in ACTUAL_RESOURCE_KEYS
    }
    # One tape-independent calendar must resource-dominate on every paired
    # tape. A different matched calendar per tape would be clairvoyant.
    resource_frontier = matched_resource_frontier(
        panel=panel,
        policy_metrics=policy_metrics,
        tolerance=tolerance,
    )

    trajectory = action_audit(calendars)
    counterfactuals = state_counterfactual_audit(
        decisions_by_tape=decisions_by_tape,
        config=config,
        scheduler=scheduler,
        model=model,
    )
    pre_placebo_rule_pass = (
        float(means["ret_visible"]) >= 0.015
        and int((deltas["ret_visible"] > 0.0).sum()) >= 34
        and bool(metric_guardrails)
        and bool(reserved_equal)
        and bool(strict_actual_use)
        and bool(resource_frontier["passed"])
        and bool(trajectory["passed"])
        and bool(counterfactuals["passed"])
    )
    row = {
        "config_id": config.config_id,
        "policy_id": config.policy_id,
        "parameter": int(config.parameter),
        "cell_id": str(cell["id"]),
        "static_index": static_index,
        "static_mean_ret_visible": float(panel["ret_visible"][:, static_index].mean()),
        "policy_mean_ret_visible": float(policy_metrics["ret_visible"].mean()),
        "mean_delta_vs_full_frontier": means["ret_visible"],
        "favorable_tapes": int((deltas["ret_visible"] > 0.0).sum()),
        "calendar_indices": indices.tolist(),
        "calendars": calendars,
        "mean_deltas": means,
        "metric_guardrails_pass": bool(metric_guardrails),
        "reserved_capacity_equal": bool(reserved_equal),
        "strict_actual_use_pass": bool(strict_actual_use),
        "resource_frontier": {
            **resource_frontier,
            "policy_resource_means": candidate_resource_mean,
        },
        "action_trajectory": trajectory,
        "state_counterfactuals": counterfactuals,
        "pre_placebo_rule_pass": bool(pre_placebo_rule_pass),
        "information_placebos": None,
        "information_placebos_pass": False,
        "primary_rule_pass": False,
        "per_tape_ret_delta": deltas["ret_visible"].astype(float).tolist(),
    }
    return row, audits


def connected_component_pass(cell_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    passing = {
        cell_id
        for cell_id, row in cell_rows.items()
        if float(row["mean_delta_vs_full_frontier"]) >= 0.015
        and int(row["favorable_tapes"]) >= 34
        and bool(row["metric_guardrails_pass"])
        and bool(row["reserved_capacity_equal"])
        and bool(row["strict_actual_use_pass"])
        and bool(row["resource_frontier"]["passed"])
        and bool(row["action_trajectory"]["passed"])
        and bool(row["state_counterfactuals"]["passed"])
        and bool(row["information_placebos_pass"])
    }
    coordinates = {
        "rho75_share75": (75, 75),
        "rho75_share90": (75, 90),
        "rho90_share75": (90, 75),
        "rho90_share90": (90, 90),
    }
    adjacency = {
        cell: {
            other
            for other in passing
            if cell != other
            and sum(a != b for a, b in zip(coordinates[cell], coordinates[other])) == 1
        }
        for cell in passing
    }
    components = []
    unseen = set(passing)
    while unseen:
        root = unseen.pop()
        component = {root}
        frontier = [root]
        while frontier:
            node = frontier.pop()
            for neighbour in adjacency[node] - component:
                component.add(neighbour)
                unseen.discard(neighbour)
                frontier.append(neighbour)
        components.append(sorted(component))
    eligible = []
    for component in components:
        rhos = {coordinates[cell][0] for cell in component}
        shares = {coordinates[cell][1] for cell in component}
        if len(component) >= 3 and len(rhos) == 2 and len(shares) == 2:
            eligible.append(component)
    return {
        "passing_cells": sorted(passing),
        "components": sorted(components),
        "eligible_components": sorted(eligible),
        "passed": bool(eligible),
    }


def run(*, contract_path: Path, parent_run: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    contract = json.loads(contract_path.read_text())
    parent_contract = json.loads(PARENT_HOBS_CONTRACT.read_text())
    parent_result_path = parent_run / "result.json"
    parent_result = json.loads(parent_result_path.read_text())
    expected_seeds = list(range(7420001, 7420049))
    if list(parent_result.get("seeds", [])) != expected_seeds:
        raise RuntimeError("parent fit seed block mismatch")
    if contract["tape_governance"]["validation_sealed"] != [7420049, 7420096]:
        raise RuntimeError("sealed validation range drift")
    if any(seed >= 7420049 for seed in expected_seeds):
        raise RuntimeError("validation seed access attempted")
    expected_shards = {
        (str(row["cell_id"]), int(row["seed"])): str(row["matrix_sha256"])
        for row in parent_result["raw_shards"]
    }
    for (cell_id, seed), expected_hash in expected_shards.items():
        path = parent_run / "raw_calendar_matrix" / cell_id / f"tape_{seed}.npz"
        if sha256(path) != expected_hash:
            raise RuntimeError(f"parent matrix hash mismatch: {cell_id}/{seed}")

    scheduler = primary_scheduler()
    cells = list(parent_contract["stability_cells"])
    model = parent_contract["policy_model_parameters"]
    configs = finite_state_rich_configurations()
    if [config.config_id for config in configs] != [
        row["config_id"] for row in contract["finite_controller_family"]["configurations"]
    ]:
        raise RuntimeError("finite family differs from contract")

    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        output / "progress.json",
        {"status": "RUNNING", "completed_cells": 0, "total_cells": len(cells)},
    )
    rows_by_config: dict[str, dict[str, Any]] = {config.config_id: {} for config in configs}
    audits_by_config: dict[str, dict[str, Any]] = {config.config_id: {} for config in configs}
    for completed, cell in enumerate(cells, start=1):
        cell_id = str(cell["id"])
        skeletons = [
            json.loads((parent_run / "skeletons" / cell_id / f"tape_{seed}.json").read_text())
            for seed in expected_seeds
        ]
        panel = load_panel(parent_run, cell_id, expected_seeds)
        if panel["ret_visible"].shape != (48, 65536):
            raise AssertionError(f"incomplete full frontier: {cell_id}")
        for config in configs:
            row, audits = evaluate_configuration(
                config=config,
                cell=cell,
                seeds=expected_seeds,
                skeletons=skeletons,
                panel=panel,
                scheduler=scheduler,
                model=model,
            )
            rows_by_config[config.config_id][cell_id] = row
            audits_by_config[config.config_id][cell_id] = audits
        write_json_atomic(
            output / "progress.json",
            {
                "status": "RUNNING",
                "completed_cells": completed,
                "total_cells": len(cells),
                "last_cell": cell_id,
                "updated_at_utc": now_utc(),
            },
        )
        del panel

    primary_id = str(contract["selection"]["primary_cell"])
    placebo_candidates = [
        config
        for config in configs
        if rows_by_config[config.config_id][primary_id]["pre_placebo_rule_pass"]
    ]
    # Placebos are computed only for the finite configurations that clear all
    # non-placebo primary gates.  Every such configuration is tested, so a
    # failed first-ranked controller cannot hide a lower-ranked valid one.
    for cell in cells:
        cell_id = str(cell["id"])
        skeletons = [
            json.loads(
                (parent_run / "skeletons" / cell_id / f"tape_{seed}.json").read_text()
            )
            for seed in expected_seeds
        ]
        panel = load_panel(parent_run, cell_id, expected_seeds)
        for config in placebo_candidates:
            row = rows_by_config[config.config_id][cell_id]
            placebo = evaluate_information_placebos(
                config=config,
                cell_id=cell_id,
                skeletons=skeletons,
                panel=panel,
                scheduler=scheduler,
                model=model,
                real_calendar_indices=row["calendar_indices"],
            )
            row["information_placebos"] = placebo
            row["information_placebos_pass"] = bool(placebo["passed"])
            row["primary_rule_pass"] = bool(
                row["pre_placebo_rule_pass"] and row["information_placebos_pass"]
            )
        del panel

    primary_candidates = [
        config for config in configs if rows_by_config[config.config_id][primary_id]["primary_rule_pass"]
    ]
    selected_config = None
    selected_cells: dict[str, Any] = {}
    selected_audits: dict[str, Any] = {}
    stability = {"passing_cells": [], "components": [], "eligible_components": [], "passed": False}
    if primary_candidates:
        selected = min(
            primary_candidates,
            key=lambda config: (
                -float(rows_by_config[config.config_id][primary_id]["policy_mean_ret_visible"]),
                -float(rows_by_config[config.config_id][primary_id]["mean_deltas"]["worst_product_fill"]),
                -float(rows_by_config[config.config_id][primary_id]["mean_deltas"]["ret_visible_cvar10"]),
                float(rows_by_config[config.config_id][primary_id]["mean_deltas"]["max_backlog_age"]),
                [row.config_id for row in configs].index(config.config_id),
            ),
        )
        selected_config = {
            "config_id": selected.config_id,
            "policy_id": selected.policy_id,
            "parameter": int(selected.parameter),
        }
        selected_cells = rows_by_config[selected.config_id]
        selected_audits = audits_by_config[selected.config_id]
        stability = connected_component_pass(selected_cells)

    if not primary_candidates:
        effect_rows = [
            rows_by_config[config.config_id][primary_id]
            for config in configs
            if float(
                rows_by_config[config.config_id][primary_id][
                    "mean_delta_vs_full_frontier"
                ]
            )
            >= 0.015
            and int(rows_by_config[config.config_id][primary_id]["favorable_tapes"])
            >= 34
            and bool(
                rows_by_config[config.config_id][primary_id]["action_trajectory"][
                    "passed"
                ]
            )
        ]
        if any(
            row["metric_guardrails_pass"]
            and row["reserved_capacity_equal"]
            and row["strict_actual_use_pass"]
            and row["resource_frontier"]["passed"]
            and (
                not row["state_counterfactuals"]["passed"]
                or not row["information_placebos_pass"]
            )
            for row in effect_rows
        ):
            status = contract["terminal_labels"]["state_independent"]
        elif any(
            row["metric_guardrails_pass"]
            and row["reserved_capacity_equal"]
            and row["strict_actual_use_pass"]
            and not row["resource_frontier"]["passed"]
            for row in effect_rows
        ):
            status = contract["terminal_labels"]["resource_frontier"]
        elif any(
            not row["metric_guardrails_pass"]
            or not row["reserved_capacity_equal"]
            or not row["strict_actual_use_pass"]
            for row in effect_rows
        ):
            status = contract["terminal_labels"]["resource_or_guardrail"]
        else:
            status = contract["terminal_labels"]["no_policy"]
    elif not stability["passed"]:
        status = contract["terminal_labels"]["no_stability"]
    else:
        status = contract["terminal_labels"]["candidate"]

    result = {
        "schema_version": "program_o_state_rich_comparator_fit_result_v1",
        "generated_at_utc": now_utc(),
        "status": status,
        "scientific_commit": git_commit(),
        "contract": str(contract_path),
        "contract_sha256": sha256(contract_path),
        "parent_result": str(parent_result_path),
        "parent_result_sha256": sha256(parent_result_path),
        "seeds": expected_seeds,
        "validation_seed_accessed": False,
        "configurations": rows_by_config,
        "selected_config": selected_config,
        "selected_cells": selected_cells,
        "selected_decision_audits": selected_audits,
        "stability": stability,
        "claim_boundary": {
            "full_des_h_pi_established": True,
            "h_obs_established": False,
            "learner_authorized": False,
            "paper2_confirmed": False,
            "paper3_authorized": False,
        },
    }
    write_json_atomic(output / "result.json", result)
    write_json_atomic(
        output / "progress.json",
        {
            "status": "COMPLETE",
            "completed_cells": len(cells),
            "total_cells": len(cells),
            "result_sha256": sha256(output / "result.json"),
            "updated_at_utc": now_utc(),
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--parent-run", type=Path, default=DEFAULT_PARENT_RUN)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        contract_path=args.contract.resolve(),
        parent_run=args.parent_run.resolve(),
        output=args.output.resolve(),
    )
    print(args.output / "result.json")
    return 0 if result["status"].startswith(("PASS_", "STOP_")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
