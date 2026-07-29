#!/usr/bin/env python3
"""Run Program O's frozen post-hoc information/resource diagnostic.

This script cannot generate tapes. It consumes the stopped state-rich result plus
the burned 7420001--7420048 skeletons and complete open-loop matrices.
"""

from __future__ import annotations

import argparse
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

from scripts.screen_program_o_state_rich_fit import (  # noqa: E402
    load_panel,
    paired_bootstrap_lcb95,
    primary_scheduler,
)
from supply_chain.program_o_hobs import calendar_index  # noqa: E402
from supply_chain.program_o_state_rich import (  # noqa: E402
    StateRichConfiguration,
    finite_state_rich_configurations,
    state_rich_calendar,
)


DEFAULT_CONTRACT = (
    ROOT / "contracts/program_o_state_rich_dual_resource_diagnostic_v1.json"
)
DEFAULT_SOURCE_RESULT = (
    ROOT / "results/program_o/state_rich_comparator_fit_v1/result.json"
)
DEFAULT_PARENT_RUN = (
    ROOT / "outputs/program_o_runs/program-o-hobs-fit-v1-20260715/artifacts/fit"
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


def connected_components(
    passing_cells: Sequence[str], *, required_size: int
) -> dict[str, Any]:
    coordinates = {
        "rho75_share75": (75, 75),
        "rho75_share90": (75, 90),
        "rho90_share75": (90, 75),
        "rho90_share90": (90, 90),
    }
    passing = set(passing_cells)
    unknown = passing - set(coordinates)
    if unknown:
        raise AssertionError(f"unknown cells: {sorted(unknown)}")
    adjacency = {
        cell: {
            other
            for other in passing
            if other != cell
            and sum(
                left != right
                for left, right in zip(coordinates[cell], coordinates[other])
            )
            == 1
        }
        for cell in passing
    }
    components: list[list[str]] = []
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
    eligible = [
        component
        for component in components
        if len(component) >= required_size
        and len({coordinates[cell][0] for cell in component}) == 2
        and len({coordinates[cell][1] for cell in component}) == 2
    ]
    return {
        "passing_cells": sorted(passing),
        "components": sorted(components),
        "eligible_components": sorted(eligible),
        "passed": bool(eligible),
    }


def placebo_calendar_indices(
    *,
    mode: str,
    config: StateRichConfiguration,
    skeletons: Sequence[Mapping[str, Any]],
    scheduler: Mapping[str, Sequence[str]],
    model: Mapping[str, Any],
) -> np.ndarray:
    indices = []
    for position, skeleton in enumerate(skeletons):
        source = (
            skeletons[(position + 17) % len(skeletons)]
            if mode == "cross_tape_shift17"
            else skeleton
        )
        observation_mode = "real" if mode == "cross_tape_shift17" else mode
        calendar, _decisions = state_rich_calendar(
            skeleton=source,
            scheduler=scheduler,
            config=config,
            regime_persistence=float(model["regime_persistence"]),
            dominant_share=float(model["dominant_product_share"]),
            observation_mode=observation_mode,
        )
        indices.append(calendar_index(calendar))
    return np.asarray(indices, dtype=np.int64)


def evaluate_placebos(
    *,
    contract: Mapping[str, Any],
    config: StateRichConfiguration,
    cell_id: str,
    real_indices: Sequence[int],
    skeletons: Sequence[Mapping[str, Any]],
    panel: Mapping[str, np.ndarray],
    scheduler: Mapping[str, Sequence[str]],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    tape_rows = np.arange(len(skeletons), dtype=np.int64)
    real = panel["ret_visible"][
        tape_rows, np.asarray(real_indices, dtype=np.int64)
    ]
    family_rows: dict[str, Any] = {}
    for family, modes in contract["placebo_families"].items():
        rows: dict[str, Any] = {}
        for mode in modes:
            indices = placebo_calendar_indices(
                mode=str(mode),
                config=config,
                skeletons=skeletons,
                scheduler=scheduler,
                model=model,
            )
            placebo = panel["ret_visible"][tape_rows, indices]
            contrast = real - placebo
            lcb = paired_bootstrap_lcb95(
                contrast,
                identity=(
                    "program-o-state-rich-dual-resource-v1:"
                    f"{config.config_id}:{cell_id}:{mode}"
                ),
            )
            rows[str(mode)] = {
                "mean_real_minus_placebo": float(contrast.mean()),
                "paired_lcb95": float(lcb),
                "favorable_tapes": int((contrast > 0.0).sum()),
                "unique_placebo_sequences": int(len(set(indices.tolist()))),
                "calendar_indices": indices.tolist(),
                "passed": bool(lcb > 0.0),
            }
        family_rows[str(family)] = {
            "placebos": rows,
            "passed": bool(all(row["passed"] for row in rows.values())),
        }
    return family_rows


def nonresource_pass(row: Mapping[str, Any], contract: Mapping[str, Any]) -> bool:
    gates = contract["common_nonresource_gates"]
    return bool(
        float(row["mean_delta_vs_full_frontier"])
        >= float(gates["mean_ret_delta_minimum"])
        and int(row["favorable_tapes"]) >= int(gates["favorable_tapes_minimum"])
        and bool(row["metric_guardrails_pass"])
        and bool(row["reserved_capacity_equal"])
        and bool(row["action_trajectory"]["passed"])
        and bool(row["state_counterfactuals"]["passed"])
    )


def run(
    *,
    contract_path: Path,
    source_result_path: Path,
    parent_run: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    contract = json.loads(contract_path.read_text())
    source_result = json.loads(source_result_path.read_text())
    expected_source_sha = str(contract["source_result"]["sha256"])
    if sha256(source_result_path) != expected_source_sha:
        raise AssertionError("source state-rich result hash mismatch")
    if source_result["status"] != contract["source_result"]["status"]:
        raise AssertionError("source state-rich status mismatch")
    if bool(source_result["validation_seed_accessed"]):
        raise AssertionError("source result reports validation access")

    seed_min = int(contract["tape_governance"]["burned_fit_min"])
    seed_max = int(contract["tape_governance"]["burned_fit_max"])
    seeds = list(range(seed_min, seed_max + 1))
    if source_result["seeds"] != seeds:
        raise AssertionError("source seed block mismatch")
    sealed_min = int(contract["tape_governance"]["sealed_validation_min"])
    sealed_max = int(contract["tape_governance"]["sealed_validation_max"])
    if set(seeds) & set(range(sealed_min, sealed_max + 1)):
        raise AssertionError("burned and sealed seed ranges overlap")

    configs = {config.config_id: config for config in finite_state_rich_configurations()}
    expected_configs = list(contract["controller_family"]["config_ids"])
    if list(configs) != expected_configs:
        raise AssertionError("finite controller family drift")
    cells = list(contract["cells"])
    if set(source_result["configurations"]) != set(expected_configs):
        raise AssertionError("source configuration set mismatch")
    scheduler = primary_scheduler()
    parent_contract = json.loads(
        (ROOT / "contracts/program_o_state_rich_comparator_fit_v1.json").read_text()
    )
    model = parent_contract["observation_contract"]["model_parameters"]

    output.mkdir(parents=True)
    results: dict[str, dict[str, Any]] = {config_id: {} for config_id in configs}
    completed = 0
    for cell_id in cells:
        skeletons = [
            json.loads(
                (parent_run / "skeletons" / cell_id / f"tape_{seed}.json").read_text()
            )
            for seed in seeds
        ]
        if [int(row["seed"]) for row in skeletons] != seeds:
            raise AssertionError(f"skeleton seed mismatch: {cell_id}")
        panel = load_panel(parent_run, cell_id, seeds)
        for config_id, config in configs.items():
            source_row = source_result["configurations"][config_id][cell_id]
            placebo_families = evaluate_placebos(
                contract=contract,
                config=config,
                cell_id=cell_id,
                real_indices=source_row["calendar_indices"],
                skeletons=skeletons,
                panel=panel,
                scheduler=scheduler,
                model=model,
            )
            base_pass = nonresource_pass(source_row, contract)
            total_info_pass = bool(placebo_families["total_information"]["passed"])
            operational_pass = bool(
                placebo_families["operational_state_given_current_belief"]["passed"]
            )
            belief_pass = bool(
                placebo_families["belief_given_current_operational_state"]["passed"]
            )
            fixed_clock_total = bool(base_pass and total_info_pass)
            fixed_clock_state = bool(fixed_clock_total and operational_pass)
            pay_per_use_total = bool(
                fixed_clock_total
                and source_row["strict_actual_use_pass"]
                and source_row["resource_frontier"]["passed"]
            )
            pay_per_use_state = bool(pay_per_use_total and operational_pass)
            results[config_id][cell_id] = {
                "mean_delta_vs_full_frontier": source_row[
                    "mean_delta_vs_full_frontier"
                ],
                "favorable_tapes": source_row["favorable_tapes"],
                "metric_guardrails_pass": source_row["metric_guardrails_pass"],
                "reserved_capacity_equal": source_row["reserved_capacity_equal"],
                "strict_actual_use_pass": source_row["strict_actual_use_pass"],
                "resource_frontier_pass": source_row["resource_frontier"]["passed"],
                "resource_frontier_eligible_calendars": source_row[
                    "resource_frontier"
                ]["eligible_calendar_count"],
                "action_trajectory_pass": source_row["action_trajectory"]["passed"],
                "state_counterfactual_pass": source_row["state_counterfactuals"][
                    "passed"
                ],
                "resource_deltas": {
                    key: source_row["mean_deltas"][key]
                    for key in (
                        "gross_production_quantity",
                        "charged_downstream_vehicle_hours",
                        "actual_loaded_departures",
                        "actual_payload",
                        "actual_downstream_vehicle_hours",
                    )
                },
                "placebo_families": placebo_families,
                "diagnostic_gates": {
                    "common_nonresource_pass": base_pass,
                    "total_information_pass": total_info_pass,
                    "incremental_operational_state_pass": operational_pass,
                    "incremental_belief_pass": belief_pass,
                    "fixed_clock_total_observable_pass": fixed_clock_total,
                    "fixed_clock_state_rich_increment_pass": fixed_clock_state,
                    "pay_per_use_total_observable_pass": pay_per_use_total,
                    "pay_per_use_state_rich_increment_pass": pay_per_use_state,
                },
            }
        completed += 1
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

    required_size = int(contract["stability"]["required_connected_cells"])
    stability: dict[str, Any] = {}
    criteria = (
        "fixed_clock_total_observable_pass",
        "fixed_clock_state_rich_increment_pass",
        "pay_per_use_total_observable_pass",
        "pay_per_use_state_rich_increment_pass",
    )
    for criterion in criteria:
        by_config = {}
        for config_id, cell_rows in results.items():
            passing = [
                cell_id
                for cell_id, row in cell_rows.items()
                if row["diagnostic_gates"][criterion]
            ]
            by_config[config_id] = connected_components(
                passing, required_size=required_size
            )
        stability[criterion] = {
            "by_configuration": by_config,
            "eligible_configurations": [
                config_id
                for config_id, row in by_config.items()
                if row["passed"]
            ],
            "passed": any(row["passed"] for row in by_config.values()),
        }

    fixed_total = stability["fixed_clock_total_observable_pass"]["passed"]
    pay_total = stability["pay_per_use_total_observable_pass"]["passed"]
    if fixed_total and pay_total:
        status = "DIAGNOSTIC_STABLE_SIGNAL_UNDER_BOTH_RESOURCE_ESTIMANDS"
    elif fixed_total:
        status = "DIAGNOSTIC_STABLE_SIGNAL_FIXED_CLOCK_ONLY"
    else:
        status = "DIAGNOSTIC_NO_STABLE_TOTAL_OBSERVABLE_SIGNAL"

    result = {
        "schema_version": "program_o_state_rich_dual_resource_diagnostic_result_v1",
        "status": status,
        "generated_at_utc": now_utc(),
        "scientific_commit": git_commit(),
        "contract_sha256": sha256(contract_path),
        "source_result_sha256": expected_source_sha,
        "parent_result_sha256": source_result["parent_result_sha256"],
        "seeds": seeds,
        "validation_seed_accessed": False,
        "configurations": results,
        "stability": stability,
        "claim_boundary": contract["claim_boundary"],
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
    parser.add_argument("--source-result", type=Path, default=DEFAULT_SOURCE_RESULT)
    parser.add_argument("--parent-run", type=Path, default=DEFAULT_PARENT_RUN)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        contract_path=args.contract.resolve(),
        source_result_path=args.source_result.resolve(),
        parent_run=args.parent_run.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
