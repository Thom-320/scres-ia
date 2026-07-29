#!/usr/bin/env python3
"""Prospective Program O classical-H_obs validation under physical fixed clock."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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

from scripts.diagnose_program_o_state_rich_dual_resource import (  # noqa: E402
    evaluate_placebos,
)
from scripts.screen_program_o_hobs_fit import (  # noqa: E402
    load_cell_panel,
    now_utc,
    primary_scheduler,
    produce_seed_cell,
    sha256,
    skeleton_path,
    write_json_atomic,
)
from scripts.screen_program_o_state_rich_fit import (  # noqa: E402
    HIGHER_KEYS,
    LOWER_KEYS,
    evaluate_configuration,
)
from supply_chain.program_o_full_des import (  # noqa: E402
    run_program_o_full_des_episode,
)
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    MATRIX_KEYS,
    direct_full_des_vector,
    simulate_full_des_frontier,
)
from supply_chain.program_o_state_rich import (  # noqa: E402
    StateRichConfiguration,
)


PARENT_FULL_DES_CONTRACT = ROOT / "contracts/program_o_full_des_hpi_translation_v1.json"
PARENT_STATE_CONTRACT = ROOT / "contracts/program_o_state_rich_comparator_fit_v1.json"
PARENT_HOBS_CONTRACT = ROOT / "contracts/program_o_hobs_prelearner_v1.json"
DEFAULT_CONTRACT = ROOT / "contracts/program_o_fixed_clock_physical_hobs_validation_v1.json"
FIT_RESULT = ROOT / "results/program_o/state_rich_comparator_fit_v1/result.json"


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def decode_calendar(index: int, *, weeks: int = 8) -> tuple[int, ...]:
    value = int(index)
    actions = [0] * weeks
    for position in range(weeks - 1, -1, -1):
        actions[position] = value % 4
        value //= 4
    if value:
        raise ValueError("calendar index exceeds frozen horizon")
    return tuple(actions)


def physical_replay_task(task: Mapping[str, Any]) -> dict[str, Any]:
    parent = json.loads(PARENT_FULL_DES_CONTRACT.read_text())
    scheduler = primary_scheduler(parent)
    calendar = decode_calendar(int(task["calendar_index"]))
    sim, panel = run_program_o_full_des_episode(
        seed=int(task["seed"]),
        calendar=calendar,
        scheduler=scheduler,
        regime_persistence=float(task["rho"]),
        dominant_share=float(task["share"]),
        downstream_freight_physics_mode="fixed_clock_physical_v1",
    )
    vector = direct_full_des_vector(sim, panel)
    resources = panel["resources"]
    return {
        "cell_id": str(task["cell_id"]),
        "seed": int(task["seed"]),
        "calendar_index": int(task["calendar_index"]),
        "vector": vector,
        "scheduled_resource_vector": [
            resources["committed_action_batch_slots"],
            resources["gross_action_production_quantity"],
            resources["scheduled_downstream_missions"],
            resources["scheduled_downstream_vehicle_hours"],
            resources["scheduled_downstream_crew_hours"],
            resources["scheduled_payload_capacity"],
            resources["setup_hours"],
        ],
        "loaded_departures": resources["actual_loaded_departures"],
        "empty_missions": resources["empty_downstream_missions"],
        "max_product_residual": panel["conservation"]["max_abs_product_residual"],
        "max_partition_residual": panel["conservation"]["max_abs_partition_residual"],
    }


def joint_bootstrap(
    *,
    panels: Mapping[str, Mapping[str, np.ndarray]],
    rows: Mapping[str, Mapping[str, Any]],
    placebo_rows: Mapping[str, Mapping[str, Any]],
    cells: Sequence[str],
    resamples: int,
    static_indices: Mapping[str, int],
) -> dict[str, Any]:
    """Studentized simultaneous LCBs against development-frozen statics."""
    n_tapes = 48
    seed = int.from_bytes(
        hashlib.sha256(b"program-o-fixed-clock-hobs-validation-v1").digest()[:8],
        "big",
    )
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(0, n_tapes, size=(resamples, n_tapes))
    counts = np.zeros((resamples, n_tapes), dtype=np.float64)
    for position, sample in enumerate(bootstrap_indices):
        counts[position] = np.bincount(sample, minlength=n_tapes)
    counts /= float(n_tapes)

    definitions: list[tuple[str, str, str | None]] = []
    points: dict[str, float] = {}
    boot_columns: list[np.ndarray] = []
    for cell_id in cells:
        panel = panels[cell_id]
        row = rows[cell_id]
        policy_indices = np.asarray(row["calendar_indices"], dtype=np.int64)
        tape_rows = np.arange(n_tapes, dtype=np.int64)
        policy = {key: panel[key][tape_rows, policy_indices] for key in MATRIX_KEYS}

        static_point = int(static_indices[cell_id])

        signed_metrics = [("ret_visible", 1.0, "primary")]
        signed_metrics.extend((key, 1.0, "guardrail") for key in HIGHER_KEYS)
        signed_metrics.extend((key, -1.0, "guardrail") for key in LOWER_KEYS)
        for key, sign, kind in signed_metrics:
            name = f"{cell_id}::{kind}::{key}"
            comparator_point = panel[key][:, static_point]
            point = float(sign * (policy[key] - comparator_point).mean())
            policy_boot = counts @ policy[key]
            static_boot = counts @ panel[key][:, static_point]
            definitions.append((name, kind, key))
            points[name] = point
            boot_columns.append(sign * (policy_boot - static_boot))

        real = panel["ret_visible"][tape_rows, policy_indices]
        for family, family_row in placebo_rows[cell_id].items():
            for mode, placebo in family_row["placebos"].items():
                name = f"{cell_id}::placebo::{family}::{mode}"
                indices = np.asarray(placebo["calendar_indices"], dtype=np.int64)
                contrast = real - panel["ret_visible"][tape_rows, indices]
                definitions.append((name, "placebo", mode))
                points[name] = float(contrast.mean())
                boot_columns.append(counts @ contrast)

    boot = np.column_stack(boot_columns)
    point_vector = np.asarray([points[name] for name, _kind, _key in definitions])
    standard_error = boot.std(axis=0, ddof=1)
    active = standard_error > 1e-15
    max_t = np.zeros(resamples, dtype=float)
    if np.any(active):
        standardized = (point_vector[None, active] - boot[:, active]) / standard_error[active]
        max_t = np.max(standardized, axis=1)
    simultaneous_critical = float(np.quantile(max_t, 0.95))
    lower_bounds = point_vector.copy()
    lower_bounds[active] = point_vector[active] - simultaneous_critical * standard_error[active]
    estimates = {}
    for index, (name, kind, key) in enumerate(definitions):
        estimates[name] = {
            "kind": kind,
            "metric_or_mode": key,
            "estimate": float(point_vector[index]),
            "bootstrap_se": float(standard_error[index]),
            "simultaneous_lcb95": float(lower_bounds[index]),
        }
    return {
        "resamples": int(resamples),
        "comparator_mode": "development_selected_full_frontier_frozen_before_validation",
        "static_reselected_each_resample": False,
        "estimand_count": len(definitions),
        "simultaneous_critical": simultaneous_critical,
        "estimates": estimates,
    }


def apply_frozen_static_comparator(
    *,
    row: Mapping[str, Any],
    panel: Mapping[str, np.ndarray],
    static_index: int,
) -> dict[str, Any]:
    """Re-score an already emitted policy against its development-frozen static."""
    updated = dict(row)
    policy_indices = np.asarray(row["calendar_indices"], dtype=np.int64)
    tape_rows = np.arange(len(policy_indices), dtype=np.int64)
    deltas = {
        key: panel[key][tape_rows, policy_indices] - panel[key][:, int(static_index)]
        for key in MATRIX_KEYS
    }
    means = {key: float(values.mean()) for key, values in deltas.items()}
    tolerance = 1e-12
    updated.update(
        {
            "static_index": int(static_index),
            "static_mean_ret_visible": float(panel["ret_visible"][:, int(static_index)].mean()),
            "mean_delta_vs_full_frontier": means["ret_visible"],
            "favorable_tapes": int(np.sum(deltas["ret_visible"] > 0.0)),
            "mean_deltas": means,
            "metric_guardrails_pass": bool(
                all(means[key] >= -tolerance for key in HIGHER_KEYS)
                and all(means[key] <= tolerance for key in LOWER_KEYS)
            ),
            "reserved_capacity_equal": bool(
                means["gross_policy_batch_slots"] == 0.0
                and means["gross_production_quantity"] == 0.0
                and means["charged_daily_dispatch_slots"] == 0.0
                and means["charged_downstream_vehicle_hours"] == 0.0
            ),
            "per_tape_ret_delta": deltas["ret_visible"].astype(float).tolist(),
            "comparator_selection": "burned_development_full_frontier_frozen_before_validation",
        }
    )
    return updated


def run(*, contract_path: Path, output: Path, workers: int) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    contract = json.loads(contract_path.read_text())
    if contract["validation_tapes"]["status"] != "SEALED_NOT_ACCESSED":
        raise RuntimeError("validation tape contract is not sealed")
    seed_min, seed_max = contract["validation_tapes"]["range"]
    seeds = list(range(int(seed_min), int(seed_max) + 1))
    if len(seeds) != 48 or int(seed_min) < 7430001:
        raise RuntimeError("corrective validation requires a fresh 48-tape block")
    parent_full = json.loads(PARENT_FULL_DES_CONTRACT.read_text())
    parent_hobs = json.loads(PARENT_HOBS_CONTRACT.read_text())
    parent_state = json.loads(PARENT_STATE_CONTRACT.read_text())
    scheduler = primary_scheduler(parent_full)
    model = parent_state["observation_contract"]["model_parameters"]
    all_cells = {str(row["id"]): row for row in parent_hobs["stability_cells"]}
    cell_ids = list(contract["connected_cells"])
    cells = [all_cells[cell_id] for cell_id in cell_ids]
    config = StateRichConfiguration("belief_mpc", 3)
    if config.config_id != contract["primary_policy"]["config_id"]:
        raise RuntimeError("primary controller drift")
    frozen_source = contract["open_loop_comparator"]["frozen_source"]
    if sha256(FIT_RESULT) != frozen_source["sha256"]:
        raise RuntimeError("frozen comparator source identity mismatch")
    frozen_indices = {
        str(cell_id): int(index)
        for cell_id, index in contract["open_loop_comparator"]["frozen_static_indices"].items()
    }
    if set(frozen_indices) != set(cell_ids):
        raise RuntimeError("frozen comparator cell set drift")

    output.mkdir(parents=True)
    write_json_atomic(
        output / "progress.json",
        {"status": "RUNNING", "completed": 0, "total": len(seeds) * len(cells)},
    )
    tasks = [(seed, cell) for cell in cells for seed in seeds]
    shard_rows = []
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        futures = {
            pool.submit(
                produce_seed_cell,
                seed=seed,
                cell=cell,
                scheduler=scheduler,
                output=output,
            ): (seed, cell)
            for seed, cell in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            shard_rows.append(future.result())
            write_json_atomic(
                output / "progress.json",
                {
                    "status": "RUNNING_MATRICES",
                    "completed": completed,
                    "total": len(tasks),
                    "updated_at_utc": now_utc(),
                },
            )

    panels = {}
    skeletons = {}
    rows = {}
    audits = {}
    placebos = {}
    placebo_contract = {"placebo_families": contract["placebos"]}
    for cell in cells:
        cell_id = str(cell["id"])
        panels[cell_id] = load_cell_panel(output, cell_id, seeds)
        skeletons[cell_id] = [
            json.loads(skeleton_path(output, cell_id, seed).read_text()) for seed in seeds
        ]
        row, decision_audits = evaluate_configuration(
            config=config,
            cell=cell,
            seeds=seeds,
            skeletons=skeletons[cell_id],
            panel=panels[cell_id],
            scheduler=scheduler,
            model=model,
        )
        rows[cell_id] = apply_frozen_static_comparator(
            row=row,
            panel=panels[cell_id],
            static_index=frozen_indices[cell_id],
        )
        audits[cell_id] = decision_audits
        placebos[cell_id] = evaluate_placebos(
            contract=placebo_contract,
            config=config,
            cell_id=cell_id,
            real_indices=rows[cell_id]["calendar_indices"],
            skeletons=skeletons[cell_id],
            panel=panels[cell_id],
            scheduler=scheduler,
            model=model,
        )

    inference = joint_bootstrap(
        panels=panels,
        rows=rows,
        placebo_rows=placebos,
        cells=cell_ids,
        resamples=int(contract["primary_gate"]["paired_bootstrap_resamples"]),
        static_indices=frozen_indices,
    )

    replay_tasks = {}
    for cell in cells:
        cell_id = str(cell["id"])
        indices_by_seed = [set() for _seed in seeds]
        for position, index in enumerate(rows[cell_id]["calendar_indices"]):
            indices_by_seed[position].add(int(index))
        for family in placebos[cell_id].values():
            for placebo in family["placebos"].values():
                for position, index in enumerate(placebo["calendar_indices"]):
                    indices_by_seed[position].add(int(index))
        static_index = int(rows[cell_id]["static_index"])
        for position, seed in enumerate(seeds):
            indices_by_seed[position].add(static_index)
            for index in indices_by_seed[position]:
                key = (cell_id, seed, index)
                replay_tasks[key] = {
                    "cell_id": cell_id,
                    "seed": seed,
                    "calendar_index": index,
                    "rho": cell["regime_persistence"],
                    "share": cell["dominant_product_share"],
                }

    replay_rows = []
    replay_failures = []
    resource_vectors = set()
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        futures = {
            pool.submit(physical_replay_task, task): key for key, task in replay_tasks.items()
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            cell_id = row["cell_id"]
            seed_position = int(row["seed"]) - int(seed_min)
            with np.load(
                output / "raw_calendar_matrix" / cell_id / f"tape_{row['seed']}.npz"
            ) as shard:
                for key in MATRIX_KEYS:
                    error = abs(
                        float(row["vector"][key]) - float(shard[key][int(row["calendar_index"])])
                    )
                    if error > 1e-10:
                        replay_failures.append(
                            {
                                "cell_id": cell_id,
                                "seed": row["seed"],
                                "calendar_index": row["calendar_index"],
                                "metric": key,
                                "error": error,
                            }
                        )
            del seed_position
            resource_vectors.add(tuple(row["scheduled_resource_vector"]))
            if (
                row["loaded_departures"] + row["empty_missions"] != 112
                or row["max_product_residual"] > 1e-8
                or row["max_partition_residual"] > 1e-8
            ):
                replay_failures.append(row)
            replay_rows.append(row)
            if completed % 25 == 0:
                write_json_atomic(
                    output / "progress.json",
                    {
                        "status": "RUNNING_PHYSICAL_REPLAY",
                        "completed": completed,
                        "total": len(replay_tasks),
                        "updated_at_utc": now_utc(),
                    },
                )

    estimates = inference["estimates"]
    primary_pass = all(
        estimates[f"{cell_id}::primary::ret_visible"]["simultaneous_lcb95"] >= 0.01
        and int(rows[cell_id]["favorable_tapes"]) >= 34
        for cell_id in cell_ids
    )
    placebo_pass = all(
        estimate["simultaneous_lcb95"] > 0.0
        for estimate in estimates.values()
        if estimate["kind"] == "placebo"
    )
    guardrail_pass = all(
        estimate["simultaneous_lcb95"] >= 0.0
        for estimate in estimates.values()
        if estimate["kind"] == "guardrail"
    )
    action_pass = all(
        rows[cell_id]["action_trajectory"]["passed"]
        and rows[cell_id]["state_counterfactuals"]["passed"]
        for cell_id in cell_ids
    )
    physical_pass = not replay_failures and len(resource_vectors) == 1
    passed = primary_pass and placebo_pass and guardrail_pass and action_pass and physical_pass
    status = contract["terminal_rules"]["pass"] if passed else contract["terminal_rules"]["fail"]
    result = {
        "schema_version": "program_o_fixed_clock_physical_hobs_validation_result_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scientific_commit": git_commit(),
        "contract_sha256": sha256(contract_path),
        "seeds": seeds,
        "validation_seed_accessed": True,
        "primary_policy": config.config_id,
        "cells": rows,
        "decision_audits": audits,
        "placebos": placebos,
        "inference": inference,
        "physical_replay": {
            "episodes": len(replay_rows),
            "failures": replay_failures,
            "unique_scheduled_resource_vectors": len(resource_vectors),
            "rows": replay_rows,
        },
        "gates": {
            "primary_pass": primary_pass,
            "placebo_pass": placebo_pass,
            "guardrail_pass": guardrail_pass,
            "action_pass": action_pass,
            "physical_pass": physical_pass,
        },
        "raw_shards": sorted(shard_rows, key=lambda row: (row["cell_id"], row["seed"])),
        "claim_boundary": {
            "h_obs_confirmed": passed,
            "learned_advantage_confirmed": False,
            "learner_authorized": passed,
            "paper2_confirmed": False,
            "paper3_authorized": False,
        },
    }
    write_json_atomic(output / "result.json", result)
    write_json_atomic(
        output / "progress.json",
        {
            "status": "COMPLETE",
            "result_sha256": sha256(output / "result.json"),
            "updated_at_utc": now_utc(),
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = run(
        contract_path=args.contract.resolve(),
        output=args.output.resolve(),
        workers=int(args.workers),
    )
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
