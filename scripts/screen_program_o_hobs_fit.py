#!/usr/bin/env python3
"""Run Program O's frozen observable-headroom fit gate.

This script is fit-only.  It may select one deterministic observable policy
configuration but cannot open the H_obs validation block or authorize a
learner.  Every tape retains the complete 4^8 open-loop matrix so the static
denominator can be recomputed without trusting a stored winner.
"""

from __future__ import annotations

import argparse
from collections import Counter
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

from supply_chain.program_o_full_des import run_program_o_full_des_episode  # noqa: E402
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    MATRIX_KEYS,
    direct_full_des_vector,
    extract_full_des_skeleton,
    simulate_full_des_frontier,
)
from supply_chain.program_o_hobs import (  # noqa: E402
    POLICY_IDS,
    calendar_index,
    decision_audit_rows,
    observable_calendar,
)

DEFAULT_CONTRACT = ROOT / "contracts/program_o_hobs_prelearner_v1.json"
PARENT_CONTRACT = ROOT / "contracts/program_o_full_des_hpi_translation_v1.json"

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


def primary_scheduler(parent: Mapping[str, Any]) -> Mapping[str, Sequence[str]]:
    scheduler_id = str(parent["action"]["primary_scheduler"])
    return parent["action"]["within_week_schedulers"][scheduler_id]


def shard_path(output: Path, cell_id: str, seed: int) -> Path:
    return output / "raw_calendar_matrix" / str(cell_id) / f"tape_{int(seed)}.npz"


def skeleton_path(output: Path, cell_id: str, seed: int) -> Path:
    return output / "skeletons" / str(cell_id) / f"tape_{int(seed)}.json"


def produce_seed_cell(
    *,
    seed: int,
    cell: Mapping[str, Any],
    scheduler: Mapping[str, Sequence[str]],
    output: Path,
) -> dict[str, Any]:
    destination = shard_path(output, str(cell["id"]), int(seed))
    skeleton_destination = skeleton_path(output, str(cell["id"]), int(seed))
    if destination.exists() or skeleton_destination.exists():
        raise FileExistsError(f"refusing to overwrite {cell['id']}/{seed}")
    skeleton, _ = extract_full_des_skeleton(
        seed=int(seed),
        scheduler=scheduler,
        regime_persistence=float(cell["regime_persistence"]),
        dominant_share=float(cell["dominant_product_share"]),
    )
    panel = simulate_full_des_frontier(
        skeleton=skeleton,
        scheduler=scheduler,
        complete_substitution=False,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **panel)
    os.replace(temporary, destination)
    write_json_atomic(skeleton_destination, skeleton.as_dict())
    return {
        "cell_id": str(cell["id"]),
        "seed": int(seed),
        "matrix": str(destination),
        "matrix_sha256": sha256(destination),
        "skeleton": str(skeleton_destination),
        "skeleton_sha256": sha256(skeleton_destination),
        "tape_sha256": skeleton.tape_sha256,
        "prefix_state_hash": skeleton.prefix_state_hash,
    }


def load_cell_panel(
    output: Path, cell_id: str, seeds: Sequence[int]
) -> dict[str, np.ndarray]:
    values = {key: [] for key in MATRIX_KEYS}
    for seed in seeds:
        with np.load(shard_path(output, cell_id, seed)) as shard:
            if tuple(shard.files) != MATRIX_KEYS:
                raise AssertionError(f"matrix schema drift: {cell_id}/{seed}")
            for key in MATRIX_KEYS:
                values[key].append(np.asarray(shard[key]))
    return {key: np.stack(rows) for key, rows in values.items()}


def configurations() -> list[dict[str, Any]]:
    return [
        {"policy_id": policy_id, "initial_action": initial_action}
        for policy_id in POLICY_IDS
        for initial_action in range(4)
    ]


def policy_calendar_for_skeleton(
    skeleton: Mapping[str, Any],
    config: Mapping[str, Any],
    model: Mapping[str, Any],
    **placebo: Any,
) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
    calendar, rows = observable_calendar(
        request_times=skeleton["order_times"],
        request_products=placebo.pop(
            "request_products", skeleton["order_products"]
        ),
        decision_start=float(skeleton["decision_start"]),
        decision_weeks=int(skeleton["decision_weeks"]),
        policy_id=str(config["policy_id"]),
        initial_action=int(config["initial_action"]),
        regime_persistence=float(model["regime_persistence"]),
        dominant_share=float(model["dominant_product_share"]),
        **placebo,
    )
    return calendar, decision_audit_rows(rows)


def evaluate_configurations(
    *,
    panel: Mapping[str, np.ndarray],
    skeletons: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    static_index = int(np.argmax(panel["ret_visible"].mean(axis=0)))
    rows: list[dict[str, Any]] = []
    for config in configurations():
        indices = []
        calendars = []
        for skeleton in skeletons:
            calendar, _ = policy_calendar_for_skeleton(skeleton, config, model)
            calendars.append(list(calendar))
            indices.append(calendar_index(calendar))
        indices_array = np.asarray(indices, dtype=np.int64)
        tape_rows = np.arange(len(skeletons), dtype=np.int64)
        deltas = {
            key: panel[key][tape_rows, indices_array] - panel[key][:, static_index]
            for key in MATRIX_KEYS
        }
        admissible = all(float(deltas[key].mean()) >= -1e-12 for key in HIGHER_KEYS)
        admissible = admissible and all(
            float(deltas[key].mean()) <= 1e-12 for key in LOWER_KEYS
        )
        rows.append(
            {
                **config,
                "admissible": bool(admissible),
                "mean_ret_visible": float(
                    panel["ret_visible"][tape_rows, indices_array].mean()
                ),
                "mean_delta_vs_full_frontier": float(
                    deltas["ret_visible"].mean()
                ),
                "favorable_tapes": int((deltas["ret_visible"] > 0.0).sum()),
                "unique_sequences": len({tuple(row) for row in calendars}),
                "guardrail_mean_deltas": {
                    key: float(deltas[key].mean())
                    for key in (*HIGHER_KEYS, *LOWER_KEYS)
                },
            }
        )
    eligible = [index for index, row in enumerate(rows) if row["admissible"]]
    if not eligible:
        return rows, -1
    selected = min(
        eligible,
        key=lambda index: (
            -float(rows[index]["mean_ret_visible"]),
            -float(rows[index]["guardrail_mean_deltas"]["ration_ret_visible"]),
            -float(rows[index]["guardrail_mean_deltas"]["worst_product_fill"]),
            int(POLICY_IDS.index(str(rows[index]["policy_id"]))),
            int(rows[index]["initial_action"]),
        ),
    )
    return rows, int(selected)


def evaluate_selected(
    *,
    panel: Mapping[str, np.ndarray],
    skeletons: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    model: Mapping[str, Any],
    donor_products: Sequence[Sequence[str]] | None = None,
    **placebo: Any,
) -> dict[str, Any]:
    calendars: list[list[int]] = []
    audits: list[list[dict[str, Any]]] = []
    indices: list[int] = []
    for position, skeleton in enumerate(skeletons):
        local = dict(placebo)
        if donor_products is not None:
            local["request_products"] = donor_products[position]
        calendar, rows = policy_calendar_for_skeleton(skeleton, config, model, **local)
        calendars.append(list(calendar))
        audits.append(rows)
        indices.append(calendar_index(calendar))
    indices_array = np.asarray(indices, dtype=np.int64)
    tape_rows = np.arange(len(skeletons), dtype=np.int64)
    static_index = int(np.argmax(panel["ret_visible"].mean(axis=0)))
    metrics = {
        key: panel[key][tape_rows, indices_array].astype(float)
        for key in MATRIX_KEYS
    }
    deltas = {
        key: metrics[key] - panel[key][:, static_index] for key in MATRIX_KEYS
    }
    counts = Counter(tuple(calendar) for calendar in calendars)
    action_counts = Counter(action for calendar in calendars for action in calendar)
    varying_weeks = sum(
        len({calendar[week] for calendar in calendars}) > 1 for week in range(8)
    )
    return {
        "config": dict(config),
        "static_index": static_index,
        "static_mean_ret_visible": float(panel["ret_visible"][:, static_index].mean()),
        "policy_mean_ret_visible": float(metrics["ret_visible"].mean()),
        "mean_delta_vs_full_frontier": float(deltas["ret_visible"].mean()),
        "favorable_tapes": int((deltas["ret_visible"] > 0.0).sum()),
        "calendars": calendars,
        "calendar_indices": indices,
        "decision_audits": audits,
        "unique_sequences": len(counts),
        "modal_sequence_fraction": max(counts.values()) / len(calendars),
        "action_counts": {str(key): value for key, value in sorted(action_counts.items())},
        "varying_week_indices": varying_weeks,
        "mean_deltas": {key: float(value.mean()) for key, value in deltas.items()},
        "per_tape_ret_delta": deltas["ret_visible"].tolist(),
    }


def direct_replay_selected(
    *,
    seeds: Sequence[int],
    cells: Sequence[Mapping[str, Any]],
    scheduler: Mapping[str, Sequence[str]],
    selected_by_cell: Mapping[str, Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    maximum_error = 0.0
    episodes = 0
    for cell in cells:
        cell_id = str(cell["id"])
        calendars = selected_by_cell[cell_id]["calendars"]
        for seed, calendar in zip(seeds, calendars, strict=True):
            sim, direct_panel = run_program_o_full_des_episode(
                seed=int(seed),
                calendar=calendar,
                scheduler=scheduler,
                regime_persistence=float(cell["regime_persistence"]),
                dominant_share=float(cell["dominant_product_share"]),
            )
            direct = direct_full_des_vector(sim, direct_panel)
            with np.load(shard_path(output, cell_id, int(seed))) as shard:
                index = calendar_index(calendar)
                for key in MATRIX_KEYS:
                    maximum_error = max(
                        maximum_error,
                        abs(float(direct[key]) - float(shard[key][index])),
                    )
            episodes += 1
    return {
        "episodes": episodes,
        "maximum_absolute_error": maximum_error,
        "passed": maximum_error <= 1e-10,
    }


def run_fit(
    *,
    contract_path: Path,
    output: Path,
    workers: int,
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text())
    parent = json.loads(PARENT_CONTRACT.read_text())
    seed_start, seed_end = map(int, contract["tape_blocks"]["fit"]["range"])
    seeds = list(range(seed_start, seed_end + 1))
    cells = list(contract["stability_cells"])
    scheduler = primary_scheduler(parent)
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        output / "progress.json",
        {"status": "RUNNING", "completed": 0, "total": len(seeds) * len(cells)},
    )
    shard_rows = []
    tasks = [(seed, cell) for cell in cells for seed in seeds]
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
                    "status": "RUNNING",
                    "completed": completed,
                    "total": len(tasks),
                    "updated_at_utc": now_utc(),
                },
            )

    panels = {}
    skeletons = {}
    for cell in cells:
        cell_id = str(cell["id"])
        panels[cell_id] = load_cell_panel(output, cell_id, seeds)
        skeletons[cell_id] = [
            json.loads(skeleton_path(output, cell_id, seed).read_text())
            for seed in seeds
        ]
    primary_id = "rho75_share90"
    fit_rows, selected_index = evaluate_configurations(
        panel=panels[primary_id],
        skeletons=skeletons[primary_id],
        model=contract["policy_model_parameters"],
    )
    if selected_index < 0:
        selected_config = None
        status = "STOP_RESOURCE_OR_GUARDRAIL_CONFOUND"
        selected = {}
        placebos = {}
        direct = {"episodes": 0, "maximum_absolute_error": None, "passed": False}
    else:
        selected_config = {
            "policy_id": fit_rows[selected_index]["policy_id"],
            "initial_action": fit_rows[selected_index]["initial_action"],
        }
        selected = {
            cell_id: evaluate_selected(
                panel=panels[cell_id],
                skeletons=skeletons[cell_id],
                config=selected_config,
                model=contract["policy_model_parameters"],
            )
            for cell_id in panels
        }
        primary_skeletons = skeletons[primary_id]
        shift = 17
        donor = [
            primary_skeletons[(position + shift) % len(primary_skeletons)][
                "order_products"
            ]
            for position in range(len(primary_skeletons))
        ]
        placebos = {
            "extra_week_delay": evaluate_selected(
                panel=panels[primary_id],
                skeletons=primary_skeletons,
                config=selected_config,
                model=contract["policy_model_parameters"],
                history_delay_weeks=1,
            ),
            "cross_tape_shift17": evaluate_selected(
                panel=panels[primary_id],
                skeletons=primary_skeletons,
                config=selected_config,
                model=contract["policy_model_parameters"],
                donor_products=donor,
            ),
            "wrong_product_swap": evaluate_selected(
                panel=panels[primary_id],
                skeletons=primary_skeletons,
                config=selected_config,
                model=contract["policy_model_parameters"],
                swap_observed_labels=True,
            ),
            "no_history": evaluate_selected(
                panel=panels[primary_id],
                skeletons=primary_skeletons,
                config=selected_config,
                model=contract["policy_model_parameters"],
                ignore_history=True,
            ),
        }
        direct = direct_replay_selected(
            seeds=seeds,
            cells=cells,
            scheduler=scheduler,
            selected_by_cell=selected,
            output=output,
        )
        primary = selected[primary_id]
        action_gate = (
            int(primary["unique_sequences"]) >= 8
            and float(primary["modal_sequence_fraction"]) <= 0.50
            and int(primary["varying_week_indices"]) >= 2
            and sum(
                count >= 0.10 * len(seeds) * 8
                for count in primary["action_counts"].values()
            )
            >= 2
        )
        fit_rule = contract["fit_pass_rule"]
        placebo_gate = all(
            float(primary["policy_mean_ret_visible"])
            > float(row["policy_mean_ret_visible"]) + 1e-12
            for row in placebos.values()
        )
        passed = (
            float(primary["mean_delta_vs_full_frontier"])
            >= float(fit_rule["primary_mean_delta_over_full_frontier_minimum"])
            and int(primary["favorable_tapes"])
            >= int(fit_rule["minimum_favorable_tapes"])
            and action_gate
            and placebo_gate
            and bool(direct["passed"])
        )
        status = (
            "PASS_PROGRAM_O_HOBS_FIT__FREEZE_SELECTION_BEFORE_VALIDATION"
            if passed
            else "STOP_NO_OBSERVABLE_HEADROOM"
        )

    result = {
        "schema_version": "program_o_hobs_fit_result_v1",
        "generated_at_utc": now_utc(),
        "status": status,
        "scientific_commit": git_commit(),
        "contract": str(contract_path),
        "contract_sha256": sha256(contract_path),
        "seeds": seeds,
        "fit_configurations": fit_rows,
        "selected_config": selected_config,
        "selected_by_cell": selected,
        "placebos_primary_cell": placebos,
        "direct_replay": direct,
        "raw_shards": sorted(shard_rows, key=lambda row: (row["cell_id"], row["seed"])),
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
            "completed": len(tasks),
            "total": len(tasks),
            "result_sha256": sha256(output / "result.json"),
            "updated_at_utc": now_utc(),
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result = run_fit(
        contract_path=args.contract.resolve(),
        output=args.output.resolve(),
        workers=int(args.workers),
    )
    print(args.output / "result.json")
    return 0 if result["status"].startswith(("PASS_", "STOP_")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
