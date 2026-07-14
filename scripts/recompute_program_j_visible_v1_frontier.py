#!/usr/bin/env python3
"""Enumerate the central Program-J 3^8 frontier under request-snapshot v2.

The historical screen used ``ret_excel_full_ledger`` and 1,041 periodic
calendars.  The first visible-v1 repair was then quarantined because its OAT-
time Bt/Ut reconstruction was not source-valid.  This retrospective burned-
tape repair enumerates all 6,561 exact
eight-week schedules, preserves raw score matrices, and compares the complete
open-loop frontier, a guardrailed convexified comparator, a tape-contingent PI
relaxation, and the two historical observable rules.  It is not confirmatory.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.maintenance_control import (
    ACTIONS,
    materialize_tape,
    periodic_policy,
    run_policy,
    wip_bottleneck_policy,
    worst_condition_policy,
)


ROOT = Path(__file__).resolve().parent.parent
WEEKS = 8
SEQUENCES = tuple(itertools.product(ACTIONS, repeat=WEEKS))
REFERENCE_PERIOD = ("PM6", "PM6", "PM5", "PM6", "PM6")
REFERENCE = tuple(REFERENCE_PERIOD[index % len(REFERENCE_PERIOD)] for index in range(WEEKS))
FIELDS = (
    "ret_visible",
    "ret_quantity",
    "ret_cvar05",
    "lost_orders",
    "service_loss_auc",
    "flow_fill_rate",
    "executed_pm_hours",
    "corrective_hours",
    "mass_residual",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def central_cell() -> dict[str, Any]:
    return {
        "sensor_balanced_accuracy": 0.75,
        "pm_restore_fraction": 0.50,
        "wip_capacity_days": 2,
        "wear_heterogeneity": "high",
        "repair_profile": "current",
    }


def extract(outcome: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(outcome["ret_excel_visible"]),
        float(outcome["ration_ret_excel"]),
        float(outcome["ret_excel_cvar05"]),
        float(outcome["lost_orders"]),
        float(outcome["service_loss_auc_ration_hours"]),
        float(outcome["flow_fill_rate"]),
        float(outcome["executed_pm_hours"]),
        float(outcome["corrective_hours"]),
        float(outcome["mass_residual"]),
    )


def evaluate_tape(seed: int, cell: dict[str, Any]) -> dict[str, Any]:
    current_tape = materialize_tape(seed, weeks=WEEKS)
    matrix = np.empty((len(SEQUENCES), len(FIELDS)), dtype=np.float64)
    for index, sequence in enumerate(SEQUENCES):
        matrix[index] = extract(
            run_policy(current_tape, periodic_policy(sequence), cell=cell)
        )
    observable = {
        "worst_condition": extract(
            run_policy(current_tape, worst_condition_policy, cell=cell)
        ),
        "wip_bottleneck": extract(
            run_policy(current_tape, wip_bottleneck_policy, cell=cell)
        ),
    }
    return {"seed": seed, "matrix": matrix, "observable": observable}


def solve_static_mixture(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    reference_index = SEQUENCES.index(REFERENCE)
    means = {field: rows.mean(axis=0) for field, rows in arrays.items()}
    reference = {field: float(rows[reference_index]) for field, rows in means.items()}
    a_ub = np.vstack(
        [
            means["lost_orders"],
            means["service_loss_auc"],
            means["corrective_hours"],
            -means["ret_quantity"],
            -means["ret_cvar05"],
        ]
    )
    b_ub = np.asarray(
        [
            reference["lost_orders"],
            reference["service_loss_auc"],
            reference["corrective_hours"],
            -reference["ret_quantity"],
            -reference["ret_cvar05"],
        ]
    )
    solution = linprog(
        -means["ret_visible"],
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=np.ones((1, len(SEQUENCES))),
        b_eq=np.ones(1),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(f"static mixture LP failed: {solution.message}")
    weights = np.asarray(solution.x, dtype=float)
    feasible = (
        (means["lost_orders"] <= reference["lost_orders"] + 1e-9)
        & (means["service_loss_auc"] <= reference["service_loss_auc"] + 1e-6)
        & (means["corrective_hours"] <= reference["corrective_hours"] + 1e-9)
        & (means["ret_quantity"] >= reference["ret_quantity"] - 1e-12)
        & (means["ret_cvar05"] >= reference["ret_cvar05"] - 1e-12)
    )
    deterministic = int(
        np.argmax(np.where(feasible, means["ret_visible"], -np.inf))
    )
    return {
        "weights": weights,
        "reference_index": reference_index,
        "reference_sequence": "|".join(REFERENCE),
        "reference_means": reference,
        "support": [
            {
                "sequence_index": index,
                "sequence": "|".join(SEQUENCES[index]),
                "weight": float(weight),
            }
            for index, weight in enumerate(weights)
            if weight > 1e-10
        ],
        "deterministic_index": deterministic,
        "deterministic_sequence": "|".join(SEQUENCES[deterministic]),
        "deterministic_means": {
            field: float(rows[deterministic]) for field, rows in means.items()
        },
        "mixture_means": {
            field: float(rows @ weights) for field, rows in means.items()
        },
        "feasible_deterministic_count": int(feasible.sum()),
    }


def solve_pi(
    arrays: dict[str, np.ndarray], comparator: dict[str, np.ndarray]
) -> dict[str, Any]:
    n_tapes, n_sequences = arrays["ret_visible"].shape
    n_variables = n_tapes * n_sequences
    objective = -arrays["ret_visible"].reshape(-1) / n_tapes
    a_ub = sparse.csr_matrix(
        np.vstack(
            [
                arrays["lost_orders"].reshape(-1) / n_tapes,
                arrays["service_loss_auc"].reshape(-1) / n_tapes,
                arrays["corrective_hours"].reshape(-1) / n_tapes,
                -arrays["ret_quantity"].reshape(-1) / n_tapes,
                -arrays["ret_cvar05"].reshape(-1) / n_tapes,
            ]
        )
    )
    b_ub = np.asarray(
        [
            float(comparator["lost_orders"].mean()),
            float(comparator["service_loss_auc"].mean()),
            float(comparator["corrective_hours"].mean()),
            -float(comparator["ret_quantity"].mean()),
            -float(comparator["ret_cvar05"].mean()),
        ]
    )
    tape_rows = np.repeat(np.arange(n_tapes), n_sequences)
    columns = np.arange(n_variables)
    a_eq = sparse.csr_matrix(
        (np.ones(n_variables), (tape_rows, columns)),
        shape=(n_tapes, n_variables),
    )
    solution = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=np.ones(n_tapes),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(f"PI LP failed: {solution.message}")
    weights = np.asarray(solution.x, dtype=float).reshape(n_tapes, n_sequences)
    expected = {
        field: np.sum(weights * values, axis=1)
        for field, values in arrays.items()
    }
    return {
        "expected": expected,
        "nonzero_weight_count": int(np.count_nonzero(weights > 1e-10)),
        "fractional_tape_count": int(
            sum(np.count_nonzero(row > 1e-10) > 1 for row in weights)
        ),
        "objective": float(-solution.fun),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=1_200_001)
    parser.add_argument("--tapes", type=int, default=12)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/paper2_maintenance/request_snapshot_v2_full_frontier/verdict.json",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=ROOT / "results/paper2_maintenance/request_snapshot_v2_full_frontier/raw_matrices.npz",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=ROOT / "outputs/program_j_request_snapshot_v2_frontier/progress.json",
    )
    args = parser.parse_args()
    if args.output.exists() or args.raw_output.exists():
        raise FileExistsError("refusing to overwrite Program-J repair output")
    started = time.perf_counter()
    cell = central_cell()
    seeds = list(range(args.seed_start, args.seed_start + args.tapes))
    results: dict[int, dict[str, Any]] = {}
    atomic_json(
        args.progress,
        {
        "schema_version": "program_j_request_snapshot_v2_frontier_progress_v1",
            "completed_tapes": 0,
            "total_tapes": args.tapes,
            "candidate_schedules_per_tape": len(SEQUENCES),
            "elapsed_seconds": 0.0,
        },
    )
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = {pool.submit(evaluate_tape, seed, cell): seed for seed in seeds}
        for future in as_completed(futures):
            row = future.result()
            results[int(row["seed"])] = row
            atomic_json(
                args.progress,
                {
                    "schema_version": "program_j_request_snapshot_v2_frontier_progress_v1",
                    "completed_tapes": len(results),
                    "total_tapes": args.tapes,
                    "completed_seeds": sorted(results),
                    "candidate_schedules_per_tape": len(SEQUENCES),
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
    ordered = [results[seed] for seed in seeds]
    matrix = np.stack([row["matrix"] for row in ordered], axis=0)
    arrays = {
        field: matrix[:, :, index] for index, field in enumerate(FIELDS)
    }
    observable = {
        name: {
            field: np.asarray(
                [row["observable"][name][index] for row in ordered], dtype=float
            )
            for index, field in enumerate(FIELDS)
        }
        for name in ("worst_condition", "wip_bottleneck")
    }
    static = solve_static_mixture(arrays)
    weights = static.pop("weights")
    comparator = {field: rows @ weights for field, rows in arrays.items()}
    pi = solve_pi(arrays, comparator)
    pi_rows = pi.pop("expected")
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    with args.raw_output.open("wb") as handle:
        np.savez_compressed(
            handle,
            seeds=np.asarray(seeds, dtype=np.int64),
            sequences=np.asarray(SEQUENCES, dtype="U3"),
            field_names=np.asarray(FIELDS, dtype="U32"),
            matrix=matrix,
        )
    sequence_rows = [
        {
            "index": index,
            "sequence": "|".join(sequence),
            "means": {
                field: float(values[:, index].mean())
                for field, values in arrays.items()
            },
        }
        for index, sequence in enumerate(SEQUENCES)
    ]
    observable_summary = {}
    for name, rows in observable.items():
        observable_summary[name] = {
            "means": {field: float(values.mean()) for field, values in rows.items()},
            "delta_ret_visible": float(
                rows["ret_visible"].mean() - comparator["ret_visible"].mean()
            ),
        }
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    result = {
        "schema_version": "program_j_request_snapshot_v2_complete_frontier_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head": head,
        "scientific_status": "RETROSPECTIVE_CENTRAL_CELL_REQUEST_SNAPSHOT_V2_COMPARATOR_REPAIR_NOT_CONFIRMATORY",
        "governing_metric": "ret_excel_request_snapshot_v2",
        "historical_metrics_replaced": [
            "ret_excel_full_ledger",
            "ret_excel_visible_v1_OAT_ledger",
        ],
        "cell": cell,
        "tapes": {
            "seed_start": args.seed_start,
            "n": args.tapes,
            "status": "historical burned Program-J development block",
            "virgin_opened": False,
        },
        "frontier": {
            "weeks": WEEKS,
            "actions": list(ACTIONS),
            "complete_schedule_count": len(SEQUENCES),
            "scheduled_pm_hours_per_policy": WEEKS * 24.0,
            "raw_output": {
                "path": str(args.raw_output.relative_to(ROOT)),
                "sha256": sha256(args.raw_output),
            },
            "sequence_rows": sequence_rows,
            "sequence_rows_sha256": json_sha256(sequence_rows),
        },
        "guardrail_rule": "Use the historical full-ledger best static calendar as an exogenous reference. Static and PI optimization require mean lost orders, service-loss AUC and corrective crew-hours no worse, and mean quantity ReT and CVaR05 no worse. Scheduled preventive hours are identical. No worst-node service metric is implemented for this serial-line adapter, so this result cannot pass the full Paper-2 gate.",
        "static_comparator": {
            **static,
            "development_block_means": {
                field: float(values.mean()) for field, values in comparator.items()
            },
        },
        "perfect_information_relaxation": {
            **pi,
            "development_block_means": {
                field: float(values.mean()) for field, values in pi_rows.items()
            },
            "delta_ret_visible": float(
                pi_rows["ret_visible"].mean()
                - comparator["ret_visible"].mean()
            ),
            "estimate_kind": "complete finite-sample 3^8 convexified PI relaxation under aggregate guardrails; retrospective point estimate",
        },
        "observable_policies": observable_summary,
        "resource_audit": {
            "executed_pm_hours_equal": bool(
                np.allclose(arrays["executed_pm_hours"], WEEKS * 24.0)
            ),
            "corrective_hours_constrained_non_superior": bool(
                pi_rows["corrective_hours"].mean()
                <= comparator["corrective_hours"].mean() + 1e-9
            ),
            "maximum_mass_residual": float(arrays["mass_residual"].max()),
            "corrective_hours_are_endogenous_but_budgeted_in_frontier": True,
        },
        "paper2_gate_eligible": False,
        "gate_blockers": [
            "historical development tapes only",
            "single central cell rather than all 108 frozen cells",
            "no implemented worst-node service endpoint",
            "no optimal observable-policy bound",
            "no confirmatory CI",
        ],
        "claim_limit": "Repairs the central-cell metric and full open-loop schedule omission only. It cannot establish H_obs, a family null, Paper 2 or Paper 3.",
        "elapsed_seconds": time.perf_counter() - started,
    }
    result["content_sha256"] = json_sha256(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    atomic_json(
        args.progress,
        {
            "schema_version": "program_j_request_snapshot_v2_frontier_progress_v1",
            "completed_tapes": args.tapes,
            "total_tapes": args.tapes,
            "candidate_schedules_per_tape": len(SEQUENCES),
            "elapsed_seconds": result["elapsed_seconds"],
            "state": "complete",
            "output": str(args.output),
            "output_sha256": sha256(args.output),
        },
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "elapsed_seconds": result["elapsed_seconds"],
                "static": result["static_comparator"]["development_block_means"],
                "pi": result["perfect_information_relaxation"]["development_block_means"],
                "pi_delta": result["perfect_information_relaxation"][
                    "delta_ret_visible"
                ],
                "observable": observable_summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
