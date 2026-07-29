#!/usr/bin/env python3
"""Produce the complete 108-cell Program-J request-snapshot-v2 frontier.

This is a retrospective development producer, not a scientific claim.  A full
run evaluates all 3^8 eight-week open-loop schedules on four burned screen tapes
in every frozen contract cell.  The comparator and PI optimization use only a
predeclared crew-resource envelope derived from the frozen historical reference
calendar.  Outcome guardrails are reported after optimization and never enter
the optimization constraints.

The producer is designed for six CPU workers.  Every cell/tape matrix is an
immutable checksummed NPZ shard, progress is checkpointed atomically, and an
explicit ``--resume`` verifies rather than overwrites existing shards.  ``--smoke``
uses a small non-scientific slice for tests.  The full grid is never launched by
this module implicitly.
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
import platform
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.maintenance_control import (
    ACTIONS,
    make_sim,
    materialize_tape,
)
from supply_chain.config import HOURS_PER_WEEK
from supply_chain.episode_metrics import compute_episode_metrics
from supply_chain.program_f import advance_including


ROOT = Path(__file__).resolve().parent.parent
WEEKS = 8
ALL_SEQUENCES = tuple(itertools.product(ACTIONS, repeat=WEEKS))
REFERENCE_PERIOD = ("PM6", "PM6", "PM5", "PM6", "PM6")
REFERENCE = tuple(
    REFERENCE_PERIOD[index % len(REFERENCE_PERIOD)] for index in range(WEEKS)
)
SCHEDULED_PM_HOURS = WEEKS * 24.0
MASS_TOLERANCE = 1e-5
FIELDS = (
    "ret_request_snapshot_v2",
    "ret_quantity",
    "ret_cvar05",
    "ret_cvar10",
    "lost_orders",
    "attended_orders",
    "backlog_qty_final",
    "backlog_age_max",
    "service_loss_auc",
    "flow_fill_rate",
    "scheduled_pm_hours",
    "executed_pm_hours",
    "corrective_hours",
    "total_crew_hours",
    "active_crew_partial_hours",
    "crew_queue_at_horizon",
    "blocked_hours",
    "starved_hours",
    "mass_residual",
)
RESOURCE_FIELDS = (
    "total_crew_hours",
)
GUARDRAIL_DIRECTIONS = {
    "ret_quantity": "higher",
    "ret_cvar05": "higher",
    "ret_cvar10": "higher",
    "lost_orders": "lower",
    "attended_orders": "higher",
    "backlog_qty_final": "lower",
    "backlog_age_max": "lower",
    "service_loss_auc": "lower",
    "flow_fill_rate": "higher",
    "blocked_hours": "lower",
    "starved_hours": "lower",
    "mass_residual": "lower",
}
CONTRACT_PATH = ROOT / "contracts/paper2_maintenance_control_v1.json"
METRIC_AUDIT_PATH = (
    ROOT
    / "research/paper2_exhaustive_search/ret_excel_request_snapshot_v2_implementation_audit_20260714.json"
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


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite temporary artifact {temporary}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if path.exists():
        raise FileExistsError(f"refusing race overwrite {path}")
    temporary.replace(path)


def all_cells() -> tuple[dict[str, Any], ...]:
    rows = []
    for index, values in enumerate(
        itertools.product(
            (0.65, 0.75, 0.85),
            (0.30, 0.50, 0.70),
            (1, 2, 3),
            ("low", "high"),
            ("current", "increased"),
        )
    ):
        q, restore, wip, heterogeneity, repair = values
        rows.append(
            {
                "cell_index": index,
                "cell_id": (
                    f"J{index:03d}_Q{int(q * 100)}_R{int(restore * 100)}"
                    f"_W{wip}_H{heterogeneity[0].upper()}_P{repair[0].upper()}"
                ),
                "sensor_balanced_accuracy": q,
                "pm_restore_fraction": restore,
                "wip_capacity_days": wip,
                "wear_heterogeneity": heterogeneity,
                "repair_profile": repair,
            }
        )
    return tuple(rows)


def scientific_cell(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in cell.items()
        if key not in {"cell_index", "cell_id"}
    }


def selected_sequences(*, smoke: bool, smoke_schedules: int) -> tuple[tuple[str, ...], ...]:
    if not smoke:
        return ALL_SEQUENCES
    if smoke_schedules < 2:
        raise ValueError("smoke_schedules must be at least 2")
    rows = list(ALL_SEQUENCES[: smoke_schedules - 1])
    if REFERENCE not in rows:
        rows.append(REFERENCE)
    return tuple(rows)


def extract(outcome: dict[str, Any]) -> tuple[float, ...]:
    if outcome.get("ret_excel_contract_version") != "ret_excel_request_snapshot_v2":
        raise RuntimeError("Program-J outcome is not request-snapshot-v2")
    blocked = sum(float(value) for value in outcome["blocked_hours"].values())
    starved = sum(float(value) for value in outcome["starved_hours"].values())
    executed = float(outcome["executed_pm_hours"])
    corrective = float(outcome["corrective_hours"])
    return (
        float(outcome["ret_excel_visible"]),
        float(outcome["ration_ret_excel"]),
        float(outcome["ret_excel_cvar05"]),
        float(outcome["ret_excel_cvar10"]),
        float(outcome["lost_orders"]),
        float(outcome["n_served"]),
        float(outcome["backorder_qty_final"]),
        float(outcome["backlog_age_max"]),
        float(outcome["service_loss_auc_ration_hours"]),
        float(outcome["flow_fill_rate"]),
        float(outcome["scheduled_pm_hours"]),
        executed,
        corrective,
        float(outcome["total_in_horizon_crew_hours"]),
        float(outcome["active_crew_partial_hours"]),
        float(outcome["crew_queue_at_horizon"]),
        blocked,
        starved,
        float(outcome["mass_residual"]),
    )


def run_sequence_with_exact_crew_ledger(
    tape: dict[str, Any],
    sequence: Sequence[str],
    *,
    cell: dict[str, Any],
) -> dict[str, Any]:
    """Run one schedule while retaining exact in-horizon crew occupancy.

    The core convenience runner exposes completed maintenance hours only. SimPy
    request objects retain ``usage_since`` for a job active at the horizon, so
    this wrapper adds its consumed partial interval without editing core physics.
    Queued-but-not-started work consumes zero crew-hours and is reported.
    """
    sim, controller, start = make_sim(tape, cell)
    end = start + int(tape["weeks"]) * HOURS_PER_WEEK
    for week, action in enumerate(sequence):
        controller.request(str(action), week)
        advance_including(sim, min(end, start + (week + 1) * HOURS_PER_WEEK))
    metrics = compute_episode_metrics(sim, treatment_start=start)
    ledger = sim.flow_ledger()
    completed_hours = sum(float(record.hours) for record in controller.records)
    active_partial = sum(
        max(0.0, end - max(start, float(request.usage_since)))
        for request in controller.crew.users
    )
    total_crew = completed_hours + active_partial
    if total_crew > end - start + 1e-9:
        raise RuntimeError("single-crew in-horizon occupancy exceeds capacity")
    metrics.update(controller.exogenous_artifacts())
    metrics.update(
        {
            "scheduled_pm_hours": float(controller.scheduled_pm_hours),
            "executed_pm_hours": float(controller.executed_pm_hours),
            "corrective_hours": float(controller.corrective_hours),
            "total_in_horizon_crew_hours": float(total_crew),
            "active_crew_partial_hours": float(active_partial),
            "crew_queue_at_horizon": float(len(controller.crew.queue)),
            "mass_residual": max(
                abs(float(ledger["raw_residual"])),
                abs(float(ledger["ration_residual"])),
            ),
            "blocked_hours": dict(controller.blocked_hours),
            "starved_hours": dict(controller.starved_hours),
        }
    )
    return metrics


def evaluate_cell_tape(
    cell: dict[str, Any],
    seed: int,
    sequences: Sequence[tuple[str, ...]],
) -> dict[str, Any]:
    tape = materialize_tape(seed, weeks=WEEKS)
    matrix = np.empty((len(sequences), len(FIELDS)), dtype=np.float64)
    for sequence_index, sequence in enumerate(sequences):
        outcome = run_sequence_with_exact_crew_ledger(
            tape,
            tuple(sequence),
            cell=scientific_cell(cell),
        )
        if outcome["base_exogenous_sha256"] != tape["base_exogenous_sha256"]:
            raise RuntimeError("base exogenous hash drift")
        matrix[sequence_index] = extract(outcome)
    scheduled_index = FIELDS.index("scheduled_pm_hours")
    mass_index = FIELDS.index("mass_residual")
    if not np.allclose(
        matrix[:, scheduled_index], SCHEDULED_PM_HOURS, atol=0.0, rtol=0.0
    ):
        raise RuntimeError("scheduled PM hours violate the frozen resource contract")
    if float(np.max(np.abs(matrix[:, mass_index]))) > MASS_TOLERANCE:
        raise RuntimeError("mass-conservation tolerance failed")
    return {
        "cell_index": int(cell["cell_index"]),
        "cell_id": str(cell["cell_id"]),
        "cell": scientific_cell(cell),
        "seed": int(seed),
        "base_exogenous_sha256": str(tape["base_exogenous_sha256"]),
        "matrix": matrix,
    }


def shard_path(shard_root: Path, cell_index: int, seed: int) -> Path:
    return shard_root / f"cell_{cell_index:03d}" / f"seed_{seed}.npz"


def write_raw_shard(
    path: Path,
    row: dict[str, Any],
    sequences: Sequence[tuple[str, ...]],
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite raw shard {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite temporary shard {temporary}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            cell_index=np.asarray([row["cell_index"]], dtype=np.int64),
            cell_id=np.asarray([row["cell_id"]], dtype="U64"),
            cell_json=np.asarray(
                [json.dumps(row["cell"], sort_keys=True, separators=(",", ":"))],
                dtype="U512",
            ),
            seed=np.asarray([row["seed"]], dtype=np.int64),
            base_exogenous_sha256=np.asarray(
                [row["base_exogenous_sha256"]], dtype="U64"
            ),
            sequence_sha256=np.asarray(
                [json_sha256([list(sequence) for sequence in sequences])], dtype="U64"
            ),
            sequences=np.asarray(sequences, dtype="U3"),
            field_names=np.asarray(FIELDS, dtype="U40"),
            matrix=row["matrix"],
        )
    if path.exists():
        raise FileExistsError(f"refusing race overwrite {path}")
    temporary.replace(path)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "cell_index": int(row["cell_index"]),
        "cell_id": str(row["cell_id"]),
        "seed": int(row["seed"]),
        "base_exogenous_sha256": str(row["base_exogenous_sha256"]),
    }


def load_raw_shard(
    record: dict[str, Any],
    *,
    sequences: Sequence[tuple[str, ...]],
) -> np.ndarray:
    path = Path(record["path"])
    if not path.exists() or sha256(path) != record["sha256"]:
        raise RuntimeError(f"raw shard checksum failed: {path}")
    with np.load(path, allow_pickle=False) as raw:
        matrix = np.asarray(raw["matrix"], dtype=float)
        if raw["field_names"].tolist() != list(FIELDS):
            raise RuntimeError(f"field schema mismatch: {path}")
        if raw["sequences"].tolist() != [list(row) for row in sequences]:
            raise RuntimeError(f"sequence schema mismatch: {path}")
        if raw["base_exogenous_sha256"].tolist() != [
            record["base_exogenous_sha256"]
        ]:
            raise RuntimeError(f"tape hash mismatch: {path}")
    if matrix.shape != (len(sequences), len(FIELDS)) or not np.isfinite(matrix).all():
        raise RuntimeError(f"raw matrix invalid: {path}")
    return matrix


def arrays_from_matrix(matrix: np.ndarray) -> dict[str, np.ndarray]:
    return {
        field: matrix[:, :, field_index]
        for field_index, field in enumerate(FIELDS)
    }


def resource_envelope(arrays: dict[str, np.ndarray], reference_index: int) -> dict[str, float]:
    return {
        "scheduled_pm_hours_exact": SCHEDULED_PM_HOURS,
        "B_total_crew_hours": float(
            arrays["total_crew_hours"][:, reference_index].mean()
        ),
        "measurement": "Exact occupied single-crew hours in the eight-week decision horizon: completed maintenance-record durations plus the consumed interval of any active SimPy crew request at the horizon. Queued work consumes zero hours.",
    }


def solve_resource_only_static(
    arrays: dict[str, np.ndarray],
    envelope: dict[str, float],
) -> dict[str, Any]:
    means = {field: values.mean(axis=0) for field, values in arrays.items()}
    feasible = np.ones_like(means["ret_request_snapshot_v2"], dtype=bool)
    for field in RESOURCE_FIELDS:
        feasible &= means[field] <= envelope[f"B_{field}"] + 1e-9
    if not feasible.any():
        raise RuntimeError("reference-derived crew envelope has no feasible schedule")
    deterministic_index = int(
        np.argmax(
            np.where(feasible, means["ret_request_snapshot_v2"], -np.inf)
        )
    )
    a_ub = np.vstack([means[field] for field in RESOURCE_FIELDS])
    b_ub = np.asarray([envelope[f"B_{field}"] for field in RESOURCE_FIELDS])
    solution = linprog(
        -means["ret_request_snapshot_v2"],
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=np.ones((1, len(feasible))),
        b_eq=np.ones(1),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(f"resource-only static LP failed: {solution.message}")
    weights = np.asarray(solution.x, dtype=float)
    expected = {field: values @ weights for field, values in arrays.items()}
    if (
        expected["ret_request_snapshot_v2"].mean()
        + 1e-10
        < means["ret_request_snapshot_v2"][deterministic_index]
    ):
        raise RuntimeError("convexified comparator weaker than deterministic comparator")
    return {
        "weights": weights,
        "expected": expected,
        "feasible_deterministic_count": int(feasible.sum()),
        "deterministic_index": deterministic_index,
        "deterministic_means": {
            field: float(values[deterministic_index])
            for field, values in means.items()
        },
        "convexified_means": {
            field: float(values.mean()) for field, values in expected.items()
        },
        "support_count": int(np.count_nonzero(weights > 1e-10)),
    }


def solve_resource_only_pi(
    arrays: dict[str, np.ndarray],
    envelope: dict[str, float],
) -> dict[str, Any]:
    n_tapes, n_sequences = arrays["ret_request_snapshot_v2"].shape
    n_variables = n_tapes * n_sequences
    a_ub = sparse.csr_matrix(
        np.vstack(
            [arrays[field].reshape(-1) / n_tapes for field in RESOURCE_FIELDS]
        )
    )
    b_ub = np.asarray([envelope[f"B_{field}"] for field in RESOURCE_FIELDS])
    tape_rows = np.repeat(np.arange(n_tapes), n_sequences)
    columns = np.arange(n_variables)
    a_eq = sparse.csr_matrix(
        (np.ones(n_variables), (tape_rows, columns)),
        shape=(n_tapes, n_variables),
    )
    solution = linprog(
        -arrays["ret_request_snapshot_v2"].reshape(-1) / n_tapes,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=np.ones(n_tapes),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(f"resource-only PI LP failed: {solution.message}")
    weights = np.asarray(solution.x, dtype=float).reshape(n_tapes, n_sequences)
    expected = {
        field: np.sum(weights * values, axis=1)
        for field, values in arrays.items()
    }
    return {
        "weights": weights,
        "expected": expected,
        "means": {field: float(values.mean()) for field, values in expected.items()},
        "nonzero_weight_count": int(np.count_nonzero(weights > 1e-10)),
        "fractional_tape_count": int(
            sum(np.count_nonzero(row > 1e-10) > 1 for row in weights)
        ),
    }


def guardrail_report(
    candidate: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
) -> dict[str, Any]:
    rows = {}
    for field, direction in GUARDRAIL_DIRECTIONS.items():
        delta = float(candidate[field].mean() - baseline[field].mean())
        rows[field] = {
            "direction": direction,
            "candidate_mean": float(candidate[field].mean()),
            "baseline_mean": float(baseline[field].mean()),
            "delta_candidate_minus_baseline": delta,
            "point_noninferiority": delta >= -1e-12 if direction == "higher" else delta <= 1e-12,
        }
    return {
        "rows": rows,
        "all_point_noninferiority": bool(
            all(row["point_noninferiority"] for row in rows.values())
        ),
        "optimization_use": "REPORT_ONLY_NOT_AN_OPTIMIZATION_CONSTRAINT",
    }


def sparse_support(
    weights: np.ndarray,
    sequences: Sequence[tuple[str, ...]],
) -> list[dict[str, Any]]:
    flat = np.asarray(weights).reshape(-1)
    n_sequences = len(sequences)
    rows = []
    for flat_index in np.flatnonzero(flat > 1e-10):
        tape_index, sequence_index = divmod(int(flat_index), n_sequences)
        row = {
            "sequence_index": sequence_index,
            "sequence": "|".join(sequences[sequence_index]),
            "weight": float(flat[flat_index]),
        }
        if np.asarray(weights).ndim == 2:
            row["tape_index"] = tape_index
        rows.append(row)
    return rows


def solve_cell(
    cell: dict[str, Any],
    shard_records: Sequence[dict[str, Any]],
    *,
    sequences: Sequence[tuple[str, ...]],
    solver_root: Path,
) -> dict[str, Any]:
    ordered = sorted(shard_records, key=lambda row: int(row["seed"]))
    matrix = np.stack(
        [load_raw_shard(row, sequences=sequences) for row in ordered], axis=0
    )
    arrays = arrays_from_matrix(matrix)
    reference_index = sequences.index(REFERENCE)
    if not np.allclose(
        arrays["scheduled_pm_hours"], SCHEDULED_PM_HOURS, atol=0.0, rtol=0.0
    ):
        raise RuntimeError("scheduled PM-hour contract failed before optimization")
    if float(np.max(np.abs(arrays["mass_residual"]))) > MASS_TOLERANCE:
        raise RuntimeError("mass-conservation gate failed before optimization")
    envelope = resource_envelope(arrays, reference_index)
    static = solve_resource_only_static(arrays, envelope)
    static_weights = static.pop("weights")
    static_expected = static.pop("expected")
    pi = solve_resource_only_pi(arrays, envelope)
    pi_weights = pi.pop("weights")
    pi_expected = pi.pop("expected")
    reference = {
        field: values[:, reference_index] for field, values in arrays.items()
    }
    solver_path = solver_root / f"cell_{int(cell['cell_index']):03d}.npz"
    if solver_path.exists():
        raise FileExistsError(f"refusing to overwrite solver shard {solver_path}")
    solver_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = solver_path.with_suffix(solver_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            static_weights=static_weights,
            pi_weights=pi_weights,
            seeds=np.asarray([row["seed"] for row in ordered], dtype=np.int64),
            sequence_sha256=np.asarray(
                [json_sha256([list(sequence) for sequence in sequences])], dtype="U64"
            ),
        )
    temporary.replace(solver_path)
    static_support = sparse_support(static_weights, sequences)
    pi_support = sparse_support(pi_weights, sequences)
    result = {
        "cell_index": int(cell["cell_index"]),
        "cell_id": str(cell["cell_id"]),
        "cell": scientific_cell(cell),
        "seeds": [int(row["seed"]) for row in ordered],
        "base_exogenous_sha256": [row["base_exogenous_sha256"] for row in ordered],
        "resource_envelope": envelope,
        "reference": {
            "sequence_index": reference_index,
            "sequence": "|".join(REFERENCE),
            "means": {field: float(values.mean()) for field, values in reference.items()},
        },
        "static_resource_only": {
            **static,
            "estimate_kind": "Strongest deterministic schedule and strongest convexified full-horizon open-loop mixture under only the frozen crew-hour envelope.",
            "deterministic_sequence": "|".join(
                sequences[static["deterministic_index"]]
            ),
            "convexified_support": static_support,
            "guardrails_vs_reference": guardrail_report(static_expected, reference),
        },
        "pi_resource_only": {
            **pi,
            "estimate_kind": "Complete finite-sample tape-contingent convexified perfect-information relaxation over every enumerated full-horizon schedule, constrained only by the frozen aggregate crew-hour envelope.",
            "support": pi_support,
            "delta_ret_vs_static_convexified": float(
                pi_expected["ret_request_snapshot_v2"].mean()
                - static_expected["ret_request_snapshot_v2"].mean()
            ),
            "guardrails_vs_static": guardrail_report(pi_expected, static_expected),
            "guardrails_vs_reference": guardrail_report(pi_expected, reference),
        },
        "solver_shard": {
            "path": str(solver_path),
            "sha256": sha256(solver_path),
            "bytes": solver_path.stat().st_size,
        },
        "raw_shards": [
            {key: row[key] for key in ("path", "sha256", "bytes", "seed", "base_exogenous_sha256")}
            for row in ordered
        ],
        "raw_matrix_shape": list(matrix.shape),
        "maximum_abs_mass_residual": float(np.max(np.abs(arrays["mass_residual"]))),
        "resource_acceptance": {
            "scheduled_pm_hours_exact_for_every_schedule": True,
            "reference_total_crew_hours": envelope["B_total_crew_hours"],
            "static_convexified_total_crew_hours": float(
                static_expected["total_crew_hours"].mean()
            ),
            "pi_total_crew_hours": float(pi_expected["total_crew_hours"].mean()),
            "static_within_envelope": bool(
                static_expected["total_crew_hours"].mean()
                <= envelope["B_total_crew_hours"] + 1e-9
            ),
            "pi_within_envelope": bool(
                pi_expected["total_crew_hours"].mean()
                <= envelope["B_total_crew_hours"] + 1e-9
            ),
            "exact_partial_active_work_included": True,
        },
        "outcome_guardrails_used_in_optimization": False,
        "scientific_claim": None,
    }
    result["content_sha256"] = json_sha256(result)
    return result


def source_manifest() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        CONTRACT_PATH,
        METRIC_AUDIT_PATH,
        ROOT / "supply_chain/maintenance_control.py",
        ROOT / "supply_chain/episode_metrics.py",
        ROOT / "supply_chain/ret_thesis.py",
        ROOT / "supply_chain/supply_chain.py",
    )
    return {
        str(path.resolve().relative_to(ROOT.resolve())): sha256(path)
        for path in paths
    }


def run_contract(
    *,
    cells: Sequence[dict[str, Any]],
    sequences: Sequence[tuple[str, ...]],
    seed_start: int,
    tapes_per_cell: int,
    smoke: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "program_j_request_snapshot_v2_108cell_run_contract_v1",
        "mode": "SMOKE_NONSCIENTIFIC" if smoke else "FULL_108_CELL_DEVELOPMENT",
        "metric": "ret_excel_request_snapshot_v2",
        "weeks": WEEKS,
        "cells": list(cells),
        "cell_count": len(cells),
        "cell_design_sha256": json_sha256(list(cells)),
        "sequences": [list(sequence) for sequence in sequences],
        "sequence_count": len(sequences),
        "sequence_sha256": json_sha256([list(sequence) for sequence in sequences]),
        "reference_sequence": list(REFERENCE),
        "seed_start": seed_start,
        "tapes_per_cell": tapes_per_cell,
        "seed_rule": "The same consecutive burned Program-J screen seeds are reused in every cell for CRN.",
        "tape_status": "BURNED_DEVELOPMENT_SCREEN_TAPES_NOT_VIRGIN",
        "screening_role": "Global finite-grid sensitivity screen of the resource-restricted finite-sample PI relaxation. Four tapes per cell locate mechanisms or conservative ceiling-stress cells; they cannot establish a population ceiling, H_obs, a family null, or a positive.",
        "scheduled_pm_hours_exact": SCHEDULED_PM_HOURS,
        "resource_envelope_rule": "Every schedule requests exactly 192 preventive crew-hours. Within each cell, freeze B_total_crew_hours to the historical reference calendar's mean exact occupied single-crew hours on the same burned tapes, including partial active work at the horizon. No outcome guardrail enters this envelope.",
        "optimization_rule": "Maximize request-snapshot-v2 ReT subject only to the crew-resource envelope. Outcome guardrails never constrain or select the comparator or PI.",
        "guardrail_directions": GUARDRAIL_DIRECTIONS,
        "guardrail_coverage": {
            "quantity_weighted_ret": "measured",
            "service_loss_auc": "measured",
            "attended_and_lost_orders": "measured",
            "backlog_quantity_and_maximum_age": "measured",
            "tail_risk": "measured",
            "resource_ledgers": "measured",
            "worst_node_or_cssu_service": "UNAVAILABLE_IN_CURRENT_PROGRAM_J_SINGLE_LINE_ADAPTER",
            "promotion_block": "No Program-J cell can be promoted to a Paper-2 candidate until a physically meaningful node-level service guardrail is defined and frozen, or Garrido confirms that it is not applicable to this single-line contract.",
        },
        "post_screen_selection_rule": "Rank every cell by the resource-restricted PI delta only to stress-test the largest possible ceiling. If any adjacent plausible cells clear the 0.01 practical gate, freeze the least-favorable cell in that connected region before any H_obs policy search. A maximum-gap cell alone is never selected as a positive contract.",
        "cell_adjacency_rule": "Two cells are adjacent only when exactly one design factor changes by one neighboring declared level; the two binary factors may flip. Connected regions use this undirected graph.",
        "metric_domain_status": "PROVISIONAL_PENDING_GARRIDO_M1_CONFIRMATION_OF_REQUEST_SNAPSHOT_AND_SAME_TIMESTAMP_ORDERING",
        "mass_tolerance": MASS_TOLERANCE,
        "source_sha256": source_manifest(),
    }


def progress_payload(
    contract: dict[str, Any],
    *,
    frozen_contract_path: Path,
    state: str,
    started_at_utc: str,
    started_perf: float,
    shard_records: Sequence[dict[str, Any]],
    completed_cells: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    total_shards = int(contract["cell_count"] * contract["tapes_per_cell"])
    return {
        "schema_version": "program_j_request_snapshot_v2_108cell_progress_v1",
        "state": state,
        "started_at_utc": started_at_utc,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_contract_sha256": json_sha256(contract),
        "frozen_run_contract": {
            "path": str(frozen_contract_path),
            "sha256": sha256(frozen_contract_path),
        },
        "mode": contract["mode"],
        "completed_raw_shards": len(shard_records),
        "total_raw_shards": total_shards,
        "completed_cells": len(completed_cells),
        "total_cells": contract["cell_count"],
        "candidate_schedules_per_tape": contract["sequence_count"],
        "completed_des_evaluations": len(shard_records) * contract["sequence_count"],
        "total_des_evaluations": total_shards * contract["sequence_count"],
        "elapsed_seconds": time.perf_counter() - started_perf,
        "raw_shards": sorted(
            shard_records,
            key=lambda row: (int(row["cell_index"]), int(row["seed"])),
        ),
        "cell_summaries": sorted(
            completed_cells, key=lambda row: int(row["cell_index"])
        ),
    }


def validate_resume(
    progress: dict[str, Any],
    contract: dict[str, Any],
    sequences: Sequence[tuple[str, ...]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if progress.get("run_contract_sha256") != json_sha256(contract):
        raise RuntimeError("resume contract hash mismatch")
    shards = list(progress.get("raw_shards", []))
    for record in shards:
        load_raw_shard(record, sequences=sequences)
    summaries = list(progress.get("cell_summaries", []))
    for record in summaries:
        path = Path(record["path"])
        if not path.exists() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"cell summary checksum failed: {path}")
    return shards, summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed-start", type=int, default=1_200_001)
    parser.add_argument("--tapes-per-cell", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-cells", type=int, default=2)
    parser.add_argument("--smoke-tapes", type=int, default=1)
    parser.add_argument("--smoke-schedules", type=int, default=3)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=ROOT / "outputs/program_j_request_snapshot_v2_108cell_frontier_v1",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=ROOT
        / "outputs/program_j_request_snapshot_v2_108cell_frontier_v1/progress.json",
    )
    parser.add_argument(
        "--run-contract",
        type=Path,
        default=None,
        help="Immutable pre-science run contract; defaults inside --work-root.",
    )
    parser.add_argument(
        "--use-precreated-run-contract",
        action="store_true",
        help="Require and verify a launcher-created contract before any DES evaluation.",
    )
    parser.add_argument(
        "--verdict",
        type=Path,
        default=ROOT
        / "results/paper2_maintenance/request_snapshot_v2_108cell_frontier_v1/verdict.json",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 6:
        raise ValueError("workers must be between 1 and the preregistered VPS limit 6")
    cells = all_cells()
    sequences = selected_sequences(
        smoke=args.smoke, smoke_schedules=args.smoke_schedules
    )
    tapes_per_cell = args.smoke_tapes if args.smoke else args.tapes_per_cell
    if args.smoke:
        cells = cells[: args.smoke_cells]
    elif len(cells) != 108 or len(sequences) != 3**8:
        raise AssertionError("full mode must retain all 108 cells and all 3^8 schedules")
    if REFERENCE not in sequences:
        raise AssertionError("historical resource reference omitted")
    if tapes_per_cell < 1:
        raise ValueError("tapes_per_cell must be positive")
    if args.verdict.exists():
        raise FileExistsError(f"refusing to overwrite final verdict {args.verdict}")

    contract = run_contract(
        cells=cells,
        sequences=sequences,
        seed_start=args.seed_start,
        tapes_per_cell=tapes_per_cell,
        smoke=args.smoke,
    )
    started_perf = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    frozen_contract_path = (
        args.run_contract
        if args.run_contract is not None
        else args.work_root / "frozen_run_contract.json"
    )
    if args.resume:
        if not args.progress.exists():
            raise FileNotFoundError("--resume requires an existing progress checkpoint")
        if not frozen_contract_path.exists():
            raise FileNotFoundError("--resume requires the frozen run contract")
        frozen = json.loads(frozen_contract_path.read_text())
        stored_content = frozen.pop("content_sha256", None)
        if stored_content != json_sha256(frozen) or frozen != contract:
            raise RuntimeError("frozen run contract content mismatch")
        checkpoint = json.loads(args.progress.read_text())
        shard_records, completed_cell_records = validate_resume(
            checkpoint, contract, sequences
        )
        started_at = str(checkpoint["started_at_utc"])
    else:
        if args.work_root.exists() or args.progress.exists():
            raise FileExistsError("fresh run refuses an existing work root or checkpoint")
        args.work_root.mkdir(parents=True)
        if args.use_precreated_run_contract:
            if not frozen_contract_path.exists():
                raise FileNotFoundError(
                    "--use-precreated-run-contract requires the frozen contract"
                )
            frozen = json.loads(frozen_contract_path.read_text())
            stored_content = frozen.pop("content_sha256", None)
            if stored_content != json_sha256(frozen) or frozen != contract:
                raise RuntimeError("precreated run contract content mismatch")
        else:
            frozen = dict(contract)
            frozen["content_sha256"] = json_sha256(frozen)
            write_json_new(frozen_contract_path, frozen)
        shard_records, completed_cell_records = [], []

    shard_root = args.work_root / "raw_shards"
    solver_root = args.work_root / "solver_shards"
    summary_root = args.work_root / "cell_summaries"
    existing_keys = {
        (int(row["cell_index"]), int(row["seed"])) for row in shard_records
    }
    tasks = []
    for cell in cells:
        for offset in range(tapes_per_cell):
            seed = args.seed_start + offset
            if (int(cell["cell_index"]), seed) not in existing_keys:
                tasks.append((cell, seed))
    atomic_json(
        args.progress,
        progress_payload(
            contract,
            state="evaluating_raw_shards",
            frozen_contract_path=frozen_contract_path,
            started_at_utc=started_at,
            started_perf=started_perf,
            shard_records=shard_records,
            completed_cells=completed_cell_records,
        ),
    )
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = {
            pool.submit(evaluate_cell_tape, cell, seed, sequences): (cell, seed)
            for cell, seed in tasks
        }
        for future in as_completed(futures):
            row = future.result()
            path = shard_path(
                shard_root, int(row["cell_index"]), int(row["seed"])
            )
            shard_records.append(write_raw_shard(path, row, sequences))
            atomic_json(
                args.progress,
                progress_payload(
                    contract,
                    state="evaluating_raw_shards",
                    frozen_contract_path=frozen_contract_path,
                    started_at_utc=started_at,
                    started_perf=started_perf,
                    shard_records=shard_records,
                    completed_cells=completed_cell_records,
                ),
            )

    completed_cell_indices = {
        int(row["cell_index"]) for row in completed_cell_records
    }
    for cell in cells:
        cell_index = int(cell["cell_index"])
        if cell_index in completed_cell_indices:
            continue
        records = [
            row for row in shard_records if int(row["cell_index"]) == cell_index
        ]
        if len(records) != tapes_per_cell:
            raise RuntimeError(f"cell {cell_index} has incomplete raw shards")
        summary = solve_cell(
            cell, records, sequences=sequences, solver_root=solver_root
        )
        summary_path = summary_root / f"cell_{cell_index:03d}.json"
        write_json_new(summary_path, summary)
        completed_cell_records.append(
            {
                "cell_index": cell_index,
                "cell_id": cell["cell_id"],
                "path": str(summary_path),
                "sha256": sha256(summary_path),
                "content_sha256": summary["content_sha256"],
            }
        )
        atomic_json(
            args.progress,
            progress_payload(
                contract,
                state="solving_cells",
                frozen_contract_path=frozen_contract_path,
                started_at_utc=started_at,
                started_perf=started_perf,
                shard_records=shard_records,
                completed_cells=completed_cell_records,
            ),
        )

    cell_results = [
        json.loads(Path(row["path"]).read_text())
        for row in sorted(completed_cell_records, key=lambda item: int(item["cell_index"]))
    ]
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    elapsed = time.perf_counter() - started_perf
    result = {
        "schema_version": "program_j_request_snapshot_v2_108cell_frontier_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head": head,
        "mode": contract["mode"],
        "scientific_status": (
            "SMOKE_ONLY_NO_SCIENTIFIC_EVIDENCE"
            if args.smoke
            else "RETROSPECTIVE_FULL_CELL_RESOURCE_ONLY_FRONTIER_DEVELOPMENT_NOT_CONFIRMATORY"
        ),
        "scientific_claim": None,
        "run_contract": contract,
        "run_contract_sha256": json_sha256(contract),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "workers": args.workers,
        },
        "execution": {
            "raw_shards": len(shard_records),
            "cell_summaries": len(cell_results),
            "des_evaluations": len(shard_records) * len(sequences),
            "elapsed_seconds": elapsed,
            "virgin_tapes_opened": False,
            "progress_path": str(args.progress),
        },
        "cell_results": cell_results,
        "max_resource_restricted_pi_delta": float(
            max(
                row["pi_resource_only"]["delta_ret_vs_static_convexified"]
                for row in cell_results
            )
        ),
        "screen_decision_rule": {
            "if_all_cells_below_0_01": "This is evidence for prioritizing a deeper confidence-bounded audit of the largest-delta and adjacent cells; four tapes do not certify a family ceiling.",
            "if_any_cells_at_or_above_0_01": "Freeze the least-favorable plausible connected cell region, add the missing node-service guardrail, and run a fresh-tape strong observable-policy gate before any learner.",
            "learner_authorized_by_this_screen": False,
        },
        "guardrail_policy": "Every outcome guardrail is reported post-optimization. None is permitted to remove a high-ReT resource-feasible schedule or weaken the comparator/PI.",
        "claim_limit": "Producer output is a finite burned-tape development calculation only. It does not establish a population ceiling, H_obs, a Program-J family null/positive, Paper 2, or Paper 3.",
    }
    result["content_sha256"] = json_sha256(result)
    write_json_new(args.verdict, result)
    atomic_json(
        args.progress,
        {
            **progress_payload(
                contract,
                state="complete",
                frozen_contract_path=frozen_contract_path,
                started_at_utc=started_at,
                started_perf=started_perf,
                shard_records=shard_records,
                completed_cells=completed_cell_records,
            ),
            "verdict": str(args.verdict),
            "verdict_sha256": sha256(args.verdict),
        },
    )
    print(
        json.dumps(
            {
                "verdict": str(args.verdict),
                "mode": result["mode"],
                "cells": len(cell_results),
                "schedules_per_tape": len(sequences),
                "des_evaluations": result["execution"]["des_evaluations"],
                "elapsed_seconds": elapsed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
