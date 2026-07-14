#!/usr/bin/env python3
"""Replay the complete Program-H four-week frontier under visible-v1.

This is a retrospective burned-tape metric repair, not a preregistration or a
confirmatory run.  It reports both the unconstrained sparse-ledger degeneracy
and a frozen loss/fairness/quantity-constrained convexified comparator and
perfect-information relaxation.  It cannot authorize a learner or Paper 3.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import HOURS_PER_WEEK
from supply_chain.program_g import ACTIONS, materialize_tape, metrics_all
from supply_chain.ret_thesis import compute_order_level_ret_excel_visible_ledger


ROOT = Path(__file__).resolve().parent.parent
REGION = [
    {
        "cell_id": f"P{p}_Q{int(q * 100)}_L{lead}_S150",
        "signal_q": q,
        "lead_weeks": lead,
        "surge_mult": 1.50,
        "persistence": p,
        "r22_weekly_prob": 0.05,
    }
    for p in ("short", "long")
    for q in (0.65, 0.75, 0.85)
    for lead in (1, 2)
]
WEEKS = 4
ARM = "TRS"
SEQUENCES = tuple(itertools.product(ACTIONS, repeat=WEEKS))
REFERENCE = ("A", "B", "A", "B")
FIELDS = ("ret_visible", "ret_quantity", "lost_orders", "worst_cssu_fill")


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


def tape(index: int, start: int):
    return materialize_tape(
        start + index,
        REGION[index % len(REGION)],
        WEEKS,
        persistent=True,
    )


def evaluate_one(current_tape, sequence: tuple[str, ...]) -> dict[str, float]:
    metrics = metrics_all(current_tape, sequence, ARM)
    visible = compute_order_level_ret_excel_visible_ledger(
        metrics["orders"], current_time=WEEKS * HOURS_PER_WEEK
    )
    return {
        "ret_visible": float(visible["mean_ret_excel"]),
        "ret_quantity": float(metrics["ret_quantity"]),
        "lost_orders": float(metrics["lost_orders"]),
        "worst_cssu_fill": float(metrics["worst_cssu_fill"]),
    }


def evaluate_block(tapes) -> dict[str, np.ndarray]:
    arrays = {
        field: np.empty((len(tapes), len(SEQUENCES)), dtype=np.float64)
        for field in FIELDS
    }
    for tape_index, current_tape in enumerate(tapes):
        for sequence_index, sequence in enumerate(SEQUENCES):
            row = evaluate_one(current_tape, sequence)
            for field in FIELDS:
                arrays[field][tape_index, sequence_index] = row[field]
    return arrays


def solve_comparator(
    arrays: dict[str, np.ndarray],
    *,
    worst_fill_margin: float,
) -> dict[str, Any]:
    reference_index = SEQUENCES.index(REFERENCE)
    means = {field: values.mean(axis=0) for field, values in arrays.items()}
    reference = {field: float(values[reference_index]) for field, values in means.items()}
    constraints = np.vstack(
        [
            means["lost_orders"],
            -means["worst_cssu_fill"],
            -means["ret_quantity"],
        ]
    )
    limits = np.asarray(
        [
            reference["lost_orders"],
            -(reference["worst_cssu_fill"] - worst_fill_margin),
            -reference["ret_quantity"],
        ]
    )
    solution = linprog(
        -means["ret_visible"],
        A_ub=constraints,
        b_ub=limits,
        A_eq=np.ones((1, len(SEQUENCES))),
        b_eq=np.ones(1),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(f"comparator LP failed: {solution.message}")
    weights = np.asarray(solution.x, dtype=float)
    feasible = (
        (means["lost_orders"] <= reference["lost_orders"] + 1e-12)
        & (
            means["worst_cssu_fill"]
            >= reference["worst_cssu_fill"] - worst_fill_margin - 1e-12
        )
        & (means["ret_quantity"] >= reference["ret_quantity"] - 1e-12)
    )
    deterministic_index = int(
        np.argmax(np.where(feasible, means["ret_visible"], -np.inf))
    )
    support = [
        {
            "sequence_index": index,
            "sequence": "".join(SEQUENCES[index]),
            "weight": float(weight),
        }
        for index, weight in enumerate(weights)
        if weight > 1e-10
    ]
    return {
        "weights": weights,
        "support": support,
        "reference": reference,
        "deterministic_index": deterministic_index,
        "deterministic_sequence": "".join(SEQUENCES[deterministic_index]),
        "deterministic_mean_ret_visible": float(means["ret_visible"][deterministic_index]),
        "mixture_calibration_means": {
            field: float(values @ weights) for field, values in means.items()
        },
        "feasible_deterministic_count": int(feasible.sum()),
    }


def solve_guardrailed_pi_relaxation(
    arrays: dict[str, np.ndarray],
    comparator_rows: dict[str, np.ndarray],
    *,
    worst_fill_margin: float,
) -> dict[str, Any]:
    n_tapes, n_sequences = arrays["ret_visible"].shape
    n_variables = n_tapes * n_sequences
    objective = -arrays["ret_visible"].reshape(-1) / n_tapes
    a_ub = sparse.csr_matrix(
        np.vstack(
            [
                arrays["lost_orders"].reshape(-1) / n_tapes,
                -arrays["worst_cssu_fill"].reshape(-1) / n_tapes,
                -arrays["ret_quantity"].reshape(-1) / n_tapes,
            ]
        )
    )
    b_ub = np.asarray(
        [
            float(comparator_rows["lost_orders"].mean()),
            -(
                float(comparator_rows["worst_cssu_fill"].mean())
                - worst_fill_margin
            ),
            -float(comparator_rows["ret_quantity"].mean()),
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
    nonzero = np.argwhere(weights > 1e-10)
    return {
        "expected_rows": expected,
        "nonzero_weight_count": int(len(nonzero)),
        "fractional_tape_count": int(
            sum(np.count_nonzero(weights[index] > 1e-10) > 1 for index in range(n_tapes))
        ),
        "solver_objective": float(-solution.fun),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-start", type=int, default=1_060_001)
    parser.add_argument("--calibration-tapes", type=int, default=200)
    parser.add_argument("--locked-start", type=int, default=1_070_001)
    parser.add_argument("--locked-tapes", type=int, default=400)
    parser.add_argument("--worst-fill-margin", type=float, default=0.02)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/program_h/visible_v1_repair/verdict.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    started = time.perf_counter()
    calibration_tapes = [
        tape(index, args.calibration_start) for index in range(args.calibration_tapes)
    ]
    locked_tapes = [
        tape(index, args.locked_start) for index in range(args.locked_tapes)
    ]
    calibration = evaluate_block(calibration_tapes)
    locked = evaluate_block(locked_tapes)
    comparator = solve_comparator(
        calibration, worst_fill_margin=args.worst_fill_margin
    )
    weights = comparator.pop("weights")
    comparator_rows = {
        field: values @ weights for field, values in locked.items()
    }
    pi = solve_guardrailed_pi_relaxation(
        locked,
        comparator_rows,
        worst_fill_margin=args.worst_fill_margin,
    )
    pi_rows = pi.pop("expected_rows")
    unrestricted_index = int(np.argmax(locked["ret_visible"].mean(axis=0)))
    sequence_rows = []
    for index, sequence in enumerate(SEQUENCES):
        sequence_rows.append(
            {
                "index": index,
                "sequence": "".join(sequence),
                "calibration_means": {
                    field: float(values[:, index].mean())
                    for field, values in calibration.items()
                },
                "locked_means": {
                    field: float(values[:, index].mean())
                    for field, values in locked.items()
                },
            }
        )
    matrices = {
        split: {
            field: [[float(value) for value in row] for row in values]
            for field, values in block.items()
        }
        for split, block in (("calibration", calibration), ("locked", locked))
    }
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    result = {
        "schema_version": "program_h_visible_v1_frontier_repair_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head": head,
        "scientific_status": "RETROSPECTIVE_BURNED_TAPE_GOVERNING_METRIC_REPAIR_NOT_CONFIRMATORY",
        "governing_metric": "ret_excel_visible_v1",
        "historical_metric_replaced": "ret_excel_full_ledger_order_adapter",
        "contract": {
            "weeks": WEEKS,
            "actions": list(ACTIONS),
            "sequence_count": len(SEQUENCES),
            "resource_rule": "Every sequence has four weekly route decisions under identical convoy physics.",
            "guardrail_rule": "The convexified comparator is calibrated subject to mean lost orders no worse than ABAB, mean quantity ReT no worse than ABAB, and mean worst-CSSU fill within 0.02. The PI relaxation faces the same aggregate constraints relative to the locked comparator.",
            "worst_fill_margin": args.worst_fill_margin,
        },
        "tapes": {
            "calibration": {
                "seed_start": args.calibration_start,
                "n": args.calibration_tapes,
                "status": "historical burned",
            },
            "locked": {
                "seed_start": args.locked_start,
                "n": args.locked_tapes,
                "status": "historical burned; opened previously by Program H",
            },
            "virgin_opened": False,
        },
        "unconstrained_sparse_metric": {
            "best_fixed_sequence": "".join(SEQUENCES[unrestricted_index]),
            "best_fixed_mean_ret_visible": float(
                locked["ret_visible"][:, unrestricted_index].mean()
            ),
            "best_fixed_mean_lost_orders": float(
                locked["lost_orders"][:, unrestricted_index].mean()
            ),
            "interpretation": "Reported to expose sparse-ledger metric degeneracy only; it is not eligible if it buys ReT by losing orders.",
        },
        "guardrailed_comparator": {
            **comparator,
            "locked_means": {
                field: float(values.mean())
                for field, values in comparator_rows.items()
            },
        },
        "guardrailed_perfect_information_relaxation": {
            **pi,
            "locked_means": {
                field: float(values.mean()) for field, values in pi_rows.items()
            },
            "delta_ret_visible": float(
                pi_rows["ret_visible"].mean()
                - comparator_rows["ret_visible"].mean()
            ),
            "estimate_kind": "finite-sample convexified full-information optimum over all 81 sequences under aggregate guardrails; retrospective point estimate, no confirmatory CI",
        },
        "sequence_rows": sequence_rows,
        "sequence_rows_sha256": json_sha256(sequence_rows),
        "raw_matrices": matrices,
        "raw_matrices_sha256": json_sha256(matrices),
        "claim_limit": "Metric/comparator repair on burned historical tapes only. No H_obs optimum, learner value, virgin confirmation, or Paper-2/Paper-3 authorization is established.",
        "elapsed_seconds": time.perf_counter() - started,
    }
    result["content_sha256"] = json_sha256(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "unconstrained": result["unconstrained_sparse_metric"],
                "comparator": result["guardrailed_comparator"]["locked_means"],
                "pi": result["guardrailed_perfect_information_relaxation"]["locked_means"],
                "delta_ret_visible": result[
                    "guardrailed_perfect_information_relaxation"
                ]["delta_ret_visible"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
