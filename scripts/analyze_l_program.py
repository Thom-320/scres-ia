#!/usr/bin/env python3
"""Analyze Program L probe rows without treating weeks as independent units."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paired_matrix(
    rows: list[dict[str, str]],
    *,
    treatment: str,
    control: str,
    metric: str,
    cycle: int,
    buffer_level: int,
) -> tuple[np.ndarray, list[int], list[str]]:
    selected = [
        row
        for row in rows
        if int(row["cycle"]) == cycle
        and int(row["buffer_level"]) == buffer_level
        and row["arm"] in {treatment, control}
    ]
    seeds = sorted({int(row["learner_seed"]) for row in selected})
    probes = sorted({row["probe_id"] for row in selected})
    lookup = {
        (row["arm"], int(row["learner_seed"]), row["probe_id"]): float(row[metric])
        for row in selected
    }
    matrix = np.empty((len(seeds), len(probes)), dtype=np.float64)
    for i, seed in enumerate(seeds):
        for j, probe in enumerate(probes):
            matrix[i, j] = lookup[(treatment, seed, probe)] - lookup[
                (control, seed, probe)
            ]
    return matrix, seeds, probes


def two_way_bootstrap(
    matrix: np.ndarray, *, seed: int = 0, n_boot: int = 10_000
) -> dict[str, Any]:
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("Expected a non-empty seed x tape matrix.")
    rng = np.random.default_rng(seed)
    n_seed, n_tape = matrix.shape
    boots = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        seed_idx = rng.integers(0, n_seed, n_seed)
        tape_idx = rng.integers(0, n_tape, n_tape)
        boots[index] = matrix[np.ix_(seed_idx, tape_idx)].mean()
    per_seed = matrix.mean(axis=1)
    per_tape = matrix.mean(axis=0)
    return {
        "mean": float(matrix.mean()),
        "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "per_seed": [float(value) for value in per_seed],
        "seeds_positive": int((per_seed > 0.0).sum()),
        "n_seeds": int(n_seed),
        "tapes_positive": int((per_tape > 0.0).sum()),
        "n_tapes": int(n_tape),
    }


def final_cycle(rows: list[dict[str, str]]) -> int:
    return max(int(row["cycle"]) for row in rows)


def analyze_buffer(
    rows: list[dict[str, str]], *, buffer_level: int, cycle: int
) -> dict[str, Any]:
    ret_matrix, seeds, probes = paired_matrix(
        rows,
        treatment="persistent_weights",
        control="reset_local",
        metric="ret_excel",
        cycle=cycle,
        buffer_level=buffer_level,
    )
    # Positive means treatment reduces the loss.
    service_matrix, _, _ = paired_matrix(
        rows,
        treatment="reset_local",
        control="persistent_weights",
        metric="service_loss_auc_ration_hours",
        cycle=cycle,
        buffer_level=buffer_level,
    )
    shift_matrix, _, _ = paired_matrix(
        rows,
        treatment="persistent_weights",
        control="reset_local",
        metric="shift_hours",
        cycle=cycle,
        buffer_level=buffer_level,
    )
    ret = two_way_bootstrap(ret_matrix)
    service = two_way_bootstrap(service_matrix)
    shift = two_way_bootstrap(shift_matrix)

    reset_rows = [
        row
        for row in rows
        if row["arm"] == "reset_local"
        and int(row["cycle"]) == cycle
        and int(row["buffer_level"]) == buffer_level
    ]
    service_baseline = float(
        np.mean([float(row["service_loss_auc_ration_hours"]) for row in reset_rows])
    )
    shift_baseline = float(np.mean([float(row["shift_hours"]) for row in reset_rows]))
    service_relative = service["mean"] / service_baseline if service_baseline > 0 else 0.0
    # Conservatively scale the absolute bootstrap interval by the fixed control mean.
    service_relative_ci = [
        value / service_baseline if service_baseline > 0 else 0.0
        for value in service["ci95"]
    ]
    shift_relative = shift["mean"] / shift_baseline if shift_baseline > 0 else 0.0
    passed = bool(
        ret["ci95"][0] > 0.0
        and service_relative >= 0.05
        and service_relative_ci[0] >= 0.05
        and abs(shift_relative) <= 0.02
    )
    return {
        "buffer_level": buffer_level,
        "cycle": cycle,
        "learner_seeds": seeds,
        "probe_ids": probes,
        "ret_excel_persistent_minus_reset": ret,
        "service_loss_reduction_persistent_vs_reset": {
            **service,
            "relative_mean": service_relative,
            "relative_ci95": service_relative_ci,
        },
        "shift_hours_persistent_minus_reset": {
            **shift,
            "relative_mean": shift_relative,
        },
        "retained_learning_gate_passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    rows = read_rows(args.run_dir / "probe_rows.csv")
    cycle = final_cycle(rows)
    buffers = sorted({int(row["buffer_level"]) for row in rows})
    results = [analyze_buffer(rows, buffer_level=value, cycle=cycle) for value in buffers]
    confirmatory = bool(manifest.get("virgin_tapes_opened", False))
    payload = {
        "kind": "l_program_retained_learning_analysis",
        "evidence_level": "confirmatory" if confirmatory else "fixed_probe_pilot",
        "cycle": cycle,
        "buffers": results,
        "any_buffer_passed": any(row["retained_learning_gate_passed"] for row in results),
        "claim_promoted": bool(
            confirmatory and any(row["retained_learning_gate_passed"] for row in results)
        ),
        "claim_rule": "confirmatory virgin tapes plus simultaneous ReT, 5% service, and resource gates",
    }
    output = args.output or (args.run_dir / "analysis.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

