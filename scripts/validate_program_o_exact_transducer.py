#!/usr/bin/env python3
"""Fresh simultaneous validation of the corrected Program O transducer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.screen_program_o_exact_transducer import (  # noqa: E402
    complete_calendars,
    make_tape,
    simulate,
)
from scripts.validate_program_m_shared_lift_hpi import (  # noqa: E402
    bootstrap_simultaneous_lcbs,
    observed_hpi,
)

DEFAULT_CONTRACT = ROOT / "contracts/program_o_exact_transducer_v1.json"
DEFAULT_FREEZE = ROOT / "research/paper2_exhaustive_search/program_o_exact_transducer_validation_freeze_20260714.json"
DEFAULT_OUTPUT_DIR = ROOT / "results/program_o/exact_transducer_validation_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def connected_components(passing: set[str]) -> list[list[str]]:
    adjacency = {
        "rho75_share75": {"rho75_share90", "rho90_share75"},
        "rho75_share90": {"rho75_share75", "rho90_share90"},
        "rho90_share75": {"rho75_share75", "rho90_share90"},
        "rho90_share90": {"rho75_share90", "rho90_share75"},
    }
    components: list[list[str]] = []
    remaining = set(passing)
    while remaining:
        pending = [min(remaining)]
        remaining.remove(pending[0])
        component: list[str] = []
        while pending:
            current = pending.pop()
            component.append(current)
            neighbors = adjacency[current] & remaining
            remaining -= neighbors
            pending.extend(sorted(neighbors))
        components.append(sorted(component))
    return sorted(components)


def produce(contract_path: Path, freeze_path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    contract = json.loads(contract_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    screen_path = ROOT / freeze["screen_result"]["path"]
    if sha256(screen_path) != freeze["screen_result"]["sha256"]:
        raise RuntimeError("screen result changed after validation freeze")
    by_id = {cell["cell_id"]: cell for cell in contract["positive_cells"]}
    calendars = complete_calendars()
    start, end = freeze["validation_seeds"]
    seeds = tuple(range(int(start), int(end) + 1))
    matrices: dict[str, np.ndarray] = {}
    guardrails: dict[str, dict[str, np.ndarray]] = {}
    tape_hashes: dict[str, list[str]] = {}
    for cell_id in freeze["selected_cells"]:
        cell = by_id[cell_id]
        ret_rows = []
        full_rows = []
        unfulfilled_rows = []
        worst_fill_rows = []
        tape_hashes[cell_id] = []
        for seed in seeds:
            tape = make_tape(
                seed,
                persistence=float(cell["regime_persistence"]),
                dominant_share=float(cell["dominant_share"]),
            )
            tape_hashes[cell_id].append(tape.sha256)
            evaluations = [simulate(tape, calendar, contract, complete_substitution=False) for calendar in calendars]
            ret_rows.append([row["ret"] for row in evaluations])
            full_rows.append([row["ret_full"] for row in evaluations])
            unfulfilled_rows.append([row["unfulfilled_quantity"] for row in evaluations])
            worst_fill_rows.append([row["worst_product_fill"] for row in evaluations])
        matrices[cell_id] = np.asarray(ret_rows, dtype=float)
        guardrails[cell_id] = {
            "ret_full": np.asarray(full_rows, dtype=float),
            "unfulfilled_quantity": np.asarray(unfulfilled_rows, dtype=float),
            "worst_product_fill": np.asarray(worst_fill_rows, dtype=float),
        }

    inference = bootstrap_simultaneous_lcbs(
        matrices,
        n_resamples=int(freeze["bootstrap"]["resamples"]),
        rng_seed=int(freeze["bootstrap"]["rng_seed"]),
    )
    cells = []
    passing: set[str] = set()
    for cell_id in freeze["selected_cells"]:
        observed = observed_hpi(matrices[cell_id])
        static_index = int(observed["best_static_calendar_index"])
        oracle_indices = np.asarray(observed["oracle_calendar_indices"], dtype=int)
        tape_indices = np.arange(len(seeds))
        g = guardrails[cell_id]
        row = {
            "cell_id": cell_id,
            "observed_h_pi": observed["observed_h_pi"],
            "simultaneous_lcb95": inference["simultaneous_lcb95"][cell_id],
            "best_static_calendar_index": static_index,
            "best_static_calendar": list(calendars[static_index]),
            "unique_oracle_calendars": observed["unique_oracle_calendars"],
            "oracle_minus_static_unfulfilled_quantity": float(np.mean(g["unfulfilled_quantity"][tape_indices, oracle_indices] - g["unfulfilled_quantity"][:, static_index])),
            "oracle_minus_static_worst_product_fill": float(np.mean(g["worst_product_fill"][tape_indices, oracle_indices] - g["worst_product_fill"][:, static_index])),
            "oracle_minus_static_full_ret": float(np.mean(g["ret_full"][tape_indices, oracle_indices] - g["ret_full"][:, static_index])),
            "tape_hashes": tape_hashes[cell_id],
        }
        rule = freeze["pass_rule"]
        row["passes"] = bool(
            row["simultaneous_lcb95"] >= float(rule["simultaneous_lcb95_minimum"])
            and row["unique_oracle_calendars"] >= int(rule["minimum_unique_oracle_calendars"])
            and row["oracle_minus_static_unfulfilled_quantity"] <= float(rule["oracle_unfulfilled_quantity_mean_delta_maximum"]) + 1e-9
            and row["oracle_minus_static_worst_product_fill"] >= float(rule["oracle_worst_product_fill_mean_delta_minimum"]) - 1e-12
            and row["oracle_minus_static_full_ret"] >= float(rule["oracle_full_ret_mean_delta_minimum"]) - 1e-12
        )
        if row["passes"]:
            passing.add(cell_id)
        cells.append(row)

    components = connected_components(passing)
    eligible_components = [
        component for component in components
        if len(component) >= 3
        and len({name[3:5] for name in component}) >= 2
        and len({name[-2:] for name in component}) >= 2
    ]
    eligible_ids = {name for component in eligible_components for name in component}
    selected = (
        min(
            (row for row in cells if row["cell_id"] in eligible_ids),
            key=lambda row: (row["observed_h_pi"], row["cell_id"]),
        )["cell_id"]
        if eligible_ids else None
    )
    result: dict[str, Any] = {
        "schema_version": "program_o_exact_transducer_validation_v1",
        "status": (
            "PASS_VALIDATED_TRANSDUCER_REGION__FULL_DES_FREEZE_ALLOWED"
            if selected else "STOP_PROGRAM_O_NO_VALIDATED_TRANSDUCER_REGION"
        ),
        "contract_sha256": sha256(contract_path),
        "freeze_sha256": sha256(freeze_path),
        "screen_result_sha256": sha256(screen_path),
        "seeds": list(seeds),
        "calendar_count": len(calendars),
        "des_or_transducer_evaluations": len(cells) * len(seeds) * len(calendars),
        "cells": cells,
        "simultaneous_inference": inference,
        "passing_components": components,
        "eligible_connected_components": eligible_components,
        "selected_full_des_development_cell": selected,
        "claim_boundary": {
            "exact_transducer_h_pi_validated": bool(selected),
            "full_des_implementation_freeze_allowed": bool(selected),
            "full_des_h_pi_established": False,
            "h_obs_authorized": False,
            "learner_authorized": False,
            "paper3_authorized": False,
        },
    }
    result["content_sha256"] = json_sha256(result)
    return result, matrices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    result, matrices = produce(args.contract.resolve(), args.freeze.resolve())
    output_dir.mkdir(parents=True)
    np.savez_compressed(output_dir / "ret_matrices.npz", **matrices)
    result["ret_matrices_sha256"] = sha256(output_dir / "ret_matrices.npz")
    result.pop("content_sha256")
    result["content_sha256"] = json_sha256(result)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(output_dir / "result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
