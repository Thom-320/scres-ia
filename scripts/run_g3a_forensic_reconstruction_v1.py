#!/usr/bin/env python3
"""Replay the manuscript-specified G3a factorial and retain all 18,360 rows.

This is a forensic reconstruction, not the lost producer.  Its contract and the
incident certificate explicitly prohibit treating agreement with the surviving
rounded aggregates as validation or confirmation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain import falsifiers as F  # noqa: E402
from supply_chain.g3a_forensic import all_cells, make_tape, policies, simulate  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / "contracts/g3a_forensic_reconstruction_v1.json"
OUT_DIR = ROOT / "results/g3a_forensic_reconstruction_v1"
SELECTION = tuple(range(8701001, 8701031))
EVALUATION = tuple(range(8701031, 8701061))
BAR = 0.01
MODULES = (
    "supply_chain/g3a_forensic.py", "supply_chain/falsifiers.py",
    "supply_chain/arm_runner.py", "supply_chain/seed_custody.py",
)


def interval(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if values.size <= 1 or float(values.std(ddof=1)) == 0.0:
        return {"mean": mean, "lcb95": mean, "ucb95": mean, "n": int(values.size)}
    half = float(stats.t.ppf(0.975, values.size - 1) * values.std(ddof=1) / np.sqrt(values.size))
    return {"mean": mean, "lcb95": mean - half, "ucb95": mean + half, "n": int(values.size)}


def run_rows() -> list[dict]:
    rows: list[dict] = []
    library = policies()
    for cell_id, process, contract in all_cells():
        for seed in (*SELECTION, *EVALUATION):
            tape = make_tape(seed, process)
            split = "selection" if seed in SELECTION else "evaluation"
            for policy in library:
                outcome = simulate(tape, contract, policy)
                rows.append({
                    "cell_id": cell_id, "process": process, "capacity_contract": contract,
                    "split": split, "seed": seed, "policy": policy.name,
                    "policy_family": policy.family, "deployable": int(policy.deployable),
                    **outcome,
                })
    return rows


def write_raw(rows: list[dict], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    # mtime=0 makes the compressed bytes reproducible as well as the CSV payload.
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="raw_rows.csv", mode="wb", fileobj=raw, mtime=0) as gz:
            import io
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    return sha256(path.read_bytes()).hexdigest()


def analyse(rows: list[dict]) -> dict:
    by_key = {(row["cell_id"], int(row["seed"]), row["policy"]): row for row in rows}
    library = policies()
    constants = [p.name for p in library if p.family == "constant"]
    structured = [p.name for p in library if p.deployable and p.family not in {"constant", "placebo"}]
    cells = {}
    for cell_id, process, contract in all_cells():
        def mean_on(names, seeds):
            return {name: float(np.mean([by_key[cell_id, seed, name]["primary_service"]
                                         for seed in seeds])) for name in names}
        fixed = max(mean_on(constants, SELECTION), key=mean_on(constants, SELECTION).get)
        struct = max(mean_on(structured, SELECTION), key=mean_on(structured, SELECTION).get)
        privileged = "privileged_true_state"
        fixed_values = np.array([by_key[cell_id, seed, fixed]["primary_service"] for seed in EVALUATION])
        struct_values = np.array([by_key[cell_id, seed, struct]["primary_service"] for seed in EVALUATION])
        privileged_values = np.array([by_key[cell_id, seed, privileged]["primary_service"]
                                      for seed in EVALUATION])
        stateful = np.array([by_key[cell_id, seed, "belief_stateful"]["primary_service"]
                             for seed in EVALUATION])
        reset = np.array([by_key[cell_id, seed, "belief_reset"]["primary_service"]
                          for seed in EVALUATION])
        cells[cell_id] = {
            "process": process, "capacity_contract": contract,
            "selected_fixed_on_selection": fixed,
            "selected_structured_on_selection": struct,
            "fixed_evaluation": interval(fixed_values),
            "structured_evaluation": interval(struct_values),
            "H_obs_structured_minus_fixed": interval(struct_values - fixed_values),
            "H_priv_privileged_minus_fixed": interval(privileged_values - fixed_values),
            "H_res_privileged_minus_structured": interval(privileged_values - struct_values),
            "H_ret_stateful_minus_reset": interval(stateful - reset),
            "structured_mean_forfeiture": float(np.mean([
                by_key[cell_id, seed, struct]["forfeited_capacity"] for seed in EVALUATION])),
            "max_abs_flow_residual": float(max(abs(by_key[cell_id, seed, p.name]["flow_residual"])
                                                for seed in EVALUATION for p in library)),
        }
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    rows = run_rows()
    raw_path = args.output_dir / "raw_rows.csv.gz"
    raw_sha = write_raw(rows, raw_path)
    cells = analyse(rows)

    expected = 9 * 60 * 34
    fifo_groups = {}
    for row in rows:
        if row["capacity_contract"] == "global_fifo":
            key = (row["cell_id"], row["seed"])
            fifo_groups.setdefault(key, []).append(float(row["primary_service"]))
    max_fifo_spread = max(max(values) - min(values) for values in fifo_groups.values())
    max_flow = max(abs(float(row["flow_residual"])) for row in rows)
    hard_forfeit = np.mean([float(row["forfeited_capacity"]) for row in rows
                            if row["capacity_contract"] == "hard_quota"])
    spare_forfeit = max(float(row["forfeited_capacity"]) for row in rows
                        if row["capacity_contract"] == "spare_reallocation")
    fifo_forfeit = max(float(row["forfeited_capacity"]) for row in rows
                       if row["capacity_contract"] == "global_fifo")
    tape_hash_counts = {}
    for row in rows:
        key = (row["cell_id"], row["seed"])
        tape_hash_counts.setdefault(key, set()).add(row["tape_sha256"])
    max_tapes_per_cell_seed = max(map(len, tape_hash_counts.values()))
    deployable_families = {p.family for p in policies() if p.deployable}

    checks = {
        "f1_expected_18360_rows": F.check(len(rows) == expected,
            "a missing policy/tape/cell silently changes the selected frontier",
            computed_from={"observed": len(rows), "expected": expected}),
        "f2_exactly_34_unique_policies": F.check(len(policies()) == 34,
            "a reconstructed library of another width is not the manuscript factorial",
            computed_from={"observed": len(policies()), "expected": 34}),
        "f3_one_exogenous_tape_per_cell_seed": F.check(max_tapes_per_cell_seed == 1,
            "policy-dependent demand or risk would invalidate common-random-number comparisons",
            computed_from={"max_distinct_hashes": max_tapes_per_cell_seed, "expected": 1}),
        "f4_flow_ledger_closes": F.lt(max_flow, 1e-8,
            "unaccounted flow can manufacture service or resource advantages"),
        "f5_global_fifo_action_invariant": F.lt(max_fifo_spread, 1e-12,
            "global pooling is the action-invariant null and must not respond to allocation"),
        "f6_hard_quota_forfeits_capacity": F.gt(hard_forfeit, 0.0,
            "without forfeiture the hard-quota mechanism is mislabeled"),
        "f7_spare_reallocation_work_conserving": F.lt(spare_forfeit, 1e-8,
            "unused capacity in the reallocation arm would recreate the hard-quota artefact"),
        "f8_global_fifo_work_conserving": F.lt(fifo_forfeit, 1e-8,
            "unused pooled capacity while backlog exists would make the FIFO null invalid"),
        "f9_both_pressure_directions_realized": F.check(
            any(row["process"].startswith("persistent") and row["demand_a"] > row["demand_b"]
                for row in rows) and
            any(row["process"].startswith("persistent") and row["demand_a"] < row["demand_b"]
                for row in rows),
            "a one-sided pressure process cannot test claimant-specific adaptation",
            computed_from={"n_rows": len(rows), "n_processes": 3}),
        "f10_action_library_moves_allocation": F.gt(
            max(float(r["mean_allocation_a"]) for r in rows if r["capacity_contract"] == "hard_quota")
            - min(float(r["mean_allocation_a"]) for r in rows if r["capacity_contract"] == "hard_quota"),
            0.5, "a nominal policy library whose actions collapse is not a control benchmark"),
        "f11_deployable_interface_excludes_privileged": F.check(
            "privileged" not in deployable_families,
            "a true-state policy marked deployable leaks the latent regime into H_obs",
            computed_from={"n_deployable_families": len(deployable_families), "n_policies": len(policies())}),
        "f12_selection_and_evaluation_are_disjoint": F.check(
            not (set(SELECTION) & set(EVALUATION)),
            "selecting a controller on its evaluation tapes biases every reported difference",
            computed_from={"n_selection": len(SELECTION), "n_evaluation": len(EVALUATION)}),
    }
    summary = F.summarise(checks)
    try:
        raw_recorded_path = str(raw_path.relative_to(ROOT))
    except ValueError:
        raw_recorded_path = str(raw_path)
    payload = {
        "schema_version": "g3a_forensic_reconstruction_result_v1",
        "claim_status": "FORENSIC_RECONSTRUCTION_COMPLETE_NOT_ORIGINAL_NOT_CONFIRMATORY",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_role": "FORENSIC_DEVELOPMENT_REPLAY",
        "nonclaim": "Agreement or disagreement with rounded manuscript aggregates cannot validate this reconstruction.",
        "seeds": list((*SELECTION, *EVALUATION)),
        "selection_seeds": list(SELECTION), "evaluation_seeds": list(EVALUATION),
        "n_rows": len(rows), "n_cells": 9, "n_policies": 34,
        "raw_rows": {"path": raw_recorded_path, "sha256": raw_sha,
                     "compression": "gzip mtime=0"},
        "cells": cells, "falsifiers": checks, "falsifier_summary": summary,
        "module_manifest": module_manifest(MODULES, script=Path(__file__)),
        "surviving_aggregates_are_not_acceptance_targets": {
            "persistent_uniform_hard_quota_H_obs": [0.0963, 0.0682, 0.1245],
            "persistent_uniform_spare_reallocation_H_obs": [0.00126, -0.00012, 0.00264],
            "persistent_uniform_global_FIFO_H_obs": [0.0, 0.0, 0.0]
        }
    }
    seal_and_write(
        payload,
        args.output_dir / "result.json",
        contract=CONTRACT,
        reference=ROOT / "contracts/g3a_asymmetric_claimants_v2.json",
    )
    print(json.dumps({"claim_status": payload["claim_status"], "n_rows": len(rows),
                      "raw_sha256": raw_sha, "falsifiers": summary,
                      "persistent_uniform": {
                          key: value["H_obs_structured_minus_fixed"] for key, value in cells.items()
                          if key.startswith("persistent_uniform")}}, indent=2))
    return 0 if summary["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
