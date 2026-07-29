#!/usr/bin/env python3
"""Program I Sobol refinement with held-out surrogate diagnostics.

The default is design-only. ``--execute`` is required to run the DES, and the
script remains incapable of authorizing RL or opening confirmatory universes.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from supply_chain.decision_right_discovery import evaluate_design
from run_program_i_sensitivity import FACTORS, run_des


def problem() -> dict:
    return {
        "num_vars": len(FACTORS),
        "names": [factor.name for factor in FACTORS],
        "bounds": [[factor.lower, factor.upper] for factor in FACTORS],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-samples", type=int, default=128)
    parser.add_argument("--tapes", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=1_090_501)
    parser.add_argument("--horizon-weeks", type=int, default=52)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/program_i/refinement"))
    args = parser.parse_args()
    if args.seed_start >= 1_110_001:
        raise SystemExit("Refusing to open Program I confirmation/virgin universes")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    design = sobol_sample.sample(problem(), args.base_samples, calc_second_order=True, seed=20260713)
    with (args.output_dir / "design.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(problem()["names"])
        writer.writerows(design)
    metadata = {
        "contract_id": "global_sensitivity_v1", "phase": "sobol_refinement",
        "design_rows": len(design), "base_samples": args.base_samples,
        "execute": args.execute, "confirmation_opened": False,
        "virgin_rl_opened": False, "promote_to_rl": False,
    }
    if not args.execute:
        (args.output_dir / "design_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata, indent=2))
        return 0
    outputs = evaluate_design(
        design, FACTORS, range(args.seed_start, args.seed_start + args.tapes),
        lambda params, tape: run_des(dict(params), tape, args.horizon_weeks),
    )
    indices = {}
    for metric, values in outputs.items():
        analysis = sobol_analyze.analyze(problem(), values, calc_second_order=True, print_to_console=False)
        indices[metric] = {
            "S1": dict(zip(problem()["names"], map(float, analysis["S1"]))),
            "ST": dict(zip(problem()["names"], map(float, analysis["ST"]))),
        }
    # A held-out emulator is diagnostic only. Grouping contiguous Saltelli rows
    # prevents near-identical design structure from leaking across the split.
    groups = np.arange(len(design)) // (2 * len(FACTORS) + 2)
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=20260713).split(design, groups=groups))
    surrogate = {}
    for metric, values in outputs.items():
        model = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=20260713, n_jobs=-1)
        model.fit(design[train], values[train])
        pred = model.predict(design[test])
        surrogate[metric] = {"r2_holdout": float(r2_score(values[test], pred)), "mae_holdout": float(mean_absolute_error(values[test], pred))}
    result = {
        **metadata, "indices": indices, "surrogate_validation": surrogate,
        "interpretation": "REFINEMENT_ONLY_REQUIRES_COUNTERFACTUAL_BRANCHING",
    }
    (args.output_dir / "verdict.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.output_dir / "outputs.npz", **outputs)
    print(json.dumps({"interpretation": result["interpretation"], "surrogate": surrogate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
