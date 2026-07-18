#!/usr/bin/env python3
"""Non-promotable end-to-end smoke of the Program Q confirmation harness."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.adjudicate_program_q import adjudicate  # noqa: E402
from scripts.audit_program_q_full_des import (  # noqa: E402
    calendar_index,
    compare_promoted_sources,
)
from scripts.evaluate_program_q_replication import (  # noqa: E402
    CONTRACT,
    scheduler,
    produce_shard,
    simultaneous_primary_inference,
    validate_shard,
)
from supply_chain.program_o_eval_custody import sha256, write_sha256_manifest  # noqa: E402
from supply_chain.program_o_full_des import run_program_o_full_des_episode  # noqa: E402
from supply_chain.program_o_full_des_transducer import direct_full_des_vector  # noqa: E402
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS  # noqa: E402


SMOKE_SEEDS = (949_200_003, 949_200_004)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    shards = args.output / "shards"
    paths: list[Path] = []
    panels: dict[str, dict[str, np.ndarray]] = {}
    direct_max_error = 0.0
    for cell_index, cell in enumerate(CONFIRMED_RET_CELLS):
        for seed in SMOKE_SEEDS:
            path = produce_shard(
                cell_index=cell_index,
                tape_seed=seed,
                models_dir=args.models,
                output=shards,
                allow_development=True,
            )
            validate_shard(path, cell_index=cell_index, tape_seed=seed)
            paths.append(path)
        open_ret = np.empty((len(SMOKE_SEEDS), 65_536), dtype=float)
        learner_ret = np.empty((10, len(SMOKE_SEEDS)), dtype=float)
        classical_ret = np.empty((10, len(SMOKE_SEEDS)), dtype=float)
        for tape_index, seed in enumerate(SMOKE_SEEDS):
            path = shards / cell.cell_id / f"tape_{seed}.npz"
            with np.load(path, allow_pickle=False) as payload:
                open_ret[tape_index] = payload["open_loop__ret_visible"]
                learner_ret[:, tape_index] = payload["learner__ret_visible"]
                classical_ret[:, tape_index] = payload["classical__ret_visible"]
                sources = []
                calendars = []
                for prefix, index, calendar_key in (
                    ("learner", 0, "learner_calendars"),
                    ("classical", 0, "classical_calendars"),
                ):
                    calendar = tuple(map(int, payload[calendar_key][index]))
                    calendars.append(calendar)
                    sources.append(
                        (
                            calendar,
                            [
                                ("open_loop", calendar_index(calendar), f"{prefix}_open_row"),
                                (prefix, index, f"{prefix}_promoted_row"),
                            ],
                        )
                    )
                for calendar, promoted_sources in sources:
                    simulation, panel = run_program_o_full_des_episode(
                        seed=seed,
                        calendar=calendar,
                        scheduler=scheduler(),
                        regime_persistence=cell.regime_persistence,
                        dominant_share=cell.dominant_share,
                        downstream_freight_physics_mode="fixed_clock_physical_v1",
                    )
                    maxima, failures = compare_promoted_sources(
                        payload,
                        direct_full_des_vector(simulation, panel),
                        calendar=calendar,
                        sources=promoted_sources,
                        atol=1e-8,
                    )
                    if failures:
                        raise RuntimeError(f"development smoke direct replay failed: {failures[0]}")
                    direct_max_error = max(direct_max_error, *maxima.values())
        panels[cell.cell_id] = {
            "learner": learner_ret,
            "open_loop": open_ret,
            "classical": classical_ret,
        }
    inference = simultaneous_primary_inference(panels, resamples=200, rng_seed=949_200_005)
    if not all(
        np.isfinite(value)
        for estimate in inference["estimates"].values()
        for value in estimate.values()
    ):
        raise RuntimeError("development smoke produced non-finite inference")
    smoke_result = {
        "schema_version": "program_q_frozen_policy_replication_evaluation_v1",
        "N": len(SMOKE_SEEDS),
        "seed_range": [SMOKE_SEEDS[0], SMOKE_SEEDS[-1]],
        "shard_count": len(paths),
        "bootstrap_resamples": 200,
        "inference": inference,
        "cell_summaries": {
            cell.cell_id: {
                "favorable_tapes_fraction_vs_open_loop": 1.0,
                "positive_learner_seeds_H_OL": 10,
            }
            for cell in CONFIRMED_RET_CELLS
        },
        "integrity_gates": {
            "feedback": True,
            "replacement_controls": True,
            "scheduled_resources_exact": True,
            "mass_partition_demand": True,
            "ret_full_noninferior": True,
            "quantity_ret_full_noninferior": True,
            "worst_product_fill_noninferior": True,
        },
    }
    terminal = adjudicate(
        smoke_result,
        json.loads(CONTRACT.read_text()),
        {"passed": True},
    )
    if terminal["verdict"] != "STOP_Q_NO_REPLICATED_LEARNED_ADAPTATION":
        raise RuntimeError("reduced development smoke was not rejected by frozen adjudication")
    report = {
        "schema_version": "program_q_confirmation_development_smoke_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_NONPROMOTABLE_END_TO_END_SMOKE",
        "scientific_status": "SANDBOX_DEVELOPMENT_ONLY_NOT_PROMOTABLE",
        "development_seeds": list(SMOKE_SEEDS),
        "scientific_749_seeds_opened": False,
        "cells": [cell.cell_id for cell in CONFIRMED_RET_CELLS],
        "shard_count": len(paths),
        "open_loop_family_size": 65_536,
        "classical_family_size": 10,
        "learner_population_size": 10,
        "inference_resamples": 200,
        "inference_all_finite": True,
        "direct_full_des_max_error": direct_max_error,
        "reduced_design_adjudication": terminal["verdict"],
        "fail_closed_design_gates": terminal["design_gates"],
        "contract_id": json.loads(CONTRACT.read_text())["contract_id"],
        "frozen_confirmation_N": json.loads(CONTRACT.read_text())["confirmation"]["N"],
        "smoke_script_sha256": sha256(Path(__file__)),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_sha256_manifest(args.output, [*paths, report_path], args.output / "smoke_files.sha256")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
