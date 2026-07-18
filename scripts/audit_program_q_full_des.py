#!/usr/bin/env python3
"""Direct SimPy replay audit for every Program Q promoted trajectory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import CONTRACT, scheduler  # noqa: E402
from supply_chain.program_o_eval_custody import (  # noqa: E402
    sha256,
    verify_sha256_manifest,
    write_sha256_manifest,
)
from supply_chain.program_o_full_des import run_program_o_full_des_episode  # noqa: E402
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    MATRIX_KEYS,
    direct_full_des_vector,
)
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS  # noqa: E402
from supply_chain.program_o_state_rich import finite_state_rich_configurations  # noqa: E402


def calendar_index(calendar: tuple[int, ...]) -> int:
    value = 0
    for action in calendar:
        value = 4 * value + int(action)
    return value


def compare_promoted_sources(
    payload: Mapping[str, np.ndarray],
    direct: Mapping[str, float],
    *,
    calendar: tuple[int, ...],
    sources: list[tuple[str, int, str]],
    atol: float,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    maximum = {key: 0.0 for key in MATRIX_KEYS}
    failures: list[dict[str, object]] = []
    for prefix, index, label in sources:
        for key in MATRIX_KEYS:
            expected = float(payload[f"{prefix}__{key}"][index])
            error = abs(float(direct[key]) - expected)
            maximum[key] = max(maximum[key], error)
            if error > atol:
                failures.append(
                    {
                        "calendar": list(calendar),
                        "source": label,
                        "metric": key,
                        "direct": float(direct[key]),
                        "stored": expected,
                        "absolute_error": error,
                    }
                )
    return maximum, failures


def audit(*, evaluation: Path, shards: Path, atol: float) -> dict:
    contract = json.loads(CONTRACT.read_text())
    result_path = evaluation / "result.json"
    evaluation_manifest = verify_sha256_manifest(
        evaluation, evaluation / "evaluation_files.sha256"
    )
    shard_manifest = verify_sha256_manifest(shards, evaluation / "shard_files.sha256")
    if "result.json" not in evaluation_manifest:
        raise RuntimeError("Program Q result is absent from evaluation manifest")
    if "shard_files.sha256" not in evaluation_manifest:
        raise RuntimeError("Program Q shard manifest is absent from evaluation manifest")
    if set(evaluation_manifest) != {"result.json", "shard_files.sha256"}:
        raise RuntimeError("Program Q evaluation manifest contains an unexpected file set")
    result = json.loads(result_path.read_text())
    if result.get("schema_version") != "program_q_frozen_policy_replication_evaluation_v1":
        raise RuntimeError("Program Q evaluator schema mismatch")
    if result.get("contract_sha256") != sha256(CONTRACT):
        raise RuntimeError("Program Q result contract hash mismatch")
    reserved_start, reserved_end = map(int, contract["confirmation"]["reserved_block"])
    expected_n = int(contract["confirmation"]["N"])
    seeds = list(range(reserved_start, reserved_start + expected_n))
    if seeds[-1] > reserved_end:
        raise RuntimeError("Program Q frozen N exceeds reserved block")
    if result.get("N") != expected_n or result.get("seed_range") != [seeds[0], seeds[-1]]:
        raise RuntimeError("Program Q result does not match the frozen seed design")
    expected_cells = {cell.cell_id for cell in CONFIRMED_RET_CELLS}
    if set(result.get("cell_summaries", {})) != expected_cells:
        raise RuntimeError("Program Q result does not contain exactly the frozen cells")
    expected_shards = {
        f"{cell.cell_id}/tape_{seed}.npz"
        for cell in CONFIRMED_RET_CELLS
        for seed in seeds
    }
    if set(shard_manifest) != expected_shards or result.get("shard_count") != len(expected_shards):
        raise RuntimeError("Program Q shard manifest does not match the frozen 768-shard design")
    maximum_error = {key: 0.0 for key in MATRIX_KEYS}
    failures: list[dict[str, object]] = []
    replay_count = 0
    stored_comparison_count = 0
    config_ids = [config.config_id for config in finite_state_rich_configurations()]
    for cell in CONFIRMED_RET_CELLS:
        summary = result["cell_summaries"][cell.cell_id]
        audit_rows = result["trajectory_audits"][cell.cell_id]
        replacement_rows = result["replacement_controls"][cell.cell_id]
        try:
            classical_index = config_ids.index(summary["best_classical_config"])
        except ValueError as error:
            raise RuntimeError("Program Q selected classical config is not frozen") from error
        for tape_index, tape_seed in enumerate(seeds):
            sources_by_calendar: dict[tuple[int, ...], list[tuple[str, int, str]]] = {}
            open_calendar = tuple(map(int, summary["best_open_loop_calendar"]))
            sources_by_calendar.setdefault(open_calendar, []).append(
                ("open_loop", calendar_index(open_calendar), "best_open_loop")
            )
            classical_calendar = tuple(
                map(int, summary["best_classical_calendars"][tape_index])
            )
            sources_by_calendar.setdefault(classical_calendar, []).extend(
                [
                    ("open_loop", calendar_index(classical_calendar), "classical_open_row"),
                    ("classical", classical_index, "selected_classical"),
                ]
            )
            shard = shards / cell.cell_id / f"tape_{tape_seed}.npz"
            relative = shard.relative_to(shards).as_posix()
            if relative not in shard_manifest:
                raise RuntimeError(f"Program Q shard missing from manifest: {relative}")
            with np.load(shard, allow_pickle=False) as payload:
                stored_learner_seeds = list(map(int, payload["learner_seeds"]))
                for learner_seed, seed_rows in audit_rows.items():
                    try:
                        learner_index = stored_learner_seeds.index(int(learner_seed))
                    except ValueError as error:
                        raise RuntimeError("Program Q learner seed missing from shard") from error
                    calendar = tuple(map(int, seed_rows["calendars"][tape_index]))
                    sources_by_calendar.setdefault(calendar, []).extend(
                        [
                            ("open_loop", calendar_index(calendar), "learner_open_row"),
                            ("learner", learner_index, f"learner_seed_{learner_seed}"),
                        ]
                    )
                for family, family_rows in replacement_rows.items():
                    for learner_seed, replacement in family_rows["per_seed"].items():
                        calendar = tuple(map(int, replacement["calendar"]))
                        sources_by_calendar.setdefault(calendar, []).append(
                            (
                                "open_loop",
                                calendar_index(calendar),
                                f"replacement_{family}_seed_{learner_seed}",
                            )
                        )
                for calendar, sources in sorted(sources_by_calendar.items()):
                    replay_count += 1
                    simulation, panel = run_program_o_full_des_episode(
                        seed=tape_seed,
                        calendar=calendar,
                        scheduler=scheduler(),
                        regime_persistence=cell.regime_persistence,
                        dominant_share=cell.dominant_share,
                        downstream_freight_physics_mode="fixed_clock_physical_v1",
                    )
                    direct = direct_full_des_vector(simulation, panel)
                    maxima, source_failures = compare_promoted_sources(
                        payload,
                        direct,
                        calendar=calendar,
                        sources=sources,
                        atol=atol,
                    )
                    stored_comparison_count += len(sources)
                    for key, error in maxima.items():
                        maximum_error[key] = max(maximum_error[key], error)
                    failures.extend(
                        {"cell": cell.cell_id, "tape_seed": tape_seed, **failure}
                        for failure in source_failures
                    )
    return {
        "schema_version": "program_q_direct_full_des_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_result_sha256": sha256(result_path),
        "contract_sha256": sha256(CONTRACT),
        "seed_range": [seeds[0], seeds[-1]],
        "N": expected_n,
        "shard_count": len(expected_shards),
        "direct_unique_replays": replay_count,
        "stored_source_comparisons": stored_comparison_count,
        "atol": atol,
        "max_absolute_error_by_metric": maximum_error,
        "failure_count": len(failures),
        "failures": failures[:100],
        "passed": not failures,
        "terminal_status": (
            "DIRECT_FULL_DES_AUDIT_PASS_ELIGIBLE_FOR_ADJUDICATION"
            if not failures
            else "STOP_Q_DIRECT_FULL_DES_PARITY_FAILURE"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = audit(evaluation=args.evaluation, shards=args.shards, atol=args.atol)
    args.output.mkdir(parents=True)
    result = args.output / "independent_full_des_audit.json"
    result.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_sha256_manifest(args.output, [result], args.output / "audit_files.sha256")
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
