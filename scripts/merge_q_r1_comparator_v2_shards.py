#!/usr/bin/env python3
"""Merge burned comparator-v2 shards from raw convergence or Pareto rows."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def _signature(row: dict[str, object]) -> tuple[int, str, float, str]:
    value = row["signature"]
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("invalid signature")
    return int(value[0]), str(value[1]), float(value[2]), str(value[3])


def _validate_common(shards: list[dict[str, object]]) -> None:
    if not shards:
        raise ValueError("at least one shard is required")
    expected = shards[0]["conditional_path_budgets"]
    for shard in shards:
        if shard.get("claim_status") != "BURNED_DEVELOPMENT_NO_CLAIM":
            raise ValueError("only burned development shards may be merged")
        if shard.get("conditional_path_budgets") != expected:
            raise ValueError("conditional path budgets differ across shards")
        if shard.get("selection_performed") is not False:
            raise ValueError("a shard performed selection")
        if shard.get("learner_return_used") is not False:
            raise ValueError("a shard used learner return")
        if shard.get("retained_minus_reset_used_for_selection") is not False:
            raise ValueError("a shard used the treatment effect for selection")


def merge_convergence(shards: list[dict[str, object]]) -> dict[str, object]:
    raw: list[dict[str, object]] = []
    summaries: dict[tuple[int, str, float, str], list[dict[str, object]]] = defaultdict(list)
    for shard in shards:
        raw.extend(shard.get("convergence_pairs", []))
        for row in shard["convergence"]:
            summaries[_signature(row)].append(row)

    identities = [
        (
            tuple(row["signature"]),
            int(row["history_root"]),
            int(row["campaign_index"]),
            str(row["persistence_mode"]),
            str(row["prior_arm"]),
        )
        for row in raw
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate convergence raw-row identity")

    merged = []
    for signature, rows in sorted(summaries.items()):
        selected = [row for row in raw if _signature(row) == signature]
        errors = np.asarray(
            [float(row["absolute_planning_value_error"]) for row in selected],
            dtype=float,
        )
        agreement = np.asarray(
            [int(row["low_action"]) == int(row["high_action"]) for row in selected],
            dtype=bool,
        )
        low_abstentions = sum(int(row["low_abstentions"]) for row in rows)
        high_abstentions = sum(int(row["high_abstentions"]) for row in rows)
        mean_error = float(errors.mean()) if len(errors) else float("inf")
        q95_error = float(np.quantile(errors, 0.95)) if len(errors) else float("inf")
        action_agreement = float(agreement.mean()) if len(agreement) else 0.0
        passed = (
            low_abstentions == 0
            and high_abstentions == 0
            and action_agreement >= 0.95
            and mean_error < 0.005
            and q95_error < 0.01
        )
        merged.append(
            {
                "signature": list(signature),
                "low_config": rows[0]["low_config"],
                "high_config": rows[0]["high_config"],
                "first_action_agreement": action_agreement,
                "mean_abs_planning_value_error": mean_error,
                "q95_abs_planning_value_error": q95_error,
                "low_abstentions": low_abstentions,
                "high_abstentions": high_abstentions,
                "comparable_arm_states": len(errors),
                "convergence_pass": passed,
            }
        )
    return {"convergence": merged, "convergence_pairs": raw}


def merge_pareto(shards: list[dict[str, object]]) -> dict[str, object]:
    raw = [row for shard in shards for row in shard.get("pareto_pairs", [])]
    identities = [
        (
            str(row["config_id"]),
            int(row["history_root"]),
            int(row["campaign_index"]),
            str(row["persistence_mode"]),
        )
        for row in raw
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate Pareto raw-row identity")
    by_config: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in raw:
        by_config[str(row["config_id"])].append(row)
    merged = []
    for config_id, rows in sorted(by_config.items()):
        def delta(key: str) -> float:
            return float(
                np.mean(
                    [
                        float(row["retained"][key]) - float(row["reset"][key])
                        for row in rows
                    ]
                )
            )

        merged.append(
            {
                "config_id": config_id,
                "pairs": len(rows),
                "retained_minus_reset_early_ret_complete_cohort": delta(
                    "early_ret_complete_cohort"
                ),
                "retained_minus_reset_worst_product_fill": delta(
                    "worst_product_fill"
                ),
                "retained_minus_reset_unresolved_orders": delta(
                    "unresolved_orders"
                ),
                "retained_minus_reset_lost_orders": delta("lost_orders"),
                "max_mass_residual": float(
                    max(
                        abs(float(metrics["mass_residual"]))
                        for row in rows
                        for metrics in (row["retained"], row["reset"])
                    )
                ),
            }
        )
    return {"pareto": merged, "pareto_pairs": raw}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("convergence", "pareto"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    shards = [json.loads(path.read_text()) for path in args.inputs]
    _validate_common(shards)
    root_blocks = [list(shard["history_roots"]) for shard in shards]
    flattened = [root for block in root_blocks for root in range(int(block[0]), int(block[1]) + 1)]
    if len(flattened) != len(set(flattened)):
        raise ValueError("root blocks overlap")
    payload = merge_convergence(shards) if args.phase == "convergence" else merge_pareto(shards)
    result = {
        "schema_version": "q_r1_comparator_v2_merged_v1",
        "claim_status": "BURNED_DEVELOPMENT_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "root_blocks": root_blocks,
        "history_roots": [min(flattened), max(flattened)],
        "states": sum(int(shard["states"]) for shard in shards),
        "conditional_path_budgets": shards[0]["conditional_path_budgets"],
        "selection_performed": False,
        "learner_return_used": False,
        "retained_minus_reset_used_for_selection": False,
        "input_paths": [str(path) for path in args.inputs],
        **payload,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if not key.endswith("_pairs")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

