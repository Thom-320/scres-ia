#!/usr/bin/env python3
"""Evaluate a frozen Q-R1 comparator on burned retained/reset pairs only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_program_q_replication import scheduler  # noqa: E402
from scripts.run_q_r1_comparator_v2_preflight import (  # noqa: E402
    BURNED_HIGH,
    BURNED_LOW,
    development_states,
)
from scripts.run_q_r1_successor_abc import fixed_theta_belief  # noqa: E402
from supply_chain.q_r1_comparator_v2 import (  # noqa: E402
    ComparatorV2Config,
    NoFeasibleStructuredAction,
    comparator_v2_calendar,
)
from supply_chain.q_r1_retained_learning import evaluate_calendar  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_freeze(path: Path) -> tuple[dict[str, object], ComparatorV2Config]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "FROZEN_BURNED_CALIBRATION_NO_FRESH_SEEDS":
        raise ValueError("comparator freeze is not executable")
    if payload.get("selection_used_learner_return") is not False:
        raise ValueError("freeze used learner return")
    if payload.get("selection_used_retained_minus_reset") is not False:
        raise ValueError("freeze used retained-minus-reset")
    config = ComparatorV2Config(**payload["config"])
    if config.config_id != payload.get("config_id"):
        raise ValueError("frozen config id does not match config fields")
    receipt = ROOT / str(payload["convergence_receipt"])
    if not receipt.exists() or sha256(receipt) != payload.get("convergence_sha256"):
        raise ValueError("frozen convergence receipt hash mismatch")
    convergence = json.loads(receipt.read_text())
    rows = [
        row
        for row in convergence.get("convergence", [])
        if row.get("low_config") == config.config_id
    ]
    if len(rows) != 1 or rows[0].get("convergence_pass") is not True:
        raise ValueError("frozen comparator did not pass its convergence receipt")
    return payload, config


def run(args: argparse.Namespace) -> dict[str, object]:
    freeze, config = load_freeze(args.freeze)
    if not (BURNED_LOW <= args.seed_start <= BURNED_HIGH):
        raise ValueError("Pareto evaluation must use burned roots")
    if args.seed_start + args.histories - 1 > BURNED_HIGH:
        raise ValueError("Pareto evaluation would leave burned roots")
    states = development_states(args)
    sched = scheduler()
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    abstentions: list[dict[str, object]] = []
    for campaign, retained_prior in states:
        pair: dict[str, object] = {
            "config_id": config.config_id,
            "history_root": campaign.history_root,
            "campaign_index": campaign.campaign_index,
            "persistence_mode": campaign.persistence_mode,
            "retained_prior": retained_prior,
        }
        complete = True
        for label, prior in (("retained", retained_prior), ("reset", 0.5)):
            if time.perf_counter() - started > args.hard_cap_seconds:
                raise TimeoutError("frozen Pareto shard hard cap exceeded")
            try:
                calendar, detail = comparator_v2_calendar(
                    campaign=campaign,
                    belief=fixed_theta_belief(prior),
                    scheduler=sched,
                    config=config,
                )
            except NoFeasibleStructuredAction as exc:
                abstentions.append(
                    {
                        "history_root": campaign.history_root,
                        "campaign_index": campaign.campaign_index,
                        "persistence_mode": campaign.persistence_mode,
                        "arm": label,
                        "reason": str(exc),
                    }
                )
                complete = False
                break
            pair[label] = evaluate_calendar(
                campaign=campaign,
                calendar=calendar,
                scheduler=sched,
            )
            pair[f"{label}_calendar"] = list(calendar)
            pair[f"{label}_diagnostics"] = detail
        if complete:
            rows.append(pair)
    return {
        "schema_version": "q_r1_comparator_v2_frozen_pareto_v1",
        "claim_status": "BURNED_DEVELOPMENT_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "frozen_pareto",
        "freeze_path": str(args.freeze.relative_to(ROOT)),
        "freeze_sha256": sha256(args.freeze),
        "convergence_receipt": freeze["convergence_receipt"],
        "config_id": config.config_id,
        "config": freeze["config"],
        "history_roots": [args.seed_start, args.seed_start + args.histories - 1],
        "states": len(states),
        "conditional_path_budgets": [config.conditional_paths, config.conditional_paths],
        "value_indifference_tolerance": config.value_indifference_tolerance,
        "tie_breaker": config.tie_breaker,
        "selection_performed": False,
        "learner_return_used": False,
        "retained_minus_reset_used_for_selection": False,
        "pareto": [],
        "pareto_pairs": rows,
        "abstentions": abstentions,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--histories", type=int, default=6)
    parser.add_argument("--campaigns", type=int, default=12)
    parser.add_argument("--states", type=int, default=12)
    parser.add_argument("--hard-cap-seconds", type=float, default=14_400.0)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "config_id": result["config_id"],
                "history_roots": result["history_roots"],
                "pairs": len(result["pareto_pairs"]),
                "abstentions": len(result["abstentions"]),
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
