#!/usr/bin/env python3
"""Fail-closed terminal adjudication for Program Q result bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CELL_IDS = ("rho75_share90", "rho90_share75", "rho90_share90")


def adjudicate(result: dict, contract: dict) -> dict:
    estimates = result.get("inference", {}).get("estimates", {})
    summaries = result.get("cell_summaries", {})
    integrity = result.get("integrity_gates", {})
    required_integrity = (
        "feedback",
        "replacement_controls",
        "scheduled_resources_exact",
        "mass_partition_demand",
        "ret_full_noninferior",
        "quantity_ret_full_noninferior",
        "worst_product_fill_noninferior",
    )
    integrity_pass = all(integrity.get(name) is True for name in required_integrity)
    adaptation = True
    premium = True
    equivalent = True
    cell_gates = {}
    for cell in CELL_IDS:
        h = estimates.get(f"{cell}::H_OL", {})
        delta = estimates.get(f"{cell}::Delta_N", {})
        summary = summaries.get(cell, {})
        h_pass = h.get("lcb95", float("-inf")) >= 0.01
        premium_pass = delta.get("lcb95", float("-inf")) >= 0.01
        equivalence_pass = (
            delta.get("lcb95", float("-inf")) >= -0.01
            and delta.get("ucb95", float("inf")) <= 0.01
        )
        favorable_pass = summary.get("favorable_tapes_fraction_vs_open_loop", 0.0) >= 0.70
        seed_pass = summary.get("positive_learner_seeds_H_OL", 0) >= 8
        adaptation &= h_pass and favorable_pass and seed_pass
        premium &= premium_pass
        equivalent &= equivalence_pass
        cell_gates[cell] = {
            "H_OL": h_pass,
            "neural_premium": premium_pass,
            "equivalence": equivalence_pass,
            "favorable_tapes": favorable_pass,
            "learner_seeds": seed_pass,
        }
    if not integrity_pass or not adaptation:
        verdict = "STOP_Q_NO_REPLICATED_LEARNED_ADAPTATION"
    elif premium:
        verdict = "PASS_Q_NEURAL_PREMIUM"
    elif equivalent:
        verdict = "PASS_Q_LEARNED_ADAPTATION_CLASSICALLY_EQUIVALENT"
    else:
        verdict = "BOUND_Q_LEARNED_ADAPTATION_ONLY"
    return {
        "schema_version": "program_q_terminal_adjudication_v1",
        "historical_verdicts_unchanged": contract["historical_verdicts_immutable"],
        "cell_gates": cell_gates,
        "integrity_gates": {name: integrity.get(name) for name in required_integrity},
        "verdict": verdict,
        "paper3_authorized": verdict in {
            "PASS_Q_NEURAL_PREMIUM",
            "PASS_Q_LEARNED_ADAPTATION_CLASSICALLY_EQUIVALENT",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("contracts/program_q_frozen_policy_replication_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = adjudicate(
        json.loads(args.result.read_text()), json.loads(args.contract.read_text())
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
