#!/usr/bin/env python3
"""Materialize Program E training/validation tapes and frozen normalizers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.dra2_convoy import static_policies  # noqa: E402
from supply_chain.dra2_experiment import FAMILIES, advance_including, make_sim, materialize_tape  # noqa: E402
from supply_chain.dra2_policy_env import OBSERVATION_KEYS  # noqa: E402


CONTRACT_ID = "program_e_policy_realizability_v1"


def make_tapes(seed_start: int, n_tapes: int, split: str) -> list[dict]:
    if n_tapes % len(FAMILIES):
        raise ValueError("Program E splits must be balanced across four families")
    per_family = n_tapes // len(FAMILIES)
    return [
        materialize_tape(
            seed_start + index,
            FAMILIES[index // per_family],
            16,
            split,
            contract_id=CONTRACT_ID,
            tape_prefix="program-e",
        )
        for index in range(n_tapes)
    ]


def collect_normalizers(tapes: list[dict]) -> dict:
    observations = {key: [] for key in OBSERVATION_KEYS}
    service_values = []
    backlog_age_values = []
    for tape in tapes:
        for policy in static_policies():
            sim, start = make_sim(tape)
            end = start + 56 * 24.0
            resource_start = sim.op8_convoy_metrics()
            while sim.env.now < end - 1e-9:
                raw = dict(sim.get_op8_convoy_observation())
                raw["departures_to_date"] = (
                    sim.op8_convoy_departures - resource_start["op8_convoy_departures"]
                )
                raw["unavailable_hours_to_date"] = (
                    sim.op8_convoy_vehicle_hours
                    - resource_start["op8_convoy_unavailable_hours"]
                )
                for key in OBSERVATION_KEYS:
                    observations[key].append(abs(float(raw[key])))
                action = policy.action(raw)
                sim.apply_op8_convoy_action(action, source="program_e_normalizer")
                advance_including(sim, min(end, float(sim.env.now) + 24.0))
                service_values.append(float(sim.pending_backorder_qty) * 24.0)
                now = float(sim.env.now)
                backlog_age_values.append(sum(
                    float(order.remaining_qty)
                    * max(0.0, now - float(order.OPTj)) * 24.0
                    for order in sim.pending_backorders
                ))
    return {
        "contract_id": CONTRACT_ID,
        "fit_split": "training",
        "observation_scales": {
            key: max(float(np.quantile(values, .95)), 1.0)
            for key, values in observations.items()
        },
        "reward_scales": {
            "daily_service_loss_p95": max(float(np.quantile(service_values, .95)), 1.0),
            "daily_backlog_age_p95": max(float(np.quantile(backlog_age_values, .95)), 1.0),
        },
        "n_static_daily_samples": len(service_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/program_e/data"))
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    training = make_tapes(900001, 80, "training")
    validation = make_tapes(910001, 20, "validation")
    (args.output_dir / "training_tapes.json").write_text(
        json.dumps(training, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "validation_tapes.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    normalizers = collect_normalizers(training)
    (args.output_dir / "normalizers.json").write_text(
        json.dumps(normalizers, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verdict = {
        "training_tapes": len(training), "validation_tapes": len(validation),
        "virgin_tapes_opened": 0, "normalizers_frozen": True,
        "interpretation": "PROGRAM_E_DATA_READY",
    }
    (args.output_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
