#!/usr/bin/env python3
"""DRA-2 same-contract finite-convoy static frontier."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_dra1_static_frontier import boot_ci  # noqa: E402
from supply_chain.dra2_convoy import ConvoyThresholdPolicy, static_policies  # noqa: E402
from supply_chain.dra2_experiment import FAMILIES, materialize_tape, run_static_policy  # noqa: E402


DEFAULT_OUTPUT = Path("results/program_d/dra2_static_frontier_smoke")
THESIS_COMPARATOR = (5_000.0, 48.0)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def relative(candidate: float, baseline: float) -> float:
    return (float(candidate) - float(baseline)) / max(abs(float(baseline)), 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed-start", type=int, default=840001)
    parser.add_argument("--n-tapes", type=int, default=4)
    parser.add_argument("--horizon-weeks", type=int, default=16)
    parser.add_argument("--n-boot", type=int, default=1_000)
    parser.add_argument("--face-validation-accepted", action="store_true")
    args = parser.parse_args()
    if args.n_tapes % 4:
        raise ValueError("n-tapes must be divisible by four")
    if args.n_tapes > 4 and not args.face_validation_accepted:
        raise RuntimeError(
            "Calibration is blocked until Garrido validates the 24h return extension."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_family = args.n_tapes // 4
    tapes = [
        materialize_tape(
            args.seed_start + i,
            FAMILIES[i // per_family],
            args.horizon_weeks,
            "smoke" if args.n_tapes == 4 else "calibration",
        )
        for i in range(args.n_tapes)
    ]
    (args.output_dir / "tapes.json").write_text(
        json.dumps(tapes, indent=2, sort_keys=True), encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    for tape in tapes:
        expected = None
        for policy in static_policies():
            result = run_static_policy(tape, policy)
            hashes = (result["risk_sha256"], result["demand_sha256"])
            if expected is None:
                expected = hashes
            elif hashes != expected:
                raise RuntimeError(f"FAIL_EXOGENOUS_CRN {tape['tape_id']}")
            rows.append(
                {"tape_id": tape["tape_id"], "family": tape["family"],
                 "seed": tape["seed"], "policy_id": policy.policy_id,
                 "inventory_threshold": policy.inventory_threshold,
                 "maximum_wait_hours": policy.maximum_wait_hours, **result}
            )
        print(f"[dra2-static] {tape['tape_id']} complete", flush=True)
    write_csv(args.output_dir / "policy_tape_rows.csv", rows)

    baseline = {
        row["tape_id"]: row for row in rows
        if float(row["inventory_threshold"]) == THESIS_COMPARATOR[0]
        and float(row["maximum_wait_hours"]) == THESIS_COMPARATOR[1]
    }
    summaries: list[dict[str, Any]] = []
    for policy in static_policies():
        selected = [row for row in rows if row["policy_id"] == policy.policy_id]
        ret = boot_ci([float(row["ret_excel"]) for row in selected], 661, args.n_boot)
        service = float(np.mean([row["service_loss_auc_ration_hours"] for row in selected]))
        departures = float(np.mean([row["op8_convoy_departures"] for row in selected]))
        vehicle_hours = float(np.mean([row["op8_convoy_vehicle_hours"] for row in selected]))
        guardrails = {}
        for name, metric in (
            ("lost", "lost_orders"),
            ("service", "service_loss_auc_ration_hours"),
            ("backlog", "backlog_auc"),
        ):
            values = [relative(row[metric], baseline[row["tape_id"]][metric]) for row in selected]
            guardrails[name] = boot_ci(values, 670 + len(summaries), args.n_boot)
        admissible = (
            max(float(row["mass_residual"]) for row in selected) <= 1e-6
            and max(float(row["op8_convoy_resource_residual"]) for row in selected) <= 1e-9
            and all(guardrails[name][2] <= 0.02 for name in guardrails)
        )
        summaries.append(
            {"policy_id": policy.policy_id,
             "inventory_threshold": policy.inventory_threshold,
             "maximum_wait_hours": policy.maximum_wait_hours,
             "mean_ret": ret[0], "ret_ci_low": ret[1], "ret_ci_high": ret[2],
             "mean_service_loss": service, "mean_departures": departures,
             "mean_vehicle_hours": vehicle_hours,
             "mean_load_factor": float(np.mean([row["op8_convoy_load_factor"] for row in selected])),
             "mean_live_fraction": float(np.mean([row["live_fraction"] for row in selected])),
             "lost_deg_ci_high": guardrails["lost"][2],
             "service_deg_ci_high": guardrails["service"][2],
             "backlog_deg_ci_high": guardrails["backlog"][2],
             "admissible": admissible}
        )
    for row in summaries:
        row["pareto_nondominated"] = not any(
            other is not row
            and float(other["mean_ret"]) >= float(row["mean_ret"])
            and float(other["mean_service_loss"]) <= float(row["mean_service_loss"])
            and float(other["mean_departures"]) <= float(row["mean_departures"])
            and (
                float(other["mean_ret"]) > float(row["mean_ret"])
                or float(other["mean_service_loss"]) < float(row["mean_service_loss"])
                or float(other["mean_departures"]) < float(row["mean_departures"])
            )
            for other in summaries
        )
    write_csv(args.output_dir / "policy_summary.csv", summaries)
    admissible = [row for row in summaries if row["admissible"]]
    best = max(admissible, key=lambda row: float(row["mean_ret"])) if admissible else None
    verdict = {
        "gate": "DRA2_STATIC_FRONTIER_SMOKE" if args.n_tapes == 4 else "DRA2_STATIC_FRONTIER",
        "n_tapes": len(tapes), "crn_pass": True,
        "mass_pass": max(float(row["mass_residual"]) for row in rows) <= 1e-6,
        "convoy_conservation_pass": max(float(row["op8_convoy_resource_residual"]) for row in rows) <= 1e-9,
        "best_admissible": best,
        "face_validation_accepted": bool(args.face_validation_accepted),
        "calibration_opened": args.n_tapes > 4,
        "virgin_tapes_opened": 0, "ppo_trained": False,
        "interpretation": (
            "IMPLEMENTATION_SMOKE_PASS" if args.n_tapes == 4 and best
            else "PASS_STATIC_FRONTIER" if best else "FAIL_NO_ADMISSIBLE_STATIC"
        ),
    }
    (args.output_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if best else 2


if __name__ == "__main__":
    raise SystemExit(main())
