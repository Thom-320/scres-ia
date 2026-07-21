#!/usr/bin/env python3
"""Causal demand-history conversion for bounded direct-SimPy U1 candidates."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.paper2_exhaustive_search.program_s_design import build_program_s_risk_tape  # noqa: E402
from research.paper2_exhaustive_search.program_s_transducer import run_program_s_direct  # noqa: E402
from scripts.run_program_s_s1_shard import make_cell, resolve_point  # noqa: E402
from supply_chain.program_o_full_des_transducer import direct_full_des_vector  # noqa: E402

DESIGN = ROOT / "research/paper2_exhaustive_search/program_s_native_morris_design_v1_1.json"
PARENT = ROOT / "contracts/program_o_full_des_hpi_translation_v1.json"
DISCOVERY = ROOT / "results/program_u1/direct_bounded_discovery_v1/result.json"
BURNED_TAPES = (7_430_001, 7_430_002, 7_430_003)


def scheduler():
    parent = json.loads(PARENT.read_text())
    return parent["action"]["within_week_schedulers"][parent["action"]["primary_scheduler"]]


def static_calendars():
    rows = {(value,) * 8 for value in range(4)}
    rows.update({(0, 3) * 4, (3, 0) * 4, (1, 2) * 4, (2, 1) * 4})
    rng = np.random.default_rng(20260720)
    while len(rows) < 32:
        rows.add(tuple(map(int, rng.integers(0, 4, size=8))))
    return tuple(sorted(rows))


def demand_counts(sim) -> tuple[int, ...]:
    start = float(sim.program_o_decision_start or 0.0); counts = [0] * 8
    for order in sim.orders:
        week = int((float(order.OPTj) - start) // 168.0)
        if 0 <= week < 8 and order.requested_product_id == "P_C":
            counts[week] += 1
    return tuple(counts)


def adaptive_calendars(counts: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
    output = {}
    for alpha in (0.25, 0.50, 0.75):
        estimate = 3.0; calendar = []
        for week in range(8):
            calendar.append(int(np.clip(round(estimate / 2.0), 0, 3)))
            estimate = alpha * counts[week] + (1.0 - alpha) * estimate
        output[f"ewma_{alpha:.2f}"] = tuple(calendar)
    lag = [2]
    trend = [2]
    for week in range(1, 8):
        lag.append(int(np.clip(round(counts[week - 1] / 2.0), 0, 3)))
        projected = counts[week - 1] if week < 2 else counts[week - 1] + 0.5 * (counts[week - 1] - counts[week - 2])
        trend.append(int(np.clip(round(projected / 2.0), 0, 3)))
    output["lag1"] = tuple(lag); output["lag_trend"] = tuple(trend)
    return output


def main() -> int:
    sched = scheduler(); design = json.loads(DESIGN.read_text()); discovery = json.loads(DISCOVERY.read_text()); statics = static_calendars(); rows = []
    for candidate in discovery["candidate_rows"]:
        group, point = resolve_point(candidate["group"], candidate["trajectory"], candidate["point"])
        cell = make_cell(group, point, candidate["product_cell"])
        per_tape = []
        for tape in BURNED_TAPES:
            built = build_program_s_risk_tape(cell, tape_id=tape, horizon_hours=8 * 168)
            reference = run_program_s_direct(seed=tape, calendar=[0] * 8, scheduler=sched, cell=cell, risk_event_tape=built["events"])
            rules = adaptive_calendars(demand_counts(reference)); policies = {f"static_{i}": calendar for i, calendar in enumerate(statics)} | rules
            metrics = {}
            for policy_id, calendar in policies.items():
                sim = run_program_s_direct(seed=tape, calendar=calendar, scheduler=sched, cell=cell, risk_event_tape=built["events"])
                vector = direct_full_des_vector(sim, sim.product_outcome_panel())
                metrics[policy_id] = {key: float(vector[key]) for key in ("ret_visible", "worst_product_fill", "lost_orders", "gross_production_quantity")}
            per_tape.append(metrics)
        deltas = []; fill_deltas = []; selected_rules = []; selected_statics = []
        rule_ids = sorted(key for key in per_tape[0] if not key.startswith("static_")); static_ids = sorted(key for key in per_tape[0] if key.startswith("static_"))
        for test_index in range(len(BURNED_TAPES)):
            train = [index for index in range(len(BURNED_TAPES)) if index != test_index]
            rule = max(rule_ids, key=lambda name: np.mean([per_tape[index][name]["ret_visible"] for index in train]))
            static = max(static_ids, key=lambda name: np.mean([per_tape[index][name]["ret_visible"] for index in train]))
            deltas.append(per_tape[test_index][rule]["ret_visible"] - per_tape[test_index][static]["ret_visible"])
            fill_deltas.append(per_tape[test_index][rule]["worst_product_fill"] - per_tape[test_index][static]["worst_product_fill"])
            selected_rules.append(rule); selected_statics.append(static)
        rows.append(candidate | {
            "classical_h_obs_loo_mean": float(np.mean(deltas)),
            "classical_h_obs_by_tape": list(map(float, deltas)),
            "worst_product_delta_mean": float(np.mean(fill_deltas)),
            "selected_rules": selected_rules, "selected_statics": selected_statics,
            "resources_exact": all(len({metrics[name]["gross_production_quantity"] for name in metrics}) == 1 for metrics in per_tape),
        })
    promoted = [row for row in rows if row["classical_h_obs_loo_mean"] >= 0.015 and row["worst_product_delta_mean"] >= -0.02 and row["resources_exact"]]
    promoted.sort(key=lambda row: (row["distance"], -row["classical_h_obs_loo_mean"]))
    payload = {
        "schema_version": "program_u1_direct_classical_conversion_v1",
        "claim_status": "EXPLORATORY_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "burned_tapes": list(BURNED_TAPES), "new_scientific_seeds_opened": [],
        "selection": "leave_one_complete_tape_out",
        "candidate_count": len(rows), "promotion_count": len(promoted), "promoted_rows": promoted,
        "rows": rows,
        "verdict": "PASS_U1_DIRECT_CLASSICAL_CONVERSION_CANDIDATES" if promoted else "STOP_U1_DIRECT_NO_CLASSICAL_OBSERVABLE_CONVERSION",
        "hybrid_training_authorized": False,
    }
    output = ROOT / "results/program_u1/direct_classical_conversion_v1/result.json"
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
