#!/usr/bin/env python3
"""Calibrate and validate the endogenous Garrido DES before any RL training.

The raw workbooks are primary order-level evidence.  This gate deliberately
does not use Excel order/risk tapes inside the DES: the objective is to test
whether endogenous demand, risks, queues and material flow reproduce held-out
CF behavior.  Odd CFs calibrate a finite 3x3 grid; even CFs are untouched
validation cases.  There is no iterative optimizer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import (  # noqa: E402
    THESIS_FAITHFUL_PROTOCOL,
    THESIS_REPLICATION_DOWNSTREAM_Q_SOURCE,
)
from supply_chain.garrido_replication import (  # noqa: E402
    DEFAULT_RAW_WORKBOOKS,
    GarridoCFTarget,
    load_raw_garrido_targets,
)
from supply_chain.ret_thesis import (  # noqa: E402
    compute_order_level_ret_excel_visible_ledger,
    order_has_ret_risk_indicator,
)
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402
from supply_chain.thesis_design import design_spec_for_cfi  # noqa: E402


CALIBRATION_CFS = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19)
VALIDATION_CFS = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
MULTIPLIERS = (1.0, 1.5, 2.0)
DELAYS = (0.0, 48.00744, 54.0)
REFINEMENT_DELAYS = (72.0, 96.0, 120.0)


def _quantile(values: Iterable[float], q: float, *, positive: bool = False) -> float:
    cleaned = [float(v) for v in values if math.isfinite(float(v))]
    if positive:
        cleaned = [v for v in cleaned if v > 0.0]
    return float(np.quantile(cleaned, q)) if cleaned else 0.0


def _target_metrics(target: GarridoCFTarget) -> dict[str, float]:
    orders = list(target.orders)
    risk_active = [order.risk_active for order in orders]
    return {
        # WORKBOOK VIEW (verified on CF1): rows are ATTENDED orders only; the
        # j column keeps the placement index, so placed = max_j and the gap
        # (placed - attended) is lost/unattended + pending + in-flight.
        "placed_orders": float(target.max_j),
        "ut_final": float(target.final_sum_ut),
        "bt_final": float(target.final_sum_bt),
        "n_orders": float(len(orders)),
        "warmup": float(target.warmup_hours),
        "ret_mean": float(target.ret_mean_excel),
        "risk_active_share": float(np.mean(risk_active)) if orders else 0.0,
        "ct_p50": _quantile((o.ctj for o in orders), 0.50),
        "ct_p95": _quantile((o.ctj for o in orders), 0.95),
        "ct_p99": _quantile((o.ctj for o in orders), 0.99),
        "rp_p50_pos": _quantile((o.rpj for o in orders), 0.50, positive=True),
        "rp_p95_pos": _quantile((o.rpj for o in orders), 0.95, positive=True),
        "rp_p99_pos": _quantile((o.rpj for o in orders), 0.99, positive=True),
        "dp_p50_pos": _quantile((o.dpj for o in orders), 0.50, positive=True),
        "dp_p95_pos": _quantile((o.dpj for o in orders), 0.95, positive=True),
        "dp_p99_pos": _quantile((o.dpj for o in orders), 0.99, positive=True),
    }


def _des_metrics(sim: MFSCSimulation) -> dict[str, float]:
    # Garrido's first visible order is placed at the warm-up boundary.  Exclude
    # the DES priming ledger rather than rewarding a different initialization.
    placed = [order for order in sim.orders if float(order.OPTj) >= sim.warmup_time]
    # WORKBOOK VIEW: Garrido's order-level table contains only ATTENDED
    # (delivered) orders — lost/unattended, still-pending, and in-flight
    # orders are excluded from CT/RP/DP/ReT rows and enter only through the
    # cumulative ΣBt/ΣUt columns. Compare like with like.
    orders = [
        order
        for order in placed
        if order.CTj is not None and not getattr(order, "lost", False)
    ]
    lost = [order for order in placed if getattr(order, "lost", False)]
    unresolved = [
        order
        for order in placed
        if order.CTj is None and not getattr(order, "lost", False)
    ]
    ret = compute_order_level_ret_excel_visible_ledger(
        placed,
        current_time=float(sim.env.now),
    )
    risk_active = [order_has_ret_risk_indicator(order) for order in orders]
    completed = orders
    return {
        "placed_orders": float(len(placed)),
        "ut_final": float(ret["final_unattended"]),
        "bt_final": float(ret["final_backorders"]),
        "n_orders": float(len(orders)),
        "warmup": float(sim.warmup_time),
        "ret_mean": float(ret["mean_ret_excel"]),
        "risk_active_share": float(np.mean(risk_active)) if orders else 0.0,
        "ct_p50": _quantile((o.CTj for o in completed), 0.50),
        "ct_p95": _quantile((o.CTj for o in completed), 0.95),
        "ct_p99": _quantile((o.CTj for o in completed), 0.99),
        "rp_p50_pos": _quantile((o.RPj for o in completed), 0.50, positive=True),
        "rp_p95_pos": _quantile((o.RPj for o in completed), 0.95, positive=True),
        "rp_p99_pos": _quantile((o.RPj for o in completed), 0.99, positive=True),
        "dp_p50_pos": _quantile((o.DPj for o in completed), 0.50, positive=True),
        "dp_p95_pos": _quantile((o.DPj for o in completed), 0.95, positive=True),
        "dp_p99_pos": _quantile((o.DPj for o in completed), 0.99, positive=True),
        "production_per_year": float(sim.total_produced)
        / max(float(sim.horizon) / float(sim.hours_per_year), 1e-9),
        "flow_raw_residual": float(sim.flow_ledger()["raw_residual"]),
        "flow_ration_residual": float(sim.flow_ledger()["ration_residual"]),
    }


def _ratio_error(actual: float, target: float) -> float:
    if actual <= 0.0 and target <= 0.0:
        return 0.0
    if actual <= 0.0 or target <= 0.0:
        return 3.0
    return abs(math.log(actual / target))


def _score(des: dict[str, float], target: dict[str, float]) -> float:
    score = 0.0
    score += _ratio_error(des["n_orders"], target["n_orders"])
    score += 0.5 * _ratio_error(des["placed_orders"], target["placed_orders"])
    score += 0.5 * _ratio_error(des["ut_final"], target["ut_final"])
    score += _ratio_error(des["warmup"], target["warmup"])
    score += _ratio_error(max(des["ret_mean"], 1e-9), max(target["ret_mean"], 1e-9))
    score += 2.0 * abs(des["risk_active_share"] - target["risk_active_share"])
    for key in (
        "ct_p50",
        "ct_p95",
        "ct_p99",
        "rp_p50_pos",
        "rp_p95_pos",
        "rp_p99_pos",
        "dp_p50_pos",
        "dp_p95_pos",
        "dp_p99_pos",
    ):
        score += 0.5 * _ratio_error(des[key], target[key])
    return float(score)


def run_cf(
    cfi: int,
    target: GarridoCFTarget,
    *,
    multiplier: float,
    delay: float,
    mechanistic_fulfillment: bool = False,
) -> dict[str, Any]:
    spec = design_spec_for_cfi(cfi)
    sim = MFSCSimulation(
        shifts=spec.shifts,
        seed=int(target.seed),
        horizon=float(spec.horizon_hours),
        risks_enabled=True,
        risk_level="current",
        enabled_risks=set(spec.enabled_risks),
        risk_overrides=dict(spec.risk_overrides),
        risk_occurrence_mode="thesis_window",
        risk_attribution_source="des_events",
        seed_stream_mode="split",
        year_basis=THESIS_FAITHFUL_PROTOCOL["year_basis"],
        warmup_trigger=THESIS_FAITHFUL_PROTOCOL["warmup_trigger"],
        downstream_q_source=THESIS_REPLICATION_DOWNSTREAM_Q_SOURCE,
        r14_defect_mode=THESIS_FAITHFUL_PROTOCOL["r14_defect_mode"],
        raw_material_flow_mode=THESIS_FAITHFUL_PROTOCOL["raw_material_flow_mode"],
        raw_material_order_up_to_multiplier=float(multiplier),
        demand_on_hand_fulfillment_delay=(0.0 if mechanistic_fulfillment else float(delay)),
        replenishment_route_aware=True,
        procurement_contract_mode="causal_coupled",
        order_fulfillment_mode=(
            "op9_linked" if mechanistic_fulfillment else "legacy_theatre_stock"
        ),
        # Freeze the exact mechanism evaluated by this gate.  Do not inherit
        # constructor defaults: fixed-clock release and tandem convoy capacity
        # are separate hypotheses, not interchangeable implementations.
        op9_dispatch_policy="fixed_clock_daily",
        downstream_transport_capacity_mode="parallel",
        # R24 surge stress window (CF11 evidence: 75% of attended orders carry
        # R24>0 ≈ 6.8 orders/event ≈ one week of placements). Opt-in here;
        # default 0.0 keeps legacy point events bitwise for frozen lanes.
        r24_attribution_window_hours=(168.0 if mechanistic_fulfillment else 0.0),
        demand_start_after_warmup=bool(mechanistic_fulfillment),
        ret_recovery_period_mode=(
            "elapsed" if mechanistic_fulfillment else "disruption"
        ),
    ).run()
    observed = _des_metrics(sim)
    expected = _target_metrics(target)
    row: dict[str, Any] = {
        "cfi": cfi,
        "split": "calibration" if cfi in CALIBRATION_CFS else "validation",
        "multiplier": float(multiplier),
        "delay": float(delay),
        "fulfillment_mode": (
            "op9_linked" if mechanistic_fulfillment else "legacy_fixed_delay"
        ),
        "op9_dispatch_policy": (
            "fixed_clock_daily" if mechanistic_fulfillment else "not_applicable"
        ),
        "downstream_transport_capacity_mode": (
            "parallel" if mechanistic_fulfillment else "not_applicable"
        ),
        "ret_recovery_period_mode": (
            "elapsed" if mechanistic_fulfillment else "disruption"
        ),
        "seed": int(target.seed),
        "score": _score(observed, expected),
    }
    for key, value in expected.items():
        row[f"excel_{key}"] = value
    for key, value in observed.items():
        row[f"des_{key}"] = value
    return row


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_cf": len(rows),
        "mean_score": float(np.mean([row["score"] for row in rows])),
        "median_score": float(np.median([row["score"] for row in rows])),
        "mean_abs_ret_gap": float(
            np.mean(
                [abs(row["des_ret_mean"] - row["excel_ret_mean"]) for row in rows]
            )
        ),
        "mean_abs_risk_share_gap": float(
            np.mean(
                [
                    abs(
                        row["des_risk_active_share"]
                        - row["excel_risk_active_share"]
                    )
                    for row in rows
                ]
            )
        ),
        "max_abs_raw_residual": float(
            max(abs(row["des_flow_raw_residual"]) for row in rows)
        ),
        "max_abs_ration_residual": float(
            max(abs(row["des_flow_ration_residual"]) for row in rows)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/audits/garrido_reference_v2_gate"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    targets = load_raw_garrido_targets(DEFAULT_RAW_WORKBOOKS)
    grid_rows: list[dict[str, Any]] = []
    grid_summary: list[dict[str, Any]] = []
    for multiplier in MULTIPLIERS:
        for delay in DELAYS:
            rows = [
                run_cf(cfi, targets[cfi], multiplier=multiplier, delay=delay)
                for cfi in CALIBRATION_CFS
            ]
            grid_rows.extend(rows)
            grid_summary.append(
                {
                    "multiplier": multiplier,
                    "delay": delay,
                    **_aggregate(rows),
                }
            )
            print(
                f"grid multiplier={multiplier:.1f} delay={delay:.5f} "
                f"score={grid_summary[-1]['mean_score']:.4f}",
                flush=True,
            )

    global_best = min(grid_summary, key=lambda row: row["mean_score"])
    # One and only one refinement around the workbook CT medians.  The raw
    # CFs center near 96 h, so test three interpretable daily-cycle values at
    # the globally selected material multiplier; do not optimize continuously.
    for delay in REFINEMENT_DELAYS:
        rows = [
            run_cf(
                cfi,
                targets[cfi],
                multiplier=float(global_best["multiplier"]),
                delay=delay,
            )
            for cfi in CALIBRATION_CFS
        ]
        grid_rows.extend(rows)
        grid_summary.append(
            {
                "stage": "refinement",
                "multiplier": float(global_best["multiplier"]),
                "delay": delay,
                **_aggregate(rows),
            }
        )
        print(
            f"refine multiplier={global_best['multiplier']:.1f} "
            f"delay={delay:.1f} score={grid_summary[-1]['mean_score']:.4f}",
            flush=True,
        )
    for row in grid_summary:
        row.setdefault("stage", "global")
    best = min(grid_summary, key=lambda row: row["mean_score"])
    # Replace the fitted ledger delay with the physical Op9->Op10->Op12 order
    # path.  This is the only candidate eligible for promotion.
    mechanistic_calibration_rows = [
        run_cf(
            cfi,
            targets[cfi],
            multiplier=float(best["multiplier"]),
            delay=0.0,
            mechanistic_fulfillment=True,
        )
        for cfi in CALIBRATION_CFS
    ]
    validation_rows = [
        run_cf(
            cfi,
            targets[cfi],
            multiplier=float(best["multiplier"]),
            delay=0.0,
            mechanistic_fulfillment=True,
        )
        for cfi in VALIDATION_CFS
    ]
    all_best_rows = mechanistic_calibration_rows + validation_rows

    # Strict promotion gates.  Failure is a scientific result, not a script error.
    validation_summary = _aggregate(validation_rows)
    gates = {
        "mass_conservation": validation_summary["max_abs_raw_residual"] <= 1e-6
        and validation_summary["max_abs_ration_residual"] <= 1e-6,
        "mean_abs_ret_gap_le_0_02": validation_summary["mean_abs_ret_gap"] <= 0.02,
        "mean_abs_risk_share_gap_le_0_05": validation_summary[
            "mean_abs_risk_share_gap"
        ]
        <= 0.05,
        "all_warmup_within_10pct": all(
            abs(row["des_warmup"] - row["excel_warmup"])
            / max(row["excel_warmup"], 1e-9)
            <= 0.10
            for row in validation_rows
        ),
        "all_ct_p50_p95_ratio_0_75_1_33": all(
            0.75
            <= row[f"des_{key}"] / max(row[f"excel_{key}"], 1e-9)
            <= 1.33
            for row in validation_rows
            for key in ("ct_p50", "ct_p95")
        ),
        "all_rp_p50_p95_ratio_0_75_1_33": all(
            0.75
            <= row[f"des_{key}"] / max(row[f"excel_{key}"], 1e-9)
            <= 1.33
            for row in validation_rows
            for key in ("rp_p50_pos", "rp_p95_pos")
        ),
    }
    promoted = all(gates.values())
    payload = {
        "artifact": "garrido_reference_v2_endogenous_gate",
        "evidence_contract": {
            "primary": [str(path) for path in DEFAULT_RAW_WORKBOOKS],
            "demand_source": "thesis_calendar",
            "risk_attribution_source": "des_events",
            "forensic_excel_tapes_used": False,
            "calibration_cfs": CALIBRATION_CFS,
            "validation_cfs": VALIDATION_CFS,
            "finite_grid": {
                "multipliers": MULTIPLIERS,
                "global_delays": DELAYS,
                "single_refinement_delays": REFINEMENT_DELAYS,
            },
        },
        "best_calibration": best,
        "mechanistic_calibration_summary": _aggregate(mechanistic_calibration_rows),
        "validation_summary": validation_summary,
        "gates": gates,
        "promoted_to_garrido_reference_v2": promoted,
        "rows": all_best_rows,
    }
    (args.output_dir / "gate.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "grid_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grid_summary[0]))
        writer.writeheader()
        writer.writerows(grid_summary)
    with (args.output_dir / "best_cf_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_best_rows[0]))
        writer.writeheader()
        writer.writerows(all_best_rows)
    print(json.dumps({"best": best, "validation": validation_summary, "gates": gates}, indent=2))
    print(f"Saved: {args.output_dir / 'gate.json'}")


if __name__ == "__main__":
    main()
