#!/usr/bin/env python3
"""Mechanism audit for endogenous Garrido fidelity (no RL, no tape replay).

This audit separates three questions that the aggregate gate previously mixed:

1. which placed orders become visible, lost, pending, or in flight;
2. which risk family is attributed to each visible order; and
3. whether downstream release and transport capacity reproduce CT/RP/DP tails.

Odd CFs are calibration cases.  Even CFs must remain untouched until a single
mechanism is frozen.  The script deliberately exposes mechanism categories,
not continuously tuneable delay constants.
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
)
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402
from supply_chain.thesis_design import design_spec_for_cfi  # noqa: E402


CALIBRATION_CFS = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19)
VALIDATION_CFS = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
VARIANTS = {
    "clock_parallel": (
        "fixed_clock_daily",
        "parallel",
        "aggregate_line",
        "completion_relative",
        "deferred_first_cycle",
    ),
    "clock_tandem": (
        "fixed_clock_daily",
        "tandem_capacity_one",
        "aggregate_line",
        "completion_relative",
        "deferred_first_cycle",
    ),
    "headway_parallel": (
        "ready_headway",
        "parallel",
        "aggregate_line",
        "completion_relative",
        "deferred_first_cycle",
    ),
    "headway_tandem": (
        "ready_headway",
        "tandem_capacity_one",
        "aggregate_line",
        "completion_relative",
        "deferred_first_cycle",
    ),
    "clock_parallel_serial": (
        "fixed_clock_daily",
        "parallel",
        "serial_wip",
        "completion_relative",
        "deferred_first_cycle",
    ),
    "headway_parallel_serial": (
        "ready_headway",
        "parallel",
        "serial_wip",
        "completion_relative",
        "deferred_first_cycle",
    ),
    "clock_parallel_release": (
        "fixed_clock_daily",
        "parallel",
        "aggregate_line",
        "start_to_start",
        "deferred_first_cycle",
    ),
    "clock_parallel_release_serial": (
        "fixed_clock_daily",
        "parallel",
        "serial_wip",
        "start_to_start",
        "deferred_first_cycle",
    ),
    "headway_parallel_release": (
        "ready_headway",
        "parallel",
        "aggregate_line",
        "start_to_start",
        "deferred_first_cycle",
    ),
    "clock_parallel_release_r12_initial": (
        "fixed_clock_daily",
        "parallel",
        "aggregate_line",
        "start_to_start",
        "include_initial_cycle",
    ),
    "clock_parallel_r12_initial": (
        "fixed_clock_daily",
        "parallel",
        "aggregate_line",
        "completion_relative",
        "include_initial_cycle",
    ),
    "clock_parallel_independent_rng": (
        "fixed_clock_daily",
        "parallel",
        "aggregate_line",
        "completion_relative",
        "deferred_first_cycle",
    ),
    "clock_parallel_release_independent_rng": (
        "fixed_clock_daily",
        "parallel",
        "aggregate_line",
        "start_to_start",
        "deferred_first_cycle",
    ),
    "clock_parallel_release_r12_initial_independent_rng": (
        "fixed_clock_daily",
        "parallel",
        "aggregate_line",
        "start_to_start",
        "include_initial_cycle",
    ),
}


def _q(values: Iterable[float], probability: float) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.quantile(clean, probability)) if clean else 0.0


def _risk_ids(target: GarridoCFTarget) -> tuple[str, ...]:
    return tuple(sorted({column.split("_")[0] for column in target.risk_columns}))


def _excel_risk_positive(order: Any, risk_id: str) -> bool:
    return any(
        str(key).split("_")[0] == risk_id and float(value) > 0.0
        for key, value in order.risk_values.items()
    )


def _des_risk_positive(order: Any, risk_id: str) -> bool:
    return any(
        str(key).split("_")[0] == risk_id and float(value) > 0.0
        for key, value in order.ret_risk_indicators.items()
    )


def _conditional_metrics(orders: list[Any], predicate: Any, prefix: str) -> dict[str, float]:
    selected = [order for order in orders if predicate(order)]
    n_total = len(orders)
    def value(order: Any, des_name: str, target_name: str) -> float:
        return float(getattr(order, des_name, getattr(order, target_name, 0.0)) or 0.0)
    return {
        f"{prefix}_share": float(len(selected) / n_total) if n_total else 0.0,
        f"{prefix}_ct_p50": _q((value(order, "CTj", "ctj") for order in selected), 0.50),
        f"{prefix}_ct_p95": _q((value(order, "CTj", "ctj") for order in selected), 0.95),
        f"{prefix}_rp_p50": _q((value(order, "RPj", "rpj") for order in selected), 0.50),
        f"{prefix}_rp_p95": _q((value(order, "RPj", "rpj") for order in selected), 0.95),
        f"{prefix}_dp_p50": _q((value(order, "DPj", "dpj") for order in selected), 0.50),
        f"{prefix}_dp_p95": _q((value(order, "DPj", "dpj") for order in selected), 0.95),
    }


def _run(cfi: int, target: GarridoCFTarget, variant: str, rp_mode: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    (
        dispatch_policy,
        transport_mode,
        assembly_mode,
        release_mode,
        risk_initialization_mode,
    ) = VARIANTS[variant]
    risk_rng_mode = "per_risk" if variant.endswith("independent_rng") else "shared"
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
        raw_material_order_up_to_multiplier=1.0,
        replenishment_route_aware=True,
        procurement_contract_mode="causal_coupled",
        order_fulfillment_mode="op9_linked",
        assembly_flow_mode=assembly_mode,
        periodic_release_mode=release_mode,
        operational_risk_initialization_mode=risk_initialization_mode,
        risk_rng_mode=risk_rng_mode,
        op9_dispatch_policy=dispatch_policy,
        downstream_transport_capacity_mode=transport_mode,
        demand_start_after_warmup=True,
        ret_recovery_period_mode=rp_mode,
    ).run()

    placed = [order for order in sim.orders if float(order.OPTj) >= sim.warmup_time]
    visible = [
        order
        for order in placed
        if order.CTj is not None and not bool(getattr(order, "lost", False))
    ]
    lost = [order for order in placed if bool(getattr(order, "lost", False))]
    unresolved = [
        order
        for order in placed
        if order.CTj is None and not bool(getattr(order, "lost", False))
    ]
    ledger = compute_order_level_ret_excel_visible_ledger(
        placed, current_time=float(sim.env.now)
    )
    event_counts: dict[str, int] = {}
    event_hours: dict[str, float] = {}
    for event in sim.risk_events:
        event_counts[event.risk_id] = event_counts.get(event.risk_id, 0) + 1
        event_hours[event.risk_id] = event_hours.get(event.risk_id, 0.0) + float(
            event.duration
        )

    summary: dict[str, Any] = {
        "cfi": cfi,
        "split": "calibration" if cfi in CALIBRATION_CFS else "validation",
        "variant": variant,
        "dispatch_policy": dispatch_policy,
        "transport_mode": transport_mode,
        "assembly_mode": assembly_mode,
        "release_mode": release_mode,
        "risk_initialization_mode": risk_initialization_mode,
        "risk_rng_mode": risk_rng_mode,
        "rp_mode": rp_mode,
        "seed": int(target.seed),
        "excel_placed": int(target.max_j),
        "des_placed": len(placed),
        "excel_visible": len(target.orders),
        "des_visible": len(visible),
        "excel_ut_final": float(target.final_sum_ut),
        "des_ut_final": float(ledger["final_unattended"]),
        "excel_bt_final": float(target.final_sum_bt),
        "des_bt_final": float(ledger["final_backorders"]),
        "des_lost": len(lost),
        "des_unresolved": len(unresolved),
        "excel_warmup": float(target.warmup_hours),
        "des_warmup": float(sim.warmup_time),
        "first_contract_completion": (
            float(sim.contract_completion_events[0][0])
            if sim.contract_completion_events
            else None
        ),
        "first_supplier_delivery": (
            float(sim.supplier_delivery_events[0][0])
            if sim.supplier_delivery_events
            else None
        ),
        "first_ration_production": (
            float(sim.daily_production[0][0]) if sim.daily_production else None
        ),
        "pre_warmup_risk_counts": {
            risk_id: sum(
                1
                for event in sim.risk_events
                if event.risk_id == risk_id and float(event.start_time) < sim.warmup_time
            )
            for risk_id in _risk_ids(target)
        },
        "excel_ret": float(target.ret_mean_excel),
        "des_ret": float(ledger["mean_ret_excel"]),
        "excel_ct_p50": _q((order.ctj for order in target.orders), 0.50),
        "des_ct_p50": _q((order.CTj for order in visible), 0.50),
        "excel_ct_p95": _q((order.ctj for order in target.orders), 0.95),
        "des_ct_p95": _q((order.CTj for order in visible), 0.95),
        "excel_rp_p50": _q((order.rpj for order in target.orders if order.rpj > 0), 0.50),
        "des_rp_p50": _q((order.RPj for order in visible if order.RPj > 0), 0.50),
        "excel_rp_p95": _q((order.rpj for order in target.orders if order.rpj > 0), 0.95),
        "des_rp_p95": _q((order.RPj for order in visible if order.RPj > 0), 0.95),
        "release_wait_p50": _q(
            (order.causal_wait_hours.get("op9_release", 0.0) for order in visible),
            0.50,
        ),
        "release_wait_p95": _q(
            (order.causal_wait_hours.get("op9_release", 0.0) for order in visible),
            0.95,
        ),
        "op10_resource_wait_p95": _q(
            (
                order.causal_wait_hours.get("op10_resource_queue", 0.0)
                for order in visible
            ),
            0.95,
        ),
        "op12_resource_wait_p95": _q(
            (
                order.causal_wait_hours.get("op12_resource_queue", 0.0)
                for order in visible
            ),
            0.95,
        ),
        "op10_down_wait_p95": _q(
            (order.causal_wait_hours.get("op10_down", 0.0) for order in visible),
            0.95,
        ),
        "op11_down_wait_p95": _q(
            (order.causal_wait_hours.get("op11_down", 0.0) for order in visible),
            0.95,
        ),
        "op12_down_wait_p95": _q(
            (order.causal_wait_hours.get("op12_down", 0.0) for order in visible),
            0.95,
        ),
        "event_counts": event_counts,
        "event_hours": event_hours,
    }

    risk_rows: list[dict[str, Any]] = []
    for risk_id in _risk_ids(target):
        excel_metrics = _conditional_metrics(
            list(target.orders),
            lambda order, rid=risk_id: _excel_risk_positive(order, rid),
            "excel",
        )
        des_metrics = _conditional_metrics(
            visible,
            lambda order, rid=risk_id: _des_risk_positive(order, rid),
            "des",
        )
        risk_rows.append(
            {
                "cfi": cfi,
                "split": summary["split"],
                "variant": variant,
                "rp_mode": rp_mode,
                "risk_id": risk_id,
                "event_count": event_counts.get(risk_id, 0),
                "event_hours": event_hours.get(risk_id, 0.0),
                **excel_metrics,
                **des_metrics,
            }
        )
    return summary, risk_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfs", nargs="+", type=int, default=list(CALIBRATION_CFS))
    parser.add_argument(
        "--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS)
    )
    parser.add_argument(
        "--rp-modes", nargs="+", choices=("elapsed", "disruption"), default=["elapsed"]
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/audits/garrido_mechanism_audit"),
    )
    args = parser.parse_args()
    if any(cfi in VALIDATION_CFS for cfi in args.cfs) and len(args.variants) != 1:
        raise SystemExit(
            "Even CFs are frozen validation cases. Run them only after freezing "
            "one single candidate variant."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = load_raw_garrido_targets(DEFAULT_RAW_WORKBOOKS)
    summaries: list[dict[str, Any]] = []
    risks: list[dict[str, Any]] = []
    for cfi in args.cfs:
        for variant in args.variants:
            for rp_mode in args.rp_modes:
                summary, risk_rows = _run(cfi, targets[cfi], variant, rp_mode)
                summaries.append(summary)
                risks.extend(risk_rows)
                print(
                    f"CF{cfi} {variant} {rp_mode}: "
                    f"visible={summary['des_visible']}/{summary['excel_visible']} "
                    f"CT95={summary['des_ct_p95']:.1f}/{summary['excel_ct_p95']:.1f} "
                    f"RP95={summary['des_rp_p95']:.1f}/{summary['excel_rp_p95']:.1f}",
                    flush=True,
                )

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    for name, rows in (("summary.csv", summaries), ("risk_attribution.csv", risks)):
        flat_rows = []
        for row in rows:
            flat = dict(row)
            for key in ("event_counts", "event_hours"):
                if key in flat:
                    flat[key] = json.dumps(flat[key], sort_keys=True)
            flat_rows.append(flat)
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)


if __name__ == "__main__":
    main()
