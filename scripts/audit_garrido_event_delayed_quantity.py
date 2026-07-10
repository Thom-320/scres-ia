#!/usr/bin/env python3
"""Paired leave-one-event-out audit of counterfactually delayed material.

For each frozen duration event, replay the identical event calendar twice:
once complete and once with exactly that event removed.  Differences between
cumulative node-arrival curves identify how many units became available later
because of that event and when the availability debt was recovered.

This is an audit lane.  It does not alter ReT attribution or DES physics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import (  # noqa: E402
    THESIS_FAITHFUL_PROTOCOL,
    THESIS_REPLICATION_DOWNSTREAM_Q_SOURCE,
)
from supply_chain.garrido_replication import (  # noqa: E402
    DEFAULT_RAW_WORKBOOKS,
    load_raw_garrido_targets,
)
from supply_chain.supply_chain import MFSCSimulation, RiskEvent  # noqa: E402
from supply_chain.thesis_design import design_spec_for_cfi  # noqa: E402


NODES = (
    "raw_material_wdc",
    "raw_material_al",
    "rations_al",
    "rations_sb",
    "order_release",
)


def _event_dict(event: RiskEvent) -> dict[str, Any]:
    return {
        "risk_id": event.risk_id,
        "start_time": float(event.start_time),
        "end_time": float(event.end_time),
        "duration": float(event.duration),
        "affected_ops": [int(op) for op in event.affected_ops],
        "description": event.description,
        "magnitude": float(event.magnitude),
        "unit": event.unit,
    }


def delayed_quantity_metrics(
    factual: Iterable[tuple[float, float]],
    no_event: Iterable[tuple[float, float]],
    *,
    horizon: float,
    tolerance: float = 1e-8,
) -> dict[str, float | None]:
    """Compare cumulative availability; positive no-event minus factual is debt."""
    factual_rows = sorted((float(t), float(q)) for t, q in factual)
    no_event_rows = sorted((float(t), float(q)) for t, q in no_event)
    times = sorted(
        {0.0, float(horizon)}
        | {t for t, _ in factual_rows if 0.0 <= t <= horizon}
        | {t for t, _ in no_event_rows if 0.0 <= t <= horizon}
    )
    factual_by_time: dict[float, float] = {}
    no_event_by_time: dict[float, float] = {}
    for time, qty in factual_rows:
        if time <= horizon:
            factual_by_time[time] = factual_by_time.get(time, 0.0) + qty
    for time, qty in no_event_rows:
        if time <= horizon:
            no_event_by_time[time] = no_event_by_time.get(time, 0.0) + qty

    factual_cum = 0.0
    no_event_cum = 0.0
    curve: list[tuple[float, float]] = []
    for time in times:
        factual_cum += factual_by_time.get(time, 0.0)
        no_event_cum += no_event_by_time.get(time, 0.0)
        curve.append((time, no_event_cum - factual_cum))

    positive = [(time, debt) for time, debt in curve if debt > tolerance]
    peak = max((debt for _, debt in positive), default=0.0)
    onset = positive[0][0] if positive else None
    peak_index = max(range(len(curve)), key=lambda i: curve[i][1]) if curve else 0
    first_recovered_at = None
    if peak > tolerance:
        for time, debt in curve[peak_index + 1 :]:
            if debt <= tolerance:
                first_recovered_at = time
                break
    last_positive_index = max(
        (index for index, (_, debt) in enumerate(curve) if debt > tolerance),
        default=-1,
    )
    recovered_at = (
        curve[last_positive_index + 1][0]
        if 0 <= last_positive_index < len(curve) - 1
        else None
    )

    area = 0.0
    for (time, debt), (next_time, _) in zip(curve, curve[1:]):
        area += max(0.0, debt) * max(0.0, next_time - time)
    terminal_debt = max(0.0, curve[-1][1]) if curve else 0.0
    return {
        "delayed_quantity_peak": float(peak),
        "delay_onset": onset,
        "first_temporary_recovery_at": first_recovered_at,
        "debt_recovered_at": recovered_at,
        "delayed_unit_hours": float(area),
        "terminal_delayed_quantity": float(terminal_debt),
        "factual_available_total": float(factual_cum),
        "no_event_available_total": float(no_event_cum),
    }


def match_order_release_counterfactual(
    factual: MFSCSimulation,
    no_event: MFSCSimulation,
    *,
    event_ref: str,
    horizon: float,
    tolerance: float = 1e-8,
) -> list[dict[str, Any]]:
    """Match the same demand order across paired runs; keep only earlier opportunities.

    The DES consumes SB inventory FIFO, while its dispatch queue selects the
    concrete order according to the configured release rule. Matching by ``j``
    therefore identifies the actual consumer rather than assigning a temporal
    exposure window to neighboring orders.
    """
    factual_by_j = {int(order.j): order for order in factual.orders}
    no_event_by_j = {int(order.j): order for order in no_event.orders}
    rows: list[dict[str, Any]] = []
    for j in sorted(set(factual_by_j) & set(no_event_by_j)):
        factual_order = factual_by_j[j]
        no_event_order = no_event_by_j[j]
        factual_release = factual_order.op9_release_time
        no_event_release = no_event_order.op9_release_time
        if no_event_release is None:
            continue
        no_event_time = float(no_event_release)
        if factual_release is None:
            delay = max(0.0, float(horizon) - no_event_time)
            if delay <= tolerance:
                continue
            status = "censored_not_released_factual"
            factual_time: float | None = None
        else:
            factual_time = float(factual_release)
            delay = factual_time - no_event_time
            if delay <= tolerance:
                continue
            status = "released_later_factual"
        quantity = float(no_event_order.quantity)
        rows.append(
            {
                "event_ref": event_ref,
                "j": j,
                "quantity": quantity,
                "no_event_release_time": no_event_time,
                "factual_release_time": factual_time,
                "release_delay_hours": float(delay),
                "delayed_unit_hours": quantity * float(delay),
                "status": status,
            }
        )
    return rows


def match_fifo_release_opportunities(
    factual: MFSCSimulation,
    no_event: MFSCSimulation,
    *,
    event_ref: str,
    tolerance: float = 1e-8,
) -> list[dict[str, Any]]:
    """Allocate only positive increments of cumulative release debt to orders.

    Factual releases at a timestamp are credited before counterfactual releases
    at that same timestamp.  A counterfactual order receives at most the slice
    by which its release increases ``cum(no_event) - cum(factual)``. Thus queue
    reshuffling is not mislabeled as additional missing material.
    """
    factual_releases = sorted(
        (
            float(order.op9_release_time),
            int(order.j),
            float(order.quantity),
        )
        for order in factual.orders
        if order.op9_release_time is not None
    )
    no_event_releases = sorted(
        (
            float(order.op9_release_time),
            int(order.j),
            float(order.quantity),
        )
        for order in no_event.orders
        if order.op9_release_time is not None
    )
    factual_release_by_j = {j: time for time, j, _ in factual_releases}
    factual_index = 0
    factual_cumulative = 0.0
    no_event_cumulative = 0.0
    rows: list[dict[str, Any]] = []
    for release_time, j, quantity in no_event_releases:
        while (
            factual_index < len(factual_releases)
            and factual_releases[factual_index][0] <= release_time + tolerance
        ):
            factual_cumulative += factual_releases[factual_index][2]
            factual_index += 1
        debt_before = max(0.0, no_event_cumulative - factual_cumulative)
        no_event_cumulative += quantity
        debt_after = max(0.0, no_event_cumulative - factual_cumulative)
        direct_quantity = min(quantity, max(0.0, debt_after - debt_before))
        if direct_quantity <= tolerance:
            continue
        factual_time = factual_release_by_j.get(j)
        same_order_delay = (
            max(0.0, factual_time - release_time)
            if factual_time is not None
            else None
        )
        rows.append(
            {
                "event_ref": event_ref,
                "j": j,
                "counterfactual_release_time": release_time,
                "direct_fifo_quantity": float(direct_quantity),
                "release_debt_before": float(debt_before),
                "release_debt_after": float(debt_after),
                "factual_same_order_release_time": factual_time,
                "same_order_release_delay_hours": same_order_delay,
            }
        )
    return rows


def _sim_kwargs(
    cfi: int,
    seed: int,
    horizon: float,
    *,
    op2_release_clock_mode: str = "inherit",
) -> dict[str, Any]:
    spec = design_spec_for_cfi(cfi)
    return {
        "shifts": spec.shifts,
        "seed": int(seed),
        "horizon": float(horizon),
        "risks_enabled": True,
        "risk_level": "current",
        "enabled_risks": set(spec.enabled_risks),
        "risk_overrides": dict(spec.risk_overrides),
        "risk_occurrence_mode": "thesis_window",
        "risk_attribution_source": "des_events",
        "strict_exogenous_crn": True,
        "year_basis": THESIS_FAITHFUL_PROTOCOL["year_basis"],
        "warmup_trigger": THESIS_FAITHFUL_PROTOCOL["warmup_trigger"],
        "downstream_q_source": THESIS_REPLICATION_DOWNSTREAM_Q_SOURCE,
        "r14_defect_mode": THESIS_FAITHFUL_PROTOCOL["r14_defect_mode"],
        "raw_material_flow_mode": THESIS_FAITHFUL_PROTOCOL["raw_material_flow_mode"],
        "raw_material_order_up_to_multiplier": 1.0,
        "replenishment_route_aware": True,
        "procurement_contract_mode": "causal_coupled",
        "order_fulfillment_mode": "op9_linked",
        "assembly_flow_mode": "aggregate_line",
        "periodic_release_mode": "start_to_start",
        "op2_release_clock_mode": op2_release_clock_mode,
        "operational_risk_initialization_mode": "include_initial_cycle",
        "risk_rng_mode": "per_risk",
        "op9_dispatch_policy": "fixed_clock_daily",
        "downstream_transport_capacity_mode": "parallel",
        "demand_start_after_warmup": True,
        "ret_recovery_period_mode": "elapsed",
    }


def run_audit(
    *,
    cfi: int,
    horizon: float,
    max_events: int,
    risk_id: str | None = None,
    op2_release_clock_mode: str = "inherit",
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    target = load_raw_garrido_targets(DEFAULT_RAW_WORKBOOKS)[cfi]
    kwargs = _sim_kwargs(
        cfi,
        int(target.seed),
        horizon,
        op2_release_clock_mode=op2_release_clock_mode,
    )

    generator = MFSCSimulation(**kwargs).run()
    tape = [
        _event_dict(event)
        for event in generator.risk_events
        if float(event.duration) > 0.0
        and any(int(op) <= 8 for op in event.affected_ops)
    ]
    if not tape:
        raise RuntimeError("No replayable upstream duration events found.")
    audited_indices = [
        index
        for index, event in enumerate(tape)
        if risk_id is None or event["risk_id"] == risk_id
    ][:max_events]
    if not audited_indices:
        raise RuntimeError("No events match the requested risk-id filter.")

    factual = MFSCSimulation(**kwargs, risk_event_tape=tape).run()
    rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    fifo_rows: list[dict[str, Any]] = []
    for event_index in audited_indices:
        event = tape[event_index]
        no_event_tape = tape[:event_index] + tape[event_index + 1 :]
        no_event = MFSCSimulation(**kwargs, risk_event_tape=no_event_tape).run()
        event_ref = f"{event['risk_id']}@{float(event['start_time']):.9f}"
        matched_orders = match_order_release_counterfactual(
            factual,
            no_event,
            event_ref=event_ref,
            horizon=horizon,
        )
        for order_row in matched_orders:
            order_row.update(
                {
                    "cfi": cfi,
                    "event_index": event_index,
                    "risk_id": event["risk_id"],
                    "event_start": event["start_time"],
                }
            )
        order_rows.extend(matched_orders)
        direct_fifo = match_fifo_release_opportunities(
            factual, no_event, event_ref=event_ref
        )
        for fifo_row in direct_fifo:
            fifo_row.update(
                {
                    "cfi": cfi,
                    "event_index": event_index,
                    "risk_id": event["risk_id"],
                    "event_start": event["start_time"],
                }
            )
        fifo_rows.extend(direct_fifo)
        for node in NODES:
            metrics = delayed_quantity_metrics(
                factual.material_availability_events[node],
                no_event.material_availability_events[node],
                horizon=horizon,
            )
            rows.append(
                {
                    "cfi": cfi,
                    "event_index": event_index,
                    "event_ref": event_ref,
                    "risk_id": event["risk_id"],
                    "event_start": event["start_time"],
                    "event_duration": event["duration"],
                    "affected_ops": ";".join(map(str, event["affected_ops"])),
                    "node": node,
                    **metrics,
                }
            )
    return tape, rows, order_rows, fifo_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cf", type=int, default=1)
    parser.add_argument("--horizon", type=float, default=12_000.0)
    parser.add_argument("--max-events", type=int, default=5)
    parser.add_argument("--risk-id")
    parser.add_argument(
        "--op2-release-clock-mode",
        choices=("inherit", "calendar_anchored"),
        default="inherit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/audits/garrido_event_delayed_quantity"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tape, rows, order_rows, fifo_rows = run_audit(
        cfi=args.cf,
        horizon=args.horizon,
        max_events=args.max_events,
        risk_id=args.risk_id,
        op2_release_clock_mode=args.op2_release_clock_mode,
    )
    (args.output_dir / "event_tape.json").write_text(
        json.dumps(tape, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "delayed_quantity_by_event_node.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if order_rows:
        with (args.output_dir / "counterfactual_order_delays.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(order_rows[0]))
            writer.writeheader()
            writer.writerows(order_rows)
    if fifo_rows:
        with (args.output_dir / "fifo_release_opportunities.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fifo_rows[0]))
            writer.writeheader()
            writer.writerows(fifo_rows)
    event_summaries = []
    for event_ref in sorted({row["event_ref"] for row in rows}):
        event_orders = [row for row in order_rows if row["event_ref"] == event_ref]
        event_fifo = [row for row in fifo_rows if row["event_ref"] == event_ref]
        release_node = next(
            row for row in rows if row["event_ref"] == event_ref and row["node"] == "order_release"
        )
        event_summaries.append(
            {
                "event_ref": event_ref,
                "matched_orders": len(event_orders),
                "matched_order_quantity": float(sum(row["quantity"] for row in event_orders)),
                "matched_order_unit_hours": float(
                    sum(row["delayed_unit_hours"] for row in event_orders)
                ),
                "direct_fifo_opportunities": len(event_fifo),
                "direct_fifo_quantity": float(
                    sum(row["direct_fifo_quantity"] for row in event_fifo)
                ),
                "max_concurrent_release_quantity_debt": float(
                    release_node["delayed_quantity_peak"]
                ),
                "release_debt_recovered_at": release_node["debt_recovered_at"],
            }
        )
    target = load_raw_garrido_targets(DEFAULT_RAW_WORKBOOKS)[args.cf]
    factual_released = {
        int(order.j)
        for order in MFSCSimulation(
            **_sim_kwargs(
                args.cf,
                int(target.seed),
                args.horizon,
                op2_release_clock_mode=args.op2_release_clock_mode,
            ),
            risk_event_tape=tape,
        ).run().orders
        if order.op9_release_time is not None
    }
    risk_summaries = []
    for audited_risk in sorted({row["risk_id"] for row in rows}):
        calendar_count = sum(event["risk_id"] == audited_risk for event in tape)
        audited_refs = {
            row["event_ref"] for row in rows if row["risk_id"] == audited_risk
        }
        fifo_orders = {
            int(row["j"]) for row in fifo_rows if row["risk_id"] == audited_risk
        }
        identity_orders = {
            int(row["j"]) for row in order_rows if row["risk_id"] == audited_risk
        }
        excel_orders = [
            order for order in target.orders if float(order.optj) <= args.horizon
        ]
        excel_marked = [
            order
            for order in excel_orders
            if any(
                str(label).startswith(audited_risk) and float(value) > 0.0
                for label, value in order.risk_values.items()
            )
        ]
        risk_summaries.append(
            {
                "risk_id": audited_risk,
                "calendar_events": calendar_count,
                "audited_events": len(audited_refs),
                "event_coverage": len(audited_refs) / max(calendar_count, 1),
                "factual_released_orders": len(factual_released),
                "unique_direct_fifo_orders": len(fifo_orders),
                "direct_fifo_order_share": len(fifo_orders)
                / max(len(factual_released), 1),
                "unique_identity_delayed_orders": len(identity_orders),
                "identity_delay_order_share": len(identity_orders)
                / max(len(factual_released), 1),
                "excel_orders_within_horizon": len(excel_orders),
                "excel_marked_orders_within_horizon": len(excel_marked),
                "excel_mark_share_within_horizon": len(excel_marked)
                / max(len(excel_orders), 1),
                "promotion_eligible": len(audited_refs) == calendar_count,
            }
        )
    verdict = {
        "estimand": "availability_without_event_minus_availability_with_event",
        "cfi": args.cf,
        "horizon": args.horizon,
        "op2_release_clock_mode": args.op2_release_clock_mode,
        "calendar_events": len(tape),
        "events_audited": len({row["event_ref"] for row in rows}),
        "rows": rows,
        "event_order_summaries": event_summaries,
        "risk_summaries": risk_summaries,
        "claim_boundary": (
            "Paired in-model counterfactual under a frozen replay calendar; "
            "not evidence of real-world causal effect."
        ),
    }
    (args.output_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )
    for row in rows:
        if float(row["delayed_quantity_peak"]) > 0.0:
            print(
                f"{row['event_ref']} {row['node']}: "
                f"peak={row['delayed_quantity_peak']:.1f}, "
                f"recovered={row['debt_recovered_at']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
