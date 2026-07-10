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


def _sim_kwargs(cfi: int, seed: int, horizon: float) -> dict[str, Any]:
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
        "operational_risk_initialization_mode": "include_initial_cycle",
        "risk_rng_mode": "per_risk",
        "op9_dispatch_policy": "fixed_clock_daily",
        "downstream_transport_capacity_mode": "parallel",
        "demand_start_after_warmup": True,
        "ret_recovery_period_mode": "elapsed",
    }


def run_audit(
    *, cfi: int, horizon: float, max_events: int, risk_id: str | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target = load_raw_garrido_targets(DEFAULT_RAW_WORKBOOKS)[cfi]
    kwargs = _sim_kwargs(cfi, int(target.seed), horizon)

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
    for event_index in audited_indices:
        event = tape[event_index]
        no_event_tape = tape[:event_index] + tape[event_index + 1 :]
        no_event = MFSCSimulation(**kwargs, risk_event_tape=no_event_tape).run()
        event_ref = f"{event['risk_id']}@{float(event['start_time']):.9f}"
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
    return tape, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cf", type=int, default=1)
    parser.add_argument("--horizon", type=float, default=12_000.0)
    parser.add_argument("--max-events", type=int, default=5)
    parser.add_argument("--risk-id")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/audits/garrido_event_delayed_quantity"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tape, rows = run_audit(
        cfi=args.cf,
        horizon=args.horizon,
        max_events=args.max_events,
        risk_id=args.risk_id,
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
    verdict = {
        "estimand": "availability_without_event_minus_availability_with_event",
        "cfi": args.cf,
        "horizon": args.horizon,
        "calendar_events": len(tape),
        "events_audited": len({row["event_ref"] for row in rows}),
        "rows": rows,
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
