#!/usr/bin/env python3
"""Run static DES experiments on the validated Garrido Excel replication lane.

Default contract:
- primary metric: mean_ret_excel_formula
- demand: workbook Q/OPTj tape
- risk attribution: workbook-visible R... + APj/RPj/DPj tape
- target set: Raw_data1+Re.xlsx and Raw_data2+Re.xlsx, CF1-CF20

This runner is intentionally static-only. Use it to test whether simple DES
levers create a non-flat Excel-formula resilience surface before spending time
on RL.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import (  # noqa: E402
    INVENTORY_BUFFERS,
    THESIS_FAITHFUL_PROTOCOL as P,
)
from supply_chain.garrido_replication import (  # noqa: E402
    DEFAULT_RAW_WORKBOOKS,
    GarridoCFTarget,
    load_raw_garrido_targets,
)
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402
from supply_chain.thesis_design import (  # noqa: E402
    ThesisDesignSpec,
    design_spec_for_cfi,
    parse_cf_range,
)


ROW_FIELDS = [
    "policy",
    "policy_kind",
    "cfi",
    "source_cfi",
    "family",
    "seed",
    "shifts",
    "inventory_period",
    "initial_buffer_profile",
    "raw_material_flow_mode",
    "raw_material_order_up_to_multiplier",
    "target_ret_excel_mean",
    "mean_ret_excel_formula",
    "mean_ret_text_formula",
    "ret_gap_vs_target",
    "abs_ret_gap_vs_target",
    "fill_rate_order_level",
    "n_orders",
    "n_completed",
    "target_n_orders",
    "case_excel_fill_rate",
    "case_excel_autotomy",
    "case_excel_recovery",
    "case_excel_risk_no_recovery",
    "total_demanded",
    "total_delivered",
    "total_backorders",
    "total_unattended_orders",
]

SUMMARY_FIELDS = [
    "policy",
    "policy_kind",
    "family",
    "n",
    "target_ret_excel_mean",
    "mean_ret_excel_formula",
    "delta_ret_vs_matched",
    "mean_abs_ret_gap_vs_target",
    "max_abs_ret_gap_vs_target",
    "fill_rate_order_level",
    "n_orders_mean",
    "n_completed_mean",
]


@dataclass(frozen=True)
class StaticPolicy:
    name: str
    kind: str
    shifts: int | None = None
    inventory_period: int | None | str = "matched"
    raw_material_flow_mode: str | None = None
    raw_material_order_up_to_multiplier: float | None = None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.fmean(vals)) if vals else math.nan


def _inventory_buffers_for(period: int | None | str) -> dict[str, float] | None:
    if period == "matched":
        raise ValueError("matched inventory must be resolved from the CF design spec.")
    if period is None:
        return None
    return {key: float(value) for key, value in INVENTORY_BUFFERS[int(period)].items()}


def policies_for_set(policy_set: str) -> list[StaticPolicy]:
    policies = [
        StaticPolicy("matched_thesis", "matched"),
        StaticPolicy("shift_S1", "shift", shifts=1),
        StaticPolicy("shift_S2", "shift", shifts=2),
        StaticPolicy("shift_S3", "shift", shifts=3),
        StaticPolicy("inventory_I0", "inventory", inventory_period=None),
        StaticPolicy("inventory_I672", "inventory", inventory_period=672),
        StaticPolicy(
            "raw_legacy_x1",
            "raw_material",
            raw_material_flow_mode="legacy_validated",
            raw_material_order_up_to_multiplier=1.0,
        ),
        StaticPolicy(
            "raw_bom_order_up_to_x1",
            "raw_material",
            raw_material_flow_mode="bom_total_units_order_up_to",
            raw_material_order_up_to_multiplier=1.0,
        ),
    ]
    if policy_set == "minimal":
        return policies
    if policy_set != "expanded":
        raise ValueError("policy_set must be 'minimal' or 'expanded'.")

    existing = {policy.name for policy in policies}
    for period in (168, 336, 504, 1344):
        name = f"inventory_I{period}"
        if name not in existing:
            policies.append(
                StaticPolicy(name, "inventory", inventory_period=period)
            )
    for multiplier in (1.5, 2.0):
        policies.append(
            StaticPolicy(
                f"raw_bom_order_up_to_x{multiplier:g}",
                "raw_material",
                raw_material_flow_mode="bom_total_units_order_up_to",
                raw_material_order_up_to_multiplier=float(multiplier),
            )
        )
    return policies


def resolve_policy(
    policy: StaticPolicy,
    *,
    spec: ThesisDesignSpec,
    default_raw_material_flow_mode: str,
    default_raw_material_order_up_to_multiplier: float,
) -> dict[str, Any]:
    if policy.inventory_period == "matched":
        inventory_period = spec.inventory_replenishment_period
        initial_buffers = spec.initial_buffers
        initial_buffer_profile = "matched"
    else:
        inventory_period = (
            None if policy.inventory_period is None else int(policy.inventory_period)
        )
        initial_buffers = _inventory_buffers_for(inventory_period)
        initial_buffer_profile = "I0" if inventory_period is None else f"I{inventory_period}"

    return {
        "shifts": int(policy.shifts if policy.shifts is not None else spec.shifts),
        "inventory_replenishment_period": inventory_period,
        "initial_buffers": initial_buffers,
        "initial_buffer_profile": initial_buffer_profile,
        "raw_material_flow_mode": (
            policy.raw_material_flow_mode or default_raw_material_flow_mode
        ),
        "raw_material_order_up_to_multiplier": float(
            policy.raw_material_order_up_to_multiplier
            if policy.raw_material_order_up_to_multiplier is not None
            else default_raw_material_order_up_to_multiplier
        ),
    }


def run_one(
    *,
    target: GarridoCFTarget,
    policy: StaticPolicy,
    args: argparse.Namespace,
) -> dict[str, Any]:
    spec = design_spec_for_cfi(target.cfi)
    resolved = resolve_policy(
        policy,
        spec=spec,
        default_raw_material_flow_mode=args.raw_material_flow_mode,
        default_raw_material_order_up_to_multiplier=args.raw_material_order_up_to_multiplier,
    )
    sim = MFSCSimulation(
        shifts=resolved["shifts"],
        initial_buffers=resolved["initial_buffers"],
        seed=int(target.seed) if args.use_workbook_seed else int(args.seed_base + target.cfi),
        horizon=float(target.horizon_hours),
        risks_enabled=True,
        risk_level=args.risk_level,
        year_basis=args.year_basis,
        stochastic_pt=args.stochastic_pt,
        warmup_trigger=args.warmup_trigger,
        downstream_q_source=args.downstream_q_source,
        r14_defect_mode=args.r14_defect_mode,
        enabled_risks=set(spec.enabled_risks),
        risk_overrides=dict(spec.risk_overrides),
        risk_occurrence_mode=args.risk_occurrence_mode,
        risk_attribution_source=args.risk_attribution_source,
        inventory_replenishment_period=resolved["inventory_replenishment_period"],
        raw_material_flow_mode=resolved["raw_material_flow_mode"],
        raw_material_order_up_to_multiplier=resolved[
            "raw_material_order_up_to_multiplier"
        ],
        demand_source=args.demand_source,
        excel_order_tape=target.order_tape(),
        seed_stream_mode=args.seed_stream_mode,
    ).run()
    ret = sim.compute_order_level_ret()
    case_counts = dict(ret["case_counts_excel_formula"])
    mean_ret = float(ret["mean_ret_excel_formula"])
    target_ret = float(target.ret_mean_excel)
    return {
        "policy": policy.name,
        "policy_kind": policy.kind,
        "cfi": int(target.cfi),
        "source_cfi": int(spec.source_cfi),
        "family": spec.family,
        "seed": int(target.seed),
        "shifts": int(resolved["shifts"]),
        "inventory_period": (
            "" if resolved["inventory_replenishment_period"] is None else int(resolved["inventory_replenishment_period"])
        ),
        "initial_buffer_profile": resolved["initial_buffer_profile"],
        "raw_material_flow_mode": resolved["raw_material_flow_mode"],
        "raw_material_order_up_to_multiplier": float(
            resolved["raw_material_order_up_to_multiplier"]
        ),
        "target_ret_excel_mean": target_ret,
        "mean_ret_excel_formula": mean_ret,
        "mean_ret_text_formula": float(ret["mean_ret_text_formula"]),
        "ret_gap_vs_target": mean_ret - target_ret,
        "abs_ret_gap_vs_target": abs(mean_ret - target_ret),
        "fill_rate_order_level": float(ret["fill_rate_order_level"]),
        "n_orders": int(ret["n_orders"]),
        "n_completed": int(ret["n_completed"]),
        "target_n_orders": int(target.n_orders),
        "case_excel_fill_rate": int(case_counts.get("excel_fill_rate", 0)),
        "case_excel_autotomy": int(case_counts.get("excel_autotomy", 0)),
        "case_excel_recovery": int(case_counts.get("excel_recovery", 0)),
        "case_excel_risk_no_recovery": int(case_counts.get("excel_risk_no_recovery", 0)),
        "total_demanded": float(sim.total_demanded),
        "total_delivered": float(sim.total_delivered),
        "total_backorders": int(sim.total_backorders),
        "total_unattended_orders": int(sim.total_unattended_orders),
    }


def summarize(rows: list[dict[str, Any]], *, group_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    matched_overall = _mean(
        row["mean_ret_excel_formula"]
        for row in rows
        if row["policy"] == "matched_thesis"
    )
    matched_by_family = {
        family: _mean(
            row["mean_ret_excel_formula"]
            for row in rows
            if row["policy"] == "matched_thesis" and row["family"] == family
        )
        for family in sorted({str(row["family"]) for row in rows})
    }
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for key, bucket in sorted(grouped.items()):
        first = bucket[0]
        family = str(first.get("family", "all"))
        matched = matched_by_family.get(family) if "family" in group_fields else matched_overall
        mean_ret = _mean(row["mean_ret_excel_formula"] for row in bucket)
        row_out = {
            "policy": first["policy"],
            "policy_kind": first["policy_kind"],
            "family": first.get("family", "all") if "family" in group_fields else "all",
            "n": len(bucket),
            "target_ret_excel_mean": _mean(row["target_ret_excel_mean"] for row in bucket),
            "mean_ret_excel_formula": mean_ret,
            "delta_ret_vs_matched": (
                mean_ret - matched if matched is not None else math.nan
            ),
            "mean_abs_ret_gap_vs_target": _mean(
                row["abs_ret_gap_vs_target"] for row in bucket
            ),
            "max_abs_ret_gap_vs_target": max(
                float(row["abs_ret_gap_vs_target"]) for row in bucket
            ),
            "fill_rate_order_level": _mean(row["fill_rate_order_level"] for row in bucket),
            "n_orders_mean": _mean(row["n_orders"] for row in bucket),
            "n_completed_mean": _mean(row["n_completed"] for row in bucket),
        }
        out.append(row_out)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbooks", nargs="*", type=Path, default=list(DEFAULT_RAW_WORKBOOKS))
    parser.add_argument("--cf-range", default="1-20")
    parser.add_argument("--policy-set", choices=["minimal", "expanded"], default="minimal")
    parser.add_argument("--demand-source", default="excel_order_tape")
    parser.add_argument("--risk-attribution-source", default="excel_risk_tape")
    parser.add_argument("--risk-occurrence-mode", default="thesis_window")
    parser.add_argument("--seed-stream-mode", default="split")
    parser.add_argument("--risk-level", default="current")
    parser.add_argument("--year-basis", default=P["year_basis"])
    parser.add_argument("--warmup-trigger", default=P["warmup_trigger"])
    parser.add_argument("--downstream-q-source", default=P["downstream_q_source"])
    parser.add_argument("--r14-defect-mode", default=P["r14_defect_mode"])
    parser.add_argument("--raw-material-flow-mode", default=P["raw_material_flow_mode"])
    parser.add_argument(
        "--raw-material-order-up-to-multiplier",
        type=float,
        default=float(P["raw_material_order_up_to_multiplier"]),
    )
    parser.add_argument("--stochastic-pt", action="store_true")
    parser.add_argument("--use-workbook-seed", action="store_true", default=True)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/garrido_excel_static"),
    )
    args = parser.parse_args()

    targets_all = load_raw_garrido_targets(args.workbooks)
    cfi_values = parse_cf_range(args.cf_range)
    targets = {cfi: targets_all[cfi] for cfi in cfi_values}
    policies = policies_for_set(args.policy_set)

    rows: list[dict[str, Any]] = []
    for target in targets.values():
        for policy in policies:
            rows.append(run_one(target=target, policy=policy, args=args))

    by_policy = summarize(rows, group_fields=("policy",))
    by_family_policy = summarize(rows, group_fields=("family", "policy"))
    payload = {
        "description": "Static experiments on the Garrido Excel replication lane.",
        "primary_metric": "mean_ret_excel_formula",
        "config": _json_safe(vars(args)),
        "n_rows": len(rows),
        "n_cfis": len(targets),
        "policies": [policy.__dict__ for policy in policies],
        "best_policy_overall": max(
            by_policy, key=lambda row: float(row["mean_ret_excel_formula"])
        ),
        "summary_by_policy": by_policy,
        "summary_by_family_policy": by_family_policy,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "rows.csv"
    by_policy_path = args.output_dir / "summary_by_policy.csv"
    by_family_path = args.output_dir / "summary_by_family_policy.csv"
    json_path = args.output_dir / "summary.json"
    write_csv(rows_path, rows, ROW_FIELDS)
    write_csv(by_policy_path, by_policy, SUMMARY_FIELDS)
    write_csv(by_family_path, by_family_policy, SUMMARY_FIELDS)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "primary_metric": payload["primary_metric"],
                "n_rows": payload["n_rows"],
                "n_cfis": payload["n_cfis"],
                "best_policy_overall": payload["best_policy_overall"],
                "outputs": {
                    "rows": str(rows_path),
                    "summary_by_policy": str(by_policy_path),
                    "summary_by_family_policy": str(by_family_path),
                    "summary": str(json_path),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
