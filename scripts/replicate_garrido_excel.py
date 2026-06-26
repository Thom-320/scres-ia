#!/usr/bin/env python3
"""Run the Garrido Excel replication harness.

The harness treats the raw workbooks as the operational target: demand/order
tapes can reproduce Q and OPTj exactly, while the DES still computes OATj, CTj,
and ReT.  In the forensic `excel_risk_tape` mode, the DES replays the workbook
visible risk gate and APj/RPj/DPj periods without copying OATj, CTj, or ReT.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.garrido_replication import (  # noqa: E402
    DEFAULT_RAW_WORKBOOKS,
    DEFAULT_RSULT_WORKBOOK,
    EXCEL_RET_FORMULA,
    GarridoCFTarget,
    audit_raw_garrido_formula,
    load_raw_garrido_targets,
    summarize_rsult_workbook,
    target_to_summary,
)
from supply_chain.ret_thesis import (  # noqa: E402
    compute_ret_per_order_excel_formula,
    order_counts_as_backorder_for_fill_rate,
)
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402
from supply_chain.thesis_design import design_spec_for_cfi, parse_cf_range  # noqa: E402


CASE_KEYS = (
    "excel_fill_rate",
    "excel_autotomy",
    "excel_recovery",
    "excel_risk_no_recovery",
)


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def _case_shares(counts: dict[str, int]) -> dict[str, float]:
    total = sum(int(counts.get(key, 0)) for key in CASE_KEYS)
    return {key: _pct(int(counts.get(key, 0)), total) for key in CASE_KEYS}


def _max_branch_gap(
    target_counts: dict[str, int], sim_counts: dict[str, int]
) -> float:
    target_shares = _case_shares(target_counts)
    sim_shares = _case_shares(sim_counts)
    return max(abs(sim_shares[key] - target_shares[key]) for key in CASE_KEYS)


def _risk_export_value(order: Any, column: str) -> float:
    indicators = getattr(order, "ret_risk_indicators", {}) or {}
    if column in indicators:
        return float(indicators[column] or 0.0)
    base = column.split("_", maxsplit=1)[0]
    return float(indicators.get(base, 0.0) or 0.0)


def build_sim(
    *,
    target: GarridoCFTarget,
    demand_source: str,
    risk_occurrence_mode: str,
    risk_attribution_source: str,
    seed_stream_mode: str,
    args: argparse.Namespace,
) -> MFSCSimulation:
    spec = design_spec_for_cfi(target.cfi)
    order_tape = target.order_tape() if demand_source == "excel_order_tape" else None
    return MFSCSimulation(
        shifts=spec.shifts,
        initial_buffers=spec.initial_buffers,
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
        risk_occurrence_mode=risk_occurrence_mode,
        risk_attribution_source=risk_attribution_source,
        inventory_replenishment_period=spec.inventory_replenishment_period,
        raw_material_flow_mode=args.raw_material_flow_mode,
        raw_material_order_up_to_multiplier=args.raw_material_order_up_to_multiplier,
        demand_source=demand_source,
        excel_order_tape=order_tape,
        seed_stream_mode=seed_stream_mode,
    )


def order_level_export_rows(
    *,
    sim: MFSCSimulation,
    target: GarridoCFTarget,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cumulative_backorders = 0
    cumulative_unattended = 0
    previous_ret: float | None = None
    orders = sorted(
        sim.orders,
        key=lambda order: (
            int(getattr(order, "j", 0) or 0),
            float(getattr(order, "OPTj", 0.0) or 0.0),
        ),
    )
    for idx, order in enumerate(orders, start=1):
        if bool(getattr(order, "lost", False)):
            cumulative_unattended += 1
        elif order_counts_as_backorder_for_fill_rate(order, current_time=float(sim.env.now)):
            cumulative_backorders += 1

        ret, case = compute_ret_per_order_excel_formula(
            order,
            j=idx,
            cumulative_backorders=cumulative_backorders,
            cumulative_unattended=cumulative_unattended,
        )
        delta_ret = 0.0 if previous_ret is None else ret - previous_ret
        previous_ret = ret
        row: dict[str, Any] = {
            "Q": float(getattr(order, "quantity", 0.0) or 0.0),
            "j": int(getattr(order, "j", idx) or idx),
            "OPTj": float(getattr(order, "OPTj", 0.0) or 0.0),
            "OATj": getattr(order, "OATj", None),
            "CTj": getattr(order, "CTj", None),
            "LT": float(getattr(order, "LTj", 0.0) or 0.0),
            "sumBt": int(cumulative_backorders),
            "APj": float(getattr(order, "APj", 0.0) or 0.0),
            "RPj": float(getattr(order, "RPj", 0.0) or 0.0),
            "DPj": float(getattr(order, "DPj", 0.0) or 0.0),
        }
        for column in target.risk_columns:
            row[column] = _risk_export_value(order, column)
        row.update(
            {
                "sumUt": int(cumulative_unattended),
                "OP9": "",
                "ReT": float(ret),
                "deltaReT": float(delta_ret),
                "excel_case": case,
            }
        )
        rows.append(row)
    return rows


def _max_abs_pair_gap(
    target_values: list[float], sim_values: list[float]
) -> float | None:
    if len(target_values) != len(sim_values):
        return None
    if not target_values:
        return 0.0
    return max(abs(a - b) for a, b in zip(target_values, sim_values, strict=True))


def compare_one(
    *,
    target: GarridoCFTarget,
    demand_source: str,
    risk_occurrence_mode: str,
    risk_attribution_source: str,
    seed_stream_mode: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], MFSCSimulation]:
    sim = build_sim(
        target=target,
        demand_source=demand_source,
        risk_occurrence_mode=risk_occurrence_mode,
        risk_attribution_source=risk_attribution_source,
        seed_stream_mode=seed_stream_mode,
        args=args,
    )
    sim.run()
    ret = sim.compute_order_level_ret()
    sim_orders = sorted(sim.orders, key=lambda order: int(getattr(order, "j", 0) or 0))
    target_orders = sorted(target.orders, key=lambda order: int(order.j))
    q_gap = _max_abs_pair_gap(
        [order.q for order in target_orders],
        [float(getattr(order, "quantity", 0.0) or 0.0) for order in sim_orders],
    )
    opt_gap = _max_abs_pair_gap(
        [order.optj for order in target_orders],
        [float(getattr(order, "OPTj", 0.0) or 0.0) for order in sim_orders],
    )
    target_counts = target.case_counts_excel_formula
    sim_counts = dict(ret["case_counts_excel_formula"])
    row = {
        "cfi": target.cfi,
        "family": design_spec_for_cfi(target.cfi).family,
        "demand_source": demand_source,
        "risk_occurrence_mode": risk_occurrence_mode,
        "risk_attribution_source": risk_attribution_source,
        "seed_stream_mode": seed_stream_mode,
        "workbook_seed": int(target.seed),
        "horizon_hours": float(sim.horizon),
        "target_horizon_hours": float(target.horizon_hours),
        "target_ret_excel_mean": float(target.ret_mean_excel),
        "sim_ret_excel_formula_mean": float(ret["mean_ret_excel_formula"]),
        "sim_ret_text_formula_mean": float(ret["mean_ret_text_formula"]),
        "signed_ret_gap": float(ret["mean_ret_excel_formula"] - target.ret_mean_excel),
        "abs_ret_gap": abs(float(ret["mean_ret_excel_formula"] - target.ret_mean_excel)),
        "target_n_orders": int(target.n_orders),
        "sim_n_orders": int(ret["n_orders"]),
        "n_order_gap": int(ret["n_orders"]) - int(target.n_orders),
        "q_max_abs_gap": q_gap,
        "optj_max_abs_gap": opt_gap,
        "target_case_counts_excel_formula": target_counts,
        "sim_case_counts_excel_formula": sim_counts,
        "max_branch_share_gap_pct": _max_branch_gap(target_counts, sim_counts),
    }
    return row, sim


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    target_counts = {key: 0 for key in CASE_KEYS}
    sim_counts = {key: 0 for key in CASE_KEYS}
    for row in rows:
        for key in CASE_KEYS:
            target_counts[key] += int(row["target_case_counts_excel_formula"].get(key, 0))
            sim_counts[key] += int(row["sim_case_counts_excel_formula"].get(key, 0))
    by_family: dict[str, Any] = {}
    for family in sorted({str(row["family"]) for row in rows}):
        fam_rows = [row for row in rows if row["family"] == family]
        by_family[family] = {
            "mean_abs_ret_gap": float(statistics.fmean(row["abs_ret_gap"] for row in fam_rows)),
            "target_ret_mean": float(statistics.fmean(row["target_ret_excel_mean"] for row in fam_rows)),
            "sim_ret_mean": float(statistics.fmean(row["sim_ret_excel_formula_mean"] for row in fam_rows)),
            "n_configurations": len(fam_rows),
        }
    return {
        "n_configurations": len(rows),
        "mean_abs_ret_gap": float(statistics.fmean(row["abs_ret_gap"] for row in rows)),
        "median_abs_ret_gap": float(statistics.median(row["abs_ret_gap"] for row in rows)),
        "max_abs_ret_gap": float(max(row["abs_ret_gap"] for row in rows)),
        "max_gap_cfi": int(max(rows, key=lambda row: row["abs_ret_gap"])["cfi"]),
        "target_ret_mean": float(statistics.fmean(row["target_ret_excel_mean"] for row in rows)),
        "sim_ret_mean": float(statistics.fmean(row["sim_ret_excel_formula_mean"] for row in rows)),
        "target_case_shares_pct": _case_shares(target_counts),
        "sim_case_shares_pct": _case_shares(sim_counts),
        "max_branch_share_gap_pct": _max_branch_gap(target_counts, sim_counts),
        "by_family": by_family,
    }


def gate_summary(
    *,
    formula_audit: dict[str, Any],
    best_rows: list[dict[str, Any]],
    best_aggregate: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    extraction_passed = (
        int(formula_audit["total_rows"]) == 47_546
        and int(formula_audit["total_mismatches"]) == 0
        and "CF2" in formula_audit["sheets"]
    )
    operational_order_passed = all(
        row["demand_source"] != "excel_order_tape"
        or (
            row["n_order_gap"] == 0
            and row["q_max_abs_gap"] is not None
            and row["q_max_abs_gap"] <= 1e-9
            and row["optj_max_abs_gap"] is not None
            and row["optj_max_abs_gap"] <= 1e-9
        )
        for row in best_rows
    )
    operational_horizon_passed = all(
        abs(float(row["horizon_hours"]) - float(row["target_horizon_hours"])) <= 1e-9
        for row in best_rows
    )
    family_gate = all(
        float(summary["mean_abs_ret_gap"]) <= 0.03
        for summary in best_aggregate.get("by_family", {}).values()
    )
    ret_gate = float(best_aggregate.get("mean_abs_ret_gap", math.inf)) <= 0.02
    branch_gate = float(best_aggregate.get("max_branch_share_gap_pct", math.inf)) <= 5.0
    gates = {
        "extraction_gate_passed": extraction_passed,
        "operational_order_gate_passed": operational_order_passed,
        "operational_horizon_gate_passed": operational_horizon_passed,
        "replication_ret_gate_passed": ret_gate,
        "replication_family_gate_passed": family_gate,
        "replication_branch_gate_passed": branch_gate,
    }
    gates["replication_status"] = (
        "passed_gate" if all(gates.values()) else "failed_gate"
    )

    blockers: list[dict[str, Any]] = []
    if not extraction_passed:
        blockers.append(
            {
                "blocker": "target_extraction",
                "severity": 1.0,
                "detail": "Formula audit did not find 47,546 clean raw rows including CF2.",
            }
        )
    if not operational_order_passed:
        max_order_gap = max(abs(int(row["n_order_gap"])) for row in best_rows)
        blockers.append(
            {
                "blocker": "demand_order_tape",
                "severity": float(max_order_gap),
                "detail": "DES did not reproduce target order count, Q, or OPTj under best config.",
            }
        )
    if not operational_horizon_passed:
        max_horizon_gap = max(
            abs(float(row["horizon_hours"]) - float(row["target_horizon_hours"]))
            for row in best_rows
        )
        blockers.append(
            {
                "blocker": "horizon_alignment",
                "severity": float(max_horizon_gap),
                "detail": "DES horizon did not match observed raw Excel target horizon.",
            }
        )
    if not branch_gate:
        blockers.append(
            {
                "blocker": "risk_order_attribution",
                "severity": float(best_aggregate["max_branch_share_gap_pct"]),
                "detail": "Branch shares remain outside the 5 percentage point gate.",
            }
        )
    if not ret_gate or not family_gate:
        blockers.append(
            {
                "blocker": "ret_level_gap",
                "severity": float(best_aggregate["mean_abs_ret_gap"]),
                "detail": "Mean Excel-formula ReT remains outside the replication threshold.",
            }
        )
    return gates, sorted(blockers, key=lambda item: item["severity"], reverse=True)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "cfi",
        "family",
        "demand_source",
        "risk_occurrence_mode",
        "risk_attribution_source",
        "seed_stream_mode",
        "target_ret_excel_mean",
        "sim_ret_excel_formula_mean",
        "sim_ret_text_formula_mean",
        "signed_ret_gap",
        "abs_ret_gap",
        "target_n_orders",
        "sim_n_orders",
        "n_order_gap",
        "q_max_abs_gap",
        "optj_max_abs_gap",
        "max_branch_share_gap_pct",
        "horizon_hours",
        "target_horizon_hours",
        "workbook_seed",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_order_exports(
    *,
    output_dir: Path,
    best_key: tuple[str, str, str, str],
    targets: dict[int, GarridoCFTarget],
    best_sims: dict[int, MFSCSimulation],
) -> list[str]:
    export_dir = output_dir / "des_order_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    for stale_csv in export_dir.glob("*.csv"):
        stale_csv.unlink()
    written: list[str] = []
    demand_source, risk_occurrence_mode, risk_attribution_source, seed_stream_mode = best_key
    for cfi, sim in sorted(best_sims.items()):
        target = targets[cfi]
        rows = order_level_export_rows(sim=sim, target=target)
        path = (
            export_dir
            / (
                f"CF{cfi:02d}_{demand_source}_{risk_occurrence_mode}_"
                f"{risk_attribution_source}_{seed_stream_mode}.csv"
            )
        )
        fieldnames = [
            "Q",
            "j",
            "OPTj",
            "OATj",
            "CTj",
            "LT",
            "sumBt",
            "APj",
            "RPj",
            "DPj",
            *target.risk_columns,
            "sumUt",
            "OP9",
            "ReT",
            "deltaReT",
            "excel_case",
        ]
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        written.append(str(path))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbooks", nargs="*", type=Path, default=list(DEFAULT_RAW_WORKBOOKS))
    parser.add_argument("--rsult-workbook", type=Path, default=DEFAULT_RSULT_WORKBOOK)
    parser.add_argument("--cf-range", default="1-20")
    parser.add_argument(
        "--demand-sources",
        default="thesis_calendar,excel_order_tape",
        help="Comma-separated demand sources to evaluate.",
    )
    parser.add_argument(
        "--risk-occurrence-modes",
        default="thesis_window,legacy_renewal",
        help="Comma-separated risk occurrence modes to evaluate.",
    )
    parser.add_argument(
        "--risk-attribution-sources",
        default="des_events,excel_risk_tape",
        help="Comma-separated risk attribution sources to evaluate.",
    )
    parser.add_argument(
        "--seed-stream-modes",
        default="split,single",
        help="Comma-separated seed stream modes to evaluate.",
    )
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
        default=Path("outputs/audits/garrido_replication"),
    )
    args = parser.parse_args()

    targets_all = load_raw_garrido_targets(args.workbooks)
    requested_cfis = parse_cf_range(args.cf_range)
    missing = [cfi for cfi in requested_cfis if cfi not in targets_all]
    if missing:
        raise SystemExit(f"Missing raw Excel targets for Cf values: {missing}")
    targets = {cfi: targets_all[cfi] for cfi in requested_cfis}
    formula_audit = audit_raw_garrido_formula(targets_all)
    rsult_summary = summarize_rsult_workbook(args.rsult_workbook)

    all_rows: list[dict[str, Any]] = []
    rows_by_key: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    sims_by_key: dict[tuple[str, str, str, str], dict[int, MFSCSimulation]] = {}
    for demand_source in _split_csv(args.demand_sources):
        for risk_occurrence_mode in _split_csv(args.risk_occurrence_modes):
            for risk_attribution_source in _split_csv(args.risk_attribution_sources):
                if (
                    risk_attribution_source == "excel_risk_tape"
                    and demand_source != "excel_order_tape"
                ):
                    continue
                for seed_stream_mode in _split_csv(args.seed_stream_modes):
                    key = (
                        demand_source,
                        risk_occurrence_mode,
                        risk_attribution_source,
                        seed_stream_mode,
                    )
                    rows_by_key[key] = []
                    sims_by_key[key] = {}
                    for cfi, target in sorted(targets.items()):
                        row, sim = compare_one(
                            target=target,
                            demand_source=demand_source,
                            risk_occurrence_mode=risk_occurrence_mode,
                            risk_attribution_source=risk_attribution_source,
                            seed_stream_mode=seed_stream_mode,
                            args=args,
                        )
                        rows_by_key[key].append(row)
                        all_rows.append(row)
                        sims_by_key[key][cfi] = sim

    matrix_summary = {
        "|".join(key): aggregate_rows(rows) for key, rows in sorted(rows_by_key.items())
    }
    eligible_keys = [
        key for key in rows_by_key if key[0] == "excel_order_tape"
    ] or list(rows_by_key)
    best_key = min(
        eligible_keys,
        key=lambda key: float(aggregate_rows(rows_by_key[key])["mean_abs_ret_gap"]),
    )
    best_aggregate = aggregate_rows(rows_by_key[best_key])
    best_rows = rows_by_key[best_key]
    gates, blockers = gate_summary(
        formula_audit=formula_audit,
        best_rows=best_rows,
        best_aggregate=best_aggregate,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "replication_audit.csv"
    json_path = args.output_dir / "replication_audit.json"
    write_csv(all_rows, csv_path)
    order_export_paths = write_order_exports(
        output_dir=args.output_dir,
        best_key=best_key,
        targets=targets,
        best_sims=sims_by_key[best_key],
    )
    payload = {
        "description": "Garrido Excel replication harness for the Python DES.",
        "formula": EXCEL_RET_FORMULA,
        "config": _json_safe(vars(args)),
        "targets": {str(cfi): target_to_summary(target) for cfi, target in sorted(targets.items())},
        "formula_audit": formula_audit,
        "rsult_secondary_summary": rsult_summary,
        "matrix_summary": matrix_summary,
        "best_config": {
            "demand_source": best_key[0],
            "risk_occurrence_mode": best_key[1],
            "risk_attribution_source": best_key[2],
            "seed_stream_mode": best_key[3],
        },
        "best_summary": best_aggregate,
        "gates": gates,
        "replication_status": gates["replication_status"],
        "blockers": blockers,
        "rows": sorted(all_rows, key=lambda row: row["abs_ret_gap"], reverse=True),
        "order_exports": order_export_paths,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({
        "best_config": payload["best_config"],
        "gates": gates,
        "replication_status": payload["replication_status"],
        "best_summary": best_aggregate,
    }, indent=2, sort_keys=True))
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Order exports: {len(order_export_paths)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
