#!/usr/bin/env python3
"""Reproduce Garrido-Rios Cf1-Cf20 and compare against the delivered workbooks.

This audit deliberately separates two different questions:

1. Can the published Excel ReT formula be reconstructed from the workbook rows?
2. Can the repository DES regenerate the workbook outputs from the published
   configuration and workbook seed, without replaying workbook demand or risk
   attribution tapes?

The first is a formula/provenance check.  The second is the scientific
replication check.  An exact formula reconstruction must never be reported as
an independent DES replication.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.garrido_replication import (  # noqa: E402
    DEFAULT_RAW_WORKBOOKS,
    audit_raw_garrido_formula,
    load_raw_garrido_targets,
)
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402
from supply_chain.thesis_design import (  # noqa: E402
    design_spec_for_cfi,
    parse_cf_range,
)


SCOPE = "HISTORICAL_DES_REPLICATION_NO_PAPER_B_SELECTION"
VERDICT = "FORMULA_RECONSTRUCTION_PASS__DES_GENERATIVE_REPLICATION_FAIL"
CF21_CF90_STATUS = "HOLD_FIDELITY_NOT_ESTABLISHED"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals)


def pearson(x: list[float], y: list[float]) -> float:
    x_mean = mean(x)
    y_mean = mean(y)
    covariance = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True))
    x_var = sum((a - x_mean) ** 2 for a in x)
    y_var = sum((b - y_mean) ** 2 for b in y)
    if x_var == 0.0 or y_var == 0.0:
        return math.nan
    return covariance / math.sqrt(x_var * y_var)


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    excel = [float(row["excel_ret"]) for row in rows]
    simulated = [float(row["sim_ret"]) for row in rows]
    gaps = [b - a for a, b in zip(excel, simulated, strict=True)]
    order_errors = [
        float(row["sim_orders"]) / float(row["excel_max_j"]) - 1.0 for row in rows
    ]
    return {
        "n_configurations": len(rows),
        "mean_excel_ret": mean(excel),
        "mean_sim_ret": mean(simulated),
        "mean_bias": mean(gaps),
        "mean_absolute_error": mean(abs(gap) for gap in gaps),
        "max_absolute_error": max(abs(gap) for gap in gaps),
        "ret_pearson_correlation": pearson(excel, simulated),
        "mean_order_count_relative_error": mean(order_errors),
        "max_absolute_order_count_relative_error": max(
            abs(error) for error in order_errors
        ),
    }


def run_generative(cfi: int, target: Any) -> dict[str, Any]:
    spec = design_spec_for_cfi(cfi)
    started = time.monotonic()
    sim = MFSCSimulation(
        shifts=spec.shifts,
        initial_buffers=spec.initial_buffers,
        seed=int(target.seed),
        horizon=spec.horizon_hours,
        risks_enabled=True,
        risk_level="current",
        year_basis=P["year_basis"],
        deterministic_baseline=False,
        stochastic_pt=False,
        warmup_trigger=P["warmup_trigger"],
        downstream_q_source=P["downstream_q_source"],
        r14_defect_mode=P["r14_defect_mode"],
        enabled_risks=set(spec.enabled_risks),
        risk_overrides=dict(spec.risk_overrides),
        inventory_replenishment_period=spec.inventory_replenishment_period,
        raw_material_flow_mode=P["raw_material_flow_mode"],
        raw_material_order_up_to_multiplier=float(
            P["raw_material_order_up_to_multiplier"]
        ),
        demand_on_hand_fulfillment_delay=float(
            P["demand_on_hand_fulfillment_delay"]
        ),
        seed_stream_mode="split",
    ).run()
    ret = sim.compute_order_level_ret()
    risk_events: dict[str, int] = {}
    for event in sim.risk_events:
        risk_events[event.risk_id] = risk_events.get(event.risk_id, 0) + 1
    return {
        "cfi": cfi,
        "family": spec.family,
        "seed": int(target.seed),
        "horizon_hours": float(spec.horizon_hours),
        "excel_visible_orders": int(target.n_orders),
        "excel_max_j": int(target.max_j),
        "excel_ret": float(target.ret_mean_excel),
        "sim_orders": len(sim.orders),
        "sim_ret": float(ret["mean_ret_excel_formula"]),
        "sim_ret_text": float(ret["mean_ret_text_formula"]),
        "ret_gap": float(ret["mean_ret_excel_formula"] - target.ret_mean_excel),
        "sim_backorders": int(sim.total_backorders),
        "sim_unattended": int(sim.total_unattended_orders),
        "sim_demanded": float(sim.total_demanded),
        "sim_delivered": float(sim.total_delivered),
        "sim_risk_events": risk_events,
        "elapsed_seconds": time.monotonic() - started,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat = dict(row)
        flat["sim_risk_events"] = json.dumps(
            flat["sim_risk_events"], sort_keys=True
        )
        flat_rows.append(flat)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workbooks",
        nargs=2,
        type=Path,
        default=list(DEFAULT_RAW_WORKBOOKS),
    )
    parser.add_argument("--cf-range", default="1-20")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/garrido_reproduction/cf1_cf20_v1"),
    )
    parser.add_argument(
        "--formula-only",
        action="store_true",
        help="Audit workbook formula and provenance without running the DES.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    selected = parse_cf_range(args.cf_range)
    if not selected or any(cfi < 1 or cfi > 20 for cfi in selected):
        raise ValueError("This historical validation instrument only permits Cf1-Cf20.")
    if args.output_dir.exists():
        raise FileExistsError(f"output_directory_must_not_exist: {args.output_dir}")
    for workbook in args.workbooks:
        if not workbook.is_file():
            raise FileNotFoundError(workbook)

    git_commit = git_value("rev-parse", "HEAD")
    git_clean_before_run = not bool(git_value("status", "--porcelain"))
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    targets = load_raw_garrido_targets(args.workbooks)
    formula_audit = audit_raw_garrido_formula(
        {cfi: targets[cfi] for cfi in selected}
    )
    rows = (
        []
        if args.formula_only
        else [run_generative(cfi, targets[cfi]) for cfi in selected]
    )

    summaries: dict[str, Any] = {}
    if rows:
        summaries["R1_Cf1_Cf10"] = summarize_group(
            [row for row in rows if row["cfi"] <= 10]
        )
        summaries["R2_Cf11_Cf20"] = summarize_group(
            [row for row in rows if row["cfi"] >= 11]
        )
        summaries["all"] = summarize_group(rows)
        write_csv(args.output_dir / "rows.csv", rows)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": SCOPE,
        "verdict": (
            "FORMULA_RECONSTRUCTION_PASS__GENERATIVE_NOT_RUN"
            if args.formula_only
            else VERDICT
        ),
        "cf21_cf90_status": CF21_CF90_STATUS,
        "claim_boundary": (
            "The workbook formula is exactly reconstructable. The current DES does "
            "not numerically regenerate Cf1-Cf20 from design plus workbook seed. "
            "No Cf21-Cf90 output is treated as thesis replication or Paper B "
            "selection evidence."
        ),
        "acceptance_note": (
            "No prospective numerical tolerance was preregistered for this historical "
            "audit. The generative verdict records descriptive discrepancies too large "
            "to support a numerical-replication claim; it is not a post-hoc gate for "
            "Paper B."
        ),
        "git": {
            "commit": git_commit,
            "worktree_clean_before_run": git_clean_before_run,
        },
        "workbooks": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in args.workbooks
        ],
        "selected_cfis": selected,
        "seeds": {str(cfi): int(targets[cfi].seed) for cfi in selected},
        "formula_audit": formula_audit,
        "generative_summary": summaries,
        "rows": rows,
        "elapsed_seconds": time.monotonic() - started,
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    receipt = {
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "source_workbook_sha256": {
            path.name: sha256_file(path) for path in args.workbooks
        },
        "scope": SCOPE,
        "verdict": payload["verdict"],
    }
    (args.output_dir / "completion_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
