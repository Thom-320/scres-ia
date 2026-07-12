#!/usr/bin/env python3
"""Audit finite downstream recovery/catch-up mechanisms for Garrido R2 runs.

The previous R2 audit showed two facts:

* direct event-overlap attribution under-marks R22/R23/R24 order shares; and
* propagating risk until backlog clearance wildly over-marks because the DES
  downstream backlog often never clears.

This script does not change production defaults. It runs audit-only variants to
isolate which missing mechanism can bound the R2 tail: faster downstream
dispatch, larger downstream dispatch lots, non-blocking backorder service, or a
combination.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_garrido_risk_calibration import (  # noqa: E402
    backlog_intervals,
    build_sim,
    quantiles,
)
from supply_chain.garrido_replication import load_raw_garrido_targets  # noqa: E402
from supply_chain.thesis_design import parse_cf_range  # noqa: E402


VARIANTS = (
    "baseline",
    "downstream_rop12",
    "downstream_double_q",
    "downstream_rop12_double_q",
    "fit_any_backorder",
    "fit_any_backorder_rop12_double_q",
    "theatre_buffer_31500",
    "cssu_buffer_31500",
    "downstream_pipeline_buffer_31500",
    "pipeline_buffer_rop12_double_q",
    "r2_window_168h",
    "r2_window_336h",
    "r2_window_672h",
    "r2_window_336h_release_2500",
    "r2_window_336h_release_5000",
    "r2_window_336h_release_10000",
    "r2_window_336h_release_31500",
    "r2_window_336h_move_2500",
    "r2_window_336h_move_5000",
    "r2_window_336h_move_10000",
)
PIPELINE_BUFFER_QTY = 31_500.0
R2_WINDOW_RISKS = {"R22", "R23", "R24"}


def parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def finite_quantiles(values: list[float]) -> dict[str, float]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return {"n": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    q = quantiles(clean)
    return {
        "n": q.get("n", float(len(clean))),
        "mean": q.get("mean", float(np.mean(clean))),
        "p50": q.get("p50", 0.0),
        "p90": q.get("p90", 0.0),
        "p95": q.get("p95", 0.0),
        "p99": q.get("p99", 0.0),
        "max": q.get("max", 0.0),
    }


def _serve_pending_backorders_fit_any(self: Any):
    """Audit-only service rule: serve any pending order that fits theatre stock.

    The production DES uses a blocking SPT queue. This sensitivity keeps SPT
    sorting but skips an oversized head order if a later order can be served
    from the available theatre inventory. It helps identify whether the tail is
    a queue-priority artifact or a true downstream throughput/catch-up problem.
    """
    while self.pending_backorders:
        available = float(self.rations_theatre.level)
        if available <= 1e-9:
            break
        fit_idx = None
        for idx, order in enumerate(self.pending_backorders):
            requested_qty = float(order.remaining_qty)
            if requested_qty <= 1e-9 or requested_qty <= available + 1e-9:
                fit_idx = idx
                break
        if fit_idx is None:
            break
        order = self.pending_backorders[fit_idx]
        requested_qty = float(order.remaining_qty)
        if requested_qty <= 1e-9:
            self._finalize_pending_backorder(order)
            self._remove_pending_backorder(order)
            continue
        yield self.rations_theatre.get(requested_qty)
        self._finalize_order_after_fulfillment_delay(order)
        self._remove_pending_backorder(order)


def _recovery_window_hours(variant: str) -> float:
    if "r2_window_168h" in variant:
        return 168.0
    if "r2_window_336h" in variant:
        return 336.0
    if "r2_window_672h" in variant:
        return 672.0
    return 0.0


def _r2_recovery_window_controller(sim: Any, variant: str):
    """Audit-only finite downstream catch-up after R2 event completion."""
    window_hours = _recovery_window_hours(variant)
    release_match = re.search(r"release_(\d+)", variant)
    release_qty = float(release_match.group(1)) if release_match else 0.0
    move_match = re.search(r"move_(\d+)", variant)
    move_qty = float(move_match.group(1)) if move_match else 0.0
    if window_hours <= 0.0:
        return

    base = {
        key: sim.params[key]
        for key in (
            "op9_rop",
            "op10_rop",
            "op12_rop",
            "op9_q_min",
            "op9_q_max",
            "op10_q_min",
            "op10_q_max",
            "op12_q_min",
            "op12_q_max",
        )
    }
    boosted_until = 0.0
    seen: set[tuple[str, float, float]] = set()
    boosted = False

    while sim.env.now < sim.horizon:
        for event in sim.risk_events:
            risk_id = str(event.risk_id)
            if risk_id not in R2_WINDOW_RISKS:
                continue
            key = (risk_id, float(event.start_time), float(event.end_time))
            if key in seen:
                continue
            seen.add(key)
            boosted_until = max(boosted_until, float(event.end_time) + window_hours)
            if release_qty > 0.0:
                sim.rations_theatre.put(release_qty)
            if move_qty > 0.0:
                sim.env.process(_move_existing_downstream_stock_to_theatre(sim, move_qty))

        should_boost = float(sim.env.now) < boosted_until
        if should_boost and not boosted:
            for key in ("op9_rop", "op10_rop", "op12_rop"):
                sim.params[key] = 12
            for prefix in ("op9", "op10", "op12"):
                sim.params[f"{prefix}_q_min"] = int(
                    round(float(base[f"{prefix}_q_min"]) * 2.0)
                )
                sim.params[f"{prefix}_q_max"] = int(
                    round(float(base[f"{prefix}_q_max"]) * 2.0)
                )
            boosted = True
        elif boosted and not should_boost:
            sim.params.update(base)
            boosted = False
        yield sim.env.timeout(1.0)


def _move_existing_downstream_stock_to_theatre(sim: Any, target_qty: float):
    """Move existing downstream inventory toward theatre without creating stock."""
    remaining = float(target_qty)
    sources = (
        sim.rations_cssu,
        sim.rations_sb_dispatch,
        sim.rations_sb,
    )
    for source in sources:
        if remaining <= 1e-9:
            break
        available = float(source.level)
        if available <= 1e-9:
            continue
        qty = min(remaining, available)
        yield source.get(qty)
        yield sim.rations_theatre.put(qty)
        remaining -= qty
        yield from sim._serve_pending_backorders()


def apply_variant(sim: Any, variant: str) -> None:
    if "theatre_buffer_31500" in variant or "downstream_pipeline_buffer_31500" in variant or "pipeline_buffer" in variant:
        sim.rations_theatre.put(PIPELINE_BUFFER_QTY)
    if "cssu_buffer_31500" in variant or "downstream_pipeline_buffer_31500" in variant or "pipeline_buffer" in variant:
        sim.rations_cssu.put(PIPELINE_BUFFER_QTY)
    if "downstream_pipeline_buffer_31500" in variant or "pipeline_buffer" in variant:
        sim.rations_sb_dispatch.put(PIPELINE_BUFFER_QTY)
    if "rop12" in variant:
        for key in ("op9_rop", "op10_rop", "op12_rop"):
            sim.params[key] = 12
    if "double_q" in variant:
        for prefix in ("op9", "op10", "op12"):
            sim.params[f"{prefix}_q_min"] = int(round(float(sim.params[f"{prefix}_q_min"]) * 2.0))
            sim.params[f"{prefix}_q_max"] = int(round(float(sim.params[f"{prefix}_q_max"]) * 2.0))
    if "fit_any_backorder" in variant:
        sim._serve_pending_backorders = MethodType(_serve_pending_backorders_fit_any, sim)
    if "r2_window" in variant:
        sim.env.process(_r2_recovery_window_controller(sim, variant))


def excel_targets(cfi_values: list[int]) -> list[dict[str, Any]]:
    targets = load_raw_garrido_targets()
    rows: list[dict[str, Any]] = []
    for cfi in cfi_values:
        orders = targets[cfi].orders
        served = [order for order in orders if order.ctj is not None]
        ct = [float(order.ctj or 0.0) for order in served]
        rp = [float(order.rpj or 0.0) for order in served]
        dp = [float(order.dpj or 0.0) for order in served]
        rows.append(
            {
                "source": "excel",
                "cfi": cfi,
                "n_orders": len(orders),
                "ct_p50": finite_quantiles(ct)["p50"],
                "ct_p90": finite_quantiles(ct)["p90"],
                "ct_p95": finite_quantiles(ct)["p95"],
                "ct_p99": finite_quantiles(ct)["p99"],
                "rp_p50": finite_quantiles(rp)["p50"],
                "rp_p90": finite_quantiles(rp)["p90"],
                "rp_p95": finite_quantiles(rp)["p95"],
                "rp_p99": finite_quantiles(rp)["p99"],
                "dp_p99": finite_quantiles(dp)["p99"],
                "mean_ret_excel": float(np.mean([float(order.ret or 0.0) for order in orders])),
            }
        )
    return rows


def run_variant(cfi_values: list[int], seeds: list[int], variants: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cfi in cfi_values:
        for variant in variants:
            for seed in seeds:
                sim = build_sim(cfi=cfi, seed=seed)
                apply_variant(sim, variant)
                sim.run()
                orders = [
                    order
                    for order in sim.orders
                    if not bool(getattr(order, "metrics_excluded", False))
                ]
                served = [order for order in orders if getattr(order, "CTj", None) is not None]
                ct = [float(order.CTj) for order in served]
                rp = [float(order.RPj) for order in served]
                dp = [float(order.DPj) for order in served]
                intervals = backlog_intervals(sim)
                terminal = sim._inventory_detail()
                rows.append(
                    {
                        "source": "des",
                        "variant": variant,
                        "cfi": cfi,
                        "seed": seed,
                        "n_orders": len(orders),
                        "served_orders": len(served),
                        "lost_orders": sum(1 for order in orders if bool(getattr(order, "lost", False))),
                        "pending_backorders_terminal": len(sim.pending_backorders),
                        "pending_backorder_qty_terminal": float(sim.pending_backorder_qty),
                        "backlog_interval_count": len(intervals),
                        "backlog_positive_hours": sum(end - start for start, end in intervals),
                        "backlog_max_interval": max((end - start for start, end in intervals), default=0.0),
                        "rations_sb_dispatch_terminal": terminal["rations_sb_dispatch"],
                        "rations_cssu_terminal": terminal["rations_cssu"],
                        "rations_theatre_terminal": terminal["rations_theatre"],
                        "delivery_events": len(sim.delivery_events),
                        "total_delivered": float(sim.total_delivered),
                        "ct_p50": finite_quantiles(ct)["p50"],
                        "ct_p90": finite_quantiles(ct)["p90"],
                        "ct_p95": finite_quantiles(ct)["p95"],
                        "ct_p99": finite_quantiles(ct)["p99"],
                        "rp_p50": finite_quantiles(rp)["p50"],
                        "rp_p90": finite_quantiles(rp)["p90"],
                        "rp_p95": finite_quantiles(rp)["p95"],
                        "rp_p99": finite_quantiles(rp)["p99"],
                        "dp_p99": finite_quantiles(dp)["p99"],
                    }
                )
    return rows


def aggregate(rows: list[dict[str, Any]], excel_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    excel_by_cf = {int(row["cfi"]): row for row in excel_rows}
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["variant"]), int(row["cfi"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (variant, cfi), group in sorted(grouped.items()):
        excel = excel_by_cf[cfi]
        ct_p99 = float(np.mean([float(row["ct_p99"]) for row in group]))
        rp_p99 = float(np.mean([float(row["rp_p99"]) for row in group]))
        out.append(
            {
                "variant": variant,
                "cfi": cfi,
                "ct_p50": float(np.mean([float(row["ct_p50"]) for row in group])),
                "ct_p90": float(np.mean([float(row["ct_p90"]) for row in group])),
                "ct_p99": ct_p99,
                "excel_ct_p99": float(excel["ct_p99"]),
                "ct_p99_ratio": ct_p99 / max(1.0, float(excel["ct_p99"])),
                "rp_p99": rp_p99,
                "excel_rp_p99": float(excel["rp_p99"]),
                "rp_p99_ratio": rp_p99 / max(1.0, float(excel["rp_p99"])),
                "lost_orders": float(np.mean([float(row["lost_orders"]) for row in group])),
                "pending_backorder_qty_terminal": float(
                    np.mean([float(row["pending_backorder_qty_terminal"]) for row in group])
                ),
                "backlog_max_interval": float(
                    np.mean([float(row["backlog_max_interval"]) for row in group])
                ),
                "rations_cssu_terminal": float(
                    np.mean([float(row["rations_cssu_terminal"]) for row in group])
                ),
                "rations_theatre_terminal": float(
                    np.mean([float(row["rations_theatre_terminal"]) for row in group])
                ),
            }
        )
    return out


def aggregate_by_variant(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        grouped.setdefault(str(row["variant"]), []).append(row)
    out: list[dict[str, Any]] = []
    for variant, group in sorted(grouped.items()):
        out.append(
            {
                "variant": variant,
                "ct_p99_ratio_mean": float(np.mean([float(row["ct_p99_ratio"]) for row in group])),
                "rp_p99_ratio_mean": float(np.mean([float(row["rp_p99_ratio"]) for row in group])),
                "lost_orders_mean": float(np.mean([float(row["lost_orders"]) for row in group])),
                "pending_qty_mean": float(np.mean([float(row["pending_backorder_qty_terminal"]) for row in group])),
                "backlog_max_interval_mean": float(np.mean([float(row["backlog_max_interval"]) for row in group])),
                "theatre_terminal_mean": float(np.mean([float(row["rations_theatre_terminal"]) for row in group])),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(output_dir: Path, by_variant: list[dict[str, Any]]) -> None:
    lines = [
        "# Garrido R2 Recovery Transient Audit",
        "",
        "Audit-only variants to isolate the finite downstream recovery mechanism missing from the endogenous DES.",
        "",
        "| Variant | CT p99 / Excel | RP p99 / Excel | Lost orders | Pending qty | Backlog max h | Theatre terminal |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in by_variant:
        lines.append(
            f"| {row['variant']} | {row['ct_p99_ratio_mean']:.2f} | "
            f"{row['rp_p99_ratio_mean']:.2f} | {row['lost_orders_mean']:.1f} | "
            f"{row['pending_qty_mean']:.0f} | {row['backlog_max_interval_mean']:.0f} | "
            f"{row['theatre_terminal_mean']:.0f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- A variant that lowers CT/RP p99 ratios and pending quantity toward Excel identifies a plausible missing recovery mechanism.",
            "- A variant that leaves the backlog max interval near the horizon is not enough to bound the endogenous R2 tail.",
            "- These variants are diagnostics only; they do not change DES defaults.",
            "",
        ]
    )
    (output_dir / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cf-range", default="11-20")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/audits/garrido_r2_recovery_transient_2026-06-26"),
    )
    args = parser.parse_args()

    cfi_values = parse_cf_range(args.cf_range)
    seeds = parse_ints(args.seeds)
    variants = [part.strip() for part in args.variants.split(",") if part.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Expected one of {VARIANTS}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    excel_rows = excel_targets(cfi_values)
    run_rows = run_variant(cfi_values, seeds, variants)
    summary_rows = aggregate(run_rows, excel_rows)
    by_variant = aggregate_by_variant(summary_rows)

    write_csv(args.output_dir / "excel_r2_targets.csv", excel_rows)
    write_csv(args.output_dir / "des_variant_runs.csv", run_rows)
    write_csv(args.output_dir / "variant_by_cf_summary.csv", summary_rows)
    write_csv(args.output_dir / "variant_summary.csv", by_variant)
    (args.output_dir / "audit.json").write_text(
        json.dumps(
            {
                "cf_range": cfi_values,
                "seeds": seeds,
                "variants": variants,
                "variant_summary": by_variant,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(args.output_dir, by_variant)

    print(f"WROTE {args.output_dir}")
    for row in by_variant:
        print(
            f"{row['variant']}: ct_p99_ratio={row['ct_p99_ratio_mean']:.2f} "
            f"rp_p99_ratio={row['rp_p99_ratio_mean']:.2f} "
            f"pending={row['pending_qty_mean']:.0f} "
            f"backlog_max={row['backlog_max_interval_mean']:.0f}h"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
