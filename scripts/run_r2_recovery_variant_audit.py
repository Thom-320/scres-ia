#!/usr/bin/env python3
"""Audit the bounded stock-release R2 recovery variant on CF11-CF20.

Implements the candidate from `docs/R2_AUDIT_DECOMPOSITION_2026-06-29.md` §5:
after each R2 event end, inject `risk_recovery_release_rations` to theatre
and drain the pending backorder queue while the recovery window is open.

This is an opt-in sensitivity, not a freeze default. The freeze
(`THESIS_FAITHFUL_PROTOCOL`) keeps `risk_recovery_window_hours=0`.

Variants tested:
  - baseline:        window=0, release=0 (no recovery; reference)
  - release_2_500:   window=336h, release=2,500 (the "sweet spot" candidate)
  - release_5_000:   window=336h, release=5,000 (intermediate)
  - release_10_000:  window=336h, release=10,000 (over-correction check)

Stop-rule (registered before running):
  - PASS:    CTj p99 ratio < 1.5x Excel AND RPj p99 ratio < 2.0x Excel AND
             theatre terminal < 200k rations AND lost < 20
  - PARTIAL: CTj p99 ratio < 2.0x Excel (was 2.09) but one of the others worse
  - FAIL:    any worse than baseline
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import (  # noqa: E402
    RET_RECOVERY_PERIOD_MODE,
    THESIS_FAITHFUL_PROTOCOL as P,
    THESIS_REPLICATION_DOWNSTREAM_Q_SOURCE as DQ,
)
from supply_chain.garrido_replication import load_raw_garrido_targets  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402
from supply_chain.thesis_design import R2_RISKS, design_spec_for_cfi, parse_cf_range  # noqa: E402


VARIANTS = {
    "baseline": {"window_hours": 0.0, "release_rations": 0.0},
    "release_2_500": {"window_hours": 336.0, "release_rations": 2_500.0},
    "release_5_000": {"window_hours": 336.0, "release_rations": 5_000.0},
    "release_10_000": {"window_hours": 336.0, "release_rations": 10_000.0},
}
DEFAULT_BOOST_DOWNSTREAM = True


def parse_ints(value: str) -> list[int]:
    return [int(p.strip()) for p in value.split(",") if p.strip()]


def build_sim(
    *,
    cfi: int,
    seed: int,
    risk_recovery_window_hours: float,
    risk_recovery_release_rations: float,
    risk_recovery_boost_downstream: bool = DEFAULT_BOOST_DOWNSTREAM,
) -> MFSCSimulation:
    spec = design_spec_for_cfi(cfi)
    return MFSCSimulation(
        shifts=1,
        initial_buffers=None,
        seed=seed,
        horizon=spec.horizon_hours,
        risks_enabled=True,
        risk_level="current",
        year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"],
        downstream_q_source=DQ,
        r14_defect_mode=P["r14_defect_mode"],
        enabled_risks=set(spec.enabled_risks),
        risk_overrides=spec.risk_overrides,
        risk_occurrence_mode=P["risk_occurrence_mode"],
        ret_recovery_period_mode=RET_RECOVERY_PERIOD_MODE,
        backorder_overflow_mode="largest",
        raw_material_flow_mode=P["raw_material_flow_mode"],
        raw_material_order_up_to_multiplier=P[
            "raw_material_order_up_to_multiplier"
        ],
        demand_on_hand_fulfillment_delay=P["demand_on_hand_fulfillment_delay"],
        risk_recovery_window_hours=risk_recovery_window_hours,
        risk_recovery_release_rations=risk_recovery_release_rations,
        risk_recovery_boost_downstream=risk_recovery_boost_downstream,
    )


def run_sim(
    sim: MFSCSimulation,
) -> dict[str, Any]:
    sim.run()
    served = [
        o
        for o in sim.orders
        if not bool(getattr(o, "metrics_excluded", False))
        and getattr(o, "OATj", None) is not None
        and not bool(getattr(o, "lost", False))
    ]
    lost = sum(1 for o in sim.orders if bool(getattr(o, "lost", False)))
    ctj = np.asarray([float(o.CTj) for o in served if o.CTj is not None])
    rpj = np.asarray(
        [float(getattr(o, "RPj", 0.0) or 0.0) for o in served]
    )
    if ctj.size == 0:
        return {
            "ct_p50": 0.0,
            "ct_p90": 0.0,
            "ct_p99": 0.0,
            "rp_p99": 0.0,
            "lost_orders": float(lost),
            "pending_backorder_qty_terminal": float(sim.pending_backorder_qty),
            "backlog_max_interval": 0.0,
            "rations_cssu_terminal": float(sim.rations_cssu.level),
            "rations_theatre_terminal": float(sim.rations_theatre.level),
            "risk_recovery_release_emitted": float(
                getattr(sim, "_risk_recovery_release_emitted", 0.0)
            ),
        }
    return {
        "ct_p50": float(np.percentile(ctj, 50)),
        "ct_p90": float(np.percentile(ctj, 90)),
        "ct_p99": float(np.percentile(ctj, 99)),
        "rp_p99": float(np.percentile(rpj, 99)),
        "lost_orders": float(lost),
        "pending_backorder_qty_terminal": float(sim.pending_backorder_qty),
        "backlog_max_interval": 0.0,
        "rations_cssu_terminal": float(sim.rations_cssu.level),
        "rations_theatre_terminal": float(sim.rations_theatre.level),
        "risk_recovery_release_emitted": float(
            getattr(sim, "_risk_recovery_release_emitted", 0.0)
        ),
    }


def excel_targets_for_cf(cfi: int) -> dict[str, float]:
    targets = load_raw_garrido_targets()
    target = targets[cfi]
    cts, rps = [], []
    for order in target.orders:
        if order.ctj is not None:
            cts.append(float(order.ctj))
        rps.append(float(getattr(order, "rpj", 0.0) or 0.0))
    return {
        "excel_ct_p99": float(np.percentile(np.asarray(cts), 99))
        if cts
        else 0.0,
        "excel_rp_p99": float(np.percentile(np.asarray(rps), 99))
        if rps
        else 0.0,
    }


def stop_rule(
    variant_summary: dict[str, Any], baseline_summary: dict[str, Any]
) -> str:
    ct_ratio = variant_summary["ct_p99_ratio_mean"]
    rp_ratio = variant_summary["rp_p99_ratio_mean"]
    theatre = variant_summary["theatre_terminal_mean"]
    lost = variant_summary["lost_orders_mean"]
    baseline_ct = baseline_summary["ct_p99_ratio_mean"]
    if ct_ratio < 1.5 and rp_ratio < 2.0 and theatre < 200_000 and lost < 20:
        return "PASS"
    if ct_ratio < 2.0 and ct_ratio < baseline_ct:
        return "PARTIAL"
    return "FAIL"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cf-range", default="11-20")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/audits/r2_recovery_variant_2026-06-29"),
    )
    args = parser.parse_args()

    cfi_values = parse_cf_range(args.cf_range)
    seeds = parse_ints(args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    started = time.time()
    for variant_name, params in VARIANTS.items():
        for cfi in cfi_values:
            excel = excel_targets_for_cf(cfi)
            for seed in seeds:
                sim = build_sim(
                    cfi=cfi,
                    seed=seed,
                    risk_recovery_window_hours=params["window_hours"],
                    risk_recovery_release_rations=params["release_rations"],
                )
                metrics = run_sim(sim)
                ct_p99 = metrics["ct_p99"]
                rp_p99 = metrics["rp_p99"]
                ct_p99_ratio = (
                    ct_p99 / excel["excel_ct_p99"]
                    if excel["excel_ct_p99"] > 0
                    else float("nan")
                )
                rp_p99_ratio = (
                    rp_p99 / excel["excel_rp_p99"]
                    if excel["excel_rp_p99"] > 0
                    else float("nan")
                )
                rows.append(
                    {
                        "variant": variant_name,
                        "cfi": cfi,
                        "seed": seed,
                        "window_hours": params["window_hours"],
                        "release_rations": params["release_rations"],
                        "ct_p50": metrics["ct_p50"],
                        "ct_p90": metrics["ct_p90"],
                        "ct_p99": ct_p99,
                        "excel_ct_p99": excel["excel_ct_p99"],
                        "ct_p99_ratio": ct_p99_ratio,
                        "rp_p99": rp_p99,
                        "excel_rp_p99": excel["excel_rp_p99"],
                        "rp_p99_ratio": rp_p99_ratio,
                        "lost_orders": metrics["lost_orders"],
                        "pending_backorder_qty_terminal": metrics[
                            "pending_backorder_qty_terminal"
                        ],
                        "rations_cssu_terminal": metrics["rations_cssu_terminal"],
                        "rations_theatre_terminal": metrics[
                            "rations_theatre_terminal"
                        ],
                        "risk_recovery_release_emitted": metrics[
                            "risk_recovery_release_emitted"
                        ],
                    }
                )

    fieldnames = list(rows[0].keys())
    with (args.output_dir / "variant_runs.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    variant_summary: dict[str, dict[str, Any]] = {}
    for variant_name in VARIANTS:
        vrows = [r for r in rows if r["variant"] == variant_name]
        if not vrows:
            continue
        variant_summary[variant_name] = {
            "ct_p99_ratio_mean": fmean(r["ct_p99_ratio"] for r in vrows),
            "rp_p99_ratio_mean": fmean(r["rp_p99_ratio"] for r in vrows),
            "lost_orders_mean": fmean(r["lost_orders"] for r in vrows),
            "pending_qty_mean": fmean(
                r["pending_backorder_qty_terminal"] for r in vrows
            ),
            "theatre_terminal_mean": fmean(
                r["rations_theatre_terminal"] for r in vrows
            ),
            "release_emitted_mean": fmean(
                r["risk_recovery_release_emitted"] for r in vrows
            ),
        }

    if "baseline" in variant_summary:
        baseline_summary = variant_summary["baseline"]
    else:
        baseline_summary = {
            "ct_p99_ratio_mean": float("nan"),
            "rp_p99_ratio_mean": float("nan"),
        }

    verdicts: dict[str, str] = {}
    for variant_name in VARIANTS:
        if variant_name == "baseline":
            verdicts[variant_name] = "REFERENCE"
        else:
            verdicts[variant_name] = stop_rule(
                variant_summary[variant_name], baseline_summary
            )

    summary_table_rows: list[dict[str, Any]] = []
    for variant_name, summary in variant_summary.items():
        summary_table_rows.append(
            {
                "variant": variant_name,
                **summary,
                "verdict": verdicts[variant_name],
            }
        )

    with (args.output_dir / "variant_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(summary_table_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(summary_table_rows)

    elapsed = time.time() - started
    audit_payload = {
        "cf_range": cfi_values,
        "seeds": seeds,
        "variants": list(VARIANTS.keys()),
        "wall_seconds": elapsed,
        "variant_summary": variant_summary,
        "verdicts": verdicts,
        "stop_rule": {
            "pass": "ct_p99_ratio < 1.5 AND rp_p99_ratio < 2.0 AND theatre < 200k AND lost < 20",
            "partial": "ct_p99_ratio < 2.0 AND improvement vs baseline",
            "fail": "any worse than baseline",
        },
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit_payload, indent=2), encoding="utf-8"
    )

    lines = [
        "# R2 Bounded Recovery Variant Audit (2026-06-29)",
        "",
        "Candidate from `docs/R2_AUDIT_DECOMPOSITION_2026-06-29.md` §5.",
        "Mechanism: after each R2 event end, inject `release_rations` to theatre and",
        "drain the pending backorder queue for `window_hours`.",
        "",
        "## Stop-rule verdicts",
        "",
        "| Variant | CT p99 / Excel | RP p99 / Excel | Lost | Theatre term | Release emitted | Verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_table_rows:
        lines.append(
            f"| `{row['variant']}` | {row['ct_p99_ratio_mean']:.2f}x "
            f"| {row['rp_p99_ratio_mean']:.2f}x "
            f"| {row['lost_orders_mean']:.1f} "
            f"| {row['theatre_terminal_mean']:,.0f} "
            f"| {row['release_emitted_mean']:,.0f} "
            f"| **{row['verdict']}** |"
        )
    lines.extend(
        [
            "",
            "## Stop-rule definitions",
            "- PASS:    CT p99 ratio < 1.5 AND RP p99 ratio < 2.0 AND theatre < 200k AND lost < 20",
            "- PARTIAL: CT p99 ratio < 2.0 AND improvement vs baseline",
            "- FAIL:    any worse than baseline",
            "",
            f"Wall time: {elapsed:.1f}s",
        ]
    )
    (args.output_dir / "audit_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print(f"WROTE {args.output_dir} (wall {elapsed:.1f}s)")
    for row in summary_table_rows:
        print(
            f"{row['variant']:18}  ct_ratio={row['ct_p99_ratio_mean']:.2f}x  "
            f"rp_ratio={row['rp_p99_ratio_mean']:.2f}x  "
            f"lost={row['lost_orders_mean']:.1f}  "
            f"theatre={row['theatre_terminal_mean']:,.0f}  "
            f"verdict={row['verdict']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
