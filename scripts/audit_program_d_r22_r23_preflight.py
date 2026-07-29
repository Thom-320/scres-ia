#!/usr/bin/env python3
"""Program D preflight for R22/R23 physical mechanics before CSSU splitting.

This is a characterization gate, not a calibration sweep. It runs the frozen
proxy contract on CF11-CF20 using Garrido's workbook seeds and reports physical
localization, realized recovery timing, downstream wait liveness, risk-to-order
share gaps, and conditional CT/RP gaps. No parameter is selected from results.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_garrido_mechanisms import _run  # noqa: E402
from supply_chain.garrido_replication import DEFAULT_RAW_WORKBOOKS, load_raw_garrido_targets  # noqa: E402


DEFAULT_OUTPUT = Path("results/program_d/r22_r23_mechanism_preflight")
PROXY_PATH = Path("supply_chain/data/garrido_proxy_v1_freeze_2026-07-10.json")
EXPECTED_OPS = {"R22": {4, 8, 10, 12}, "R23": {11}}
EXPECTED_RECOVERY_MEAN = {"R22": 24.0, "R23": 120.0}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    output = DEFAULT_OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    targets = load_raw_garrido_targets(DEFAULT_RAW_WORKBOOKS)
    summaries: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    for cfi in range(11, 21):
        summary, risks = _run(
            cfi,
            targets[cfi],
            "clock_parallel",
            "elapsed",
            attribution_source="des_events",
            r24_window_hours=168.0,
            material_lineage_mode="off",
        )
        summaries.append(summary)
        risk_rows.extend(row for row in risks if row["risk_id"] in {"R22", "R23"})
        print(
            f"[r2-preflight] CF{cfi}: ReT={summary['des_ret']:.4f}/"
            f"{summary['excel_ret']:.4f} events={summary['event_counts']}",
            flush=True,
        )

    checks: dict[str, Any] = {}
    risk_summary: dict[str, Any] = {}
    for risk_id in ("R22", "R23"):
        enabled = [
            r for r in risk_rows
            if r["risk_id"] == risk_id and int(r["event_count"]) > 0
        ]
        total_events = sum(int(r["event_count"]) for r in enabled)
        total_hours = sum(float(r["event_hours"]) for r in enabled)
        realized_mean = total_hours / total_events if total_events else 0.0
        shares = [float(r["des_share"]) - float(r["excel_share"]) for r in enabled]
        ct50_ratios = [
            float(r["des_ct_p50"]) / float(r["excel_ct_p50"])
            for r in enabled if float(r["excel_ct_p50"]) > 0
        ]
        rp50_ratios = [
            float(r["des_rp_p50"]) / float(r["excel_rp_p50"])
            for r in enabled if float(r["excel_rp_p50"]) > 0
        ]
        observed_ops: set[int] = set()
        invalid_ops: set[int] = set()
        for summary in summaries:
            by_op = summary["event_affected_op_counts"].get(risk_id, {})
            observed_ops.update(int(op) for op, count in by_op.items() if int(count) > 0)
            invalid_ops.update(int(op) for op, count in by_op.items() if int(count) > 0 and int(op) not in EXPECTED_OPS[risk_id])
        timing_ratio = realized_mean / EXPECTED_RECOVERY_MEAN[risk_id]
        risk_summary[risk_id] = {
            "n_enabled_configs": len(enabled),
            "total_events": total_events,
            "realized_recovery_mean_hours": realized_mean,
            "target_recovery_mean_hours": EXPECTED_RECOVERY_MEAN[risk_id],
            "recovery_mean_ratio": timing_ratio,
            "observed_affected_ops": sorted(observed_ops),
            "invalid_affected_ops": sorted(invalid_ops),
            "mean_order_share_gap": float(np.mean(shares)) if shares else None,
            "mean_abs_order_share_gap": float(np.mean(np.abs(shares))) if shares else None,
            "median_conditional_ct_p50_ratio": float(np.median(ct50_ratios)) if ct50_ratios else None,
            "median_conditional_rp_p50_ratio": float(np.median(rp50_ratios)) if rp50_ratios else None,
        }
        checks[f"{risk_id}_events_live"] = total_events > 0
        checks[f"{risk_id}_localization_valid"] = not invalid_ops and bool(observed_ops) and observed_ops <= EXPECTED_OPS[risk_id]
        checks[f"{risk_id}_timing_plausible"] = 0.75 <= timing_ratio <= 1.25

    # The current aggregate proxy can only expose waits at Op10/11/12. R22 may
    # also strike Op4/8 upstream of the recorded order cohort.
    r22_wait = any(
        int(s["op10_down_wait_positive_orders"]) > 0
        or int(s["op12_down_wait_positive_orders"]) > 0
        for s in summaries if int(s["event_counts"].get("R22", 0)) > 0
    )
    r23_wait = any(
        int(s["op11_down_wait_positive_orders"]) > 0
        for s in summaries if int(s["event_counts"].get("R23", 0)) > 0
    )
    checks["R22_order_path_wait_live"] = r22_wait
    checks["R23_order_path_wait_live"] = r23_wait
    physical_pass = all(bool(value) for value in checks.values())

    ret_rel_gaps = [
        (float(s["des_ret"]) - float(s["excel_ret"])) / max(abs(float(s["excel_ret"])), 1e-12)
        for s in summaries
    ]
    attribution_disclosure_required = any(
        (risk_summary[rid]["mean_abs_order_share_gap"] or 0.0) > 0.05
        for rid in ("R22", "R23")
    )
    verdict = {
        "kind": "program_d_r22_r23_mechanism_preflight",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha_input": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
        "proxy_sha256": sha256(PROXY_PATH.read_bytes()).hexdigest(),
        "configurations": list(range(11, 21)),
        "workbook_seeds": [int(targets[cfi].seed) for cfi in range(11, 21)],
        "metric": "ret_excel_visible_v1",
        "risk_summary": risk_summary,
        "checks": checks,
        "r2_ret_mean_relative_gap": float(np.mean(ret_rel_gaps)),
        "r2_ret_mean_absolute_relative_gap": float(np.mean(np.abs(ret_rel_gaps))),
        "physical_mechanism_pass": physical_pass,
        "attribution_disclosure_required": attribution_disclosure_required,
        "interpretation": (
            "PASS_PHYSICAL_WITH_ATTRIBUTION_LIMITATION" if physical_pass
            else "STOP_R22_R23_PHYSICAL_MECHANISM"
        ),
        "dra1_refactor_authorized": physical_pass,
        "virgin_tapes_opened": 0,
        "ppo_trained": False,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
    }
    write_csv(output / "config_summary.csv", summaries)
    write_csv(output / "risk_conditional_metrics.csv", risk_rows)
    (output / "verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if physical_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
