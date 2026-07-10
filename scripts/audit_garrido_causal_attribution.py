#!/usr/bin/env python3
"""Odd-CF bakeoff: raw overlap vs fixed R24 window vs causal ledger."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_garrido_mechanisms import CALIBRATION_CFS, _run  # noqa: E402
from supply_chain.garrido_replication import (  # noqa: E402
    DEFAULT_RAW_WORKBOOKS,
    load_raw_garrido_targets,
)


ARMS = {
    "raw_overlap": ("des_events", 0.0),
    "fixed_r24_168h": ("des_events", 168.0),
    "causal_order_ledger": ("causal_exposure", 0.0),
}


def _log_error(actual: float, expected: float) -> float:
    return abs(math.log(max(float(actual), 1e-9) / max(float(expected), 1e-9)))


def main() -> None:
    output_dir = Path("outputs/audits/garrido_causal_attribution_odd_cfs")
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = load_raw_garrido_targets(DEFAULT_RAW_WORKBOOKS)
    summaries = []
    risk_rows = []
    for arm, (source, window) in ARMS.items():
        for cfi in CALIBRATION_CFS:
            summary, risks = _run(
                cfi,
                targets[cfi],
                "clock_parallel_independent_rng",
                "elapsed",
                attribution_source=source,
                r24_window_hours=window,
            )
            summary["arm"] = arm
            for row in risks:
                row["arm"] = arm
            summaries.append(summary)
            risk_rows.extend(risks)
            print(
                f"{arm} CF{cfi}: ReT={summary['des_ret']:.6f}/"
                f"{summary['excel_ret']:.6f} RP95={summary['des_rp_p95']:.1f}/"
                f"{summary['excel_rp_p95']:.1f}",
                flush=True,
            )

    aggregate = []
    for arm in ARMS:
        rows = [row for row in summaries if row["arm"] == arm]
        risks = [row for row in risk_rows if row["arm"] == arm]
        aggregate.append(
            {
                "arm": arm,
                "mean_abs_ret_gap": float(
                    np.mean([abs(row["des_ret"] - row["excel_ret"]) for row in rows])
                ),
                "mean_rp95_log_error": float(
                    np.mean(
                        [
                            _log_error(row["des_rp_p95"], row["excel_rp_p95"])
                            for row in rows
                        ]
                    )
                ),
                "mean_per_risk_share_gap": float(
                    np.mean(
                        [abs(float(row["des_share"]) - float(row["excel_share"])) for row in risks]
                    )
                ),
                "mean_per_risk_rp95_log_error": float(
                    np.mean(
                        [
                            _log_error(row["des_rp_p95"], row["excel_rp_p95"])
                            for row in risks
                            if float(row["excel_rp_p95"]) > 0.0
                        ]
                    )
                ),
                "max_physical_ct_delta_vs_raw": 0.0,
            }
        )

    raw_by_cf = {
        row["cfi"]: row for row in summaries if row["arm"] == "raw_overlap"
    }
    for aggregate_row in aggregate:
        arm_rows = [row for row in summaries if row["arm"] == aggregate_row["arm"]]
        aggregate_row["max_physical_ct_delta_vs_raw"] = max(
            abs(row["des_ct_p50"] - raw_by_cf[row["cfi"]]["des_ct_p50"])
            + abs(row["des_ct_p95"] - raw_by_cf[row["cfi"]]["des_ct_p95"])
            for row in arm_rows
        )

    (output_dir / "verdict.json").write_text(
        json.dumps(
            {
                "split": "odd_cf_calibration_only",
                "arms": aggregate,
                "promotion_rule": (
                    "causal_order_ledger must improve ReT, per-risk share, and "
                    "per-risk RP95 over both comparators with zero physical delta"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for filename, rows in (
        ("cf_summary.csv", summaries),
        ("risk_summary.csv", risk_rows),
        ("aggregate.csv", aggregate),
    ):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
