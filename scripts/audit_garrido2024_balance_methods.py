#!/usr/bin/env python3
"""Audit Garrido-2024 index balancing methods on a static policy surface.

This is a metric-quality gate, not an RL result finder. A candidate balance method
must not choose a zero-buffer or poor-service policy as the best policy merely
because it is cheap. The script reuses one DES sample of static policies, then
scores the same rows under several balance methods and cost scales.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.calibrate_cd_exponents import (
    SIGNS,
    build_parser as build_calibration_parser,
    calibrate_from_rows,
    collect_episode_rows,
)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-x)))


def _row_variables(row: dict[str, Any], payload: dict[str, Any]) -> dict[str, float]:
    kappa_ref = max(float(payload["kappa_ref"]), 1e-6)
    return {
        "zeta": max(float(row["zeta_avg"]), 1e-6),
        "epsilon": max(float(row["epsilon_avg"]), 1e-6),
        "phi": max(float(row["phi_avg"]), 1e-6),
        "tau": max(float(row["tau_avg"]), 1e-6),
        "kappa_dot": max(float(row["average_cost"]) / kappa_ref, 1e-6),
    }


def score_row(
    row: dict[str, Any], payload: dict[str, Any], *, kappa_scale: float
) -> tuple[float, float]:
    variables = _row_variables(row, payload)
    method = str(payload["balance_method"])
    score = 0.0
    for key, value in variables.items():
        sign = SIGNS[key]
        scale = float(kappa_scale) if key == "kappa_dot" else 1.0
        if method == "variance_log":
            mean = float(payload["log_mean"][key])
            std = max(float(payload["log_std"][key]), 1e-6)
            term = (float(payload["balance_c"]) / std) * (math.log(value) - mean)
        elif method == "minmax":
            lo = float(payload["minmax_min"][key])
            hi = float(payload["minmax_max"][key])
            norm = (value - lo) / max(hi - lo, 1e-6)
            norm = min(max(norm, 1e-6), 1.0)
            utility = norm if sign > 0.0 else 1.0 - norm
            utility = min(max(utility, 1e-6), 1.0)
            gmean = max(float(payload["minmax_utility_gmean"][key]), 1e-6)
            term = 0.2 * math.log(utility / gmean)
            sign = 1.0
        else:
            key_map = {
                "zeta": "a_zeta",
                "epsilon": "b_epsilon",
                "phi": "c_phi",
                "tau": "d_tau",
                "kappa_dot": "n_kappa",
            }
            term = float(payload[key_map[key]]) * math.log(value)
        score += sign * scale * term
    return float(score), _sigmoid(score)


def enrich_rows(
    rows: list[dict[str, Any]],
    *,
    balance_methods: list[str],
    kappa_scales: list[float],
) -> pd.DataFrame:
    scored: list[dict[str, Any]] = []
    payloads = {method: calibrate_from_rows(rows, balance_method=method) for method in balance_methods}
    for method, payload in payloads.items():
        for row in rows:
            demand = float(row.get("cumulative_demanded", 0.0))
            backorder = float(row.get("cumulative_backorder_qty", 0.0))
            flow_fill = 1.0 - backorder / demand if demand > 0.0 else float("nan")
            for kappa_scale in kappa_scales:
                log_score, index = score_row(row, payload, kappa_scale=kappa_scale)
                scored.append(
                    {
                        **row,
                        "balance_method": method,
                        "kappa_scale": float(kappa_scale),
                        "log_score": log_score,
                        "g24_index": index,
                        "flow_fill_rate": flow_fill,
                        "kappa_dot": _row_variables(row, payload)["kappa_dot"],
                    }
                )
    return pd.DataFrame(scored)


def summarize(scored: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        scored.groupby(["balance_method", "kappa_scale", "risk_level", "policy"], as_index=False)
        .agg(
            g24_index=("g24_index", "mean"),
            log_score=("log_score", "mean"),
            flow_fill_rate=("flow_fill_rate", "mean"),
            epsilon_avg=("epsilon_avg", "mean"),
            pending_backorder_qty=("pending_backorder_qty", "mean"),
            average_cost=("average_cost", "mean"),
            kappa_dot=("kappa_dot", "mean"),
            zeta_avg=("zeta_avg", "mean"),
            episode_count=("episode", "count"),
        )
    )
    top_flow = (
        grouped.groupby(["balance_method", "kappa_scale", "risk_level"])["flow_fill_rate"]
        .max()
        .rename("top_flow_fill_rate")
    )
    top_index = grouped.loc[
        grouped.groupby(["balance_method", "kappa_scale", "risk_level"])["g24_index"].idxmax()
    ].copy()
    top_index = top_index.merge(
        top_flow.reset_index(),
        on=["balance_method", "kappa_scale", "risk_level"],
        how="left",
    )
    top_index["flow_retention_vs_best"] = (
        top_index["flow_fill_rate"] / top_index["top_flow_fill_rate"].replace(0.0, np.nan)
    )
    top_index["chooses_zero_buffer"] = top_index["policy"].astype(str).str.contains(
        "I0|b0.00|b0.0", regex=True
    )
    top_index["passes_surface_gate"] = (
        (~top_index["chooses_zero_buffer"])
        & (top_index["flow_retention_vs_best"] >= 0.95)
        & (top_index["g24_index"].between(0.05, 0.95))
    )
    return top_index.sort_values(["risk_level", "balance_method", "kappa_scale"])


def correlation_summary(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, df in scored.groupby(["balance_method", "kappa_scale", "risk_level"]):
        method, kappa_scale, risk_level = keys
        rows.append(
            {
                "balance_method": method,
                "kappa_scale": float(kappa_scale),
                "risk_level": risk_level,
                "spearman_index_flow": float(
                    df["g24_index"].corr(df["flow_fill_rate"], method="spearman")
                ),
                "spearman_index_epsilon": float(
                    df["g24_index"].corr(df["epsilon_avg"], method="spearman")
                ),
                "spearman_index_cost": float(
                    df["g24_index"].corr(df["kappa_dot"], method="spearman")
                ),
                "index_std": float(df["g24_index"].std()),
                "index_min": float(df["g24_index"].min()),
                "index_max": float(df["g24_index"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["risk_level", "balance_method", "kappa_scale"])


def build_parser() -> argparse.ArgumentParser:
    base = build_calibration_parser()
    parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[base],
        conflict_handler="resolve",
    )
    parser.set_defaults(output=Path("/tmp/scresia_g24_balance_unused.json"))
    parser.add_argument(
        "--balance-methods",
        nargs="+",
        default=["max_offset", "variance_log", "minmax"],
        choices=["max_offset", "variance_log", "minmax"],
    )
    parser.add_argument(
        "--kappa-scales",
        nargs="+",
        type=float,
        default=[1.0, 0.2, 0.0],
        help="Cost-term scales to audit; 0.0 is a sensitivity, not the default metric.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/audits/garrido2024_balance_methods"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = collect_episode_rows(args)
    scored = enrich_rows(
        rows,
        balance_methods=list(args.balance_methods),
        kappa_scales=[float(v) for v in args.kappa_scales],
    )
    surface = summarize(scored)
    correlations = correlation_summary(scored)
    args.output_root.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_root / "scored_rows.csv", index=False)
    surface.to_csv(args.output_root / "surface_gate_summary.csv", index=False)
    correlations.to_csv(args.output_root / "correlation_summary.csv", index=False)
    manifest = {
        "row_count": len(rows),
        "balance_methods": list(args.balance_methods),
        "kappa_scales": [float(v) for v in args.kappa_scales],
        "run_config": {
            "action_space_mode": args.action_space_mode,
            "risk_levels": list(args.risk_levels),
            "risk_occurrence_mode": args.risk_occurrence_mode,
            "raw_material_flow_mode": args.raw_material_flow_mode,
            "raw_material_order_up_to_multiplier": args.raw_material_order_up_to_multiplier,
            "stochastic_pt": bool(args.stochastic_pt),
            "stochastic_pt_spread": args.stochastic_pt_spread,
            "episodes_per_risk_level": args.episodes,
            "max_steps": args.max_steps,
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    pd.set_option("display.width", 220)
    print("=== Surface gate summary ===")
    print(surface.to_string(index=False))
    print("\n=== Correlation summary ===")
    print(correlations.to_string(index=False))
    print(f"\nWrote: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
