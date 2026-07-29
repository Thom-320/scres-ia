#!/usr/bin/env python3
"""Diagnose why a dynamic Track-A policy is not beating static baselines.

The script consumes outputs from ``compare_garrido_dynamic_vs_static.py`` and
turns them into an explicit failure/smoke report: train/eval mismatch, action
mix, component deltas, service deltas, and whether any apparent C-D win looks
like service resilience or metric gaming.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_GLOB = "outputs/benchmarks/garrido_dynamic_cd/*/summary.json"
COMPONENT_FIELDS = (
    "cd_zeta_avg",
    "cd_epsilon_avg",
    "cd_phi_avg",
    "cd_tau_avg",
    "cd_kappa_dot",
)
SERVICE_FIELDS = (
    "mean_ret_excel_formula",
    "flow_fill_rate",
    "fill_rate_order_level",
    "service_loss_mean",
    "service_loss_p95",
    "service_loss_cvar95",
    "backorder_qty_total",
    "pending_backorder_qty_terminal",
    "unattended_orders_terminal",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(statistics.fmean(finite)) if finite else float("nan")


def _latest_summary() -> Path:
    candidates = [path for path in REPO_ROOT.glob(DEFAULT_RUN_GLOB)]
    if not candidates:
        raise FileNotFoundError(f"No dynamic summary found via {DEFAULT_RUN_GLOB}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _git_info() -> dict[str, str]:
    def git(*parts: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *parts],
                cwd=REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return ""

    return {
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "status_short": git("status", "--short"),
    }


def _summary_index(summary_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["regime"]), str(row["policy"])): row for row in summary_rows}


def _static_rows(summary_rows: list[dict[str, Any]], regime: str) -> list[dict[str, Any]]:
    return [
        row
        for row in summary_rows
        if str(row.get("regime")) == regime and str(row.get("policy", "")).startswith("static_")
        or (
            str(row.get("regime")) == regime
            and str(row.get("policy")) == "original_S1_I0"
        )
    ]


def _best_static(
    summary_rows: list[dict[str, Any]], regime: str, metric: str
) -> dict[str, Any] | None:
    rows = _static_rows(summary_rows, regime)
    field = f"{metric}_mean"
    rows = [row for row in rows if field in row and math.isfinite(_float(row[field]))]
    return max(rows, key=lambda row: _float(row[field])) if rows else None


def _dynamic_row(summary_rows: list[dict[str, Any]], regime: str) -> dict[str, Any] | None:
    for row in summary_rows:
        if str(row.get("regime")) == regime and str(row.get("policy")) == "ppo_dynamic":
            return row
    return None


def _field(row: dict[str, Any], metric: str) -> float:
    return _float(row.get(f"{metric}_mean"))


def _component_delta(dynamic: dict[str, Any], static: dict[str, Any]) -> dict[str, float]:
    return {field: _field(dynamic, field) - _field(static, field) for field in COMPONENT_FIELDS}


def _service_delta(dynamic: dict[str, Any], static: dict[str, Any]) -> dict[str, float]:
    return {field: _field(dynamic, field) - _field(static, field) for field in SERVICE_FIELDS}


def _diagnose_config(config: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    reward = str(config.get("reward_mode", ""))
    train_frac = _float(config.get("ret_g24_kappa_train_frac"), default=1.0)
    if reward == "ReT_garrido2024_train" and train_frac < 0.95:
        flags.append(
            "reward/eval mismatch: train reward discounts kappa capacity cost while cd_sigmoid eval uses full cost"
        )
    if bool(config.get("stochastic_pt")):
        flags.append("stochastic PT enabled: exploratory unless pre-registered for the lane")
    if _float(config.get("risk_frequency_multiplier"), default=1.0) >= 4.0:
        flags.append("war-stress frequency >=4: exploratory, not thesis-faithful primary")
    return flags


def _action_diagnostics(dynamic: dict[str, Any], static: dict[str, Any]) -> dict[str, Any]:
    dyn_s3 = _field(dynamic, "pct_steps_S3")
    sta_s3 = _field(static, "pct_steps_S3")
    dyn_s2 = _field(dynamic, "pct_steps_S2")
    dyn_s1 = _field(dynamic, "pct_steps_S1")
    return {
        "dynamic_pct_steps_S1": dyn_s1,
        "dynamic_pct_steps_S2": dyn_s2,
        "dynamic_pct_steps_S3": dyn_s3,
        "static_pct_steps_S3": sta_s3,
        "s3_overuse_vs_static_pp": dyn_s3 - sta_s3,
        "shift_mix_warning": dyn_s3 > sta_s3 + 20.0,
    }


def diagnose_run(summary_path: Path) -> dict[str, Any]:
    data = _read_json(summary_path)
    summary_rows = list(data.get("policy_summary", []))
    config = dict(data.get("config", {}))
    regimes = sorted({str(row["regime"]) for row in summary_rows})
    config_flags = _diagnose_config(config)
    rows: list[dict[str, Any]] = []
    blockers: list[str] = list(config_flags)
    for regime in regimes:
        dynamic = _dynamic_row(summary_rows, regime)
        best_cd = _best_static(summary_rows, regime, "cd_sigmoid_mean")
        best_excel = _best_static(summary_rows, regime, "mean_ret_excel_formula")
        if dynamic is None or best_cd is None:
            continue
        delta_cd = _field(dynamic, "cd_sigmoid_mean") - _field(best_cd, "cd_sigmoid_mean")
        service = _service_delta(dynamic, best_cd)
        components = _component_delta(dynamic, best_cd)
        action = _action_diagnostics(dynamic, best_cd)
        service_worse = (
            service["flow_fill_rate"] < -0.005
            or service["mean_ret_excel_formula"] < -0.0005
            or service["service_loss_cvar95"] > 0.02
        )
        metric_gaming_risk = delta_cd > 0.0 and service_worse
        if delta_cd < 0.0:
            blockers.append(f"{regime}: PPO below best static on cd_sigmoid_mean ({delta_cd:+.6f})")
        if action["shift_mix_warning"]:
            blockers.append(
                f"{regime}: PPO overuses S3 vs best static by {action['s3_overuse_vs_static_pp']:.1f}pp"
            )
        if metric_gaming_risk:
            blockers.append(
                f"{regime}: C-D win is paired with worse service/Excel metrics; treat as metric-gaming risk"
            )
        row = {
            "regime": regime,
            "dynamic_policy": "ppo_dynamic",
            "best_static_cd_policy": best_cd["policy"],
            "best_static_excel_policy": best_excel["policy"] if best_excel else "",
            "dynamic_cd_sigmoid_mean": _field(dynamic, "cd_sigmoid_mean"),
            "best_static_cd_sigmoid_mean": _field(best_cd, "cd_sigmoid_mean"),
            "delta_cd_sigmoid_mean": delta_cd,
            "dynamic_excel_ret": _field(dynamic, "mean_ret_excel_formula"),
            "best_static_excel_ret_on_cd_static": _field(best_cd, "mean_ret_excel_formula"),
            "delta_excel_ret_vs_cd_static": service["mean_ret_excel_formula"],
            "dynamic_flow_fill_rate": _field(dynamic, "flow_fill_rate"),
            "best_static_flow_fill_rate": _field(best_cd, "flow_fill_rate"),
            "delta_flow_fill_rate": service["flow_fill_rate"],
            "dynamic_service_loss_cvar95": _field(dynamic, "service_loss_cvar95"),
            "best_static_service_loss_cvar95": _field(best_cd, "service_loss_cvar95"),
            "delta_service_loss_cvar95": service["service_loss_cvar95"],
            "dynamic_resource_composite_total": _field(dynamic, "resource_composite_total"),
            "best_static_resource_composite_total": _field(best_cd, "resource_composite_total"),
            "delta_resource_composite_total": _field(dynamic, "resource_composite_total")
            - _field(best_cd, "resource_composite_total"),
            **{f"delta_{key}": value for key, value in components.items()},
            **action,
            "metric_gaming_risk": metric_gaming_risk,
            "passes_minimum_win_gate": delta_cd >= 0.0 and not service_worse,
        }
        rows.append(row)
    cd_deltas = [row["delta_cd_sigmoid_mean"] for row in rows]
    passed = bool(rows) and all(row["passes_minimum_win_gate"] for row in rows)
    verdict = (
        "candidate_win_requires_confirmation"
        if passed
        else "no_defensible_dynamic_win_currently"
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_path": str(summary_path),
        "git": _git_info(),
        "config": config,
        "config_flags": config_flags,
        "diagnostics": rows,
        "aggregate": {
            "mean_delta_cd_sigmoid": _mean(cd_deltas),
            "min_delta_cd_sigmoid": min(cd_deltas) if cd_deltas else float("nan"),
            "max_delta_cd_sigmoid": max(cd_deltas) if cd_deltas else float("nan"),
            "n_regimes": len(rows),
            "passes_all_regime_win_gate": passed,
            "verdict": verdict,
        },
        "blockers": sorted(set(blockers)),
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Dynamic Policy Failure Diagnostic",
        "",
        f"Generated: {payload['generated_at_utc']}",
        f"Input summary: `{payload['summary_path']}`",
        "",
        "## Verdict",
        "",
        f"**{payload['aggregate']['verdict']}**",
        "",
        f"- Mean PPO minus best-static C-D: {payload['aggregate']['mean_delta_cd_sigmoid']:+.6f}",
        f"- Minimum regime delta: {payload['aggregate']['min_delta_cd_sigmoid']:+.6f}",
        f"- Maximum regime delta: {payload['aggregate']['max_delta_cd_sigmoid']:+.6f}",
        "",
        "## Main Blockers",
        "",
    ]
    if payload["blockers"]:
        lines.extend(f"- {blocker}" for blocker in payload["blockers"])
    else:
        lines.append("- None detected by the current gates.")
    lines.extend(["", "## Regime Diagnostics", ""])
    for row in payload["diagnostics"]:
        lines.extend(
            [
                f"### {row['regime']}",
                "",
                f"- Best static C-D: `{row['best_static_cd_policy']}`",
                f"- Delta C-D: {row['delta_cd_sigmoid_mean']:+.6f}",
                f"- Delta Excel ReT vs C-D static: {row['delta_excel_ret_vs_cd_static']:+.6f}",
                f"- Delta flow fill: {row['delta_flow_fill_rate']:+.6f}",
                f"- Delta service CVaR95: {row['delta_service_loss_cvar95']:+.6f}",
                f"- Delta resource composite: {row['delta_resource_composite_total']:+.3f}",
                f"- PPO shift mix: S1 {row['dynamic_pct_steps_S1']:.1f}%, "
                f"S2 {row['dynamic_pct_steps_S2']:.1f}%, "
                f"S3 {row['dynamic_pct_steps_S3']:.1f}%",
                f"- Metric-gaming risk: {row['metric_gaming_risk']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/diagnostics/dynamic_policy_failure"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary_path = (args.summary_json or _latest_summary()).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = diagnose_run(summary_path)
    (output_dir / "dynamic_policy_failure_diagnostic.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(output_dir / "dynamic_policy_failure_diagnostic.csv", payload["diagnostics"])
    _write_report(output_dir / "dynamic_policy_failure_diagnostic.md", payload)
    print(json.dumps({
        "verdict": payload["aggregate"]["verdict"],
        "mean_delta_cd_sigmoid": payload["aggregate"]["mean_delta_cd_sigmoid"],
        "output_dir": str(output_dir),
        "blockers": payload["blockers"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
