#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CORE_BACKBONE_FIELDS: tuple[str, ...] = (
    "observation_version",
    "frame_stack",
    "year_basis",
    "risk_level",
    "stochastic_pt",
    "step_size_hours",
    "max_steps",
)
COMPARISON_CORE_COLUMNS: tuple[str, ...] = (
    "static_s2_fill_rate_mean",
    "ppo_pct_steps_S1_mean",
    "ppo_pct_steps_S2_mean",
    "ppo_pct_steps_S3_mean",
)


@dataclass(frozen=True)
class BundleRef:
    label: str
    root_dir: Path
    summary_path: Path
    comparison_csv_path: Path | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit benchmark bundles for provenance drift, schema drift, and "
            "cross-run comparability."
        )
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help="Run directories or summary.json paths to audit.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Optional reference run path. Defaults to the first positional run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/audits/benchmark_bundle_audit"),
    )
    return parser


def infer_reward_family(reward_mode: str | None) -> str | None:
    if reward_mode in ("control_v1", "control_v1_pbrs"):
        return "operational_penalty"
    if reward_mode in (
        "ReT_seq_v1",
        "ReT_unified_v1",
        "ReT_garrido2024_raw",
        "ReT_garrido2024",
        "ReT_garrido2024_train",
        "ReT_cd_v1",
        "ReT_cd_sigmoid",
    ):
        return "resilience_index"
    if reward_mode:
        return "unknown"
    return None


def normalize_run_ref(value: str) -> BundleRef:
    raw_path = Path(value).expanduser()
    if raw_path.name == "summary.json":
        summary_path = raw_path
        root_dir = raw_path.parent
    else:
        root_dir = raw_path
        summary_path = root_dir / "summary.json"
    comparison_csv_path = root_dir / "comparison_table.csv"
    return BundleRef(
        label=root_dir.name,
        root_dir=root_dir,
        summary_path=summary_path,
        comparison_csv_path=(
            comparison_csv_path if comparison_csv_path.exists() else None
        ),
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_comparison_rows(
    bundle: BundleRef, summary: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = summary.get("comparison_table", [])
    if rows:
        return list(rows)
    if bundle.comparison_csv_path is None:
        return []
    with bundle.comparison_csv_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float | None:
    if value in (None, "", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_comparison_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return dict(rows[0]) if rows else {}


def build_backbone(
    summary: dict[str, Any], comparison_row: dict[str, Any]
) -> dict[str, Any]:
    config = summary.get("config", {})
    backbone = dict(summary.get("backbone", {}))
    if backbone:
        return backbone
    fallback: dict[str, Any] = {"env_variant": "shift_control"}
    for key in CORE_BACKBONE_FIELDS:
        if key in config and config.get(key) is not None:
            fallback[key] = config.get(key)
        elif key in comparison_row and comparison_row.get(key) not in (None, ""):
            fallback[key] = comparison_row.get(key)
    reward_mode = (
        config.get("reward_mode")
        or comparison_row.get("reward_mode")
        or summary.get("reward_contract", {}).get("reward_mode")
    )
    if reward_mode is not None:
        fallback["reward_mode"] = reward_mode
    return fallback


def extract_metric_contract(summary: dict[str, Any]) -> dict[str, Any]:
    return dict(summary.get("metric_contract", {}))


def extract_reward_contract(
    summary: dict[str, Any],
    backbone: dict[str, Any],
    comparison_row: dict[str, Any],
) -> dict[str, Any]:
    reward_contract = dict(summary.get("reward_contract", {}))
    reward_mode = (
        reward_contract.get("reward_mode")
        or backbone.get("reward_mode")
        or summary.get("config", {}).get("reward_mode")
        or comparison_row.get("reward_mode")
    )
    if reward_mode is not None and "reward_mode" not in reward_contract:
        reward_contract["reward_mode"] = reward_mode
    if reward_contract.get("reward_family") is None:
        reward_contract["reward_family"] = infer_reward_family(reward_mode)
    if "cross_mode_reward_comparison_allowed" not in reward_contract:
        reward_contract["cross_mode_reward_comparison_allowed"] = False
    return reward_contract


def extract_run_metrics(comparison_row: dict[str, Any]) -> dict[str, Any]:
    learned_fill = as_float(
        comparison_row.get(
            "learned_fill_rate_mean", comparison_row.get("ppo_fill_rate_mean")
        )
    )
    learned_reward = as_float(
        comparison_row.get("learned_reward_mean", comparison_row.get("ppo_reward_mean"))
    )
    learned_backorder = as_float(
        comparison_row.get(
            "learned_backorder_rate_mean",
            comparison_row.get("ppo_backorder_rate_mean"),
        )
    )
    learned_order_ret = as_float(
        comparison_row.get(
            "learned_order_level_ret_mean",
            comparison_row.get("ppo_order_level_ret_mean"),
        )
    )
    return {
        "learned_policy": comparison_row.get("learned_policy"),
        "learned_reward_mean": learned_reward,
        "learned_fill_rate_mean": learned_fill,
        "learned_backorder_rate_mean": learned_backorder,
        "learned_order_level_ret_mean": learned_order_ret,
        "static_s2_fill_rate_mean": as_float(
            comparison_row.get("static_s2_fill_rate_mean")
        ),
        "static_s2_backorder_rate_mean": as_float(
            comparison_row.get("static_s2_backorder_rate_mean")
        ),
        "ppo_pct_steps_S1_mean": as_float(comparison_row.get("ppo_pct_steps_S1_mean")),
        "ppo_pct_steps_S2_mean": as_float(comparison_row.get("ppo_pct_steps_S2_mean")),
        "ppo_pct_steps_S3_mean": as_float(comparison_row.get("ppo_pct_steps_S3_mean")),
    }


def audit_bundle(bundle: BundleRef) -> dict[str, Any]:
    reasons: list[str] = []
    if not bundle.summary_path.exists():
        return {
            "label": bundle.label,
            "run_dir": str(bundle.root_dir.resolve()),
            "summary_path": str(bundle.summary_path.resolve()),
            "summary_exists": False,
            "audit_status": "invalid",
            "reasons": ["missing_summary_json"],
        }

    summary = load_json(bundle.summary_path)
    comparison_rows = load_comparison_rows(bundle, summary)
    comparison_row = first_comparison_row(comparison_rows)
    backbone = build_backbone(summary, comparison_row)
    metric_contract = extract_metric_contract(summary)
    reward_contract = extract_reward_contract(summary, backbone, comparison_row)
    benchmark_metadata = dict(summary.get("benchmark_metadata", {}))

    if not summary.get("backbone"):
        reasons.append("missing_backbone_metadata")
    if not metric_contract:
        reasons.append("missing_metric_contract")
    if not benchmark_metadata:
        reasons.append("missing_benchmark_metadata")
    if not comparison_rows:
        reasons.append("missing_comparison_table")
    else:
        missing_columns = [
            column for column in COMPARISON_CORE_COLUMNS if column not in comparison_row
        ]
        if missing_columns:
            reasons.append(
                "legacy_comparison_schema:" + ",".join(sorted(missing_columns))
            )

    audit_status = "auditable" if not reasons else "historical_artifact"
    return {
        "label": bundle.label,
        "run_dir": str(bundle.root_dir.resolve()),
        "summary_path": str(bundle.summary_path.resolve()),
        "summary_exists": True,
        "comparison_table_exists": bool(comparison_rows),
        "comparison_column_count": len(comparison_row) if comparison_row else 0,
        "audit_status": audit_status,
        "backbone": backbone,
        "metric_contract": metric_contract,
        "reward_contract": reward_contract,
        "benchmark_metadata": benchmark_metadata,
        "metrics": extract_run_metrics(comparison_row),
        "reasons": reasons,
    }


def compare_to_reference(
    audited_row: dict[str, Any],
    reference_row: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    audited_backbone = audited_row.get("backbone", {})
    reference_backbone = reference_row.get("backbone", {})

    env_backbone_match = True
    for key in CORE_BACKBONE_FIELDS:
        if audited_backbone.get(key) != reference_backbone.get(key):
            env_backbone_match = False
            reasons.append(
                f"backbone_mismatch:{key}={audited_backbone.get(key)!r}!= {reference_backbone.get(key)!r}"
            )

    metric_contract_match = (
        audited_row.get("metric_contract", {})
        == reference_row.get("metric_contract", {})
        and bool(audited_row.get("metric_contract"))
        and bool(reference_row.get("metric_contract"))
    )
    if not metric_contract_match:
        reasons.append("metric_contract_mismatch_or_missing")

    audited_reward = audited_row.get("reward_contract", {})
    reference_reward = reference_row.get("reward_contract", {})
    raw_reward_comparable = (
        audited_reward.get("reward_mode") == reference_reward.get("reward_mode")
        and audited_reward.get("reward_family") == reference_reward.get("reward_family")
        and audited_row.get("audit_status") == "auditable"
        and reference_row.get("audit_status") == "auditable"
    )
    if not raw_reward_comparable:
        reasons.append("raw_reward_not_comparable")

    service_metrics_comparable = (
        audited_row.get("audit_status") == "auditable"
        and reference_row.get("audit_status") == "auditable"
        and env_backbone_match
        and metric_contract_match
    )
    if not service_metrics_comparable:
        reasons.append("service_metrics_not_comparable")

    metrics = audited_row.get("metrics", {})
    reference_metrics = reference_row.get("metrics", {})
    static_s2_fill_delta = None
    if (
        metrics.get("static_s2_fill_rate_mean") is not None
        and reference_metrics.get("static_s2_fill_rate_mean") is not None
    ):
        static_s2_fill_delta = (
            metrics["static_s2_fill_rate_mean"]
            - reference_metrics["static_s2_fill_rate_mean"]
        )
        if abs(static_s2_fill_delta) >= 0.02:
            reasons.append("large_static_s2_fill_delta")

    return {
        "label": audited_row["label"],
        "reference_label": reference_row["label"],
        "audit_status": audited_row.get("audit_status"),
        "env_backbone_match": env_backbone_match,
        "metric_contract_match": metric_contract_match,
        "service_metrics_comparable": service_metrics_comparable,
        "raw_reward_comparable": raw_reward_comparable,
        "static_s2_fill_delta_vs_reference": static_s2_fill_delta,
        "reasons": reasons,
    }


def build_markdown_report(
    *,
    audited_rows: list[dict[str, Any]],
    compatibility_rows: list[dict[str, Any]],
    reference_label: str,
) -> str:
    lines = [
        "# Benchmark Bundle Audit",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Reference run: `{reference_label}`",
        "",
        "## Bundle status",
        "",
        "| Label | Audit status | Reward mode | Reward family | obs | fs | risk | static_s2 fill | Reasons |",
        "| --- | --- | --- | --- | --- | ---: | --- | ---: | --- |",
    ]
    for row in audited_rows:
        backbone = row.get("backbone", {})
        reward_contract = row.get("reward_contract", {})
        metrics = row.get("metrics", {})
        lines.append(
            f"| `{row['label']}` | `{row['audit_status']}` | "
            f"`{reward_contract.get('reward_mode')}` | "
            f"`{reward_contract.get('reward_family')}` | "
            f"`{backbone.get('observation_version')}` | "
            f"{backbone.get('frame_stack')} | "
            f"`{backbone.get('risk_level')}` | "
            f"{metrics.get('static_s2_fill_rate_mean') if metrics.get('static_s2_fill_rate_mean') is not None else float('nan'):.3f} | "
            f"{', '.join(row.get('reasons', [])) or 'ok'} |"
        )

    lines.extend(
        [
            "",
            "## Compatibility vs reference",
            "",
            "| Label | Service-metric comparable | Raw-reward comparable | Static_s2 delta vs ref | Reasons |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for row in compatibility_rows:
        delta = row.get("static_s2_fill_delta_vs_reference")
        delta_str = f"{delta:+.3f}" if delta is not None else "n/a"
        lines.append(
            f"| `{row['label']}` | `{row['service_metrics_comparable']}` | "
            f"`{row['raw_reward_comparable']}` | {delta_str} | "
            f"{', '.join(row['reasons']) or 'ok'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle_refs = [normalize_run_ref(run) for run in args.runs]
    reference_ref = (
        normalize_run_ref(args.reference) if args.reference else bundle_refs[0]
    )

    audited_rows = [audit_bundle(bundle) for bundle in bundle_refs]
    audited_by_label = {row["label"]: row for row in audited_rows}
    reference_row = audited_by_label.get(reference_ref.label)
    if reference_row is None:
        reference_row = audit_bundle(reference_ref)
        audited_rows.insert(0, reference_row)
        audited_by_label[reference_row["label"]] = reference_row

    compatibility_rows = [
        compare_to_reference(row, reference_row) for row in audited_rows
    ]

    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_label": reference_row["label"],
        "audited_rows": audited_rows,
        "compatibility_rows": compatibility_rows,
    }

    json_path = args.output_dir / "summary.json"
    md_path = args.output_dir / "audit_report.md"
    csv_path = args.output_dir / "compatibility.csv"

    json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    md_path.write_text(
        build_markdown_report(
            audited_rows=audited_rows,
            compatibility_rows=compatibility_rows,
            reference_label=reference_row["label"],
        ),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "label",
            "reference_label",
            "audit_status",
            "env_backbone_match",
            "metric_contract_match",
            "service_metrics_comparable",
            "raw_reward_comparable",
            "static_s2_fill_delta_vs_reference",
            "reasons",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in compatibility_rows:
            csv_row = dict(row)
            csv_row["reasons"] = ";".join(row["reasons"])
            writer.writerow(csv_row)


if __name__ == "__main__":
    main()
