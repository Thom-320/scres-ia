#!/usr/bin/env python3
"""Analyze Garrido static-fidelity output folders.

The runner writes raw episode rows. This script is a post-processor for local
or Kaggle artifacts: it recomputes the H1/H2/H3 gates from `episode_metrics.csv`
and writes a compact markdown/json analysis beside the source file.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

METRICS = (
    "fill_rate_order_level",
    "order_level_ret_mean",
    "cumulative_disruption_hours",
)
H1_PROFILE_ORDER = ("current", "increased", "severe", "severe_extended")


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else float("nan")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def policy_mean_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["profile"],
                row["family"],
                row["policy"],
                row.get("policy_kind", ""),
            )
        ].append(row)

    out = []
    for (profile, family, policy, policy_kind), bucket in sorted(groups.items()):
        record: dict[str, object] = {
            "profile": profile,
            "family": family,
            "policy": policy,
            "policy_kind": policy_kind,
            "episode_count": len(bucket),
        }
        for metric in METRICS:
            record[f"{metric}_mean"] = mean(float(row[metric]) for row in bucket)
        out.append(record)
    return out


def nested_policy_means(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str, str], dict[str, float]]:
    means = {}
    for row in policy_mean_rows(rows):
        key = (str(row["profile"]), str(row["family"]), str(row["policy"]))
        means[key] = {metric: float(row[f"{metric}_mean"]) for metric in METRICS}
    return means


def scenario_policy_means(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str, str, int], dict[str, float]]:
    groups: dict[tuple[str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["profile"],
                row["family"],
                row["policy"],
                int(row["cfi"]),
            )
        ].append(row)

    out = {}
    for key, bucket in groups.items():
        out[key] = {
            metric: mean(float(row[metric]) for row in bucket) for metric in METRICS
        }
    return out


def h1_gate(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    means = nested_policy_means(rows)
    families = sorted({row["family"] for row in rows})
    result = []
    for family in families:
        values = []
        missing = []
        for profile in H1_PROFILE_ORDER:
            key = (profile, family, "garrido_matched_DOE_baseline")
            if key not in means:
                missing.append(profile)
                continue
            values.append((profile, means[key]))
        if missing:
            result.append(
                {"family": family, "status": "missing", "missing_profiles": missing}
            )
            continue

        fill = [item[1]["fill_rate_order_level"] for item in values]
        ret = [item[1]["order_level_ret_mean"] for item in values]
        disruption = [item[1]["cumulative_disruption_hours"] for item in values]
        result.append(
            {
                "family": family,
                "status": (
                    "passed"
                    if all(a >= b for a, b in zip(fill, fill[1:]))
                    and all(a >= b for a, b in zip(ret, ret[1:]))
                    and all(a <= b for a, b in zip(disruption, disruption[1:]))
                    else "failed"
                ),
                "profiles": list(H1_PROFILE_ORDER),
                "fill": fill,
                "ret": ret,
                "disruption_hours": disruption,
            }
        )
    return result


def contrast_gate(
    rows: list[dict[str, str]],
    *,
    name: str,
    policy_hi: str,
    policy_lo: str,
) -> list[dict[str, object]]:
    scenario_means = scenario_policy_means(rows)
    keys = sorted(
        {(profile, family, cfi) for profile, family, _policy, cfi in scenario_means}
    )
    result = []
    for profile, family, cfi in keys:
        hi = scenario_means.get((profile, family, policy_hi, cfi))
        lo = scenario_means.get((profile, family, policy_lo, cfi))
        if hi is None or lo is None:
            continue
        result.append(
            {
                "gate": name,
                "profile": profile,
                "family": family,
                "cfi": cfi,
                "fill_delta": hi["fill_rate_order_level"] - lo["fill_rate_order_level"],
                "ret_delta": hi["order_level_ret_mean"] - lo["order_level_ret_mean"],
            }
        )
    return result


def summarize_contrast(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["profile"]), str(row["family"]))].append(row)

    out = []
    for (profile, family), bucket in sorted(groups.items()):
        fill = [float(row["fill_delta"]) for row in bucket]
        ret = [float(row["ret_delta"]) for row in bucket]
        out.append(
            {
                "profile": profile,
                "family": family,
                "scenario_count": len(bucket),
                "fill_positive": sum(value > 0.0 for value in fill),
                "fill_delta_mean": mean(fill),
                "ret_positive": sum(value > 0.0 for value in ret),
                "ret_delta_mean": mean(ret),
            }
        )
    return out


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        rendered = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                rendered.append(f"{value:.4f}")
            elif isinstance(value, list):
                rendered.append(format_list_cell(value))
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return lines


def format_list_cell(values: list[object]) -> str:
    """Render compact list values for markdown tables."""
    rendered = []
    for value in values:
        if isinstance(value, float):
            rendered.append(f"{value:.4f}")
        else:
            rendered.append(str(value))
    return " -> ".join(rendered)


def analyze_run(run_dir: Path) -> dict[str, object]:
    csv_path = run_dir / "episode_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    rows = read_rows(csv_path)
    h2_rows = contrast_gate(
        rows,
        name="H2",
        policy_hi="pure_inventory_I672_S1",
        policy_lo="pure_inventory_I0_S1",
    )
    h3_rows = contrast_gate(
        rows,
        name="H3",
        policy_hi="pure_capacity_I0_S3",
        policy_lo="pure_capacity_I0_S1",
    )
    payload = {
        "run_dir": str(run_dir),
        "episode_count": len(rows),
        "profiles": sorted({row["profile"] for row in rows}),
        "families": sorted({row["family"] for row in rows}),
        "policies": sorted({row["policy"] for row in rows}),
        "h1": h1_gate(rows),
        "h2": summarize_contrast(h2_rows),
        "h3": summarize_contrast(h3_rows),
    }
    write_json(run_dir / "fidelity_gate_analysis.json", payload)
    write_markdown(run_dir / "FIDELITY_GATE_ANALYSIS.md", payload)
    return payload


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# Fidelity Gate Analysis",
        "",
        f"Run: `{payload['run_dir']}`",
        f"Episodes: `{payload['episode_count']}`",
        f"Profiles: `{', '.join(payload['profiles'])}`",
        f"Families: `{', '.join(payload['families'])}`",
        "",
        "## H1 Risk Degradation",
        "",
    ]
    lines.extend(
        markdown_table(
            payload["h1"],
            ["family", "status", "profiles", "fill", "ret", "disruption_hours"],
        )
    )
    lines.extend(["", "## H2 Inventory Moderation", ""])
    lines.extend(
        markdown_table(
            payload["h2"],
            [
                "profile",
                "family",
                "scenario_count",
                "fill_positive",
                "fill_delta_mean",
                "ret_positive",
                "ret_delta_mean",
            ],
        )
    )
    lines.extend(["", "## H3 Capacity Moderation", ""])
    lines.extend(
        markdown_table(
            payload["h3"],
            [
                "profile",
                "family",
                "scenario_count",
                "fill_positive",
                "fill_delta_mean",
                "ret_positive",
                "ret_delta_mean",
            ],
        )
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    return parser


def iter_run_dirs(paths: Iterable[Path]) -> list[Path]:
    """Return directories containing an episode_metrics.csv file."""
    run_dirs: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path.is_file() and path.name == "episode_metrics.csv":
            candidate = path.parent
            if candidate not in seen:
                run_dirs.append(candidate)
                seen.add(candidate)
            continue
        if (path / "episode_metrics.csv").exists():
            if path not in seen:
                run_dirs.append(path)
                seen.add(path)
            continue
        for csv_path in sorted(path.rglob("episode_metrics.csv")):
            candidate = csv_path.parent
            if candidate not in seen:
                run_dirs.append(candidate)
                seen.add(candidate)
    return run_dirs


def main() -> int:
    args = build_parser().parse_args()
    run_dirs = iter_run_dirs(args.run_dirs)
    if not run_dirs:
        raise FileNotFoundError("No episode_metrics.csv files found.")
    for run_dir in run_dirs:
        payload = analyze_run(run_dir)
        print(
            f"Wrote {run_dir / 'FIDELITY_GATE_ANALYSIS.md'} "
            f"({payload['episode_count']} episodes)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
