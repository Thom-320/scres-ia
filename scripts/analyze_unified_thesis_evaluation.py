#!/usr/bin/env python3
"""Post-process the unified thesis-decision evaluation.

This script treats the scenario (Cf row) as the inferential unit. Replications
are averaged within each (policy, Cf), then paired differences are computed
across the 60 Cf scenarios. This avoids treating the 3 replications per Cf as
independent evidence for paper-level contrasts.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy may be unavailable in minimal envs.
    stats = None


METRICS = {
    "fill": "fill_rate_order_level",
    "ret": "order_level_ret_mean",
    "reward": "reward_total",
}

PURE_THESIS_POLICIES = {
    "L1a_uniform_I0_S1",
    "L1a_uniform_I0_S2",
    "L1a_uniform_I0_S3",
    "L1a_uniform_I168_S1",
    "L1a_uniform_I336_S1",
    "L1a_uniform_I504_S1",
    "L1a_uniform_I672_S1",
    "L1a_uniform_I1344_S1",
}


def source_risk_group(source_cfi: int) -> str:
    if 1 <= source_cfi <= 10:
        return "R1"
    if 11 <= source_cfi <= 20:
        return "R2"
    if 21 <= source_cfi <= 30:
        return "R3"
    return "unknown"


def policy_summary(scenario_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, bucket in scenario_df.groupby("policy"):
        first = bucket.iloc[0]
        rows.append(
            {
                "policy": policy,
                "kind": first["kind"],
                "space": first["space"],
                "scenario_n": int(bucket["cfi"].nunique()),
                "fill_mean": float(bucket[METRICS["fill"]].mean()),
                "ret_mean": float(bucket[METRICS["ret"]].mean()),
                "reward_mean": float(bucket[METRICS["reward"]].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["fill_mean", "ret_mean", "reward_mean"], ascending=False
    )


def choose_best(summary: pd.DataFrame, mask: pd.Series) -> str:
    sub = summary[mask].sort_values(
        ["fill_mean", "ret_mean", "reward_mean"], ascending=False
    )
    if sub.empty:
        raise ValueError("No policy matched selection mask.")
    return str(sub.iloc[0]["policy"])


def bootstrap_ci(
    values: np.ndarray, *, rng: np.random.Generator, draws: int
) -> tuple[float, float]:
    if len(values) == 0:
        return math.nan, math.nan
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).astype(float))


def paired_contrast(
    scenario_df: pd.DataFrame,
    *,
    policy_a: str,
    policy_b: str,
    label: str,
    metric_key: str,
    rng: np.random.Generator,
    bootstrap_draws: int,
) -> dict[str, Any]:
    metric = METRICS[metric_key]
    a = scenario_df[scenario_df["policy"].eq(policy_a)][["cfi", metric]]
    b = scenario_df[scenario_df["policy"].eq(policy_b)][["cfi", metric]]
    merged = a.merge(b, on="cfi", suffixes=("_a", "_b"))
    diff = (merged[f"{metric}_a"] - merged[f"{metric}_b"]).to_numpy(dtype=float)
    ci_low, ci_high = bootstrap_ci(diff, rng=rng, draws=bootstrap_draws)
    wilcoxon_p = math.nan
    paired_t_p = math.nan
    if stats is not None and len(diff) > 0:
        try:
            wilcoxon_p = float(stats.wilcoxon(diff, zero_method="pratt").pvalue)
        except ValueError:
            wilcoxon_p = 1.0
        paired_t_p = float(stats.ttest_1samp(diff, 0.0).pvalue)
    return {
        "contrast": label,
        "policy_a": policy_a,
        "policy_b": policy_b,
        "metric": metric_key,
        "scenario_n": int(len(diff)),
        "mean_delta": float(np.mean(diff)),
        "median_delta": float(np.median(diff)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "wilcoxon_p": wilcoxon_p,
        "paired_t_p": paired_t_p,
        "wins": int(np.sum(diff > 0)),
        "losses": int(np.sum(diff < 0)),
        "ties": int(np.sum(diff == 0)),
    }


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    fill_rows = [
        row
        for row in rows
        if row["metric"] == "fill" and not math.isnan(row["wilcoxon_p"])
    ]
    ordered = sorted(fill_rows, key=lambda row: row["wilcoxon_p"])
    m = len(ordered)
    running = 0.0
    for idx, row in enumerate(ordered):
        adjusted = min(1.0, (m - idx) * row["wilcoxon_p"])
        running = max(running, adjusted)
        row["holm_p_fill_family"] = running
    for row in rows:
        row.setdefault("holm_p_fill_family", math.nan)


def family_breakdown(
    scenario_df: pd.DataFrame, *, policy_a: str, policy_b: str, label: str
) -> list[dict[str, Any]]:
    out = []
    for group_key in ["family", "source_risk_group"]:
        for group_name, group in scenario_df.groupby(group_key):
            a = group[group["policy"].eq(policy_a)][["cfi", *METRICS.values()]]
            b = group[group["policy"].eq(policy_b)][["cfi", *METRICS.values()]]
            merged = a.merge(b, on="cfi", suffixes=("_a", "_b"))
            if merged.empty:
                continue
            row = {
                "contrast": label,
                "group_type": group_key,
                "group": group_name,
                "scenario_n": int(merged["cfi"].nunique()),
            }
            for metric_key, metric in METRICS.items():
                diff = merged[f"{metric}_a"] - merged[f"{metric}_b"]
                row[f"{metric_key}_mean_delta"] = float(diff.mean())
                row[f"{metric_key}_median_delta"] = float(diff.median())
            out.append(row)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 4) -> str:
    if math.isnan(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Unified evaluation directory containing unified_per_scenario.csv.",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260612)
    args = parser.parse_args()

    run_dir = args.run_dir
    df = pd.read_csv(run_dir / "unified_per_scenario.csv")
    df["source_risk_group"] = df["source_cfi"].astype(int).map(source_risk_group)
    scenario_df = (
        df.groupby(
            [
                "policy",
                "kind",
                "space",
                "cfi",
                "source_cfi",
                "family",
                "source_risk_group",
            ]
        )
        .agg({metric: "mean" for metric in METRICS.values()})
        .reset_index()
    )

    summary = policy_summary(scenario_df)
    summary_path = run_dir / "decision_ladder_policy_summary.csv"
    summary.to_csv(summary_path, index=False)

    matched = "garrido_oracle"
    crossed = "L1a_uniform_I504_S3"
    best_pure = choose_best(summary, summary["policy"].isin(PURE_THESIS_POLICIES))
    best_uniform = choose_best(
        summary, summary["kind"].eq("static") & summary["space"].eq("uniform")
    )
    best_per_node = choose_best(
        summary, summary["kind"].eq("static") & summary["space"].eq("per_node")
    )
    best_ppo_e2a = choose_best(
        summary, summary["kind"].eq("ppo") & summary["space"].eq("uniform")
    )
    best_ppo_e2b = choose_best(
        summary, summary["kind"].eq("ppo") & summary["space"].eq("per_node")
    )

    contrast_specs = [
        ("crossed_uniform_I504_S3_minus_matched_DOE", crossed, matched),
        ("crossed_uniform_I504_S3_minus_best_pure_thesis", crossed, best_pure),
        ("best_per_node_minus_best_uniform", best_per_node, best_uniform),
        ("best_PPO_E2a_minus_best_uniform", best_ppo_e2a, best_uniform),
        ("best_PPO_E2b_minus_best_per_node", best_ppo_e2b, best_per_node),
    ]
    rng = np.random.default_rng(args.seed)
    contrast_rows = []
    for label, a_policy, b_policy in contrast_specs:
        for metric_key in METRICS:
            contrast_rows.append(
                paired_contrast(
                    scenario_df,
                    policy_a=a_policy,
                    policy_b=b_policy,
                    label=label,
                    metric_key=metric_key,
                    rng=rng,
                    bootstrap_draws=args.bootstrap_draws,
                )
            )
    holm_adjust(contrast_rows)
    write_csv(run_dir / "decision_ladder_contrasts.csv", contrast_rows)

    family_rows = []
    for label, a_policy, b_policy in contrast_specs:
        family_rows.extend(
            family_breakdown(
                scenario_df,
                policy_a=a_policy,
                policy_b=b_policy,
                label=label,
            )
        )
    write_csv(run_dir / "decision_ladder_family_breakdown.csv", family_rows)

    selected = summary[
        summary["policy"].isin(
            [
                matched,
                best_pure,
                crossed,
                best_uniform,
                best_per_node,
                best_ppo_e2a,
                best_ppo_e2b,
            ]
        )
    ].copy()
    selected["display_policy"] = selected["policy"].replace(
        {"garrido_oracle": "garrido_matched_DOE_baseline"}
    )

    lines = [
        "# Decision-Ladder Statistical Audit",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "Inferential unit: scenario-level paired differences, `n=60`; the 3 replications are averaged within each `(policy, Cf)` first.",
        "",
        "Terminology: `garrido_oracle` is renamed here as `garrido_matched_DOE_baseline`; it is the thesis DOE assignment for each Cf row, not an optimizing oracle.",
        "",
        "## Selected Policies",
        "",
        "| role | policy | fill | ReT | reward |",
        "|---|---|---:|---:|---:|",
    ]
    roles = {
        matched: "matched DOE baseline",
        best_pure: "best pure thesis-isolated static",
        crossed: "crossed uniform I504,S3",
        best_uniform: "best crossed uniform static",
        best_per_node: "best per-node static",
        best_ppo_e2a: "best PPO E2a [6,3]",
        best_ppo_e2b: "best PPO E2b [6,6,6,3]",
    }
    for policy, role in roles.items():
        row = summary[summary["policy"].eq(policy)].iloc[0]
        display = "garrido_matched_DOE_baseline" if policy == matched else policy
        lines.append(
            f"| {role} | `{display}` | {row.fill_mean:.4f} | {row.ret_mean:.4f} | {row.reward_mean:.2f} |"
        )

    lines += [
        "",
        "## Paired Contrasts",
        "",
        "| contrast | metric | mean delta | median delta | 95% bootstrap CI | Wilcoxon p | Holm p (fill only) | wins/losses/ties |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in contrast_rows:
        lines.append(
            "| {contrast} | {metric} | {mean_delta} | {median_delta} | [{ci_low}, {ci_high}] | {p} | {holm} | {wins}/{losses}/{ties} |".format(
                contrast=row["contrast"],
                metric=row["metric"],
                mean_delta=fmt(row["mean_delta"]),
                median_delta=fmt(row["median_delta"]),
                ci_low=fmt(row["ci95_low"]),
                ci_high=fmt(row["ci95_high"]),
                p=fmt(row["wilcoxon_p"], 5),
                holm=fmt(row["holm_p_fill_family"], 5),
                wins=row["wins"],
                losses=row["losses"],
                ties=row["ties"],
            )
        )

    lines += [
        "",
        "## Mechanism Breakdown",
        "",
        "Mean deltas by thesis family and source risk group. Positive values favor the first policy in the contrast.",
        "",
        "| contrast | group type | group | n | fill delta | ReT delta | reward delta |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in family_rows:
        lines.append(
            f"| {row['contrast']} | {row['group_type']} | {row['group']} | {row['scenario_n']} | "
            f"{row['fill_mean_delta']:.4f} | {row['ret_mean_delta']:.4f} | {row['reward_mean_delta']:.2f} |"
        )

    lines += [
        "",
        "## Immediate Reading",
        "",
        "- The primary interaction test is `crossed_uniform_I504_S3_minus_best_pure_thesis`; use that row before claiming an I x S effect.",
        "- `best_per_node_minus_best_uniform` is the granularity test; if its CI straddles zero tightly, per-node should be framed as negligible in this panel.",
        "- PPO rows are short-budget adaptivity tests; keep them secondary unless a confirmatory run changes the sign and interval.",
        "",
    ]
    (run_dir / "decision_ladder_statistical_audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(run_dir / "decision_ladder_statistical_audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
