#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.control_reward_seed_inference import (
    exact_sign_flip_pvalue,
    paired_bootstrap_ci,
    paired_cohens_d,
)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RUN_DIR = REPO / "outputs/track_b_benchmarks/track_b_ret_seq_k020_500k_rerun1"
DEFAULT_AUDIT_DIR = DEFAULT_RUN_DIR / "posthoc_resilience_audit"
DEFAULT_TRACK_A_DIR = REPO / "outputs/paper_benchmarks/paper_ret_seq_k020_500k"
DEFAULT_OUTPUT_DIR = REPO / "outputs/track_b_benchmarks/track_b_results_package"
PAIRWISE_METRICS = (
    ("fill_rate", "higher"),
    ("backorder_rate", "lower"),
    ("order_level_ret_mean", "higher"),
    ("terminal_rolling_fill_rate_4w", "higher"),
    ("ret_thesis_corrected_total", "higher"),
    ("ret_unified_total", "higher"),
)
POLICY_ORDER = ("ppo", "s2_d1.00", "s3_d2.00")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the final Track B statistical package for Results and Discussion."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=DEFAULT_AUDIT_DIR,
        help="Posthoc resilience audit bundle. Falls back to run-dir if missing.",
    )
    parser.add_argument("--track-a-dir", type=Path, default=DEFAULT_TRACK_A_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260331)
    return parser


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def policy_row_lookup(rows: list[dict[str, str]], policy: str) -> dict[str, str]:
    for row in rows:
        if row["policy"] == policy:
            return row
    raise KeyError(f"Policy {policy!r} not found")


def pick_source_dir(run_dir: Path, audit_dir: Path) -> Path:
    if (audit_dir / "seed_metrics.csv").exists() and (
        audit_dir / "policy_summary.csv"
    ).exists():
        return audit_dir
    return run_dir


def seed_metric_values(
    rows: list[dict[str, str]], *, policy: str, metric: str
) -> dict[int, float]:
    values: dict[int, float] = {}
    field = f"{metric}_mean"
    for row in rows:
        if row["policy"] != policy:
            continue
        values[int(row["seed"])] = float(row[field])
    return values


def pairwise_metric_stats(
    seed_rows: list[dict[str, str]],
    *,
    comparator: str,
    metric: str,
    goal: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    ppo_values = seed_metric_values(seed_rows, policy="ppo", metric=metric)
    cmp_values = seed_metric_values(seed_rows, policy=comparator, metric=metric)
    shared_seeds = sorted(set(ppo_values) & set(cmp_values))
    if not shared_seeds:
        raise ValueError(f"No shared seeds for metric={metric} comparator={comparator}")

    ppo_arr = np.asarray([ppo_values[s] for s in shared_seeds], dtype=np.float64)
    cmp_arr = np.asarray([cmp_values[s] for s in shared_seeds], dtype=np.float64)
    diffs = ppo_arr - cmp_arr
    ci_low, ci_high = paired_bootstrap_ci(diffs, n_samples=bootstrap_samples, rng=rng)
    better_diffs = diffs if goal == "higher" else -diffs
    ppo_wins = int(np.sum(better_diffs > 0.0))

    return {
        "comparator_policy": comparator,
        "metric": metric,
        "goal": goal,
        "seed_count": len(shared_seeds),
        "shared_seeds": ",".join(str(seed) for seed in shared_seeds),
        "ppo_mean": float(ppo_arr.mean()),
        "comparator_mean": float(cmp_arr.mean()),
        "ppo_minus_comparator": float(diffs.mean()),
        "improvement_toward_goal": float(better_diffs.mean()),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "sign_flip_pvalue": float(exact_sign_flip_pvalue(diffs)),
        "cohens_d": float(paired_cohens_d(diffs)),
        "ppo_wins": ppo_wins,
        "comparator_wins": len(shared_seeds) - ppo_wins,
    }


def build_policy_overview(policy_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    overview: list[dict[str, Any]] = []
    for policy in POLICY_ORDER:
        row = policy_row_lookup(policy_rows, policy)
        overview.append(
            {
                "policy": policy,
                "reward_total_mean": float(row["reward_total_mean"]),
                "reward_total_ci95_low": float(row["reward_total_ci95_low"]),
                "reward_total_ci95_high": float(row["reward_total_ci95_high"]),
                "fill_rate_mean": float(row["fill_rate_mean"]),
                "fill_rate_ci95_low": float(row["fill_rate_ci95_low"]),
                "fill_rate_ci95_high": float(row["fill_rate_ci95_high"]),
                "backorder_rate_mean": float(row["backorder_rate_mean"]),
                "order_level_ret_mean": float(row["order_level_ret_mean_mean"]),
                "ret_thesis_corrected_total_mean": float(
                    row["ret_thesis_corrected_total_mean"]
                ),
                "ret_unified_total_mean": float(row["ret_unified_total_mean"]),
                "rolling_fill_rate_4w_mean": float(
                    row["terminal_rolling_fill_rate_4w_mean"]
                ),
                "pct_steps_S1_mean": float(row["pct_steps_S1_mean"]),
                "pct_steps_S2_mean": float(row["pct_steps_S2_mean"]),
                "pct_steps_S3_mean": float(row["pct_steps_S3_mean"]),
                "order_case_fill_rate_share_mean": float(
                    row["order_case_fill_rate_share_mean"]
                ),
                "order_case_autotomy_share_mean": float(
                    row["order_case_autotomy_share_mean"]
                ),
                "order_case_recovery_share_mean": float(
                    row["order_case_recovery_share_mean"]
                ),
                "order_case_non_recovery_share_mean": float(
                    row["order_case_non_recovery_share_mean"]
                ),
                "order_case_unfulfilled_share_mean": float(
                    row["order_case_unfulfilled_share_mean"]
                ),
            }
        )
    return overview


def build_track_a_vs_track_b(
    track_a_dir: Path, track_b_source_dir: Path, track_b_run_dir: Path
) -> list[dict[str, Any]]:
    track_a_row = load_csv_rows(track_a_dir / "comparison_table.csv")[0]
    track_b_comparison_path = track_b_source_dir / "comparison_table.csv"
    if not track_b_comparison_path.exists():
        track_b_comparison_path = track_b_run_dir / "comparison_table.csv"
    track_b_row = load_csv_rows(track_b_comparison_path)[0]
    return [
        {
            "track": "Track A",
            "env_family": "v1/thesis/upstream-only",
            "learned_policy": "ppo",
            "baseline_policy": "static_s2",
            "fill_rate_mean": float(track_a_row["ppo_fill_rate_mean"]),
            "baseline_fill_rate_mean": float(track_a_row["static_s2_fill_rate_mean"]),
            "fill_gap_vs_baseline_pp": 100.0
            * (
                float(track_a_row["ppo_fill_rate_mean"])
                - float(track_a_row["static_s2_fill_rate_mean"])
            ),
            "ret_thesis_corrected_mean": float(
                track_a_row["ppo_ret_thesis_corrected_total_mean"]
            ),
            "note": "RL fails when downstream bottleneck is not controllable.",
        },
        {
            "track": "Track B",
            "env_family": "v7/thesis/downstream-control",
            "learned_policy": "ppo",
            "baseline_policy": "s2_d1.00",
            "fill_rate_mean": float(track_b_row["ppo_fill_rate_mean"]),
            "baseline_fill_rate_mean": float(track_b_row["baseline_fill_rate_mean"]),
            "fill_gap_vs_baseline_pp": float(
                track_b_row["ppo_fill_gap_vs_baseline_pp"]
            ),
            "ret_thesis_corrected_mean": float(
                track_b_row.get(
                    "ppo_ret_thesis_corrected_mean",
                    track_b_row.get("ppo_ret_thesis_corrected_total_mean"),
                )
            ),
            "note": "RL wins once Op10/Op12 become controllable.",
        },
    ]


def render_markdown(
    *,
    run_summary: dict[str, Any],
    policy_overview: list[dict[str, Any]],
    pairwise_rows: list[dict[str, Any]],
    track_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Track B Results Package",
        "",
        "## Frozen Run",
        "",
        f"- Run dir: `{run_summary['artifacts']['summary_json']}`",
        f"- Reward mode: `{run_summary['config']['reward_mode']}`",
        f"- Action contract: `{run_summary['config']['action_contract']}`",
        f"- Observation version: `{run_summary['config']['observation_version']}`",
        f"- Risk level: `{run_summary['config']['risk_level']}`",
        "",
        "## Policy Overview",
        "",
        "| Policy | Reward | Fill | Backorder | Order-level ReT | ReT corrected | ReT unified | Rolling fill 4w | Shift mix |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in policy_overview:
        shift_mix = (
            f"{row['pct_steps_S1_mean']:.1f}/"
            f"{row['pct_steps_S2_mean']:.1f}/"
            f"{row['pct_steps_S3_mean']:.1f}"
        )
        lines.append(
            f"| {row['policy']} | {row['reward_total_mean']:.2f} | {row['fill_rate_mean']:.5f} | "
            f"{row['backorder_rate_mean']:.5f} | {row['order_level_ret_mean']:.4f} | "
            f"{row['ret_thesis_corrected_total_mean']:.2f} | {row['ret_unified_total_mean']:.2f} | "
            f"{row['rolling_fill_rate_4w_mean']:.4f} | {shift_mix} |"
        )

    lines.extend(
        [
            "",
            "## Pairwise Seed-Level Statistics",
            "",
            "| Comparator | Metric | Goal | PPO mean | Comparator mean | PPO - comparator | CI95 | Sign-flip p | Cohen d | PPO wins |",
            "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in pairwise_rows:
        lines.append(
            f"| {row['comparator_policy']} | {row['metric']} | {row['goal']} | "
            f"{row['ppo_mean']:.5f} | {row['comparator_mean']:.5f} | "
            f"{row['ppo_minus_comparator']:+.5f} | "
            f"[{row['ci95_low']:+.5f}, {row['ci95_high']:+.5f}] | "
            f"{row['sign_flip_pvalue']:.4f} | {row['cohens_d']:+.2f} | "
            f"{row['ppo_wins']}/{row['seed_count']} |"
        )

    lines.extend(
        [
            "",
            "## Track A vs Track B",
            "",
            "| Track | Env family | PPO fill | Baseline fill | Gap vs baseline | ReT corrected | Note |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in track_rows:
        lines.append(
            f"| {row['track']} | {row['env_family']} | {row['fill_rate_mean']:.5f} | "
            f"{row['baseline_fill_rate_mean']:.5f} | {row['fill_gap_vs_baseline_pp']:+.2f} pp | "
            f"{row['ret_thesis_corrected_mean']:.2f} | {row['note']} |"
        )

    ppo = next(row for row in policy_overview if row["policy"] == "ppo")
    s2 = next(row for row in policy_overview if row["policy"] == "s2_d1.00")
    best_static = next(row for row in policy_overview if row["policy"] == "s3_d2.00")
    lines.extend(
        [
            "",
            "## Results Draft",
            "",
            (
                "Under the frozen Track B contract, PPO learned a policy that outperformed both the "
                "neutral static baseline and the strongest static comparator. PPO reached "
                f"`fill_rate={ppo['fill_rate_mean']:.5f}` versus `s2_d1.00={s2['fill_rate_mean']:.5f}` "
                f"and `s3_d2.00={best_static['fill_rate_mean']:.5f}`. The same policy improved "
                f"`order_level_ret_mean` to `{ppo['order_level_ret_mean']:.4f}`, compared with "
                f"`{s2['order_level_ret_mean']:.4f}` and `{best_static['order_level_ret_mean']:.4f}` "
                "for the static policies."
            ),
            "",
            "## Discussion Draft",
            "",
            (
                "Track A and Track B together support a structural interpretation rather than a pure "
                "algorithmic one. In Track A, PPO and RecurrentPPO failed to beat `static_s2` despite "
                "multiple reward formulations. In Track B, the same PPO family became decisively better "
                "once the action contract exposed downstream control at `Op10/Op12`. The contrast "
                "supports the claim that RL performance in this MFSC benchmark depends critically on "
                "whether the agent can intervene on the active bottleneck."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    source_dir = pick_source_dir(args.run_dir, args.audit_dir)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_summary = load_json(source_dir / "summary.json")
    seed_rows = load_csv_rows(source_dir / "seed_metrics.csv")
    policy_rows = load_csv_rows(source_dir / "policy_summary.csv")

    rng = np.random.default_rng(args.seed)
    pairwise_rows: list[dict[str, Any]] = []
    for comparator in ("s2_d1.00", "s3_d2.00"):
        for metric, goal in PAIRWISE_METRICS:
            pairwise_rows.append(
                pairwise_metric_stats(
                    seed_rows,
                    comparator=comparator,
                    metric=metric,
                    goal=goal,
                    bootstrap_samples=args.bootstrap_samples,
                    rng=rng,
                )
            )

    policy_overview = build_policy_overview(policy_rows)
    track_rows = build_track_a_vs_track_b(args.track_a_dir, source_dir, args.run_dir)

    save_csv(output_dir / "policy_overview.csv", policy_overview)
    save_csv(output_dir / "pairwise_statistics.csv", pairwise_rows)
    save_csv(output_dir / "track_a_vs_track_b.csv", track_rows)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir.resolve()),
        "run_dir": str(args.run_dir.resolve()),
        "audit_dir": str(args.audit_dir.resolve()),
        "track_a_dir": str(args.track_a_dir.resolve()),
        "bootstrap_samples": args.bootstrap_samples,
        "policy_overview": policy_overview,
        "pairwise_statistics": pairwise_rows,
        "track_a_vs_track_b": track_rows,
    }
    (output_dir / "results_discussion_package.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (output_dir / "results_discussion_package.md").write_text(
        render_markdown(
            run_summary=run_summary,
            policy_overview=policy_overview,
            pairwise_rows=pairwise_rows,
            track_rows=track_rows,
        ),
        encoding="utf-8",
    )
    print(f"Wrote Track B results package to {output_dir}")


if __name__ == "__main__":
    main()
