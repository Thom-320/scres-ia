#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DEFAULT_RUN_DIRS = (
    Path("outputs/paper_benchmarks/paper_ret_seq_k020_500k"),
    Path("outputs/paper_benchmarks/paper_ret_seq_k010_500k"),
    Path("outputs/paper_benchmarks/paper_control_v1_500k"),
)
DEFAULT_SECONDARY_COMPARATOR = Path("outputs/benchmarks/final_ret_seq_v1_500k")
DEFAULT_OUTPUT_DIR = Path("outputs/paper_benchmarks/family_a_summary")

PRIMARY_FIELDS = [
    "label",
    "reward_mode",
    "reward_family",
    "ret_seq_kappa",
    "git_commit",
    "seed_count",
    "eval_episodes",
    "observation_version",
    "year_basis",
    "risk_level",
    "fill_rate_mean",
    "static_s2_fill_rate_mean",
    "delta_fill_vs_static_s2",
    "backorder_rate_mean",
    "static_s2_backorder_rate_mean",
    "delta_backorder_vs_static_s2",
    "order_level_ret_mean",
    "static_s2_order_level_ret_mean",
    "delta_order_level_ret_vs_static_s2",
    "reward_total_mean",
    "static_s2_reward_total_mean",
    "ppo_pct_steps_S1_mean",
    "ppo_pct_steps_S2_mean",
    "ppo_pct_steps_S3_mean",
]
SEVERE_FIELDS = [
    "label",
    "severe_available",
    "reward_mode",
    "fill_rate_mean",
    "static_s2_fill_rate_mean",
    "static_s3_fill_rate_mean",
    "delta_fill_vs_static_s2",
    "delta_fill_vs_static_s3",
    "backorder_rate_mean",
    "static_s2_backorder_rate_mean",
    "static_s3_backorder_rate_mean",
    "order_level_ret_mean",
    "static_s2_order_level_ret_mean",
    "static_s3_order_level_ret_mean",
    "reward_total_mean",
    "static_s2_reward_total_mean",
    "static_s3_reward_total_mean",
    "ppo_pct_steps_S1_mean",
    "ppo_pct_steps_S2_mean",
    "ppo_pct_steps_S3_mean",
]
COMPARATOR_FIELDS = [
    "label",
    "role",
    "reward_mode",
    "git_commit",
    "seed_count",
    "eval_episodes",
    "observation_version",
    "year_basis",
    "risk_level",
    "service_metrics_comparable_to_primary",
    "raw_reward_comparable_to_primary",
    "notes",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the Family A summary bundle from the current auditable "
            "paper-facing benchmark runs."
        )
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        type=Path,
        default=list(DEFAULT_RUN_DIRS),
        help="Primary Family A bundle directories.",
    )
    parser.add_argument(
        "--secondary-comparator",
        type=Path,
        default=DEFAULT_SECONDARY_COMPARATOR,
        help="Optional auditable comparator kept outside the thesis-basis core family.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the Family A summary bundle will be written.",
    )
    return parser


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def policy_row(
    rows: list[dict[str, str]],
    *,
    phase: str,
    policy: str,
) -> dict[str, str] | None:
    for row in rows:
        if row.get("phase") == phase and row.get("policy") == policy:
            return row
    return None


def as_float(row: dict[str, str] | None, key: str) -> float | None:
    if row is None:
        return None
    value = row.get(key)
    if value in (None, "", "nan"):
        return None
    return float(value)


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
    return None


def build_primary_row(run_dir: Path) -> dict[str, Any]:
    summary = load_json(run_dir / "summary.json")
    policy_rows = load_csv_rows(run_dir / "policy_summary.csv")
    config = summary.get("config", {})
    backbone = summary.get("backbone", {})
    reward_contract = summary.get("reward_contract", {})
    algo = str(config["algo"])
    learned = policy_row(policy_rows, phase=f"{algo}_eval", policy=algo)
    static_s2 = policy_row(policy_rows, phase="static_screen", policy="static_s2")
    if learned is None or static_s2 is None:
        raise ValueError(f"Missing primary comparison rows in {run_dir}")

    reward_mode = str(config.get("reward_mode", "unknown"))
    reward_family = reward_contract.get("reward_family") or infer_reward_family(
        reward_mode
    )
    fill_rate_mean = as_float(learned, "fill_rate_mean")
    static_fill_rate_mean = as_float(static_s2, "fill_rate_mean")
    backorder_rate_mean = as_float(learned, "backorder_rate_mean")
    static_backorder_rate_mean = as_float(static_s2, "backorder_rate_mean")
    order_level_ret_mean = as_float(learned, "order_level_ret_mean_mean")
    static_order_level_ret_mean = as_float(static_s2, "order_level_ret_mean_mean")

    return {
        "label": run_dir.name,
        "reward_mode": reward_mode,
        "reward_family": reward_family,
        "ret_seq_kappa": config.get("ret_seq_kappa"),
        "git_commit": backbone.get("git_commit"),
        "seed_count": len(config.get("seeds", [])),
        "eval_episodes": config.get("eval_episodes"),
        "observation_version": backbone.get("observation_version"),
        "year_basis": backbone.get("year_basis"),
        "risk_level": backbone.get("risk_level"),
        "fill_rate_mean": fill_rate_mean,
        "static_s2_fill_rate_mean": static_fill_rate_mean,
        "delta_fill_vs_static_s2": (
            fill_rate_mean - static_fill_rate_mean
            if fill_rate_mean is not None and static_fill_rate_mean is not None
            else None
        ),
        "backorder_rate_mean": backorder_rate_mean,
        "static_s2_backorder_rate_mean": static_backorder_rate_mean,
        "delta_backorder_vs_static_s2": (
            backorder_rate_mean - static_backorder_rate_mean
            if backorder_rate_mean is not None
            and static_backorder_rate_mean is not None
            else None
        ),
        "order_level_ret_mean": order_level_ret_mean,
        "static_s2_order_level_ret_mean": static_order_level_ret_mean,
        "delta_order_level_ret_vs_static_s2": (
            order_level_ret_mean - static_order_level_ret_mean
            if order_level_ret_mean is not None
            and static_order_level_ret_mean is not None
            else None
        ),
        "reward_total_mean": as_float(learned, "reward_total_mean"),
        "static_s2_reward_total_mean": as_float(static_s2, "reward_total_mean"),
        "ppo_pct_steps_S1_mean": as_float(learned, "pct_steps_S1_mean"),
        "ppo_pct_steps_S2_mean": as_float(learned, "pct_steps_S2_mean"),
        "ppo_pct_steps_S3_mean": as_float(learned, "pct_steps_S3_mean"),
    }


def build_severe_row(run_dir: Path) -> dict[str, Any]:
    summary = load_json(run_dir / "summary.json")
    policy_rows = load_csv_rows(run_dir / "policy_summary.csv")
    config = summary.get("config", {})
    reward_mode = str(config.get("reward_mode", "unknown"))
    algo = str(config["algo"])
    learned = policy_row(policy_rows, phase="cross_eval_severe", policy=algo)
    static_s2 = policy_row(policy_rows, phase="cross_eval_severe", policy="static_s2")
    static_s3 = policy_row(policy_rows, phase="cross_eval_severe", policy="static_s3")

    if learned is None:
        return {
            "label": run_dir.name,
            "severe_available": False,
            "reward_mode": reward_mode,
            "fill_rate_mean": None,
            "static_s2_fill_rate_mean": None,
            "static_s3_fill_rate_mean": None,
            "delta_fill_vs_static_s2": None,
            "delta_fill_vs_static_s3": None,
            "backorder_rate_mean": None,
            "static_s2_backorder_rate_mean": None,
            "static_s3_backorder_rate_mean": None,
            "order_level_ret_mean": None,
            "static_s2_order_level_ret_mean": None,
            "static_s3_order_level_ret_mean": None,
            "reward_total_mean": None,
            "static_s2_reward_total_mean": None,
            "static_s3_reward_total_mean": None,
            "ppo_pct_steps_S1_mean": None,
            "ppo_pct_steps_S2_mean": None,
            "ppo_pct_steps_S3_mean": None,
        }

    fill_rate_mean = as_float(learned, "fill_rate_mean")
    static_s2_fill_rate_mean = as_float(static_s2, "fill_rate_mean")
    static_s3_fill_rate_mean = as_float(static_s3, "fill_rate_mean")
    backorder_rate_mean = as_float(learned, "backorder_rate_mean")
    static_s2_backorder_rate_mean = as_float(static_s2, "backorder_rate_mean")
    static_s3_backorder_rate_mean = as_float(static_s3, "backorder_rate_mean")
    order_level_ret_mean = as_float(learned, "order_level_ret_mean_mean")
    static_s2_order_level_ret_mean = as_float(static_s2, "order_level_ret_mean_mean")
    static_s3_order_level_ret_mean = as_float(static_s3, "order_level_ret_mean_mean")

    return {
        "label": run_dir.name,
        "severe_available": True,
        "reward_mode": reward_mode,
        "fill_rate_mean": fill_rate_mean,
        "static_s2_fill_rate_mean": static_s2_fill_rate_mean,
        "static_s3_fill_rate_mean": static_s3_fill_rate_mean,
        "delta_fill_vs_static_s2": (
            fill_rate_mean - static_s2_fill_rate_mean
            if fill_rate_mean is not None and static_s2_fill_rate_mean is not None
            else None
        ),
        "delta_fill_vs_static_s3": (
            fill_rate_mean - static_s3_fill_rate_mean
            if fill_rate_mean is not None and static_s3_fill_rate_mean is not None
            else None
        ),
        "backorder_rate_mean": backorder_rate_mean,
        "static_s2_backorder_rate_mean": static_s2_backorder_rate_mean,
        "static_s3_backorder_rate_mean": static_s3_backorder_rate_mean,
        "order_level_ret_mean": order_level_ret_mean,
        "static_s2_order_level_ret_mean": static_s2_order_level_ret_mean,
        "static_s3_order_level_ret_mean": static_s3_order_level_ret_mean,
        "reward_total_mean": as_float(learned, "reward_total_mean"),
        "static_s2_reward_total_mean": as_float(static_s2, "reward_total_mean"),
        "static_s3_reward_total_mean": as_float(static_s3, "reward_total_mean"),
        "ppo_pct_steps_S1_mean": as_float(learned, "pct_steps_S1_mean"),
        "ppo_pct_steps_S2_mean": as_float(learned, "pct_steps_S2_mean"),
        "ppo_pct_steps_S3_mean": as_float(learned, "pct_steps_S3_mean"),
    }


def build_secondary_comparator_row(
    run_dir: Path,
    *,
    primary_reference: dict[str, Any],
) -> dict[str, Any]:
    summary = load_json(run_dir / "summary.json")
    config = summary.get("config", {})
    backbone = summary.get("backbone", {})
    reward_mode = str(config.get("reward_mode", "unknown"))
    service_metrics_comparable = (
        backbone.get("observation_version") == primary_reference["observation_version"]
        and backbone.get("year_basis") == primary_reference["year_basis"]
        and backbone.get("risk_level") == primary_reference["risk_level"]
        and backbone.get("stochastic_pt") == primary_reference["stochastic_pt"]
    )
    raw_reward_comparable = (
        reward_mode == primary_reference["reward_mode"] and service_metrics_comparable
    )
    notes: list[str] = []
    if backbone.get("year_basis") != primary_reference["year_basis"]:
        notes.append("year_basis differs from primary Family A bundles")
    if config.get("eval_episodes") != primary_reference["eval_episodes"]:
        notes.append("eval_episodes differs from primary Family A bundles")
    if len(config.get("seeds", [])) != primary_reference["seed_count"]:
        notes.append("seed count differs from primary Family A bundles")
    if not notes:
        notes.append("auditable comparator")
    return {
        "label": run_dir.name,
        "role": "secondary_auditable_comparator",
        "reward_mode": reward_mode,
        "git_commit": backbone.get("git_commit"),
        "seed_count": len(config.get("seeds", [])),
        "eval_episodes": config.get("eval_episodes"),
        "observation_version": backbone.get("observation_version"),
        "year_basis": backbone.get("year_basis"),
        "risk_level": backbone.get("risk_level"),
        "service_metrics_comparable_to_primary": service_metrics_comparable,
        "raw_reward_comparable_to_primary": raw_reward_comparable,
        "notes": "; ".join(notes),
    }


def choose_increased_family_leader(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    seq_rows = [row for row in rows if row["reward_mode"] == "ReT_seq_v1"]
    if not seq_rows:
        seq_rows = rows
    return max(
        seq_rows,
        key=lambda row: (
            float(row["fill_rate_mean"]),
            -float(row["backorder_rate_mean"]),
            float(row["order_level_ret_mean"]),
        ),
    )


def render_markdown(
    primary_rows: list[dict[str, Any]],
    severe_rows: list[dict[str, Any]],
    comparator_rows: list[dict[str, Any]],
    leader: dict[str, Any] | None,
) -> str:
    lines = [
        "# Family A Summary",
        "",
        "This bundle freezes the current Family A readout around the auditable post-audit paper-facing runs.",
        "",
        "Family A definition:",
        "- `reward_mode=ReT_seq_v1` as the leading lane",
        "- `observation_version=v1`",
        "- `year_basis=thesis`",
        "- `step_size_hours=168`",
        "- `risk_level=increased` with severe cross-eval when available",
        "- `stochastic_pt=True`",
        "",
        "## Increased comparison",
        "",
        "| Label | Reward mode | Seeds | Eval eps | Fill | Delta fill vs S2 | Backorder | Delta ReT vs S2 | Shift mix S1/S2/S3 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in primary_rows:
        shift_mix = (
            f"{row['ppo_pct_steps_S1_mean']:.1f} / "
            f"{row['ppo_pct_steps_S2_mean']:.1f} / "
            f"{row['ppo_pct_steps_S3_mean']:.1f}"
        )
        lines.append(
            f"| `{row['label']}` | `{row['reward_mode']}` | {row['seed_count']} | "
            f"{row['eval_episodes']} | {row['fill_rate_mean']:.3f} | "
            f"{row['delta_fill_vs_static_s2']:.3f} | {row['backorder_rate_mean']:.3f} | "
            f"{row['delta_order_level_ret_vs_static_s2']:.3f} | {shift_mix} |"
        )

    lines.extend(["", "## Severe cross-eval", ""])
    lines.append(
        "| Label | Available | Fill | Delta vs S2 | Delta vs S3 | Shift mix S1/S2/S3 |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    for row in severe_rows:
        if not row["severe_available"]:
            lines.append(f"| `{row['label']}` | `False` | n/a | n/a | n/a | n/a |")
            continue
        shift_mix = (
            f"{row['ppo_pct_steps_S1_mean']:.1f} / "
            f"{row['ppo_pct_steps_S2_mean']:.1f} / "
            f"{row['ppo_pct_steps_S3_mean']:.1f}"
        )
        lines.append(
            f"| `{row['label']}` | `True` | {row['fill_rate_mean']:.3f} | "
            f"{row['delta_fill_vs_static_s2']:.3f} | {row['delta_fill_vs_static_s3']:.3f} | "
            f"{shift_mix} |"
        )

    lines.extend(["", "## Secondary comparator", ""])
    lines.append(
        "| Label | Reward mode | Service-comparable | Raw-reward comparable | Notes |"
    )
    lines.append("| --- | --- | --- | --- | --- |")
    for row in comparator_rows:
        lines.append(
            f"| `{row['label']}` | `{row['reward_mode']}` | "
            f"`{row['service_metrics_comparable_to_primary']}` | "
            f"`{row['raw_reward_comparable_to_primary']}` | {row['notes']} |"
        )

    lines.extend(["", "## Current reading", ""])
    if leader is None:
        lines.append("- No leader could be determined.")
    else:
        lines.append(
            f"- Current increased-scenario Family A leader: `{leader['label']}` "
            f"(`{leader['reward_mode']}`, fill={leader['fill_rate_mean']:.3f}, "
            f"delta_vs_static_s2={leader['delta_fill_vs_static_s2']:.3f})."
        )
    lines.append(
        "- No currently valid Family A RL lane clearly beats `static_s2` on fill rate."
    )
    lines.append(
        "- `paper_control_v1_500k` remains the operational comparator, not the leading lane."
    )
    lines.append(
        "- `paper_ret_seq_k020_500k` is the only current Family A bundle with severe cross-eval in the thesis-basis paper-facing family."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    primary_rows = [build_primary_row(run_dir) for run_dir in args.run_dirs]
    severe_rows = [build_severe_row(run_dir) for run_dir in args.run_dirs]
    leader = choose_increased_family_leader(primary_rows)
    reference_row = primary_rows[0]

    comparator_rows: list[dict[str, Any]] = []
    if args.secondary_comparator.exists():
        comparator_rows.append(
            build_secondary_comparator_row(
                args.secondary_comparator,
                primary_reference={
                    "reward_mode": reference_row["reward_mode"],
                    "observation_version": reference_row["observation_version"],
                    "year_basis": reference_row["year_basis"],
                    "risk_level": reference_row["risk_level"],
                    "stochastic_pt": True,
                    "eval_episodes": reference_row["eval_episodes"],
                    "seed_count": reference_row["seed_count"],
                },
            )
        )

    payload = {
        "family": "A",
        "primary_runs": [str(run_dir.resolve()) for run_dir in args.run_dirs],
        "secondary_comparator": (
            str(args.secondary_comparator.resolve())
            if args.secondary_comparator.exists()
            else None
        ),
        "primary_rows": primary_rows,
        "severe_rows": severe_rows,
        "secondary_comparator_rows": comparator_rows,
        "leader": leader,
    }

    save_csv(args.output_dir / "primary_comparison.csv", primary_rows, PRIMARY_FIELDS)
    save_csv(args.output_dir / "severe_cross_eval.csv", severe_rows, SEVERE_FIELDS)
    save_csv(
        args.output_dir / "secondary_comparators.csv",
        comparator_rows,
        COMPARATOR_FIELDS,
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "summary.md").write_text(
        render_markdown(primary_rows, severe_rows, comparator_rows, leader),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
