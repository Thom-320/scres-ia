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
    Path("outputs/paper_benchmarks/paper_control_v1_500k"),
    Path("outputs/paper_benchmarks/paper_ret_seq_k010_500k"),
    Path("outputs/paper_benchmarks/paper_ret_seq_k020_500k"),
)
DEFAULT_OUTPUT_DIR = Path("outputs/paper_benchmarks/paper_trio_analysis")
STATUS_FIELDS = [
    "label",
    "run_dir",
    "state",
    "reward_mode",
    "reward_family",
    "started_at_utc",
    "finished_at_utc",
    "heartbeat_last_activity_utc",
    "summary_exists",
    "policy_summary_exists",
    "comparison_table_exists",
]
COMPARABLE_FIELDS = [
    "label",
    "run_dir",
    "state",
    "reward_mode",
    "reward_family",
    "fill_rate_mean",
    "backorder_rate_mean",
    "order_level_ret_mean",
    "pct_steps_S1_mean",
    "pct_steps_S2_mean",
    "pct_steps_S3_mean",
    "static_s2_fill_rate_mean",
    "static_s2_backorder_rate_mean",
    "static_s2_order_level_ret_mean",
    "delta_fill_rate_vs_static_s2",
    "delta_backorder_rate_vs_static_s2",
    "delta_order_level_ret_mean_vs_static_s2",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the paper-facing trio of benchmark runs using only "
            "cross-mode comparable metrics."
        )
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        type=Path,
        default=list(DEFAULT_RUN_DIRS),
        help="Directories produced by scripts/run_paper_benchmark.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the analysis bundle will be written.",
    )
    return parser


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
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


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def infer_reward_mode(command_text: str) -> str:
    marker = "--reward-mode "
    if marker not in command_text:
        return "unknown"
    tail = command_text.split(marker, maxsplit=1)[1].strip()
    return tail.split()[0] if tail else "unknown"


def build_status_row(run_dir: Path) -> dict[str, Any]:
    label = run_dir.name
    status_path = run_dir / "status.json"
    heartbeat_path = run_dir / "heartbeat.json"
    summary_path = run_dir / "summary.json"
    policy_summary_path = run_dir / "policy_summary.csv"
    comparison_table_path = run_dir / "comparison_table.csv"
    command_path = run_dir / "command.txt"

    status_payload: dict[str, Any] = (
        load_json(status_path) if status_path.exists() else {}
    )
    heartbeat_payload: dict[str, Any] = (
        load_json(heartbeat_path) if heartbeat_path.exists() else {}
    )
    summary_payload: dict[str, Any] = (
        load_json(summary_path) if summary_path.exists() else {}
    )
    reward_contract = summary_payload.get("reward_contract", {})
    config = summary_payload.get("config", {})
    benchmark_command = str(status_payload.get("benchmark_command", ""))
    if not benchmark_command and command_path.exists():
        benchmark_command = command_path.read_text(encoding="utf-8")
    inferred_reward_mode = infer_reward_mode(benchmark_command)

    return {
        "label": label,
        "run_dir": str(run_dir.resolve()),
        "state": str(status_payload.get("state", "missing")),
        "reward_mode": str(
            config.get(
                "reward_mode",
                reward_contract.get("reward_mode", inferred_reward_mode),
            )
        ),
        "reward_family": str(reward_contract.get("reward_family", "unknown")),
        "started_at_utc": status_payload.get("started_at_utc"),
        "finished_at_utc": status_payload.get("finished_at_utc"),
        "heartbeat_last_activity_utc": heartbeat_payload.get("last_activity_utc"),
        "summary_exists": summary_path.exists(),
        "policy_summary_exists": policy_summary_path.exists(),
        "comparison_table_exists": comparison_table_path.exists(),
    }


def build_comparable_row(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "summary.json"
    policy_summary_path = run_dir / "policy_summary.csv"
    status_path = run_dir / "status.json"
    if not summary_path.exists() or not policy_summary_path.exists():
        return None

    summary = load_json(summary_path)
    status = load_json(status_path) if status_path.exists() else {}
    reward_contract = summary.get("reward_contract", {})
    config = summary.get("config", {})
    rows = load_csv_rows(policy_summary_path)

    learned = policy_row(
        rows,
        phase=f"{config['algo']}_eval",
        policy=str(config["algo"]),
    )
    static_s2 = policy_row(rows, phase="static_screen", policy="static_s2")
    if learned is None or static_s2 is None:
        return None

    fill_rate_mean = as_float(learned, "fill_rate_mean")
    backorder_rate_mean = as_float(learned, "backorder_rate_mean")
    order_level_ret_mean = as_float(learned, "order_level_ret_mean_mean")
    static_s2_fill_rate_mean = as_float(static_s2, "fill_rate_mean")
    static_s2_backorder_rate_mean = as_float(static_s2, "backorder_rate_mean")
    static_s2_order_level_ret_mean = as_float(static_s2, "order_level_ret_mean_mean")

    return {
        "label": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "state": str(status.get("state", "unknown")),
        "reward_mode": str(config.get("reward_mode", "unknown")),
        "reward_family": str(reward_contract.get("reward_family", "unknown")),
        "fill_rate_mean": fill_rate_mean,
        "backorder_rate_mean": backorder_rate_mean,
        "order_level_ret_mean": order_level_ret_mean,
        "pct_steps_S1_mean": as_float(learned, "pct_steps_S1_mean"),
        "pct_steps_S2_mean": as_float(learned, "pct_steps_S2_mean"),
        "pct_steps_S3_mean": as_float(learned, "pct_steps_S3_mean"),
        "static_s2_fill_rate_mean": static_s2_fill_rate_mean,
        "static_s2_backorder_rate_mean": static_s2_backorder_rate_mean,
        "static_s2_order_level_ret_mean": static_s2_order_level_ret_mean,
        "delta_fill_rate_vs_static_s2": fill_rate_mean - static_s2_fill_rate_mean,
        "delta_backorder_rate_vs_static_s2": (
            backorder_rate_mean - static_s2_backorder_rate_mean
        ),
        "delta_order_level_ret_mean_vs_static_s2": (
            order_level_ret_mean - static_s2_order_level_ret_mean
        ),
    }


def choose_pragmatic_leader(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row["order_level_ret_mean"]),
            float(row["fill_rate_mean"]),
            -float(row["backorder_rate_mean"]),
        ),
    )


def render_markdown(
    status_rows: list[dict[str, Any]],
    comparable_rows: list[dict[str, Any]],
    leader: dict[str, Any] | None,
) -> str:
    lines = [
        "# Paper Benchmark Trio Analysis",
        "",
        "This bundle applies the pragmatic reconciliation protocol across the three paper-facing runs.",
        "",
        "Comparable metrics only:",
        "- `fill_rate_mean`",
        "- `backorder_rate_mean`",
        "- `order_level_ret_mean`",
        "- learned shift mix (`pct_steps_S1_mean`, `pct_steps_S2_mean`, `pct_steps_S3_mean`)",
        "",
        "Do not compare raw `reward_total` across different reward modes.",
        "",
        "## Run Status",
        "",
        "| Label | State | Reward mode | Reward family | Summary | Policy CSV | Comparison CSV | Last heartbeat |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in status_rows:
        lines.append(
            f"| `{row['label']}` | `{row['state']}` | `{row['reward_mode']}` | "
            f"`{row['reward_family']}` | `{row['summary_exists']}` | "
            f"`{row['policy_summary_exists']}` | `{row['comparison_table_exists']}` | "
            f"`{row['heartbeat_last_activity_utc'] or 'NA'}` |"
        )

    lines.extend(
        [
            "",
            "## Comparable Summary",
            "",
            "| Label | Reward mode | Fill rate | Backorder rate | Order-level ReT | Shift mix S1/S2/S3 | Delta fill vs S2 | Delta backorder vs S2 | Delta ReT vs S2 |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in comparable_rows:
        shift_mix = (
            f"{row['pct_steps_S1_mean']:.1f} / "
            f"{row['pct_steps_S2_mean']:.1f} / "
            f"{row['pct_steps_S3_mean']:.1f}"
        )
        lines.append(
            f"| `{row['label']}` | `{row['reward_mode']}` | "
            f"{row['fill_rate_mean']:.3f} | {row['backorder_rate_mean']:.3f} | "
            f"{row['order_level_ret_mean']:.3f} | {shift_mix} | "
            f"{row['delta_fill_rate_vs_static_s2']:.3f} | "
            f"{row['delta_backorder_rate_vs_static_s2']:.3f} | "
            f"{row['delta_order_level_ret_mean_vs_static_s2']:.3f} |"
        )

    lines.extend(["", "## Leader", ""])
    if leader is None:
        lines.append(
            "No completed run has enough artifacts yet to compute the comparable trio summary."
        )
    else:
        lines.append(
            "Pragmatic leader by `(order_level_ret_mean, fill_rate_mean, -backorder_rate_mean)`:"
        )
        lines.append(
            f"- `{leader['label']}` with `reward_mode={leader['reward_mode']}`, "
            f"`order_level_ret_mean={leader['order_level_ret_mean']:.3f}`, "
            f"`fill_rate_mean={leader['fill_rate_mean']:.3f}`, "
            f"`backorder_rate_mean={leader['backorder_rate_mean']:.3f}`"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    status_rows = [build_status_row(run_dir) for run_dir in args.run_dirs]
    comparable_rows = [
        row
        for run_dir in args.run_dirs
        if (row := build_comparable_row(run_dir)) is not None
    ]
    leader = choose_pragmatic_leader(comparable_rows)

    save_csv(args.output_dir / "run_status.csv", status_rows, STATUS_FIELDS)
    save_csv(
        args.output_dir / "comparable_summary.csv",
        comparable_rows,
        COMPARABLE_FIELDS,
    )

    payload = {
        "protocol": "pragmatic_reconciliation_v1",
        "selection_rule": [
            "order_level_ret_mean",
            "fill_rate_mean",
            "-backorder_rate_mean",
        ],
        "status_rows": status_rows,
        "comparable_rows": comparable_rows,
        "leader": leader,
    }
    (args.output_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (args.output_dir / "analysis.md").write_text(
        render_markdown(status_rows, comparable_rows, leader),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
