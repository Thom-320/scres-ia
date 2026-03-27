#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


STATIC_POLICIES = ("static_s1", "static_s2", "static_s3")
CROSS_SCENARIO_FIELDNAMES = [
    "risk_level",
    "learned_phase",
    "static_phase",
    "learned_policy",
    "best_static_policy",
    "learned_fill_rate",
    "static_s2_fill_rate",
    "best_static_fill_rate",
    "delta_fill_rate_vs_static_s2",
    "delta_fill_rate_vs_best_static",
    "learned_backorder_rate",
    "static_s2_backorder_rate",
    "best_static_backorder_rate",
    "delta_backorder_rate_vs_static_s2",
    "delta_backorder_rate_vs_best_static",
    "learned_order_level_ret_mean",
    "static_s2_order_level_ret_mean",
    "best_static_order_level_ret_mean",
    "delta_order_level_ret_mean_vs_static_s2",
    "delta_order_level_ret_mean_vs_best_static",
    "learned_pct_steps_S1",
    "learned_pct_steps_S2",
    "learned_pct_steps_S3",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate proof-of-learning artifacts for a paper benchmark run."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory produced by scripts/run_paper_benchmark.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory. Defaults to <run-dir>/proof_of_learning.",
    )
    return parser


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def resolve_primary_weight_combo(summary: dict[str, Any]) -> dict[str, float] | None:
    survivors = summary.get("survivors", [])
    if survivors:
        row = survivors[0]
        return {
            "w_bo": float(row["w_bo"]),
            "w_cost": float(row["w_cost"]),
            "w_disr": float(row["w_disr"]),
        }
    combos = summary.get("weight_combinations", [])
    if combos:
        row = combos[0]
        return {
            "w_bo": float(row["w_bo"]),
            "w_cost": float(row["w_cost"]),
            "w_disr": float(row["w_disr"]),
        }
    return None


def row_matches_weight_combo(
    row: dict[str, str] | dict[str, Any], combo: dict[str, float] | None
) -> bool:
    if combo is None:
        return True
    return (
        float(row["w_bo"]) == float(combo["w_bo"])
        and float(row["w_cost"]) == float(combo["w_cost"])
        and float(row["w_disr"]) == float(combo["w_disr"])
    )


def policy_row(
    rows: list[dict[str, str]],
    *,
    phase: str,
    policy: str,
    weight_combo: dict[str, float] | None,
) -> dict[str, str] | None:
    for row in rows:
        if (
            row.get("phase") == phase
            and row.get("policy") == policy
            and row_matches_weight_combo(row, weight_combo)
        ):
            return row
    return None


def best_static_row(
    rows: list[dict[str, str]],
    *,
    phase: str,
    weight_combo: dict[str, float] | None,
) -> dict[str, str] | None:
    candidates = [
        row
        for row in rows
        if row.get("phase") == phase
        and row.get("policy") in STATIC_POLICIES
        and row_matches_weight_combo(row, weight_combo)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            as_float(row, "order_level_ret_mean_mean"),
            as_float(row, "fill_rate_mean"),
            -as_float(row, "backorder_rate_mean"),
        ),
    )


def learned_phase_name(config: dict[str, Any], risk_level: str) -> str:
    train_risk_level = str(config["risk_level"])
    algo = str(config["algo"])
    if risk_level == train_risk_level:
        return f"{algo}_eval"
    return f"cross_eval_{risk_level}"


def static_phase_name(config: dict[str, Any], risk_level: str) -> str:
    train_risk_level = str(config["risk_level"])
    if risk_level == train_risk_level:
        return "static_screen"
    return f"cross_eval_{risk_level}"


def build_cross_scenario_rows(run_dir: Path) -> list[dict[str, Any]]:
    summary = load_json(run_dir / "summary.json")
    policy_rows = load_csv_rows(run_dir / "policy_summary.csv")
    config = summary.get("config", {})
    weight_combo = resolve_primary_weight_combo(summary)
    eval_risk_levels = list(config.get("eval_risk_levels") or [config["risk_level"]])
    learned_policy = str(config["algo"])

    output_rows: list[dict[str, Any]] = []
    for risk_level in eval_risk_levels:
        learned_phase = learned_phase_name(config, risk_level)
        baseline_phase = static_phase_name(config, risk_level)
        learned = policy_row(
            policy_rows,
            phase=learned_phase,
            policy=learned_policy,
            weight_combo=weight_combo,
        )
        static_s2 = policy_row(
            policy_rows,
            phase=baseline_phase,
            policy="static_s2",
            weight_combo=weight_combo,
        )
        best_static = best_static_row(
            policy_rows,
            phase=baseline_phase,
            weight_combo=weight_combo,
        )
        if learned is None or static_s2 is None or best_static is None:
            continue

        learned_fill_rate = as_float(learned, "fill_rate_mean")
        learned_backorder_rate = as_float(learned, "backorder_rate_mean")
        learned_ret = as_float(learned, "order_level_ret_mean_mean")
        static_s2_fill_rate = as_float(static_s2, "fill_rate_mean")
        static_s2_backorder_rate = as_float(static_s2, "backorder_rate_mean")
        static_s2_ret = as_float(static_s2, "order_level_ret_mean_mean")
        best_static_fill_rate = as_float(best_static, "fill_rate_mean")
        best_static_backorder_rate = as_float(best_static, "backorder_rate_mean")
        best_static_ret = as_float(best_static, "order_level_ret_mean_mean")

        output_rows.append(
            {
                "risk_level": risk_level,
                "learned_phase": learned_phase,
                "static_phase": baseline_phase,
                "learned_policy": learned_policy,
                "best_static_policy": str(best_static["policy"]),
                "learned_fill_rate": learned_fill_rate,
                "static_s2_fill_rate": static_s2_fill_rate,
                "best_static_fill_rate": best_static_fill_rate,
                "delta_fill_rate_vs_static_s2": learned_fill_rate - static_s2_fill_rate,
                "delta_fill_rate_vs_best_static": (
                    learned_fill_rate - best_static_fill_rate
                ),
                "learned_backorder_rate": learned_backorder_rate,
                "static_s2_backorder_rate": static_s2_backorder_rate,
                "best_static_backorder_rate": best_static_backorder_rate,
                "delta_backorder_rate_vs_static_s2": (
                    learned_backorder_rate - static_s2_backorder_rate
                ),
                "delta_backorder_rate_vs_best_static": (
                    learned_backorder_rate - best_static_backorder_rate
                ),
                "learned_order_level_ret_mean": learned_ret,
                "static_s2_order_level_ret_mean": static_s2_ret,
                "best_static_order_level_ret_mean": best_static_ret,
                "delta_order_level_ret_mean_vs_static_s2": learned_ret - static_s2_ret,
                "delta_order_level_ret_mean_vs_best_static": (
                    learned_ret - best_static_ret
                ),
                "learned_pct_steps_S1": as_float(learned, "pct_steps_S1_mean"),
                "learned_pct_steps_S2": as_float(learned, "pct_steps_S2_mean"),
                "learned_pct_steps_S3": as_float(learned, "pct_steps_S3_mean"),
            }
        )
    return output_rows


def rolling_mean(values: list[float], window: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    window = max(1, min(window, len(arr)))
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.convolve(arr, kernel, mode="valid")
    prefix = np.full(window - 1, np.nan)
    return np.concatenate([prefix, smoothed])


def plot_learning_curve(
    training_rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    if not training_rows:
        raise ValueError("training_trace.csv is empty.")

    grouped: dict[int, list[dict[str, str]]] = {}
    for row in training_rows:
        grouped.setdefault(int(row["seed"]), []).append(row)

    max_timestep = max(int(row["timesteps"]) for row in training_rows)
    common_grid = np.linspace(0, max_timestep, 200)
    interpolated: list[np.ndarray] = []

    fig, ax = plt.subplots(figsize=(10, 5.4))
    for seed, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["timesteps"]))
        timesteps = np.asarray([int(row["timesteps"]) for row in rows], dtype=float)
        rewards = np.asarray(
            [float(row["episode_reward"]) for row in rows], dtype=float
        )
        smoothed = rolling_mean(rewards.tolist(), window=5)
        ax.plot(timesteps, rewards, color="#5c7cfa", alpha=0.14, linewidth=1.0)
        ax.plot(timesteps, smoothed, color="#1c4ed8", alpha=0.28, linewidth=1.4)
        valid = ~np.isnan(smoothed)
        if valid.sum() >= 2:
            interpolated.append(
                np.interp(
                    common_grid,
                    timesteps[valid],
                    smoothed[valid],
                    left=smoothed[valid][0],
                    right=smoothed[valid][-1],
                )
            )

    if interpolated:
        mean_curve = np.nanmean(np.vstack(interpolated), axis=0)
        ax.plot(
            common_grid,
            mean_curve,
            color="#111827",
            linewidth=2.5,
            linestyle="--",
            label="Mean rolling reward",
        )
        ax.legend(loc="best")

    ax.set_title("Learning Curve")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Episode reward")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def choose_timeline_risk_level(proof_rows: list[dict[str, str]]) -> str:
    available = {
        str(row["risk_level"])
        for row in proof_rows
        if str(row.get("policy", "")) != "static_s2"
    }
    if "severe" in available:
        return "severe"
    if not available:
        raise ValueError("proof_trajectories.csv does not contain learned-policy rows.")

    disruption_by_level: dict[str, list[float]] = {}
    for row in proof_rows:
        if str(row.get("policy", "")) == "static_s2":
            continue
        disruption_by_level.setdefault(str(row["risk_level"]), []).append(
            float(row["disruption_fraction"])
        )
    return max(
        disruption_by_level,
        key=lambda level: float(np.mean(disruption_by_level[level])),
    )


def first_episode_rows(
    proof_rows: list[dict[str, str]],
    *,
    risk_level: str,
    policy: str,
) -> list[dict[str, str]]:
    candidates = [
        row
        for row in proof_rows
        if str(row["risk_level"]) == risk_level and str(row["policy"]) == policy
    ]
    if not candidates:
        return []
    candidates.sort(
        key=lambda row: (
            int(row["seed"]),
            int(row["episode"]),
            int(row["step"]),
        )
    )
    seed = int(candidates[0]["seed"])
    episode = int(candidates[0]["episode"])
    return [
        row
        for row in candidates
        if int(row["seed"]) == seed and int(row["episode"]) == episode
    ]


def plot_shift_timeline(
    proof_rows: list[dict[str, str]],
    output_path: Path,
) -> str:
    if not proof_rows:
        raise ValueError("proof_trajectories.csv is empty.")

    risk_level = choose_timeline_risk_level(proof_rows)
    learned_rows = first_episode_rows(
        proof_rows, risk_level=risk_level, policy=str(proof_rows[0]["algo"])
    )
    if not learned_rows:
        learned_policy_names = sorted(
            {
                str(row["policy"])
                for row in proof_rows
                if str(row["policy"]) != "static_s2"
            }
        )
        if not learned_policy_names:
            raise ValueError(
                "No learned-policy trajectory found in proof_trajectories.csv."
            )
        learned_rows = first_episode_rows(
            proof_rows,
            risk_level=risk_level,
            policy=learned_policy_names[0],
        )
    static_rows = first_episode_rows(
        proof_rows, risk_level=risk_level, policy="static_s2"
    )

    learned_steps = [int(row["step"]) for row in learned_rows]
    learned_shifts = [int(row["shifts_active"]) for row in learned_rows]
    learned_fill = [float(row["fill_rate"]) for row in learned_rows]
    learned_disruption = [float(row["disruption_fraction"]) for row in learned_rows]

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.2), sharex=True)

    axes[0].step(
        learned_steps,
        learned_shifts,
        where="post",
        color="#0f766e",
        linewidth=2.2,
        label="Learned policy",
    )
    if static_rows:
        axes[0].step(
            [int(row["step"]) for row in static_rows],
            [int(row["shifts_active"]) for row in static_rows],
            where="post",
            color="#b45309",
            linewidth=1.8,
            linestyle="--",
            label="Static S2",
        )
    axes[0].set_ylabel("Shift level")
    axes[0].set_yticks([1, 2, 3])
    axes[0].set_ylim(0.8, 3.2)
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="best")

    axes[1].fill_between(
        learned_steps,
        learned_disruption,
        step="post",
        color="#ef4444",
        alpha=0.22,
        label="Disruption fraction",
    )
    axes[1].plot(
        learned_steps,
        learned_fill,
        color="#1d4ed8",
        linewidth=2.0,
        label="Learned fill rate",
    )
    if static_rows:
        axes[1].plot(
            [int(row["step"]) for row in static_rows],
            [float(row["fill_rate"]) for row in static_rows],
            color="#92400e",
            linewidth=1.5,
            linestyle="--",
            label="Static S2 fill rate",
        )
    axes[1].set_xlabel("Episode step")
    axes[1].set_ylabel("Fraction")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="best")

    fig.suptitle(f"Adaptive Shift Behavior Under {risk_level.capitalize()} Risk")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return risk_level


def write_cross_scenario_markdown(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    lines = [
        "# Cross-Scenario Comparison",
        "",
        "| Risk | Learned Fill | Static S2 Fill | Best Static | Learned ReT | Static S2 ReT | Best Static ReT |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {risk} | {learned_fill:.3f} | {static_fill:.3f} | {best_static} | {learned_ret:.3f} | {static_ret:.3f} | {best_ret:.3f} |".format(
                risk=row["risk_level"],
                learned_fill=row["learned_fill_rate"],
                static_fill=row["static_s2_fill_rate"],
                best_static=row["best_static_policy"],
                learned_ret=row["learned_order_level_ret_mean"],
                static_ret=row["static_s2_order_level_ret_mean"],
                best_ret=row["best_static_order_level_ret_mean"],
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_proof_of_learning_artifacts(
    run_dir: Path,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    summary = load_json(run_dir / "summary.json")
    weight_combo = resolve_primary_weight_combo(summary)
    training_rows = load_csv_rows(run_dir / "training_trace.csv")
    proof_rows = load_csv_rows(run_dir / "proof_trajectories.csv")

    if weight_combo is not None:
        training_rows = [
            row for row in training_rows if row_matches_weight_combo(row, weight_combo)
        ]
        proof_rows = [
            row for row in proof_rows if row_matches_weight_combo(row, weight_combo)
        ]

    cross_rows = build_cross_scenario_rows(run_dir)
    if not cross_rows:
        raise ValueError(
            "No cross-scenario rows could be built from policy_summary.csv."
        )

    resolved_output_dir = output_dir or (run_dir / "proof_of_learning")
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    learning_curve_path = resolved_output_dir / "learning_curve.png"
    shift_timeline_path = resolved_output_dir / "shift_vs_disruption_timeline.png"
    cross_csv_path = resolved_output_dir / "cross_scenario_comparison.csv"
    cross_md_path = resolved_output_dir / "cross_scenario_comparison.md"
    manifest_path = resolved_output_dir / "manifest.json"

    plot_learning_curve(training_rows, learning_curve_path)
    selected_timeline_risk = plot_shift_timeline(proof_rows, shift_timeline_path)
    save_csv(cross_csv_path, cross_rows, CROSS_SCENARIO_FIELDNAMES)
    write_cross_scenario_markdown(cross_rows, cross_md_path)

    manifest = {
        "artifact_type": "proof_of_learning",
        "generated_at_utc": utc_now_iso(),
        "run_dir": str(run_dir.resolve()),
        "output_dir": str(resolved_output_dir.resolve()),
        "primary_weight_combo": weight_combo,
        "selected_timeline_risk_level": selected_timeline_risk,
        "files": {
            "learning_curve_png": str(learning_curve_path.resolve()),
            "shift_vs_disruption_timeline_png": str(shift_timeline_path.resolve()),
            "cross_scenario_comparison_csv": str(cross_csv_path.resolve()),
            "cross_scenario_comparison_md": str(cross_md_path.resolve()),
            "manifest_json": str(manifest_path.resolve()),
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    manifest = generate_proof_of_learning_artifacts(
        args.run_dir,
        output_dir=args.output_dir,
    )
    print(
        "Generated proof-of-learning artifacts at "
        f"{Path(manifest['output_dir']).resolve()}"
    )


if __name__ == "__main__":
    main()
