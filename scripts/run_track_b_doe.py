#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import statistics
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import OPERATIONS
from supply_chain.external_env_interface import (
    get_episode_terminal_metrics,
    get_track_b_env_spec,
    make_track_b_env,
)

DEFAULT_OUTPUT_ROOT = Path("outputs/doe")
DEFAULT_SHIFT_LEVELS = (1, 2, 3)
DEFAULT_DOWNSTREAM_MULTIPLIERS = (0.5, 1.0, 2.0)
DEFAULT_SEEDS = (11, 22, 33, 44, 55)
DEFAULT_MAX_STEPS = 260

SEED_FIELDS = [
    "policy",
    "seed",
    "assembly_shifts",
    "downstream_multiplier",
    "reward_total",
    "fill_rate",
    "backorder_rate",
    "order_level_ret_mean",
    "flow_fill_rate",
    "flow_backorder_rate",
    "terminal_rolling_fill_rate_4w",
    "terminal_rolling_backorder_rate_4w",
]

SUMMARY_FIELDS = [
    "policy",
    "assembly_shifts",
    "downstream_multiplier",
    "seed_count",
    "reward_total_mean",
    "reward_total_std",
    "fill_rate_mean",
    "fill_rate_std",
    "backorder_rate_mean",
    "backorder_rate_std",
    "order_level_ret_mean",
    "order_level_ret_std",
    "terminal_rolling_fill_rate_4w_mean",
    "terminal_rolling_backorder_rate_4w_mean",
]


@dataclass(frozen=True)
class PolicySpec:
    label: str
    assembly_shifts: int
    downstream_multiplier: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a static DOE over the minimal Track B environment to measure "
            "whether downstream control opens non-trivial headroom."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the DOE bundle. Defaults to a timestamped path.",
    )
    parser.add_argument(
        "--reward-mode",
        default="ReT_seq_v1",
        help="Training reward used only for within-DOE comparison.",
    )
    parser.add_argument(
        "--risk-level",
        default="adaptive_benchmark_v2",
        help="Track B risk profile to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Replicate seeds used as static-policy DOE repetitions.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Episode length in decision steps.",
    )
    parser.add_argument(
        "--step-size-hours",
        type=float,
        default=168.0,
        help="Decision cadence in hours.",
    )
    parser.add_argument(
        "--shift-levels",
        nargs="+",
        type=int,
        default=list(DEFAULT_SHIFT_LEVELS),
        help="Shift levels included in the DOE grid.",
    )
    parser.add_argument(
        "--downstream-multipliers",
        nargs="+",
        type=float,
        default=list(DEFAULT_DOWNSTREAM_MULTIPLIERS),
        help="Downstream Op10/Op12 dispatch multipliers included in the DOE grid.",
    )
    return parser


def default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / f"track_b_minimal_doe_{timestamp}"


def save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_policy_grid(
    shift_levels: tuple[int, ...], downstream_multipliers: tuple[float, ...]
) -> list[PolicySpec]:
    return [
        PolicySpec(
            label=f"s{shifts}_d{multiplier:.2f}",
            assembly_shifts=shifts,
            downstream_multiplier=float(multiplier),
        )
        for shifts in shift_levels
        for multiplier in downstream_multipliers
    ]


def build_direct_policy_action(policy: PolicySpec) -> dict[str, float | int]:
    downstream_multiplier = float(policy.downstream_multiplier)
    return {
        "op3_q": float(OPERATIONS[3]["q"]),
        "op3_rop": float(OPERATIONS[3]["rop"]),
        "op9_q_min": float(OPERATIONS[9]["q"][0]),
        "op9_q_max": float(OPERATIONS[9]["q"][1]),
        "op9_rop": float(OPERATIONS[9]["rop"]),
        "op10_q_min": float(OPERATIONS[10]["q"][0]) * downstream_multiplier,
        "op10_q_max": float(OPERATIONS[10]["q"][1]) * downstream_multiplier,
        "op12_q_min": float(OPERATIONS[12]["q"][0]) * downstream_multiplier,
        "op12_q_max": float(OPERATIONS[12]["q"][1]) * downstream_multiplier,
        "assembly_shifts": int(policy.assembly_shifts),
    }


def run_static_policy_episode(
    policy: PolicySpec,
    *,
    seed: int,
    env_kwargs: dict[str, Any],
) -> dict[str, Any]:
    env = make_track_b_env(**env_kwargs)
    obs, info = env.reset(seed=seed)
    terminated = False
    truncated = False
    reward_total = 0.0
    demanded_total = 0.0
    backorder_qty_total = 0.0
    final_info: dict[str, Any] = info
    action_payload = build_direct_policy_action(policy)

    while not (terminated or truncated):
        obs, reward, terminated, truncated, final_info = env.step(action_payload)
        reward_total += float(reward)
        demanded_total += float(final_info.get("new_demanded", 0.0))
        backorder_qty_total += float(final_info.get("new_backorder_qty", 0.0))

    terminal_metrics = get_episode_terminal_metrics(env)
    track_b_context = final_info["state_constraint_context"]["track_b_context"]
    if demanded_total > 0.0:
        flow_backorder_rate = backorder_qty_total / demanded_total
        flow_fill_rate = 1.0 - flow_backorder_rate
    else:
        flow_backorder_rate = 0.0
        flow_fill_rate = 1.0
    row = {
        "policy": policy.label,
        "seed": seed,
        "assembly_shifts": policy.assembly_shifts,
        "downstream_multiplier": policy.downstream_multiplier,
        "reward_total": reward_total,
        "fill_rate": float(terminal_metrics["fill_rate_order_level"]),
        "backorder_rate": float(terminal_metrics["backorder_rate_order_level"]),
        "order_level_ret_mean": float(terminal_metrics["order_level_ret_mean"]),
        "flow_fill_rate": flow_fill_rate,
        "flow_backorder_rate": flow_backorder_rate,
        "terminal_rolling_fill_rate_4w": float(track_b_context["rolling_fill_rate_4w"]),
        "terminal_rolling_backorder_rate_4w": float(
            track_b_context["rolling_backorder_rate_4w"]
        ),
    }
    env.close()
    return row


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["policy"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for policy, policy_rows in grouped.items():

        def mean(key: str) -> float:
            return float(statistics.fmean(float(row[key]) for row in policy_rows))

        def std(key: str) -> float:
            if len(policy_rows) < 2:
                return 0.0
            return float(statistics.stdev(float(row[key]) for row in policy_rows))

        first = policy_rows[0]
        summary_rows.append(
            {
                "policy": policy,
                "assembly_shifts": int(first["assembly_shifts"]),
                "downstream_multiplier": float(first["downstream_multiplier"]),
                "seed_count": len(policy_rows),
                "reward_total_mean": mean("reward_total"),
                "reward_total_std": std("reward_total"),
                "fill_rate_mean": mean("fill_rate"),
                "fill_rate_std": std("fill_rate"),
                "backorder_rate_mean": mean("backorder_rate"),
                "backorder_rate_std": std("backorder_rate"),
                "order_level_ret_mean": mean("order_level_ret_mean"),
                "order_level_ret_std": std("order_level_ret_mean"),
                "terminal_rolling_fill_rate_4w_mean": mean(
                    "terminal_rolling_fill_rate_4w"
                ),
                "terminal_rolling_backorder_rate_4w_mean": mean(
                    "terminal_rolling_backorder_rate_4w"
                ),
            }
        )
    return sorted(summary_rows, key=lambda row: str(row["policy"]))


def choose_best_policy(summary_rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return max(summary_rows, key=lambda row: float(row[key]))


def baseline_row(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in summary_rows:
        if (
            int(row["assembly_shifts"]) == 2
            and float(row["downstream_multiplier"]) == 1.0
        ):
            return row
    raise ValueError("Baseline row s2_d1.00 not present in DOE summary.")


def build_decision_summary(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = baseline_row(summary_rows)
    best_fill = choose_best_policy(summary_rows, "fill_rate_mean")
    best_reward = choose_best_policy(summary_rows, "reward_total_mean")
    delta_fill_pp = 100.0 * (
        float(best_fill["fill_rate_mean"]) - float(baseline["fill_rate_mean"])
    )
    baseline_reward = float(baseline["reward_total_mean"])
    reward_gain_fraction = (
        float(best_reward["reward_total_mean"]) - baseline_reward
    ) / max(abs(baseline_reward), 1e-9)
    return {
        "baseline_policy": baseline["policy"],
        "best_by_fill": best_fill["policy"],
        "best_by_reward": best_reward["policy"],
        "delta_fill_pp_vs_s2_neutral": delta_fill_pp,
        "reward_gain_fraction_vs_s2_neutral": reward_gain_fraction,
        "headroom_open_by_fill": delta_fill_pp > 1.0,
        "headroom_open_by_reward": reward_gain_fraction > 0.01,
    }


def render_markdown(
    spec: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    decision: dict[str, Any],
) -> str:
    lines = [
        "# Track B Minimal DOE",
        "",
        "This bundle measures static-policy headroom in the minimal Track B contract before any PPO training.",
        "",
        "## Contract",
        "",
        f"- `env_variant={spec['env_variant']}`",
        f"- `reward_mode={spec['reward_mode']}`",
        f"- `observation_version={spec['observation_version']}`",
        f"- `action_dims={len(spec['action_fields'])}`",
        "",
        "## Policy summary",
        "",
        "| Policy | Shifts | Downstream mult | Reward mean | Fill | Backorder | Order-level ReT | Rolling fill 4w |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['policy']}` | {int(row['assembly_shifts'])} | "
            f"{float(row['downstream_multiplier']):.2f} | "
            f"{float(row['reward_total_mean']):.2f} | {float(row['fill_rate_mean']):.3f} | "
            f"{float(row['backorder_rate_mean']):.3f} | {float(row['order_level_ret_mean']):.3f} | "
            f"{float(row['terminal_rolling_fill_rate_4w_mean']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Baseline policy: `{decision['baseline_policy']}`.",
            f"- Best by fill: `{decision['best_by_fill']}`.",
            f"- Best by reward: `{decision['best_by_reward']}`.",
            f"- Delta fill vs neutral S2: {float(decision['delta_fill_pp_vs_s2_neutral']):.2f} percentage points.",
            f"- Reward gain fraction vs neutral S2: {100.0 * float(decision['reward_gain_fraction_vs_s2_neutral']):.2f}%.",
            f"- Headroom open by fill (>1pp): `{decision['headroom_open_by_fill']}`.",
            f"- Headroom open by reward (>1%): `{decision['headroom_open_by_reward']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_grid = build_policy_grid(
        tuple(args.shift_levels),
        tuple(args.downstream_multipliers),
    )
    env_kwargs = {
        "reward_mode": args.reward_mode,
        "observation_version": "v7",
        "action_contract": "track_b_v1",
        "risk_level": args.risk_level,
        "year_basis": "thesis",
        "stochastic_pt": True,
        "step_size_hours": float(args.step_size_hours),
        "max_steps": int(args.max_steps),
    }

    seed_rows: list[dict[str, Any]] = []
    for policy in policy_grid:
        for seed in args.seeds:
            seed_rows.append(
                run_static_policy_episode(policy, seed=int(seed), env_kwargs=env_kwargs)
            )

    summary_rows = summarize_rows(seed_rows)
    decision = build_decision_summary(summary_rows)
    spec = {
        **get_track_b_env_spec(
            reward_mode=args.reward_mode, observation_version="v7"
        ).__dict__,
        "risk_level": args.risk_level,
        "max_steps": int(args.max_steps),
        "seeds": list(args.seeds),
    }
    payload = {
        "spec": spec,
        "policies": [policy.__dict__ for policy in policy_grid],
        "seed_rows": seed_rows,
        "summary_rows": summary_rows,
        "decision": decision,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    save_csv(output_dir / "seed_metrics.csv", seed_rows, SEED_FIELDS)
    save_csv(output_dir / "policy_summary.csv", summary_rows, SUMMARY_FIELDS)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        render_markdown(spec, summary_rows, decision),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
