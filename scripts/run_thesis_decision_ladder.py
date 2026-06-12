#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.external_env_interface import (  # noqa: E402
    THESIS_INVENTORY_PERIODS,
    get_episode_terminal_metrics,
    make_dkana_thesis_faithful_env,
)
from supply_chain.thesis_design import (  # noqa: E402
    ThesisDesignSpec,
    design_spec_for_cfi,
    parse_cf_range,
)

DEFAULT_OUTPUT_ROOT = Path("outputs/benchmarks/thesis_decision_ladder")
LADDER_LEVELS = ("L0_garrido", "L1a_uniform_IxS", "L1b_per_node_IxS")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run static thesis-decision ladder policies without RL training. "
            "L0 reproduces Garrido rows, L1a combines common I_t,S with S, "
            "and L1b declares the per-node inventory extension."
        )
    )
    parser.add_argument("--label", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--levels",
        choices=LADDER_LEVELS,
        nargs="+",
        default=["L0_garrido", "L1a_uniform_IxS"],
    )
    parser.add_argument("--replications", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--garrido-cfis", default="31-90")
    parser.add_argument("--reward-mode", default="ReT_cd_v1")
    parser.add_argument("--risk-level", default="increased")
    parser.add_argument("--observation-version", default="v5")
    parser.add_argument(
        "--observation-mode",
        choices=[
            "decision_reward",
            "env_reward",
            "env_state_reward",
            "env_sdm_history_reward",
        ],
        default="env_sdm_history_reward",
    )
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--stochastic-pt", action="store_true")
    parser.add_argument("--l1b-screening-replications", type=int, default=5)
    parser.add_argument("--l1b-top-k", type=int, default=20)
    parser.add_argument("--l1b-top-replications", type=int, default=None)
    return parser


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def base_env_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "reward_mode": args.reward_mode,
        "risk_level": args.risk_level,
        "observation_version": args.observation_version,
        "observation_mode": args.observation_mode,
        "step_size_hours": args.step_size_hours,
        "max_steps": args.max_steps,
        "stochastic_pt": args.stochastic_pt,
        "learn_initial_decision": False,
    }


def thesis_factorized_action(period: int | None, shifts: int) -> np.ndarray:
    if period is None:
        return np.array([0, shifts - 1], dtype=np.int64)
    return np.array(
        [THESIS_INVENTORY_PERIODS.index(int(period)) + 1, shifts - 1],
        dtype=np.int64,
    )


def per_node_action(
    op3_period: int | None,
    op5_period: int | None,
    op9_period: int | None,
    shifts: int,
) -> np.ndarray:
    levels = []
    for period in (op3_period, op5_period, op9_period):
        levels.append(0 if period is None else THESIS_INVENTORY_PERIODS.index(period) + 1)
    return np.array([*levels, shifts - 1], dtype=np.int64)


def thesis_design_action(spec: ThesisDesignSpec) -> np.ndarray:
    period = spec.inventory_replenishment_period
    return thesis_factorized_action(None if period is None else int(period), spec.shifts)


def summarize_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for policy in sorted({str(row["policy"]) for row in rows}):
        bucket = [row for row in rows if row["policy"] == policy]
        out.append(
            {
                "policy": policy,
                "ladder_level": bucket[0]["ladder_level"],
                "stage": bucket[0]["stage"],
                "episode_count": len(bucket),
                "reward_total_mean": float(np.mean([row["reward_total"] for row in bucket])),
                "fill_rate_order_level_mean": float(
                    np.mean([row["fill_rate_order_level"] for row in bucket])
                ),
                "order_level_ret_mean": float(
                    np.mean([row["order_level_ret_mean"] for row in bucket])
                ),
                "assembly_shift_hours_mean": float(
                    np.mean([row["assembly_shift_hours"] for row in bucket])
                ),
                "inventory_target_total_mean": float(
                    np.mean([row["inventory_target_total_mean"] for row in bucket])
                ),
                "pending_backorders_count_mean": float(
                    np.mean([row["pending_backorders_count"] for row in bucket])
                ),
                "pending_backorder_qty_mean": float(
                    np.mean([row["pending_backorder_qty"] for row in bucket])
                ),
            }
        )
    return out


def run_policy(
    *,
    args: argparse.Namespace,
    policy_name: str,
    ladder_level: str,
    stage: str,
    action_space_mode: str,
    inventory_period_mode: str,
    action: np.ndarray,
    replications: int,
    seed_offset: int,
    env_override: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    policy_metadata = dict(metadata or {})
    kwargs = base_env_kwargs(args)
    kwargs.update(
        {
            "action_space_mode": action_space_mode,
            "inventory_period_mode": inventory_period_mode,
            "initial_action": action,
        }
    )
    kwargs.update(env_override or {})

    for replication in range(replications):
        eval_seed = args.seed + seed_offset + replication
        env = make_dkana_thesis_faithful_env(**kwargs)
        obs, info = env.reset(seed=eval_seed)
        terminated = truncated = False
        reward_total = 0.0
        steps = 0
        assembly_shift_hours = 0.0
        inventory_target_total_sum = 0.0

        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
            reward_total += float(reward)
            if info.get("action_phase") != "weekly_decision":
                continue
            steps += 1
            decision = info.get("thesis_decision", {})
            shifts = int(decision.get("assembly_shifts", 1))
            assembly_shift_hours += float(shifts) * float(args.step_size_hours)
            targets = decision.get("inventory_buffer_targets", {})
            if isinstance(targets, dict):
                inventory_target_total_sum += float(sum(float(v) for v in targets.values()))

        terminal = get_episode_terminal_metrics(env)
        sim = getattr(env.unwrapped, "sim", None)
        total_steps = max(1, steps)
        row = {
            "policy": policy_name,
            "ladder_level": ladder_level,
            "stage": stage,
            "replication": replication,
            "seed": eval_seed,
            "action_space_mode": action_space_mode,
            "inventory_period_mode": inventory_period_mode,
            "action": json.dumps(np.asarray(action).astype(int).tolist()),
            "steps": steps,
            "reward_total": reward_total,
            "fill_rate_order_level": terminal["fill_rate_order_level"],
            "backorder_rate_order_level": terminal["backorder_rate_order_level"],
            "order_level_ret_mean": terminal["order_level_ret_mean"],
            "assembly_shift_hours": assembly_shift_hours,
            "inventory_target_total_mean": inventory_target_total_sum / total_steps,
            "pending_backorders_count": float(
                len(getattr(sim, "pending_backorders", [])) if sim is not None else 0.0
            ),
            "pending_backorder_qty": float(
                sum(
                    float(getattr(order, "remaining_qty", 0.0))
                    for order in getattr(sim, "pending_backorders", [])
                )
                if sim is not None
                else 0.0
            ),
        }
        row.update(policy_metadata)
        rows.append(row)
        env.close()
    return rows


def l0_garrido_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cfi in parse_cf_range(args.garrido_cfis):
        spec = design_spec_for_cfi(cfi)
        rows.extend(
            run_policy(
                args=args,
                policy_name=f"L0_garrido_{spec.label}_{spec.family}",
                ladder_level="L0_garrido",
                stage="thesis_static",
                action_space_mode="thesis_factorized",
                inventory_period_mode="thesis_strict",
                action=thesis_design_action(spec),
                replications=args.replications,
                seed_offset=cfi * 100_000,
                env_override={
                    "enabled_risks": set(spec.enabled_risks),
                    "risk_overrides": dict(spec.risk_overrides),
                },
                metadata={
                    "cfi": spec.cfi,
                    "source_cfi": spec.source_cfi,
                    "family": spec.family,
                    "horizon_hours_thesis_design": spec.horizon_hours,
                    "common_inventory_period": spec.inventory_replenishment_period,
                    "op3_inventory_period": spec.inventory_replenishment_period,
                    "op5_inventory_period": spec.inventory_replenishment_period,
                    "op9_inventory_period": spec.inventory_replenishment_period,
                    "shifts": spec.shifts,
                },
            )
        )
    return rows


def l1a_uniform_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    period_levels: list[int | None] = [None, *THESIS_INVENTORY_PERIODS]
    for period in period_levels:
        period_label = "I0" if period is None else f"I{period}"
        for shifts in (1, 2, 3):
            rows.extend(
                run_policy(
                    args=args,
                    policy_name=f"L1a_uniform_{period_label}_S{shifts}",
                    ladder_level="L1a_uniform_IxS",
                    stage="static_grid",
                    action_space_mode="thesis_factorized",
                    inventory_period_mode="thesis_strict",
                    action=thesis_factorized_action(period, shifts),
                    replications=args.replications,
                    seed_offset=1_000_000 + (0 if period is None else period) * 10 + shifts,
                    metadata={
                        "cfi": "",
                        "source_cfi": "",
                        "family": "uniform_inventory_capacity_grid",
                        "horizon_hours_thesis_design": "",
                        "common_inventory_period": "" if period is None else period,
                        "op3_inventory_period": "" if period is None else period,
                        "op5_inventory_period": "" if period is None else period,
                        "op9_inventory_period": "" if period is None else period,
                        "shifts": shifts,
                    },
                )
            )
    return rows


def l1b_per_node_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    period_levels: list[int | None] = [None, *THESIS_INVENTORY_PERIODS]
    configs = []
    for op3_period in period_levels:
        for op5_period in period_levels:
            for op9_period in period_levels:
                for shifts in (1, 2, 3):
                    configs.append((op3_period, op5_period, op9_period, shifts))

    for index, (op3_period, op5_period, op9_period, shifts) in enumerate(configs):
        action = per_node_action(op3_period, op5_period, op9_period, shifts)
        labels = [
            "I0" if period is None else f"I{period}"
            for period in (op3_period, op5_period, op9_period)
        ]
        rows.extend(
            run_policy(
                args=args,
                policy_name=f"L1b_per_node_{labels[0]}_{labels[1]}_{labels[2]}_S{shifts}",
                ladder_level="L1b_per_node_IxS",
                stage="screening",
                action_space_mode="factorized",
                inventory_period_mode="per_node",
                action=action,
                replications=args.l1b_screening_replications,
                seed_offset=2_000_000 + index * 100,
                metadata={
                    "cfi": "",
                    "source_cfi": "",
                    "family": "per_node_inventory_capacity_grid",
                    "horizon_hours_thesis_design": "",
                    "common_inventory_period": "",
                    "op3_inventory_period": "" if op3_period is None else op3_period,
                    "op5_inventory_period": "" if op5_period is None else op5_period,
                    "op9_inventory_period": "" if op9_period is None else op9_period,
                    "shifts": shifts,
                },
            )
        )

    if args.l1b_top_k <= 0:
        return rows

    screening_summary = summarize_group(rows)
    top_policies = {
        row["policy"]
        for row in sorted(
            screening_summary,
            key=lambda row: (
                row["fill_rate_order_level_mean"],
                row["order_level_ret_mean"],
            ),
            reverse=True,
        )[: args.l1b_top_k]
    }
    config_by_policy = {}
    for index, (op3_period, op5_period, op9_period, shifts) in enumerate(configs):
        labels = [
            "I0" if period is None else f"I{period}"
            for period in (op3_period, op5_period, op9_period)
        ]
        policy = f"L1b_per_node_{labels[0]}_{labels[1]}_{labels[2]}_S{shifts}"
        config_by_policy[policy] = (index, op3_period, op5_period, op9_period, shifts)

    top_reps = args.replications if args.l1b_top_replications is None else args.l1b_top_replications
    for policy in sorted(top_policies):
        index, op3_period, op5_period, op9_period, shifts = config_by_policy[policy]
        rows.extend(
            run_policy(
                args=args,
                policy_name=f"{policy}_top",
                ladder_level="L1b_per_node_IxS",
                stage="top_k",
                action_space_mode="factorized",
                inventory_period_mode="per_node",
                action=per_node_action(op3_period, op5_period, op9_period, shifts),
                replications=top_reps,
                seed_offset=3_000_000 + index * 100,
                metadata={
                    "cfi": "",
                    "source_cfi": "",
                    "family": "per_node_inventory_capacity_grid",
                    "horizon_hours_thesis_design": "",
                    "common_inventory_period": "",
                    "op3_inventory_period": "" if op3_period is None else op3_period,
                    "op5_inventory_period": "" if op5_period is None else op5_period,
                    "op9_inventory_period": "" if op9_period is None else op9_period,
                    "shifts": shifts,
                },
            )
        )
    return rows


def main() -> int:
    args = build_parser().parse_args()
    label = args.label or utc_now_iso().replace(":", "").replace("+", "Z")
    run_dir = args.output_root / label
    run_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    if "L0_garrido" in args.levels:
        rows.extend(l0_garrido_rows(args))
    if "L1a_uniform_IxS" in args.levels:
        rows.extend(l1a_uniform_rows(args))
    if "L1b_per_node_IxS" in args.levels:
        rows.extend(l1b_per_node_rows(args))

    policy_summary = summarize_group(rows)
    summary = {
        "created_at": utc_now_iso(),
        "levels": list(args.levels),
        "replications": args.replications,
        "garrido_cfis": args.garrido_cfis,
        "reward_mode": args.reward_mode,
        "risk_level": args.risk_level,
        "stochastic_pt": args.stochastic_pt,
        "observation_version": args.observation_version,
        "observation_mode": args.observation_mode,
        "step_size_hours": args.step_size_hours,
        "max_steps": args.max_steps,
        "l1b_screening_replications": args.l1b_screening_replications,
        "l1b_top_k": args.l1b_top_k,
        "l1b_top_replications": (
            args.replications if args.l1b_top_replications is None else args.l1b_top_replications
        ),
        "policy_count": len(policy_summary),
        "episode_count": len(rows),
        "best_policy_by_fill_rate": (
            max(policy_summary, key=lambda row: row["fill_rate_order_level_mean"])
            if policy_summary
            else None
        ),
        "policy_summary": policy_summary,
    }
    write_csv(run_dir / "episode_metrics.csv", rows)
    write_csv(run_dir / "policy_summary.csv", policy_summary)
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary["best_policy_by_fill_rate"], indent=2))
    print(f"Saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
