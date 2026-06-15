#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.external_env_interface import (  # noqa: E402
    THESIS_INVENTORY_PERIODS,
    get_episode_terminal_metrics,
    make_dkana_thesis_faithful_env,
)
from supply_chain.thesis_design import design_spec_for_cfi, parse_cf_range  # noqa: E402

DEFAULT_REWARD_MODES = ("ReT_cd_v1", "ReT_seq_v1", "ReT_ladder_v1")
DEFAULT_OUTPUT_ROOT = Path("outputs/benchmarks/thesis_reward_surface")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def thesis_factorized_action(period: int | None, shifts: int) -> np.ndarray:
    period_level = 0 if period is None else THESIS_INVENTORY_PERIODS.index(period) + 1
    return np.array([period_level, shifts - 1], dtype=np.int64)


def policy_grid() -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for period in [None, *THESIS_INVENTORY_PERIODS]:
        period_label = "I0" if period is None else f"I{period}"
        for shifts in (1, 2, 3):
            policies.append(
                {
                    "policy": f"L1a_uniform_{period_label}_S{shifts}",
                    "period": period,
                    "period_label": period_label,
                    "shifts": shifts,
                    "action": thesis_factorized_action(period, shifts),
                }
            )
    return policies


def rankdata(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    rx = rankdata(x)
    ry = rankdata(y)
    if float(np.std(rx)) <= 0.0 or float(np.std(ry)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def scenario_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.garrido_cfis:
        return [
            {
                "scenario": args.risk_level,
                "cfi": "",
                "family": "common_risk_level",
                "env_override": {},
            }
        ]
    specs: list[dict[str, Any]] = []
    for cfi in parse_cf_range(args.garrido_cfis):
        spec = design_spec_for_cfi(cfi)
        specs.append(
            {
                "scenario": f"Cf{spec.cfi}_{spec.family}",
                "cfi": spec.cfi,
                "family": spec.family,
                "env_override": {
                    "enabled_risks": set(spec.enabled_risks),
                    "risk_overrides": dict(spec.risk_overrides),
                },
            }
        )
    return specs


def run_episode(
    *,
    args: argparse.Namespace,
    reward_mode: str,
    policy: dict[str, Any],
    scenario: dict[str, Any],
    replication: int,
    scenario_index: int,
    policy_index: int,
) -> dict[str, Any]:
    eval_seed = (
        int(args.seed) + scenario_index * 100_000 + policy_index * 1_000 + replication
    )
    kwargs = {
        "reward_mode": reward_mode,
        "risk_level": args.risk_level,
        "observation_version": args.observation_version,
        "observation_mode": args.observation_mode,
        "action_space_mode": "thesis_factorized",
        "inventory_period_mode": "thesis_strict",
        "step_size_hours": args.step_size_hours,
        "max_steps": args.max_steps,
        "stochastic_pt": args.stochastic_pt,
        "learn_initial_decision": False,
        "initial_action": policy["action"],
    }
    kwargs.update(scenario["env_override"])
    env = make_dkana_thesis_faithful_env(**kwargs)
    _obs, _info = env.reset(seed=eval_seed)
    terminated = truncated = False
    reward_total = 0.0
    steps = 0
    ret_ladder_total = 0.0
    ret_cd_total = 0.0
    ret_seq_total = 0.0
    while not (terminated or truncated):
        _obs, reward, terminated, truncated, info = env.step(policy["action"])
        reward_total += float(reward)
        ret_ladder_total += float(info.get("ret_ladder_step", 0.0))
        ret_cd_total += float(info.get("ret_cd_v1_step", info.get("ret_cd_step", 0.0)))
        ret_seq_total += float(info.get("ret_seq_step", 0.0))
        steps += 1

    terminal = get_episode_terminal_metrics(env)
    sim = getattr(env.unwrapped, "sim", None)
    pending_qty = (
        float(
            sum(
                float(getattr(order, "remaining_qty", 0.0))
                for order in getattr(sim, "pending_backorders", [])
            )
        )
        if sim is not None
        else 0.0
    )
    pending_count = (
        float(len(getattr(sim, "pending_backorders", []))) if sim is not None else 0.0
    )
    env.close()
    return {
        "reward_mode": reward_mode,
        "scenario": scenario["scenario"],
        "cfi": scenario["cfi"],
        "family": scenario["family"],
        "policy": policy["policy"],
        "period": "" if policy["period"] is None else policy["period"],
        "shifts": policy["shifts"],
        "action": json.dumps(np.asarray(policy["action"]).astype(int).tolist()),
        "replication": replication,
        "seed": eval_seed,
        "steps": steps,
        "reward_total": reward_total,
        "ret_ladder_total": ret_ladder_total,
        "ret_cd_total": ret_cd_total,
        "ret_seq_total": ret_seq_total,
        "fill_rate_order_level": terminal["fill_rate_order_level"],
        "backorder_rate_order_level": terminal["backorder_rate_order_level"],
        "order_level_ret_mean": terminal["order_level_ret_mean"],
        "pending_backorder_qty": pending_qty,
        "pending_backorders_count": pending_count,
    }


def summarize_policy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(row["reward_mode"], row["policy"]) for row in rows})
    for reward_mode, policy in keys:
        bucket = [
            row
            for row in rows
            if row["reward_mode"] == reward_mode and row["policy"] == policy
        ]
        first = bucket[0]
        out.append(
            {
                "reward_mode": reward_mode,
                "policy": policy,
                "period": first["period"],
                "shifts": first["shifts"],
                "episode_count": len(bucket),
                "reward_total_mean": float(
                    np.mean([row["reward_total"] for row in bucket])
                ),
                "reward_total_std": float(
                    np.std([row["reward_total"] for row in bucket], ddof=1)
                    if len(bucket) > 1
                    else 0.0
                ),
                "fill_rate_order_level_mean": float(
                    np.mean([row["fill_rate_order_level"] for row in bucket])
                ),
                "backorder_rate_order_level_mean": float(
                    np.mean([row["backorder_rate_order_level"] for row in bucket])
                ),
                "order_level_ret_mean": float(
                    np.mean([row["order_level_ret_mean"] for row in bucket])
                ),
                "pending_backorder_qty_mean": float(
                    np.mean([row["pending_backorder_qty"] for row in bucket])
                ),
                "pending_backorders_count_mean": float(
                    np.mean([row["pending_backorders_count"] for row in bucket])
                ),
            }
        )
    return out


def summarize_reward_modes(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for reward_mode in sorted({row["reward_mode"] for row in policy_rows}):
        bucket = [row for row in policy_rows if row["reward_mode"] == reward_mode]
        rewards = [float(row["reward_total_mean"]) for row in bucket]
        rets = [float(row["order_level_ret_mean"]) for row in bucket]
        fill = [float(row["fill_rate_order_level_mean"]) for row in bucket]
        backlog_good = [-float(row["pending_backorder_qty_mean"]) for row in bucket]
        best_reward = max(bucket, key=lambda row: float(row["reward_total_mean"]))
        spread = float(max(rewards) - min(rewards))
        mean_abs = float(np.mean(np.abs(rewards))) if rewards else 0.0
        spread_ratio = spread / max(mean_abs, 1e-9)
        ret_corr = spearman(rewards, rets)
        fill_corr = spearman(rewards, fill)
        backlog_corr = spearman(rewards, backlog_good)
        s1_collapse_penalty = 0.5 if int(best_reward["shifts"]) == 1 else 0.0
        diagnostic_score = (
            spread_ratio
            + max(0.0, ret_corr if not np.isnan(ret_corr) else 0.0)
            + max(0.0, backlog_corr if not np.isnan(backlog_corr) else 0.0)
            - s1_collapse_penalty
        )
        out.append(
            {
                "reward_mode": reward_mode,
                "policy_count": len(bucket),
                "reward_min": float(min(rewards)),
                "reward_max": float(max(rewards)),
                "reward_mean_abs": mean_abs,
                "reward_spread": spread,
                "reward_spread_ratio": spread_ratio,
                "spearman_reward_vs_order_level_ret": ret_corr,
                "spearman_reward_vs_fill": fill_corr,
                "spearman_reward_vs_negative_pending_backlog": backlog_corr,
                "best_policy_by_reward": best_reward["policy"],
                "best_policy_shifts": int(best_reward["shifts"]),
                "best_policy_order_level_ret": float(
                    best_reward["order_level_ret_mean"]
                ),
                "best_policy_fill_rate": float(
                    best_reward["fill_rate_order_level_mean"]
                ),
                "best_policy_pending_backorder_qty": float(
                    best_reward["pending_backorder_qty_mean"]
                ),
                "s1_collapse_penalty": s1_collapse_penalty,
                "diagnostic_score": diagnostic_score,
            }
        )
    return sorted(out, key=lambda row: row["diagnostic_score"], reverse=True)


def render_report(
    args: argparse.Namespace,
    mode_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Thesis Reward Surface Audit",
        "",
        "This is a reward-shaping diagnostic, not a final policy claim.  It asks "
        "whether each training reward creates separable signal over the same "
        "`thesis_factorized [6,3]` static action grid.",
        "",
        "Raw `reward_total` is compared only within each reward mode.  Cross-mode "
        "interpretation uses spread, rank correlation with external metrics, and "
        "the selected best policy.",
        "",
        "## Config",
        "",
        f"- reward_modes: `{', '.join(args.reward_modes)}`",
        f"- risk_level: `{args.risk_level}`",
        f"- garrido_cfis: `{args.garrido_cfis or 'none/common risk level'}`",
        f"- replications: `{args.replications}`",
        f"- max_steps: `{args.max_steps}`",
        f"- stochastic_pt: `{args.stochastic_pt}`",
        "",
        "## Reward-Mode Ranking",
        "",
        "| reward_mode | spread_ratio | rho(ReT) | rho(-backlog) | best_policy | best fill | best ReT | diagnostic_score |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in mode_rows:
        lines.append(
            f"| `{row['reward_mode']}` | {row['reward_spread_ratio']:.4f} | "
            f"{row['spearman_reward_vs_order_level_ret']:.4f} | "
            f"{row['spearman_reward_vs_negative_pending_backlog']:.4f} | "
            f"`{row['best_policy_by_reward']}` | {row['best_policy_fill_rate']:.4f} | "
            f"{row['best_policy_order_level_ret']:.4f} | {row['diagnostic_score']:.4f} |"
        )
    lines.extend(["", "## Top Policies Per Reward Mode", ""])
    for reward_mode in sorted({row["reward_mode"] for row in policy_rows}):
        lines.extend([f"### `{reward_mode}`", ""])
        top = sorted(
            [row for row in policy_rows if row["reward_mode"] == reward_mode],
            key=lambda row: float(row["reward_total_mean"]),
            reverse=True,
        )[:5]
        lines.extend(
            [
                "| policy | reward_mean | fill | ReT | pending_backlog |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in top:
            lines.append(
                f"| `{row['policy']}` | {row['reward_total_mean']:.4f} | "
                f"{row['fill_rate_order_level_mean']:.4f} | "
                f"{row['order_level_ret_mean']:.4f} | "
                f"{row['pending_backorder_qty_mean']:.1f} |"
            )
        lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--reward-modes",
        nargs="+",
        default=list(DEFAULT_REWARD_MODES),
        help="Reward modes to compare over the same static thesis action grid.",
    )
    parser.add_argument("--replications", type=int, default=3)
    parser.add_argument("--seed", type=int, default=550000)
    parser.add_argument(
        "--garrido-cfis",
        default=None,
        help="Optional Cf range/list. If omitted, use one common risk_level scenario.",
    )
    parser.add_argument("--risk-level", default="increased")
    parser.add_argument("--observation-version", default="v5")
    parser.add_argument("--observation-mode", default="env_sdm_history_reward")
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--stochastic-pt", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_label = args.label or f"reward_surface_{utc_stamp()}"
    run_dir = args.output_root / run_label
    run_dir.mkdir(parents=True, exist_ok=False)
    policies = policy_grid()
    scenarios = scenario_specs(args)
    total = len(args.reward_modes) * len(scenarios) * len(policies) * args.replications
    print(
        f"reward_modes={len(args.reward_modes)}, scenarios={len(scenarios)}, "
        f"policies={len(policies)}, reps={args.replications}, total={total}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    done = 0
    for reward_mode in args.reward_modes:
        for scenario_index, scenario in enumerate(scenarios):
            for policy_index, policy in enumerate(policies):
                for replication in range(args.replications):
                    rows.append(
                        run_episode(
                            args=args,
                            reward_mode=reward_mode,
                            policy=policy,
                            scenario=scenario,
                            replication=replication,
                            scenario_index=scenario_index,
                            policy_index=policy_index,
                        )
                    )
                    done += 1
                    if done % max(1, args.progress_every) == 0:
                        print(f"progress {done}/{total}", flush=True)

    policy_rows = summarize_policy(rows)
    mode_rows = summarize_reward_modes(policy_rows)
    report = render_report(args, mode_rows, policy_rows)
    write_csv(run_dir / "episode_metrics.csv", rows)
    write_csv(run_dir / "policy_summary.csv", policy_rows)
    write_csv(run_dir / "reward_mode_summary.csv", mode_rows)
    write_json(
        run_dir / "summary.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "label": run_label,
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "episode_count": len(rows),
            "policy_count": len(policy_rows),
            "reward_mode_summary": mode_rows,
        },
    )
    (run_dir / "REWARD_SURFACE_AUDIT.md").write_text(report + "\n", encoding="utf-8")
    print(report, flush=True)
    print(f"Saved to: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
