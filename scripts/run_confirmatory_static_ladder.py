#!/usr/bin/env python3
"""Confirmatory static decision-ladder run with pre-registered policies.

This is the follow-up to the exploratory unified panel. It avoids best-of-N
selection by evaluating a fixed policy set under new seeds:

- Garrido matched DOE baseline
- Best thesis-pure inventory policy from the exploratory panel: I672,S1
- Best thesis-pure capacity-only policy from the exploratory panel: I0,S3
- Crossed uniform policy: I504,S3
- Best per-node policy from the exploratory panel: I1344,I504,I504,S3

The output schema matches the unified evaluation enough for downstream pandas
analysis, but this script writes rows incrementally because the DES can run for
long enough that partial progress is valuable.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import RISK_OCCURRENCE_MODE_OPTIONS  # noqa: E402
from supply_chain.external_env_interface import (
    get_episode_terminal_metrics,
    make_dkana_thesis_faithful_env,
)
from supply_chain.thesis_design import design_spec_for_cfi, parse_cf_range
from scripts.run_unified_thesis_evaluation import (
    base_kwargs,
    fixed_action_fn,
    per_node_action,
    thesis_design_action,
    thesis_factorized_action,
)

POLICIES: dict[str, dict[str, Any]] = {
    "garrido_matched_DOE_baseline": {
        "kind": "matched_doe",
        "space": "thesis_matched",
        "action_space_mode": "thesis_factorized",
        "inventory_period_mode": "thesis_strict",
        "action": "matched",
    },
    "pure_inventory_I672_S1": {
        "kind": "static",
        "space": "thesis_pure_inventory",
        "action_space_mode": "thesis_factorized",
        "inventory_period_mode": "thesis_strict",
        "action": thesis_factorized_action(672, 1),
    },
    "pure_capacity_I0_S3": {
        "kind": "static",
        "space": "thesis_pure_capacity",
        "action_space_mode": "thesis_factorized",
        "inventory_period_mode": "thesis_strict",
        "action": thesis_factorized_action(None, 3),
    },
    "crossed_uniform_I504_S3": {
        "kind": "static",
        "space": "crossed_uniform",
        "action_space_mode": "thesis_factorized",
        "inventory_period_mode": "thesis_strict",
        "action": thesis_factorized_action(504, 3),
    },
    "per_node_I1344_I504_I504_S3": {
        "kind": "static",
        "space": "per_node",
        "action_space_mode": "factorized",
        "inventory_period_mode": "per_node",
        "action": per_node_action(1344, 504, 504, 3),
    },
}

ROW_FIELDS = [
    "policy",
    "kind",
    "space",
    "cfi",
    "source_cfi",
    "family",
    "replication",
    "eval_seed",
    "reward_total",
    "fill_rate_order_level",
    "order_level_ret_mean",
    "backorder_rate_order_level",
    "pct_steps_S1",
    "pct_steps_S2",
    "pct_steps_S3",
    "steps",
]

METRICS = {
    "fill": "fill_rate_order_level",
    "ret": "order_level_ret_mean",
    "reward": "reward_total",
}


def rollout(
    args: argparse.Namespace, *, cfi: int, rep: int, policy_name: str
) -> dict[str, Any]:
    spec = design_spec_for_cfi(cfi)
    policy = POLICIES[policy_name]
    action = policy["action"]
    if isinstance(action, str) and action == "matched":
        action = thesis_design_action(spec)
    kwargs = base_kwargs(args)
    kwargs.update(
        {
            "action_space_mode": policy["action_space_mode"],
            "inventory_period_mode": policy["inventory_period_mode"],
            "initial_action": action,
            "enabled_risks": set(spec.enabled_risks),
            "risk_overrides": dict(spec.risk_overrides),
        }
    )
    env = make_dkana_thesis_faithful_env(**kwargs)
    eval_seed = args.base_seed + cfi * 1000 + rep
    obs, info = env.reset(seed=eval_seed)
    action_fn = fixed_action_fn(np.asarray(action))
    terminated = truncated = False
    reward_total = 0.0
    steps = 0
    shift_counts = {1: 0, 2: 0, 3: 0}
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(action_fn(obs, info))
        reward_total += float(reward)
        if info.get("action_phase") != "weekly_decision":
            continue
        steps += 1
        shift = int(info.get("thesis_decision", {}).get("assembly_shifts", 1))
        shift_counts[shift] = shift_counts.get(shift, 0) + 1
    terminal = get_episode_terminal_metrics(env)
    total_steps = max(1, steps)
    env.close()
    return {
        "policy": policy_name,
        "kind": policy["kind"],
        "space": policy["space"],
        "cfi": cfi,
        "source_cfi": spec.source_cfi,
        "family": spec.family,
        "replication": rep,
        "eval_seed": eval_seed,
        "reward_total": reward_total,
        "fill_rate_order_level": float(terminal["fill_rate_order_level"]),
        "order_level_ret_mean": float(terminal["order_level_ret_mean"]),
        "backorder_rate_order_level": float(terminal["backorder_rate_order_level"]),
        "pct_steps_S1": 100.0 * shift_counts.get(1, 0) / total_steps,
        "pct_steps_S2": 100.0 * shift_counts.get(2, 0) / total_steps,
        "pct_steps_S3": 100.0 * shift_counts.get(3, 0) / total_steps,
        "steps": steps,
    }


def bootstrap_ci(
    values: np.ndarray, *, rng: np.random.Generator, draws: int
) -> tuple[float, float]:
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[idx].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).astype(float))


def paired_contrasts(
    df: pd.DataFrame, *, bootstrap_draws: int, seed: int
) -> pd.DataFrame:
    scenario_df = (
        df.groupby(["policy", "kind", "space", "cfi", "source_cfi", "family"])
        .agg({metric: "mean" for metric in METRICS.values()})
        .reset_index()
    )
    specs = [
        (
            "crossed_uniform_minus_matched_DOE",
            "crossed_uniform_I504_S3",
            "garrido_matched_DOE_baseline",
        ),
        (
            "crossed_uniform_minus_pure_inventory",
            "crossed_uniform_I504_S3",
            "pure_inventory_I672_S1",
        ),
        (
            "crossed_uniform_minus_pure_capacity",
            "crossed_uniform_I504_S3",
            "pure_capacity_I0_S3",
        ),
        (
            "per_node_minus_crossed_uniform",
            "per_node_I1344_I504_I504_S3",
            "crossed_uniform_I504_S3",
        ),
        (
            "pure_inventory_minus_matched_DOE",
            "pure_inventory_I672_S1",
            "garrido_matched_DOE_baseline",
        ),
    ]
    rng = np.random.default_rng(seed)
    rows = []
    for label, a_policy, b_policy in specs:
        for metric_key, metric in METRICS.items():
            a = scenario_df[scenario_df.policy.eq(a_policy)][["cfi", metric]]
            b = scenario_df[scenario_df.policy.eq(b_policy)][["cfi", metric]]
            merged = a.merge(b, on="cfi", suffixes=("_a", "_b"))
            diff = (merged[f"{metric}_a"] - merged[f"{metric}_b"]).to_numpy(float)
            ci_low, ci_high = bootstrap_ci(diff, rng=rng, draws=bootstrap_draws)
            wilcoxon_p = math.nan
            paired_t_p = math.nan
            if stats is not None:
                try:
                    wilcoxon_p = float(stats.wilcoxon(diff, zero_method="pratt").pvalue)
                except ValueError:
                    wilcoxon_p = 1.0
                paired_t_p = float(stats.ttest_1samp(diff, 0.0).pvalue)
            rows.append(
                {
                    "contrast": label,
                    "metric": metric_key,
                    "policy_a": a_policy,
                    "policy_b": b_policy,
                    "scenario_n": int(len(diff)),
                    "mean_delta": float(diff.mean()),
                    "median_delta": float(np.median(diff)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "wilcoxon_p": wilcoxon_p,
                    "paired_t_p": paired_t_p,
                    "wins": int(np.sum(diff > 0)),
                    "losses": int(np.sum(diff < 0)),
                    "ties": int(np.sum(diff == 0)),
                }
            )
    return pd.DataFrame(rows)


def write_outputs(out_dir: Path, args: argparse.Namespace) -> None:
    df = pd.read_csv(out_dir / "confirmatory_per_scenario.csv")
    summary = (
        df.groupby(["policy", "kind", "space"])
        .agg(
            fill_mean=("fill_rate_order_level", "mean"),
            ret_mean=("order_level_ret_mean", "mean"),
            reward_mean=("reward_total", "mean"),
            n=("fill_rate_order_level", "size"),
        )
        .reset_index()
        .sort_values(["fill_mean", "ret_mean", "reward_mean"], ascending=False)
    )
    summary.to_csv(out_dir / "confirmatory_summary.csv", index=False)
    contrasts = paired_contrasts(
        df, bootstrap_draws=args.bootstrap_draws, seed=args.base_seed + 99
    )
    contrasts.to_csv(out_dir / "confirmatory_contrasts.csv", index=False)

    lines = [
        "# Confirmatory Static Ladder",
        "",
        f"Created UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"Panel: `{args.panel_cfis}`, reps: `{args.replications}`, base_seed: `{args.base_seed}`, max_steps: `{args.max_steps}`.",
        "",
        "Rows are evaluated with a common-seed paired panel, not strict CRN.",
        "",
        "## Summary",
        "",
        "| policy | fill | ReT | reward | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| `{row.policy}` | {row.fill_mean:.4f} | {row.ret_mean:.4f} | {row.reward_mean:.2f} | {int(row.n)} |"
        )
    lines += [
        "",
        "## Scenario-Level Paired Contrasts",
        "",
        "| contrast | metric | mean delta | 95% CI | Wilcoxon p | wins/losses/ties |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in contrasts.iterrows():
        lines.append(
            f"| {row.contrast} | {row.metric} | {row.mean_delta:.4f} | "
            f"[{row.ci95_low:.4f}, {row.ci95_high:.4f}] | {row.wilcoxon_p:.5f} | "
            f"{int(row.wins)}/{int(row.losses)}/{int(row.ties)} |"
        )
    (out_dir / "CONFIRMATORY_STATIC_LADDER.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/benchmarks/confirmatory_static_ladder"),
    )
    parser.add_argument("--panel-cfis", default="31-90")
    parser.add_argument("--replications", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=910000)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument("--reward-mode", default="ReT_cd_v1")
    parser.add_argument("--risk-level", default="increased")
    parser.add_argument("--observation-version", default="v5")
    parser.add_argument("--observation-mode", default="env_sdm_history_reward")
    parser.add_argument("--stochastic-pt", action="store_true", default=True)
    parser.add_argument(
        "--stochastic-pt-spread",
        type=float,
        default=1.0,
        help=(
            "Scales stochastic processing-time variability when --stochastic-pt "
            "is enabled. Historical default 1.0 is Tri(0.75*PT, PT, 1.5*PT)."
        ),
    )
    parser.add_argument(
        "--stochastic-pt-mean-preserving",
        action="store_true",
        help=(
            "Use a symmetric triangular PT envelope around the thesis PT, so "
            "changing --stochastic-pt-spread changes variance without changing "
            "the expected processing time."
        ),
    )
    parser.add_argument(
        "--raw-material-flow-mode",
        default="legacy_validated",
        help="Raw-material flow semantics for post-fix thesis-inventory reruns.",
    )
    parser.add_argument(
        "--raw-material-order-up-to-multiplier", type=float, default=2.0
    )
    parser.add_argument(
        "--risk-occurrence-mode",
        choices=RISK_OCCURRENCE_MODE_OPTIONS,
        default="legacy_renewal",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()

    label = args.label or datetime.now(timezone.utc).strftime(
        "confirmatory_static_%Y%m%dT%H%M%SZ"
    )
    out_dir = args.output_root / label
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = parse_cf_range(args.panel_cfis)
    total = len(panel) * args.replications * len(POLICIES)
    done = 0
    csv_path = out_dir / "confirmatory_per_scenario.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        handle.flush()
        for policy in POLICIES:
            for cfi in panel:
                for rep in range(args.replications):
                    row = rollout(args, cfi=cfi, rep=rep, policy_name=policy)
                    writer.writerow(row)
                    done += 1
                if done % 100 < args.replications:
                    handle.flush()
                    print(
                        f"progress {done}/{total} ({100*done/total:.0f}%)", flush=True
                    )

    write_outputs(out_dir, args)
    print(out_dir / "CONFIRMATORY_STATIC_LADDER.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
