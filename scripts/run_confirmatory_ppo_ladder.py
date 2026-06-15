#!/usr/bin/env python3
"""Confirmatory static + PPO evaluation on the same scenario panel.

This script is intentionally evaluation-only: it does not train PPO. It loads
pretrained PPO models plus VecNormalize statistics and scores them against the
pre-registered static ladder policies from ``run_confirmatory_static_ladder``.
Rows are evaluated on the same common-seed Cf panel. The design is not strict
CRN because exogenous streams are not action-invariant in the DES.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import zipfile
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

from scripts.run_confirmatory_static_ladder import (  # noqa: E402
    METRICS,
    POLICIES,
    ROW_FIELDS,
    bootstrap_ci,
)
from scripts.run_unified_thesis_evaluation import (  # noqa: E402
    base_kwargs,
    fixed_action_fn,
    load_ppo,
    model_action_fn,
    thesis_design_action,
)
from supply_chain.external_env_interface import (  # noqa: E402
    get_episode_terminal_metrics,
    make_dkana_thesis_faithful_env,
)
from supply_chain.thesis_design import design_spec_for_cfi, parse_cf_range  # noqa: E402


def discover_ppo_runs(ppo_root: Path) -> list[dict[str, Any]]:
    """Find pretrained PPO zip/VecNormalize pairs under a Kaggle/local output root."""
    candidates: list[Path] = []
    candidates.extend(ppo_root.glob("kaggle_ppo_bestshot_seed*"))
    candidates.extend(ppo_root.glob("out/kaggle_ppo_bestshot_seed*"))
    candidates.extend(ppo_root.glob("**/kaggle_ppo_bestshot_seed*"))
    seen: set[Path] = set()
    runs: list[dict[str, Any]] = []
    for run_dir in sorted(candidates):
        run_dir = run_dir.resolve()
        if run_dir in seen:
            continue
        seen.add(run_dir)
        model_zip = run_dir / "ppo_mlp_thesis_decision.zip"
        model_dir = run_dir / "ppo_mlp_thesis_decision"
        if not model_zip.exists() and model_dir.is_dir():
            with zipfile.ZipFile(
                model_zip, "w", compression=zipfile.ZIP_DEFLATED
            ) as zf:
                for file_path in model_dir.rglob("*"):
                    if file_path.is_file():
                        zf.write(file_path, file_path.relative_to(model_dir))
        vecnorm = run_dir / "vecnormalize.pkl"
        summary = run_dir / "summary.json"
        if not (model_zip.exists() and vecnorm.exists() and summary.exists()):
            continue
        meta = json.loads(summary.read_text())
        seed = meta.get("seed")
        if seed is None:
            m = re.search(r"seed(\d+)", run_dir.name)
            seed = int(m.group(1)) if m else len(runs)
        runs.append(
            {
                "name": f"ppo500k_seed{seed}",
                "kind": "ppo",
                "space": "ppo_thesis_factorized",
                "action_space_mode": meta.get("action_space_mode", "thesis_factorized"),
                "inventory_period_mode": meta.get(
                    "inventory_period_mode", "thesis_strict"
                ),
                "learn_initial_decision": bool(
                    meta.get("env_kwargs", {}).get("learn_initial_decision", False)
                ),
                "model_zip": model_zip,
                "vecnormalize": vecnorm,
                "train_timesteps": meta.get("train_timesteps"),
                "train_cfis": meta.get("train_cfis"),
            }
        )
    return runs


def rollout_with_action_fn(
    args: argparse.Namespace,
    *,
    cfi: int,
    rep: int,
    policy_name: str,
    policy: dict[str, Any],
    action_fn: Any,
    initial_action: np.ndarray | None,
) -> dict[str, Any]:
    spec = design_spec_for_cfi(cfi)
    kwargs = base_kwargs(args)
    kwargs.update(
        {
            "action_space_mode": policy["action_space_mode"],
            "inventory_period_mode": policy["inventory_period_mode"],
            "enabled_risks": set(spec.enabled_risks),
            "risk_overrides": dict(spec.risk_overrides),
            "learn_initial_decision": bool(policy.get("learn_initial_decision", False)),
        }
    )
    if initial_action is not None:
        kwargs["initial_action"] = initial_action

    env = make_dkana_thesis_faithful_env(**kwargs)
    eval_seed = args.base_seed + cfi * 1000 + rep
    obs, info = env.reset(seed=eval_seed)
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


def paired_contrasts(
    df: pd.DataFrame, *, bootstrap_draws: int, seed: int, ppo_names: list[str]
) -> pd.DataFrame:
    scenario_df = (
        df.groupby(["policy", "kind", "space", "cfi", "source_cfi", "family"])
        .agg({metric: "mean" for metric in METRICS.values()})
        .reset_index()
    )
    specs = [
        (
            "crossed_uniform_minus_pure_inventory",
            "crossed_uniform_I504_S3",
            "pure_inventory_I672_S1",
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
        (
            "crossed_uniform_minus_matched_DOE",
            "crossed_uniform_I504_S3",
            "garrido_matched_DOE_baseline",
        ),
    ]
    for ppo_name in ppo_names:
        specs.extend(
            [
                (
                    f"{ppo_name}_minus_pure_inventory",
                    ppo_name,
                    "pure_inventory_I672_S1",
                ),
                (
                    f"{ppo_name}_minus_crossed_uniform",
                    ppo_name,
                    "crossed_uniform_I504_S3",
                ),
                (
                    f"{ppo_name}_minus_matched_DOE",
                    ppo_name,
                    "garrido_matched_DOE_baseline",
                ),
            ]
        )
    rng = np.random.default_rng(seed)
    rows = []
    for label, a_policy, b_policy in specs:
        for metric_key, metric in METRICS.items():
            a = scenario_df[scenario_df.policy.eq(a_policy)][["cfi", metric]]
            b = scenario_df[scenario_df.policy.eq(b_policy)][["cfi", metric]]
            merged = a.merge(b, on="cfi", suffixes=("_a", "_b"))
            if merged.empty:
                continue
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


def write_outputs(
    out_dir: Path, args: argparse.Namespace, ppo_names: list[str]
) -> None:
    df = pd.read_csv(out_dir / "confirmatory_ppo_per_scenario.csv")
    summary = (
        df.groupby(["policy", "kind", "space"])
        .agg(
            fill_mean=("fill_rate_order_level", "mean"),
            ret_mean=("order_level_ret_mean", "mean"),
            reward_mean=("reward_total", "mean"),
            backorder_mean=("backorder_rate_order_level", "mean"),
            n=("fill_rate_order_level", "size"),
        )
        .reset_index()
        .sort_values(["fill_mean", "ret_mean", "reward_mean"], ascending=False)
    )
    summary.to_csv(out_dir / "confirmatory_ppo_summary.csv", index=False)
    contrasts = paired_contrasts(
        df,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.bootstrap_seed,
        ppo_names=ppo_names,
    )
    contrasts.to_csv(out_dir / "confirmatory_ppo_contrasts.csv", index=False)

    lines = [
        "# Confirmatory Static + PPO Ladder",
        "",
        f"Created UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"Panel: `{args.panel_cfis}`, reps: `{args.replications}`, "
        f"base_seed: `{args.base_seed}`, max_steps: `{args.max_steps}`.",
        "",
        "Rows are evaluated with a common-seed paired panel, not strict CRN.",
        "PPO models are loaded from pretrained 500k best-shot runs; no training is performed here.",
        "",
        "## Summary",
        "",
        "| policy | kind | fill | ReT | reward | n |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| `{row.policy}` | {row.kind} | {row.fill_mean:.4f} | "
            f"{row.ret_mean:.4f} | {row.reward_mean:.2f} | {int(row.n)} |"
        )
    lines.extend(
        [
            "",
            "## Primary PPO Contrasts",
            "",
            "| contrast | metric | mean delta | 95% CI | Wilcoxon p | wins/losses/ties |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    primary = contrasts[
        contrasts["contrast"].str.contains("ppo500k")
        & contrasts["contrast"].str.contains("pure_inventory|crossed_uniform")
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.contrast} | {row.metric} | {row.mean_delta:.4f} | "
            f"[{row.ci95_low:.4f}, {row.ci95_high:.4f}] | {row.wilcoxon_p:.5f} | "
            f"{row.wins}/{row.losses}/{row.ties} |"
        )
    (out_dir / "CONFIRMATORY_PPO_LADDER.md").write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--output-root", default="outputs/benchmarks/confirmatory_ppo_ladder"
    )
    parser.add_argument("--ppo-root", required=True)
    parser.add_argument("--panel-cfis", default="31-90")
    parser.add_argument("--replications", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=940000)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument("--reward-mode", default="ReT_cd_v1")
    parser.add_argument("--risk-level", default="increased")
    parser.add_argument("--observation-version", default="v5")
    parser.add_argument("--observation-mode", default="env_sdm_history_reward")
    parser.add_argument("--stochastic-pt", action="store_true")
    parser.add_argument(
        "--raw-material-flow-mode",
        default="legacy_validated",
        help="Raw-material flow semantics for post-fix thesis-inventory reruns.",
    )
    parser.add_argument(
        "--raw-material-order-up-to-multiplier", type=float, default=2.0
    )
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ppo_runs = discover_ppo_runs(Path(args.ppo_root))
    if not ppo_runs:
        raise FileNotFoundError(f"No PPO best-shot runs found under {args.ppo_root}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = args.label or f"confirmatory_ppo_{stamp}"
    out_dir = Path(args.output_root) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = parse_cf_range(args.panel_cfis)
    static_names = list(POLICIES)
    ppo_names = [run["name"] for run in ppo_runs]
    total = len(panel) * args.replications * (len(static_names) + len(ppo_runs))
    print(
        f"panel={args.panel_cfis}, reps={args.replications}, "
        f"static={len(static_names)}, ppo={len(ppo_runs)}, total={total}",
        flush=True,
    )
    print("ppo runs:", ", ".join(ppo_names), flush=True)

    ppo_action_fns: dict[str, Any] = {}
    for run in ppo_runs:
        model, vec = load_ppo(run["model_zip"], run["vecnormalize"])
        ppo_action_fns[run["name"]] = model_action_fn(model, vec)

    out_csv = out_dir / "confirmatory_ppo_per_scenario.csv"
    done = 0
    with out_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS)
        writer.writeheader()

        for cfi in panel:
            spec = design_spec_for_cfi(cfi)
            for rep in range(args.replications):
                for policy_name in static_names:
                    policy = POLICIES[policy_name]
                    action = policy["action"]
                    if isinstance(action, str) and action == "matched":
                        action = thesis_design_action(spec)
                    row = rollout_with_action_fn(
                        args,
                        cfi=cfi,
                        rep=rep,
                        policy_name=policy_name,
                        policy=policy,
                        action_fn=fixed_action_fn(np.asarray(action)),
                        initial_action=np.asarray(action),
                    )
                    writer.writerow(row)
                    done += 1

                for run in ppo_runs:
                    policy = {
                        "kind": "ppo",
                        "space": run["space"],
                        "action_space_mode": run["action_space_mode"],
                        "inventory_period_mode": run["inventory_period_mode"],
                        "learn_initial_decision": run["learn_initial_decision"],
                    }
                    row = rollout_with_action_fn(
                        args,
                        cfi=cfi,
                        rep=rep,
                        policy_name=run["name"],
                        policy=policy,
                        action_fn=ppo_action_fns[run["name"]],
                        initial_action=None,
                    )
                    writer.writerow(row)
                    done += 1

                if done % max(1, len(static_names) + len(ppo_runs)) == 0:
                    stream.flush()
                if done % 200 == 0:
                    print(
                        f"progress {done}/{total} ({100*done/total:.1f}%)", flush=True
                    )

    write_outputs(out_dir, args, ppo_names)
    print((out_dir / "CONFIRMATORY_PPO_LADDER.md").read_text(), flush=True)
    print(f"wrote: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
