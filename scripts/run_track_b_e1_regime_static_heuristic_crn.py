#!/usr/bin/env python3
"""E1 go/no-go Track B regime-table and heuristic CRN baseline.

This script evaluates zero-learning comparators against the canonical Track B
v7 setting. It reuses the dense-static CRN episode plan so each policy sees the
same `(seed, episode, eval_seed)` keys:

- common static constants over shift/op10/op12;
- a greedy regime-conditioned lookup table over the true adaptive regime;
- existing Track B heuristics plus a forecast-threshold heuristic.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_b_dense_crn_static import (  # noqa: E402
    DEFAULT_RUN_DIR,
    StaticPolicy,
    action_for,
    build_env_kwargs,
    fallback_eval_plan,
    infer_config,
    infer_eval_plan,
    parse_float_list,
    parse_int_list,
)
from scripts.run_track_b_smoke import (  # noqa: E402
    _finalize_episode_row,
    aggregate_policy_metrics,
    aggregate_seed_metrics,
    append_order_ledger_rows,
    extract_downstream_multipliers,
    get_episode_terminal_metrics,
    init_cd_totals,
    update_cd_totals,
)
from scripts.track_b_heuristics import make_heuristic_defaults  # noqa: E402
from supply_chain.config import ADAPTIVE_BENCHMARK_REGIMES  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.external_env_interface import make_track_b_env  # noqa: E402


METRICS = [
    "reward_total",
    "order_level_ret_mean",
    "flow_fill_rate",
    "flow_backorder_rate",
    "terminal_rolling_fill_rate_4w",
    "terminal_rolling_backorder_rate_4w",
    "assembly_cost_index",
    "assembly_hours_total",
    "ret_garrido2024_raw_total",
    "ret_garrido2024_train_total",
    "ret_garrido2024_sigmoid_total",
    "ret_garrido2024_sigmoid_mean",
    "terminal_kappa_dot",
    "terminal_epsilon_avg",
    "order_ret_excel",
    "order_lost_rate",
    "order_service_loss_auc_per_order",
    "order_ctj_p99",
    "order_rpj_p99",
    "order_dpj_p99",
    "order_delivered_rations",
    "order_demanded_rations",
]


@dataclass(frozen=True)
class RegimeTablePolicy:
    label: str
    by_regime: dict[str, StaticPolicy]


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return Path("outputs/experiments") / f"track_b_e1_regime_static_heuristic_crn_{stamp}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def candidate_policies(args: argparse.Namespace) -> list[StaticPolicy]:
    policies = [
        StaticPolicy(shift=shift, op10_mult=op10, op12_mult=op12)
        for shift in parse_int_list(args.shifts)
        for op10 in parse_float_list(args.op10_mults)
        for op12 in parse_float_list(args.op12_mults)
    ]
    return policies[: int(args.max_fit_policies)] if args.max_fit_policies else policies


def current_regime(env: Any) -> str:
    sim = getattr(env.unwrapped, "sim", None)
    return str(getattr(sim, "adaptive_regime", ADAPTIVE_BENCHMARK_REGIMES[0]))


def run_action_episode(
    *,
    policy_label: str,
    action_fn: Callable[[np.ndarray, dict[str, Any], Any], dict[str, float | int] | np.ndarray],
    eval_item: dict[str, int],
    args: argparse.Namespace,
    env_kwargs: dict[str, Any],
    order_ledger_rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    env = make_track_b_env(**env_kwargs)
    obs, info = env.reset(seed=int(eval_item["eval_seed"]))
    terminated = truncated = False
    reward_total = 0.0
    demanded_total = 0.0
    backorder_qty_total = 0.0
    steps = 0
    shift_counts = {1: 0, 2: 0, 3: 0}
    op10_multipliers: list[float] = []
    op12_multipliers: list[float] = []
    cd_totals = init_cd_totals()
    final_info = info

    while not (terminated or truncated):
        action = action_fn(obs, final_info, env)
        obs, reward, terminated, truncated, final_info = env.step(action)
        reward_total += float(reward)
        update_cd_totals(cd_totals, final_info)
        demanded_total += float(final_info.get("new_demanded", 0.0))
        backorder_qty_total += float(final_info.get("new_backorder_qty", 0.0))
        shift_counts[int(final_info.get("shifts_active", 1))] += 1
        op10_mult, op12_mult = extract_downstream_multipliers(final_info)
        op10_multipliers.append(op10_mult)
        op12_multipliers.append(op12_mult)
        steps += 1

    row = _finalize_episode_row(
        policy=policy_label,
        seed=int(eval_item["seed"]),
        episode=int(eval_item["episode"]),
        eval_seed=int(eval_item["eval_seed"]),
        steps=steps,
        reward_total=reward_total,
        demanded_total=demanded_total,
        backorder_qty_total=backorder_qty_total,
        shift_counts=shift_counts,
        op10_multipliers=op10_multipliers,
        op12_multipliers=op12_multipliers,
        track_b_context=final_info["state_constraint_context"]["track_b_context"],
        terminal_metrics=get_episode_terminal_metrics(env),
        final_info=final_info,
        cd_totals=cd_totals,
        full_episode_metrics=compute_episode_metrics(env.unwrapped.sim),
    )
    append_order_ledger_rows(
        order_ledger_rows,
        env,
        policy=policy_label,
        seed=int(eval_item["seed"]),
        episode=int(eval_item["episode"]),
        eval_seed=int(eval_item["eval_seed"]),
    )
    env.close()
    return row


def evaluate_fixed_static(
    policy: StaticPolicy,
    *,
    eval_plan: list[dict[str, int]],
    args: argparse.Namespace,
    env_kwargs: dict[str, Any],
    order_ledger_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    action = action_for(policy)
    return [
        run_action_episode(
            policy_label=policy.label,
            action_fn=lambda _obs, _info, _env, action=action: action,
            eval_item=item,
            args=args,
            env_kwargs=env_kwargs,
            order_ledger_rows=order_ledger_rows,
        )
        for item in eval_plan
    ]


def evaluate_regime_table(
    policy: RegimeTablePolicy,
    *,
    eval_plan: list[dict[str, int]],
    args: argparse.Namespace,
    env_kwargs: dict[str, Any],
    order_ledger_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    actions = {regime: action_for(static) for regime, static in policy.by_regime.items()}
    fallback = actions[ADAPTIVE_BENCHMARK_REGIMES[0]]

    def choose(_obs: np.ndarray, _info: dict[str, Any], env: Any) -> dict[str, float | int]:
        return actions.get(current_regime(env), fallback)

    return [
        run_action_episode(
            policy_label=policy.label,
            action_fn=choose,
            eval_item=item,
            args=args,
            env_kwargs=env_kwargs,
            order_ledger_rows=order_ledger_rows,
        )
        for item in eval_plan
    ]


def evaluate_heuristic(
    label: str,
    heuristic: Any,
    *,
    eval_plan: list[dict[str, int]],
    args: argparse.Namespace,
    env_kwargs: dict[str, Any],
    order_ledger_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in eval_plan:
        heuristic.reset()
        rows.append(
            run_action_episode(
                policy_label=label,
                action_fn=lambda obs, info, _env, heuristic=heuristic: np.asarray(
                    heuristic(obs, info), dtype=np.float32
                ),
                eval_item=item,
                args=args,
                env_kwargs=env_kwargs,
                order_ledger_rows=order_ledger_rows,
            )
        )
    return rows


def best_policy(policy_rows: list[dict[str, Any]]) -> str:
    return str(max(policy_rows, key=lambda row: float(row["order_ret_excel_mean"]))["policy"])


def policy_mean(policy_rows: list[dict[str, Any]], label: str, metric: str) -> float:
    row = next(item for item in policy_rows if str(item["policy"]) == label)
    return float(row[f"{metric}_mean"])


def fit_regime_table(
    *,
    base_policy: StaticPolicy,
    candidates: list[StaticPolicy],
    eval_plan: list[dict[str, int]],
    args: argparse.Namespace,
    env_kwargs: dict[str, Any],
) -> tuple[RegimeTablePolicy, list[dict[str, Any]]]:
    table = {regime: base_policy for regime in ADAPTIVE_BENCHMARK_REGIMES}
    trace: list[dict[str, Any]] = []
    for regime in ADAPTIVE_BENCHMARK_REGIMES:
        rows_for_regime: list[dict[str, Any]] = []
        for candidate in candidates:
            trial = dict(table)
            trial[regime] = candidate
            label = f"fit_{regime}_{candidate.label}"
            episode_rows = evaluate_regime_table(
                RegimeTablePolicy(label=label, by_regime=trial),
                eval_plan=eval_plan,
                args=args,
                env_kwargs=env_kwargs,
                order_ledger_rows=None,
            )
            seed_rows = aggregate_seed_metrics(episode_rows)
            policy_rows = aggregate_policy_metrics(seed_rows)
            rows_for_regime.append(policy_rows[0])
        winner = max(rows_for_regime, key=lambda row: float(row["order_ret_excel_mean"]))
        winner_label = str(winner["policy"]).removeprefix(f"fit_{regime}_")
        table[regime] = next(candidate for candidate in candidates if candidate.label == winner_label)
        trace.append({"regime": regime, "selected_static": winner_label, **winner})
    label = "regime_table_" + "__".join(f"{r}:{p.label}" for r, p in table.items())
    return RegimeTablePolicy(label=label, by_regime=table), trace


def build_gap_decomposition(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    common = best_policy(
        [row for row in policy_rows if str(row["policy"]).startswith("S")]
    )
    regime = best_policy(
        [row for row in policy_rows if str(row["policy"]).startswith("regime_table_")]
    )
    heuristic_rows = [
        row for row in policy_rows if str(row["policy"]).startswith("heur_")
    ]
    heuristic = best_policy(heuristic_rows) if heuristic_rows else ""
    rows = []
    baseline = policy_mean(policy_rows, common, "order_ret_excel")
    for stage, label in [
        ("common_static", common),
        ("regime_table", regime),
        ("best_heuristic", heuristic),
    ]:
        if not label:
            continue
        value = policy_mean(policy_rows, label, "order_ret_excel")
        rows.append(
            {
                "stage": stage,
                "policy": label,
                "order_ret_excel_mean": value,
                "gap_vs_common_static": value - baseline,
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--shifts", default="1,2,3")
    parser.add_argument("--op10-mults", default="0.5,0.75,1.0,1.25,1.5,2.0,2.5")
    parser.add_argument("--op12-mults", default="0.5,0.75,1.0,1.25,1.5,2.0,2.5")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--reward-mode", default="control_v1")
    parser.add_argument("--risk-level", default="adaptive_benchmark_v2")
    parser.add_argument("--observation-version", default="v7")
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument("--max-steps", type=int, default=104)
    parser.add_argument("--max-fit-policies", type=int, default=0)
    parser.add_argument("--skip-regime-fit", action="store_true")
    parser.add_argument("--export-order-ledger", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = infer_config(args.run_dir)
    eval_plan = infer_eval_plan(args.run_dir)
    if args.seeds:
        seeds = parse_int_list(args.seeds)
        eval_plan = fallback_eval_plan(seeds, int(args.eval_episodes or 12))
    if args.eval_episodes is not None and not args.seeds:
        seeds = sorted({int(item["seed"]) for item in eval_plan}) or [1, 2, 3, 4, 5]
        eval_plan = fallback_eval_plan(seeds, int(args.eval_episodes))
    if not eval_plan:
        eval_plan = fallback_eval_plan([1, 2, 3, 4, 5], int(args.eval_episodes or 12))

    env_kwargs = build_env_kwargs(args, config)
    candidates = candidate_policies(args)
    order_ledger_rows: list[dict[str, Any]] | None = [] if args.export_order_ledger else None

    episode_rows: list[dict[str, Any]] = []
    for policy in candidates:
        episode_rows.extend(
            evaluate_fixed_static(
                policy,
                eval_plan=eval_plan,
                args=args,
                env_kwargs=env_kwargs,
                order_ledger_rows=order_ledger_rows,
            )
        )

    seed_rows = aggregate_seed_metrics(episode_rows)
    policy_rows = aggregate_policy_metrics(seed_rows)
    common_label = best_policy(policy_rows)
    common_policy = next(policy for policy in candidates if policy.label == common_label)
    fit_trace: list[dict[str, Any]] = []
    if args.skip_regime_fit:
        regime_table = RegimeTablePolicy(
            label=f"regime_table_all_{common_policy.label}",
            by_regime={regime: common_policy for regime in ADAPTIVE_BENCHMARK_REGIMES},
        )
    else:
        regime_table, fit_trace = fit_regime_table(
            base_policy=common_policy,
            candidates=candidates,
            eval_plan=eval_plan,
            args=args,
            env_kwargs=env_kwargs,
        )
    episode_rows.extend(
        evaluate_regime_table(
            regime_table,
            eval_plan=eval_plan,
            args=args,
            env_kwargs=env_kwargs,
            order_ledger_rows=order_ledger_rows,
        )
    )
    for label, heuristic in make_heuristic_defaults().items():
        episode_rows.extend(
            evaluate_heuristic(
                label,
                heuristic,
                eval_plan=eval_plan,
                args=args,
                env_kwargs=env_kwargs,
                order_ledger_rows=order_ledger_rows,
            )
        )

    seed_rows = aggregate_seed_metrics(episode_rows)
    policy_rows = aggregate_policy_metrics(seed_rows)
    gap_rows = build_gap_decomposition(policy_rows)

    write_csv(output_dir / "episode_metrics.csv", episode_rows)
    write_csv(output_dir / "seed_metrics.csv", seed_rows)
    write_csv(output_dir / "policy_summary.csv", policy_rows)
    write_csv(output_dir / "gap_decomposition.csv", gap_rows)
    write_csv(output_dir / "regime_fit_trace.csv", fit_trace)
    if order_ledger_rows is not None:
        write_csv(output_dir / "order_ledger.csv", order_ledger_rows)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "run_dir": str(args.run_dir),
            "output_dir": str(output_dir),
            "env_kwargs": env_kwargs,
            "eval_plan_count": len(eval_plan),
            "candidate_count": len(candidates),
            "skip_regime_fit": bool(args.skip_regime_fit),
        },
        "best_common_static": common_label,
        "regime_table": regime_table.by_regime | {},
        "gap_decomposition": gap_rows,
        "artifacts": {
            "episode_metrics_csv": str((output_dir / "episode_metrics.csv").resolve()),
            "policy_summary_csv": str((output_dir / "policy_summary.csv").resolve()),
            "gap_decomposition_csv": str((output_dir / "gap_decomposition.csv").resolve()),
        },
    }
    summary["regime_table"] = {
        regime: policy.label for regime, policy in regime_table.by_regime.items()
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# E1 Regime Static Heuristic CRN\n\n"
        f"- Best common static: `{common_label}`\n"
        f"- Regime table: `{summary['regime_table']}`\n"
        f"- Eval episodes: `{len(eval_plan)}`\n",
        encoding="utf-8",
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
