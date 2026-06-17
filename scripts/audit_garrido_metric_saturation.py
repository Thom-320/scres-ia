#!/usr/bin/env python3
"""Audit Garrido-style ReT saturation against repository episode metrics.

This script is intentionally static-only. It does not tune rewards or train RL.
It asks whether the apparent ceiling in Track A comes from the physical system,
the Garrido Eq. 5.5 cases, or the way episode summaries aggregate order-level
metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_garrido_static_fidelity_stress import (  # noqa: E402
    RISK_PROFILES,
    POLICY_SETS,
    max_steps_for_spec,
    policy_candidates,
    risk_kwargs_for_profile,
    thesis_design_action,
)
from supply_chain.config import (  # noqa: E402
    HOURS_PER_WEEK,
    RAW_MATERIAL_FLOW_MODE_OPTIONS,
    RISK_OCCURRENCE_MODE_OPTIONS,
)
from supply_chain.external_env_interface import (  # noqa: E402
    get_episode_terminal_metrics,
    make_dkana_thesis_faithful_env,
)
from supply_chain.env_experimental_shifts import (  # noqa: E402
    RET_TAIL_BETA,
    RET_TAIL_BOOST,
    RET_TAIL_CAP_KAPPA,
    RET_TAIL_GAMMA,
    RET_TAIL_INV_KAPPA,
    RET_TAIL_TRANSFORM,
    RET_TAIL_W_CE,
    RET_TAIL_W_RC,
    RET_TAIL_W_SC,
)
from supply_chain.ret_thesis import compute_ret_per_order  # noqa: E402
from supply_chain.thesis_design import (  # noqa: E402
    ThesisDesignSpec,
    design_spec_for_cfi,
    parse_cf_range,
)

DEFAULT_OUTPUT_ROOT = Path("outputs/benchmarks/garrido_metric_saturation_audit")

EPISODE_FIELDS = [
    "profile",
    "family",
    "cfi",
    "source_cfi",
    "policy",
    "policy_kind",
    "replication",
    "seed",
    "horizon_mode",
    "max_steps_used",
    "reward_total",
    "fill_rate_order_level",
    "fill_rate_state_terminal",
    "flow_fill_rate",
    "order_level_ret_mean_existing",
    "ret_mean_completed_orders",
    "ret_mean_all_orders_zero_unfulfilled",
    "ret_gap_existing_minus_all",
    "re_fr_episode_value",
    "re_fr_case_mean",
    "re_ap_case_mean",
    "re_rp_case_mean",
    "re_dp_rp_case_mean",
    "re_fr_contribution_all",
    "re_ap_contribution_all",
    "re_rp_contribution_all",
    "re_dp_rp_contribution_all",
    "dynamic_ret_contribution_all",
    "static_ret_contribution_all",
    "period_weighted_ret_proxy",
    "cycle_time_weighted_ret_completed",
    "period_total_exposure_hours",
    "period_static_exposure_pct",
    "period_ap_exposure_pct",
    "period_rp_exposure_pct",
    "period_dp_rp_exposure_pct",
    "period_dynamic_exposure_pct",
    "period_unfulfilled_exposure_pct",
    "dynamic_case_pct",
    "dynamic_case_ret_mean",
    "ret_p10_all",
    "ret_p50_all",
    "ret_p90_all",
    "ret_p99_all",
    "pct_ret_eq_1",
    "pct_ret_ge_095",
    "pct_ret_lt_05",
    "pct_ret_eq_0",
    "pct_case_fill_rate",
    "pct_case_autotomy",
    "pct_case_recovery",
    "pct_case_non_recovery",
    "pct_case_unfulfilled",
    "n_orders",
    "n_completed",
    "n_unfulfilled",
    "bt_pending_orders",
    "ut_lost_orders",
    "pending_backorder_qty",
    "stockout_week_pct",
    "mean_step_flow_fill",
    "p10_step_flow_fill",
    "mean_ct_hours",
    "p90_ct_hours",
    "mean_delay_hours",
    "p90_delay_hours",
    "max_delay_hours",
    "pct_delayed_completed_orders",
    "cumulative_disruption_hours",
    "action",
]

SUMMARY_NUMERIC_FIELDS = [
    field
    for field in EPISODE_FIELDS
    if field
    not in {
        "profile",
        "family",
        "policy",
        "policy_kind",
        "horizon_mode",
        "action",
    }
]


def utc_label(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def pct(values: Iterable[bool]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return 100.0 * sum(1 for value in vals if value) / len(vals)


def mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if not math.isnan(float(value))]
    return float(np.mean(vals)) if vals else float("nan")


def quantile(values: Iterable[float], q: float) -> float:
    vals = [float(value) for value in values if not math.isnan(float(value))]
    return float(np.quantile(vals, q)) if vals else float("nan")


def order_metric_distribution(sim: Any) -> dict[str, float]:
    orders = list(getattr(sim, "orders", []))
    fill_rate = float(sim._order_level_fill_rate())
    now = float(getattr(getattr(sim, "env", None), "now", 0.0) or 0.0)
    ret_all: list[float] = []
    ret_completed: list[float] = []
    ret_by_case: dict[str, list[float]] = defaultdict(list)
    case_counts: Counter[str] = Counter()
    cycle_times: list[float] = []
    delays: list[float] = []
    period_numerator = 0.0
    period_denominator = 0.0
    cycle_weighted_numerator = 0.0
    cycle_weighted_denominator = 0.0
    period_exposures: Counter[str] = Counter()

    for order in orders:
        ret, case = compute_ret_per_order(order, fill_rate=fill_rate)
        ret_all.append(float(ret))
        case_counts[str(case)] += 1
        ret_by_case[str(case)].append(float(ret))
        lt = float(getattr(order, "LTj", 0.0) or 0.0)
        if getattr(order, "OATj", None) is not None:
            ret_completed.append(float(ret))
            ct = getattr(order, "CTj", None)
            if ct is not None:
                ct_value = float(ct)
                cycle_times.append(ct_value)
                delays.append(max(0.0, ct_value - float(getattr(order, "LTj", 0.0))))
                cycle_exposure = max(ct_value, lt, 1e-9)
                cycle_weighted_numerator += float(ret) * cycle_exposure
                cycle_weighted_denominator += cycle_exposure
                ap = min(
                    max(0.0, float(getattr(order, "APj", 0.0) or 0.0)), cycle_exposure
                )
                remaining = max(0.0, cycle_exposure - ap)
                rp = min(max(0.0, float(getattr(order, "RPj", 0.0) or 0.0)), remaining)
                remaining = max(0.0, remaining - rp)
                dp_rp = min(
                    max(
                        0.0,
                        float(getattr(order, "DPj", 0.0) or 0.0)
                        - float(getattr(order, "RPj", 0.0) or 0.0),
                    ),
                    remaining,
                )
                static_exposure = max(0.0, cycle_exposure - ap - rp - dp_rp)
                re_ap = ap / max(lt, 1e-9) if ap > 0.0 else 0.0
                re_rp = 0.5 * (1.0 / max(rp, 1e-9)) if rp > 0.0 else 0.0
                period_numerator += (
                    fill_rate * static_exposure + re_ap * ap + re_rp * rp
                )
                period_denominator += cycle_exposure
                period_exposures["static"] += static_exposure
                period_exposures["ap"] += ap
                period_exposures["rp"] += rp
                period_exposures["dp_rp"] += dp_rp
        else:
            unfulfilled_exposure = max(
                lt,
                now - float(getattr(order, "OPTj", now) or now),
                1e-9,
            )
            period_denominator += unfulfilled_exposure
            period_exposures["unfulfilled"] += unfulfilled_exposure

    n_orders = len(orders)
    n_completed = sum(1 for order in orders if getattr(order, "OATj", None) is not None)
    n_unfulfilled = n_orders - n_completed
    bt_pending_orders = len(getattr(sim, "pending_backorders", []))
    ut_lost_orders = int(getattr(sim, "total_unattended_orders", 0))
    pending_backorder_qty = sum(
        float(getattr(order, "remaining_qty", 0.0))
        for order in getattr(sim, "pending_backorders", [])
    )
    ret_mean_completed = mean(ret_completed)
    ret_mean_all = mean(ret_all)
    dynamic_values = (
        ret_by_case["autotomy"] + ret_by_case["recovery"] + ret_by_case["non_recovery"]
    )

    def case_mean(case: str) -> float:
        values = ret_by_case[case]
        return mean(values) if values else 0.0

    def case_contribution(case: str) -> float:
        return float(sum(ret_by_case[case]) / max(1, n_orders))

    return {
        "ret_mean_completed_orders": ret_mean_completed,
        "ret_mean_all_orders_zero_unfulfilled": ret_mean_all,
        "re_fr_episode_value": fill_rate,
        "re_fr_case_mean": case_mean("fill_rate"),
        "re_ap_case_mean": case_mean("autotomy"),
        "re_rp_case_mean": case_mean("recovery"),
        "re_dp_rp_case_mean": case_mean("non_recovery"),
        "re_fr_contribution_all": case_contribution("fill_rate"),
        "re_ap_contribution_all": case_contribution("autotomy"),
        "re_rp_contribution_all": case_contribution("recovery"),
        "re_dp_rp_contribution_all": case_contribution("non_recovery"),
        "dynamic_ret_contribution_all": (
            case_contribution("autotomy")
            + case_contribution("recovery")
            + case_contribution("non_recovery")
        ),
        "static_ret_contribution_all": case_contribution("fill_rate"),
        "period_weighted_ret_proxy": (
            period_numerator / period_denominator
            if period_denominator > 0.0
            else float("nan")
        ),
        "cycle_time_weighted_ret_completed": (
            cycle_weighted_numerator / cycle_weighted_denominator
            if cycle_weighted_denominator > 0.0
            else float("nan")
        ),
        "period_total_exposure_hours": float(period_denominator),
        "period_static_exposure_pct": 100.0
        * period_exposures["static"]
        / max(period_denominator, 1e-9),
        "period_ap_exposure_pct": 100.0
        * period_exposures["ap"]
        / max(period_denominator, 1e-9),
        "period_rp_exposure_pct": 100.0
        * period_exposures["rp"]
        / max(period_denominator, 1e-9),
        "period_dp_rp_exposure_pct": 100.0
        * period_exposures["dp_rp"]
        / max(period_denominator, 1e-9),
        "period_dynamic_exposure_pct": 100.0
        * (period_exposures["ap"] + period_exposures["rp"] + period_exposures["dp_rp"])
        / max(period_denominator, 1e-9),
        "period_unfulfilled_exposure_pct": 100.0
        * period_exposures["unfulfilled"]
        / max(period_denominator, 1e-9),
        "dynamic_case_pct": 100.0
        * (
            case_counts["autotomy"]
            + case_counts["recovery"]
            + case_counts["non_recovery"]
        )
        / max(1, n_orders),
        "dynamic_case_ret_mean": mean(dynamic_values) if dynamic_values else 0.0,
        "ret_p10_all": quantile(ret_all, 0.10),
        "ret_p50_all": quantile(ret_all, 0.50),
        "ret_p90_all": quantile(ret_all, 0.90),
        "ret_p99_all": quantile(ret_all, 0.99),
        "pct_ret_eq_1": pct(abs(value - 1.0) <= 1e-9 for value in ret_all),
        "pct_ret_ge_095": pct(value >= 0.95 for value in ret_all),
        "pct_ret_lt_05": pct(value < 0.5 for value in ret_all),
        "pct_ret_eq_0": pct(value <= 1e-12 for value in ret_all),
        "pct_case_fill_rate": 100.0 * case_counts["fill_rate"] / max(1, n_orders),
        "pct_case_autotomy": 100.0 * case_counts["autotomy"] / max(1, n_orders),
        "pct_case_recovery": 100.0 * case_counts["recovery"] / max(1, n_orders),
        "pct_case_non_recovery": 100.0 * case_counts["non_recovery"] / max(1, n_orders),
        "pct_case_unfulfilled": 100.0 * case_counts["unfulfilled"] / max(1, n_orders),
        "n_orders": float(n_orders),
        "n_completed": float(n_completed),
        "n_unfulfilled": float(n_unfulfilled),
        "bt_pending_orders": float(bt_pending_orders),
        "ut_lost_orders": float(ut_lost_orders),
        "pending_backorder_qty": float(pending_backorder_qty),
        "mean_ct_hours": mean(cycle_times),
        "p90_ct_hours": quantile(cycle_times, 0.90),
        "mean_delay_hours": mean(delays),
        "p90_delay_hours": quantile(delays, 0.90),
        "max_delay_hours": max(delays) if delays else 0.0,
        "pct_delayed_completed_orders": pct(value > 0.0 for value in delays),
    }


def resolve_action(policy: dict[str, Any], spec: ThesisDesignSpec) -> np.ndarray:
    action = policy["action"]
    if isinstance(action, str) and action == "matched":
        action = thesis_design_action(spec)
    return np.asarray(action, dtype=np.int64)


def build_env_kwargs(
    *,
    args: argparse.Namespace,
    profile: str,
    spec: ThesisDesignSpec,
    action: np.ndarray,
    max_steps: int,
) -> dict[str, Any]:
    env_kwargs = {
        "reward_mode": args.reward_mode,
        "observation_version": args.observation_version,
        "observation_mode": args.observation_mode,
        "step_size_hours": args.step_size_hours,
        "max_steps": max_steps,
        "stochastic_pt": bool(args.stochastic_pt),
        "stochastic_pt_spread": args.stochastic_pt_spread,
        "stochastic_pt_mean_preserving": bool(args.stochastic_pt_mean_preserving),
        "learn_initial_decision": False,
        "action_space_mode": "thesis_factorized",
        "inventory_period_mode": "thesis_strict",
        "initial_action": action,
        "raw_material_flow_mode": args.raw_material_flow_mode,
        "raw_material_order_up_to_multiplier": args.raw_material_order_up_to_multiplier,
        "risk_occurrence_mode": args.risk_occurrence_mode,
        "ret_tail_w_sc": args.ret_tail_w_sc,
        "ret_tail_w_rc": args.ret_tail_w_rc,
        "ret_tail_w_ce": args.ret_tail_w_ce,
        "ret_tail_cap_kappa": args.ret_tail_cap_kappa,
        "ret_tail_inv_kappa": args.ret_tail_inv_kappa,
        "ret_tail_boost": args.ret_tail_boost,
        "ret_tail_transform": args.ret_tail_transform,
        "ret_tail_gamma": args.ret_tail_gamma,
        "ret_tail_beta": args.ret_tail_beta,
    }
    env_kwargs.update(
        risk_kwargs_for_profile(
            spec=spec,
            profile=profile,
            thesis_pattern_risk_level=args.thesis_pattern_risk_level,
        )
    )
    return env_kwargs


def rollout(
    *,
    args: argparse.Namespace,
    profile: str,
    spec: ThesisDesignSpec,
    policy: dict[str, Any],
    replication: int,
    seed: int,
) -> dict[str, Any]:
    action = resolve_action(policy, spec)
    max_steps = max_steps_for_spec(
        spec,
        horizon_mode=args.horizon_mode,
        fixed_max_steps=args.max_steps,
        step_size_hours=args.step_size_hours,
    )
    env_kwargs = build_env_kwargs(
        args=args,
        profile=profile,
        spec=spec,
        action=action,
        max_steps=max_steps,
    )

    env = make_dkana_thesis_faithful_env(**env_kwargs)
    env.reset(seed=seed)
    terminated = truncated = False
    reward_total = 0.0
    weekly_flow_fills: list[float] = []
    weekly_stockout_flags: list[bool] = []

    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(action)
        reward_total += float(reward)
        if info.get("action_phase") != "weekly_decision":
            continue
        demanded = float(info.get("new_demanded", 0.0))
        backorder_qty = float(info.get("new_backorder_qty", 0.0))
        if demanded > 0:
            weekly_flow_fills.append(max(0.0, min(1.0, 1.0 - backorder_qty / demanded)))
        base_env = getattr(env, "unwrapped", env)
        sim_now = getattr(base_env, "sim", None)
        pending_qty = 0.0
        if sim_now is not None:
            pending_qty = sum(
                float(getattr(order, "remaining_qty", 0.0))
                for order in getattr(sim_now, "pending_backorders", [])
            )
        weekly_stockout_flags.append(backorder_qty > 0.0 or pending_qty > 0.0)

    terminal = get_episode_terminal_metrics(env)
    sim = getattr(env.unwrapped, "sim", None)
    if sim is None:
        raise RuntimeError("Expected DES simulator on env.unwrapped.sim")
    order_dist = order_metric_distribution(sim)
    flow_fill_rate = mean(weekly_flow_fills)
    cumulative_disruption_hours = float(
        getattr(
            sim,
            "_cumulative_down_hours",
            getattr(sim, "cumulative_disruption_hours", 0.0),
        )
    )
    row = {
        "profile": profile,
        "family": spec.family,
        "cfi": spec.cfi,
        "source_cfi": spec.source_cfi,
        "policy": policy["name"],
        "policy_kind": policy["kind"],
        "replication": replication,
        "seed": seed,
        "horizon_mode": args.horizon_mode,
        "max_steps_used": max_steps,
        "reward_total": reward_total,
        "fill_rate_order_level": float(terminal["fill_rate_order_level"]),
        "fill_rate_state_terminal": float(terminal["fill_rate_state_terminal"]),
        "flow_fill_rate": flow_fill_rate,
        "order_level_ret_mean_existing": float(terminal["order_level_ret_mean"]),
        **order_dist,
        "stockout_week_pct": pct(weekly_stockout_flags),
        "mean_step_flow_fill": flow_fill_rate,
        "p10_step_flow_fill": quantile(weekly_flow_fills, 0.10),
        "cumulative_disruption_hours": cumulative_disruption_hours,
        "action": json.dumps(action.astype(int).tolist()),
    }
    row["ret_gap_existing_minus_all"] = float(
        row["order_level_ret_mean_existing"]
    ) - float(row["ret_mean_all_orders_zero_unfulfilled"])
    env.close()
    return row


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["profile"]),
                str(row["family"]),
                str(row["policy"]),
                str(row["policy_kind"]),
            )
        ].append(row)

    summary: list[dict[str, Any]] = []
    for (profile, family, policy, policy_kind), bucket in sorted(groups.items()):
        out: dict[str, Any] = {
            "profile": profile,
            "family": family,
            "policy": policy,
            "policy_kind": policy_kind,
            "episode_count": len(bucket),
        }
        for field in SUMMARY_NUMERIC_FIELDS:
            out[f"{field}_mean"] = mean(float(row[field]) for row in bucket)
        summary.append(out)
    return summary


def write_report(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
) -> None:
    best_rows: list[dict[str, Any]] = []
    for profile in args.profiles:
        candidates = [
            row
            for row in summary
            if row["profile"] == profile
            and row["policy_kind"] in {"matched_doe", "pure_inventory", "pure_capacity"}
        ]
        if candidates:
            best_rows.append(
                max(candidates, key=lambda row: row["fill_rate_order_level_mean"])
            )

    all_ret_gaps = [abs(float(row["ret_gap_existing_minus_all"])) for row in rows]
    max_ret_gap = max(all_ret_gaps) if all_ret_gaps else 0.0
    top_saturated = sorted(
        summary,
        key=lambda row: (
            row["pct_ret_eq_1_mean"],
            row["fill_rate_order_level_mean"],
        ),
        reverse=True,
    )[:12]

    lines = [
        "# Garrido Metric Saturation Audit",
        "",
        f"Created UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"Panel: `{args.panel_cfis}`; profiles: `{','.join(args.profiles)}`; "
        f"policy_set: `{args.policy_set}`; reps: `{args.replications}`.",
        f"Horizon: `{args.horizon_mode}`; max_steps: `{args.max_steps}`; "
        f"risk_occurrence_mode: `{args.risk_occurrence_mode}`.",
        f"Raw-material flow: `{args.raw_material_flow_mode}`; "
        f"stochastic_pt: `{args.stochastic_pt}`; spread: "
        f"`{args.stochastic_pt_spread}`; mean_preserving: "
        f"`{args.stochastic_pt_mean_preserving}`.",
        "",
        "## What This Audits",
        "",
        "- `fill_rate_order_level`: Garrido Eq. 5.4, `1 - (Bt + Ut) / Dt`.",
        "- Component contributions use all orders as denominator. For example, "
        "`Re(APj) contribution = sum(Re(APj) for AP-case orders) / all orders`.",
        "- `period_weighted_ret_proxy` is a secondary diagnostic, not a "
        "replacement for Garrido Eq. 5.5. It weights branch values by observed "
        "order lifecycle exposure to check whether single-case assignment hides "
        "AP/RP/DP exposure.",
        "- `order_level_ret_mean_existing`: current repo terminal ReT mean.",
        "- `ret_mean_all_orders_zero_unfulfilled`: same per-order Eq. 5.5 pass, "
        "but keeps unfulfilled orders as zero instead of dropping them.",
        "- `% case fill_rate/autotomy/recovery/non_recovery`: which branch of "
        "Garrido Eq. 5.5 is dominating.",
        "- `stockout_week_pct` and delay columns: operational evidence hidden by "
        "episode means.",
        "",
        "## Best Fill Policy Per Profile",
        "",
        "| profile | policy | fill Eq5.4 | existing ReT | all-order ReT | ReT gap | %ReT=1 | %ReT<0.5 | %fill-rate case | stockout weeks % | mean delay h |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_rows:
        lines.append(
            f"| {row['profile']} | `{row['policy']}` | "
            f"{row['fill_rate_order_level_mean']:.4f} | "
            f"{row['order_level_ret_mean_existing_mean']:.4f} | "
            f"{row['ret_mean_all_orders_zero_unfulfilled_mean']:.4f} | "
            f"{row['ret_gap_existing_minus_all_mean']:.4f} | "
            f"{row['pct_ret_eq_1_mean']:.1f} | "
            f"{row['pct_ret_lt_05_mean']:.1f} | "
            f"{row['pct_case_fill_rate_mean']:.1f} | "
            f"{row['stockout_week_pct_mean']:.1f} | "
            f"{row['mean_delay_hours_mean']:.1f} |"
        )

    lines += [
        "",
        "## ReT Component Decomposition For Best Fill Policies",
        "",
        "Contributions sum to the all-order ReT mean, except for rounding. "
        "`Re(FRt)` is the static resilience branch; the other three are "
        "dynamic resilience branches in the presence of risks.",
        "",
        "| profile | policy | Re(FRt) contrib | Re(APj) contrib | Re(RPj) contrib | Re(DPj,RPj) contrib | dynamic total | dynamic cases % | Re(FRt) value |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_rows:
        lines.append(
            f"| {row['profile']} | `{row['policy']}` | "
            f"{row['re_fr_contribution_all_mean']:.4f} | "
            f"{row['re_ap_contribution_all_mean']:.4f} | "
            f"{row['re_rp_contribution_all_mean']:.4f} | "
            f"{row['re_dp_rp_contribution_all_mean']:.4f} | "
            f"{row['dynamic_ret_contribution_all_mean']:.4f} | "
            f"{row['dynamic_case_pct_mean']:.1f} | "
            f"{row['re_fr_episode_value_mean']:.4f} |"
        )

    lines += [
        "",
        "## Single-Case ReT vs Period-Exposure Proxy",
        "",
        "This proxy is intentionally secondary. It uses the AP/RP/DP indicators "
        "already populated by the simulator and exposes whether a policy spends "
        "meaningful lifecycle time under disruption states even when the "
        "single-case Eq. 5.5 branch is dominated by `Re(FRt)`.",
        "",
        "| profile | policy | all-order ReT | period-weighted proxy | dynamic case % | dynamic exposure % | static exposure % | unfulfilled exposure % | p10 ReT |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_rows:
        lines.append(
            f"| {row['profile']} | `{row['policy']}` | "
            f"{row['ret_mean_all_orders_zero_unfulfilled_mean']:.4f} | "
            f"{row['period_weighted_ret_proxy_mean']:.4f} | "
            f"{row['dynamic_case_pct_mean']:.1f} | "
            f"{row['period_dynamic_exposure_pct_mean']:.1f} | "
            f"{row['period_static_exposure_pct_mean']:.1f} | "
            f"{row['period_unfulfilled_exposure_pct_mean']:.1f} | "
            f"{row['ret_p10_all_mean']:.4f} |"
        )

    lines += [
        "",
        "## Most Saturated Summary Rows",
        "",
        "| profile | family | policy | fill | ReT p10 | ReT p50 | ReT p90 | %ReT=1 | %fill-rate case | %autotomy | %recovery |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top_saturated:
        lines.append(
            f"| {row['profile']} | {row['family']} | `{row['policy']}` | "
            f"{row['fill_rate_order_level_mean']:.4f} | "
            f"{row['ret_p10_all_mean']:.4f} | "
            f"{row['ret_p50_all_mean']:.4f} | "
            f"{row['ret_p90_all_mean']:.4f} | "
            f"{row['pct_ret_eq_1_mean']:.1f} | "
            f"{row['pct_case_fill_rate_mean']:.1f} | "
            f"{row['pct_case_autotomy_mean']:.1f} | "
            f"{row['pct_case_recovery_mean']:.1f} |"
        )

    saturated_profiles = [
        row
        for row in best_rows
        if float(row["fill_rate_order_level_mean"]) >= 0.98
        and float(row["pct_ret_ge_095_mean"]) >= 80.0
    ]
    lines += [
        "",
        "## Diagnosis",
        "",
    ]
    if saturated_profiles:
        lines.append(
            "- Several best-policy profile rows are genuinely near-ceiling: "
            "`fill_rate_order_level >= 0.98` and at least 80% of order ReT "
            "values are >= 0.95."
        )
    else:
        lines.append(
            "- The best-policy rows are not uniformly at ceiling once order-level "
            "distribution is inspected."
        )
    if max_ret_gap > 0.01:
        lines.append(
            f"- Existing terminal ReT can exceed all-order ReT by up to "
            f"{max_ret_gap:.4f}, because the current aggregate drops "
            "unfulfilled orders from the ReT mean. Eq. 5.4 fill still counts "
            "them through Bt/Ut."
        )
    else:
        lines.append(
            "- Dropping unfulfilled orders from the current ReT mean is not a "
            "large driver in this run; the largest existing-vs-all ReT gap is "
            f"{max_ret_gap:.4f}."
        )
    lines.append(
        "- If `%fill-rate case` dominates, most orders never enter dynamic "
        "autotomy/recovery/non-recovery branches; the episode is measuring "
        "static service rather than disruption response."
    )
    lines.append(
        "- Use this report before reward tuning: if p10/p50/p90 are already "
        "near 1.0, a steeper reward mostly magnifies a saturated signal."
    )
    lines.append(
        "- Garrido-Rios (2017) publishes the ReT equation, SDM schema, example "
        "ReT(Cfi) series, and rank-test comparisons. It does not publish a "
        "machine-readable table of mean ReT by every Cf row, so this report "
        "compares our environment to the thesis metric contract and hypothesis "
        "patterns rather than to a thesis CSV of means."
    )

    lines += [
        "",
        "## Files",
        "",
        "- `episode_metric_audit.csv`: one row per evaluated episode.",
        "- `metric_saturation_summary.csv`: grouped profile/family/policy means.",
        "- `manifest.json`: exact run configuration.",
    ]
    (out_dir / "GARRIDO_METRIC_SATURATION_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--panel-cfis", default="31-90")
    parser.add_argument(
        "--profiles",
        type=parse_csv_list,
        default=["current", "increased", "severe", "severe_extended"],
    )
    parser.add_argument("--policy-set", choices=POLICY_SETS, default="minimal")
    parser.add_argument("--replications", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=935000)
    parser.add_argument("--reward-mode", default="ReT_thesis")
    parser.add_argument("--ret-tail-w-sc", type=float, default=RET_TAIL_W_SC)
    parser.add_argument("--ret-tail-w-rc", type=float, default=RET_TAIL_W_RC)
    parser.add_argument("--ret-tail-w-ce", type=float, default=RET_TAIL_W_CE)
    parser.add_argument("--ret-tail-cap-kappa", type=float, default=RET_TAIL_CAP_KAPPA)
    parser.add_argument("--ret-tail-inv-kappa", type=float, default=RET_TAIL_INV_KAPPA)
    parser.add_argument("--ret-tail-boost", type=float, default=RET_TAIL_BOOST)
    parser.add_argument(
        "--ret-tail-transform",
        choices=["identity", "power", "exp_norm"],
        default=RET_TAIL_TRANSFORM,
    )
    parser.add_argument("--ret-tail-gamma", type=float, default=RET_TAIL_GAMMA)
    parser.add_argument("--ret-tail-beta", type=float, default=RET_TAIL_BETA)
    parser.add_argument(
        "--raw-material-flow-mode",
        default="kit_equivalent_order_up_to",
        choices=RAW_MATERIAL_FLOW_MODE_OPTIONS,
    )
    parser.add_argument(
        "--raw-material-order-up-to-multiplier", type=float, default=2.0
    )
    parser.add_argument(
        "--risk-occurrence-mode",
        choices=RISK_OCCURRENCE_MODE_OPTIONS,
        default="thesis_periodic",
    )
    parser.add_argument("--observation-version", default="v5")
    parser.add_argument("--observation-mode", default="env_sdm_history_reward")
    parser.add_argument("--step-size-hours", type=float, default=float(HOURS_PER_WEEK))
    parser.add_argument("--horizon-mode", choices=("fixed", "thesis"), default="fixed")
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--thesis-pattern-risk-level", default="increased")
    parser.add_argument("--stochastic-pt", action="store_true", default=True)
    parser.add_argument(
        "--no-stochastic-pt", dest="stochastic_pt", action="store_false"
    )
    parser.add_argument("--stochastic-pt-spread", type=float, default=1.0)
    parser.add_argument("--stochastic-pt-mean-preserving", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    unknown = set(args.profiles).difference(RISK_PROFILES)
    if unknown:
        raise ValueError(f"Unknown risk profiles: {sorted(unknown)}")

    label = args.label or utc_label("garrido_metric_saturation")
    out_dir = args.output_root / label
    out_dir.mkdir(parents=True, exist_ok=False)

    specs = [design_spec_for_cfi(cfi) for cfi in parse_cf_range(args.panel_cfis)]
    policies = policy_candidates(args.policy_set)
    total = len(args.profiles) * len(specs) * len(policies) * args.replications
    rows: list[dict[str, Any]] = []

    csv_path = out_dir / "episode_metric_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        done = 0
        for profile_index, profile in enumerate(args.profiles):
            for spec in specs:
                for policy_index, policy in enumerate(policies):
                    for replication in range(args.replications):
                        seed = (
                            args.base_seed
                            + profile_index * 10_000_000
                            + spec.cfi * 10_000
                            + policy_index * 100
                            + replication
                        )
                        row = rollout(
                            args=args,
                            profile=profile,
                            spec=spec,
                            policy=policy,
                            replication=replication,
                            seed=seed,
                        )
                        rows.append(row)
                        writer.writerow(row)
                        done += 1
                        if done % max(1, args.progress_every) == 0:
                            handle.flush()
                            print(
                                f"progress {done}/{total} "
                                f"({100.0 * done / total:.1f}%)",
                                flush=True,
                            )

    summary = summarize(rows)
    summary_fields = list(summary[0].keys()) if summary else []
    write_csv(out_dir / "metric_saturation_summary.csv", summary, summary_fields)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "panel_cfis": args.panel_cfis,
        "profiles": args.profiles,
        "policy_set": args.policy_set,
        "replications": args.replications,
        "base_seed": args.base_seed,
        "reward_mode": args.reward_mode,
        "ret_tail_w_sc": args.ret_tail_w_sc,
        "ret_tail_w_rc": args.ret_tail_w_rc,
        "ret_tail_w_ce": args.ret_tail_w_ce,
        "ret_tail_cap_kappa": args.ret_tail_cap_kappa,
        "ret_tail_inv_kappa": args.ret_tail_inv_kappa,
        "ret_tail_boost": args.ret_tail_boost,
        "ret_tail_transform": args.ret_tail_transform,
        "ret_tail_gamma": args.ret_tail_gamma,
        "ret_tail_beta": args.ret_tail_beta,
        "raw_material_flow_mode": args.raw_material_flow_mode,
        "raw_material_order_up_to_multiplier": args.raw_material_order_up_to_multiplier,
        "risk_occurrence_mode": args.risk_occurrence_mode,
        "horizon_mode": args.horizon_mode,
        "max_steps": args.max_steps,
        "step_size_hours": args.step_size_hours,
        "stochastic_pt": args.stochastic_pt,
        "stochastic_pt_spread": args.stochastic_pt_spread,
        "stochastic_pt_mean_preserving": args.stochastic_pt_mean_preserving,
        "policy_count": len(policies),
        "episode_count": len(rows),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(out_dir, args=args, rows=rows, summary=summary)
    print(out_dir / "GARRIDO_METRIC_SATURATION_AUDIT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
