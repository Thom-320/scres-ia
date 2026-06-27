#!/usr/bin/env python3
"""Compare Track-A PPO against the frozen efficient static frontier.

This runner uses the same thesis-factorized Track A action surface for both
arms: ``Discrete(18)`` = 6 inventory levels x 3 shift levels.  The paper-facing
question is whether a dynamic policy can beat the best cost-aware static policy
identified by the Garrido-2024 Cobb-Douglas gate:

* current/increased: S1_I168
* severe: S2_I168

The primary same-bar outcome for this runner is the Garrido-2024 Cobb-Douglas
sigmoid index. Garrido's 2017 Excel ReT is still reported as the continuity
metric, but it is not the primary comparison lens in the C-D lane.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

try:  # Optional dependency; installed in the project venv, absent in lean envs.
    from sb3_contrib import RecurrentPPO
except Exception:  # pragma: no cover - dependency availability is environment-specific.
    RecurrentPPO = None  # type: ignore[assignment]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.external_env_interface import (  # noqa: E402
    get_episode_terminal_metrics,
    make_discrete18_track_a_env,
)
from supply_chain.thesis_decision_env import Discrete18TrackAEnv  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("outputs/benchmarks/garrido_dynamic_vs_static")
INVENTORY_LEVEL_LABELS = {
    0: "I0",
    1: "I168",
    2: "I336",
    3: "I504",
    4: "I672",
    5: "I1344",
}


def _static_baselines() -> dict[str, tuple[int, int]]:
    baselines: dict[str, tuple[int, int]] = {"original_S1_I0": (0, 0)}
    for level, inventory_label in INVENTORY_LEVEL_LABELS.items():
        for shift_index in range(3):
            if level == 0 and shift_index == 0:
                continue
            baselines[f"static_S{shift_index + 1}_{inventory_label}"] = (
                level,
                shift_index,
            )
    return baselines


STATIC_BASELINES = _static_baselines()
HEURISTIC_POLICIES = {
    "heuristic_threshold_lean": {
        "default": (0, 0),
        "stress": (1, 1),
        "description": "Default S1/I0; switch to S2/I168 after backlog/service-loss stress.",
    },
    "heuristic_threshold_buffer": {
        "default": (1, 0),
        "stress": (1, 1),
        "description": "Default S1/I168; switch to S2/I168 after backlog/service-loss stress.",
    },
}
FROZEN_STATIC_BY_REGIME = {
    "current": ("static_S1_I168", 1, 0),
    "increased": ("static_S1_I168", 1, 0),
    "severe": ("static_S2_I168", 1, 1),
}


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.fmean(finite)) if finite else float("nan")


def _std(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.stdev(finite)) if len(finite) > 1 else 0.0


def _ci95(values: list[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan"), float("nan")
    mean = _mean(finite)
    if len(finite) < 2:
        return mean, mean
    half = 1.96 * _std(finite) / (len(finite) ** 0.5)
    return mean - half, mean + half


def _p95(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95))


def _cvar95(values: list[float]) -> float:
    if not values:
        return float("nan")
    threshold = _p95(values)
    tail = [value for value in values if value >= threshold]
    return _mean(tail)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _order_audit_metrics(env: Any) -> dict[str, Any]:
    base_env = getattr(env, "unwrapped", env)
    sim = getattr(base_env, "sim", None)
    if sim is None:
        return {}
    order_summary = (
        sim.compute_order_level_ret() if hasattr(sim, "compute_order_level_ret") else {}
    )
    case_counts = dict(order_summary.get("case_counts_excel_formula", {}))
    orders = list(getattr(sim, "orders", []))
    served = [order for order in orders if getattr(order, "OATj", None) is not None]
    n_orders = int(order_summary.get("n_orders", len(orders)) or 0)
    total_denominator = max(n_orders, 1)
    served_denominator = max(len(served), 1)

    def attr_values(attr: str) -> list[float]:
        return [float(getattr(order, attr, 0.0) or 0.0) for order in served]

    out: dict[str, Any] = {
        "orders_total": n_orders,
        "orders_completed": int(order_summary.get("n_completed", len(served)) or 0),
        "excel_branch_fill_rate_share": (
            int(case_counts.get("excel_fill_rate", 0)) / total_denominator
        ),
        "excel_branch_autotomy_share": (
            int(case_counts.get("excel_autotomy", 0)) / total_denominator
        ),
        "excel_branch_recovery_share": (
            int(case_counts.get("excel_recovery", 0)) / total_denominator
        ),
        "excel_branch_risk_no_recovery_share": (
            int(case_counts.get("excel_risk_no_recovery", 0)) / total_denominator
        ),
        "excel_branch_unfulfilled_share": (
            int(case_counts.get("excel_unfulfilled", 0)) / total_denominator
        ),
    }
    for attr in ("CTj", "APj", "RPj", "DPj"):
        values = attr_values(attr)
        out[f"{attr}_mean"] = _mean(values)
        out[f"{attr}_p50"] = _quantile(values, 50)
        out[f"{attr}_p95"] = _quantile(values, 95)
        out[f"{attr}_p99"] = _quantile(values, 99)
        out[f"{attr}_positive_share"] = (
            sum(1 for value in values if value > 0.0) / served_denominator
        )
    return out


def static_action(level: int, shift_index: int) -> int:
    return Discrete18TrackAEnv.encode_discrete_action(level, shift_index)


def is_heuristic_policy(policy_name: str) -> bool:
    return str(policy_name) in HEURISTIC_POLICIES


def heuristic_initial_action(policy_name: str) -> int:
    spec = HEURISTIC_POLICIES[str(policy_name)]
    level, shift_index = spec["default"]
    return static_action(int(level), int(shift_index))


def heuristic_action(policy_name: str, *, stress_hold_steps: int) -> int:
    spec = HEURISTIC_POLICIES[str(policy_name)]
    key = "stress" if stress_hold_steps > 0 else "default"
    level, shift_index = spec[key]
    return static_action(int(level), int(shift_index))


def heuristic_is_stressed(info: dict[str, Any], args: argparse.Namespace) -> bool:
    service_loss = float(info.get("service_loss_step", 0.0) or 0.0)
    pending_backorder_qty = float(info.get("pending_backorder_qty", 0.0) or 0.0)
    new_backorder_qty = float(info.get("new_backorder_qty", 0.0) or 0.0)
    disruption_fraction = float(info.get("disruption_fraction_step", 0.0) or 0.0)
    return (
        service_loss >= float(args.heuristic_service_loss_threshold)
        or pending_backorder_qty >= float(args.heuristic_pending_backorder_threshold)
        or new_backorder_qty >= float(args.heuristic_new_backorder_threshold)
        or disruption_fraction >= float(args.heuristic_disruption_threshold)
    )


def _decision_resource(info: dict[str, Any]) -> tuple[float, float, int]:
    decision = info.get("thesis_decision", {})
    if not isinstance(decision, dict):
        return 0.0, 0.0, 0
    period = decision.get("inventory_period_hours")
    period_hours = 0.0 if period is None else float(period)
    targets = decision.get("inventory_buffer_targets", {})
    target_total = (
        sum(float(value) for value in targets.values())
        if isinstance(targets, dict)
        else 0.0
    )
    inventory_level = int(decision.get("common_inventory_level", 0) or 0)
    return period_hours, target_total, inventory_level


def build_env_kwargs(args: argparse.Namespace, regime: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "reward_mode": args.reward_mode,
        "observation_version": args.observation_version,
        "risk_level": regime,
        "stochastic_pt": bool(args.stochastic_pt),
        "step_size_hours": float(args.step_size_hours),
        "max_steps": int(args.max_steps),
        "risk_occurrence_mode": args.risk_occurrence_mode,
        "risk_frequency_multiplier": float(args.risk_frequency_multiplier),
        "risk_impact_multiplier": float(args.risk_impact_multiplier),
        "ret_g24_shift_cost": float(args.ret_g24_shift_cost),
        "ret_g24_kappa_train_frac": float(args.ret_g24_kappa_train_frac),
        "w_bo": float(args.w_bo),
        "w_cost": float(args.w_cost),
        "w_disr": float(args.w_disr),
        "control_v2_w_fill": float(args.control_v2_w_fill),
        "control_v2_w_service": float(args.control_v2_w_service),
        "control_v2_w_lost": float(args.control_v2_w_lost),
        "control_v2_w_inventory": float(args.control_v2_w_inventory),
        "control_v2_w_shift": float(args.control_v2_w_shift),
        "control_v2_w_switch": float(args.control_v2_w_switch),
        "raw_material_flow_mode": args.raw_material_flow_mode,
        "raw_material_order_up_to_multiplier": float(
            args.raw_material_order_up_to_multiplier
        ),
        "demand_on_hand_fulfillment_delay": float(
            args.demand_on_hand_fulfillment_delay
        ),
    }
    return kwargs


def _env_kwargs_with_initial_policy(
    args: argparse.Namespace,
    regime: str,
    policy_name: str | None,
) -> dict[str, Any]:
    kwargs = build_env_kwargs(args, regime)
    if policy_name:
        level, shift_index = STATIC_BASELINES[str(policy_name)]
        kwargs["initial_action"] = static_action(level, shift_index)
    return kwargs


def make_env_factory(args: argparse.Namespace, regime: str, seed: int):
    def _factory():
        env = make_discrete18_track_a_env(
            **_env_kwargs_with_initial_policy(
                args,
                regime,
                str(getattr(args, "ppo_initial_static_policy", "")),
            )
        )
        env.reset(seed=seed)
        return Monitor(env)

    return _factory


def train_ppo(args: argparse.Namespace, *, regime: str, seed: int) -> Any:
    vec_env = DummyVecEnv([make_env_factory(args, regime, seed)])
    common_kwargs = {
        "env": vec_env,
        "learning_rate": float(args.learning_rate),
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
        "n_epochs": int(args.n_epochs),
        "gamma": float(args.gamma),
        "gae_lambda": float(args.gae_lambda),
        "clip_range": float(args.clip_range),
        "seed": int(seed),
        "verbose": 0,
        "device": "cpu",
    }
    if args.algo == "recurrent_ppo":
        if RecurrentPPO is None:
            raise RuntimeError("sb3_contrib.RecurrentPPO is not available.")
        model = RecurrentPPO("MlpLstmPolicy", **common_kwargs)
    else:
        model = PPO("MlpPolicy", **common_kwargs)
    model.learn(total_timesteps=int(args.train_timesteps))
    vec_env.close()
    return model


def evaluate_episode(
    args: argparse.Namespace,
    *,
    regime: str,
    seed: int,
    policy_name: str,
    model: Any | None = None,
    fixed_action: int | None = None,
) -> dict[str, Any]:
    initial_policy = (
        None
        if fixed_action is not None or is_heuristic_policy(policy_name)
        else str(getattr(args, "ppo_initial_static_policy", ""))
    )
    env = make_discrete18_track_a_env(
        **_env_kwargs_with_initial_policy(args, regime, initial_policy)
    )
    initial_options: dict[str, Any] = {}
    if fixed_action is not None:
        initial_options["initial_discrete_action"] = int(fixed_action)
    elif is_heuristic_policy(policy_name):
        initial_options["initial_discrete_action"] = heuristic_initial_action(policy_name)
    obs, _info = env.reset(seed=int(seed), options=initial_options)
    done = False
    steps = 0
    reward_total = 0.0
    cd_sigmoid_total = 0.0
    cd_train_total = 0.0
    raw_total = 0.0
    ret_cvar_cd_total = 0.0
    cd_loss_values: list[float] = []
    cvar_estimates: list[float] = []
    demanded_total = 0.0
    delivered_total = 0.0
    backorder_qty_total = 0.0
    shift_hours_total = 0.0
    extra_shift_hours_total = 0.0
    strategic_buffer_period_hours_total = 0.0
    strategic_buffer_target_unit_hours_total = 0.0
    service_losses: list[float] = []
    shift_counts = {1: 0, 2: 0, 3: 0}
    inventory_level_counts: dict[int, int] = {}
    action_counts: dict[int, int] = {}
    final_info: dict[str, Any] = {}
    stress_hold_steps = 0
    recurrent_state: Any = None
    episode_start = np.ones((1,), dtype=bool)

    while not done:
        if fixed_action is not None:
            action = int(fixed_action)
        elif is_heuristic_policy(policy_name):
            action = heuristic_action(policy_name, stress_hold_steps=stress_hold_steps)
        else:
            if model is None:
                raise ValueError("model is required for dynamic policy evaluation")
            if args.algo == "recurrent_ppo":
                predicted, recurrent_state = model.predict(
                    obs,
                    state=recurrent_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
            else:
                predicted, _ = model.predict(obs, deterministic=True)
            action = int(np.asarray(predicted).item())
        obs, reward, terminated, truncated, info = env.step(action)
        episode_start = np.asarray([bool(terminated or truncated)], dtype=bool)
        if is_heuristic_policy(policy_name):
            if heuristic_is_stressed(info, args):
                stress_hold_steps = int(args.heuristic_hold_steps)
            else:
                stress_hold_steps = max(0, stress_hold_steps - 1)
        reward_total += float(reward)
        cd_sigmoid_total += float(info.get("ret_garrido2024_sigmoid_step", 0.0))
        cd_train_total += float(info.get("ret_garrido2024_train_step", 0.0))
        raw_total += float(info.get("ret_garrido2024_raw_step", 0.0))
        ret_cvar_cd_total += float(info.get("ret_cvar_cd_step", 0.0))
        if "cd_loss_step" in info:
            cd_loss_values.append(float(info.get("cd_loss_step", 0.0)))
        if "cvar_estimate" in info:
            cvar_estimates.append(float(info.get("cvar_estimate", 0.0)))
        new_demanded = float(info.get("new_demanded", 0.0))
        new_backorder_qty = float(info.get("new_backorder_qty", 0.0))
        demanded_total += new_demanded
        delivered_total += float(info.get("new_delivered", 0.0))
        backorder_qty_total += new_backorder_qty
        shifts_active = int(info.get("shifts_active", 1))
        shift_counts[shifts_active] += 1
        shift_hours_total += float(shifts_active) * float(args.step_size_hours)
        extra_shift_hours_total += max(0, shifts_active - 1) * float(
            args.step_size_hours
        )
        period_hours, target_total, inventory_level = _decision_resource(info)
        strategic_buffer_period_hours_total += period_hours
        strategic_buffer_target_unit_hours_total += target_total * float(
            args.step_size_hours
        )
        inventory_level_counts[inventory_level] = (
            inventory_level_counts.get(inventory_level, 0) + 1
        )
        service_loss_step = info.get("service_loss_step")
        if service_loss_step is None:
            service_loss_step = (
                new_backorder_qty / new_demanded if new_demanded > 0.0 else 0.0
            )
        service_losses.append(float(service_loss_step))
        action_counts[action] = action_counts.get(action, 0) + 1
        final_info = dict(info)
        steps += 1
        done = bool(terminated or truncated)

    terminal = get_episode_terminal_metrics(env)
    order_audit = _order_audit_metrics(env)
    env.close()
    total_steps = max(1, steps)
    flow_fill_rate = (
        delivered_total / demanded_total if demanded_total > 0.0 else float("nan")
    )
    metrics = {
        "regime": regime,
        "policy": policy_name,
        "seed": int(seed),
        "eval_seed": int(seed),
        "steps": int(steps),
        "reward_total": reward_total,
        "cd_sigmoid_total": cd_sigmoid_total,
        "cd_sigmoid_mean": cd_sigmoid_total / total_steps,
        "cd_train_total": cd_train_total,
        "cd_train_mean": cd_train_total / total_steps,
        "cd_raw_total": raw_total,
        "cd_raw_mean": raw_total / total_steps,
        "ret_cvar_cd_total": ret_cvar_cd_total,
        "ret_cvar_cd_mean": ret_cvar_cd_total / total_steps,
        "cd_loss_mean": _mean(cd_loss_values),
        "cd_loss_cvar95": _cvar95(cd_loss_values),
        "cd_cvar_estimate_terminal": float(final_info.get("cvar_estimate", 0.0)),
        "cd_cvar_estimate_mean": _mean(cvar_estimates),
        "cd_zeta_avg": float(final_info.get("zeta_avg", 0.0)),
        "cd_epsilon_avg": float(final_info.get("epsilon_avg", 0.0)),
        "cd_phi_avg": float(final_info.get("phi_avg", 0.0)),
        "cd_tau_avg": float(final_info.get("tau_avg", 0.0)),
        "cd_kappa_dot": float(final_info.get("kappa_dot", 0.0)),
        "mean_ret_excel_formula": float(
            terminal["order_level_ret_excel_formula_mean"]
        ),
        "mean_ret_text_formula": float(terminal["order_level_ret_text_formula_mean"]),
        "fill_rate_order_level": float(terminal["fill_rate_order_level"]),
        "backorder_rate_order_level": float(terminal["backorder_rate_order_level"]),
        "flow_fill_rate": flow_fill_rate,
        "demanded_total": demanded_total,
        "delivered_total": delivered_total,
        "backorder_qty_total": backorder_qty_total,
        "pending_backorder_qty_terminal": float(
            final_info.get("pending_backorder_qty", 0.0)
        ),
        "unattended_orders_terminal": float(
            final_info.get("unattended_orders_total", 0.0)
        ),
        "service_loss_mean": _mean(service_losses),
        "service_loss_p95": _p95(service_losses),
        "service_loss_cvar95": _cvar95(service_losses),
        "service_loss_positive_step_share": (
            sum(1 for value in service_losses if value > 0.0) / total_steps
        ),
        "shift_hours_total": shift_hours_total,
        "extra_shift_hours_total": extra_shift_hours_total,
        "strategic_buffer_period_hours_mean": (
            strategic_buffer_period_hours_total / total_steps
        ),
        "strategic_buffer_target_unit_hours_total": (
            strategic_buffer_target_unit_hours_total
        ),
        "strategic_buffer_target_units_mean": (
            strategic_buffer_target_unit_hours_total
            / max(float(args.step_size_hours) * total_steps, 1.0)
        ),
        "resource_composite_total": (
            extra_shift_hours_total
            + strategic_buffer_target_unit_hours_total
            / max(float(args.step_size_hours), 1.0)
        ),
        "pct_steps_S1": 100.0 * shift_counts[1] / total_steps,
        "pct_steps_S2": 100.0 * shift_counts[2] / total_steps,
        "pct_steps_S3": 100.0 * shift_counts[3] / total_steps,
        "dominant_inventory_level": max(
            inventory_level_counts.items(), key=lambda item: item[1]
        )[0],
        "inventory_level_counts": json.dumps(inventory_level_counts, sort_keys=True),
        "dominant_action": max(action_counts.items(), key=lambda item: item[1])[0],
        "action_counts": json.dumps(action_counts, sort_keys=True),
    }
    metrics.update(order_audit)
    return metrics


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["regime"]), str(row["policy"])), []).append(row)
    metric_names = [
        "cd_sigmoid_mean",
        "cd_sigmoid_total",
        "cd_train_mean",
        "cd_train_total",
        "cd_raw_mean",
        "cd_raw_total",
        "ret_cvar_cd_mean",
        "ret_cvar_cd_total",
        "cd_loss_mean",
        "cd_loss_cvar95",
        "cd_cvar_estimate_terminal",
        "cd_cvar_estimate_mean",
        "cd_zeta_avg",
        "cd_epsilon_avg",
        "cd_phi_avg",
        "cd_tau_avg",
        "cd_kappa_dot",
        "mean_ret_excel_formula",
        "mean_ret_text_formula",
        "fill_rate_order_level",
        "flow_fill_rate",
        "backorder_qty_total",
        "pending_backorder_qty_terminal",
        "unattended_orders_terminal",
        "service_loss_mean",
        "service_loss_p95",
        "service_loss_cvar95",
        "service_loss_positive_step_share",
        "shift_hours_total",
        "extra_shift_hours_total",
        "strategic_buffer_period_hours_mean",
        "strategic_buffer_target_unit_hours_total",
        "strategic_buffer_target_units_mean",
        "resource_composite_total",
        "orders_total",
        "orders_completed",
        "excel_branch_fill_rate_share",
        "excel_branch_autotomy_share",
        "excel_branch_recovery_share",
        "excel_branch_risk_no_recovery_share",
        "excel_branch_unfulfilled_share",
        "CTj_mean",
        "CTj_p50",
        "CTj_p95",
        "CTj_p99",
        "CTj_positive_share",
        "APj_mean",
        "APj_p50",
        "APj_p95",
        "APj_p99",
        "APj_positive_share",
        "RPj_mean",
        "RPj_p50",
        "RPj_p95",
        "RPj_p99",
        "RPj_positive_share",
        "DPj_mean",
        "DPj_p50",
        "DPj_p95",
        "DPj_p99",
        "DPj_positive_share",
        "pct_steps_S1",
        "pct_steps_S2",
        "pct_steps_S3",
    ]
    out: list[dict[str, Any]] = []
    for (regime, policy), group in sorted(grouped.items()):
        item: dict[str, Any] = {
            "regime": regime,
            "policy": policy,
            "n": len(group),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in group]
            lo, hi = _ci95(values)
            item[f"{metric}_mean"] = _mean(values)
            item[f"{metric}_std"] = _std(values)
            item[f"{metric}_ci95_low"] = lo
            item[f"{metric}_ci95_high"] = hi
        out.append(item)
    return out


def build_comparison(
    summary_rows: list[dict[str, Any]],
    *,
    excel_noninferiority_tol: float = 0.0,
) -> list[dict[str, Any]]:
    by_key = {
        (str(row["regime"]), str(row["policy"])): row for row in summary_rows
    }
    comparisons: list[dict[str, Any]] = []
    regimes = sorted({str(row["regime"]) for row in summary_rows})
    dynamic_policies = sorted(
        {
            str(row["policy"])
            for row in summary_rows
            if str(row["policy"]) == "ppo_dynamic" or is_heuristic_policy(str(row["policy"]))
        }
    )
    for regime in regimes:
        for static_name in STATIC_BASELINES:
            static = by_key.get((regime, static_name))
            if static is None:
                continue
            for dynamic_name in dynamic_policies:
                dynamic = by_key.get((regime, dynamic_name))
                if dynamic is None:
                    continue
                delta_excel = (
                    dynamic["mean_ret_excel_formula_mean"]
                    - static["mean_ret_excel_formula_mean"]
                )
                excel_noninferior = delta_excel >= -abs(float(excel_noninferiority_tol))
                fewer_extra_shift_hours = (
                    dynamic["extra_shift_hours_total_mean"]
                    < static["extra_shift_hours_total_mean"]
                )
                lower_buffer_target = (
                    dynamic["strategic_buffer_target_units_mean_mean"]
                    < static["strategic_buffer_target_units_mean_mean"]
                )
                fill_noninferior = (
                    dynamic["fill_rate_order_level_mean"]
                    >= static["fill_rate_order_level_mean"]
                    - abs(float(excel_noninferiority_tol))
                )
                p95_not_worse = dynamic["service_loss_p95_mean"] <= static[
                    "service_loss_p95_mean"
                ]
                cvar95_not_worse = dynamic["service_loss_cvar95_mean"] <= static[
                    "service_loss_cvar95_mean"
                ]
                resource_pareto = (
                    excel_noninferior
                    and fewer_extra_shift_hours
                    and lower_buffer_target
                )
                comparisons.append(
                    {
                        "regime": regime,
                        "dynamic_policy": dynamic_name,
                        "static_policy": static_name,
                        "is_frozen_efficient_static": (
                            FROZEN_STATIC_BY_REGIME.get(regime, ("", 0, 0))[0]
                            == static_name
                        ),
                        "primary_metric": "cd_sigmoid_mean",
                        "secondary_metric": "mean_ret_excel_formula",
                        "dynamic_cd_sigmoid_mean": dynamic["cd_sigmoid_mean_mean"],
                        "ppo_cd_sigmoid_mean": dynamic["cd_sigmoid_mean_mean"],
                        "static_cd_sigmoid_mean": static["cd_sigmoid_mean_mean"],
                        "delta_cd_sigmoid_mean": (
                            dynamic["cd_sigmoid_mean_mean"]
                            - static["cd_sigmoid_mean_mean"]
                        ),
                        "dynamic_excel_ret": dynamic["mean_ret_excel_formula_mean"],
                        "ppo_excel_ret": dynamic["mean_ret_excel_formula_mean"],
                        "static_excel_ret": static["mean_ret_excel_formula_mean"],
                        "delta_excel_ret": delta_excel,
                        "ppo_beats_static_excel": (
                            dynamic["mean_ret_excel_formula_mean"]
                            > static["mean_ret_excel_formula_mean"]
                        ),
                        "excel_noninferiority_tol": abs(float(excel_noninferiority_tol)),
                        "excel_noninferior": excel_noninferior,
                        "dynamic_fill_rate": dynamic["fill_rate_order_level_mean"],
                        "ppo_fill_rate": dynamic["fill_rate_order_level_mean"],
                        "static_fill_rate": static["fill_rate_order_level_mean"],
                        "delta_fill_rate": (
                            dynamic["fill_rate_order_level_mean"]
                            - static["fill_rate_order_level_mean"]
                        ),
                        "fill_noninferior": fill_noninferior,
                        "dynamic_extra_shift_hours": dynamic[
                            "extra_shift_hours_total_mean"
                        ],
                        "ppo_extra_shift_hours": dynamic["extra_shift_hours_total_mean"],
                        "static_extra_shift_hours": static["extra_shift_hours_total_mean"],
                        "delta_extra_shift_hours": (
                            dynamic["extra_shift_hours_total_mean"]
                            - static["extra_shift_hours_total_mean"]
                        ),
                        "dynamic_buffer_target_units": dynamic[
                            "strategic_buffer_target_units_mean_mean"
                        ],
                        "ppo_buffer_target_units": dynamic[
                            "strategic_buffer_target_units_mean_mean"
                        ],
                        "static_buffer_target_units": static[
                            "strategic_buffer_target_units_mean_mean"
                        ],
                        "delta_buffer_target_units": (
                            dynamic["strategic_buffer_target_units_mean_mean"]
                            - static["strategic_buffer_target_units_mean_mean"]
                        ),
                        "dynamic_resource_composite_total": dynamic[
                            "resource_composite_total_mean"
                        ],
                        "static_resource_composite_total": static[
                            "resource_composite_total_mean"
                        ],
                        "delta_resource_composite_total": (
                            dynamic["resource_composite_total_mean"]
                            - static["resource_composite_total_mean"]
                        ),
                        "dynamic_service_loss_p95": dynamic["service_loss_p95_mean"],
                        "ppo_service_loss_p95": dynamic["service_loss_p95_mean"],
                        "static_service_loss_p95": static["service_loss_p95_mean"],
                        "dynamic_service_loss_cvar95": dynamic[
                            "service_loss_cvar95_mean"
                        ],
                        "ppo_service_loss_cvar95": dynamic["service_loss_cvar95_mean"],
                        "static_service_loss_cvar95": static["service_loss_cvar95_mean"],
                        "p95_not_worse": p95_not_worse,
                        "cvar95_not_worse": cvar95_not_worse,
                        "dynamic_pct_steps_S1": dynamic["pct_steps_S1_mean"],
                        "dynamic_pct_steps_S2": dynamic["pct_steps_S2_mean"],
                        "dynamic_pct_steps_S3": dynamic["pct_steps_S3_mean"],
                        "ppo_pct_steps_S1": dynamic["pct_steps_S1_mean"],
                        "ppo_pct_steps_S2": dynamic["pct_steps_S2_mean"],
                        "ppo_pct_steps_S3": dynamic["pct_steps_S3_mean"],
                        "static_pct_steps_S1": static["pct_steps_S1_mean"],
                        "static_pct_steps_S2": static["pct_steps_S2_mean"],
                        "static_pct_steps_S3": static["pct_steps_S3_mean"],
                        "ppo_beats_static_cd": (
                            dynamic["cd_sigmoid_mean_mean"]
                            > static["cd_sigmoid_mean_mean"]
                        ),
                        "fewer_extra_shift_hours": fewer_extra_shift_hours,
                        "lower_buffer_target": lower_buffer_target,
                        "resource_pareto_dominates": resource_pareto,
                        "strict_service_resource_dominates": (
                            resource_pareto
                            and fill_noninferior
                            and p95_not_worse
                            and cvar95_not_worse
                        ),
                    }
                )
    return comparisons


def build_best_static_by_metric(
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metric_specs = {
        "mean_ret_excel_formula": "max",
        "cd_sigmoid_mean": "max",
        "cd_raw_mean": "max",
        "fill_rate_order_level": "max",
        "flow_fill_rate": "max",
        "service_loss_mean": "min",
        "service_loss_p95": "min",
        "service_loss_cvar95": "min",
        "unattended_orders_terminal": "min",
        "extra_shift_hours_total": "min",
        "strategic_buffer_target_units_mean": "min",
        "resource_composite_total": "min",
    }
    out: list[dict[str, Any]] = []
    regimes = sorted({str(row["regime"]) for row in summary_rows})
    for regime in regimes:
        static_rows = [
            row
            for row in summary_rows
            if str(row["regime"]) == regime and str(row["policy"]) in STATIC_BASELINES
        ]
        for metric, direction in metric_specs.items():
            field = f"{metric}_mean"
            rows_with_metric = [row for row in static_rows if field in row]
            if not rows_with_metric:
                continue
            best = (
                max(rows_with_metric, key=lambda row: float(row[field]))
                if direction == "max"
                else min(rows_with_metric, key=lambda row: float(row[field]))
            )
            out.append(
                {
                    "regime": regime,
                    "metric": metric,
                    "direction": direction,
                    "best_static_policy": best["policy"],
                    "best_static_value": float(best[field]),
                }
            )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="smoke")
    parser.add_argument("--regimes", default="current,increased,severe")
    parser.add_argument("--seeds", default="8201,8202,8203")
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--train-timesteps", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=52)
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument("--reward-mode", default="ReT_garrido2024_raw")
    parser.add_argument("--observation-version", default="v4")
    parser.add_argument(
        "--ppo-initial-static-policy",
        choices=sorted(STATIC_BASELINES),
        default="static_S1_I168",
        help="Initial Track-A decision used before PPO starts acting.",
    )
    parser.add_argument(
        "--stochastic-pt",
        action="store_true",
        default=bool(P.get("stochastic_pt", False)),
        help="Enable stochastic processing times. Off by default in the 1:1 thesis lane.",
    )
    parser.add_argument("--risk-occurrence-mode", default="thesis_window")
    parser.add_argument("--risk-frequency-multiplier", type=float, default=1.0)
    parser.add_argument("--risk-impact-multiplier", type=float, default=1.0)
    parser.add_argument("--ret-g24-shift-cost", type=float, default=0.5)
    parser.add_argument("--ret-g24-kappa-train-frac", type=float, default=0.2)
    parser.add_argument("--w-bo", type=float, default=4.0)
    parser.add_argument("--w-cost", type=float, default=0.02)
    parser.add_argument("--w-disr", type=float, default=0.0)
    parser.add_argument("--control-v2-w-fill", type=float, default=1.0)
    parser.add_argument("--control-v2-w-service", type=float, default=4.0)
    parser.add_argument("--control-v2-w-lost", type=float, default=2.0)
    parser.add_argument("--control-v2-w-inventory", type=float, default=0.05)
    parser.add_argument("--control-v2-w-shift", type=float, default=0.08)
    parser.add_argument("--control-v2-w-switch", type=float, default=0.02)
    parser.add_argument("--raw-material-flow-mode", default="kit_equivalent_order_up_to")
    parser.add_argument("--raw-material-order-up-to-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--demand-on-hand-fulfillment-delay",
        type=float,
        default=P["demand_on_hand_fulfillment_delay"],
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--algo", choices=("ppo", "recurrent_ppo"), default="ppo")
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument(
        "--excel-noninferiority-tol",
        type=float,
        default=0.005,
        help="Allowed Excel ReT loss when declaring resource dominance.",
    )
    parser.add_argument("--skip-ppo", action="store_true")
    parser.add_argument("--include-threshold-heuristics", action="store_true")
    parser.add_argument("--heuristic-service-loss-threshold", type=float, default=0.05)
    parser.add_argument("--heuristic-pending-backorder-threshold", type=float, default=1_000.0)
    parser.add_argument("--heuristic-new-backorder-threshold", type=float, default=500.0)
    parser.add_argument("--heuristic-disruption-threshold", type=float, default=0.10)
    parser.add_argument("--heuristic-hold-steps", type=int, default=4)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    regimes = _parse_csv_strings(args.regimes)
    seeds = _parse_csv_ints(args.seeds)
    run_dir = args.output_dir / args.label
    model_dir = run_dir / "models"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    episode_rows: list[dict[str, Any]] = []
    trained_models: list[dict[str, Any]] = []
    for regime in regimes:
        if regime not in FROZEN_STATIC_BY_REGIME:
            raise ValueError(f"No frozen static policy declared for regime {regime!r}")
        for seed in seeds:
            model = None
            if not bool(args.skip_ppo):
                model = train_ppo(args, regime=regime, seed=seed)
                model_path = model_dir / f"ppo_{regime}_seed{seed}.zip"
                model.save(str(model_path))
                trained_models.append(
                    {
                        "algo": args.algo,
                        "regime": regime,
                        "seed": int(seed),
                        "model_path": str(model_path),
                        "train_timesteps": int(args.train_timesteps),
                    }
                )
            for ep in range(int(args.eval_episodes)):
                eval_seed = int(seed) * 1_000 + ep
                if model is not None:
                    episode_rows.append(
                        evaluate_episode(
                            args,
                            regime=regime,
                            seed=eval_seed,
                            policy_name="ppo_dynamic",
                            model=model,
                        )
                    )
                if bool(args.include_threshold_heuristics):
                    for heuristic_name in HEURISTIC_POLICIES:
                        episode_rows.append(
                            evaluate_episode(
                                args,
                                regime=regime,
                                seed=eval_seed,
                                policy_name=heuristic_name,
                            )
                        )
                for static_name, (static_level, static_shift_index) in (
                    STATIC_BASELINES.items()
                ):
                    episode_rows.append(
                        evaluate_episode(
                            args,
                            regime=regime,
                            seed=eval_seed,
                            policy_name=static_name,
                            fixed_action=static_action(
                                static_level, static_shift_index
                            ),
                        )
                    )

    summary_rows = summarize(episode_rows)
    comparison_rows = build_comparison(
        summary_rows,
        excel_noninferiority_tol=float(args.excel_noninferiority_tol),
    )
    best_static_by_metric_rows = build_best_static_by_metric(summary_rows)
    write_csv(run_dir / "episode_metrics.csv", episode_rows)
    write_csv(run_dir / "policy_summary.csv", summary_rows)
    write_csv(run_dir / "comparison_table.csv", comparison_rows)
    write_csv(run_dir / "best_static_by_metric.csv", best_static_by_metric_rows)
    payload = {
        "description": "Track-A Discrete(18) PPO vs static frontier on the Garrido-2024 C-D bar.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_metric": "cd_sigmoid_mean",
        "primary_resilience_metric": "cd_sigmoid_mean",
        "secondary_metric": "mean_ret_excel_formula",
        "secondary_metrics": [
            "mean_ret_text_formula",
            "mean_ret_excel_formula",
            "cd_sigmoid_mean",
            "cd_raw_mean",
            "cd_train_mean",
            "ret_cvar_cd_mean",
            "cd_loss_cvar95",
            "cd_cvar_estimate_terminal",
            "cd_zeta_avg/epsilon_avg/phi_avg/tau_avg/kappa_dot",
            "fill_rate_order_level",
            "backorder_qty_total",
            "pending_backorder_qty_terminal",
            "unattended_orders_terminal",
            "service_loss_p95",
            "service_loss_cvar95",
            "extra_shift_hours_total",
            "strategic_buffer_target_units_mean",
            "resource_composite_total",
            "excel_branch_fill_rate/autotomy/recovery/unfulfilled_share",
            "CTj/APj/RPj/DPj_p50/p95/p99",
            "pct_steps_S1/S2/S3",
        ],
        "config": {
            "label": args.label,
            "regimes": regimes,
            "seeds": seeds,
            "eval_episodes": int(args.eval_episodes),
            "train_timesteps": int(args.train_timesteps),
            "max_steps": int(args.max_steps),
            "step_size_hours": float(args.step_size_hours),
            "reward_mode": args.reward_mode,
            "algo": args.algo,
            "stochastic_pt": bool(args.stochastic_pt),
            "ppo_initial_static_policy": args.ppo_initial_static_policy,
            "risk_occurrence_mode": args.risk_occurrence_mode,
            "risk_frequency_multiplier": float(args.risk_frequency_multiplier),
            "risk_impact_multiplier": float(args.risk_impact_multiplier),
            "ret_g24_shift_cost": float(args.ret_g24_shift_cost),
            "ret_g24_kappa_train_frac": float(args.ret_g24_kappa_train_frac),
            "w_bo": float(args.w_bo),
            "w_cost": float(args.w_cost),
            "w_disr": float(args.w_disr),
            "control_v2_w_fill": float(args.control_v2_w_fill),
            "control_v2_w_service": float(args.control_v2_w_service),
            "control_v2_w_lost": float(args.control_v2_w_lost),
            "control_v2_w_inventory": float(args.control_v2_w_inventory),
            "control_v2_w_shift": float(args.control_v2_w_shift),
            "control_v2_w_switch": float(args.control_v2_w_switch),
            "excel_noninferiority_tol": float(args.excel_noninferiority_tol),
            "skip_ppo": bool(args.skip_ppo),
            "include_threshold_heuristics": bool(args.include_threshold_heuristics),
            "heuristic_service_loss_threshold": float(
                args.heuristic_service_loss_threshold
            ),
            "heuristic_pending_backorder_threshold": float(
                args.heuristic_pending_backorder_threshold
            ),
            "heuristic_new_backorder_threshold": float(
                args.heuristic_new_backorder_threshold
            ),
            "heuristic_disruption_threshold": float(args.heuristic_disruption_threshold),
            "heuristic_hold_steps": int(args.heuristic_hold_steps),
            "raw_material_flow_mode": args.raw_material_flow_mode,
            "raw_material_order_up_to_multiplier": float(
                args.raw_material_order_up_to_multiplier
            ),
            "demand_on_hand_fulfillment_delay": float(
                args.demand_on_hand_fulfillment_delay
            ),
        },
        "static_baselines": {
            policy: {
                "inventory_level": level,
                "shift_index": shift_index,
                "discrete_action": static_action(level, shift_index),
            }
            for policy, (level, shift_index) in STATIC_BASELINES.items()
        },
        "threshold_heuristics": HEURISTIC_POLICIES,
        "frozen_static_by_regime": {
            regime: {
                "policy": policy,
                "inventory_level": level,
                "shift_index": shift_index,
                "discrete_action": static_action(level, shift_index),
            }
            for regime, (policy, level, shift_index) in FROZEN_STATIC_BY_REGIME.items()
        },
        "trained_models": trained_models,
        "policy_summary": summary_rows,
        "comparison_table": comparison_rows,
        "best_static_by_metric": best_static_by_metric_rows,
        "artifacts": {
            "episode_metrics_csv": str(run_dir / "episode_metrics.csv"),
            "policy_summary_csv": str(run_dir / "policy_summary.csv"),
            "comparison_table_csv": str(run_dir / "comparison_table.csv"),
            "best_static_by_metric_csv": str(run_dir / "best_static_by_metric.csv"),
            "summary_json": str(run_dir / "summary.json"),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    args = build_parser().parse_args()
    summary = run(args)
    print(f"Wrote {summary['artifacts']['summary_json']}")
    for row in summary["comparison_table"]:
        print(
            f"{row['regime']} {row['dynamic_policy']}: "
            f"dynamic CD={row['dynamic_cd_sigmoid_mean']:.6f}, "
            f"static {row['static_policy']} CD={row['static_cd_sigmoid_mean']:.6f}, "
            f"delta={row['delta_cd_sigmoid_mean']:+.6f}; "
            f"Excel delta={row['delta_excel_ret']:+.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
