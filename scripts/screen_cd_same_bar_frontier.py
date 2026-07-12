#!/usr/bin/env python3
"""Screen the Track-A static frontier under a Cobb-Douglas same-bar metric.

This is the gate before spending more PPO/DQN budget on Cobb-Douglas rewards.
If the training reward is Cobb-Douglas, the primary evaluation bar should be
the same Cobb-Douglas resilience index.  This runner therefore evaluates the
full thesis Track-A static grid, ``Discrete(18)`` = 6 inventory levels x 3
shift levels, in multiple environment cells and reports whether any cell has:

* a non-corner Cobb-Douglas optimum,
* a regime-dependent optimum,
* enough static spread/headroom to justify dynamic learning,
* and non-degenerate service behavior.

Excel ReT remains exported as a secondary continuity metric, not the selector.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.external_env_interface import (  # noqa: E402
    get_episode_terminal_metrics,
    make_discrete18_track_a_env,
)
from supply_chain.thesis_decision_env import Discrete18TrackAEnv  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("outputs/experiments/cd_same_bar_frontier")
INVENTORY_LEVEL_LABELS = {
    0: "I0",
    1: "I168",
    2: "I336",
    3: "I504",
    4: "I672",
    5: "I1344",
}
PRIMARY_METRIC_CHOICES = (
    "cd_sigmoid_mean",
    "cd_train_mean",
    "cd_raw_mean",
)


@dataclass(frozen=True)
class StaticPolicy:
    label: str
    inventory_level: int
    shift_index: int

    @property
    def shifts(self) -> int:
        return int(self.shift_index) + 1

    @property
    def inventory_label(self) -> str:
        return INVENTORY_LEVEL_LABELS[int(self.inventory_level)]

    @property
    def is_max_corner(self) -> bool:
        return self.inventory_level == 5 and self.shift_index == 2

    @property
    def is_interior_inventory(self) -> bool:
        return self.inventory_level in {1, 2, 3, 4}


def static_policies() -> list[StaticPolicy]:
    out: list[StaticPolicy] = []
    for inventory_level, inventory_label in INVENTORY_LEVEL_LABELS.items():
        for shift_index in range(3):
            out.append(
                StaticPolicy(
                    label=f"static_S{shift_index + 1}_{inventory_label}",
                    inventory_level=int(inventory_level),
                    shift_index=int(shift_index),
                )
            )
    return out


STATIC_POLICIES = static_policies()
STATIC_POLICY_BY_LABEL = {policy.label: policy for policy in STATIC_POLICIES}


def static_action(policy: StaticPolicy) -> int:
    return Discrete18TrackAEnv.encode_discrete_action(
        policy.inventory_level,
        policy.shift_index,
    )


def _parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _parse_csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _parse_csv_bools(value: str) -> list[bool]:
    out: list[bool] = []
    for part in _parse_csv_strings(value):
        normalized = part.lower()
        if normalized in {"1", "true", "t", "yes", "y"}:
            out.append(True)
        elif normalized in {"0", "false", "f", "no", "n"}:
            out.append(False)
        else:
            raise ValueError(f"Cannot parse boolean value {part!r}.")
    return out


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.fmean(finite)) if finite else float("nan")


def _std(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.stdev(finite)) if len(finite) > 1 else 0.0


def _pctl(values: list[float], q: float) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan")
    return float(np.percentile(np.asarray(finite, dtype=np.float64), q))


def _cvar95(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan")
    threshold = _pctl(finite, 95)
    tail = [value for value in finite if value >= threshold]
    return _mean(tail)


def _cell_id(
    *,
    phi: float,
    psi: float,
    stochastic_pt: bool,
    demand_multiplier: float,
    shift_cost: float,
    kappa_train_frac: float,
) -> str:
    spt = "spt" if stochastic_pt else "detpt"
    return (
        f"phi{phi:g}_psi{psi:g}_{spt}_dm{demand_multiplier:g}"
        f"_sc{shift_cost:g}_kf{kappa_train_frac:g}"
    ).replace(".", "p")


def _env_kwargs(args: argparse.Namespace, *, regime: str, cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "reward_mode": "ReT_garrido2024_raw",
        "observation_version": args.observation_version,
        "risk_level": str(regime),
        "stochastic_pt": bool(cell["stochastic_pt"]),
        "step_size_hours": float(args.step_size_hours),
        "max_steps": int(args.max_steps),
        "risk_occurrence_mode": args.risk_occurrence_mode,
        "risk_frequency_multiplier": float(cell["phi"]),
        "risk_impact_multiplier": float(cell["psi"]),
        "demand_mean_multiplier": float(cell["demand_multiplier"]),
        "ret_g24_shift_cost": float(cell["shift_cost"]),
        "ret_g24_kappa_train_frac": float(cell["kappa_train_frac"]),
        "raw_material_flow_mode": args.raw_material_flow_mode,
        "raw_material_order_up_to_multiplier": float(
            args.raw_material_order_up_to_multiplier
        ),
        "demand_on_hand_fulfillment_delay": float(
            args.demand_on_hand_fulfillment_delay
        ),
    }


def _decision_resource(info: dict[str, Any]) -> tuple[float, float, int]:
    decision = info.get("thesis_decision", {})
    if not isinstance(decision, dict):
        return 0.0, 0.0, 0
    targets = decision.get("inventory_buffer_targets", {})
    target_total = (
        sum(float(value) for value in targets.values())
        if isinstance(targets, dict)
        else 0.0
    )
    period = decision.get("inventory_period_hours")
    period_hours = 0.0 if period is None else float(period)
    inventory_level = int(decision.get("common_inventory_level", 0) or 0)
    return period_hours, target_total, inventory_level


def evaluate_static_episode(
    args: argparse.Namespace,
    *,
    cell: dict[str, Any],
    regime: str,
    policy: StaticPolicy,
    seed: int,
) -> dict[str, Any]:
    env = make_discrete18_track_a_env(**_env_kwargs(args, regime=regime, cell=cell))
    action = static_action(policy)
    obs, _info = env.reset(seed=int(seed), options={"initial_discrete_action": action})
    del obs
    done = False
    steps = 0
    reward_total = 0.0
    cd_sigmoid_total = 0.0
    cd_sigmoid_train_total = 0.0
    cd_train_total = 0.0
    cd_raw_total = 0.0
    demanded_total = 0.0
    delivered_total = 0.0
    backorder_qty_total = 0.0
    shift_hours_total = 0.0
    extra_shift_hours_total = 0.0
    strategic_buffer_target_unit_hours_total = 0.0
    service_losses: list[float] = []
    final_info: dict[str, Any] = {}

    while not done:
        _obs, reward, terminated, truncated, info = env.step(action)
        reward_total += float(reward)
        cd_sigmoid_total += float(info.get("ret_garrido2024_sigmoid_step", 0.0))
        cd_sigmoid_train_total += float(
            info.get("ret_garrido2024_sigmoid_train_step", 0.0)
        )
        cd_train_total += float(info.get("ret_garrido2024_train_step", 0.0))
        cd_raw_total += float(info.get("ret_garrido2024_raw_step", 0.0))
        new_demanded = float(info.get("new_demanded", 0.0))
        new_backorder_qty = float(info.get("new_backorder_qty", 0.0))
        demanded_total += new_demanded
        delivered_total += float(info.get("new_delivered", 0.0))
        backorder_qty_total += new_backorder_qty
        shifts_active = int(info.get("shifts_active", policy.shifts))
        shift_hours_total += float(shifts_active) * float(args.step_size_hours)
        extra_shift_hours_total += max(0, shifts_active - 1) * float(
            args.step_size_hours
        )
        _period_hours, target_total, _inventory_level = _decision_resource(info)
        strategic_buffer_target_unit_hours_total += target_total * float(
            args.step_size_hours
        )
        service_loss_step = info.get("service_loss_step")
        if service_loss_step is None:
            service_loss_step = (
                new_backorder_qty / new_demanded if new_demanded > 0.0 else 0.0
            )
        service_losses.append(float(service_loss_step))
        final_info = dict(info)
        steps += 1
        done = bool(terminated or truncated)

    terminal = get_episode_terminal_metrics(env)
    env.close()
    total_steps = max(1, steps)
    flow_delivery_ratio = delivered_total / demanded_total if demanded_total > 0 else math.nan
    flow_fill_rate = (
        min(1.0, max(0.0, flow_delivery_ratio))
        if math.isfinite(flow_delivery_ratio)
        else math.nan
    )
    return {
        "cell_id": cell["cell_id"],
        "regime": regime,
        "seed": int(seed),
        "policy": policy.label,
        "inventory_level": int(policy.inventory_level),
        "inventory_label": policy.inventory_label,
        "shift_index": int(policy.shift_index),
        "shifts": int(policy.shifts),
        "is_max_corner": bool(policy.is_max_corner),
        "is_interior_inventory": bool(policy.is_interior_inventory),
        "phi": float(cell["phi"]),
        "psi": float(cell["psi"]),
        "stochastic_pt": bool(cell["stochastic_pt"]),
        "demand_multiplier": float(cell["demand_multiplier"]),
        "ret_g24_shift_cost": float(cell["shift_cost"]),
        "ret_g24_kappa_train_frac": float(cell["kappa_train_frac"]),
        "steps": int(steps),
        "reward_total": float(reward_total),
        "cd_sigmoid_total": float(cd_sigmoid_total),
        "cd_sigmoid_mean": float(cd_sigmoid_total / total_steps),
        "cd_sigmoid_train_total": float(cd_sigmoid_train_total),
        "cd_sigmoid_train_mean": float(cd_sigmoid_train_total / total_steps),
        "cd_train_total": float(cd_train_total),
        "cd_train_mean": float(cd_train_total / total_steps),
        "cd_raw_total": float(cd_raw_total),
        "cd_raw_mean": float(cd_raw_total / total_steps),
        "mean_ret_excel_formula": float(
            terminal["order_level_ret_excel_formula_mean"]
        ),
        "mean_ret_text_formula": float(terminal["order_level_ret_text_formula_mean"]),
        "fill_rate_order_level": float(terminal["fill_rate_order_level"]),
        "backorder_rate_order_level": float(terminal["backorder_rate_order_level"]),
        "flow_delivery_ratio": float(flow_delivery_ratio),
        "flow_fill_rate": float(flow_fill_rate),
        "demanded_total": float(demanded_total),
        "delivered_total": float(delivered_total),
        "backorder_qty_total": float(backorder_qty_total),
        "pending_backorder_qty_terminal": float(
            final_info.get("pending_backorder_qty", 0.0)
        ),
        "unattended_orders_terminal": float(
            final_info.get("unattended_orders_total", 0.0)
        ),
        "service_loss_mean": _mean(service_losses),
        "service_loss_p95": _pctl(service_losses, 95),
        "service_loss_cvar95": _cvar95(service_losses),
        "shift_hours_total": float(shift_hours_total),
        "extra_shift_hours_total": float(extra_shift_hours_total),
        "strategic_buffer_target_units_mean": float(
            strategic_buffer_target_unit_hours_total
            / max(float(args.step_size_hours) * total_steps, 1.0)
        ),
        "strategic_buffer_target_unit_hours_total": float(
            strategic_buffer_target_unit_hours_total
        ),
        "resource_composite_total": float(
            extra_shift_hours_total + strategic_buffer_target_unit_hours_total
        ),
        "zeta_avg": float(final_info.get("zeta_avg", math.nan)),
        "epsilon_avg": float(final_info.get("epsilon_avg", math.nan)),
        "phi_avg": float(final_info.get("phi_avg", math.nan)),
        "tau_avg": float(final_info.get("tau_avg", math.nan)),
        "kappa_dot": float(final_info.get("kappa_dot", math.nan)),
    }


def _numeric_mean_fields(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    return {f"{key}_mean": _mean(float(row[key]) for row in rows) for key in keys}


def summarize_by_policy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["cell_id"]), str(row["regime"]), str(row["policy"])),
            [],
        ).append(row)

    out: list[dict[str, Any]] = []
    for (cell_id, regime, policy), bucket in sorted(grouped.items()):
        first = bucket[0]
        out.append(
            {
                "cell_id": cell_id,
                "regime": regime,
                "policy": policy,
                "inventory_level": int(first["inventory_level"]),
                "inventory_label": str(first["inventory_label"]),
                "shift_index": int(first["shift_index"]),
                "shifts": int(first["shifts"]),
                "is_max_corner": bool(first["is_max_corner"]),
                "is_interior_inventory": bool(first["is_interior_inventory"]),
                "phi": float(first["phi"]),
                "psi": float(first["psi"]),
                "stochastic_pt": bool(first["stochastic_pt"]),
                "demand_multiplier": float(first["demand_multiplier"]),
                "ret_g24_shift_cost": float(first["ret_g24_shift_cost"]),
                "ret_g24_kappa_train_frac": float(first["ret_g24_kappa_train_frac"]),
                "n": len(bucket),
                **_numeric_mean_fields(bucket),
            }
        )
    return out


def _top_policy(
    policy_rows: list[dict[str, Any]],
    *,
    metric: str,
) -> dict[str, Any]:
    mean_field = f"{metric}_mean"
    return max(policy_rows, key=lambda row: float(row[mean_field]))


def _regime_wrong_policy_penalty(
    policy_rows: list[dict[str, Any]],
    *,
    metric: str,
    top_by_regime: dict[str, dict[str, Any]],
) -> float:
    mean_field = f"{metric}_mean"
    rows_by_regime_policy = {
        (str(row["regime"]), str(row["policy"])): row for row in policy_rows
    }
    penalties: list[float] = []
    regimes = sorted(top_by_regime)
    for regime in regimes:
        optimal = top_by_regime[regime]
        optimal_value = float(optimal[mean_field])
        for other_regime in regimes:
            if other_regime == regime:
                continue
            wrong_policy = str(top_by_regime[other_regime]["policy"])
            wrong_row = rows_by_regime_policy.get((regime, wrong_policy))
            if wrong_row is None:
                continue
            penalties.append(optimal_value - float(wrong_row[mean_field]))
    return _mean(penalties)


def summarize_by_cell(
    policy_rows: list[dict[str, Any]],
    *,
    primary_metric: str,
    headroom_threshold: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in policy_rows:
        grouped.setdefault(str(row["cell_id"]), []).append(row)

    mean_field = f"{primary_metric}_mean"
    out: list[dict[str, Any]] = []
    for cell_id, bucket in sorted(grouped.items()):
        first = bucket[0]
        regimes = sorted({str(row["regime"]) for row in bucket})
        top_by_regime: dict[str, dict[str, Any]] = {}
        spread_by_regime: dict[str, float] = {}
        best_flow_by_regime: dict[str, float] = {}
        for regime in regimes:
            regime_rows = [row for row in bucket if str(row["regime"]) == regime]
            top = _top_policy(regime_rows, metric=primary_metric)
            top_by_regime[regime] = top
            values = [float(row[mean_field]) for row in regime_rows]
            spread_by_regime[regime] = max(values) - min(values)
            best_flow_by_regime[regime] = float(top["flow_fill_rate_mean"])

        top_policies = [str(top["policy"]) for top in top_by_regime.values()]
        all_non_corner = all(
            not bool(STATIC_POLICY_BY_LABEL[policy].is_max_corner)
            for policy in top_policies
        )
        all_interior_inventory = all(
            bool(STATIC_POLICY_BY_LABEL[policy].is_interior_inventory)
            for policy in top_policies
        )
        regime_dependent = len(set(top_policies)) > 1
        avg_spread = _mean(spread_by_regime.values())
        min_spread = min(spread_by_regime.values()) if spread_by_regime else math.nan
        wrong_penalty = _regime_wrong_policy_penalty(
            bucket,
            metric=primary_metric,
            top_by_regime=top_by_regime,
        )
        eligible = (
            all_non_corner
            and all_interior_inventory
            and regime_dependent
            and avg_spread >= float(headroom_threshold)
        )
        out.append(
            {
                "cell_id": cell_id,
                "primary_metric": primary_metric,
                "phi": float(first["phi"]),
                "psi": float(first["psi"]),
                "stochastic_pt": bool(first["stochastic_pt"]),
                "demand_multiplier": float(first["demand_multiplier"]),
                "ret_g24_shift_cost": float(first["ret_g24_shift_cost"]),
                "ret_g24_kappa_train_frac": float(first["ret_g24_kappa_train_frac"]),
                "n_policy_rows": len(bucket),
                "regimes": ",".join(regimes),
                "top_policy_by_regime": json.dumps(
                    {
                        regime: {
                            "policy": str(row["policy"]),
                            primary_metric: float(row[mean_field]),
                            "flow_fill_rate": float(row["flow_fill_rate_mean"]),
                            "excel_ret": float(row["mean_ret_excel_formula_mean"]),
                            "resource_composite_total": float(
                                row["resource_composite_total_mean"]
                            ),
                        }
                        for regime, row in top_by_regime.items()
                    },
                    sort_keys=True,
                ),
                "top_policies": ",".join(top_policies),
                "top_policy_set_size": len(set(top_policies)),
                "all_regime_tops_non_corner": all_non_corner,
                "all_regime_tops_interior_inventory": all_interior_inventory,
                "regime_dependent_top": regime_dependent,
                "avg_static_spread": float(avg_spread),
                "min_static_spread": float(min_spread),
                "avg_wrong_regime_penalty": float(wrong_penalty),
                "best_flow_min": min(best_flow_by_regime.values()),
                "best_flow_max": max(best_flow_by_regime.values()),
                "best_flow_mean": _mean(best_flow_by_regime.values()),
                "eligible_for_cd_training": bool(eligible),
                "selection_score": float(
                    (1.0 if eligible else 0.0)
                    + avg_spread
                    + max(0.0, wrong_penalty)
                    + (0.1 if regime_dependent else 0.0)
                    + (0.1 if all_non_corner else 0.0)
                    + (0.1 if all_interior_inventory else 0.0)
                ),
            }
        )
    return sorted(out, key=lambda row: float(row["selection_score"]), reverse=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_cells(args: argparse.Namespace) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for phi in _parse_csv_floats(args.phis):
        for psi in _parse_csv_floats(args.psis):
            for stochastic_pt in _parse_csv_bools(args.stochastic_pt_values):
                for demand_multiplier in _parse_csv_floats(args.demand_multipliers):
                    for shift_cost in _parse_csv_floats(args.ret_g24_shift_costs):
                        for kappa_frac in _parse_csv_floats(args.ret_g24_kappa_train_fracs):
                            cell = {
                                "phi": float(phi),
                                "psi": float(psi),
                                "stochastic_pt": bool(stochastic_pt),
                                "demand_multiplier": float(demand_multiplier),
                                "shift_cost": float(shift_cost),
                                "kappa_train_frac": float(kappa_frac),
                            }
                            cell["cell_id"] = _cell_id(**cell)
                            cells.append(cell)
    return cells


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    regimes = _parse_csv_strings(args.regimes)
    seeds = _parse_csv_ints(args.seeds)
    cells = build_cells(args)
    policies = STATIC_POLICIES

    rows: list[dict[str, Any]] = []
    total = len(cells) * len(regimes) * len(policies) * len(seeds)
    completed = 0
    for cell in cells:
        for regime in regimes:
            for policy in policies:
                for seed in seeds:
                    completed += 1
                    if completed == 1 or completed % max(1, int(args.progress_every)) == 0:
                        print(
                            f"[{completed}/{total}] {cell['cell_id']} "
                            f"{regime} {policy.label} seed={seed}",
                            flush=True,
                        )
                    rows.append(
                        evaluate_static_episode(
                            args,
                            cell=cell,
                            regime=regime,
                            policy=policy,
                            seed=seed,
                        )
                    )

    policy_rows = summarize_by_policy(rows)
    cell_rows = summarize_by_cell(
        policy_rows,
        primary_metric=str(args.primary_cd_metric),
        headroom_threshold=float(args.headroom_threshold),
    )
    write_csv(output_dir / "static_episode_rows.csv", rows)
    write_csv(output_dir / "summary_by_policy.csv", policy_rows)
    write_csv(output_dir / "summary_by_cell.csv", cell_rows)
    recommended = cell_rows[0] if cell_rows else None
    payload = {
        "description": "Static Track-A Cobb-Douglas same-bar frontier screen.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_metric": str(args.primary_cd_metric),
        "secondary_metrics": [
            "mean_ret_excel_formula",
            "flow_fill_rate",
            "service_loss_cvar95",
            "resource_composite_total",
            "cd_raw_mean",
            "cd_train_mean",
            "cd_sigmoid_train_mean",
        ],
        "config": {
            "regimes": regimes,
            "seeds": seeds,
            "phis": _parse_csv_floats(args.phis),
            "psis": _parse_csv_floats(args.psis),
            "stochastic_pt_values": _parse_csv_bools(args.stochastic_pt_values),
            "demand_multipliers": _parse_csv_floats(args.demand_multipliers),
            "ret_g24_shift_costs": _parse_csv_floats(args.ret_g24_shift_costs),
            "ret_g24_kappa_train_fracs": _parse_csv_floats(
                args.ret_g24_kappa_train_fracs
            ),
            "max_steps": int(args.max_steps),
            "step_size_hours": float(args.step_size_hours),
            "observation_version": args.observation_version,
            "risk_occurrence_mode": args.risk_occurrence_mode,
            "demand_on_hand_fulfillment_delay": float(
                args.demand_on_hand_fulfillment_delay
            ),
            "headroom_threshold": float(args.headroom_threshold),
        },
        "n_episode_rows": len(rows),
        "n_policy_rows": len(policy_rows),
        "n_cells": len(cells),
        "recommended_cell": recommended,
        "top_cells": cell_rows[:10],
        "continuous_buffer_note": (
            "Not evaluated here: this runner deliberately stays on the thesis "
            "Discrete(18) Track-A action surface. Continuous buffers require a "
            "separate diagnostic wrapper or a new action-space claim."
        ),
        "artifacts": {
            "static_episode_rows_csv": str(output_dir / "static_episode_rows.csv"),
            "summary_by_policy_csv": str(output_dir / "summary_by_policy.csv"),
            "summary_by_cell_csv": str(output_dir / "summary_by_cell.csv"),
            "summary_json": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_dir / 'summary.json'}")
    if recommended:
        print(
            "Recommended C-D cell: "
            f"{recommended['cell_id']} "
            f"eligible={recommended['eligible_for_cd_training']} "
            f"tops={recommended['top_policies']} "
            f"spread={float(recommended['avg_static_spread']):.6f}"
        )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--regimes", default="current,increased,severe")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--max-steps", type=int, default=52)
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument(
        "--primary-cd-metric",
        choices=PRIMARY_METRIC_CHOICES,
        default="cd_sigmoid_mean",
    )
    parser.add_argument("--headroom-threshold", type=float, default=0.005)
    parser.add_argument("--observation-version", default="v4")
    parser.add_argument("--risk-occurrence-mode", default="thesis_window")
    parser.add_argument("--phis", default="1.0,1.5,2.0")
    parser.add_argument("--psis", default="1.0,1.25,1.5")
    parser.add_argument("--stochastic-pt-values", default="False,True")
    parser.add_argument("--demand-multipliers", default="1.0,1.1")
    parser.add_argument("--ret-g24-shift-costs", default="0.5,1.0")
    parser.add_argument("--ret-g24-kappa-train-fracs", default="0.2,1.0")
    parser.add_argument("--raw-material-flow-mode", default="kit_equivalent_order_up_to")
    parser.add_argument("--raw-material-order-up-to-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--demand-on-hand-fulfillment-delay",
        type=float,
        default=P["demand_on_hand_fulfillment_delay"],
    )
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
