#!/usr/bin/env python3
"""Screen cost-aware metrics on the Garrido Excel static experiment grid.

This is a post-processing harness: it does not rerun the DES and it does not
change the faithful Excel replication metric.  It reads the static experiment
rows produced by ``scripts/run_garrido_excel_experiments.py`` and scores each
policy with three cost-aware candidates:

- multiplicative Excel discount:
  ReT_cost = ReT_excel * sqrt(cap_eff * inv_eff)
- subtractive Excel net-benefit:
  ReT_cost = ReT_excel - A_S * (S-1)/2 - A_I * (I/I_ref)
- Garrido-2024 family pointer:
  recomputed by static DES reruns because the five-variable index needs
  step-level inventory/backorder/capacity/cost state

The screening goal is modest: find whether a cost-aware metric produces an
interior static optimum before spending RL budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import INVENTORY_BUFFERS  # noqa: E402
from supply_chain.env_experimental_shifts import MFSCGymEnvShifts  # noqa: E402


DEFAULT_INPUT = Path("outputs/experiments/garrido_excel_static_expanded_2026-06-26/rows.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/experiments/garrido_cost_metric_screen")
DEFAULT_INVENTORY_REF = 11_344.0


@dataclass(frozen=True)
class ScreenConfig:
    inventory_ref: float = DEFAULT_INVENTORY_REF
    k_s_values: tuple[float, ...] = (
        0.0,
        0.005,
        0.01,
        0.02,
        0.03,
        0.05,
        0.075,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
    )
    k_i_values: tuple[float, ...] = (
        0.0,
        0.005,
        0.01,
        0.02,
        0.03,
        0.05,
        0.075,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
    )
    a_s_values: tuple[float, ...] = (
        0.0,
        0.001,
        0.0025,
        0.005,
        0.0075,
        0.01,
        0.015,
        0.02,
        0.03,
        0.05,
    )
    a_i_values: tuple[float, ...] = (
        0.0,
        0.001,
        0.0025,
        0.005,
        0.0075,
        0.01,
        0.015,
        0.02,
        0.03,
        0.05,
    )
    cd_regimes: tuple[str, ...] = ("current", "increased", "severe")
    cd_seeds: tuple[int, ...] = (7, 13, 29)
    cd_max_steps: int = 52
    cd_step_size_hours: float = 168.0
    cd_kappa_train_fracs: tuple[float, ...] = (0.20, 0.40, 0.60, 1.00)
    cd_shift_costs: tuple[float, ...] = (0.50, 1.00, 2.00)
    cd_risk_frequency_multipliers: tuple[float, ...] = (1.0,)
    cd_risk_impact_multipliers: tuple[float, ...] = (1.0,)
    cd_policy_grid: str = "source_rows"
    cd_stochastic_pt: bool = True


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.fmean(vals)) if vals else math.nan


def inventory_total_from_profile(profile: str, inventory_period: str | int | None) -> float:
    """Return the initial buffer total represented by a static experiment row."""
    if profile in ("", "I0"):
        return 0.0
    if profile == "matched":
        if inventory_period in ("", None):
            return 0.0
        return float(sum(float(v) for v in INVENTORY_BUFFERS[int(inventory_period)].values()))
    if profile.startswith("I"):
        return float(sum(float(v) for v in INVENTORY_BUFFERS[int(profile[1:])].values()))
    return 0.0


def row_inventory_total(row: dict[str, Any]) -> float:
    return inventory_total_from_profile(
        str(row.get("initial_buffer_profile", "")),
        row.get("inventory_period", ""),
    )


def multiplicative_score(
    row: dict[str, Any],
    *,
    k_s: float,
    k_i: float,
    inventory_ref: float = DEFAULT_INVENTORY_REF,
) -> dict[str, float]:
    ret = float(row["mean_ret_excel_formula"])
    shifts = int(row["shifts"])
    inventory_total = row_inventory_total(row)
    cap_eff = max(0.0, 1.0 - float(k_s) * (shifts - 1.0) / 2.0)
    inv_eff = 1.0 / (1.0 + float(k_i) * inventory_total / max(inventory_ref, 1e-9))
    cost_eff = math.sqrt(cap_eff * inv_eff)
    return {
        "score": ret * cost_eff,
        "ret_excel": ret,
        "cap_eff": cap_eff,
        "inv_eff": inv_eff,
        "cost_eff": cost_eff,
        "inventory_total": inventory_total,
    }


def subtractive_score(
    row: dict[str, Any],
    *,
    a_s: float,
    a_i: float,
    inventory_ref: float = DEFAULT_INVENTORY_REF,
) -> dict[str, float]:
    ret = float(row["mean_ret_excel_formula"])
    shifts = int(row["shifts"])
    inventory_total = row_inventory_total(row)
    shift_penalty = float(a_s) * (shifts - 1.0) / 2.0
    inventory_penalty = float(a_i) * inventory_total / max(inventory_ref, 1e-9)
    return {
        "score": ret - shift_penalty - inventory_penalty,
        "ret_excel": ret,
        "shift_penalty": shift_penalty,
        "inventory_penalty": inventory_penalty,
        "inventory_total": inventory_total,
    }


def _policy_key(row: dict[str, Any]) -> tuple[str, str, int, str, str, str, float]:
    return (
        str(row["policy"]),
        str(row.get("policy_kind", "")),
        int(row["shifts"]),
        str(row.get("inventory_period", "")),
        str(row.get("initial_buffer_profile", "")),
        str(row.get("raw_material_flow_mode", "kit_equivalent_order_up_to")),
        float(row.get("raw_material_order_up_to_multiplier", 2.0) or 2.0),
    )


def static_policy_descriptors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    descriptors: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _policy_key(row)
        policy = key[0]
        descriptors.setdefault(
            policy,
            {
                "policy": policy,
                "policy_kind": key[1],
                "shifts": key[2],
                "inventory_period": key[3],
                "initial_buffer_profile": key[4],
                "raw_material_flow_mode": key[5],
                "raw_material_order_up_to_multiplier": key[6],
            },
        )
    return [descriptors[name] for name in sorted(descriptors)]


def joint_6x3_descriptors() -> list[dict[str, Any]]:
    """Return the Track-A joint inventory-buffer x shift static grid."""
    descriptors: list[dict[str, Any]] = []
    for inventory_period in ("", "168", "336", "504", "672", "1344"):
        profile = "I0" if inventory_period == "" else f"I{inventory_period}"
        inv_label = "I0" if inventory_period == "" else f"I{inventory_period}"
        for shifts in (1, 2, 3):
            descriptors.append(
                {
                    "policy": f"joint_{inv_label}_S{shifts}",
                    "policy_kind": "joint_inventory_shift",
                    "shifts": shifts,
                    "inventory_period": inventory_period,
                    "initial_buffer_profile": profile,
                    "raw_material_flow_mode": "kit_equivalent_order_up_to",
                    "raw_material_order_up_to_multiplier": 2.0,
                }
            )
    return descriptors


def policy_descriptors_for_screen(
    rows: list[dict[str, Any]], config: ScreenConfig
) -> list[dict[str, Any]]:
    if config.cd_policy_grid == "source_rows":
        return static_policy_descriptors(rows)
    if config.cd_policy_grid == "joint_6x3":
        return joint_6x3_descriptors()
    raise ValueError(f"Unsupported cd_policy_grid={config.cd_policy_grid!r}.")


def _initial_buffers_for_descriptor(desc: dict[str, Any]) -> dict[str, float] | None:
    profile = str(desc.get("initial_buffer_profile", ""))
    if profile in ("", "I0"):
        return None
    if profile == "matched":
        # The Excel-matched CF buffers are forensic per-CF values.  For this
        # Track-A DES screen, keep the env's normal reset inventory instead of
        # replaying CF-specific workbook state.
        return None
    if profile.startswith("I"):
        return {
            key: float(value)
            for key, value in INVENTORY_BUFFERS[int(profile[1:])].items()
        }
    return None


def _inventory_period_for_descriptor(desc: dict[str, Any]) -> int | None:
    period = str(desc.get("inventory_period", ""))
    return None if period == "" else int(period)


def _run_garrido2024_episode(
    desc: dict[str, Any],
    *,
    risk_level: str,
    seed: int,
    kappa_train_frac: float,
    shift_cost: float,
    risk_frequency_multiplier: float,
    risk_impact_multiplier: float,
    config: ScreenConfig,
) -> dict[str, Any]:
    shifts = int(desc["shifts"])
    env = MFSCGymEnvShifts(
        reward_mode="ReT_garrido2024_train",
        observation_version="v4",
        step_size_hours=float(config.cd_step_size_hours),
        max_steps=int(config.cd_max_steps),
        risk_level=str(risk_level),
        stochastic_pt=bool(config.cd_stochastic_pt),
        year_basis="thesis",
        warmup_trigger="op9_arrival",
        downstream_q_source="figure_6_2",
        r14_defect_mode="thesis_strict_op6",
        risk_occurrence_mode="thesis_window",
        risk_frequency_multiplier=float(risk_frequency_multiplier),
        risk_impact_multiplier=float(risk_impact_multiplier),
        raw_material_flow_mode=str(desc["raw_material_flow_mode"]),
        raw_material_order_up_to_multiplier=float(
            desc["raw_material_order_up_to_multiplier"]
        ),
        ret_g24_kappa_train_frac=float(kappa_train_frac),
        ret_g24_shift_cost=float(shift_cost),
    )
    reset_options = {
        "initial_buffers": _initial_buffers_for_descriptor(desc),
        "initial_shifts": shifts,
        "inventory_replenishment_period": _inventory_period_for_descriptor(desc),
    }
    env.reset(seed=int(seed), options=reset_options)
    action = {"assembly_shifts": shifts}
    reward_total = 0.0
    raw_total = 0.0
    train_total = 0.0
    sigmoid_total = 0.0
    final_info: dict[str, Any] = {}
    done = False
    while not done:
        _, reward, terminated, truncated, info = env.step(action)
        reward_total += float(reward)
        raw_total += float(info["ret_garrido2024_raw_step"])
        train_total += float(info["ret_garrido2024_train_step"])
        sigmoid_total += float(info["ret_garrido2024_sigmoid_step"])
        final_info = dict(info)
        done = bool(terminated or truncated)
    ret = env.sim.compute_order_level_ret() if env.sim is not None else {}
    env.close()
    return {
        "policy": str(desc["policy"]),
        "policy_kind": str(desc["policy_kind"]),
        "risk_level": str(risk_level),
        "seed": int(seed),
        "shifts": shifts,
        "inventory_period": str(desc.get("inventory_period", "")),
        "initial_buffer_profile": str(desc.get("initial_buffer_profile", "")),
        "raw_material_flow_mode": str(desc["raw_material_flow_mode"]),
        "raw_material_order_up_to_multiplier": float(
            desc["raw_material_order_up_to_multiplier"]
        ),
        "policy_grid": str(config.cd_policy_grid),
        "ret_garrido2024_raw_total": raw_total,
        "ret_garrido2024_train_total": train_total,
        "ret_garrido2024_sigmoid_total": sigmoid_total,
        "reward_total": reward_total,
        "risk_frequency_multiplier": float(risk_frequency_multiplier),
        "risk_impact_multiplier": float(risk_impact_multiplier),
        "mean_ret_excel_formula": float(ret.get("mean_ret_excel_formula", math.nan)),
        "fill_rate_order_level": float(ret.get("fill_rate_order_level", math.nan)),
        "zeta_avg": float(final_info.get("zeta_avg", math.nan)),
        "epsilon_avg": float(final_info.get("epsilon_avg", math.nan)),
        "phi_avg": float(final_info.get("phi_avg", math.nan)),
        "tau_avg": float(final_info.get("tau_avg", math.nan)),
        "kappa_dot": float(final_info.get("kappa_dot", math.nan)),
    }


def rank_policies(
    rows: list[dict[str, Any]],
    scorer: Callable[[dict[str, Any]], float],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        policy = str(row["policy"])
        grouped.setdefault(policy, []).append(float(scorer(row)))
        meta.setdefault(
            policy,
            {
                "policy": policy,
                "policy_kind": row.get("policy_kind", ""),
            },
        )
    ranked = []
    for policy, values in grouped.items():
        ranked.append({**meta[policy], "mean_score": _mean(values), "n": len(values)})
    return sorted(ranked, key=lambda item: float(item["mean_score"]), reverse=True)


def _is_corner(policy: str) -> bool:
    return policy in {
        "shift_S1",
        "shift_S3",
        "inventory_I0",
        "inventory_I1344",
        "raw_legacy_x1",
    }


def _has_positive_params(item: dict[str, Any]) -> bool:
    return any(
        float(item.get(key, 0.0)) > 0.0
        for key in ("k_s", "k_i", "a_s", "a_i")
    )


def calibrate_multiplicative(
    rows: list[dict[str, Any]], config: ScreenConfig
) -> dict[str, Any]:
    candidates = []
    for k_s in config.k_s_values:
        for k_i in config.k_i_values:
            ranked = rank_policies(
                rows,
                lambda row, k_s=k_s, k_i=k_i: multiplicative_score(
                    row, k_s=k_s, k_i=k_i, inventory_ref=config.inventory_ref
                )["score"],
            )
            top = ranked[0]
            candidates.append(
                {
                    "metric": "excel_cost_multiplicative",
                    "k_s": k_s,
                    "k_i": k_i,
                    "top_policy": top["policy"],
                    "top_score": top["mean_score"],
                    "top_is_corner": _is_corner(str(top["policy"])),
                    "ranking": ranked[:8],
                }
            )
    positive_non_corner = [
        item
        for item in candidates
        if _has_positive_params(item) and not item["top_is_corner"]
    ]
    non_corner = [item for item in candidates if not item["top_is_corner"]]
    selected = positive_non_corner[0] if positive_non_corner else candidates[0]
    default = next(
        (
            item
            for item in candidates
            if item["k_s"] == 0.40 and item["k_i"] == 0.25
        ),
        None,
    )
    return {
        "selected": selected,
        "first_positive_non_corner": (
            positive_non_corner[0] if positive_non_corner else None
        ),
        "first_non_corner": non_corner[0] if non_corner else None,
        "default_k04_k025": default,
    }


def calibrate_subtractive(rows: list[dict[str, Any]], config: ScreenConfig) -> dict[str, Any]:
    candidates = []
    for a_s in config.a_s_values:
        for a_i in config.a_i_values:
            ranked = rank_policies(
                rows,
                lambda row, a_s=a_s, a_i=a_i: subtractive_score(
                    row, a_s=a_s, a_i=a_i, inventory_ref=config.inventory_ref
                )["score"],
            )
            top = ranked[0]
            candidates.append(
                {
                    "metric": "excel_cost_subtractive",
                    "a_s": a_s,
                    "a_i": a_i,
                    "top_policy": top["policy"],
                    "top_score": top["mean_score"],
                    "top_is_corner": _is_corner(str(top["policy"])),
                    "ranking": ranked[:8],
                }
            )
    positive_non_corner = [
        item
        for item in candidates
        if _has_positive_params(item) and not item["top_is_corner"]
    ]
    non_corner = [item for item in candidates if not item["top_is_corner"]]
    selected = positive_non_corner[0] if positive_non_corner else candidates[0]
    default = next(
        (
            item
            for item in candidates
            if item["a_s"] == 0.02 and item["a_i"] == 0.02
        ),
        None,
    )
    return {
        "selected": selected,
        "first_positive_non_corner": (
            positive_non_corner[0] if positive_non_corner else None
        ),
        "first_non_corner": non_corner[0] if non_corner else None,
        "default_a02_a02": default,
    }


def family_rankings(
    rows: list[dict[str, Any]], scorer: Callable[[dict[str, Any]], float]
) -> dict[str, list[dict[str, Any]]]:
    families = sorted({str(row.get("family", "")) for row in rows})
    return {
        family: rank_policies(
            [row for row in rows if str(row.get("family", "")) == family], scorer
        )[:8]
        for family in families
    }


def run_garrido2024_des_screen(
    rows: list[dict[str, Any]], config: ScreenConfig
) -> list[dict[str, Any]]:
    descriptors = policy_descriptors_for_screen(rows, config)
    out: list[dict[str, Any]] = []
    for kappa_frac in config.cd_kappa_train_fracs:
        for shift_cost in config.cd_shift_costs:
            for risk_frequency_multiplier in config.cd_risk_frequency_multipliers:
                for risk_impact_multiplier in config.cd_risk_impact_multipliers:
                    for risk_level in config.cd_regimes:
                        for desc in descriptors:
                            for seed in config.cd_seeds:
                                row = _run_garrido2024_episode(
                                    desc,
                                    risk_level=str(risk_level),
                                    seed=int(seed),
                                    kappa_train_frac=float(kappa_frac),
                                    shift_cost=float(shift_cost),
                                    risk_frequency_multiplier=float(
                                        risk_frequency_multiplier
                                    ),
                                    risk_impact_multiplier=float(
                                        risk_impact_multiplier
                                    ),
                                    config=config,
                                )
                                row["ret_g24_kappa_train_frac"] = float(kappa_frac)
                                row["ret_g24_shift_cost"] = float(shift_cost)
                                out.append(row)
    return out


def _rank_g24_policies(rows: list[dict[str, Any]], score_field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        policy = str(row["policy"])
        grouped.setdefault(policy, []).append(float(row[score_field]))
        meta.setdefault(policy, {"policy": policy, "policy_kind": row.get("policy_kind", "")})
    ranked = [
        {
            **meta[policy],
            "mean_score": _mean(values),
            "n": len(values),
        }
        for policy, values in grouped.items()
    ]
    return sorted(ranked, key=lambda item: float(item["mean_score"]), reverse=True)


def _ci95(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        value = float(values[0])
        return value, value
    mean = statistics.fmean(values)
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return float(mean - half), float(mean + half)


def paired_top_margin(
    rows: list[dict[str, Any]],
    *,
    top_policy: str,
    runner_up_policy: str,
    score_field: str,
) -> dict[str, Any]:
    by_policy_seed: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        by_policy_seed.setdefault(
            (str(row["policy"]), int(row["seed"])), []
        ).append(float(row[score_field]))

    diffs: list[float] = []
    for policy, seed in sorted(by_policy_seed):
        if policy != top_policy:
            continue
        other_key = (runner_up_policy, seed)
        if other_key not in by_policy_seed:
            continue
        top_score = statistics.fmean(by_policy_seed[(top_policy, seed)])
        other_score = statistics.fmean(by_policy_seed[other_key])
        diffs.append(float(top_score - other_score))

    ci_low, ci_high = _ci95(diffs)
    return {
        "top_policy": top_policy,
        "runner_up_policy": runner_up_policy,
        "n_paired_seeds": len(diffs),
        "mean_diff": _mean(diffs),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "signs": [
            "+" if value > 0.0 else ("-" if value < 0.0 else "0")
            for value in diffs
        ],
        "robust_positive": bool(diffs and ci_low > 0.0),
    }


def summarize_garrido2024_des(
    rows: list[dict[str, Any]],
    *,
    score_field: str = "ret_garrido2024_train_total",
) -> dict[str, Any]:
    if not rows:
        return {
            "status": "not_run",
            "score_field": score_field,
            "note": "Pass DES rows to build_summary or run script without --skip-cd-des.",
        }

    policy_grids = sorted({str(row.get("policy_grid", "")) for row in rows})
    candidates: list[dict[str, Any]] = []
    combos = sorted(
        {
            (
                float(row["ret_g24_kappa_train_frac"]),
                float(row["ret_g24_shift_cost"]),
                float(row.get("risk_frequency_multiplier", 1.0)),
                float(row.get("risk_impact_multiplier", 1.0)),
            )
            for row in rows
        }
    )
    for kappa_frac, shift_cost, risk_frequency_multiplier, risk_impact_multiplier in combos:
        combo_rows = [
            row
            for row in rows
            if float(row["ret_g24_kappa_train_frac"]) == kappa_frac
            and float(row["ret_g24_shift_cost"]) == shift_cost
            and float(row.get("risk_frequency_multiplier", 1.0))
            == risk_frequency_multiplier
            and float(row.get("risk_impact_multiplier", 1.0))
            == risk_impact_multiplier
        ]
        ranked_overall = _rank_g24_policies(combo_rows, score_field)
        top_by_regime: dict[str, dict[str, Any]] = {}
        top_margin_by_regime: dict[str, dict[str, Any]] = {}
        for regime in sorted({str(row["risk_level"]) for row in combo_rows}):
            regime_rows = [
                row for row in combo_rows if str(row["risk_level"]) == regime
            ]
            regime_ranked = _rank_g24_policies(regime_rows, score_field)
            top_by_regime[regime] = regime_ranked[0]
            if len(regime_ranked) > 1:
                top_margin_by_regime[regime] = paired_top_margin(
                    regime_rows,
                    top_policy=str(regime_ranked[0]["policy"]),
                    runner_up_policy=str(regime_ranked[1]["policy"]),
                    score_field=score_field,
                )
        top_policies = [str(item["policy"]) for item in top_by_regime.values()]
        robust_regime_tops = all(
            margin.get("robust_positive", False)
            for margin in top_margin_by_regime.values()
        )
        candidate = {
            "metric": "garrido2024_cobb_douglas_des",
            "score_field": score_field,
            "ret_g24_kappa_train_frac": kappa_frac,
            "ret_g24_shift_cost": shift_cost,
            "risk_frequency_multiplier": risk_frequency_multiplier,
            "risk_impact_multiplier": risk_impact_multiplier,
            "top_policy": ranked_overall[0]["policy"],
            "top_score": ranked_overall[0]["mean_score"],
            "top_is_corner": _is_corner(str(ranked_overall[0]["policy"])),
            "top_by_regime": top_by_regime,
            "top_margin_by_regime": top_margin_by_regime,
            "regime_dependent_top": len(set(top_policies)) > 1,
            "robust_regime_tops": robust_regime_tops,
            "robust_regime_dependent_top": (
                len(set(top_policies)) > 1 and robust_regime_tops
            ),
            "all_regime_tops_non_corner": all(
                not _is_corner(policy) for policy in top_policies
            ),
            "ranking": ranked_overall[:8],
        }
        candidates.append(candidate)

    eligible = [
        item
        for item in candidates
        if item["all_regime_tops_non_corner"] and item["regime_dependent_top"]
        and item["robust_regime_tops"]
    ]
    non_corner = [item for item in candidates if not item["top_is_corner"]]
    selected = eligible[0] if eligible else (non_corner[0] if non_corner else candidates[0])
    return {
        "status": "computed",
        "score_field": score_field,
        "policy_grid": policy_grids[0] if len(policy_grids) == 1 else policy_grids,
        "n_rows": len(rows),
        "n_candidates": len(candidates),
        "selected": selected,
        "first_eligible_interior_regime_dependent": eligible[0] if eligible else None,
        "first_non_corner": non_corner[0] if non_corner else None,
        "candidates": candidates,
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def build_summary(
    rows: list[dict[str, Any]],
    config: ScreenConfig,
    *,
    garrido2024_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    mult = calibrate_multiplicative(rows, config)
    sub = calibrate_subtractive(rows, config)
    mult_sel = mult["selected"]
    sub_sel = sub["selected"]
    return {
        "description": "Cost-aware screening over Garrido Excel static rows; no DES rerun.",
        "primary_ret_metric": "mean_ret_excel_formula",
        "inventory_ref": config.inventory_ref,
        "n_rows": len(rows),
        "n_policies": len({row["policy"] for row in rows}),
        "baseline_excel_ranking": rank_policies(
            rows, lambda row: float(row["mean_ret_excel_formula"])
        )[:10],
        "excel_cost_multiplicative": {
            **mult,
            "family_rankings": family_rankings(
                rows,
                lambda row: multiplicative_score(
                    row,
                    k_s=float(mult_sel["k_s"]),
                    k_i=float(mult_sel["k_i"]),
                    inventory_ref=config.inventory_ref,
                )["score"],
            ),
        },
        "excel_cost_subtractive": {
            **sub,
            "family_rankings": family_rankings(
                rows,
                lambda row: subtractive_score(
                    row,
                    a_s=float(sub_sel["a_s"]),
                    a_i=float(sub_sel["a_i"]),
                    inventory_ref=config.inventory_ref,
                )["score"],
            ),
        },
        "garrido2024_family": summarize_garrido2024_des(garrido2024_rows or []),
    }


def write_outputs(
    summary: dict[str, Any],
    output_dir: Path,
    *,
    garrido2024_rows: list[dict[str, Any]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    rows = []
    for metric_key in ("excel_cost_multiplicative", "excel_cost_subtractive"):
        selected = summary[metric_key]["selected"]
        for rank, item in enumerate(selected["ranking"], start=1):
            rows.append(
                {
                    "metric": metric_key,
                    "rank": rank,
                    "policy": item["policy"],
                    "policy_kind": item["policy_kind"],
                    "mean_score": item["mean_score"],
                    "n": item["n"],
                    "selected_params": json.dumps(
                        {
                            key: value
                            for key, value in selected.items()
                            if key
                            in {
                                "k_s",
                                "k_i",
                                "a_s",
                                "a_i",
                            }
                        },
                        sort_keys=True,
                    ),
                }
            )
    g24 = summary.get("garrido2024_family", {})
    if g24.get("status") == "computed":
        selected = g24["selected"]
        for rank, item in enumerate(selected["ranking"], start=1):
            rows.append(
                {
                    "metric": "garrido2024_cobb_douglas_des",
                    "rank": rank,
                    "policy": item["policy"],
                    "policy_kind": item["policy_kind"],
                    "mean_score": item["mean_score"],
                    "n": item["n"],
                    "selected_params": json.dumps(
                        {
                            "ret_g24_kappa_train_frac": selected[
                                "ret_g24_kappa_train_frac"
                            ],
                            "ret_g24_shift_cost": selected["ret_g24_shift_cost"],
                            "risk_frequency_multiplier": selected[
                                "risk_frequency_multiplier"
                            ],
                            "risk_impact_multiplier": selected[
                                "risk_impact_multiplier"
                            ],
                            "policy_grid": summary["garrido2024_family"].get(
                                "policy_grid", ""
                            ),
                            "score_field": selected["score_field"],
                        },
                        sort_keys=True,
                    ),
                }
            )
    with (output_dir / "selected_rankings.csv").open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "metric",
                "rank",
                "policy",
                "policy_kind",
                "mean_score",
                "n",
                "selected_params",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if garrido2024_rows:
        fieldnames = list(garrido2024_rows[0].keys())
        with (output_dir / "garrido2024_des_rows.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(garrido2024_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--inventory-ref", type=float, default=DEFAULT_INVENTORY_REF)
    parser.add_argument(
        "--skip-cd-des",
        action="store_true",
        help="Skip the Garrido-2024 static DES rerun and only post-process Excel rows.",
    )
    parser.add_argument(
        "--cd-regimes",
        nargs="+",
        default=["current", "increased", "severe"],
    )
    parser.add_argument("--cd-seeds", type=int, nargs="+", default=[7, 13, 29])
    parser.add_argument("--cd-max-steps", type=int, default=52)
    parser.add_argument("--cd-step-size-hours", type=float, default=168.0)
    parser.add_argument(
        "--cd-kappa-train-fracs",
        type=float,
        nargs="+",
        default=[0.20, 0.40, 0.60, 1.00],
    )
    parser.add_argument(
        "--cd-shift-costs",
        type=float,
        nargs="+",
        default=[0.50, 1.00, 2.00],
    )
    parser.add_argument(
        "--cd-risk-frequency-multipliers",
        type=float,
        nargs="+",
        default=[1.0],
    )
    parser.add_argument(
        "--cd-risk-impact-multipliers",
        type=float,
        nargs="+",
        default=[1.0],
    )
    parser.add_argument(
        "--cd-policy-grid",
        choices=["source_rows", "joint_6x3"],
        default="source_rows",
    )
    parser.add_argument("--cd-deterministic-pt", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = load_rows(args.input_csv)
    config = ScreenConfig(
        inventory_ref=float(args.inventory_ref),
        cd_regimes=tuple(str(value) for value in args.cd_regimes),
        cd_seeds=tuple(int(value) for value in args.cd_seeds),
        cd_max_steps=int(args.cd_max_steps),
        cd_step_size_hours=float(args.cd_step_size_hours),
        cd_kappa_train_fracs=tuple(float(value) for value in args.cd_kappa_train_fracs),
        cd_shift_costs=tuple(float(value) for value in args.cd_shift_costs),
        cd_risk_frequency_multipliers=tuple(
            float(value) for value in args.cd_risk_frequency_multipliers
        ),
        cd_risk_impact_multipliers=tuple(
            float(value) for value in args.cd_risk_impact_multipliers
        ),
        cd_policy_grid=str(args.cd_policy_grid),
        cd_stochastic_pt=not bool(args.cd_deterministic_pt),
    )
    garrido2024_rows = [] if args.skip_cd_des else run_garrido2024_des_screen(rows, config)
    summary = build_summary(rows, config, garrido2024_rows=garrido2024_rows)
    write_outputs(summary, args.output_dir, garrido2024_rows=garrido2024_rows)
    print(f"Wrote {args.output_dir / 'summary.json'}")
    print(f"Wrote {args.output_dir / 'selected_rankings.csv'}")
    if garrido2024_rows:
        print(f"Wrote {args.output_dir / 'garrido2024_des_rows.csv'}")
    for metric_key in ("excel_cost_multiplicative", "excel_cost_subtractive"):
        selected = summary[metric_key]["selected"]
        print(
            f"{metric_key}: top={selected['top_policy']} "
            f"score={float(selected['top_score']):.6f}"
        )
    g24 = summary["garrido2024_family"]
    if g24.get("status") == "computed":
        selected = g24["selected"]
        print(
            "garrido2024_cobb_douglas_des: "
            f"top={selected['top_policy']} "
            f"score={float(selected['top_score']):.6f} "
            f"regime_dependent={selected['regime_dependent_top']} "
            f"robust_regime_dependent={selected['robust_regime_dependent_top']} "
            f"risk_freq={float(selected['risk_frequency_multiplier']):g} "
            f"risk_impact={float(selected['risk_impact_multiplier']):g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
