#!/usr/bin/env python3
"""
Calibrate the paper-faithful Garrido-2024 Cobb-Douglas coefficients.

Methodology (Garrido et al. 2024, Section 3.3)
----------------------------------------------
1. Run Monte-Carlo episodes under decision rules that SPAN the comparison space.
2. Compute the five explicit C-D variables from the DES: ζ, ε, φ, τ, κ̇.
3. Identify the maximum of each variable across the Monte-Carlo sample.
4. Equate each log-argument to 1/5 = 0.20:  exponent = 0.20 / ln(max_value),
   so every term contributes ≈0.20 at its maximum (the offsets make the five
   variables comparable).
κ̇ is computed as cost / mean(cost) over the sample, which equals the paper's
7κ(S_ij)/Σκ(S_ij) (this substrategy's cost relative to the mean across the set).

IMPORTANT (2026-06-18 fix): the calibration MUST be run on the SAME faithful env
and the SAME decision space used for evaluation. The previous calibration was run
on the legacy env (legacy_renewal risk + legacy_validated inert inventory, obs v1)
with only 3 pure-capacity statics, so the maxima were unrepresentative and the
offsets mis-balanced the index on the faithful env. This script now exposes the
faithful-mode flags and sweeps the inventory x shift decision grid in the chosen
action contract.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import (  # noqa: E402
    RAW_MATERIAL_FLOW_MODE_OPTIONS,
    RISK_OCCURRENCE_MODE_OPTIONS,
)
from supply_chain.env_experimental_shifts import (  # noqa: E402
    G24_DEFAULT_CALIBRATION_PATH,
    G24_EQUATED_TARGET,
)
from supply_chain.external_env_interface import (  # noqa: E402
    THESIS_INVENTORY_PERIODS,
    make_dkana_thesis_faithful_env,
)

ActionFn = Callable[[Any], np.ndarray]


def _shift_signal(shifts: int) -> float:
    return {-1: -1.0, 1: -1.0, 2: 0.0, 3: 1.0}[shifts]


def decision_grid(action_space_mode: str) -> list[tuple[str, ActionFn]]:
    """Decision rules that SPAN the inventory x shift comparison space.

    These mirror the static baselines used at evaluation, so the calibration
    maxima reflect the range of ζ/ε/φ/τ/κ the compared policies actually reach.
    """
    grid: list[tuple[str, ActionFn]] = []
    if action_space_mode == "continuous_it_s":
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            for shifts in (1, 2, 3):
                act = np.array([float(frac), _shift_signal(shifts)], dtype=np.float32)
                grid.append((f"b{frac:.2f}_S{shifts}", lambda env, a=act: a.copy()))
    elif action_space_mode == "thesis_factorized":
        periods = [0] + [THESIS_INVENTORY_PERIODS.index(p) + 1 for p in THESIS_INVENTORY_PERIODS]
        for period_idx in periods:
            for shifts in (1, 2, 3):
                act = np.array([period_idx, shifts - 1], dtype=np.int64)
                grid.append((f"I{period_idx}_S{shifts}", lambda env, a=act: a.copy()))
    else:
        raise ValueError(
            f"Unsupported action_space_mode={action_space_mode!r}; use "
            "'continuous_it_s' or 'thesis_factorized'."
        )
    grid.append(("random", lambda env: env.action_space.sample()))
    return grid


def run_episode(
    *, env: Any, action_fn: ActionFn, label: str, seed: int, max_steps: int
) -> dict[str, Any]:
    env.reset(seed=seed)
    final_components: dict[str, Any] | None = None
    steps = 0
    while steps < max_steps:
        _, _, terminated, truncated, info = env.step(action_fn(env))
        final_components = info.get("ret_garrido2024_components")
        steps += 1
        if terminated or truncated:
            break
    if final_components is None:
        raise RuntimeError("Calibration episode produced no Garrido-2024 components.")
    return {
        "policy": label,
        "seed": seed,
        "steps": steps,
        "zeta_avg": float(final_components["zeta_avg"]),
        "epsilon_avg": float(final_components["epsilon_avg"]),
        "phi_avg": float(final_components["phi_avg"]),
        "tau_avg": float(final_components["tau_avg"]),
        "average_cost": float(final_components["average_cost"]),
        "cumulative_demanded": float(
            info.get("cumulative_demanded_post_warmup", 0.0)
        ),
        "cumulative_backorder_qty": float(
            info.get("cumulative_backorder_qty_post_warmup", 0.0)
        ),
        "pending_backorder_qty": float(info.get("pending_backorder_qty", 0.0)),
    }


def make_env(args: argparse.Namespace) -> Any:
    return make_dkana_thesis_faithful_env(
        reward_mode="ReT_garrido2024_raw",
        action_space_mode=args.action_space_mode,
        risk_level=args.risk_level,
        risk_occurrence_mode=args.risk_occurrence_mode,
        raw_material_flow_mode=args.raw_material_flow_mode,
        raw_material_order_up_to_multiplier=args.raw_material_order_up_to_multiplier,
        stochastic_pt=args.stochastic_pt,
        stochastic_pt_spread=args.stochastic_pt_spread,
        step_size_hours=args.step_size_hours,
        max_steps=args.max_steps,
        observation_version=args.observation_version,
        ret_g24_calibration_path=None,
    )


def collect_episode_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    grid = decision_grid(args.action_space_mode)
    rows: list[dict[str, Any]] = []
    for risk_level in args.risk_levels:
        args.risk_level = risk_level
        env = make_env(args)
        for episode_idx in range(args.episodes):
            label, action_fn = grid[episode_idx % len(grid)]
            row = run_episode(
                env=env,
                action_fn=action_fn,
                label=label,
                seed=args.seed + episode_idx,
                max_steps=args.max_steps,
            )
            row["episode"] = episode_idx + 1
            row["risk_level"] = risk_level
            rows.append(row)
        env.close()
    return rows


SIGNS = {"zeta": 1.0, "epsilon": -1.0, "phi": 1.0, "tau": -1.0, "kappa_dot": -1.0}
TARGET_LOGSCORE_STD = 0.30  # variance_log: scale so good static policies score ~0.85 (de-saturated)


def calibrate_from_rows(
    rows: list[dict[str, Any]], *, balance_method: str = "max_offset"
) -> dict[str, Any]:
    if not rows:
        raise ValueError("At least one episode row is required for calibration.")
    target = G24_EQUATED_TARGET
    mean_cost = float(np.mean([float(row["average_cost"]) for row in rows]))
    if mean_cost <= 0.0:
        raise ValueError("average_cost mean must be > 0 to compute κ̇.")
    kappa_dot_values = [float(row["average_cost"]) / mean_cost for row in rows]
    # Per-variable value arrays over the policy sample (κ̇ = cost/mean).
    var_values = {
        "zeta": [float(r["zeta_avg"]) for r in rows],
        "epsilon": [float(r["epsilon_avg"]) for r in rows],
        "phi": [float(r["phi_avg"]) for r in rows],
        "tau": [float(r["tau_avg"]) for r in rows],
        "kappa_dot": kappa_dot_values,
    }
    maxima = {k: float(max(v)) for k, v in var_values.items()}

    payload: dict[str, Any] = {
        "source": "garrido2024_monte_carlo_maxima",
        "balance_method": balance_method,
        "target_contribution": target,
        "episode_count": len(rows),
        "kappa_ref": mean_cost,
        "maxima": maxima,
    }

    if balance_method == "variance_log":
        # Centered log-ratio form:
        #   term_i = (c/std(ln x_i)) * (ln x_i - mean(ln x_i)).
        # This is equivalent to Cobb-Douglas on x_i / geometric_mean(x_i), so
        # absolute DES units cannot dominate the index. Pick c so the combined
        # log-score has a controlled, non-saturated spread on the calibration set.
        log_arrays = {
            k: np.log(np.maximum(np.asarray(v, dtype=float), 1e-6))
            for k, v in var_values.items()
        }
        log_mean = {k: float(np.mean(arr)) for k, arr in log_arrays.items()}
        log_std = {k: float(max(np.std(arr), 1e-6)) for k, arr in log_arrays.items()}
        base_scores = []
        for i in range(len(rows)):
            s = sum(
                SIGNS[k]
                * (1.0 / log_std[k])
                * (math.log(max(var_values[k][i], 1e-6)) - log_mean[k])
                for k in var_values
            )
            base_scores.append(s)
        base_std = float(max(np.std(base_scores), 1e-6))
        balance_c = float(TARGET_LOGSCORE_STD / base_std)
        payload["log_mean"] = log_mean
        payload["log_std"] = log_std
        payload["balance_c"] = balance_c
    elif balance_method == "minmax":
        # Min-max ablation in utility space. Positive variables use high-is-good;
        # negative variables use low-is-good. We center each utility by its
        # geometric mean so the best attainable policy is not artificially capped
        # at sigmoid(0)=0.5.
        lo = {k: float(min(v)) for k, v in var_values.items()}
        hi = {k: float(max(v)) for k, v in var_values.items()}
        utilities: dict[str, list[float]] = {k: [] for k in var_values}
        for k, values in var_values.items():
            span = max(hi[k] - lo[k], 1e-6)
            for x in values:
                norm = (float(x) - lo[k]) / span
                norm = min(max(norm, 1e-6), 1.0)
                utility = norm if SIGNS[k] > 0.0 else 1.0 - norm
                utilities[k].append(min(max(utility, 1e-6), 1.0))
        utility_gmean = {
            k: float(np.exp(np.mean(np.log(np.asarray(v, dtype=float)))))
            for k, v in utilities.items()
        }
        payload["minmax_min"] = lo
        payload["minmax_max"] = hi
        payload["minmax_utility_gmean"] = utility_gmean
    else:  # max_offset (paper/legacy)
        def exponent_for(max_value: float) -> float:
            if max_value <= 1.0:
                raise ValueError(
                    f"max_offset requires max>1.0 for paper logs; got {max_value:.6f}."
                )
            return float(target / math.log(max_value))

        payload.update(
            {
                "a_zeta": exponent_for(maxima["zeta"]),
                "b_epsilon": exponent_for(maxima["epsilon"]),
                "c_phi": exponent_for(maxima["phi"]),
                "d_tau": exponent_for(maxima["tau"]),
                "n_kappa": exponent_for(maxima["kappa_dot"]),
            }
        )

    payload.update({
        "paper_reference_exponents": {
            "a_zeta": 0.0240,
            "b_epsilon": 0.0260,
            "c_phi": 0.0400,
            "d_tau": 0.0600,
            "n_kappa": 0.1771,
        },
        "des_semantics": {
            "zeta": "mean finished-goods ration inventory over the episode",
            "epsilon": "mean pending backorder quantity over the episode",
            "phi": "mean spare assembly capacity over the episode",
            "tau": "mean net-requirement coverage time proxy over the episode",
            "kappa_dot": "mean cost divided by the Monte-Carlo reference cost "
            "(= paper 7k/Sum k)",
            "kappa_ref": "mean episode cost across the calibration Monte-Carlo sample",
        },
    })
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate the paper-faithful Garrido-2024 C-D exponents."
    )
    parser.add_argument("--episodes", type=int, default=300,
                        help="Episodes PER risk level (cycles the decision grid).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--action-space-mode",
        choices=["continuous_it_s", "thesis_factorized"],
        default="continuous_it_s",
        help="Decision contract; the grid spans inventory x shift in this contract.",
    )
    parser.add_argument(
        "--balance-method",
        choices=["max_offset", "variance_log", "minmax"],
        default="max_offset",
        help=(
            "How to balance the five C-D terms. max_offset = paper rule "
            "(0.20/ln(max)); variance_log / minmax = scale-robust so no term "
            "dominates on this DES."
        ),
    )
    parser.add_argument(
        "--risk-levels",
        nargs="+",
        default=["increased", "severe"],
        help="Calibrate maxima across the evaluated risk levels.",
    )
    # --- faithful environment contract (the 2026-06-18 fix) ---
    parser.add_argument(
        "--risk-occurrence-mode",
        choices=RISK_OCCURRENCE_MODE_OPTIONS,
        default="thesis_periodic",
    )
    parser.add_argument(
        "--raw-material-flow-mode",
        choices=RAW_MATERIAL_FLOW_MODE_OPTIONS,
        default="kit_equivalent_order_up_to",
    )
    parser.add_argument("--raw-material-order-up-to-multiplier", type=float, default=2.0)
    parser.add_argument("--stochastic-pt", action="store_true", default=True)
    parser.add_argument("--no-stochastic-pt", dest="stochastic_pt", action="store_false")
    parser.add_argument("--stochastic-pt-spread", type=float, default=1.0)
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument(
        "--observation-version",
        choices=["v1", "v2", "v3", "v4", "v5"],
        default="v5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=G24_DEFAULT_CALIBRATION_PATH,
        help="Destination JSON path for the calibration payload.",
    )
    return parser


def main() -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args()

    rows = collect_episode_rows(args)
    calibration = calibrate_from_rows(rows, balance_method=args.balance_method)
    payload = {
        **calibration,
        "run_config": {
            "episodes_per_risk_level": int(args.episodes),
            "seed_start": int(args.seed),
            "action_space_mode": str(args.action_space_mode),
            "risk_levels": list(args.risk_levels),
            "risk_occurrence_mode": str(args.risk_occurrence_mode),
            "raw_material_flow_mode": str(args.raw_material_flow_mode),
            "raw_material_order_up_to_multiplier": float(
                args.raw_material_order_up_to_multiplier
            ),
            "stochastic_pt": bool(args.stochastic_pt),
            "stochastic_pt_spread": float(args.stochastic_pt_spread),
            "step_size_hours": float(args.step_size_hours),
            "max_steps": int(args.max_steps),
            "observation_version": str(args.observation_version),
            "decision_grid": "inventory x shift span (+ random)",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Garrido-2024 faithful-env calibration complete")
    print(f"  episodes: {len(rows)}  (action={args.action_space_mode}, risks={args.risk_levels})")
    print(f"  balance_method: {payload['balance_method']}")
    print(f"  kappa_ref: {payload['kappa_ref']:.6f}")
    for key, value in payload["maxima"].items():
        print(f"  max {key}: {value:.6f}")
    if payload["balance_method"] == "max_offset":
        for key in ("a_zeta", "b_epsilon", "c_phi", "d_tau", "n_kappa"):
            print(
                f"  {key}: {payload[key]:.6f}  "
                f"(paper {payload['paper_reference_exponents'][key]})"
            )
    elif payload["balance_method"] == "variance_log":
        print(f"  balance_c: {payload['balance_c']:.6f}")
        for key, value in payload["log_std"].items():
            print(
                f"  log_mean/std {key}: "
                f"{payload['log_mean'][key]:.6f} / {value:.6f}"
            )
    elif payload["balance_method"] == "minmax":
        for key, value in payload["minmax_utility_gmean"].items():
            print(f"  utility_gmean {key}: {value:.6f}")
    print(f"  output: {args.output}")
    return payload


if __name__ == "__main__":
    main()
