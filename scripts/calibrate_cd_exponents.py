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


def calibrate_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("At least one episode row is required for calibration.")
    target = G24_EQUATED_TARGET
    mean_cost = float(np.mean([float(row["average_cost"]) for row in rows]))
    if mean_cost <= 0.0:
        raise ValueError("average_cost mean must be > 0 to compute κ̇.")
    kappa_dot_values = [float(row["average_cost"]) / mean_cost for row in rows]
    maxima = {
        "zeta": float(max(float(row["zeta_avg"]) for row in rows)),
        "epsilon": float(max(float(row["epsilon_avg"]) for row in rows)),
        "phi": float(max(float(row["phi_avg"]) for row in rows)),
        "tau": float(max(float(row["tau_avg"]) for row in rows)),
        "kappa_dot": float(max(kappa_dot_values)),
    }

    def exponent_for(max_value: float) -> float:
        if max_value <= 1.0:
            raise ValueError(
                f"Calibration requires max_value > 1.0 for paper-faithful logs; got {max_value:.6f}."
            )
        return float(target / math.log(max_value))

    exponents = {
        "a_zeta": exponent_for(maxima["zeta"]),
        "b_epsilon": exponent_for(maxima["epsilon"]),
        "c_phi": exponent_for(maxima["phi"]),
        "d_tau": exponent_for(maxima["tau"]),
        "n_kappa": exponent_for(maxima["kappa_dot"]),
    }

    return {
        "source": "garrido2024_monte_carlo_maxima",
        "target_contribution": target,
        "episode_count": len(rows),
        "kappa_ref": mean_cost,
        "maxima": maxima,
        **exponents,
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
    }


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
    calibration = calibrate_from_rows(rows)
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
    print(f"  kappa_ref: {payload['kappa_ref']:.6f}")
    for key, value in payload["maxima"].items():
        print(f"  max {key}: {value:.6f}")
    for key in ("a_zeta", "b_epsilon", "c_phi", "d_tau", "n_kappa"):
        print(f"  {key}: {payload[key]:.6f}  (paper {payload['paper_reference_exponents'][key]})")
    print(f"  output: {args.output}")
    return payload


if __name__ == "__main__":
    main()
