#!/usr/bin/env python3
"""
Calibrate the paper-faithful Garrido-2024 Cobb-Douglas coefficients.

Methodology
-----------
This implements the procedure described in Garrido et al. (2024), Section 3.3:

1. Run Monte-Carlo episodes under representative decision rules.
2. Compute the five explicit C-D variables from the DES:
   ζ, ε, φ, τ, κ̇.
3. Identify the maximum value of each variable across the Monte-Carlo sample.
4. Equate each log-argument to 1/5 = 0.20:

       exponent * ln(max_value) = 0.20

   yielding:

       exponent = 0.20 / ln(max_value)

The output JSON can be passed back into the environment via
``--ret-g24-calibration``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.env_experimental_shifts import (  # noqa: E402
    G24_DEFAULT_CALIBRATION_PATH,
    G24_EQUATED_TARGET,
    MFSCGymEnvShifts,
)

POLICY_CHOICES = ("static_s1", "static_s2", "static_s3", "random")


def policy_action(policy_name: str, env: MFSCGymEnvShifts) -> np.ndarray:
    """Return the 5D action used for a calibration policy."""
    if policy_name == "static_s1":
        return np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    if policy_name == "static_s2":
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if policy_name == "static_s3":
        return np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if policy_name == "random":
        return env.action_space.sample()
    raise ValueError(f"Unsupported policy {policy_name!r}.")


def run_episode(
    *,
    env: MFSCGymEnvShifts,
    policy_name: str,
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    """Run one episode and return the final Garrido-2024 averages."""
    _, _ = env.reset(seed=seed)
    final_components: dict[str, Any] | None = None
    steps = 0

    while steps < max_steps:
        action = policy_action(policy_name, env)
        _, _, terminated, truncated, info = env.step(action)
        final_components = info.get("ret_garrido2024_components")
        steps += 1
        if terminated or truncated:
            break

    if final_components is None:
        raise RuntimeError("Calibration episode produced no Garrido-2024 components.")

    return {
        "policy": policy_name,
        "seed": seed,
        "steps": steps,
        "zeta_avg": float(final_components["zeta_avg"]),
        "epsilon_avg": float(final_components["epsilon_avg"]),
        "phi_avg": float(final_components["phi_avg"]),
        "tau_avg": float(final_components["tau_avg"]),
        "average_cost": float(final_components["average_cost"]),
    }


def collect_episode_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Collect final episode averages over the requested Monte-Carlo sample."""
    env = MFSCGymEnvShifts(
        reward_mode="ReT_garrido2024_raw",
        risk_level=args.risk_level,
        stochastic_pt=args.stochastic_pt,
        step_size_hours=args.step_size_hours,
        max_steps=args.max_steps,
        observation_version=args.observation_version,
        ret_g24_calibration_path=None,
    )

    rows: list[dict[str, Any]] = []
    policies = list(args.policies)
    for episode_idx in range(args.episodes):
        policy_name = policies[episode_idx % len(policies)]
        row = run_episode(
            env=env,
            policy_name=policy_name,
            seed=args.seed + episode_idx,
            max_steps=args.max_steps,
        )
        row["episode"] = episode_idx + 1
        rows.append(row)

    env.close()
    return rows


def calibrate_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the Garrido-2024 maxima calibration rule to episode rows."""
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
            "kappa_dot": "mean cost divided by the Monte-Carlo reference cost",
            "kappa_ref": "mean episode cost across the calibration Monte-Carlo sample",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate the paper-faithful Garrido-2024 C-D exponents."
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICY_CHOICES,
        default=list(POLICY_CHOICES),
        help="Decision rules to cycle through during calibration.",
    )
    parser.add_argument(
        "--risk-level",
        choices=["current", "increased", "severe", "severe_training"],
        default="increased",
    )
    parser.add_argument("--stochastic-pt", action="store_true")
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument(
        "--observation-version",
        choices=["v1", "v2", "v3", "v4", "v5"],
        default="v1",
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
            "episodes": int(args.episodes),
            "seed_start": int(args.seed),
            "policies": list(args.policies),
            "risk_level": str(args.risk_level),
            "stochastic_pt": bool(args.stochastic_pt),
            "step_size_hours": float(args.step_size_hours),
            "max_steps": int(args.max_steps),
            "observation_version": str(args.observation_version),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Garrido-2024 paper-faithful calibration complete")
    print(f"  episodes: {len(rows)}")
    print(f"  kappa_ref: {payload['kappa_ref']:.6f}")
    for key, value in payload["maxima"].items():
        print(f"  max {key}: {value:.6f}")
    for key in ("a_zeta", "b_epsilon", "c_phi", "d_tau", "n_kappa"):
        print(f"  {key}: {payload[key]:.6f}")
    print(f"  output: {args.output}")
    return payload


if __name__ == "__main__":
    main()
