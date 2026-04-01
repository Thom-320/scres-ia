#!/usr/bin/env python3
"""Deterministically calibrate ReT_unified_v1 on static MFSC baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_control_reward import static_policy_action
from supply_chain.external_env_interface import run_episodes

GARRIDO_POLICY_ORDER: tuple[str, ...] = (
    "garrido_cf_s1",
    "garrido_cf_s2",
    "garrido_cf_s3",
)
STATIC_POLICY_ORDER: tuple[str, ...] = ("static_s1", "static_s2", "static_s3")
THETA_SC_GRID: tuple[float, ...] = (0.76, 0.78, 0.80)
THETA_BC_GRID: tuple[float, ...] = (0.70, 0.75, 0.80)
KAPPA_GRID: tuple[float, ...] = (0.10, 0.15, 0.20)
DEFAULT_BETA = 12.0
DEFAULT_W_FR = 0.60
DEFAULT_W_RC = 0.25
DEFAULT_W_CE = 0.15
DEFAULT_SELECTION_RULE = (
    "1) garrido_cf_s2 > garrido_cf_s3 > garrido_cf_s1; "
    "2) static_s2 > static_s3 > static_s1; "
    "3) S2-S3 >= 1%; 4) lowest kappa; "
    "5) tie-break theta_sc=0.78 then theta_bc=0.75"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate ReT_unified_v1 on exact static Garrido baselines."
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("supply_chain/data/ret_unified_v1_calibration.json"),
    )
    parser.add_argument(
        "--output-grid-csv",
        type=Path,
        default=Path("supply_chain/data/ret_unified_v1_calibration_grid.csv"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("supply_chain/data/ret_unified_v1_calibration_summary.json"),
    )
    parser.add_argument(
        "--risk-level", choices=["current", "increased", "severe"], default="increased"
    )
    parser.add_argument("--stochastic-pt", action="store_true", default=True)
    parser.add_argument("--step-size-hours", type=float, default=168.0)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument(
        "--observation-version",
        choices=["v1", "v2", "v3", "v4", "v5"],
        default="v4",
    )
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument(
        "--theta-sc-grid",
        type=float,
        nargs="+",
        default=list(THETA_SC_GRID),
    )
    parser.add_argument(
        "--theta-bc-grid",
        type=float,
        nargs="+",
        default=list(THETA_BC_GRID),
    )
    parser.add_argument(
        "--kappa-grid",
        type=float,
        nargs="+",
        default=list(KAPPA_GRID),
    )
    return parser.parse_args()


def parameter_grid(
    theta_sc_grid: Iterable[float] = THETA_SC_GRID,
    theta_bc_grid: Iterable[float] = THETA_BC_GRID,
    kappa_grid: Iterable[float] = KAPPA_GRID,
) -> list[dict[str, float]]:
    return [
        {
            "theta_sc": theta_sc,
            "theta_bc": theta_bc,
            "beta": DEFAULT_BETA,
            "kappa": kappa,
        }
        for theta_sc in theta_sc_grid
        for theta_bc in theta_bc_grid
        for kappa in kappa_grid
    ]


def policy_mean_reward(
    policy_name: str,
    *,
    params: dict[str, float],
    args: argparse.Namespace,
) -> float:
    reward_values: list[float] = []
    action_payload = static_policy_action(policy_name)

    def _policy_fn(
        obs: np.ndarray, info: dict[str, Any]
    ) -> np.ndarray | dict[str, float | int]:
        if isinstance(action_payload, dict):
            return dict(action_payload)
        return np.asarray(action_payload, dtype=np.float32)

    for seed in args.seeds:
        rows = run_episodes(
            _policy_fn,
            n_episodes=args.eval_episodes,
            seed=seed,
            env_kwargs={
                "reward_mode": "ReT_unified_v1",
                "risk_level": args.risk_level,
                "observation_version": args.observation_version,
                "stochastic_pt": args.stochastic_pt,
                "step_size_hours": args.step_size_hours,
                "max_steps": args.max_steps,
                "ret_unified_theta_sc": params["theta_sc"],
                "ret_unified_theta_bc": params["theta_bc"],
                "ret_unified_beta": params["beta"],
                "ret_unified_kappa": params["kappa"],
            },
            policy_name=policy_name,
        )
        reward_values.extend(float(row["reward_total"]) for row in rows)
    return float(np.mean(reward_values))


def evaluate_combo(
    params: dict[str, float], args: argparse.Namespace
) -> dict[str, Any]:
    rewards = {
        policy_name: policy_mean_reward(policy_name, params=params, args=args)
        for policy_name in (*GARRIDO_POLICY_ORDER, *STATIC_POLICY_ORDER)
    }
    garrido_ok = (
        rewards["garrido_cf_s2"] > rewards["garrido_cf_s3"] > rewards["garrido_cf_s1"]
    )
    static_ok = rewards["static_s2"] > rewards["static_s3"] > rewards["static_s1"]
    s2_s3_margin = rewards["garrido_cf_s2"] - rewards["garrido_cf_s3"]
    margin_ok = s2_s3_margin >= 0.01 * max(abs(rewards["garrido_cf_s2"]), 1e-9)
    return {
        **params,
        **{f"{policy}_reward_mean": value for policy, value in rewards.items()},
        "garrido_rank_ok": garrido_ok,
        "static_rank_ok": static_ok,
        "garrido_s2_minus_s3": s2_s3_margin,
        "garrido_margin_ok": margin_ok,
        "accepted": bool(garrido_ok and static_ok and margin_ok),
    }


def _theta_sc_priority(value: float) -> int:
    return 0 if np.isclose(value, 0.78) else 1


def _theta_bc_priority(value: float) -> int:
    return 0 if np.isclose(value, 0.75) else 1


def select_best_combo(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if bool(row["accepted"])]
    if not accepted:
        raise ValueError("No ReT_unified_v1 calibration candidate satisfied the rules.")
    accepted.sort(
        key=lambda row: (
            float(row["kappa"]),
            _theta_sc_priority(float(row["theta_sc"])),
            _theta_bc_priority(float(row["theta_bc"])),
            -float(row["garrido_s2_minus_s3"]),
        )
    )
    return accepted[0]


def fallback_default_combo(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        if (
            np.isclose(float(row["theta_sc"]), 0.78)
            and np.isclose(float(row["theta_bc"]), 0.78)
            and np.isclose(float(row["beta"]), DEFAULT_BETA)
            and np.isclose(float(row["kappa"]), 0.20)
        ):
            return dict(row)
    return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = [
        evaluate_combo(params, args)
        for params in parameter_grid(
            theta_sc_grid=args.theta_sc_grid,
            theta_bc_grid=args.theta_bc_grid,
            kappa_grid=args.kappa_grid,
        )
    ]

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_grid_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)

    accepted_candidates = sum(1 for row in rows if bool(row["accepted"]))
    selection_status = "selected"
    try:
        best_row = select_best_combo(rows)
        selected_theta_sc = float(best_row["theta_sc"])
        selected_theta_bc = float(best_row["theta_bc"])
        selected_beta = float(best_row["beta"])
        selected_kappa = float(best_row["kappa"])
        source = "scripts/calibrate_ret_unified_v1.py"
    except ValueError:
        selection_status = "no_valid_candidate_fallback_defaults"
        best_row = fallback_default_combo(rows) or {
            "theta_sc": 0.78,
            "theta_bc": 0.78,
            "beta": DEFAULT_BETA,
            "kappa": 0.20,
        }
        selected_theta_sc = 0.78
        selected_theta_bc = 0.78
        selected_beta = DEFAULT_BETA
        selected_kappa = 0.20
        source = "scripts/calibrate_ret_unified_v1.py:fallback_defaults"

    calibration_payload = {
        "theta_sc": selected_theta_sc,
        "theta_bc": selected_theta_bc,
        "beta": selected_beta,
        "kappa": selected_kappa,
        "w_fr": DEFAULT_W_FR,
        "w_rc": DEFAULT_W_RC,
        "w_ce": DEFAULT_W_CE,
        "selection_rule": DEFAULT_SELECTION_RULE,
        "source": source,
        "selection_status": selection_status,
        "risk_level": args.risk_level,
        "stochastic_pt": bool(args.stochastic_pt),
        "observation_version": args.observation_version,
        "step_size_hours": float(args.step_size_hours),
        "max_steps": int(args.max_steps),
        "eval_episodes": int(args.eval_episodes),
        "seeds": [int(seed) for seed in args.seeds],
        "theta_sc_grid": [float(value) for value in args.theta_sc_grid],
        "theta_bc_grid": [float(value) for value in args.theta_bc_grid],
        "kappa_grid": [float(value) for value in args.kappa_grid],
    }
    summary_payload = {
        "selected": calibration_payload,
        "accepted_candidates": accepted_candidates,
        "total_candidates": len(rows),
        "selection_status": selection_status,
        "best_candidate_metrics": best_row,
    }

    args.output_json.write_text(
        json.dumps(calibration_payload, indent=2),
        encoding="utf-8",
    )
    write_csv(args.output_grid_csv, rows)
    args.output_summary.write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    print("Selected ReT_unified_v1 calibration:")
    print(
        f"  theta_sc={selected_theta_sc:.2f} "
        f"theta_bc={selected_theta_bc:.2f} "
        f"beta={selected_beta:.2f} "
        f"kappa={selected_kappa:.2f}"
    )
    if "garrido_cf_s1_reward_mean" in best_row:
        print(
            "  Garrido means: "
            f"S1={best_row['garrido_cf_s1_reward_mean']:.3f}, "
            f"S2={best_row['garrido_cf_s2_reward_mean']:.3f}, "
            f"S3={best_row['garrido_cf_s3_reward_mean']:.3f}"
        )
    if selection_status != "selected":
        print(
            "  No candidate satisfied the strict acceptance rules; defaults were frozen as fallback."
        )
    print(f"Saved calibration JSON to {args.output_json}")
    print(f"Saved calibration grid CSV to {args.output_grid_csv}")
    print(f"Saved calibration summary JSON to {args.output_summary}")


if __name__ == "__main__":
    main()
