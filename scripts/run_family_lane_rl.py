#!/usr/bin/env python3
"""RL family-lane runner (Garrido-pure, 2026-06-29).

Trains PPO with `continuous_its` and `per_op_buffer` action contracts on
R1-only, R2-only, and R3-only environments. Compares against the dense
static frontier from `outputs/benchmarks/family_static_frontier_2026-06-29/`.

Each (family, action_contract, cfi) combination runs with N seeds at
horizon h104 (2 years post-warmup), freeze defaults, no war multipliers.

Output:
  outputs/experiments/family_lane_rl_2026-06-29/
    {family}_{contract}_{cfi_label}/         — per-seed training output
      model.zip
      metrics.json
      vec_normalize.pkl
    family_lane_summary.csv                  — (family, contract, cfi, seed, mean_ret, ...)
    family_lane_audit.json                   — machine-readable summary
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from supply_chain.continuous_its_env import (  # noqa: E402
    make_continuous_its_track_a_env,
    make_per_op_buffer_track_a_env,
)
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.thesis_design import (  # noqa: E402
    R1_RISKS,
    R2_RISKS,
    R3_RISKS,
)


FAMILIES = {
    "R1": {
        "risks": R1_RISKS,
        "horizon_hours": 161_280.0,
        "default_static": "L168_S3",  # from 3.1 frontier
    },
    "R2": {
        "risks": R2_RISKS,
        "horizon_hours": 80_640.0,
        "default_static": "L168_S1",
    },
    "R3": {
        "risks": R3_RISKS,
        "horizon_hours": 161_280.0,
        "default_static": "L0_S1",
    },
}

CFIS_PER_FAMILY = {
    "R1": [1, 2, 3, 4, 5],
    "R2": [11, 12, 13, 14, 15],
    "R3": [21, 22, 23, 24, 25],
}

CONTRACTS = ("continuous_its", "per_op_buffer")


def make_env_fn(
    *,
    family: str,
    contract: str,
    cfi: int,
    seed: int,
    horizon: float = 104.0,
):
    """Build the env factory for a (family, contract, cfi) tuple."""
    fam = FAMILIES[family]
    enabled_risks = fam["risks"]
    horizon_h = float(horizon) * 168.0

    def _build():
        common_overrides = dict(
            reward_mode="ReT_excel_plus_cvar",
            observation_version="v6",
            risk_level="current",
            enabled_risks=enabled_risks,
            risk_frequency_multiplier=1.0,
            risk_impact_multiplier=1.0,
            stochastic_pt=False,
            max_steps=int(horizon),
            step_size_hours=168.0,
            demand_mean_multiplier=1.0,
            ret_excel_cvar_alpha=0.2,
        )
        if contract == "continuous_its":
            return make_continuous_its_track_a_env(
                **common_overrides,
                init_frac=0.0,
                risk_obs=True,
                holding_cost=0.0,
                shift_cost=0.0,
                replenishment_period=168.0,
            )
        if contract == "per_op_buffer":
            return make_per_op_buffer_track_a_env(
                **common_overrides,
                init_fracs=(0.0, 0.0, 0.0),
                risk_obs=True,
                holding_cost=0.0,
                shift_cost=0.0,
                replenishment_period=168.0,
            )
        raise ValueError(f"Unknown contract {contract!r}")

    return _build


def evaluate(
    model: PPO,
    env_factory,
    n_episodes: int,
    seed0: int,
) -> dict[str, float]:
    """Evaluate the trained policy on N episodes with fixed seeds."""
    rets: list[float] = []
    service_losses: list[float] = []
    resources: list[float] = []
    lost_counts: list[int] = []
    fill_rates: list[float] = []
    for ep in range(int(n_episodes)):
        env = env_factory()
        try:
            obs, _info = env.reset(seed=int(seed0) + ep)
            done = truncated = False
            ep_resource = []
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _reward, done, truncated, info = env.step(action)
                ep_resource.append(
                    float(info.get("resource_composite", 0.0))
                )
            sim = getattr(env, "unwrapped", env).sim
            metrics = compute_episode_metrics(sim)
            rets.append(float(metrics.get("ret_excel", 0.0)))
            service_losses.append(
                float(metrics.get("service_loss_auc_per_order", 0.0))
            )
            resources.append(float(np.mean(ep_resource)))
            lost_counts.append(int(metrics.get("n_lost", 0)))
            fill_rates.append(float(metrics.get("fill_rate", 0.0)))
        finally:
            try:
                env.close()
            except Exception:
                pass
    return {
        "mean_ret_mean": fmean(rets) if rets else 0.0,
        "mean_ret_sd": float(np.std(rets)) if len(rets) > 1 else 0.0,
        "service_loss_mean": fmean(service_losses) if service_losses else 0.0,
        "resource_mean": fmean(resources) if resources else 0.0,
        "resource_sd": float(np.std(resources)) if len(resources) > 1 else 0.0,
        "lost_orders_mean": fmean(lost_counts) if lost_counts else 0.0,
        "fill_rate_mean": fmean(fill_rates) if fill_rates else 0.0,
        "n_episodes": int(n_episodes),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families", default="R1,R2,R3", help="comma-separated family list"
    )
    parser.add_argument(
        "--contracts", default="continuous_its,per_op_buffer"
    )
    parser.add_argument(
        "--cfi",
        default=None,
        help="comma-separated CFIs; default: first 2 of each family",
    )
    parser.add_argument(
        "--seeds", default="1,2,3", help="comma-separated seeds"
    )
    parser.add_argument(
        "--timesteps", type=int, default=40_000, help="PPO timesteps per run"
    )
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument(
        "--eval-episodes", type=int, default=4
    )
    parser.add_argument(
        "--horizon-weeks", type=int, default=104
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/family_lane_rl_2026-06-29"),
    )
    parser.add_argument(
        "--quick", action="store_true", help="10k timesteps, 1 seed, 1 CFi"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.quick:
        args.timesteps = 10_000
        args.seeds = "1"
        args.cfi = "1,11,21"

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    contracts = [c.strip() for c in args.contracts.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.cfi:
        cfis_by_family: dict[str, list[int]] = {
            f: [int(c) for c in args.cfi.split(",") if c.strip()]
            for f in families
        }
    else:
        cfis_by_family = {
            f: list(CFIS_PER_FAMILY[f][:2]) for f in families
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    rows: list[dict[str, Any]] = []
    for family in families:
        for contract in contracts:
            for cfi in cfis_by_family[family]:
                for seed in seeds:
                    label = (
                        f"{family}_{contract}_cf{cfi}_s{seed}"
                    )
                    run_dir = args.output_dir / label
                    run_dir.mkdir(parents=True, exist_ok=True)
                    print(
                        f"[{label}] building env (family={family}, "
                        f"contract={contract}, cfi={cfi}, seed={seed})"
                    )
                    env_factory = make_env_fn(
                        family=family,
                        contract=contract,
                        cfi=cfi,
                        seed=seed,
                        horizon=float(args.horizon_weeks),
                    )

                    def _vec_env_factory():
                        return DummyVecEnv([env_factory])

                    vec_env = _vec_env_factory()
                    t0 = time.time()
                    model = PPO(
                        "MlpPolicy",
                        vec_env,
                        n_steps=512,
                        batch_size=64,
                        learning_rate=3e-4,
                        verbose=0,
                        seed=int(seed),
                    )
                    model.learn(total_timesteps=int(args.timesteps))
                    train_seconds = time.time() - t0
                    print(
                        f"[{label}] trained in {train_seconds:.1f}s, "
                        f"evaluating..."
                    )
                    metrics = evaluate(
                        model,
                        env_factory,
                        n_episodes=int(args.eval_episodes),
                        seed0=int(seed) * 1000,
                    )
                    metrics["train_seconds"] = train_seconds
                    metrics["family"] = family
                    metrics["contract"] = contract
                    metrics["cfi"] = cfi
                    metrics["seed"] = seed
                    metrics["timesteps"] = int(args.timesteps)
                    metrics["horizon_weeks"] = int(args.horizon_weeks)
                    rows.append(metrics)
                    metrics_path = run_dir / "metrics.json"
                    metrics_path.write_text(
                        json.dumps(metrics, indent=2), encoding="utf-8"
                    )
                    try:
                        vec_env.close()
                    except Exception:
                        pass

    fieldnames = list(rows[0].keys())
    with (args.output_dir / "family_lane_summary.csv").open(
        "w", newline=""
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - started
    audit = {
        "families": families,
        "contracts": contracts,
        "cfis_by_family": {k: v for k, v in cfis_by_family.items()},
        "seeds": seeds,
        "timesteps": int(args.timesteps),
        "horizon_weeks": int(args.horizon_weeks),
        "n_runs": len(rows),
        "wall_seconds": elapsed,
    }
    (args.output_dir / "family_lane_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    print(f"\nWROTE {args.output_dir} (wall {elapsed:.1f}s, {len(rows)} runs)")
    for row in rows:
        print(
            f"  {row['family']}/{row['contract']}/cf{row['cfi']}/s{row['seed']}: "
            f"mean_ret={row['mean_ret_mean']:.4f} "
            f"lost={row['lost_orders_mean']:.1f} "
            f"resource={row['resource_mean']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
