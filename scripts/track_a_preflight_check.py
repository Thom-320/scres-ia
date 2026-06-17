#!/usr/bin/env python3
"""Preflight gate for Track A Kaggle runs.

This script does not train.  It validates the contract that must hold before
spending Kaggle time on Track A:

- Garrido decision variables only: `thesis_factorized`.
- Thesis-faithful backbone: `thesis_periodic`, `kit_equivalent_order_up_to`,
  and raw-material order-up-to multiplier `m2.0`.
- Tail/recovery reward: `ReT_tail_v1`, with auditable steepness transforms.
- Stochastic processing time enabled at the historical spread.
- Panel Cf profile and held-out eval seed base are explicit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import run_track_a_exhaustion_sweep as sweep  # noqa: E402
from supply_chain.dkana_env import make_dkana_thesis_faithful_env  # noqa: E402


def command_value(command: list[str], flag: str) -> str | None:
    if flag not in command:
        return None
    idx = command.index(flag)
    if idx + 1 >= len(command):
        return None
    return command[idx + 1]


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "pass": bool(passed), "detail": detail})


def build_strict_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        train_timesteps=args.train_timesteps,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        eval_seed_base=args.eval_seed_base,
        max_steps=args.max_steps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        device=args.device,
        history_window=args.history_window,
        norm_reward=args.norm_reward,
        use_cf_risk_profile=True,
        panel_cfis=args.panel_cfis,
        ret_tail_w_sc=args.ret_tail_w_sc,
        ret_tail_w_rc=args.ret_tail_w_rc,
        ret_tail_w_ce=args.ret_tail_w_ce,
        ret_tail_cap_kappa=args.ret_tail_cap_kappa,
        ret_tail_inv_kappa=args.ret_tail_inv_kappa,
        ret_tail_boost=args.ret_tail_boost,
        ret_tail_transform=args.ret_tail_transform,
        ret_tail_gamma=args.ret_tail_gamma,
        ret_tail_beta=args.ret_tail_beta,
    )


def build_probe_command(args: argparse.Namespace) -> list[str]:
    strict = build_strict_args(args)
    return sweep.build_command(
        args=strict,
        run_root=args.output_root / "runs",
        label="track_a_preflight_probe",
        algo=args.algo,
        action_space_mode="thesis_factorized",
        reward_profile="ret_tail",
        risk_level=args.risk_level,
        pt_profile="stoch_pt_hist",
    )


def validate_command(command: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    expected_pairs = {
        "--algo": args.algo,
        "--action-space-mode": "thesis_factorized",
        "--reward-mode": "ReT_tail_v1",
        "--risk-level": args.risk_level,
        "--risk-occurrence-mode": "thesis_periodic",
        "--raw-material-flow-mode": "kit_equivalent_order_up_to",
        "--raw-material-order-up-to-multiplier": "2.0",
        "--inventory-period-mode": "thesis_strict",
        "--train-cfis": args.panel_cfis,
        "--garrido-cfis": args.panel_cfis,
        "--train-risk-profile": args.risk_level,
        "--eval-risk-profile": args.risk_level,
        "--n-envs": str(args.n_envs),
        "--eval-seed-base": str(args.eval_seed_base),
        "--ret-tail-transform": args.ret_tail_transform,
    }
    if args.ret_tail_transform == "power":
        expected_pairs["--ret-tail-gamma"] = str(args.ret_tail_gamma)
    if args.ret_tail_transform == "exp_norm":
        expected_pairs["--ret-tail-beta"] = str(args.ret_tail_beta)

    for flag, expected in expected_pairs.items():
        actual = command_value(command, flag)
        add_check(
            checks,
            f"command {flag}",
            actual == expected,
            f"expected {expected!r}, got {actual!r}",
        )

    for flag in ["--stochastic-pt", "--include-static-grid", "--no-eval-ai-on-garrido-cfis"]:
        add_check(checks, f"command has {flag}", flag in command, "required flag")

    add_check(
        checks,
        "command uses historical stochastic PT spread",
        command_value(command, "--stochastic-pt-spread") == "1.0",
        f"spread={command_value(command, '--stochastic-pt-spread')!r}",
    )
    add_check(
        checks,
        "command normalizes reward for PPO scale",
        ("--norm-reward" in command and "--no-norm-reward" not in command),
        "VecNormalize reward normalization must be explicit for serious scouts",
    )
    add_check(
        checks,
        "command does not use continuous action extension",
        "continuous_it_s" not in command,
        "Track A must keep Garrido variables before continuous_it_s/Track B",
    )
    return checks


def validate_env(args: argparse.Namespace) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    env = make_dkana_thesis_faithful_env(
        risk_level=args.risk_level,
        risk_occurrence_mode="thesis_periodic",
        raw_material_flow_mode="kit_equivalent_order_up_to",
        raw_material_order_up_to_multiplier=2.0,
        stochastic_pt=True,
        stochastic_pt_spread=1.0,
        reward_mode="ReT_tail_v1",
        action_space_mode="thesis_factorized",
        inventory_period_mode="thesis_strict",
        observation_version="v5",
        observation_mode="env_sdm_history_reward",
        ret_tail_w_sc=args.ret_tail_w_sc,
        ret_tail_w_rc=args.ret_tail_w_rc,
        ret_tail_w_ce=args.ret_tail_w_ce,
        ret_tail_cap_kappa=args.ret_tail_cap_kappa,
        ret_tail_inv_kappa=args.ret_tail_inv_kappa,
        ret_tail_boost=args.ret_tail_boost,
        ret_tail_transform=args.ret_tail_transform,
        ret_tail_gamma=args.ret_tail_gamma,
        ret_tail_beta=args.ret_tail_beta,
    )
    try:
        env.reset(seed=args.seed)
        _, reward, _, _, info = env.step(env.action_space.sample())
    finally:
        env.close()

    add_check(
        checks,
        "env reward bounded",
        0.0 < float(reward) <= 1.0,
        f"reward={float(reward):.6f}",
    )
    for key in [
        "ret_tail_step",
        "ret_tail_base_step",
        "ret_tail_service_continuity",
        "ret_tail_recovery_containment",
        "ret_tail_cost_efficiency",
        "ret_tail_stress",
    ]:
        add_check(checks, f"env info has {key}", key in info, "required metadata")
    add_check(
        checks,
        "env transform propagated",
        info.get("ret_tail_transform") == args.ret_tail_transform,
        f"info transform={info.get('ret_tail_transform')!r}",
    )
    return checks


def validate_reward_surface(path: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    df = pd.read_csv(path)
    ret_tail = df[df["reward"].astype(str).str.startswith("ReT_tail_v1")]
    add_check(
        checks,
        "reward surface has ReT_tail_v1 rows",
        not ret_tail.empty,
        str(path),
    )
    if ret_tail.empty:
        return checks
    for _, row in ret_tail.iterrows():
        label = f"{row['reward']}:{row.get('profile', '')}"
        add_check(
            checks,
            f"surface PASS {label}",
            bool(row.get("PASS", False)),
            f"best_by_reward={row.get('best_by_reward')} p10_rank={row.get('best_p10_rank')}",
        )
        if "rho_ret_p10_all" in row:
            add_check(
                checks,
                f"surface ret_p10 correlation {label}",
                float(row["rho_ret_p10_all"]) > 0.0,
                f"rho={row['rho_ret_p10_all']}",
            )
        if "rho_flow_fill_rate" in row:
            add_check(
                checks,
                f"surface flow_fill correlation {label}",
                float(row["rho_flow_fill_rate"]) > 0.0,
                f"rho={row['rho_flow_fill_rate']}",
            )
        if "rho_stockout_week_pct" in row:
            add_check(
                checks,
                f"surface stockout anti-correlation {label}",
                float(row["rho_stockout_week_pct"]) < 0.0,
                f"rho={row['rho_stockout_week_pct']}",
            )
    return checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPO / "outputs/benchmarks/track_a_preflight")
    parser.add_argument("--algo", choices=["ppo_mlp", "recurrent_ppo", "dmlpa_ppo"], default="ppo_mlp")
    parser.add_argument("--risk-level", default="increased")
    parser.add_argument("--panel-cfis", default="31-90")
    parser.add_argument("--train-timesteps", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed-base", type=int, default=900_000)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--history-window", type=int, default=30)
    parser.add_argument("--norm-reward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ret-tail-w-sc", type=float, default=0.30)
    parser.add_argument("--ret-tail-w-rc", type=float, default=0.60)
    parser.add_argument("--ret-tail-w-ce", type=float, default=0.10)
    parser.add_argument("--ret-tail-cap-kappa", type=float, default=0.40)
    parser.add_argument("--ret-tail-inv-kappa", type=float, default=0.25)
    parser.add_argument("--ret-tail-boost", type=float, default=0.0)
    parser.add_argument("--ret-tail-transform", choices=["identity", "power", "exp_norm"], default="power")
    parser.add_argument("--ret-tail-gamma", type=float, default=1.25)
    parser.add_argument("--ret-tail-beta", type=float, default=2.0)
    parser.add_argument(
        "--reward-surface-summary",
        type=Path,
        default=None,
        help="Optional reward_surface_summary.csv to enforce static reward gate.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    command = build_probe_command(args)
    checks = validate_command(command, args)
    checks.extend(validate_env(args))
    if args.reward_surface_summary is not None:
        checks.extend(validate_reward_surface(args.reward_surface_summary))

    args.output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "pass": all(item["pass"] for item in checks),
        "command": command,
        "checks": checks,
    }
    report_path = args.output_root / "track_a_preflight_latest.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for item in checks:
        mark = "PASS" if item["pass"] else "FAIL"
        print(f"[{mark}] {item['check']}: {item['detail']}")
    print("\nKaggle/scout command:")
    print(" ".join(command))
    print(f"\nSaved {report_path}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
