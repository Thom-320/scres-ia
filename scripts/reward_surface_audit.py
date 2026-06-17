#!/usr/bin/env python3
"""Reward-surface audit: does a training reward rank policies the way the TAIL-resilience
metrics do, or does it just track the mean?

Motivation (2026-06-17): we discovered we were training PPO on `ReT_ladder_v1`, whose
reward_total correlates with the MEAN ReT (~0.86) but barely with the tail `ret_p10` (~0.24);
its argmax (I504_S3) has a FLOORED tail. So PPO was optimizing the wrong objective and every
"RL loses on the tail" was structurally doomed. Codex's acceptance rule: *if the best policy by
reward has a bad tail, that reward must not be used for training.*

This script evaluates the STATIC thesis policy grid (no SB3 / no training needed) under each
candidate reward on the thesis-faithful backbone (m2.0), then reports, per reward:
  - Spearman rank-corr(reward_total, each metric): ret_p10, flow_fill, stockout(-), ret_mean
  - best-by-reward policy and its rank on ret_p10 (1 = best tail)
  - PASS/FAIL: best-by-reward is in the top-K by ret_p10 AND its p10 is not floored.

A reward PASSES only if optimizing it also moves you toward the tail optimum. Use a PASSing
reward to train; reject FAILing rewards.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
AUDITOR = REPO / "scripts" / "audit_garrido_metric_saturation.py"

# Metrics where HIGHER is better tail resilience; stockout is inverted (lower better).
TAIL_HIGHER = ["ret_p10_all", "flow_fill_rate", "ret_mean_all_orders_zero_unfulfilled"]
TAIL_LOWER = ["stockout_week_pct"]
DEFAULT_REWARDS = [
    "ReT_ladder_v1",
    "ReT_tail_v1",
    "ReT_seq_v1",
    "ReT_garrido2024",
    "ReT_unified_v1",
    "rt_v0",
    "control_v1",
]
DEFAULT_RET_TAIL_TRANSFORM_GRID = "identity,power:1.25,power:1.5,power:2.0,exp_norm:2,exp_norm:4"

RET_TAIL_FLAGS = [
    ("--ret-tail-w-sc", "ret_tail_w_sc"),
    ("--ret-tail-w-rc", "ret_tail_w_rc"),
    ("--ret-tail-w-ce", "ret_tail_w_ce"),
    ("--ret-tail-cap-kappa", "ret_tail_cap_kappa"),
    ("--ret-tail-inv-kappa", "ret_tail_inv_kappa"),
    ("--ret-tail-boost", "ret_tail_boost"),
]


@dataclass(frozen=True)
class RewardSpec:
    reward_mode: str
    label: str
    ret_tail_transform: str | None = None
    ret_tail_gamma: float | None = None
    ret_tail_beta: float | None = None


def slug_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def parse_ret_tail_transform_spec(spec: str) -> tuple[str, float, float]:
    raw = spec.strip()
    if raw == "identity":
        return "identity", 1.0, 2.0
    if ":" not in raw:
        raise ValueError(
            f"Invalid ret-tail transform spec {raw!r}; expected identity, "
            "power:<gamma>, or exp_norm:<beta>"
        )
    kind, value = raw.split(":", 1)
    number = float(value)
    if kind == "power":
        if number <= 0.0:
            raise ValueError("power gamma must be positive")
        return "power", number, 2.0
    if kind == "exp_norm":
        if number <= 0.0:
            raise ValueError("exp_norm beta must be positive")
        return "exp_norm", 1.0, number
    raise ValueError(
        f"Invalid ret-tail transform kind {kind!r}; expected identity, power, or exp_norm"
    )


def ret_tail_label(transform: str, gamma: float, beta: float) -> str:
    if transform == "power":
        return f"ReT_tail_v1__power_g{slug_float(gamma)}"
    if transform == "exp_norm":
        return f"ReT_tail_v1__exp_norm_b{slug_float(beta)}"
    return "ReT_tail_v1__identity"


def iter_reward_specs(args: argparse.Namespace) -> list[RewardSpec]:
    rewards = [r.strip() for r in args.rewards.split(",") if r.strip()]
    transform_specs = [
        parse_ret_tail_transform_spec(item)
        for item in args.ret_tail_transform_grid.split(",")
        if item.strip()
    ]
    specs: list[RewardSpec] = []
    for reward in rewards:
        if reward != "ReT_tail_v1":
            specs.append(RewardSpec(reward_mode=reward, label=reward))
            continue
        for transform, gamma, beta in transform_specs:
            specs.append(
                RewardSpec(
                    reward_mode=reward,
                    label=ret_tail_label(transform, gamma, beta),
                    ret_tail_transform=transform,
                    ret_tail_gamma=gamma,
                    ret_tail_beta=beta,
                )
            )
    return specs


def build_auditor_command(
    reward: str | RewardSpec, out_root: Path, args: argparse.Namespace
) -> list[str]:
    spec = (
        reward
        if isinstance(reward, RewardSpec)
        else RewardSpec(
            reward_mode=reward,
            label=(
                ret_tail_label(
                    args.ret_tail_transform,
                    args.ret_tail_gamma,
                    args.ret_tail_beta,
                )
                if reward == "ReT_tail_v1"
                else reward
            ),
            ret_tail_transform=args.ret_tail_transform
            if reward == "ReT_tail_v1"
            else None,
            ret_tail_gamma=args.ret_tail_gamma if reward == "ReT_tail_v1" else None,
            ret_tail_beta=args.ret_tail_beta if reward == "ReT_tail_v1" else None,
        )
    )
    label = f"rwd_{spec.label}"
    cmd = [
        sys.executable, str(AUDITOR),
        "--profiles", args.profiles,
        "--panel-cfis", args.panel_cfis,
        "--policy-set", args.policy_set,
        "--replications", str(args.replications),
        "--raw-material-flow-mode", "kit_equivalent_order_up_to",
        "--raw-material-order-up-to-multiplier", str(args.multiplier),
        "--risk-occurrence-mode", "thesis_periodic",
        "--reward-mode", spec.reward_mode,
        "--label", label,
        "--output-root", str(out_root),
    ]
    for cli_flag, attr in RET_TAIL_FLAGS:
        value = getattr(args, attr, None)
        if value is not None:
            cmd.extend([cli_flag, str(value)])
    if spec.reward_mode == "ReT_tail_v1":
        cmd.extend(
            [
                "--ret-tail-transform",
                str(spec.ret_tail_transform or "identity"),
                "--ret-tail-gamma",
                str(spec.ret_tail_gamma if spec.ret_tail_gamma is not None else 1.0),
                "--ret-tail-beta",
                str(spec.ret_tail_beta if spec.ret_tail_beta is not None else 2.0),
            ]
        )
    return cmd


def run_auditor(spec: RewardSpec, out_root: Path, args: argparse.Namespace) -> Path:
    label = f"rwd_{spec.label}"
    csv = out_root / label / "episode_metric_audit.csv"
    if csv.exists() and csv.stat().st_size > 0 and not args.force:
        print(f"[skip] {spec.label} (cached)")
        return csv
    cmd = build_auditor_command(spec, out_root, args)
    print(f"[run ] {spec.label} ...", flush=True)
    env_extra = {"KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1"}
    import os
    proc = subprocess.run(cmd, cwd=str(REPO), env={**os.environ, **env_extra},
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0 or not csv.exists():
        print(proc.stdout[-1500:])
        raise RuntimeError(f"auditor failed for reward={spec.label}")
    return csv


def spearman(a: pd.Series, b: pd.Series) -> float:
    if a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(a.rank().corr(b.rank()))


def audit_reward(csv: Path, spec: RewardSpec, profile: str, top_k: int,
                 p10_floor: float) -> dict:
    df = pd.read_csv(csv)
    df = df[df["profile"] == profile]
    g = df.groupby("policy").mean(numeric_only=True).reset_index()
    if g.empty or "reward_total" not in g:
        return {}
    row = {
        "reward": spec.label,
        "reward_mode": spec.reward_mode,
        "ret_tail_transform": spec.ret_tail_transform or "",
        "ret_tail_gamma": spec.ret_tail_gamma,
        "ret_tail_beta": spec.ret_tail_beta,
        "profile": profile,
        "n_policy": len(g),
    }
    for m in TAIL_HIGHER:
        if m in g:
            row[f"rho_{m}"] = round(spearman(g["reward_total"], g[m]), 3)
    for m in TAIL_LOWER:
        if m in g:
            # invert: good reward should anti-correlate with stockout
            row[f"rho_{m}"] = round(spearman(g["reward_total"], g[m]), 3)
    # best by reward and its tail rank
    best = g.loc[g["reward_total"].idxmax()]
    g_sorted = g.sort_values("ret_p10_all", ascending=False).reset_index(drop=True)
    best_p10 = float(best["ret_p10_all"])
    top_p10 = float(g["ret_p10_all"].max())
    p10_rank = int((g["ret_p10_all"] > best_p10 + 1e-9).sum()) + 1
    row["best_by_reward"] = best["policy"]
    row["best_reward_p10"] = round(best_p10, 4)
    row["best_p10_rank"] = p10_rank
    row["top_p10_policy"] = g_sorted.iloc[0]["policy"]
    row["top_p10"] = round(top_p10, 4)
    row["PASS"] = bool(p10_rank <= top_k and best_p10 >= p10_floor * max(top_p10, 1e-9))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rewards", default=",".join(DEFAULT_REWARDS))
    ap.add_argument("--profiles", default="increased,severe")
    ap.add_argument("--panel-cfis", default="31-90")
    ap.add_argument("--policy-set", default="with_crossed",
                    help=(
                        "Use with_crossed: it spans the FULL I x S grid incl. high-inventory "
                        "x high-shift combos (e.g. I504_S3). thesis_static only varies "
                        "inventory at S1 and gives FALSE PASSes because it never tests "
                        "whether the reward over-rewards shifts at the cost of the tail."
                    ))
    ap.add_argument("--multiplier", type=float, default=2.0,
                    help="2.0 = thesis-faithful backbone (Table 6.10).")
    ap.add_argument("--replications", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=3,
                    help="best-by-reward must rank within top-K by ret_p10 to PASS.")
    ap.add_argument("--p10-floor", type=float, default=0.5,
                    help="and its p10 must be >= this fraction of the best p10.")
    ap.add_argument("--ret-tail-w-sc", type=float, default=None)
    ap.add_argument("--ret-tail-w-rc", type=float, default=None)
    ap.add_argument("--ret-tail-w-ce", type=float, default=None)
    ap.add_argument("--ret-tail-cap-kappa", type=float, default=None)
    ap.add_argument("--ret-tail-inv-kappa", type=float, default=None)
    ap.add_argument("--ret-tail-boost", type=float, default=None)
    ap.add_argument(
        "--ret-tail-transform",
        choices=["identity", "power", "exp_norm"],
        default="identity",
        help=(
            "Single ReT_tail_v1 transform used when build_auditor_command is called directly."
        ),
    )
    ap.add_argument("--ret-tail-gamma", type=float, default=1.0)
    ap.add_argument("--ret-tail-beta", type=float, default=2.0)
    ap.add_argument(
        "--ret-tail-transform-grid",
        default=DEFAULT_RET_TAIL_TRANSFORM_GRID,
        help=(
            "Comma-separated ReT_tail_v1 transform grid. Examples: "
            "identity,power:1.25,power:1.5,exp_norm:2. Applies only when "
            "ReT_tail_v1 is included in --rewards."
        ),
    )
    ap.add_argument("--output-root", type=Path,
                    default=REPO / "outputs/benchmarks/reward_surface_audit")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in iter_reward_specs(args):
        try:
            csv = run_auditor(spec, args.output_root, args)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {spec.label}: {exc}")
            continue
        for profile in args.profiles.split(","):
            r = audit_reward(csv, spec, profile.strip(), args.top_k, args.p10_floor)
            if r:
                rows.append(r)

    if not rows:
        print("No rows produced.")
        return 1
    out = pd.DataFrame(rows)
    cols = ["reward", "reward_mode", "ret_tail_transform", "ret_tail_gamma",
            "ret_tail_beta", "profile", "rho_ret_p10_all", "rho_flow_fill_rate",
            "rho_stockout_week_pct", "rho_ret_mean_all_orders_zero_unfulfilled",
            "best_by_reward", "best_reward_p10", "best_p10_rank",
            "top_p10_policy", "top_p10", "PASS"]
    cols = [c for c in cols if c in out.columns]
    summary_path = args.output_root / "reward_surface_summary.csv"
    out.to_csv(summary_path, index=False)
    pd.set_option("display.width", 240)
    print("\n" + "=" * 100)
    print("REWARD-SURFACE AUDIT  (PASS = best-by-reward also good on the tail ret_p10)")
    print("=" * 100)
    print(out[cols].to_string(index=False))
    print(f"\nSaved {summary_path}")
    print("\nReading: rho_ret_p10 near +1 = reward tracks the tail (good); near 0 or negative = "
          "reward ignores/opposes the tail.\nA reward with PASS=False must NOT be used to train "
          "PPO/Recurrent — its optimum has a bad tail.")
    passing = sorted({r["reward"] for r in rows if r.get("PASS")})
    print(
        "\nPASSING rewards (usable for training): "
        f"{passing or 'NONE — need a new tail-aligned reward (ReT_tail_v1)'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
