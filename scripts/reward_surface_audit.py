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

RET_TAIL_FLAGS = [
    ("--ret-tail-w-sc", "ret_tail_w_sc"),
    ("--ret-tail-w-rc", "ret_tail_w_rc"),
    ("--ret-tail-w-ce", "ret_tail_w_ce"),
    ("--ret-tail-cap-kappa", "ret_tail_cap_kappa"),
    ("--ret-tail-inv-kappa", "ret_tail_inv_kappa"),
    ("--ret-tail-boost", "ret_tail_boost"),
]


def build_auditor_command(
    reward: str, out_root: Path, args: argparse.Namespace
) -> list[str]:
    label = f"rwd_{reward}"
    cmd = [
        sys.executable, str(AUDITOR),
        "--profiles", args.profiles,
        "--policy-set", args.policy_set,
        "--replications", str(args.replications),
        "--raw-material-flow-mode", "kit_equivalent_order_up_to",
        "--raw-material-order-up-to-multiplier", str(args.multiplier),
        "--risk-occurrence-mode", "thesis_periodic",
        "--reward-mode", reward,
        "--label", label,
        "--output-root", str(out_root),
    ]
    for cli_flag, attr in RET_TAIL_FLAGS:
        value = getattr(args, attr, None)
        if value is not None:
            cmd.extend([cli_flag, str(value)])
    return cmd


def run_auditor(reward: str, out_root: Path, args: argparse.Namespace) -> Path:
    label = f"rwd_{reward}"
    csv = out_root / label / "episode_metric_audit.csv"
    if csv.exists() and csv.stat().st_size > 0 and not args.force:
        print(f"[skip] {reward} (cached)")
        return csv
    cmd = build_auditor_command(reward, out_root, args)
    print(f"[run ] {reward} ...", flush=True)
    env_extra = {"KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1"}
    import os
    proc = subprocess.run(cmd, cwd=str(REPO), env={**os.environ, **env_extra},
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0 or not csv.exists():
        print(proc.stdout[-1500:])
        raise RuntimeError(f"auditor failed for reward={reward}")
    return csv


def spearman(a: pd.Series, b: pd.Series) -> float:
    if a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(a.rank().corr(b.rank()))


def audit_reward(csv: Path, reward: str, profile: str, top_k: int,
                 p10_floor: float) -> dict:
    df = pd.read_csv(csv)
    df = df[df["profile"] == profile]
    g = df.groupby("policy").mean(numeric_only=True).reset_index()
    if g.empty or "reward_total" not in g:
        return {}
    row = {"reward": reward, "profile": profile, "n_policy": len(g)}
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
    ap.add_argument("--output-root", type=Path,
                    default=REPO / "outputs/benchmarks/reward_surface_audit")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rewards = [r.strip() for r in args.rewards.split(",") if r.strip()]
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for reward in rewards:
        try:
            csv = run_auditor(reward, args.output_root, args)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {reward}: {exc}")
            continue
        for profile in args.profiles.split(","):
            r = audit_reward(csv, reward, profile.strip(), args.top_k, args.p10_floor)
            if r:
                rows.append(r)

    if not rows:
        print("No rows produced.")
        return 1
    out = pd.DataFrame(rows)
    cols = ["reward", "profile", "rho_ret_p10_all", "rho_flow_fill_rate",
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
    print(f"\nPASSING rewards (usable for training): {passing or 'NONE — need a new tail-aligned reward (ReT_tail_v1)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
