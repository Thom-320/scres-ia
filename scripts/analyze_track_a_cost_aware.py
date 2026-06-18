#!/usr/bin/env python3
"""Judge a Track A run on the COST-AWARE Garrido 2024 index (not just ret_p10).

ret_p10 is a pure tail/service metric: more buffer never hurts it, it plateaus, so a
fixed max-ish static is unbeatable and there is no temporal structure for an adaptive
policy to exploit. The Garrido 2024 cost-aware index (ret_garrido2024_sigmoid, surfaced
per policy in policy_summary.csv) has an INTERIOR optimum — inventory helps via zeta but
its holding/production/backlog cost penalizes via kappa. On that index an adaptive
policy that times the buffer (high before predictable disruptions, low in calm) can beat
the best FIXED static. This is the L_{t-1} payoff and it is Garrido 2024's own index.

Self-contained: reads every <run>/**/policy_summary.csv, pairs the learned (ppo) policy
against the best static (per metric) within the same run, and applies the cost-aware
promotion rule. Does not touch Codex's sweep_summary/analyzer chain.

Promotion rule (declared): a config PROMOTES if the learned policy beats the best static
on the cost-aware index `ret_garrido2024_sigmoid` by >= --primary-threshold WITHOUT
losing more than --max-fill-loss on raw service `flow_fill_rate` — i.e. better
cost-adjusted resilience via temporal adaptation, not by tanking service.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import pandas as pd

LEARNED_TOKEN = "ppo"
STATIC_PREFIX = "static_grid"
COST_AWARE = "ret_garrido2024_sigmoid_mean"
SERVICE = "flow_fill_rate_mean"
P10 = "ret_p10_all_mean"


def parse_algo_risk(run_name: str) -> tuple[str, str]:
    algo = (
        "dmlpa" if "dmlpa" in run_name else "recurrent" if "recurrent" in run_name else "mlp"
    )
    risk = "severe" if "severe" in run_name else "increased"
    return algo, risk


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="A sweep/kernel output dir to scan recursively.")
    ap.add_argument("--primary-threshold", type=float, default=0.02)
    ap.add_argument("--max-fill-loss", type=float, default=0.01)
    args = ap.parse_args()

    summaries = sorted(glob.glob(str(args.run_dir / "**" / "policy_summary.csv"), recursive=True))
    if not summaries:
        print(f"No policy_summary.csv under {args.run_dir}")
        return 1

    rows = []
    for path in summaries:
        df = pd.read_csv(path)
        if COST_AWARE not in df.columns:
            print(f"[skip] {path}: no cost-aware column (re-run with the cost-aware eval)")
            continue
        df["policy"] = df["policy"].astype(str)
        learned = df[df["policy"].str.contains(LEARNED_TOKEN)]
        grid = df[df["policy"].str.startswith(STATIC_PREFIX)]
        if learned.empty or grid.empty:
            continue
        L = learned.iloc[0]
        algo, risk = parse_algo_risk(os.path.basename(os.path.dirname(path)))
        # best static by the cost-aware index (the interior optimum)
        best = grid.loc[grid[COST_AWARE].astype(float).idxmax()]
        d_cost = float(L[COST_AWARE]) - float(best[COST_AWARE])
        d_fill = float(L[SERVICE]) - float(best[SERVICE])
        d_p10 = float(L[P10]) - float(best[P10]) if P10 in df.columns else float("nan")
        promote = d_cost >= args.primary_threshold and d_fill >= -args.max_fill_loss
        rows.append(
            {
                "algo": algo,
                "risk": risk,
                "best_static_costaware": best["policy"],
                "RL_costaware": round(float(L[COST_AWARE]), 4),
                "static_costaware": round(float(best[COST_AWARE]), 4),
                "delta_costaware": round(d_cost, 4),
                "delta_flow": round(d_fill, 4),
                "delta_p10": round(d_p10, 4),
                "RL_zeta": round(float(L.get("zeta_avg_mean", 0.0)), 0),
                "static_zeta": round(float(best.get("zeta_avg_mean", 0.0)), 0),
                "PROMOTE": promote,
            }
        )

    if not rows:
        print("No comparable (learned vs static) rows with cost-aware columns found.")
        return 1
    out = pd.DataFrame(rows).sort_values(["risk", "algo"])
    pd.set_option("display.width", 240)
    print("=" * 96)
    print("TRACK A COST-AWARE DECISION (Garrido 2024 index; PROMOTE = RL beats best static)")
    print("=" * 96)
    print(out.to_string(index=False))
    n_promote = int(out["PROMOTE"].sum())
    print(
        f"\nPromotion threshold: delta_costaware >= {args.primary_threshold} and "
        f"delta_flow >= {-args.max_fill_loss}"
    )
    if n_promote:
        print(f"\n>>> {n_promote} config(s) PROMOTE: RL adds cost-adjusted resilience via "
              "temporal adaptation. Candidate for a confirmatory run.")
    else:
        print("\n>>> NO config promotes on the cost-aware index either. RL does not beat the "
              "best fixed static even on cost-adjusted resilience; report Track A closed.")
    print("\nNote: RL_zeta vs static_zeta shows whether RL holds LESS inventory than the best "
          "static (the mechanism of a cost-aware win).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
