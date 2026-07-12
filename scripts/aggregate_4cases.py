#!/usr/bin/env python3
"""Aggregator: PPO 4-case results vs static frontier (cd_sigmoid same-bar).

Reads each case's PPO summary (outputs/4cases/caseN/summary.csv) and the
corresponding static frontier panel from the war-sweep run
(outputs/experiments/cd_same_bar_risk_war_sweep_2026-06-27/panel_war_phi*.csv),
computes the static robust (best fixed by mean across regimes) and oracle
(per-regime best) on cd_sigmoid_mean, and reports:

  PPO_cd      = mean(PPO cd_sigmoid across seeds)
  robust_cd   = static robust (best fixed policy, mean across regimes)
  oracle_cd   = mean of per-regime best
  mean_gap    = oracle_cd - robust_cd  (honest dynamic headroom ceiling)
  captured    = (PPO_cd - robust_cd) / mean_gap  (fraction of headroom captured)

Outputs a unified table for all 4 cases.
"""
from __future__ import annotations
import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def ppo_rows(case_dir: Path):
    f = case_dir / "summary.csv"
    if not f.exists():
        return []
    return list(csv.DictReader(open(f)))


def static_frontier(panel_path: Path):
    """Return (robust, oracle_by_regime, mean_gap) for a war-sweep panel."""
    rows = list(csv.DictReader(open(panel_path)))
    policies = sorted({r["policy"] for r in rows})
    regimes = sorted({r["regime"] for r in rows})
    by = {(r["policy"], r["regime"]): float(r["cd_sigmoid_mean"]) for r in rows}
    def m(p):
        return statistics.mean(by[(p, rg)] for rg in regimes)
    robust = max(policies, key=m)
    oracle = {rg: max(policies, key=lambda p: by[(p, rg)]) for rg in regimes}
    gaps = {rg: by[(oracle[rg], rg)] - by[(robust, rg)] for rg in regimes}
    mean_gap = statistics.mean(gaps.values())
    robust_cd = m(robust)
    oracle_cd = statistics.mean(by[(oracle[rg], rg)] for rg in regimes)
    return {
        "robust_policy": robust, "robust_cd": robust_cd,
        "oracle_policies": oracle, "oracle_cd": oracle_cd,
        "mean_gap": mean_gap, "gaps_by_regime": {k: round(v, 4) for k, v in gaps.items()},
    }


# Case -> (phi, psi, env_label, reward, static_panel)
CASES = {
    "case1_faithful_resilience": dict(
        env="faithful (phi=1,psi=1)", reward="control_v1",
        panel="outputs/experiments/cd_same_bar_risk_war_sweep_2026-06-27/panel_war_phi1.0_psi1.0.csv",
        metric="resilience / service",
    ),
    "case2_faithful_cd": dict(
        env="faithful (phi=1,psi=1)", reward="ReT_garrido2024_raw",
        panel="outputs/experiments/cd_same_bar_risk_war_sweep_2026-06-27/panel_war_phi1.0_psi1.0.csv",
        metric="CD same-bar (Garrido 2024)",
    ),
    "case3_war_resilience": dict(
        env="war (phi=4,psi=1.5, max-headroom)", reward="ReT_cvar_cd",
        panel="outputs/experiments/cd_same_bar_risk_war_sweep_2026-06-27/panel_war_phi4.0_psi1.5.csv",
        metric="resilience / service",
    ),
    "case4_war_cd": dict(
        env="war (phi=4,psi=1.5, max-headroom)", reward="ReT_garrido2024_raw",
        panel="outputs/experiments/cd_same_bar_risk_war_sweep_2026-06-27/panel_war_phi4.0_psi1.5.csv",
        metric="CD same-bar (Garrido 2024)",
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ppo-root", default="outputs/4cases")
    ap.add_argument("--output", default="outputs/4cases/aggregated_results.csv")
    args = ap.parse_args()
    ppo_root = Path(args.ppo_root)

    print("=== PPO 4-case vs static frontier (cd_sigmoid same-bar) ===\n")
    print(f"{'case':28} {'env':32} {'reward':20} {'metric':22} "
          f"{'PPO_cd':>7} {'robust_cd':>9} {'oracle_cd':>9} {'gap':>7} {'capture%':>9} {'win?':>5}")
    rows_out = []
    for case, spec in CASES.items():
        case_dir = ppo_root / case
        ppos = ppo_rows(case_dir)
        panel = Path(spec["panel"])
        if not ppos or not panel.exists():
            print(f"{case:28} (missing: ppo={bool(ppos)} panel={panel.exists()})")
            continue
        sf = static_frontier(panel)
        ppo_cd = statistics.mean(float(r["cd_sigmoid_mean"]) for r in ppos)
        gap = sf["mean_gap"]
        # capture = how much of the headroom (oracle - robust) the PPO captured.
        # PPO_cd vs robust_cd: if >0, PPO beats the best fixed policy.
        delta_vs_robust = ppo_cd - sf["robust_cd"]
        capture_pct = (100.0 * delta_vs_robust / gap) if gap > 1e-9 else float("nan")
        win = "YES" if delta_vs_robust > 0 and gap > 0.001 else ("~" if delta_vs_robust > -0.001 else "no")
        print(f"{case:28} {spec['env']:32} {spec['reward']:20} {spec['metric']:22} "
              f"{ppo_cd:7.4f} {sf['robust_cd']:9.4f} {sf['oracle_cd']:9.4f} "
              f"{gap:7.4f} {capture_pct:8.1f}% {win:>5}")
        rows_out.append({
            "case": case, "env": spec["env"], "reward": spec["reward"],
            "metric": spec["metric"], "ppo_cd_sigmoid": ppo_cd,
            "static_robust_cd": sf["robust_cd"], "static_oracle_cd": sf["oracle_cd"],
            "static_mean_gap": gap, "ppo_minus_robust": delta_vs_robust,
            "headroom_capture_pct": capture_pct, "win": win,
            "robust_policy": sf["robust_policy"],
        })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows_out:
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
        print(f"\nWROTE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
