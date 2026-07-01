#!/usr/bin/env python3
"""Risk-regime headroom gate: does toggling R14 + R24/R22 create a real frontier?

Regime A ("manufacturing rescue"): R14=OFF, R24=SEVERE+high surge, R11=INCREASED
  → S3 is GOOD (no defect penalty, production rescues from demand surges)
  → Buffer is LESS valuable (can produce through disruptions)

Regime B ("distribution crisis"): R14=ON, R22=SEVERE, R23=SEVERE
  → S1 is BETTER (R14 penalty, downstream blocks make production useless)
  → Op9 buffer is CRITICAL (protects against LOC/CSSU destruction)

If the optimal policy CHANGES between regimes, headroom exists.
Dynamic policy that switches per regime > any single constant.
"""
from __future__ import annotations

import csv, json, sys, time
from pathlib import Path
from statistics import fmean
from typing import Any
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.supply_chain import MFSCSimulation
from supply_chain.config import (
    INVENTORY_BUFFERS, THESIS_FAITHFUL_PROTOCOL as P,
    THESIS_REPLICATION_DOWNSTREAM_Q_SOURCE as DQ,
)

REGIMES = {
    "buffer_king": {
        "risk_level": "current",
        "enabled_risks": ("R14", "R22", "R23"),
        "risk_overrides": {
            "R14": "increased",  # p=8%
            "R22": "increased",  # b=1344 → ~6/year
            "R23": "increased",  # b=1344 → ~6/year
        },
        "risk_frequency_multiplier": 1.5,   # modest
        "risk_impact_multiplier": 1.5,      # modest
        "horizon_hours": 80_640.0,
        "note": "Downstream disrupted, moderate. R14 active. Op9 buffer most valuable, S1 best.",
    },
    "speed_demon": {
        "risk_level": "current",
        "enabled_risks": ("R24",),
        "risk_overrides": {},
        "risk_frequency_multiplier": 6.0,   # R24 every ~56h
        "risk_impact_multiplier": 5.0,      # surge = 12000-13000
        "horizon_hours": 80_640.0,
        "note": "Massive demand surges. NO R14 → S3 is free. S3 + LOW buffer optimal. Produce like crazy.",
    },
    "calm_seas": {
        "risk_level": "current",
        "enabled_risks": ("R14",),
        "risk_overrides": {"R14": "current"},  # p=3%
        "risk_frequency_multiplier": 1.0,
        "risk_impact_multiplier": 1.0,
        "horizon_hours": 80_640.0,
        "note": "Calm. R14 only. S1 adequate. No buffer needed.",
    },
}

BUFFER_PERIODS = [0, 168, 336, 504, 672, 1344]
SHIFTS = [1, 2, 3]


def build_sim(regime_name: str, period: int, shifts: int, seed: int) -> MFSCSimulation:
    r = REGIMES[regime_name]
    bufs = dict(INVENTORY_BUFFERS[period]) if period else None
    enabled = set(r["enabled_risks"])
    overrides = {k: v for k, v in r["risk_overrides"].items() if k in enabled}
    return MFSCSimulation(
        shifts=shifts, initial_buffers=bufs, seed=seed, horizon=r["horizon_hours"],
        risks_enabled=True, risk_level=r["risk_level"],
        enabled_risks=enabled, risk_overrides=overrides,
        risk_occurrence_mode=P["risk_occurrence_mode"],
        risk_frequency_multiplier=r["risk_frequency_multiplier"],
        risk_impact_multiplier=r["risk_impact_multiplier"],
        year_basis=P["year_basis"], warmup_trigger=P["warmup_trigger"],
        r14_defect_mode=P["r14_defect_mode"], downstream_q_source=DQ,
        raw_material_flow_mode=P["raw_material_flow_mode"],
        raw_material_order_up_to_multiplier=P["raw_material_order_up_to_multiplier"],
        demand_on_hand_fulfillment_delay=P["demand_on_hand_fulfillment_delay"],
    )


def run_sim(sim: MFSCSimulation) -> dict:
    sim.run()
    served = [o for o in sim.orders
              if not getattr(o, "metrics_excluded", False)
              and getattr(o, "OATj", None) is not None
              and not getattr(o, "lost", False)]
    lost = sum(1 for o in sim.orders if getattr(o, "lost", False))
    ctj = [float(o.CTj) for o in served if o.CTj is not None]
    if not ctj:
        return {"mean_ret": 0.0, "fill_rate": 0.0, "lost": float(lost), "ct_p99": 0.0}
    ret = sim.compute_order_level_ret()
    return {
        "mean_ret": float(ret["mean_ret"]),
        "fill_rate": float(ret["fill_rate_order_level"]),
        "lost": float(lost),
        "ct_p99": float(np.percentile(np.asarray(ctj), 99)),
    }


def main():
    seeds = [1, 2, 3]
    out = Path("outputs/experiments/risk_regime_headroom_2026-06-29")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    started = time.time()
    total = len(REGIMES) * len(BUFFER_PERIODS) * len(SHIFTS) * len(seeds)
    n = 0
    for rname, rspec in REGIMES.items():
        for period in BUFFER_PERIODS:
            for s in SHIFTS:
                for seed in seeds:
                    n += 1
                    label = f"{rname}_I{period}_S{s}_seed{seed}"
                    sim = build_sim(rname, period, s, seed)
                    metrics = run_sim(sim)
                    rows.append({"regime": rname, "period": period, "shift": s,
                                 "seed": seed, **{k: float(v) for k, v in metrics.items()}})
                    if n % 10 == 0:
                        print(f"  {n}/{total}...", end=" ", flush=True)
    elapsed = time.time() - started

    with (out / "cells.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Per-regime best policy
    regime_best = {}
    regime_max_ret = {}
    for rname in REGIMES:
        rrows = [r for r in rows if r["regime"] == rname]
        grid = {}
        for r in rrows:
            k = (r["period"], r["shift"])
            grid.setdefault(k, []).append(r["mean_ret"])
        best_cell = max(grid, key=lambda k: fmean(grid[k]))
        regime_best[rname] = {
            "period": best_cell[0], "shift": best_cell[1],
            "mean_ret": fmean(grid[best_cell]),
        }
        regime_max_ret[rname] = fmean(grid[best_cell])

    # Oracle = best per regime, averaged (raw)
    oracle_ret = fmean(v["mean_ret"] for v in regime_best.values())

    # Per-regime normalized score for each policy
    normalized_scores = {}  # { (period, shift): avg_normalized_score }
    for r in rows:
        k = (r["period"], r["shift"])
        rname = r["regime"]
        norm_score = r["mean_ret"] / max(regime_max_ret[rname], 1e-9)
        normalized_scores.setdefault(k, []).append(norm_score)

    # Best constant = policy with highest avg normalized score
    best_norm_cell = max(normalized_scores, key=lambda k: fmean(normalized_scores[k]))
    best_norm_score = fmean(normalized_scores[best_norm_cell])
    oracle_norm = 1.0  # best per regime = 1.0 normalized
    norm_headroom = oracle_norm - best_norm_score

    # Also compute best constant for raw ReT (traditional)
    constant_grid = {}
    for r in rows:
        k = (r["period"], r["shift"])
        constant_grid.setdefault(k, []).append(r["mean_ret"])
    best_constant_cell = max(constant_grid, key=lambda k: fmean(constant_grid[k]))
    best_constant_ret = fmean(constant_grid[best_constant_cell])
    raw_headroom = oracle_ret - best_constant_ret

    headroom = norm_headroom  # use normalized as primary

    report = [
        "# Risk-Regime Headroom Gate",
        f"Wall: {elapsed:.0f}s | {len(REGIMES)} regimes × 6 periods × 3 shifts × {len(seeds)} seeds = {len(rows)} cells",
        "",
        "## Regimes",
    ]
    for rname, rspec in REGIMES.items():
        report.append(f"- **{rname}**: {rspec['note']}")
        report.append(f"  enabled={rspec['enabled_risks']} overrides={rspec['risk_overrides']} φ={rspec['risk_frequency_multiplier']} ψ={rspec['risk_impact_multiplier']}")

    report.extend([
        "",
        "## Best per Regime",
        "| Regime | Best Period | Best Shift | Mean ReT |",
        "|---|---:|---:|---:|",
    ])
    for rname, best in regime_best.items():
        report.append(f"| {rname} | {best['period']} | {best['shift']} | {best['mean_ret']:.5f} |")

    report.extend([
        "",
        f"**Oracle (best per regime, normalized):** 1.0000",
        f"**Best constant (normalized score):** I{best_norm_cell[0]}_S{best_norm_cell[1]} → {best_norm_score:.4f}",
        f"**NORMALIZED HEADROOM:** {headroom:+.4f} ({headroom*100:+.1f}% of oracle)",
        f"**Raw headroom:** {raw_headroom:+.5f} (oracle={oracle_ret:.5f} vs constant={best_constant_ret:.5f})",
        "",
    ])

    if headroom > 0.05:
        report.append(f"✅ HEADROOM CONFIRMED — oracle beats constant by {headroom:.4f} normalized. Dynamic policy has a real target.")
        report.append(f"   Promote to PPO training with these 3 regimes alternating within an episode.")
        verdict = "PROMOTE"
    elif headroom > 0.02:
        report.append(f"⚠️ MARGINAL HEADROOM — {headroom:.4f} may be exploitable. Consider amplifying regime divergence.")
        verdict = "MARGINAL"
    else:
        report.append(f"❌ NO HEADROOM — oracle ≈ best constant. Regime divergence insufficient even when normalized.")
        verdict = "NULL"

    rn = list(REGIMES.keys())
    report.extend([
        "",
        f"## Verdict: **{verdict}**",
        "",
        "## Interpretation",
        f"- If the best policy DIFFERS between regimes, a dynamic agent can beat the constant.",
        f"- Current divergence: {rn[0]} best=I{regime_best[rn[0]]['period']}_S{regime_best[rn[0]]['shift']} vs {rn[1]} best=I{regime_best[rn[1]]['period']}_S{regime_best[rn[1]]['shift']} vs {rn[2]} best=I{regime_best[rn[2]]['period']}_S{regime_best[rn[2]]['shift']}.",
        f"- Headroom = {headroom:.5f} = the max gain from switching between regime-specific optima vs using one compromise policy.",
    ])
    (out / "report.md").write_text("\n".join(report))
    (out / "gate.json").write_text(json.dumps({
        "regimes": list(REGIMES.keys()),
        "regime_best": regime_best,
        "oracle_ret_raw": oracle_ret,
        "best_constant_raw": best_constant_ret,
        "raw_headroom": raw_headroom,
        "oracle_norm": 1.0,
        "best_constant_norm_cell": list(best_norm_cell),
        "best_constant_norm_score": best_norm_score,
        "norm_headroom": headroom,
        "verdict": verdict,
    }, indent=2))

    print(f"\n{'='*60}")
    for rname, best in regime_best.items():
        print(f"  {rname:20s}: best=I{best['period']}_S{best['shift']} ret={best['mean_ret']:.5f}")
    print(f"  {'ORACLE':20s}: {oracle_ret:.5f}")
    print(f"  {'BEST CONSTANT':20s}: I{best_constant_cell[0]}_S{best_constant_cell[1]} = {best_constant_ret:.5f}")
    print(f"  {'HEADROOM':20s}: {headroom:+.5f} → {verdict}")
    print(f"WROTE {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
