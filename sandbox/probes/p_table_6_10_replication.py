"""Replication probe: Table 6.10 (Garrido-Rios 2017).

Thesis Table 6.10 compares 8 years of historical Colombian-army deliveries (Pt)
against 8 years of simulation output (ECS), each ECS = avg of 3 stochastic runs.

Thesis values:
    Pt  = [711808, 901131, 806454, 719344, 731016, 629429, 707203, 728878]
    ECS = [725021, 773675, 735389, 771434, 888776, 712315, 732883, 801239]
    RMSE(Pt, ECS) = 87,918

This probe reproduces the comparison under the thesis-faithful configuration
(post-fix defaults: thesis_periodic + op9_arrival). We compute:
  1. Mean annual delivery across N seeds (analogous to "ECS avg of 3 runs")
  2. Per-year delivery (years 1..8 post-warmup), averaged across seeds
  3. RMSE of our per-year ECS vs thesis Pt
  4. RMSE of our per-year ECS vs thesis ECS
  5. Mean comparisons

We also re-compute the audit doc's claimed number (deterministic_baseline=True,
risks_enabled=False) to confirm what that config actually measures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from supply_chain.config import (
    HOURS_PER_YEAR_THESIS,
    SIMULATION_HORIZON,
    VALIDATION_TABLE_6_10,
)
from supply_chain.supply_chain import MFSCSimulation

# Thesis Table 6.10 (also in config.py VALIDATION_TABLE_6_10, duplicated for clarity)
PT = np.array(VALIDATION_TABLE_6_10["Pt_observed"], dtype=float)
THESIS_ECS = np.array(VALIDATION_TABLE_6_10["ECS_simulated"], dtype=float)
THESIS_RMSE = float(VALIDATION_TABLE_6_10["RMSE"])
N_YEARS = len(PT)  # 8
SEEDS = [101, 202, 303]  # 3 seeds, as thesis "avg of 3 runs"


def get_yearly_delivery(sim: MFSCSimulation, n_years: int) -> np.ndarray:
    """Return post-warmup yearly deliveries for years 1..n."""
    t = sim.get_annual_throughput(start_time=sim.warmup_time)
    by_year = t["produced_by_year"]
    # Year keys may be 1..20 or 0..19 — normalize
    out = []
    for k in sorted(by_year.keys()):
        if len(out) >= n_years:
            break
        out.append(float(by_year[k]))
    if len(out) < n_years:
        # Pad with last value (shouldn't happen for 20-year horizon)
        out.extend([out[-1]] * (n_years - len(out)))
    return np.array(out[:n_years])


def run_stochastic_thesis_faithful(seed: int) -> np.ndarray:
    """Run thesis-faithful stochastic sim; return 8-year delivery array."""
    sim = MFSCSimulation(
        shifts=1,
        risks_enabled=True,
        risk_level="current",
        seed=seed,
        horizon=SIMULATION_HORIZON,
        year_basis="thesis",
        # Post-fix defaults already set these, but pass explicitly for clarity:
        risk_occurrence_mode="thesis_periodic",
        warmup_trigger="op9_arrival",
    ).run()
    return get_yearly_delivery(sim, N_YEARS)


def run_deterministic_no_risks(seed: int) -> np.ndarray:
    """Run the AUDIT-DOC config: deterministic_baseline=True, risks_enabled=False.

    This is what the audit doc used to compute its 'RMSE 62,055' claim.
    """
    sim = MFSCSimulation(
        shifts=1,
        risks_enabled=False,
        seed=seed,
        horizon=SIMULATION_HORIZON,
        year_basis="thesis",
        deterministic_baseline=True,
        # Inherit post-fix defaults for warmup_trigger and risk_occurrence_mode
        # (though risks_enabled=False makes the latter irrelevant).
    ).run()
    return get_yearly_delivery(sim, N_YEARS)


def report(label: str, ecs: np.ndarray) -> dict:
    """Compute & print comparison metrics for an 8-year ECS array."""
    rmse_vs_pt = float(np.sqrt(np.mean((ecs - PT) ** 2)))
    rmse_vs_thesis_ecs = float(np.sqrt(np.mean((ecs - THESIS_ECS) ** 2)))
    mean_ours = float(ecs.mean())
    mean_thesis_ecs = float(THESIS_ECS.mean())
    mean_pt = float(PT.mean())

    print(f"\n=== {label} ===")
    print(f"  Per-year delivery: {[f'{x:,.0f}' for x in ecs]}")
    print(f"  Mean (ours):       {mean_ours:>12,.0f}")
    print(f"  Mean (thesis ECS): {mean_thesis_ecs:>12,.0f}  Δ = {(mean_ours - mean_thesis_ecs) / mean_thesis_ecs:+.2%}")
    print(f"  Mean (Pt):         {mean_pt:>12,.0f}  Δ = {(mean_ours - mean_pt) / mean_pt:+.2%}")
    print(f"  RMSE vs Pt:        {rmse_vs_pt:>12,.0f}")
    print(f"  RMSE vs thesis ECS:{rmse_vs_thesis_ecs:>12,.0f}")
    print(f"  Thesis RMSE (Pt vs ECS): {THESIS_RMSE:,.0f}")
    if THESIS_RMSE > 0:
        ratio = rmse_vs_pt / THESIS_RMSE
        print(f"  RMSE ratio (ours / thesis): {ratio:.2f}x")

    return {
        "label": label,
        "ecs": ecs.tolist(),
        "mean": mean_ours,
        "rmse_vs_pt": rmse_vs_pt,
        "rmse_vs_thesis_ecs": rmse_vs_thesis_ecs,
    }


def main() -> None:
    print(f"Thesis Table 6.10 — {N_YEARS}-year replication")
    print(f"Seeds: {SEEDS}")
    print(f"Pt (historical):    {[f'{x:,.0f}' for x in PT]}")
    print(f"Thesis ECS:         {[f'{x:,.0f}' for x in THESIS_ECS]}")
    print(f"Thesis RMSE:        {THESIS_RMSE:,.0f}")

    out = {"seeds": SEEDS, "n_years": N_YEARS, "thesis": {"Pt": PT.tolist(), "ECS": THESIS_ECS.tolist(), "RMSE": THESIS_RMSE}}

    # ----- Stochastic thesis-faithful -----
    print("\nRunning stochastic thesis-faithful (this takes ~60-90s)...")
    sto_runs = np.array([run_stochastic_thesis_faithful(s) for s in SEEDS])
    sto_mean_per_year = sto_runs.mean(axis=0)
    out["stochastic_thesis_faithful"] = {
        "per_seed": sto_runs.tolist(),
        "mean_per_year": sto_mean_per_year.tolist(),
        "report": report("Stochastic thesis-faithful (mean of 3 seeds)", sto_mean_per_year),
    }

    # ----- Deterministic no-risks (audit-doc config) -----
    print("\nRunning deterministic no-risks (audit-doc config)...")
    # Single seed (deterministic doesn't depend on seed)
    det_run = run_deterministic_no_risks(SEEDS[0])
    out["deterministic_no_risks_audit_config"] = {
        "per_year": det_run.tolist(),
        "report": report("Deterministic no-risks (audit-doc config)", det_run),
    }

    out_path = Path(__file__).resolve().parents[1] / "results" / "p_table_6_10_replication.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
