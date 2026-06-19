"""Empirical confirmation of forensic finding B1.

Verifies that the DEFAULT risk_occurrence_mode ('legacy_renewal') makes uniform
risks fire ~2x too often compared to Table 6.11 of the Garrido-Rios 2017 thesis,
and that 'thesis_periodic' mode matches Table 6.11.

Table 6.11 expected events per year (current level):
    R11 = 48        R21 = 1/2     R3  = 1/20
    R12 = 2 1/6     R22 = 2
    R13 = 58        R23 = 1
    R14 = 22,153    R24 = 12

We run short horizons (5 thesis-years = 40,320 h) and many seeds, then compare
events/year against Table 6.11.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from supply_chain.supply_chain import MFSCSimulation

# Table 6.11 events per year (Garrido-Rios 2017, current level of risk)
THESIS_EVENTS_PER_YEAR = {
    "R11": 48.0,
    "R12": 2.0 + 1.0 / 6.0,
    "R13": 58.0,
    "R14": 22153.0,
    "R21": 0.5,
    "R22": 2.0,
    "R23": 1.0,
    "R24": 12.0,
    "R3": 1.0 / 20.0,
}

HOURS_PER_YEAR_THESIS = 8064
HORIZON_HOURS = 5 * HOURS_PER_YEAR_THESIS  # 40,320 h ≈ 5 thesis-years
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]


def count_risk_events(sim: MFSCSimulation) -> dict[str, int]:
    """Count events per risk_id from sim.risk_events (or whatever the engine exposes)."""
    counts: dict[str, int] = {}
    events = getattr(sim, "risk_events", None)
    if events is None:
        # Fallbacks: try other plausible attribute names
        for attr in ("events", "disruption_log", "risk_log"):
            events = getattr(sim, attr, None)
            if events is not None:
                break
    if events is None:
        raise AttributeError(
            "Cannot find risk event log on MFSCSimulation. Inspect attributes."
        )
    for ev in events:
        rid = ev.get("risk_id") if isinstance(ev, dict) else getattr(ev, "risk_id", None)
        if rid is None and isinstance(ev, dict):
            rid = ev.get("id") or ev.get("type") or ev.get("risk")
        if rid is None:
            # Best effort: dump one event so the user can see its shape
            raise AttributeError(f"Cannot find risk_id field on event: {ev!r}")
        counts[rid] = counts.get(rid, 0) + 1
    return counts


def run_mode(mode: str) -> dict[str, dict]:
    """Run N seeds for the given risk_occurrence_mode; return per-risk stats."""
    per_seed: list[dict[str, int]] = []
    for seed in SEEDS:
        sim = MFSCSimulation(
            shifts=1,
            risks_enabled=True,
            risk_level="current",
            seed=seed,
            horizon=HORIZON_HOURS,
            year_basis="thesis",
            risk_occurrence_mode=mode,
            # Use op9_arrival so warm-up is thesis-faithful; irrelevant for
            # event-frequency counts but cleaner.
            warmup_trigger="op9_arrival",
        ).run()
        per_seed.append(count_risk_events(sim))

    by_risk: dict[str, dict] = {}
    all_risk_ids = set().union(*[set(c.keys()) for c in per_seed])
    for rid in sorted(all_risk_ids):
        per_year = np.array([c.get(rid, 0) for c in per_seed]) / 5.0  # 5-year horizon
        by_risk[rid] = {
            "mean_per_year": float(per_year.mean()),
            "std_per_year": float(per_year.std(ddof=1)) if len(per_year) > 1 else 0.0,
            "n_seeds": len(per_year),
        }
    return by_risk


def main() -> None:
    print(f"Horizon: {HORIZON_HOURS} h ({HORIZON_HOURS / HOURS_PER_YEAR_THESIS} thesis-years)")
    print(f"Seeds: {SEEDS}")
    print()

    out = {"horizon_hours": HORIZON_HOURS, "seeds": SEEDS, "modes": {}}

    for mode in ("legacy_renewal", "thesis_periodic"):
        print(f"=== risk_occurrence_mode = {mode} ===")
        stats = run_mode(mode)
        out["modes"][mode] = stats
        print(
            f"{'Risk':<6}{'Thesis/yr':>12}{'Mean/yr':>12}{'Std':>10}{'Ratio':>10}{'Verdict':>14}"
        )
        for rid in sorted(THESIS_EVENTS_PER_YEAR.keys()):
            thesis = THESIS_EVENTS_PER_YEAR[rid]
            obs = stats.get(rid, {"mean_per_year": 0.0, "std_per_year": 0.0})
            ratio = obs["mean_per_year"] / thesis if thesis > 0 else float("nan")
            if 0.85 <= ratio <= 1.15:
                verdict = "OK"
            elif ratio >= 1.85:
                verdict = "2x OVER"  # legacy_renewal signature
            elif ratio <= 0.55:
                verdict = "2x UNDER"  # inverse bias like R13
            else:
                verdict = "?"
            print(
                f"{rid:<6}{thesis:>12.3f}{obs['mean_per_year']:>12.2f}"
                f"{obs['std_per_year']:>10.2f}{ratio:>10.2f}{verdict:>14}"
            )
        print()

    out_path = Path(__file__).resolve().parents[1] / "results" / "p_b1_risk_frequency.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
