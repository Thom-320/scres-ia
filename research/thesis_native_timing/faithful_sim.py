"""Thesis-faithful MFSC sim builder + dual-metric scorer for the timing ablation.

Encapsulates the representative evaluation setup (the ``THESIS_FAITHFUL_PROTOCOL`` knobs,
``op9_arrival`` warm-up, benchmark horizon, post-warm-up treatment window) so every policy in
the metric x cadence x risk ablation is scored on an identical, thesis-validated footing --
the condition for the "flags-off identity" gate (cell A must reproduce the known null).

A bare ``MFSCSimulation(...)`` uses non-thesis defaults (warmup_trigger='production',
r14_defect_mode='reprocess', ...) and cold-start orders drag ret_excel down; this module fixes
that by threading the protocol knobs and scoring only post-warm-up orders.
"""

from __future__ import annotations

from typing import Any

from supply_chain import config as C
from supply_chain.supply_chain import MFSCSimulation

from .canonical_panel import compute_episode_metrics
from .resilience_timeresolved import resilience_timeresolved

_P = C.THESIS_FAITHFUL_PROTOCOL

# Default evaluation horizon = the RL benchmark reference (5y), long enough for the sim to settle
# past the op9-arrival warm-up while keeping the oracle sweep tractable.
BENCHMARK_HORIZON_HOURS: float = float(C.BENCHMARK_REFERENCE_MAX_STEPS * C.HOURS_PER_WEEK)


def build_faithful_sim(
    *,
    seed: int,
    horizon: float = BENCHMARK_HORIZON_HOURS,
    risks_enabled: bool = True,
    risk_frequency_multiplier: float = 1.0,   # phi
    risk_impact_multiplier: float = 1.0,      # psi
    shifts: int = 1,
    initial_buffers: Any = None,
) -> MFSCSimulation:
    """Construct (not run) a thesis-faithful MFSCSimulation with optional risk modulation."""
    return MFSCSimulation(
        seed=int(seed),
        horizon=float(horizon),
        risks_enabled=bool(risks_enabled),
        risk_level="current",
        year_basis=_P["year_basis"],
        warmup_trigger=_P["warmup_trigger"],
        downstream_q_source=_P["downstream_q_source"],
        r14_defect_mode=_P["r14_defect_mode"],
        risk_occurrence_mode=_P["risk_occurrence_mode"],
        raw_material_flow_mode=_P["raw_material_flow_mode"],
        raw_material_order_up_to_multiplier=_P["raw_material_order_up_to_multiplier"],
        risk_frequency_multiplier=float(risk_frequency_multiplier),
        risk_impact_multiplier=float(risk_impact_multiplier),
        shifts=int(shifts),
        initial_buffers=initial_buffers,
    )


def score_faithful(sim: MFSCSimulation) -> dict[str, float]:
    """Score a *completed* faithful sim on both endpoints (post-warm-up).

    Returns the canonical panel (incl. ``ret_excel`` -- the primary) plus the bounded
    time-resolved ``resilience_triangle_v1`` (co-primary) and its diagnostics.
    """
    panel = compute_episode_metrics(sim)  # treatment_start defaults to sim.warmup_time
    return resilience_timeresolved(panel, sim)


def run_and_score(
    *,
    seed: int,
    horizon: float = BENCHMARK_HORIZON_HOURS,
    risks_enabled: bool = True,
    risk_frequency_multiplier: float = 1.0,
    risk_impact_multiplier: float = 1.0,
    shifts: int = 1,
) -> dict[str, float]:
    """Convenience: build a static-policy faithful sim, run to horizon, return dual-metric panel."""
    sim = build_faithful_sim(
        seed=seed,
        horizon=horizon,
        risks_enabled=risks_enabled,
        risk_frequency_multiplier=risk_frequency_multiplier,
        risk_impact_multiplier=risk_impact_multiplier,
        shifts=shifts,
    ).run()
    return score_faithful(sim)
