"""The 2026-07-30 arm options must leave the shipped default bitwise identical.

Five switches were added in one day (`procurement_delay_accumulation`,
`rpj_onset_admission`, `autotomy_predicate`, `fulfillment_delay_distribution`, plus the
`RET_RECOVERY_PERIOD_MODE` migration). Each was claimed to be default-preserving, and the
only evidence was a runner falsifier comparing against the *previous* run's numbers with a
**5% band** -- a 4% perturbation would have passed every one of them.

This pins the claim as an invariant instead: toggling any non-default arm must not move a
single order field on the default path, and the default path must reproduce a frozen
per-order digest.
"""
from __future__ import annotations

import hashlib

import pytest

from supply_chain.config import HOURS_PER_WEEK
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P
from supply_chain.supply_chain import MFSCSimulation

FAMILIES = {"R1r": ("R11", "R12", "R13", "R14"), "R2r": ("R21", "R22", "R23", "R24")}
HORIZON = 8 * HOURS_PER_WEEK


def _run(family: str, seed: int, horizon: float = HORIZON, **kwargs) -> MFSCSimulation:
    risks = FAMILIES[family]
    sim = MFSCSimulation(
        shifts=1,
        initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0,
        seed=seed, horizon=horizon, risks_enabled=True, risk_level="current",
        enabled_risks=set(risks), risk_overrides={r: "increased" for r in risks},
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"],
        **kwargs)
    sim.step(action=None, step_hours=horizon)
    return sim


def _digest(sim: MFSCSimulation) -> str:
    rows = [
        (o.OPTj, o.OATj, o.CTj, o.APj, o.RPj, o.DPj,
         tuple(sorted((o.ret_risk_indicators or {}).items())))
        for o in sim.orders
    ]
    events = [(e.risk_id, e.start_time, e.end_time, e.duration) for e in sim.risk_events]
    return hashlib.sha256(repr((rows, events)).encode()).hexdigest()


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("seed", [2_300_001, 2_500_001])
@pytest.mark.parametrize(
    "override",
    [
        {"procurement_delay_accumulation": "serial"},
        {"rpj_onset_admission": "clamped"},
        {"autotomy_predicate": "le"},
        {"autotomy_predicate": "band", "autotomy_tolerance_hours": 0.0},
        {"fulfillment_delay_distribution": "constant"},
    ],
)
def test_explicit_default_matches_implicit_default(family, seed, override):
    """Passing a switch at its documented default must change nothing.

    `autotomy_predicate="band"` with tolerance 0.0 is included because it is
    mathematically equivalent to "le" and is accepted silently.
    """
    assert _digest(_run(family, seed)) == _digest(_run(family, seed, **override))


def test_non_default_arms_change_exactly_the_families_they_can_reach():
    """Guard against the opposite failure: an arm that is silently a no-op.

    Three of the 2026-07-30 falsifiers passed only because the quantity they checked
    could not vary. An arm that changes nothing makes its whole contract vacuous -- but
    an arm that changes a family it cannot physically reach is just as wrong.

    `procurement_delay_accumulation` touches R12 and R13 only, so it MUST be bitwise
    inert in R2r (which has neither). That is falsifier 1 of
    docs/PREREGISTRO_DURACION_R12_R13_2026-07-30.md, pinned here so it stops depending
    on a runner remembering to check it.

    Horizon matters: the warm-up alone is ~1,299 h, so at 8 weeks only ~47 orders are
    scored and `within_window` and the autotomy band are both no-ops by accident. At 26
    weeks they separate. A check can pass or fail purely on how much data it is given.
    """
    long_h = 26 * HOURS_PER_WEEK
    expected_effect = {
        "procurement_delay_accumulation": {"R1r": True, "R2r": False},
        "rpj_onset_admission": {"R1r": True, "R2r": True},
        "autotomy_band": {"R1r": True, "R2r": True},
    }
    overrides = {
        "procurement_delay_accumulation": {"procurement_delay_accumulation": "parallel"},
        "rpj_onset_admission": {"rpj_onset_admission": "within_window"},
        "autotomy_band": {"demand_on_hand_fulfillment_delay": 48.0074,
                          "autotomy_predicate": "band",
                          "autotomy_tolerance_hours": 0.05},
    }
    for arm, per_family in expected_effect.items():
        for family, should_change in per_family.items():
            base = _digest(_run(family, 2_300_001, horizon=long_h))
            got = _digest(_run(family, 2_300_001, horizon=long_h, **overrides[arm]))
            assert (got != base) is should_change, (
                f"{arm} in {family}: expected "
                f"{'a change' if should_change else 'bitwise inertness'}")


def test_mutually_silent_combinations_are_refused():
    """Combinations where one switch silently discarded another now raise."""
    with pytest.raises(ValueError):
        MFSCSimulation(seed=1, deterministic_baseline=True,
                       rpj_onset_admission="within_window",
                       ret_recovery_period_mode="disruption")
    with pytest.raises(ValueError):
        MFSCSimulation(seed=1, deterministic_baseline=True,
                       fulfillment_delay_distribution="exponential")
    with pytest.raises(ValueError):
        MFSCSimulation(seed=1, deterministic_baseline=True,
                       fulfillment_delay_distribution="exponential",
                       fulfillment_delay_params={"beta": 1.0},
                       on_hand_transit_mode="modelled_legs")


def test_seed_none_is_not_a_constant_stream():
    """`seed=None` must stay OS-seeded, not collapse onto seed=0."""
    a = MFSCSimulation(seed=None, deterministic_baseline=True).fulfillment_rng.random()
    b = MFSCSimulation(seed=0, deterministic_baseline=True).fulfillment_rng.random()
    c = MFSCSimulation(seed=None, deterministic_baseline=True).fulfillment_rng.random()
    assert a != b and a != c
