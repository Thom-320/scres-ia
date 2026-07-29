"""Two ReT invariants that the DES does not currently satisfy.

Both are marked `xfail(strict=True)`: they encode the contract we want, they fail
today for known reasons, and they will fail *loudly* the day someone fixes the
cause without updating this file. Findings of 2026-07-29, see
`docs/RET_METRIC_DEFECTS_2026-07-29.md`.
"""
from __future__ import annotations

import pytest

from supply_chain.config import HOURS_PER_WEEK
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P
from supply_chain.episode_metrics import compute_episode_metrics
from supply_chain.expanded_contract_controllers import level_targets
from supply_chain.supply_chain import MFSCSimulation

HORIZON = 52 * HOURS_PER_WEEK
R1R = ("R11", "R12", "R13", "R14")


def _run(step_hours: float, *, risks: bool = True) -> dict:
    sim = MFSCSimulation(
        shifts=1, initial_buffers=level_targets(168),
        inventory_replenishment_period=168.0, seed=1_620_001, horizon=HORIZON,
        risks_enabled=risks, risk_level="current",
        enabled_risks=set(R1R) if risks else set(),
        risk_overrides={r: "increased" for r in R1R} if risks else {},
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
    elapsed = 0.0
    while elapsed < HORIZON:
        step = min(step_hours, HORIZON - elapsed)
        sim.step(action=None, step_hours=step)
        elapsed += step
    m = compute_episode_metrics(sim)
    return {"m": m, "sim": sim}


def test_physical_endpoints_are_step_cadence_invariant():
    """The control: the trajectory itself does not depend on step() cadence."""
    a, b = _run(HORIZON)["m"], _run(24.0)["m"]
    assert a["flow_fill_rate"] == pytest.approx(b["flow_fill_rate"])
    assert a["delivered_rations"] == pytest.approx(b["delivered_rations"])
    assert a["lost_orders"] == pytest.approx(b["lost_orders"])


@pytest.mark.xfail(strict=True, reason=(
    "ret_excel depends on how often step() is called. `_op_down_since` is reset to "
    "env.now at every step boundary (supply_chain.py:1856), and the legacy "
    "ongoing-disruption attribution then measures overlap from that reset value "
    "(supply_chain.py:5743), so RPj shrinks as cadence rises and ReT rises with it. "
    "Measured 0.004369 at one step against 0.005981 hourly on an identical "
    "trajectory. Fix: derive RPj from immutable risk intervals."))
def test_ret_excel_is_step_cadence_invariant():
    a, b = _run(HORIZON)["m"], _run(24.0)["m"]
    assert a["ret_excel"] == pytest.approx(b["ret_excel"], rel=1e-6)


@pytest.mark.xfail(strict=True, reason=(
    "The autotomy branch of ReT is unreachable. It requires CTj <= LTj, but the "
    "minimum cycle time is 54 h against a 48 h lead-time promise, in EVERY "
    "configuration tested: risks on and off, 1 and 3 shifts, zero and maximum "
    "buffer. So excel_case_pct_autotomy is 0.00 always, fill_rate_on_time is 0.00 "
    "always, and no lever can move either. Garrido's own output has mean APj > 0."))
def test_autotomy_case_is_reachable():
    """Absorption -- disruption occurs and the order is still on time -- is the
    thesis's central resilience mechanism. Our DES cannot express it."""
    m = _run(HORIZON)["m"]
    assert m["excel_case_pct_autotomy"] > 0.0


@pytest.mark.xfail(strict=True, reason="same 54 h > 48 h floor as above")
def test_some_order_is_ever_delivered_on_time():
    m = _run(HORIZON, risks=False)["m"]
    assert m["fill_rate_on_time"] > 0.0
