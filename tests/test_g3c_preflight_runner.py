"""G3c minimum-dwell physics, tested against the SIMULATOR rather than against a runner.

Written this way on purpose. Two runners for G3c were authored in parallel on 2026-08-05, and a
test that imports one of them proves nothing about the mechanism -- it proves that a module agrees
with itself, which is the exact failure shape this project has caught six times.

The finding these tests encode: **the frozen level set {1, 3, 7} has a dead middle**. The natural
re-decision spacing of the CSSU action, given the 24 h activation latency and the daily cadence, is
about three days, so every dwell up to four days is INERT and only `7` is a real treatment.
"""
from __future__ import annotations

import warnings

import pytest

from supply_chain.config import HOURS_PER_WEEK
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P
from supply_chain.g3c_temporal import (
    G3C_MIN_DWELL_LEVELS_DAYS,
    G3C_NULL_MIN_DWELL_DAYS,
    validate_min_dwell_days,
)
from supply_chain.supply_chain import MFSCSimulation

warnings.filterwarnings("ignore")

SEED = 5_200_001                    # burned block: a fixture, never scientific power
RISKS = ("R11", "R12", "R13", "R14", "R21", "R22", "R23", "R24")
WEEKS = 52


def run_under_maximal_switching_pressure(dwell: int) -> MFSCSimulation:
    """Alternate the requested split every step. Nothing in the model asks harder than this, so a
    dwell that never holds HERE cannot hold under any policy at this cadence."""
    sim = MFSCSimulation(
        shifts=1, initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0, seed=SEED,
        horizon=float(WEEKS * HOURS_PER_WEEK), risks_enabled=True, risk_level="current",
        enabled_risks=set(RISKS), cssu_topology_mode="split_v1", cssu_allocation_a=0.5,
        cssu_service_rule="FIFO_PARTIAL", cssu_reallocate_unused=False,
        order_fulfillment_mode="op9_linked", op9_dispatch_policy="fixed_clock_daily",
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"],
        cssu_min_dwell_days=float(dwell))
    step, done = 0, False
    while not done:
        target = 0.9 if step % 2 == 0 else 0.1
        wants = abs(float(sim.cssu_allocation_a) - target) > 1e-9
        action = ({"cssu_allocation_a": target}
                  if wants and sim._pending_cssu_action is None else None)
        _, _, done, _ = sim.step(action=action, step_hours=24.0)
        step += 1
    return sim


@pytest.fixture(scope="module")
def held() -> dict[int, int]:
    return {d: run_under_maximal_switching_pressure(d).cssu_blocked_by_dwell_count
            for d in (1, 2, 3, 4, 7)}


def test_the_null_level_is_inert_which_is_what_makes_it_a_regression_null(held):
    assert G3C_NULL_MIN_DWELL_DAYS == 1
    assert held[1] == 0, "a null level that holds an action is not a regression null"


def test_dwell_seven_actually_binds(held):
    """Liveness. A treatment level that never holds makes every downstream number vacuous."""
    assert held[7] > 0


def test_MEASURED_the_middle_level_of_the_frozen_grid_is_DEAD(held):
    """The preflight finding, pinned so it cannot be lost.

    `cssu_min_dwell_days=3` holds ZERO actions even under maximal switching pressure, and so do 2
    and 4. The contract's `{1, 3, 7}` therefore contains one null and one inert cell: the grid has
    a single real treatment, not two, and any power calculation over three levels overstates what
    the design can learn.
    """
    assert held[2] == held[3] == held[4] == 0, (
        "a dwell of 2-4 days now binds; the dead-middle finding no longer holds and the G3c level "
        "grid must be re-derived rather than assumed")
    assert 3 in G3C_MIN_DWELL_LEVELS_DAYS, "the contract still freezes the inert level"


def test_the_dwell_that_binds_suppresses_switching(held):
    """Direction check: holding actions must REDUCE realised switches, not merely count events."""
    loose = run_under_maximal_switching_pressure(1).cssu_switch_count
    tight = run_under_maximal_switching_pressure(7).cssu_switch_count
    assert tight < loose


@pytest.mark.parametrize("bad", [0, 2, 5, 8, 1.5, True])
def test_unregistered_levels_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_min_dwell_days(bad)
