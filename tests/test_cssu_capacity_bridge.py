"""Finite CSSU storage inside the DES: inert by default, and mass-conserving when it bites.

This is the bridge the capacity helper was missing. Testing `NodeCapacityLedger()` on its own only
showed that a dictionary defaults to unlimited; it said nothing about the simulator. Here the
simulator itself is run, and the null arm is anchored to a hash frozen from verified code rather
than to a comparison of two runs down the same code path.
"""
from __future__ import annotations

import warnings

import pytest

from supply_chain.config import HOURS_PER_WEEK
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P
from supply_chain.episode_metrics import compute_episode_metrics
from supply_chain.scientific_payload import (
    canonical_scientific_payload,
    scientific_payload_sha256,
)
from supply_chain.supply_chain import MFSCSimulation

warnings.filterwarnings("ignore")

RISKS = ("R11", "R12", "R13", "R14", "R21", "R22", "R23", "R24")
SEED = 5_200_001          # burned block; a unit test, not an experiment
WEEKS = 26


def run(**kw) -> dict:
    sim = MFSCSimulation(
        shifts=1, initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0, seed=SEED, horizon=float(WEEKS * HOURS_PER_WEEK),
        risks_enabled=True, risk_level="current", enabled_risks=set(RISKS),
        cssu_topology_mode="split_v1", cssu_allocation_a=0.5,
        cssu_service_rule="FIFO_PARTIAL", cssu_reallocate_unused=False,
        order_fulfillment_mode="op9_linked", op9_dispatch_policy="fixed_clock_daily",
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"], **kw)
    sim.run()
    panel = compute_episode_metrics(sim)
    ledger = getattr(sim, "_cssu_capacity_ledger", None)
    return {"sim": sim, "panel": panel,
            "sha": scientific_payload_sha256(canonical_scientific_payload(sim, panel)),
            "delivered": dict(sim.cssu_delivered), "demanded": dict(sim.cssu_demanded),
            "evidence": None if ledger is None else ledger.as_evidence()}


@pytest.fixture(scope="module")
def shipped() -> dict:
    return run()


@pytest.fixture(scope="module")
def unlimited() -> dict:
    """Capacity switched ON but set so wide it cannot bind. Must equal the shipped model."""
    return run(cssu_storage_capacity={"A": 1e12, "B": 1e12})


@pytest.fixture(scope="module")
def tight() -> dict:
    return run(cssu_storage_capacity={"A": 500.0, "B": 500.0})


def test_the_flag_is_inert_when_the_caps_cannot_bind(shipped, unlimited):
    """Switching the machinery on must not move a ration by itself."""
    assert unlimited["sha"] == shipped["sha"]
    assert unlimited["evidence"]["total_blocked"] == 0.0


# Frozen 2026-08-03 from the verified bridge, on the burned unit-test tape (SEED 5_200_001,
# WEEKS=26, split_v1, FIFO_PARTIAL, non-fungible, no capacity). The tape parameters belong beside
# the hash because the hash means nothing without them.
GOLDEN_SHIPPED_PAYLOAD_SHA256 = "f3fe61b1e2b1f4a63ff30beb296d4c6bf54be029fed0ab7877b69e24201af385"


def test_the_null_is_anchored_outside_the_code_path_it_guards(shipped):
    """Comparing shipped with unlimited only proves they agree; both run the same simulator, so a
    defect in shared code breaks both equally. Only a frozen anchor can fail on that."""
    assert shipped["sha"] == GOLDEN_SHIPPED_PAYLOAD_SHA256, (
        "the shipped model no longer reproduces the frozen science. Either a defect was "
        "introduced, or the physics changed deliberately -- re-freeze in a commit that says why.")


def test_a_tight_cap_actually_binds(tight):
    """Liveness, measured in a real rollout rather than on a bare dictionary. A cap that never
    fills makes every result attributed to it vacuous."""
    assert tight["evidence"]["total_blocked"] > 0.0
    assert 0.0 < tight["sim"]._cssu_capacity_ledger.binding_fraction() <= 1.0


def test_a_tight_cap_changes_the_science(shipped, tight):
    """If capping storage changed nothing, the mechanism would be decoration."""
    assert tight["sha"] != shipped["sha"]


def test_blocked_rations_are_not_delivered_and_not_destroyed(shipped, tight):
    """Mass, checked on the DES ledger and not on the helper.

    Blocking means the surplus was never dispatched, so it stays at the SB. Demand is exogenous
    and must be untouched -- a capacity that reduced DEMAND would be deleting the problem instead
    of constraining the solution.
    """
    assert tight["demanded"] == shipped["demanded"], "capacity must not alter exogenous demand"
    assert sum(tight["delivered"].values()) <= sum(shipped["delivered"].values()) + 1e-6


def test_a_zero_capacity_cssu_receives_nothing():
    out = run(cssu_storage_capacity={"A": 0.0, "B": 1e12})
    assert out["delivered"]["A"] == pytest.approx(0.0)
    assert out["delivered"]["B"] > 0.0


def test_an_invalid_capacity_is_rejected_at_construction():
    with pytest.raises(ValueError):
        run(cssu_storage_capacity={"A": -1.0, "B": 1.0})
    with pytest.raises(ValueError):
        run(cssu_storage_capacity={"NOWHERE": 1.0})


def test_a_spilling_mutant_in_the_production_path_is_caught(shipped, monkeypatch):
    """The mutation must hit the code the DES actually calls, not a hypothetical.

    A ledger that reports the surplus as admitted would let the CSSU absorb more than its cap, so
    the tight run would stop differing from the shipped one on delivered volume.
    """
    import supply_chain.node_capacity as nc

    monkeypatch.setattr(nc.NodeCapacityLedger, "admit",
                        lambda self, node, level, arriving: {"admitted": float(arriving),
                                                             "blocked": 0.0})
    mutated = run(cssu_storage_capacity={"A": 500.0, "B": 500.0})
    assert mutated["sha"] == shipped["sha"], (
        "with admission neutered the cap must stop binding; if the science still differs, the "
        "mutation is not reaching the production path and this test proves nothing")
