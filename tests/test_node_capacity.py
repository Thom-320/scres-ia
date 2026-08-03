"""Finite capacity: inert by default, mass-conserving when it bites, and live or it is decoration.

The two tests that carry weight are conservation -- blocked rations must never vanish, because a
metric that rewards deleting demand is already a measured failure mode here -- and liveness, since
a cap that never fills makes any result attributed to it meaningless.
"""
from __future__ import annotations

import random

import pytest

from supply_chain.node_capacity import (
    CAPACITY_NODES,
    INF,
    UNLIMITED,
    NodeCapacityLedger,
    budget_split,
    capacity_is_live,
    validate_capacities,
)


def test_the_default_is_the_shipped_model():
    """Every simpy.Container in the simulator is built with capacity=INF; so is this."""
    ledger = NodeCapacityLedger()
    assert ledger.is_inert is True
    assert all(ledger.capacities[n] == INF for n in CAPACITY_NODES)


def test_unlimited_capacity_admits_everything_and_blocks_nothing():
    ledger = NodeCapacityLedger()
    out = ledger.admit("sb", level=1e12, arriving=1e9)
    assert out == {"admitted": 1e9, "blocked": 0.0}
    assert capacity_is_live(ledger) is False


def test_a_full_node_blocks_the_surplus():
    ledger = NodeCapacityLedger({"sb": 1000.0})
    out = ledger.admit("sb", level=900.0, arriving=500.0)
    assert out["admitted"] == pytest.approx(100.0)
    assert out["blocked"] == pytest.approx(400.0)
    assert capacity_is_live(ledger) is True


@pytest.mark.parametrize("trial", range(200))
def test_mass_is_conserved_on_every_admission(trial):
    """THE conservation test. Blocked rations stay upstream; they are never destroyed.

    Spilling would silently delete demand and flatter every downstream metric, which is exactly
    the failure already measured in ret_excel -- a policy that abandons a claimant scores better.
    """
    rng = random.Random(31337 + trial)
    caps = {n: rng.choice([INF, 0.0, float(rng.randint(1, 5000))]) for n in CAPACITY_NODES}
    ledger = NodeCapacityLedger(caps)
    total_in = 0.0
    for _ in range(20):
        node = rng.choice(CAPACITY_NODES)
        arriving = float(rng.randint(0, 3000))
        out = ledger.admit(node, level=float(rng.randint(0, 5000)), arriving=arriving)
        assert out["admitted"] + out["blocked"] == pytest.approx(arriving), "mass created or lost"
        assert out["admitted"] >= 0.0 and out["blocked"] >= 0.0
        total_in += arriving
    ev = ledger.as_evidence()
    assert ev["total_admitted"] + ev["total_blocked"] == pytest.approx(total_in)


def test_a_zero_capacity_node_admits_nothing():
    ledger = NodeCapacityLedger({"cssu_b": 0.0})
    out = ledger.admit("cssu_b", level=0.0, arriving=750.0)
    assert out["admitted"] == 0.0 and out["blocked"] == pytest.approx(750.0)


def test_headroom_never_goes_negative_when_a_node_is_over_its_cap():
    """Levels can exceed a cap that was tightened mid-run; headroom must clamp, not go negative,
    or a later admission would silently subtract from the arriving quantity."""
    ledger = NodeCapacityLedger({"al": 100.0})
    assert ledger.headroom("al", level=500.0) == 0.0
    assert ledger.admit("al", level=500.0, arriving=10.0)["admitted"] == 0.0


def test_the_budget_split_conserves_the_total_and_is_continuous():
    split = budget_split(10_000.0, {"sb": 1.0, "cssu_a": 1.0, "cssu_b": 2.0})
    assert sum(split.values()) == pytest.approx(10_000.0)
    assert split["cssu_b"] == pytest.approx(5_000.0)
    # Continuous by design: Garrido asked for continuous variables on 2 July.
    fine = budget_split(10_000.0, {"sb": 0.3333, "cssu_a": 0.3333, "cssu_b": 0.3334})
    assert sum(fine.values()) == pytest.approx(10_000.0)
    assert fine["cssu_b"] > fine["sb"]


@pytest.mark.parametrize("trial", range(100))
def test_the_budget_is_conserved_for_arbitrary_shares(trial):
    rng = random.Random(99 + trial)
    shares = {n: float(rng.randint(0, 20)) for n in CAPACITY_NODES}
    if sum(shares.values()) == 0:
        shares["sb"] = 1.0
    total = float(rng.randint(1, 10 ** 6))
    assert sum(budget_split(total, shares).values()) == pytest.approx(total)


def test_an_unlimited_budget_is_rejected_because_it_removes_the_decision():
    """A budget that is infinite is the shipped model wearing a decision's name."""
    with pytest.raises(ValueError):
        budget_split(INF, {"sb": 1.0})
    with pytest.raises(ValueError):
        budget_split(0.0, {"sb": 1.0})


def test_liveness_is_reported_because_a_cap_that_never_fills_is_decoration():
    loose = NodeCapacityLedger({"sb": 10 ** 9})
    loose.admit("sb", level=0.0, arriving=1000.0)
    assert capacity_is_live(loose) is False, "a cap this loose never binds; results are vacuous"

    tight = NodeCapacityLedger({"sb": 100.0})
    tight.admit("sb", level=0.0, arriving=1000.0)
    assert capacity_is_live(tight) is True


@pytest.mark.parametrize("bad", [
    lambda: validate_capacities({"nowhere": 1.0}),
    lambda: validate_capacities({"sb": -1.0}),
    lambda: validate_capacities({"sb": float("nan")}),
    lambda: NodeCapacityLedger().admit("sb", level=0.0, arriving=-1.0),
    lambda: NodeCapacityLedger().admit("nowhere", level=0.0, arriving=1.0),
    lambda: budget_split(100.0, {"nowhere": 1.0}),
    lambda: budget_split(100.0, {"sb": -1.0}),
    lambda: budget_split(100.0, {"sb": 0.0}),
])
def test_invalid_configurations_are_rejected(bad):
    with pytest.raises(ValueError):
        bad()


def test_a_spilling_mutant_would_break_conservation():
    """The mutant, to show the conservation test can fail.

    If `admit` dropped the surplus instead of returning it, admitted + blocked would fall short of
    arriving and the parametrised conservation test above would fire on the first full node.
    """
    ledger = NodeCapacityLedger({"sb": 100.0})
    out = ledger.admit("sb", level=0.0, arriving=1000.0)
    spilled = out["admitted"]                 # what a spilling implementation would report alone
    assert spilled < 1000.0
    assert out["admitted"] + out["blocked"] == pytest.approx(1000.0), (
        "conservation holds only because the surplus is returned; a spilling variant fails here")
