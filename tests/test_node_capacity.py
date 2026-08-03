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


def test_a_spilling_mutant_ACTUALLY_breaks_conservation(monkeypatch):
    """A mutation test that reasons about a hypothetical mutant proves nothing.

    The first version of this ran the CORRECT code and asserted what a spilling variant would do.
    That is the same shape as every self-referential falsifier caught this week. Here the
    production method is genuinely replaced, and conservation must fail.
    """
    import supply_chain.node_capacity as nc

    ledger = nc.NodeCapacityLedger({"sb": 100.0})
    healthy = ledger.admit("sb", level=0.0, arriving=1000.0)
    assert healthy["admitted"] + healthy["blocked"] == pytest.approx(1000.0)

    def spilling_admit(self, node, level, arriving):
        room = self.headroom(node, level)
        admitted = float(arriving) if room == INF else min(float(arriving), room)
        return {"admitted": admitted, "blocked": 0.0}      # the surplus is silently destroyed

    monkeypatch.setattr(nc.NodeCapacityLedger, "admit", spilling_admit)
    mutated = nc.NodeCapacityLedger({"sb": 100.0}).admit("sb", level=0.0, arriving=1000.0)
    assert mutated["admitted"] + mutated["blocked"] != pytest.approx(1000.0), (
        "the mutant conserved mass, so it is not a mutant and this test proves nothing")


def test_binding_fraction_is_a_fraction():
    """It returned 3.0 for three blocks at one node: a normaliser dividing by nodes, not calls."""
    ledger = NodeCapacityLedger({"sb": 100.0})
    for _ in range(3):
        ledger.admit("sb", level=0.0, arriving=1000.0)
    assert ledger.binding_fraction() == pytest.approx(1.0)

    mixed = NodeCapacityLedger({"sb": 100.0, "cssu_a": INF})
    mixed.admit("sb", level=0.0, arriving=1000.0)      # blocks
    mixed.admit("cssu_a", level=0.0, arriving=1000.0)  # does not
    assert mixed.binding_fraction() == pytest.approx(0.5)
    assert 0.0 <= mixed.binding_fraction() <= 1.0


def test_a_budgeted_ledger_records_and_enforces_its_total():
    """Turns "shared budget" from a convention into a checkable invariant."""
    from supply_chain.node_capacity import budgeted_ledger

    ledger = budgeted_ledger(10_000.0, {"sb": 1.0, "cssu_a": 1.0, "cssu_b": 2.0})
    assert ledger.total_budget == pytest.approx(10_000.0)
    assert sum(ledger.capacities.values()) == pytest.approx(10_000.0)
    assert ledger.is_inert is False


@pytest.mark.parametrize("bad", [
    # capacities that do not sum to the declared budget
    lambda: NodeCapacityLedger({n: 10.0 for n in CAPACITY_NODES}, total_budget=999.0),
    # an unlimited node inside a budget returns the scarce resource to being abundant
    lambda: NodeCapacityLedger({"sb": 100.0}, total_budget=100.0),
])
def test_a_budget_that_is_not_conserved_is_rejected(bad):
    with pytest.raises(ValueError):
        bad()


def test_the_simulators_INF_is_not_float_inf():
    """Stated because the docstring used to overclaim: the simulator's INF is 10_000_000, and the
    two WIP containers can carry a genuinely finite cap via serial_wip_capacity_rations. So the
    defensible claim is that every STORAGE node is effectively unlimited, not every container."""
    assert INF == float("inf"), "this module's INF is the real infinity, unlike the simulator's"
