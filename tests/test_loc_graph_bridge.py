"""The flags-off bridge: `graph_v1` over the shipped topology must BE the shipped model.

This is the E*-R null arm, and the whole expansion rests on it. The check is anchored to a hash
frozen from verified code, not to a comparison between two runs of the same code path -- that
mistake made the G3c null test unable to fail, and it is the reason `self_sha256` cannot be used
here either: it seals `created_at` and provenance, so it must change whenever code changes.
"""
from __future__ import annotations

import warnings

import pytest

from supply_chain.config import HOURS_PER_WEEK
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P
from supply_chain.episode_metrics import compute_episode_metrics
from supply_chain.loc_graph import arc_for_operation, baseline_graph
from supply_chain.scientific_payload import (
    canonical_scientific_payload,
    scientific_payload_sha256,
)
from supply_chain.supply_chain import MFSCSimulation

warnings.filterwarnings("ignore")

RISKS = ("R11", "R12", "R13", "R14", "R21", "R22", "R23", "R24")
SEED = 5_200_001          # burned block; a unit test, not an experiment
WEEKS = 52
# R22 fires on a Uniform(1, 4032h) clock, so a short tape yields ONE event. The first version of
# an earlier diagnostic version used WEEKS=12 and drew a single op4 event -- the exact arc the mutant maps everything
# onto, which made the mutation indistinguishable and the mapping test vacuous. The multiplier
# buys several events across different operations so the check has something to bite on.
R22_FREQUENCY = 6.0


def run(**kw) -> dict:
    sim = MFSCSimulation(
        shifts=1, initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0, seed=SEED, horizon=float(WEEKS * HOURS_PER_WEEK),
        risks_enabled=True, risk_level="current", enabled_risks=set(RISKS),
        risk_frequency_multipliers_by_id={"R22": R22_FREQUENCY},
        cssu_topology_mode="split_v1", cssu_allocation_a=0.5,
        cssu_service_rule="FIFO_PARTIAL", cssu_reallocate_unused=False,
        order_fulfillment_mode="op9_linked", op9_dispatch_policy="fixed_clock_daily",
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"], **kw)
    sim.run()
    payload = canonical_scientific_payload(sim, compute_episode_metrics(sim))
    return {"sim": sim, "payload": payload, "sha": scientific_payload_sha256(payload),
            "arc_events": list(getattr(sim, "loc_arc_down_events", []))}


@pytest.fixture(scope="module")
def serial() -> dict:
    return run()


@pytest.fixture(scope="module")
def graph() -> dict:
    return run(loc_topology_mode="graph_v1")


def test_graph_mode_reproduces_the_shipped_science_exactly(serial, graph):
    """The bridge. Naming an arc must not move a single ration."""
    assert graph["sha"] == serial["sha"]
    assert graph["payload"] == serial["payload"]


# Frozen 2026-08-02 from the verified bridge, on the burned unit-test tape (SEED 5_200_001,
# WEEKS=52 with R22_FREQUENCY=6.0, split_v1, non-fungible). Anchors the null OUTSIDE the code
# path it guards. The tape parameters belong in this comment because the hash is meaningless
# without them -- an earlier version said 12 weeks after the tape had already been lengthened.
GOLDEN_SERIAL_PAYLOAD_SHA256 = "371c5975386868a57c3fa71b16fd02df082d3c90d74408a8cc75b3f303e2bbc1"


def test_the_null_is_anchored_to_a_frozen_hash(serial):
    """Comparing serial with graph only proves they agree, not that either is still correct.

    Both run the same simulator, so a defect in shared code breaks both equally and the bridge
    test above would keep passing. Only an external anchor can fail on that.
    """
    assert serial["sha"] == GOLDEN_SERIAL_PAYLOAD_SHA256, (
        "the shipped model no longer reproduces the frozen science. Either a real defect was "
        "introduced, or the physics changed deliberately -- in which case re-freeze in a commit "
        "that says why, never silently.")


def test_the_tape_fires_enough_r22_events_to_test_the_mapping(graph):
    """A tape with one event, on one operation, cannot distinguish a broken mapping from a good
    one. This guards the guard: if the tape ever goes quiet, the mapping test is vacuous."""
    ops = {e["op_id"] for e in graph["arc_events"]}
    assert len(graph["arc_events"]) >= 3, "too few R22 events to exercise the mapping"
    assert len(ops) >= 2, f"all events hit the same operation {ops}: the mutant would be invisible"


def test_graph_mode_records_arcs_and_serial_mode_records_none(serial, graph):
    """The telemetry must exist under graph_v1 and be absent otherwise, or the flag does nothing."""
    assert serial["arc_events"] == []
    assert graph["arc_events"], "R22 fired but no arc was named: the flag is not wired"
    known = {a.arc_id for a in baseline_graph().arcs}
    assert all(e["arc_id"] in known for e in graph["arc_events"])


# Written out by hand from the thesis chain, NOT derived from `arc_for_operation`. An expectation
# computed with the function under test agrees with it trivially, including when both are wrong --
# which is precisely how the suffix defect (`"SB".endswith("B")`) survived the first version of
# this test.
EXPECTED_ARC = {
    (4, None): "op4_wdc_al",
    (8, None): "op8_al_sb",
    (10, "A"): "op10_sb_cssu_a",
    (10, "B"): "op10_sb_cssu_b",
    (12, "A"): "op12_cssu_a_theatre",
    (12, "B"): "op12_cssu_b_theatre",
}


def test_the_mapping_table_is_a_bijection_onto_the_shipped_arcs():
    """Independent of the simulator: every hop maps to its own arc, and no arc is orphaned."""
    assert sorted(EXPECTED_ARC.values()) == sorted(a.arc_id for a in baseline_graph().arcs)
    assert len(set(EXPECTED_ARC.values())) == len(EXPECTED_ARC)


@pytest.mark.parametrize("key,expected", sorted(EXPECTED_ARC.items(), key=lambda kv: str(kv[0])))
def test_arc_for_operation_matches_the_hand_written_table(key, expected):
    assert arc_for_operation(*key) == expected


def test_every_realized_arc_event_matches_the_hand_written_table(graph):
    """Realized events, checked against the table rather than against the function that made them."""
    for event in graph["arc_events"]:
        assert event["arc_id"] == EXPECTED_ARC[(event["op_id"], event["cssu"])]


def test_arcs_come_back_up_after_the_event(graph):
    """A down set that only grows would silently disconnect the chain for the rest of the run."""
    assert graph["sim"].loc_arcs_down == set()


def test_an_unknown_topology_mode_is_rejected():
    with pytest.raises(ValueError):
        run(loc_topology_mode="graph_v2_does_not_exist")


def test_a_mutated_arc_mapping_breaks_the_bridge(serial, monkeypatch):
    """The mutant: prove the bridge CAN fail.

    Mapping every event onto one arc is wrong but invisible to the science payload, since arc
    telemetry is not part of it. So the mutant has to be caught by the mapping test instead --
    which is exactly why that test exists separately from the hash.
    """
    import supply_chain.supply_chain as sc

    # Map everything onto an arc that no op4/op8 event can legitimately produce.
    monkeypatch.setattr(sc, "_loc_arc_for_operation", lambda op, cssu: "op12_cssu_b_theatre")
    mutated = run(loc_topology_mode="graph_v1")
    assert mutated["sha"] == serial["sha"], "science must be untouched: the flag is telemetry only"
    assert any(e["arc_id"] != arc_for_operation(e["op_id"], e["cssu"])
               for e in mutated["arc_events"]), (
        "the mutant produced a correct mapping; rebuild it or the mapping check proves nothing")
