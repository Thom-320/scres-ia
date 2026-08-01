"""The falsifier for `service_first_resilience_v2`, and the control that proves it can fail.

The audit found that `v1`'s leading component, `lost_orders == 0`, is a proxy for backlog-queue
OVERFLOW rather than for abandonment: an order is labelled `lost` only when the queue passes
`BACKORDER_QUEUE_CAP = 60`. A policy that parks orders permanently just under the cap therefore
abandons them and still scores a perfect gate.

The test that matters is the exploit written as an assertion: converting lost orders into
permanently-pending ones must NOT improve the key. `v1` is kept as the control -- it must improve,
because if it does not the test would pass for both and prove nothing. That pairing is what makes
this a validated falsifier rather than an assertion.
"""
from __future__ import annotations

import pytest

from supply_chain.config import HOURS_PER_WEEK, THESIS_FAITHFUL_PROTOCOL as P
from supply_chain.episode_metrics import compute_episode_metrics
from supply_chain.service_first_metric import (
    claimant_fills,
    service_first_key,
    service_first_key_v2,
    service_first_v2_components,
)
from supply_chain.supply_chain import MFSCSimulation


def _panel(*, lost: float, fill: float, backorders: float = 0.0, ret: float = 0.5) -> dict:
    return {
        "lost_orders": lost,
        "flow_fill_rate": fill,
        "backorder_qty_final": backorders,
        "ret_excel_visible_clipped_0_1": ret,
    }


def test_v2_is_not_fooled_by_converting_losses_into_permanent_backlog():
    """THE falsifier. Same delivered quantity, losses relabelled as pending: key must not rise."""
    # Identical physical outcome -- nothing more was delivered -- but the orders that used to
    # overflow the queue now sit in it forever, so `lost_orders` reads 0.
    abandoning = _panel(lost=60, fill=0.50, backorders=0.0)
    relabelled = _panel(lost=0, fill=0.50, backorders=150_000.0)
    fills = {"A": 0.99, "B": 0.01}          # the shortfall is concentrated: this IS abandonment

    v2_before = service_first_key_v2(abandoning, fills)
    v2_after = service_first_key_v2(relabelled, fills)
    assert v2_after <= v2_before, "v2 rewarded relabelling a loss as permanent backlog"

    # The control: v1 MUST be fooled. Without this the test could pass vacuously.
    assert service_first_key(relabelled) > service_first_key(abandoning), (
        "v1 was expected to be fooled by the relabelling; if it is not, this test proves nothing"
    )


def test_v2_leading_component_is_the_worst_claimant_not_the_average():
    """Equal aggregate fill, opposite distributions: the abandoning split must lose."""
    balanced = _panel(lost=0, fill=0.70)
    lopsided = _panel(lost=0, fill=0.70)
    assert service_first_key_v2(balanced, {"A": 0.70, "B": 0.70}) > service_first_key_v2(
        lopsided, {"A": 0.99, "B": 0.41}
    )


def test_v2_degenerates_to_aggregate_fill_without_a_claimant_partition():
    """With one claimant, abandoning a claimant is undefined; the key must not invent a value."""
    panel = _panel(lost=3, fill=0.62)
    assert service_first_key_v2(panel, {})[0] == pytest.approx(0.62)
    assert service_first_key_v2(panel, None)[0] == pytest.approx(0.62)


def test_v2_ranks_the_measured_contention_sweep_the_way_service_does():
    """End to end on the real DES: the balanced split must win, as it does on fill and loses on ReT."""
    horizon = 26.0 * HOURS_PER_WEEK
    keys = {}
    for share in (0.1, 0.5, 0.9):
        sim = MFSCSimulation(
            shifts=1,
            initial_buffers={name: 0.0 for name in ("op3_rm", "op5_rm", "op9_rations")},
            inventory_replenishment_period=0.0, seed=9_100_001, horizon=horizon,
            risks_enabled=True, risk_level="current",
            enabled_risks={"R21", "R22", "R23", "R24"},
            cssu_topology_mode="split_v1", cssu_allocation_a=share,
            cssu_service_rule="FIFO_PARTIAL", cssu_reallocate_unused=False,
            order_fulfillment_mode="op9_linked", op9_dispatch_policy="fixed_clock_daily",
            strict_exogenous_crn=True, year_basis=P["year_basis"],
            warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])
        sim.run()
        panel = compute_episode_metrics(sim)
        keys[share] = service_first_key_v2(panel, claimant_fills(sim))

    assert max(keys, key=lambda s: keys[s]) == 0.5, (
        f"v2 did not prefer the balanced split: {keys}"
    )


def test_v2_components_are_named_and_json_safe():
    components = service_first_v2_components(_panel(lost=0, fill=0.8), {"A": 0.9, "B": 0.7})
    assert components["worst_claimant_fill"] == pytest.approx(0.7)
    assert all(isinstance(value, float) for value in components.values())
