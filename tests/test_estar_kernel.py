from __future__ import annotations

import pytest

from supply_chain.estar_kernel import (
    EStarAction,
    EStarKernel,
    MASKS,
    PlannerStats,
)


def _kernel(mask: str = "M111") -> EStarKernel:
    return EStarKernel(
        mask_id=mask,
        node_capacities={
            "wdc": 1_000.0,
            "al": 1_000.0,
            "sb": 1_000.0,
            "cssu_a": 500.0,
            "cssu_b": 500.0,
        },
        supplier_capacity={
            "supplier_wdc": 100.0,
            "supplier_al": 100.0,
            "supplier_sb": 100.0,
        },
        source_stock={
            "supplier_wdc": 1_000.0,
            "supplier_al": 1_000.0,
            "supplier_sb": 1_000.0,
        },
        transport_capacity={
            "supplier_wdc": 100.0,
            "supplier_al": 100.0,
            "supplier_sb": 100.0,
            "sb_to_cssu_a": 100.0,
            "sb_to_cssu_b": 100.0,
        },
        initial_inventory={
            "wdc": 0.0,
            "al": 0.0,
            "sb": 200.0,
            "cssu_a": 0.0,
            "cssu_b": 0.0,
        },
        lead_times={
            "supplier_wdc": 200.0,
            "supplier_al": 200.0,
            "supplier_sb": 200.0,
            "sb_to_cssu_a": 200.0,
            "sb_to_cssu_b": 200.0,
        },
    )


def test_factorial_masks_are_complete() -> None:
    assert list(MASKS) == ["M000", "M100", "M010", "M001", "M110", "M101", "M011", "M111"]


def test_masked_rights_fail_closed() -> None:
    with pytest.raises(ValueError, match="does not permit procurement"):
        _kernel("M000").step(
            EStarAction(
                procurement_qty={"supplier_wdc": 1.0},
                active_supplier_lanes=("supplier_wdc",),
            )
        )


def test_procurement_waits_for_lead_time_and_does_not_create_stock() -> None:
    kernel = _kernel()
    transition = kernel.step(
        EStarAction(
            procurement_qty={"supplier_wdc": 50.0},
            buffer_targets={"wdc": 100.0},
            active_supplier_lanes=("supplier_wdc",),
        )
    )
    assert transition.state.inventory["wdc"] == pytest.approx(0.0)
    assert transition.state.on_order["wdc"] == pytest.approx(50.0)
    assert transition.ledger["physical_residual"] == pytest.approx(0.0)
    assert transition.ledger["procurement_ordered"]["total"] == pytest.approx(50.0)


def test_full_buffer_blocks_without_spilling() -> None:
    kernel = _kernel()
    kernel.node_capacities["wdc"] = 10.0
    kernel.lead_times["supplier_wdc"] = 0.0
    transition = kernel.step(
        EStarAction(
            procurement_qty={"supplier_wdc": 100.0},
            buffer_targets={"wdc": 10.0},
            active_supplier_lanes=("supplier_wdc",),
        )
    )
    assert transition.state.inventory["wdc"] == pytest.approx(10.0)
    assert transition.ledger["blocked_qty"]["wdc"] == pytest.approx(90.0)
    assert transition.ledger["physical_residual"] == pytest.approx(0.0)


def test_dispatch_is_conserving_and_demand_is_a_sink() -> None:
    kernel = _kernel()
    transition = kernel.step(
        EStarAction(
            dispatch_qty={"sb_to_cssu_a": 50.0, "sb_to_cssu_b": 50.0},
            active_dispatch_lanes=("sb_to_cssu_a", "sb_to_cssu_b"),
        ),
        demand={"cssu_a": 25.0, "cssu_b": 25.0},
        planner_stats=PlannerStats("test", 0.0, 1, 0, 0),
    )
    assert transition.ledger["physical_residual"] == pytest.approx(0.0)
    assert transition.ledger["dispatch_sent"]["total"] == pytest.approx(100.0)
    assert transition.ledger["delivered"]["total"] == pytest.approx(0.0)
    assert sum(transition.ledger["delivered"].values()) <= 50.0


def test_observation_has_no_future_fields_and_action_is_live() -> None:
    kernel = _kernel()
    before = kernel.payload_sha256()
    observation = kernel.observe()
    assert "future_demand" not in observation
    assert "future_risk" not in observation
    kernel.step(
        EStarAction(
            buffer_targets={"wdc": 100.0, "al": 100.0, "sb": 100.0},
        )
    )
    assert kernel.payload_sha256() != before


def test_nonfinite_values_fail_closed() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        EStarKernel(node_capacities={"wdc": float("inf")})


def test_positive_quantity_must_use_active_lane() -> None:
    with pytest.raises(ValueError, match="its lane to be active"):
        _kernel().step(
            EStarAction(procurement_qty={"supplier_wdc": 1.0}),
        )


def test_procurement_respects_transport_capacity() -> None:
    kernel = _kernel()
    kernel.transport_capacity["supplier_wdc"] = 10.0
    with pytest.raises(ValueError, match="transport capacity"):
        kernel.step(
            EStarAction(
                procurement_qty={"supplier_wdc": 11.0},
                active_supplier_lanes=("supplier_wdc",),
            )
        )
