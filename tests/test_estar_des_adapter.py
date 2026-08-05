from __future__ import annotations

import pytest

from supply_chain.estar_bridge import (
    check_expanded_bridge_smoke,
    check_flags_off_adapter_golden,
    load_tape,
    make_expanded_sim,
)
from supply_chain.estar_des_adapter import EStarDESAdapter
from supply_chain.estar_kernel import EStarAction
from supply_chain.supply_chain import OrderRecord


TAPE = "results/expanded_contract_comparators_v2_preflight_1dc40c1/preflight/R1r_actual_tapes.json"
M000_GOLDEN = "e24d4ef6408276d30987dc27c065fe7f2a274d139c468b8eb6354bcb4d22725d"


def test_adapter_m000_matches_historical_golden() -> None:
    tape = load_tape(TAPE)
    result = check_flags_off_adapter_golden(tape, M000_GOLDEN)
    assert result["passed"] is True


def test_expanded_bridge_smoke_closes_all_masks_without_strategic_injection() -> None:
    tape = load_tape(TAPE)
    result = check_expanded_bridge_smoke(
        tape, None, m000_expected_digest=M000_GOLDEN
    )
    assert result["passed"] is True
    assert len(result["residuals"]) == 8
    assert all(
        row["raw_residual"] == pytest.approx(0.0)
        and row["ration_residual"] == pytest.approx(0.0)
        and row["strategic_raw_injected"] == 0.0
        and row["strategic_rations_injected"] == 0.0
        for row in result["residuals"]
    )


def test_upstream_target_is_telemetry_not_inventory_creation() -> None:
    tape = load_tape(TAPE)
    sim = make_expanded_sim(tape, "M010")
    sim.step_e_star(
        EStarAction(buffer_targets={"wdc": 500.0, "al": 500.0, "sb": 500.0}),
        step_hours=168.0,
    )
    assert sim.inventory_buffer_targets["op3_rm"] == pytest.approx(500.0)
    assert sim.inventory_buffer_targets["op5_rm"] == pytest.approx(500.0)
    assert sim.inventory_buffer_targets["op9_rations"] == pytest.approx(500.0)
    assert sim.total_strategic_raw_injected == pytest.approx(0.0)
    assert sim.total_strategic_rations_injected == pytest.approx(0.0)


def test_dispatch_cannot_exceed_existing_claimant_shortfall() -> None:
    tape = load_tape(TAPE)
    sim = make_expanded_sim(tape, "M001")
    sim.rations_sb.put(10.0)
    with pytest.raises(ValueError, match="claimant shortfall"):
        sim.step_e_star(
            EStarAction(
                dispatch_qty={"sb_to_cssu_a": 1.0},
                active_dispatch_lanes=("sb_to_cssu_a",),
            ),
            step_hours=168.0,
        )


def test_bridge_smoke_catches_unaccounted_procurement(monkeypatch: pytest.MonkeyPatch) -> None:
    """The conservation receipt must reject a delivery that omits its source ledger."""

    def unaccounted_delivery(self: EStarDESAdapter, record):
        yield self.env.timeout(max(0.0, record.due_at - float(self.env.now)))
        # Deliberately bypass the source/receipt accounting.  This is a real
        # production-path mutation: the physical container changes, but the
        # external source ledger does not.
        yield self._container_for_destination(record.destination).put(record.quantity)

    monkeypatch.setattr(
        EStarDESAdapter,
        "_deliver_e_star_procurement",
        unaccounted_delivery,
    )
    tape = load_tape(TAPE)
    result = check_expanded_bridge_smoke(
        tape, None, m000_expected_digest=M000_GOLDEN
    )
    assert result["passed"] is False
    assert any(
        row["raw_residual"] < -1e-9 or row["ration_residual"] < -1e-9
        for row in result["residuals"]
    )


def test_expanded_procurement_and_dispatch_respect_lead_times() -> None:
    tape = load_tape(TAPE)
    sim = make_expanded_sim(tape, "M101")
    sim.apply_e_star_action(
        EStarAction(
            procurement_qty={"supplier_wdc": 100.0},
            active_supplier_lanes=("supplier_wdc",),
        )
    )
    sim.env.run(until=12.0)
    assert sim.raw_material_wdc.level == pytest.approx(0.0)
    assert sim.e_star_evidence()["in_transit"]
    sim.env.run(until=25.0)
    assert sim.raw_material_wdc.level == pytest.approx(100.0)

    sim = make_expanded_sim(tape, "M001")
    sim.rations_sb.put(10.0)
    sim.pending_backorders.append(
        OrderRecord(
            j=0,
            OPTj=0.0,
            quantity=1.0,
            remaining_qty=1.0,
            cssu_destination="A",
        )
    )
    sim._refresh_pending_backorder_qty()
    sim.apply_e_star_action(
        EStarAction(
            dispatch_qty={"sb_to_cssu_a": 1.0},
            active_dispatch_lanes=("sb_to_cssu_a",),
        )
    )
    sim.env.run(until=12.0)
    assert sim.rations_sb.level == pytest.approx(10.0)
    assert sim.e_star_evidence()["dispatch_in_transit"]["sb_to_cssu_a"] == pytest.approx(1.0)
    sim.env.run(until=25.0)
    assert sim.rations_sb.level == pytest.approx(9.0)
