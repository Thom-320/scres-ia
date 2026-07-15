import json
from pathlib import Path

import pytest

from scripts.screen_program_o_exact_transducer import make_tape
from supply_chain.program_o_full_des import (
    ProductTagLedger,
    product_demand_tape,
    run_program_o_full_des_episode,
)
from supply_chain.program_o_full_des_transducer import (
    MATRIX_KEYS,
    direct_full_des_vector,
    extract_full_des_skeleton,
    full_action_calendars,
    simulate_full_des_frontier,
)

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = json.loads(
    (ROOT / "contracts/program_o_full_des_hpi_translation_v1.json").read_text()
)
SCHEDULER = CONTRACT["action"]["within_week_schedulers"][
    CONTRACT["action"]["primary_scheduler"]
]


def test_full_des_contract_is_hpi_only_and_seed_blocks_are_fail_closed():
    assert CONTRACT["base_des"]["risks_enabled"] is False
    assert CONTRACT["product_physics"]["setup_hours"] == 0
    assert CONTRACT["product_physics"]["substitution_fraction"] == 0
    assert CONTRACT["action"]["complete_open_loop_frontier"] == 65536
    assert CONTRACT["metric"]["primary"] == "ret_excel_request_snapshot_v2"
    assert CONTRACT["metric"]["cobb_douglas"] == "outside contract"
    assert CONTRACT["tape_blocks"]["development"] == {
        "range": [7400049, 7400072],
        "status": "SEALED_NOT_OPENED_PENDING_IMPLEMENTATION_TESTS_AND_COMMIT",
    }
    assert CONTRACT["tape_blocks"]["validation"] == {
        "range": [7400073, 7400096],
        "status": "SEALED_NOT_AUTHORIZED_PENDING_DEVELOPMENT_PASS_AND_ADDITIVE_FREEZE",
    }
    assert CONTRACT["h_obs_authorized"] is False
    assert CONTRACT["learner_authorized"] is False
    assert CONTRACT["paper3_authorized"] is False


def test_full_des_frontier_is_complete_and_unique():
    calendars = full_action_calendars()
    assert calendars.shape == (65536, 8)
    assert len({tuple(row) for row in calendars.tolist()}) == 65536
    assert calendars[0].tolist() == [0] * 8
    assert calendars[-1].tolist() == [3] * 8


def run(calendar, *, fungible=False, seed=7400000):
    return run_program_o_full_des_episode(
        seed=seed,
        calendar=calendar,
        scheduler=SCHEDULER,
        regime_persistence=0.75,
        dominant_share=0.90,
        complete_substitution=fungible,
    )


def test_product_tape_matches_frozen_program_o_labels_and_hash():
    expected = make_tape(7400048, persistence=0.75, dominant_share=0.90)
    actual = product_demand_tape(7400048, regime_persistence=0.75, dominant_share=0.90)

    assert tuple(actual["regimes"]) == expected.regimes
    assert tuple(actual["order_products"]) == expected.order_products
    assert actual["sha256"] == expected.sha256


def test_product_tag_ledger_is_metadata_only_fifo_partition():
    ledger = ProductTagLedger()
    ledger.put("rations_sb", "P_C", 5000, lot_id="C1", created_at=0)
    ledger.put("rations_sb", "P_H", 5000, lot_id="H1", created_at=0)

    tokens = ledger.take_product("rations_sb", "P_H", 2500)
    ledger.put_slices("order_transit", tokens, now=1)
    ledger.remove_tokens("order_transit", tokens)
    ledger.put_slices("delivered", tokens, now=49)

    assert ledger.quantity("rations_sb", "P_C") == 5000
    assert ledger.quantity("rations_sb", "P_H") == 2500
    assert ledger.quantity("delivered", "P_H") == 2500
    assert ledger.quantity("order_transit") == 0


def test_direct_full_des_episode_uses_real_pipeline_and_closes_ledgers():
    sim, panel = run([2] * 8)

    assert sim.program_o_decision_start is not None
    assert panel["metrics"]["n_orders"] == 48
    assert panel["resources"]["committed_action_batch_slots"] == 24
    assert panel["resources"]["completed_action_batch_slots"] == 24
    assert panel["resources"]["gross_action_production_quantity"] == 120000
    assert panel["conservation"]["max_abs_product_residual"] <= 1e-8
    assert panel["conservation"]["max_abs_partition_residual"] <= 1e-8
    assert (
        abs(panel["conservation"]["aggregate_flow_ledger"]["ration_residual"]) <= 1e-8
    )
    assert all(
        event["oat_is_none"]
        for event in sim.program_o_order_route_events
        if event["event"] == "op9_reserved"
    )
    assert all(
        event["oat_is_none_before_finalize"]
        for event in sim.program_o_order_route_events
        if event["event"] == "post_op12_physical_delivery"
    )
    completed = [order for order in sim.orders if order.OATj is not None]
    assert completed
    assert all(
        float(order.OATj) >= float(order.op9_release_time) + 48 for order in completed
    )


def test_prefix_and_gross_resources_are_calendar_independent():
    sim_h, panel_h = run([0] * 8)
    sim_c, panel_c = run([3] * 8)

    assert panel_h["prefix_state_hash"] == panel_c["prefix_state_hash"]
    for key in (
        "committed_action_batch_slots",
        "completed_action_batch_slots",
        "gross_action_production_quantity",
        "charged_daily_dispatch_slots",
        "charged_downstream_vehicle_hours",
        "setup_hours",
    ):
        assert panel_h["resources"][key] == panel_c["resources"][key]
    assert sim_h.total_produced == sim_c.total_produced == 130000
    assert sim_h.total_raw_material_consumed == sim_c.total_raw_material_consumed
    assert panel_h["tape_sha256"] == panel_c["tape_sha256"]


def test_fungible_null_has_identical_aggregate_outcome_across_extremes():
    _, panel_h = run([0] * 8, fungible=True)
    _, panel_c = run([3] * 8, fungible=True)

    assert panel_h["aggregate_state_hash"] == panel_c["aggregate_state_hash"]
    assert panel_h["metrics"] == panel_c["metrics"]
    assert panel_h["resources"] == panel_c["resources"]


def test_nonfungible_full_des_decision_is_physically_live():
    _, panel_h = run([0] * 8)
    _, panel_c = run([3] * 8)

    assert panel_h["metrics"]["ret_excel"] != pytest.approx(
        panel_c["metrics"]["ret_excel"], abs=1e-12
    )
    assert panel_h["tape_sha256"] == panel_c["tape_sha256"]


PARITY_CALENDARS = (
    [0] * 8,
    [1] * 8,
    [2] * 8,
    [3] * 8,
    [0, 3] * 4,
    [3, 0] * 4,
    [0, 0, 3, 3, 0, 0, 3, 3],
    [3, 3, 0, 0, 3, 3, 0, 0],
    [0, 1, 2, 3, 0, 1, 2, 3],
    [3, 2, 1, 0, 3, 2, 1, 0],
    [3, 0, 2, 1, 3, 2, 0, 1],
    [1, 3, 0, 2, 2, 0, 3, 1],
)


@pytest.mark.parametrize("seed", (7400047, 7400048))
@pytest.mark.parametrize("calendar", PARITY_CALENDARS)
def test_full_des_transducer_matches_direct_simpy(calendar, seed):
    skeleton, _ = extract_full_des_skeleton(
        seed=seed,
        scheduler=SCHEDULER,
        regime_persistence=0.75,
        dominant_share=0.90,
    )
    direct_sim, direct = run(calendar, seed=seed)
    expected = direct_full_des_vector(direct_sim, direct)
    transduced = simulate_full_des_frontier(
        skeleton=skeleton,
        scheduler=SCHEDULER,
        calendars=[calendar],
        complete_substitution=False,
    )
    assert tuple(expected) == MATRIX_KEYS
    for key, value in expected.items():
        assert transduced[key][0] == pytest.approx(value, abs=1e-10), key
