from types import SimpleNamespace

import pytest
import simpy

from supply_chain.garrido_replication import (
    DEFAULT_RAW_WORKBOOKS,
    load_raw_garrido_targets,
)
from supply_chain.ret_thesis import (
    compute_order_level_ret_excel_request_snapshot_ledger,
    compute_order_level_ret_excel_visible_ledger,
)
from supply_chain.supply_chain import BACKORDER_QUEUE_CAP, MFSCSimulation, OrderRecord


def order(**kwargs):
    defaults = dict(
        j=1,
        OPTj=0.0,
        OATj=48.0,
        CTj=48.0,
        LTj=48.0,
        APj=0.0,
        RPj=0.0,
        DPj=0.0,
        quantity=2500.0,
        remaining_qty=0.0,
        lost=False,
        lost_time=None,
        ret_risk_indicators={},
        ret_bt_at_request=None,
        ret_ut_at_request=None,
        ret_ledger_snapshot_time=None,
        ret_ledger_event_sequence=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_request_snapshot_orders_rows_by_j_opt_not_completion():
    orders = [
        order(j=1, OPTj=0.0, OATj=100.0, CTj=100.0),
        order(j=2, OPTj=24.0, OATj=200.0, CTj=176.0),
        order(j=3, OPTj=60.0, OATj=80.0, CTj=20.0),
    ]

    result = compute_order_level_ret_excel_request_snapshot_ledger(orders)

    assert [row["j"] for row in result["ret_rows"]] == [1, 2, 3]
    assert [row["sum_bt"] for row in result["ret_rows"]] == [0, 0, 1]
    assert result["mean_ret_excel"] == pytest.approx((1.0 + 1.0 + 2 / 3) / 3)
    assert (
        result["mean_ret_excel"]
        != compute_order_level_ret_excel_visible_ledger(orders)["mean_ret_excel"]
    )


def test_request_snapshot_counts_prior_lost_order_at_generation_time():
    lost = order(
        j=1,
        OPTj=0.0,
        OATj=None,
        CTj=None,
        lost=True,
        lost_time=50.0,
    )
    visible = order(j=2, OPTj=60.0, OATj=100.0, CTj=40.0)

    result = compute_order_level_ret_excel_request_snapshot_ledger([lost, visible])

    assert result["n_visible_rows"] == 1
    assert result["ret_rows"][0]["sum_bt"] == 0
    assert result["ret_rows"][0]["sum_ut"] == 1
    assert result["ret_rows"][0]["ret"] == pytest.approx(0.5)


def test_captured_request_snapshot_overrides_retrospective_reconstruction():
    visible = order(
        j=7,
        OPTj=100.0,
        OATj=180.0,
        CTj=80.0,
        ret_bt_at_request=2,
        ret_ut_at_request=1,
        ret_ledger_snapshot_time=100.0,
        ret_ledger_event_sequence=7,
    )

    result = compute_order_level_ret_excel_request_snapshot_ledger([visible])

    row = result["ret_rows"][0]
    assert row["snapshot_source"] == "captured_at_request"
    assert row["sum_bt"] == 2
    assert row["sum_ut"] == 1
    assert row["snapshot_time"] == 100.0
    assert row["snapshot_event_sequence"] == 7
    assert row["ret"] == pytest.approx(1 - 3 / 7)


def test_request_snapshot_preserves_sparse_population_and_unclipped_risk_branch():
    completed = order(
        j=7,
        OPTj=100.0,
        OATj=160.0,
        CTj=60.0,
        RPj=0.25,
        DPj=60.0,
        ret_risk_indicators={"R22": 1.0},
        ret_bt_at_request=3,
        ret_ut_at_request=2,
        ret_ledger_snapshot_time=100.0,
        ret_ledger_event_sequence=7,
    )
    lost = order(
        j=8,
        OPTj=124.0,
        OATj=None,
        CTj=None,
        lost=True,
        lost_time=200.0,
    )

    result = compute_order_level_ret_excel_request_snapshot_ledger(
        [completed, lost]
    )

    assert result["contract_version"] == "ret_excel_request_snapshot_v2"
    assert result["n_generated_orders"] == 2
    assert result["n_visible_rows"] == 1
    assert result["n_omitted_rows"] == 1
    assert result["mean_ret_excel"] == 2.0


def test_request_snapshot_aggregator_reproduces_all_local_workbook_cells():
    if not all(path.exists() for path in DEFAULT_RAW_WORKBOOKS):
        pytest.skip("Local Garrido raw workbooks are not available.")
    targets = load_raw_garrido_targets(DEFAULT_RAW_WORKBOOKS)
    compared = 0
    for target in targets.values():
        orders = [
            order(
                j=row.j,
                OPTj=row.optj,
                OATj=row.oatj,
                CTj=row.ctj,
                LTj=row.ltj,
                APj=row.apj,
                RPj=row.rpj,
                DPj=row.dpj,
                quantity=row.q,
                ret_risk_indicators=row.risk_values,
                ret_bt_at_request=int(row.sum_bt),
                ret_ut_at_request=int(row.sum_ut),
                ret_ledger_snapshot_time=row.optj,
                ret_ledger_event_sequence=row.j,
            )
            for row in target.orders
        ]
        result = compute_order_level_ret_excel_request_snapshot_ledger(orders)
        assert result["n_visible_rows"] == len(target.orders)
        for actual, expected in zip(
            result["ret_values"],
            (row.ret for row in target.orders),
            strict=True,
        ):
            assert actual == pytest.approx(expected, abs=1e-12)
            compared += 1
    assert compared == 47_546


def test_native_des_captures_request_snapshots_before_queue_entry():
    simulation = MFSCSimulation(
        seed=7,
        horizon=672,
        risks_enabled=False,
        warmup_trigger="production",
    )
    simulation.run()

    assert simulation.orders
    assert [order.ret_ledger_event_sequence for order in simulation.orders] == list(
        range(1, len(simulation.orders) + 1)
    )
    assert all(
        order.ret_ledger_snapshot_time == pytest.approx(order.OPTj)
        for order in simulation.orders
    )
    assert all(order.ret_bt_at_request is not None for order in simulation.orders)
    assert all(order.ret_ut_at_request is not None for order in simulation.orders)
    assert simulation.orders[0].ret_bt_at_request == 0
    assert simulation.orders[2].ret_bt_at_request == 1

    ledger = compute_order_level_ret_excel_request_snapshot_ledger(
        simulation.orders
    )
    assert ledger["contract_version"] == "ret_excel_request_snapshot_v2"
    assert all(
        row["snapshot_source"] == "captured_at_request"
        for row in ledger["ret_rows"]
    )


def _native_snapshot_sim(prior_orders):
    simulation = MFSCSimulation(
        seed=17,
        horizon=1,
        risks_enabled=False,
        warmup_trigger="production",
        order_fulfillment_mode="op9_linked",
    )
    simulation.orders = list(prior_orders)
    simulation.pending_backorders = []
    return simulation


def test_native_request_snapshot_uses_half_open_time_interval_not_queue_flag():
    prior = OrderRecord(
        j=99,
        OPTj=0.0,
        LTj=48.0,
        quantity=2500.0,
        remaining_qty=2500.0,
        backorder=False,
    )
    simulation = _native_snapshot_sim([prior])

    assert simulation._ret_request_snapshot_counts_at(47.999) == (0, 0)
    assert simulation._ret_request_snapshot_counts_at(48.0) == (1, 0)
    prior.backorder = True
    assert simulation._ret_request_snapshot_counts_at(49.0) == (1, 0)
    # Dispatch may remove an order from the live queue while it is in flight;
    # OATj, not queue membership, closes the request-ledger interval.
    assert prior not in simulation.pending_backorders
    prior.OATj = 60.0
    assert simulation._ret_request_snapshot_counts_at(59.999) == (1, 0)
    assert simulation._ret_request_snapshot_counts_at(60.0) == (0, 0)


def test_native_request_snapshot_loss_boundary_cap_and_exclusions():
    lost = OrderRecord(
        j=1,
        OPTj=0.0,
        LTj=48.0,
        quantity=2500.0,
        remaining_qty=2500.0,
        lost=True,
        lost_time=60.0,
    )
    excluded = OrderRecord(
        j=2,
        OPTj=0.0,
        LTj=48.0,
        quantity=2500.0,
        remaining_qty=2500.0,
        metrics_excluded=True,
    )
    active = [
        OrderRecord(
            j=10 + idx,
            OPTj=0.0,
            LTj=48.0,
            quantity=2500.0,
            remaining_qty=2500.0,
        )
        for idx in range(BACKORDER_QUEUE_CAP + 5)
    ]
    simulation = _native_snapshot_sim([lost, excluded, *active])

    assert simulation._ret_request_snapshot_counts_at(59.0) == (
        BACKORDER_QUEUE_CAP,
        0,
    )
    assert simulation._ret_request_snapshot_counts_at(60.0) == (
        BACKORDER_QUEUE_CAP,
        1,
    )


@pytest.mark.parametrize("completion_registered_first", [False, True])
def test_native_request_snapshot_is_scheduler_invariant_at_deadline(
    completion_registered_first,
):
    prior = OrderRecord(
        j=1,
        OPTj=0.0,
        LTj=48.0,
        quantity=2500.0,
        remaining_qty=2500.0,
    )
    simulation = _native_snapshot_sim([prior])
    simulation.env = simpy.Environment()
    observed = []

    def completion():
        yield simulation.env.timeout(48.0)
        prior.OATj = 48.0
        prior.CTj = 48.0
        prior.remaining_qty = 0.0

    def snapshot():
        yield simulation.env.timeout(48.0)
        # Mirrors the zero-time phase barrier at the start of
        # _place_demand_order without entering the physical order path.
        yield simulation.env.timeout(0)
        observed.append(
            simulation._ret_request_snapshot_counts_at(simulation.env.now)
        )

    processes = (completion, snapshot) if completion_registered_first else (snapshot, completion)
    for process in processes:
        simulation.env.process(process())
    simulation.env.run(until=48.0001)

    assert observed == [(0, 0)]


def test_fallback_snapshot_uses_request_time_not_nonmonotone_order_id():
    prior = order(
        j=99,
        OPTj=0.0,
        OATj=100.0,
        CTj=100.0,
        ret_bt_at_request=None,
        ret_ut_at_request=None,
    )
    row = order(
        j=1,
        OPTj=60.0,
        OATj=108.0,
        CTj=48.0,
        ret_bt_at_request=None,
        ret_ut_at_request=None,
    )

    result = compute_order_level_ret_excel_request_snapshot_ledger([row, prior])

    row_result = next(item for item in result["ret_rows"] if item["j"] == 1)
    assert row_result["sum_bt"] == 1
