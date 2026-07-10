from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from openpyxl import Workbook
import pytest

from supply_chain.garrido_replication import (
    DEFAULT_RAW_WORKBOOKS,
    audit_raw_garrido_formula,
    load_raw_garrido_targets,
)
from supply_chain.ret_thesis import (
    compute_order_level_ret_excel_formula,
    compute_order_level_ret_excel_visible_ledger,
    compute_ret_per_order_excel_formula,
)
from supply_chain.config import BACKORDER_QUEUE_CAP, GARRIDO_R14_RET_PERIOD_HOURS
from supply_chain.supply_chain import MFSCSimulation, OrderRecord, RiskEvent


def _order(**kwargs):
    data = {
        "j": 1,
        "OPTj": 0.0,
        "OATj": 48.0,
        "CTj": 48.0,
        "LTj": 48.0,
        "APj": 0.0,
        "RPj": 0.0,
        "DPj": 0.0,
        "lost": False,
        "backorder": False,
        "remaining_qty": 0.0,
    }
    data.update(kwargs)
    return SimpleNamespace(**data)


def test_excel_ret_autotomy_uses_ap_over_lt_without_clip_or_ct_gate() -> None:
    order = _order(APj=50.0, CTj=60.0, LTj=48.0)

    value, case = compute_ret_per_order_excel_formula(
        order,
        j=1,
        cumulative_backorders=0,
        cumulative_unattended=0,
        risk_active=True,
    )

    assert case == "excel_autotomy"
    assert value == pytest.approx(50.0 / 48.0)


def test_excel_ret_recovery_matches_raw_workbook_branch() -> None:
    order = _order(APj=0.0, RPj=90.1992, DPj=96.0)

    value, case = compute_ret_per_order_excel_formula(
        order,
        j=1,
        cumulative_backorders=0,
        cumulative_unattended=0,
        risk_active=True,
    )

    assert case == "excel_recovery"
    assert value == pytest.approx(0.5 * (1.0 / 90.1992))


def test_visible_ledger_uses_sparse_j_and_time_varying_bt_ut() -> None:
    delivered_late = _order(j=1, OPTj=0.0, OATj=100.0, CTj=100.0)
    lost = _order(
        j=2,
        OPTj=24.0,
        OATj=None,
        CTj=None,
        lost=True,
        lost_time=90.0,
    )
    delivered_at_promise = _order(j=3, OPTj=48.0, OATj=96.0, CTj=48.0)

    result = compute_order_level_ret_excel_visible_ledger(
        [delivered_late, lost, delivered_at_promise],
        current_time=120.0,
    )

    assert result["n_generated_orders"] == 3
    assert result["n_visible_rows"] == 2
    assert result["n_omitted_rows"] == 1
    # At j=3/OAT=96: j=1 is pending beyond LT and j=2 is already unattended.
    assert result["final_backorders"] == 1
    assert result["final_unattended"] == 1
    assert result["mean_ret_excel"] == pytest.approx((0.0 + 1.0 / 3.0) / 2.0)


def test_excel_ret_uses_explicit_order_risk_indicators() -> None:
    order = _order(ret_risk_indicators={"R14": 3.0})

    value, case = compute_ret_per_order_excel_formula(
        order,
        j=1,
        cumulative_backorders=0,
        cumulative_unattended=0,
    )

    assert case == "excel_risk_no_recovery"
    assert value == 0.0


def test_excel_fill_rate_branch_caps_backlog_like_garrido_ledger() -> None:
    order = _order()

    value, case = compute_ret_per_order_excel_formula(
        order,
        j=1000,
        cumulative_backorders=1000,
        cumulative_unattended=5,
        risk_active=False,
    )

    assert case == "excel_fill_rate"
    assert value == pytest.approx(1.0 - ((BACKORDER_QUEUE_CAP + 5) / 1000))


def test_quantity_risk_attribution_marks_later_orders() -> None:
    sim = MFSCSimulation(risks_enabled=False)
    event = RiskEvent(
        "R14",
        start_time=0.0,
        end_time=0.0,
        duration=0.0,
        affected_ops=[7],
        magnitude=2500.0,
        unit="defective_products",
    )
    sim._add_ret_quantity_risk(event)
    order = OrderRecord(
        j=1,
        OPTj=24.0,
        OATj=72.0,
        CTj=48.0,
        quantity=2500.0,
        remaining_qty=0.0,
    )

    sim._set_order_ret_indicators(order)

    assert order.ret_risk_indicators["R14"] == pytest.approx(1.0)
    assert order.APj == pytest.approx(
        min(GARRIDO_R14_RET_PERIOD_HOURS, order.LTj)
    )

    delayed_order = OrderRecord(
        j=2,
        OPTj=96.0,
        OATj=168.0,
        CTj=72.0,
        quantity=2500.0,
        remaining_qty=0.0,
    )

    sim._set_order_ret_indicators(delayed_order)

    assert delayed_order.ret_risk_indicators["R14"] == pytest.approx(1.0)
    assert delayed_order.RPj == pytest.approx(GARRIDO_R14_RET_PERIOD_HOURS)


def test_excel_order_level_ret_uses_running_order_count_fill_branch() -> None:
    orders = [
        _order(j=1, OATj=48.0, CTj=48.0),
        _order(j=2, OATj=100.0, CTj=100.0),
        _order(j=3, OATj=None, CTj=None, lost=True),
    ]

    summary = compute_order_level_ret_excel_formula(orders, current_time=120.0)

    # Row-wise no-risk fill values:
    # j1: 1 - (0 + 0)/1 = 1
    # j2: 1 - (1 + 0)/2 = 0.5
    # j3: unfulfilled/lost order with no OATj = 0
    assert summary["case_counts"]["excel_fill_rate"] == 2
    assert summary["case_counts"]["excel_unfulfilled"] == 1
    assert summary["mean_ret_excel"] == pytest.approx((1.0 + 0.5 + 0.0) / 3.0)


def test_excel_order_level_ret_can_preserve_original_workbook_j() -> None:
    orders = [
        _order(j=10, OATj=48.0, CTj=48.0),
        _order(j=20, OATj=100.0, CTj=100.0),
    ]

    row_index_summary = compute_order_level_ret_excel_formula(
        orders, current_time=120.0
    )
    workbook_j_summary = compute_order_level_ret_excel_formula(
        orders, current_time=120.0, j_source="order_j"
    )

    assert row_index_summary["mean_ret_excel"] == pytest.approx((1.0 + 0.5) / 2.0)
    assert workbook_j_summary["mean_ret_excel"] == pytest.approx((1.0 + (19.0 / 20.0)) / 2.0)


def test_sim_order_level_ret_accepts_forensic_ledger_and_j_source() -> None:
    sim = MFSCSimulation(horizon=0.0, risks_enabled=False)
    orders = [
        OrderRecord(j=10, OPTj=0.0, OATj=48.0, CTj=48.0, quantity=1.0),
        OrderRecord(j=20, OPTj=0.0, OATj=100.0, CTj=100.0, quantity=1.0),
    ]

    summary = sim.compute_order_level_ret(orders=orders, j_source="order_j")

    assert summary["j_source"] == "order_j"
    assert summary["n_orders"] == 2
    assert summary["mean_ret_excel_formula"] == pytest.approx(
        (1.0 + (19.0 / 20.0)) / 2.0
    )


def test_garrido_target_extractor_detects_shifted_header(tmp_path: Path) -> None:
    workbook_path = tmp_path / "mini_raw.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "CF2"
    headers = [
        "Cfi",
        "Cf2",
        None,
        "Q",
        "j",
        "OPTj",
        "OATj",
        "CTj",
        "LT",
        "\u2211Bt",
        "APj",
        "RPj",
        "DPj",
        "R11_1",
        "R11_2",
        "R12",
        "R13",
        "R14",
        "\u2211Ut",
        "OP9",
        "ReT",
        "\u0394ReT",
    ]
    ws.append([None] * len(headers))
    ws.append(headers)
    ws.append(["Seed", 123, None, 2500, 1, 24, 72, 48, 48, 0, 0, 100, 48, 0, 0, 1, 0, 0, 0, 42, 0.005, 0])
    ws.append(["Warm-up period", 888, None, 2600, 2, 48, 96, 48, 48, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 43, 0.5, 0.495])
    wb.save(workbook_path)

    targets = load_raw_garrido_targets([workbook_path])
    target = targets[2]

    assert target.header_row == 2
    assert target.seed == 123
    assert target.warmup_hours == 888
    assert target.n_orders == 2
    assert target.risk_columns == ("R11_1", "R11_2", "R12", "R13", "R14")
    assert audit_raw_garrido_formula(targets)["total_mismatches"] == 0


def test_local_garrido_raw_targets_recalculate_if_available() -> None:
    if not all(path.exists() for path in DEFAULT_RAW_WORKBOOKS):
        pytest.skip("Local Garrido raw workbooks are not available.")

    targets = load_raw_garrido_targets(DEFAULT_RAW_WORKBOOKS)
    audit = audit_raw_garrido_formula(targets)

    assert sorted(targets) == list(range(1, 21))
    assert targets[2].header_row == 2
    assert audit["total_rows"] == 47_546
    assert audit["total_mismatches"] == 0
    assert audit["max_abs_diff"] == 0.0


def test_excel_order_tape_reproduces_order_count_q_and_optj() -> None:
    tape = [
        {"j": 1, "OPTj": 24.0, "Q": 2501.0},
        {"j": 2, "OPTj": 48.0, "Q": 2602.0, "contingent": True},
        {"j": 3, "OPTj": 72.0, "Q": 2499.0},
    ]

    sim = MFSCSimulation(
        horizon=96.0,
        risks_enabled=False,
        demand_source="excel_order_tape",
        excel_order_tape=tape,
    ).run()

    assert [order.j for order in sim.orders] == [1, 2, 3]
    assert [order.OPTj for order in sim.orders] == pytest.approx([24.0, 48.0, 72.0])
    assert [order.quantity for order in sim.orders] == pytest.approx([2501.0, 2602.0, 2499.0])
    assert sim.orders[1].contingent is True


def test_excel_risk_tape_sets_order_periods_and_indicators() -> None:
    tape = [
        {
            "j": 1,
            "OPTj": 24.0,
            "Q": 2500.0,
            "ret_attribution": {
                "APj": 0.0,
                "RPj": 90.0,
                "DPj": 96.0,
                "LTj": 48.0,
                "risk_values": {"R12": 1.0, "R14": 0.0},
            },
        }
    ]

    sim = MFSCSimulation(
        horizon=120.0,
        risks_enabled=False,
        demand_source="excel_order_tape",
        excel_order_tape=tape,
        risk_attribution_source="excel_risk_tape",
    )
    sim.rations_theatre.put(2500.0)
    sim.run()

    assert len(sim.orders) == 1
    assert sim.orders[0].APj == 0.0
    assert sim.orders[0].RPj == 90.0
    assert sim.orders[0].DPj == 96.0
    assert sim.orders[0].ret_risk_indicators == {"R12": 1.0}
    ret = sim.compute_order_level_ret()
    assert ret["case_counts_excel_formula"]["excel_recovery"] == 1
    assert ret["mean_ret_excel_formula"] == pytest.approx(0.5 / 90.0)


def test_garrido_replication_options_validate_and_are_deterministic() -> None:
    tape = [{"j": 1, "OPTj": 24.0, "Q": 2500.0}]
    with pytest.raises(ValueError, match="demand_source"):
        MFSCSimulation(demand_source="not_real", excel_order_tape=tape)
    with pytest.raises(ValueError, match="seed_stream_mode"):
        MFSCSimulation(seed_stream_mode="forked")
    with pytest.raises(ValueError, match="risk_attribution_source"):
        MFSCSimulation(risk_attribution_source="not_real")
    with pytest.raises(ValueError, match="requires demand_source"):
        MFSCSimulation(risk_attribution_source="excel_risk_tape")
    with pytest.raises(ValueError, match="ret_attribution"):
        MFSCSimulation(
            demand_source="excel_order_tape",
            excel_order_tape=tape,
            risk_attribution_source="excel_risk_tape",
        )
    with pytest.raises(ValueError, match="requires excel_order_tape"):
        MFSCSimulation(demand_source="excel_order_tape")

    sim_a = MFSCSimulation(
        seed=77,
        horizon=48.0,
        risks_enabled=False,
        demand_source="excel_order_tape",
        excel_order_tape=tape,
        seed_stream_mode="split",
    ).run()
    sim_b = MFSCSimulation(
        seed=77,
        horizon=48.0,
        risks_enabled=False,
        demand_source="excel_order_tape",
        excel_order_tape=tape,
        seed_stream_mode="split",
    ).run()

    assert [(order.j, order.OPTj, order.quantity) for order in sim_a.orders] == [
        (order.j, order.OPTj, order.quantity) for order in sim_b.orders
    ]
