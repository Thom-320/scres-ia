from __future__ import annotations

from types import SimpleNamespace

from supply_chain.ret_thesis import (
    compute_order_level_ret_excel_formula,
    compute_order_level_ret_excel_visible_ledger,
)


def order(**kwargs):
    defaults = dict(
        j=1, OPTj=0.0, OATj=48.0, CTj=48.0, LTj=48.0,
        APj=0.0, RPj=0.0, DPj=0.0, quantity=2500.0,
        remaining_qty=0.0, lost=False, lost_time=None,
        backorder=False, ret_risk_indicators={}, metrics_excluded=False,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_visible_contract_omits_lost_row_but_keeps_original_j() -> None:
    completed = order(j=1)
    lost = order(j=2, OPTj=100.0, OATj=None, CTj=None, remaining_qty=2500.0,
                 lost=True, lost_time=160.0)
    full = compute_order_level_ret_excel_formula([completed, lost], current_time=200.0)
    visible = compute_order_level_ret_excel_visible_ledger(
        [completed, lost], current_time=200.0
    )
    assert full["mean_ret_excel"] == 0.5
    assert visible["mean_ret_excel"] == 1.0
    assert visible["n_visible_rows"] == 1
    assert visible["n_omitted_rows"] == 1
    assert visible["ret_rows"][0]["j"] == 1


def test_visible_contract_preserves_unclipped_workbook_recovery_value() -> None:
    fast_recovery = order(
        j=7, OATj=60.0, CTj=60.0, RPj=0.25, DPj=60.0,
        ret_risk_indicators={"R22": 1.0},
    )
    visible = compute_order_level_ret_excel_visible_ledger([fast_recovery])
    assert visible["mean_ret_excel"] == 2.0
    assert min(1.0, max(0.0, visible["ret_values"][0])) == 1.0
