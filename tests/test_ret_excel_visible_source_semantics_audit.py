from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.audit_ret_excel_visible_source_semantics import (
    DEFAULT_OUTPUT,
    build_audit,
)
from supply_chain.garrido_replication import DEFAULT_RAW_WORKBOOKS
from supply_chain.ret_thesis import (
    compute_order_level_ret_excel_visible_ledger,
    compute_ret_per_order_excel_formula,
)


def _order(**changes):
    values = {
        "j": 1,
        "OPTj": 0.0,
        "OATj": 48.0,
        "CTj": 48.0,
        "LTj": 48.0,
        "APj": 0.0,
        "RPj": 0.0,
        "DPj": 0.0,
        "quantity": 1.0,
        "remaining_qty": 0.0,
        "lost": False,
        "lost_time": None,
        "ret_risk_indicators": {},
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_frozen_request_snapshot_score_is_invariant_to_later_oat() -> None:
    order = _order(j=17, OATj=200.0, CTj=200.0)
    first, first_case = compute_ret_per_order_excel_formula(
        order,
        j=17,
        cumulative_backorders=4,
        cumulative_unattended=2,
        risk_active=False,
    )
    order.OATj = 800.0
    order.CTj = 800.0
    second, second_case = compute_ret_per_order_excel_formula(
        order,
        j=17,
        cumulative_backorders=4,
        cumulative_unattended=2,
        risk_active=False,
    )
    assert first_case == second_case == "excel_fill_rate"
    assert first == second == pytest.approx(1.0 - 6.0 / 17.0)


def test_current_visible_ledger_uses_completion_time_not_frozen_request_state() -> None:
    early = _order(j=1, OPTj=0.0, OATj=100.0, CTj=100.0)
    later = _order(j=2, OPTj=24.0, OATj=96.0, CTj=72.0)
    score_a = compute_order_level_ret_excel_visible_ledger([early, later])[
        "mean_ret_excel"
    ]
    early.OATj = 70.0
    early.CTj = 70.0
    score_b = compute_order_level_ret_excel_visible_ledger([early, later])[
        "mean_ret_excel"
    ]
    assert score_a != score_b


def test_machine_artifact_is_fail_closed() -> None:
    payload = json.loads(DEFAULT_OUTPUT.read_text())
    assert payload["status"].endswith("OAT_BT_UT_RECONSTRUCTION_NOT_SOURCE_VALIDATED")
    assert payload["no_tape_access"]["virgin_tapes_accessed"] is False
    assert payload["scientific_disposition"]["current_ret_excel_visible_v1"] == (
        "HOLD_FOR_LEDGER_SEMANTICS_REPAIR"
    )
    assert payload["proposed_ret_excel_visible_v2"]["status"] == (
        "PROPOSED_NOT_IMPLEMENTED"
    )


def test_local_workbook_source_semantics_if_available() -> None:
    if not all(path.is_file() for path in DEFAULT_RAW_WORKBOOKS):
        pytest.skip("local Garrido raw workbooks are unavailable")
    payload = build_audit()
    formula = payload["workbook_formula_replay"]
    totals = payload["timing_and_population_evidence"]["totals"]
    assert formula["formula_rows"] == 47_546
    assert formula["mismatches"] == 0
    assert formula["max_abs_diff"] == 0.0
    assert totals["OATj_adjacent_inversions_in_j_order"] == 16_391
    assert totals["sumUt_adjacent_decreases_in_j_order"] == 0
    assert totals["sumUt_adjacent_decreases_in_OATj_order"] == 8_340
    assert totals["same_exact_OATj_group_count"] == 0
    assert totals["visible_rows_only_OAT_Bt_matches"] == 590
    assert totals["visible_rows_only_OAT_Bt_total"] == 47_546
    assert totals["initial_contiguous_j_prefix_rows"] == 687
    assert totals["initial_contiguous_j_prefix_matches"] == 331
