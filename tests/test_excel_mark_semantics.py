from __future__ import annotations

from supply_chain.garrido_replication import GarridoCFTarget, GarridoOrderTarget
from scripts.audit_garrido_excel_mark_semantics import superset_violations


def _order(j: int, opt: float, oat: float, mark: float) -> GarridoOrderTarget:
    return GarridoOrderTarget(
        source_file="synthetic",
        sheet="CF1",
        cfi=1,
        row_index=j,
        q=1.0,
        j=j,
        optj=opt,
        oatj=oat,
        ctj=oat - opt,
        ltj=48.0,
        sum_bt=0.0,
        apj=0.0,
        rpj=0.0,
        dpj=0.0,
        risk_values={"R13": mark},
        sum_ut=0.0,
        op9=0.0,
        ret=0.0,
        delta_ret=0.0,
    )


def _target(orders: tuple[GarridoOrderTarget, ...]) -> GarridoCFTarget:
    return GarridoCFTarget(
        source_file="synthetic",
        sheet="CF1",
        cfi=1,
        seed=1,
        warmup_hours=0.0,
        header_row=1,
        risk_columns=("R13",),
        orders=orders,
    )


def test_contained_mark_missing_from_container_is_violation() -> None:
    result = superset_violations(
        _target((_order(1, 0.0, 20.0, 0.0), _order(2, 5.0, 10.0, 1.0)))
    )
    assert result["nested_pairs"] == 1
    assert result["families"]["R13"]["opportunities"] == 1
    assert result["families"]["R13"]["violations"] == 1


def test_monotone_marks_are_consistent_with_window_null() -> None:
    result = superset_violations(
        _target((_order(1, 0.0, 20.0, 1.0), _order(2, 5.0, 10.0, 1.0)))
    )
    assert result["families"]["R13"]["violations"] == 0
