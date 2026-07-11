from __future__ import annotations

from scripts.run_program_d_d1_ret_tree import FEATURE_NAMES, RULES, vector


def observable_row() -> dict[str, float | str]:
    return {
        "sb_inventory": 1.0,
        "queue_qty": 2.0,
        "queue_count": 3.0,
        "queue_occupancy": 0.5,
        "contingent_share": 0.25,
        "size_p25": 4.0,
        "size_p50": 5.0,
        "size_p75": 6.0,
        "age_p25": 7.0,
        "age_p50": 8.0,
        "age_p75": 9.0,
        "oldest_age": 10.0,
        "in_transit": 11.0,
        "op9_down": 0.0,
        "op10_down": 1.0,
        "op11_down": 0.0,
        "op12_down": 0.0,
        "recent_demand": 12.0,
        "recent_delivered": 13.0,
        "recent_fill": 0.9,
        "operational_day": 4.0,
        "prior_rule": "spt_flat",
    }


def test_vector_is_fixed_and_one_hot() -> None:
    values = vector(observable_row())
    assert len(values) == len(FEATURE_NAMES)
    assert sum(values[-len(RULES):]) == 1.0
    assert values[-len(RULES) + RULES.index("spt_flat")] == 1.0


def test_vector_ignores_privileged_fields() -> None:
    row = observable_row()
    baseline = vector(row)
    row.update({"family": "R2", "risk_level": "severe", "future_repair": 999.0})
    assert vector(row) == baseline
