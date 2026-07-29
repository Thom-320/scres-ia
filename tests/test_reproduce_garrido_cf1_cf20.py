from __future__ import annotations

import math

from scripts.reproduce_garrido_cf1_cf20 import pearson, summarize_group


def test_pearson_identity() -> None:
    assert math.isclose(pearson([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)


def test_summary_keeps_ret_and_order_errors_separate() -> None:
    rows = [
        {
            "excel_ret": 0.1,
            "sim_ret": 0.2,
            "excel_max_j": 100,
            "sim_orders": 110,
        },
        {
            "excel_ret": 0.3,
            "sim_ret": 0.2,
            "excel_max_j": 100,
            "sim_orders": 90,
        },
    ]
    summary = summarize_group(rows)
    assert math.isclose(summary["mean_bias"], 0.0, abs_tol=1e-15)
    assert math.isclose(summary["mean_absolute_error"], 0.1)
    assert math.isclose(summary["max_absolute_order_count_relative_error"], 0.1)
