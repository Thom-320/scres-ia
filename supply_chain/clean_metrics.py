"""Treatment-window outcome metric (corrected retained-vs-reset protocol).

Fixes the contaminated-metric confounder flagged in the 2026-06-24 audit: the
default order-level ReT aggregates ALL orders, including those placed during warm-up
(under I0,S1) and before the policy's decision takes effect. Those pre-treatment
orders carry inherited backlog that no evaluated action could have avoided, which
compresses the difference between policies.

This module recomputes the thesis-exact order-level ReT (Garrido-Rios 2017,
Eq. 5.1-5.5) using ONLY orders with ``OPTj >= treatment_start`` (default: end of
warm-up). It reuses the same aggregation as the env so the numbers are comparable.
"""

from __future__ import annotations

from typing import Any

from .config import THESIS_FAITHFUL_PROTOCOL
from .ret_thesis import (
    compute_fill_rate_from_orders,
    compute_order_level_ret_excel_formula,
    compute_order_level_ret as _aggregate_order_ret,
)


def treatment_filtered_order_ret(
    sim: Any, *, treatment_start: float | None = None
) -> dict[str, Any]:
    """Order-level ReT computed only on post-treatment orders.

    ``treatment_start`` defaults to ``sim.warmup_time`` (the policy acts after
    warm-up). Fill rate is recomputed on the filtered orders so it matches the
    thesis Eq. 5.4 definition (1 - (Bt+Ut)/Dt) on the treatment window only.
    """
    if treatment_start is None:
        treatment_start = float(getattr(sim, "warmup_time", 0.0))

    all_orders = list(getattr(sim, "orders", []))
    kept = [o for o in all_orders if float(o.OPTj) >= treatment_start]
    n_pre = len(all_orders) - len(kept)
    dt = len(kept)
    if dt == 0:
        return {
            "mean_ret": float("nan"),
            "mean_ret_excel_formula": float("nan"),
            "mean_ret_text_formula": float("nan"),
            "fill_rate_order_level": float("nan"),
            "case_counts": {},
            "case_counts_excel_formula": {},
            "case_counts_text_formula": {},
            "n_orders": 0,
            "n_orders_pre_treatment": n_pre,
            "treatment_start": float(treatment_start),
        }

    sim_env = getattr(sim, "env", None)
    current_time = None if sim_env is None else float(getattr(sim_env, "now", 0.0))
    fill = compute_fill_rate_from_orders(kept, current_time=current_time)

    res = _aggregate_order_ret(
        kept, fill_rate=fill, ret_weights=THESIS_FAITHFUL_PROTOCOL["ret_weights"]
    )
    excel_res = compute_order_level_ret_excel_formula(
        kept, current_time=current_time
    )
    res["mean_ret_text_formula"] = res["mean_ret"]
    res["case_counts_text_formula"] = res["case_counts"]
    res["mean_ret"] = excel_res["mean_ret_excel"]
    res["case_counts"] = excel_res["case_counts"]
    res["mean_ret_excel_formula"] = excel_res["mean_ret_excel"]
    res["case_counts_excel_formula"] = excel_res["case_counts"]
    res["treatment_start"] = float(treatment_start)
    res["n_orders_pre_treatment"] = n_pre
    return res
