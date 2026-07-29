"""Prospective, selectable repairs for the order-level Excel ReT endpoint.

The historical ``ret_excel`` implementation is intentionally untouched.  These
helpers call the official request-snapshot ledger and transform only cloned order
inputs or returned per-order values.  They exist for separately preregistered
prospective experiments.

``clip_0_1`` is the minimal range repair and makes no causal attribution claim.
``quantity_time_clip_0_1`` is a disclosed proxy sensitivity: when an order carries
an R14/R24 quantity-risk indicator, its effective recovery period is at least its
realised lateness, then every per-order value is clipped to [0, 1].  Presence of a
quantity-risk indicator does not prove that the quantity risk caused all lateness,
so this mode must not be labelled an exact causal repair.
"""
from __future__ import annotations

import copy
from typing import Any, Iterable

import numpy as np

from supply_chain.ret_thesis import (
    compute_order_level_ret_excel_request_snapshot_ledger,
)

REPAIR_MODES = (
    "canonical",
    "clip_0_1",
    "quantity_time_clip_0_1",
)
QUANTITY_RISKS = ("R14", "R24")


def _has_quantity_risk(order: Any) -> bool:
    indicators = dict(getattr(order, "ret_risk_indicators", {}) or {})
    return any(
        key == risk or key.startswith(f"{risk}_")
        for key in indicators
        for risk in QUANTITY_RISKS
    )


def repaired_ret_values(
    orders: Iterable[Any],
    *,
    current_time: float,
    mode: str,
) -> np.ndarray:
    """Return official-ledger per-order values under one named repair."""
    if mode not in REPAIR_MODES:
        raise ValueError(f"unknown ReT repair mode: {mode!r}")

    order_list = list(orders)
    if mode == "canonical":
        ledger = compute_order_level_ret_excel_request_snapshot_ledger(
            order_list,
            current_time=current_time,
        )
        return np.asarray(ledger["ret_values"], dtype=float)

    if mode == "clip_0_1":
        ledger = compute_order_level_ret_excel_request_snapshot_ledger(
            order_list,
            current_time=current_time,
        )
        return np.clip(np.asarray(ledger["ret_values"], dtype=float), 0.0, 1.0)

    patched: list[Any] = []
    for order in order_list:
        clone = copy.copy(order)
        if _has_quantity_risk(clone):
            rpj = float(getattr(clone, "RPj", 0.0) or 0.0)
            ctj = float(getattr(clone, "CTj", 0.0) or 0.0)
            ltj = float(getattr(clone, "LTj", 0.0) or 0.0)
            lateness = max(0.0, ctj - ltj)
            if lateness > 0.0:
                clone.RPj = max(rpj, lateness)
        patched.append(clone)
    ledger = compute_order_level_ret_excel_request_snapshot_ledger(
        patched,
        current_time=current_time,
    )
    return np.clip(np.asarray(ledger["ret_values"], dtype=float), 0.0, 1.0)


def repaired_ret_mean(
    orders: Iterable[Any],
    *,
    current_time: float,
    mode: str,
) -> float:
    values = repaired_ret_values(orders, current_time=current_time, mode=mode)
    return float(values.mean()) if values.size else 0.0
