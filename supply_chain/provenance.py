"""Calibration provenance stamp — which constants produced a number.

Measured 2026-07-30: of 845 result artifacts in the repository, **10** record the
fulfilment delay they were produced under, and all four sealed confirmation runs record
none. That gap is what makes both candidate futures impossible today. Operating two
calibration lines in parallel requires knowing which line produced which number, and
migrating to a different constant requires the same. 835 artifacts cannot say.

This module is the fix, and it is deliberately tiny: one call, embedded in every artifact
written from here on. It does not decide anything and it changes no behaviour.

    payload["calibration_provenance"] = calibration_stamp()

The stamp records the constants that are known to move `ret_excel` without moving the
physical trajectory, so a reader can tell at a glance whether two artifacts are
comparable at all.
"""
from __future__ import annotations

from typing import Any

from supply_chain.config import (
    GARRIDO_FULFILLMENT_DELAY_HOURS,
    HOURS_PER_WEEK,
    LEAD_TIME_PROMISE,
)

# The historical line. Not a thesis quantity: fitted on 2026-06-26 as "the smallest
# tested value that crosses the LT=48 cliff and reproduces the Garrido raw-Excel order
# of magnitude for ReT", and labelled there "provisional reproduction default, not a
# complete behavioral calibration".
HISTORICAL_FULFILMENT_DELAY_HOURS: float = 54.0

# Thesis §6.8.2 p.111: "the availability of finished products at this point allows
# troops to be supplied within a pre-set lead-time of 48 hours". This one IS a thesis
# quantity. `config.py` cites §6.3.4 for it, which is wrong -- that section is "Demand
# for combat rations" and defines no lead time.
THESIS_LEAD_TIME_SOURCE: str = "Garrido 2017 thesis §6.8.2 p.111"

# Reconstructed from the canonical workbooks on 2026-07-30, 43,360 rows over 19 sheets
# of Raw_data1+Re.xlsx and Raw_data2+Re.xlsx. Recorded here because it is the reference
# any calibration line has to hit, and because it refutes two readings we held:
#
#   * autotomy is NOT a tolerance band on CTj. Rows with APj > 0 have CTj - LT in
#     [0.00744, 0.048], and rows WITHOUT autotomy that also exceed LT start at exactly
#     0.00744 -- the same value. CTj alone does not determine the class.
#   * in Garrido's data essentially every order lands within 0.048 h of LT, so "on
#     schedule" is the norm and the discriminator is risk incidence, per Algorithm 1.
#
# Our pipeline puts every order 6 h past LT, which is 125x his largest excess. That is
# a pipeline gap, not a metric gap.
CANONICAL_AUTOTOMY_REFERENCE: dict[str, Any] = {
    "source_workbooks": ["Raw_data1+Re.xlsx", "Raw_data2+Re.xlsx"],
    "excluded_workbook": "Rsult_1.xlsx",
    "excluded_reason": (
        "not the thesis's final data; its 12 configurations differ from the thesis row "
        "counts by -1,949 to +735"),
    "n_rows": 43_360,
    "n_sheets": 19,
    "autotomy_share_overall": 0.00254,
    "autotomy_share_by_sheet_min": 0.00045,
    "autotomy_share_by_sheet_max": 0.00557,
    "ctj_minus_lt_when_autotomy": [0.00744, 0.048],
    "ctj_minus_lt_min_when_not_autotomy": 0.00744,
    "classification_is_determined_by_ctj": False,
    "note": (
        "computed as count(APj > 0) / scored rows, directly from the raw rows -- not "
        "from the 'Media APj' summary of the excluded workbook"),
}


def calibration_stamp(**extra: Any) -> dict[str, Any]:
    """The constants an artifact was produced under. Embed in every result payload."""
    stamp: dict[str, Any] = {
        "schema": "calibration_provenance_v1",
        "fulfilment_delay_hours": float(GARRIDO_FULFILLMENT_DELAY_HOURS),
        "lead_time_promise_hours": float(LEAD_TIME_PROMISE),
        "lead_time_source": THESIS_LEAD_TIME_SOURCE,
        "hours_per_week": float(HOURS_PER_WEEK),
        "calibration_line": (
            "historical"
            if float(GARRIDO_FULFILLMENT_DELAY_HOURS)
            == HISTORICAL_FULFILMENT_DELAY_HOURS
            else "prospective"),
        "autotomy_reachable": (
            float(GARRIDO_FULFILLMENT_DELAY_HOURS) <= float(LEAD_TIME_PROMISE)),
        "comparability_note": (
            "ret_excel is comparable only across artifacts sharing fulfilment_delay_hours "
            "and lead_time_promise_hours; step cadence no longer matters after the "
            "immutable-onset RPj fix (commit 125b94f)"),
    }
    stamp.update(extra)
    return stamp


def assert_same_calibration(*stamps: dict[str, Any]) -> None:
    """Refuse to compare artifacts from different calibration lines."""
    keys = ("fulfilment_delay_hours", "lead_time_promise_hours")
    seen = {tuple(s.get(k) for k in keys) for s in stamps}
    if len(seen) > 1:
        raise ValueError(
            f"artifacts span {len(seen)} calibration lines {sorted(seen)}; ret_excel is "
            "not comparable across them")
