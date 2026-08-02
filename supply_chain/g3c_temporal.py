"""Machine-readable G3c mechanism boundary.

G3c v2 tests one mechanism only: minimum dwell.  Switching cost is deliberately
outside this contract and requires a future, separate preregistration.
"""
from __future__ import annotations

G3C_MECHANISM = "min_dwell"
G3C_MIN_DWELL_LEVELS_DAYS = (1, 3, 7)
G3C_NULL_MIN_DWELL_DAYS = 1


def validate_min_dwell_days(value: int | float) -> int:
    """Validate the frozen G3c level; reject unregistered levels."""
    if isinstance(value, bool) or int(value) != value:
        raise ValueError("min_dwell_days must be an integer number of days")
    days = int(value)
    if days not in G3C_MIN_DWELL_LEVELS_DAYS:
        raise ValueError(
            f"min_dwell_days must be one of {G3C_MIN_DWELL_LEVELS_DAYS}; got {value!r}"
        )
    return days


def g3c_arm_grid() -> tuple[dict[str, int | str], ...]:
    """Return the complete frozen arm grid, including the regression null."""
    return tuple(
        {
            "mechanism": G3C_MECHANISM,
            "min_dwell_days": days,
            "role": "null_legacy_regression" if days == G3C_NULL_MIN_DWELL_DAYS
            else "temporal_coupling",
        }
        for days in G3C_MIN_DWELL_LEVELS_DAYS
    )
