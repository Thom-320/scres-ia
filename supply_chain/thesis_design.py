from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .config import HOURS_PER_YEAR_THESIS, INVENTORY_BUFFERS

RiskLevel = Literal["current", "increased"]
ScenarioFamily = Literal["risk_r1", "risk_r2", "risk_r3", "inventory", "capacity"]

R1_RISKS = ("R11", "R12", "R13", "R14")
R2_RISKS = ("R21", "R22", "R23", "R24")
R3_RISKS = ("R3",)

RISK_PATTERNS: dict[int, tuple[bool, ...]] = {
    1: (False, False, True, True),
    2: (False, True, False, False),
    3: (True, False, True, True),
    4: (True, True, True, False),
    5: (False, False, True, False),
    6: (True, True, False, True),
    7: (True, False, False, True),
    8: (True, False, False, False),
    9: (False, True, True, True),
    10: (False, True, False, True),
    11: (True, False, True, True),
    12: (True, False, False, False),
    13: (True, True, False, True),
    14: (True, True, True, False),
    15: (False, False, True, True),
    16: (False, True, False, False),
    17: (False, True, True, False),
    18: (False, False, True, False),
    19: (False, True, False, True),
    20: (True, True, True, True),
    21: (False,),
    22: (True,),
    23: (True,),
    24: (True,),
    25: (True,),
    26: (False,),
    27: (False,),
    28: (False,),
    29: (True,),
    30: (False,),
}

INVENTORY_PERIOD_BY_CFI = {
    31: 504,
    32: 336,
    33: 168,
    34: 1344,
    35: 336,
    36: 1344,
    37: 672,
    38: 672,
    39: 168,
    40: 504,
    41: 1344,
    42: 336,
    43: 504,
    44: 168,
    45: 504,
    46: 1344,
    47: 168,
    48: 336,
    49: 672,
    50: 672,
    51: 672,
    52: 1344,
    53: 672,
    54: 1344,
    55: 504,
    56: 504,
    57: 336,
    58: 336,
    59: 168,
    60: 168,
}

SHIFTS_BY_CFI = {
    61: 2,
    62: 1,
    63: 3,
    64: 3,
    65: 1,
    66: 2,
    67: 1,
    68: 2,
    69: 3,
    70: 3,
    71: 1,
    72: 3,
    73: 2,
    74: 3,
    75: 2,
    76: 3,
    77: 2,
    78: 1,
    79: 2,
    80: 1,
    81: 1,
    82: 3,
    83: 2,
    84: 3,
    85: 2,
    86: 3,
    87: 2,
    88: 1,
    89: 2,
    90: 1,
}


@dataclass(frozen=True)
class ThesisDesignSpec:
    cfi: int
    family: ScenarioFamily
    source_cfi: int
    enabled_risks: tuple[str, ...]
    risk_overrides: dict[str, RiskLevel]
    shifts: int
    initial_buffers: dict[str, float] | None
    inventory_replenishment_period: float | None
    horizon_hours: float

    @property
    def label(self) -> str:
        return f"Cf{self.cfi}"


def parse_cf_range(value: str) -> list[int]:
    """Parse CLI ranges like '1-30,47,85' into sorted Cfi values."""
    cfi_values: set[int] = set()
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            cfi_values.update(range(start, end + 1))
        else:
            cfi_values.add(int(token))
    return sorted(cfi_values)


def source_cfi_for(cfi: int) -> int:
    """Map inventory/capacity design rows back to their paired risk row."""
    if 1 <= cfi <= 30:
        return cfi
    if 31 <= cfi <= 60:
        return cfi - 30
    if 61 <= cfi <= 90:
        return cfi - 60
    raise ValueError(f"Cf{cfi} is outside the thesis design range 1..90.")


def risk_group_for_source(source_cfi: int) -> tuple[str, ...]:
    if 1 <= source_cfi <= 10:
        return R1_RISKS
    if 11 <= source_cfi <= 20:
        return R2_RISKS
    if 21 <= source_cfi <= 30:
        return R3_RISKS
    raise ValueError(f"Cf{source_cfi} has no thesis risk group.")


def risk_overrides_for_source(source_cfi: int) -> dict[str, RiskLevel]:
    risks = risk_group_for_source(source_cfi)
    pattern = RISK_PATTERNS[source_cfi]
    return {
        risk_id: "increased" if is_increased else "current"
        for risk_id, is_increased in zip(risks, pattern, strict=True)
    }


def family_for(cfi: int) -> ScenarioFamily:
    if 1 <= cfi <= 10:
        return "risk_r1"
    if 11 <= cfi <= 20:
        return "risk_r2"
    if 21 <= cfi <= 30:
        return "risk_r3"
    if 31 <= cfi <= 60:
        return "inventory"
    if 61 <= cfi <= 90:
        return "capacity"
    raise ValueError(f"Cf{cfi} is outside the thesis design range 1..90.")


def horizon_hours_for_source(source_cfi: int) -> float:
    years = 20 if 21 <= source_cfi <= 30 else 10
    return float(years * HOURS_PER_YEAR_THESIS)


def design_spec_for_cfi(cfi: int) -> ThesisDesignSpec:
    source_cfi = source_cfi_for(cfi)
    family = family_for(cfi)
    inventory_period = INVENTORY_PERIOD_BY_CFI.get(cfi)
    shifts = SHIFTS_BY_CFI.get(cfi, 1)
    initial_buffers = None
    if inventory_period is not None:
        initial_buffers = {
            key: float(value)
            for key, value in INVENTORY_BUFFERS[inventory_period].items()
        }
    return ThesisDesignSpec(
        cfi=cfi,
        family=family,
        source_cfi=source_cfi,
        enabled_risks=risk_group_for_source(source_cfi),
        risk_overrides=risk_overrides_for_source(source_cfi),
        shifts=shifts,
        initial_buffers=initial_buffers,
        inventory_replenishment_period=(
            float(inventory_period) if inventory_period is not None else None
        ),
        horizon_hours=horizon_hours_for_source(source_cfi),
    )


def design_matrix(cfi_values: list[int] | None = None) -> list[ThesisDesignSpec]:
    values = list(range(1, 91)) if cfi_values is None else cfi_values
    return [design_spec_for_cfi(cfi) for cfi in values]
