"""Executable transcription of Garrido-Rios (2017), Tables 6.12-6.23.

The module contains design inputs only.  Scientific authority belongs to a
frozen execution contract plus completed receipts, never to this transcription
alone.
"""
from __future__ import annotations

from dataclasses import dataclass

HOURS_PER_YEAR = 8_064.0

FACTOR_CODING: dict[str, dict[str, tuple[str, float, float]]] = {
    "R11": {"-": ("U", 1, 168), "+": ("U", 1, 42)},
    "R12": {"-": ("B", 1, 11), "+": ("B", 4, 11)},
    "R13": {"-": ("B", 1, 10), "+": ("B", 4, 10)},
    "R14": {"-": ("B", 3, 100), "+": ("B", 8, 100)},
    "R21": {"-": ("U", 1, 16_128), "+": ("U", 1, 4_032)},
    "R22": {"-": ("U", 1, 4_032), "+": ("U", 1, 1_344)},
    "R23": {"-": ("U", 1, 8_064), "+": ("U", 1, 1_344)},
    "R24": {"-": ("U", 1, 672), "+": ("U", 1, 336)},
    "R3": {"-": ("U", 1, 161_280), "+": ("U", 1, 80_640)},
}

R1R_FACTORS = ("R11", "R12", "R13", "R14")
R2R_FACTORS = ("R21", "R22", "R23", "R24")

TABLE_6_13 = {
    1: "--++", 2: "-+--", 3: "+-++", 4: "+++-", 5: "--+-",
    6: "++-+", 7: "+--+", 8: "+---", 9: "-+++", 10: "-+-+",
}
TABLE_6_14 = {
    11: "+-++", 12: "+---", 13: "++-+", 14: "+++-", 15: "--++",
    16: "-+--", 17: "-++-", 18: "--+-", 19: "-+-+", 20: "++++",
}
TABLE_6_15 = {
    21: "-", 22: "+", 23: "+", 24: "+", 25: "+",
    26: "-", 27: "-", 28: "-", 29: "+", 30: "-",
}

TABLE_6_17 = {
    31: 504, 32: 336, 33: 168, 34: 1344, 35: 336,
    36: 1344, 37: 672, 38: 672, 39: 168, 40: 504,
}
TABLE_6_18 = {
    41: 1344, 42: 336, 43: 504, 44: 168, 45: 504,
    46: 1344, 47: 168, 48: 336, 49: 672, 50: 672,
}
TABLE_6_19 = {
    51: 672, 52: 1344, 53: 672, 54: 1344, 55: 504,
    56: 504, 57: 336, 58: 336, 59: 168, 60: 168,
}

TABLE_6_21 = {
    61: 2, 62: 1, 63: 3, 64: 3, 65: 1,
    66: 2, 67: 1, 68: 2, 69: 3, 70: 3,
}
TABLE_6_22 = {
    71: 1, 72: 3, 73: 2, 74: 3, 75: 2,
    76: 3, 77: 2, 78: 1, 79: 2, 80: 1,
}
TABLE_6_23 = {
    81: 1, 82: 3, 83: 2, 84: 3, 85: 2,
    86: 3, 87: 2, 88: 1, 89: 2, 90: 1,
}

# All twenty seeds are recoverable by locating the row labelled "Seed".
THESIS_SEEDS: dict[int, int] = {
    1: 375, 2: 91, 3: 22, 4: 4, 5: 77,
    6: 206, 7: 82, 8: 33, 9: 827, 10: 27,
    11: 21, 12: 47, 13: 101, 14: 28, 15: 10,
    16: 11, 17: 96, 18: 498, 19: 95, 20: 85,
}

SOURCE_VALIDATION_QUARANTINE = {
    1: "workbook horizon is approximately 20 years, published design says 10",
    2: "workbook horizon is approximately 20 years, published design says 10",
    5: "workbook ends at approximately 8.94 years, published design says 10",
}


@dataclass(frozen=True)
class Configuration:
    index: int
    scenario: int
    hypothesis: str
    risk_family: str
    risk_pattern: str
    buffer_hours: int
    shifts: int
    horizon_years: int
    source_seed: int | None
    base_index: int

    @property
    def horizon_hours(self) -> float:
        return self.horizon_years * HOURS_PER_YEAR

    @property
    def factors(self) -> tuple[str, ...]:
        return {
            "R1r": R1R_FACTORS,
            "R2r": R2R_FACTORS,
            "R3": ("R3",),
        }[self.risk_family]

    @property
    def increased_risks(self) -> tuple[str, ...]:
        return tuple(
            factor
            for factor, sign in zip(self.factors, self.risk_pattern)
            if sign == "+"
        )


def _family(base: int) -> tuple[str, str, int]:
    if base <= 10:
        return "R1r", TABLE_6_13[base], 10
    if base <= 20:
        return "R2r", TABLE_6_14[base], 10
    return "R3", TABLE_6_15[base], 20


def build_design() -> dict[int, Configuration]:
    hypotheses = {
        (1, "R1r"): "H1a", (1, "R2r"): "H1b", (1, "R3"): "H1c",
        (2, "R1r"): "H2a", (2, "R2r"): "H2b", (2, "R3"): "H2c",
        (3, "R1r"): "H3a", (3, "R2r"): "H3b", (3, "R3"): "H3c",
    }
    buffers = {**TABLE_6_17, **TABLE_6_18, **TABLE_6_19}
    shifts = {**TABLE_6_21, **TABLE_6_22, **TABLE_6_23}
    design: dict[int, Configuration] = {}
    for index in range(1, 91):
        scenario = 1 if index <= 30 else 2 if index <= 60 else 3
        base = index if scenario == 1 else index - 30 if scenario == 2 else index - 60
        family, pattern, horizon = _family(base)
        design[index] = Configuration(
            index=index,
            scenario=scenario,
            hypothesis=hypotheses[(scenario, family)],
            risk_family=family,
            risk_pattern=pattern,
            buffer_hours=buffers[index] if scenario == 2 else 0,
            shifts=shifts[index] if scenario == 3 else 1,
            horizon_years=horizon,
            source_seed=THESIS_SEEDS.get(base),
            base_index=base,
        )
    return design


DESIGN = build_design()
VALIDATABLE = tuple(range(1, 21))
