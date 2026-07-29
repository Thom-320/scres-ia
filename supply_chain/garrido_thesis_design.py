"""The complete 90-configuration experimental design of Garrido-Rios (2017), as data.

Source: WRAP_Theses_Garrido_Rios_2017.pdf, University of Warwick, Chapter 6, section 6.7,
Tables 6.11-6.23. Every value below is transcribed from the published thesis, not inferred.

Why this file exists. Garrido delivered three workbooks, and only two of them are the thesis
data (`Raw_data1+Re.xlsx` = Cf1-Cf10, `Raw_data2+Re.xlsx` = Cf11-Cf20; verified row-for-row
against the degrees of freedom published in Tables 6.26 and 6.27). Those cover DS1 and DS2
only -- hypotheses H1a and H1b. **Cf21-Cf90 were not delivered, and Cf31-Cf90 are precisely
the configurations that test on-hand inventory buffers and short-term manufacturing capacity**,
i.e. the decision variables this project is trying to extend.

We do not need them. The design is fully published, and `supply_chain/config.py` already
implements Table 6.16 (buffer ladder) and Table 6.20 (capacity by shifts) verbatim -- both
verified to match exactly. So Cf21-Cf90 can be regenerated rather than requested; what the
delivered files buy us is a validation target for Cf1-Cf20, which is the useful half.

Structure of the design (section 6.7 and Equation 6.4):

    Scenario I   -- risk frequency only, buffers at zero, S = 1
        Cf1-Cf10   R1r increased (Table 6.13, H1a)   10-year horizon
        Cf11-Cf20  R2r increased (Table 6.14, H1b)   10-year horizon
        Cf21-Cf30  R3  increased (Table 6.15, H1c)   20-year horizon

    Scenario II  -- on-hand inventory buffers varied, S = 1 held fixed
        Cf31-Cf40  buffers x R1r (Table 6.17, H2a)   10-year horizon
        Cf41-Cf50  buffers x R2r (Table 6.18, H2b)   10-year horizon
        Cf51-Cf60  buffers x R3  (Table 6.19, H2c)   20-year horizon

    Scenario III -- work shifts varied, buffers held at ZERO to isolate capacity
        Cf61-Cf70  shifts x R1r (Table 6.21, H3a)    10-year horizon
        Cf71-Cf80  shifts x R2r (Table 6.22, H3b)    10-year horizon
        Cf81-Cf90  shifts x R3  (Table 6.23, H3c)    20-year horizon

Scenario II and III inherit the risk pattern of the corresponding Scenario I configuration:
Cf31 and Cf61 reuse Cf1's pattern, Cf47 and Cf77 reuse Cf17's, and so on with a stride of 30.
The thesis states the same holds for seeds: "the seed used for ReT(Cf7) is the same for
ReT(Cf37) and ReT(Cf67)."
"""
from __future__ import annotations

from dataclasses import dataclass

HOURS_PER_YEAR = 8_064.0

# -- Table 6.12: factor coding for R_cr -------------------------------------------------
# '-' is the current level of risk the MFSC faces; '+' is the increased level.
# U(a, b) is uniform over an inter-arrival window in hours; B(n, p) is binomial.
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

# -- Table 6.11: frequency of occurrence at the current level ---------------------------
CURRENT_EVENTS_PER_YEAR = {
    "R11": 48.0, "R12": 2 + 1 / 6, "R13": 58.0, "R14": 22_153.0,
    "R21": 0.5, "R22": 2.0, "R23": 1.0, "R24": 12.0, "R3": 0.05,
}

# -- Table 6.13: Cf1-Cf10, increased levels of R1r (H1a) --------------------------------
TABLE_6_13 = {
    1: "--++", 2: "-+--", 3: "+-++", 4: "+++-",
    5: "--+-", 6: "++-+", 7: "+--+", 8: "+---", 9: "-+++", 10: "-+-+",
}
R1R_FACTORS = ("R11", "R12", "R13", "R14")

# -- Table 6.14: Cf11-Cf20, increased levels of R2r (H1b) -------------------------------
TABLE_6_14 = {
    11: "+-++", 12: "+---", 13: "++-+", 14: "+++-", 15: "--++",
    16: "-+--", 17: "-++-", 18: "--+-", 19: "-+-+", 20: "++++",
}
R2R_FACTORS = ("R21", "R22", "R23", "R24")

# -- Table 6.15: Cf21-Cf30, increased levels of R3 (H1c) --------------------------------
TABLE_6_15 = {
    21: "-", 22: "+", 23: "+", 24: "+", 25: "+",
    26: "-", 27: "-", 28: "-", 29: "+", 30: "-",
}

# -- Tables 6.17 / 6.18 / 6.19: buffer level per configuration (hours index of Table 6.16)
TABLE_6_17 = {31: 504, 32: 336, 33: 168, 34: 1344, 35: 336,
              36: 1344, 37: 672, 38: 672, 39: 168, 40: 504}
TABLE_6_18 = {41: 1344, 42: 336, 43: 504, 44: 168, 45: 504,
              46: 1344, 47: 168, 48: 336, 49: 672, 50: 672}
TABLE_6_19 = {51: 672, 52: 1344, 53: 672, 54: 1344, 55: 504,
              56: 504, 57: 336, 58: 336, 59: 168, 60: 168}

# -- Tables 6.21 / 6.22 / 6.23: number of work shifts per configuration -----------------
TABLE_6_21 = {61: 2, 62: 1, 63: 3, 64: 3, 65: 1, 66: 2, 67: 1, 68: 2, 69: 3, 70: 3}
TABLE_6_22 = {71: 1, 72: 3, 73: 2, 74: 3, 75: 2, 76: 3, 77: 2, 78: 1, 79: 2, 80: 1}
TABLE_6_23 = {81: 1, 82: 3, 83: 2, 84: 3, 85: 2, 86: 3, 87: 2, 88: 1, 89: 2, 90: 1}

# -- Seeds recovered from the delivered workbooks ---------------------------------------
# Cell B2 of each CF sheet -- except CF2, whose sheet carries a leading blank row so the
# header sits at r2 and the seed at B3. All 20 are recovered.
THESIS_SEEDS: dict[int, int | None] = {
    1: 375, 2: 91, 3: 22, 4: 4, 5: 77, 6: 206, 7: 82, 8: 33, 9: 827, 10: 27,
    11: 21, 12: 47, 13: 101, 14: 28, 15: 10, 16: 11, 17: 96, 18: 498, 19: 95, 20: 85,
}


@dataclass(frozen=True)
class Configuration:
    """One Cf_i of the published design."""

    index: int
    scenario: int          # 1 = risk frequency, 2 = buffers, 3 = shifts
    hypothesis: str        # H1a .. H3c
    risk_family: str       # R1r | R2r | R3
    risk_pattern: str      # '+'/'-' per factor of the family, in family order
    buffer_hours: int      # 0 means no strategic buffer held
    shifts: int
    horizon_years: int
    seed: int | None
    base_index: int        # the Scenario-I configuration whose risk pattern it inherits

    @property
    def horizon_hours(self) -> float:
        return self.horizon_years * HOURS_PER_YEAR

    @property
    def factors(self) -> tuple[str, ...]:
        return {"R1r": R1R_FACTORS, "R2r": R2R_FACTORS, "R3": ("R3",)}[self.risk_family]

    @property
    def increased_risks(self) -> tuple[str, ...]:
        """Risk ids set to the '+' level for this configuration."""
        return tuple(f for f, s in zip(self.factors, self.risk_pattern) if s == "+")


def _family_for(base: int) -> tuple[str, str, int]:
    if 1 <= base <= 10:
        return "R1r", TABLE_6_13[base], 10
    if 11 <= base <= 20:
        return "R2r", TABLE_6_14[base], 10
    return "R3", TABLE_6_15[base], 20


def build_design() -> dict[int, Configuration]:
    """All 90 configurations of the thesis, keyed by index."""
    out: dict[int, Configuration] = {}
    hyp = {(1, "R1r"): "H1a", (1, "R2r"): "H1b", (1, "R3"): "H1c",
           (2, "R1r"): "H2a", (2, "R2r"): "H2b", (2, "R3"): "H2c",
           (3, "R1r"): "H3a", (3, "R2r"): "H3b", (3, "R3"): "H3c"}
    buffers = {**TABLE_6_17, **TABLE_6_18, **TABLE_6_19}
    shifts = {**TABLE_6_21, **TABLE_6_22, **TABLE_6_23}

    for i in range(1, 91):
        scenario = 1 if i <= 30 else (2 if i <= 60 else 3)
        base = i if i <= 30 else (i - 30 if i <= 60 else i - 60)
        family, pattern, horizon = _family_for(base)
        out[i] = Configuration(
            index=i,
            scenario=scenario,
            hypothesis=hyp[(scenario, family)],
            risk_family=family,
            risk_pattern=pattern,
            # Scenario I holds no strategic buffer; Scenario III explicitly zeroes it
            # "in order to isolate the effect of adding more manufacturing capacity".
            buffer_hours=buffers[i] if scenario == 2 else 0,
            # Scenario I and II hold S = 1 fixed; only Scenario III varies it.
            shifts=shifts[i] if scenario == 3 else 1,
            horizon_years=horizon,
            seed=THESIS_SEEDS.get(base),
            base_index=base,
        )
    return out


DESIGN = build_design()

# Configurations for which Garrido delivered output data, and which can therefore be
# used as a validation target rather than merely regenerated.
VALIDATABLE = tuple(range(1, 21))
