"""Garrido's Cobb-Douglas factory resilience index, ported to the MFSC DES.

Source: Garrido, Pongutá & García-Reyes (2024), "Zero-inventory plans, constant
workforce, or hybrid approach? Analysing pure production strategies for enhancing
factory resilience with demand variability", *International Journal of Production
Research*, DOI 10.1080/00207543.2024.2425771. The index is §3.4, Equations (2)-(6);
the five output variables are Algorithm 2 lines 33-38.

    R(z, e, p, t, k) = (z^a)(1/e^b)(p^c)(1/t^d)(1/k^n)                      Eq. (3)
    ln-linear:        a*ln z - b*ln e + c*ln p - d*ln t - n*ln k            Eq. (4)
    squashed:         R = 1 / (1 + exp(-(that)))                            Eq. (6)

with (z, e, p, t, k) = (zeta, epsilon, phi, tau, kappa_dot).

## The exponents are scale normalisers, not preference weights

The paper is explicit (§3.4, after Table 3): the maxima of the five variables were
taken over 10,000 runs *of their own APP model*, and "each function argument was
equated to 1/5. For example, in the case of zeta, zeta^max ~ 3,612, from which
a*Ln3,612 = 0.20, resulting in a = 0.024." So

    exponent_x = 0.20 / ln(x_max)

and their published 0.024 / 0.026 / 0.04 / 0.06 / 0.1771 encode a 36-week planning
model whose inventories run in the thousands. Ours run in the millions. Copying those
five numbers would silently rescale every term by orders of magnitude. `derive_exponents`
re-derives them with Garrido's own rule from our own maxima; `GARRIDO_2024_EXPONENTS`
is kept only so the test suite can prove the rule reproduces his published numbers.

## Two hazards that are properties of the index, not of this port

**zeta enters positively.** More inventory raises R. An index built to punish a
controller for hoarding stock will not do it through zeta -- it does it through the
holding cost inside kappa, whose exponent is ~7x zeta's in Garrido's own fit. Without
a costed ledger this index *rewards* overstocking, which is worse than ReT for that
question rather than better. The cost term is not optional.

**kappa_dot is set-relative.** Eq. (5)'s `kappa_dot = 7*kappa(S_ij)/sum_ij kappa(S_ij)`
normalises by the sum over the seven substrategies being compared; the 7 is that set's
cardinality. R therefore depends on the comparison set: adding or removing a policy
changes every other policy's R. The set must be frozen before evaluation and declared
in any table. Generalised here as `|S| * kappa(s) / sum_{s in S} kappa(s)`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping, Sequence

from supply_chain.config import CAPACITY_BY_SHIFTS, NUM_RAW_MATERIALS

# Garrido's published fit. Present for verification only -- never used to score.
GARRIDO_2024_EXPONENTS: dict[str, float] = {
    "zeta": 0.024, "epsilon": 0.026, "phi": 0.04, "tau": 0.06, "kappa_dot": 0.1771,
}
# "each function argument was equated to 1/5" (§3.4).
SHARE_PER_TERM: float = 0.20

# The seven cost coefficients of kappa(S_ij) = sum[c_p P + c_h H + c_l L + c_u U +
# c_i I + c_b B + c_o O]. Garrido's own baseline is assumption (6) of §3.1: "all
# decision variables of the output variable kappa(S_ij) share a constant cost
# parameter c, with c = 1". His §5 then varies the sensitive ones over [1, 2] and
# finds the ranking of S_12 unchanged, so c = 1 is his published baseline rather
# than a placeholder of ours.
UNIT_COSTS: dict[str, float] = {
    "c_p": 1.0,  # regular production
    "c_h": 1.0,  # hiring          -> shift increases
    "c_l": 1.0,  # firing          -> shift decreases
    "c_u": 1.0,  # marginal capacity (spare)
    "c_i": 1.0,  # holding
    "c_b": 1.0,  # backorders
    "c_o": 1.0,  # overtime        -> structurally absent from the MFSC DES
}

COST_COMPONENT_KEYS: dict[str, str] = {
    "c_p": "mean_regular_production",
    "c_h": "mean_shift_increases",
    "c_l": "mean_shift_decreases",
    "c_u": "mean_spare_capacity",
    "c_i": "mean_inventory",
    "c_b": "mean_backorders",
    "c_o": "mean_overtime",
}


def validate_costs(costs: Mapping[str, float]) -> dict[str, float]:
    """Return a complete, finite, non-negative seven-coefficient cost vector."""
    missing = sorted(set(COST_COMPONENT_KEYS) - set(costs))
    extra = sorted(set(costs) - set(COST_COMPONENT_KEYS))
    if missing or extra:
        raise ValueError(f"invalid cost keys: missing={missing}, extra={extra}")
    out = {name: float(costs[name]) for name in COST_COMPONENT_KEYS}
    bad = {name: value for name, value in out.items()
           if not math.isfinite(value) or value < 0.0}
    if bad:
        raise ValueError(f"cost coefficients must be finite and non-negative: {bad}")
    return out


def kappa_from_components(
    aggregate: Mapping[str, float],
    costs: Mapping[str, float],
) -> float:
    """Reprice a recorded episode without replaying its physical trajectory.

    The recorder persists the seven unpriced means independently of ``kappa``.
    This keeps Garrido's published ``c=1`` replication baseline separate from
    prospective economic sensitivity grids or a later domain-calibrated vector.
    """
    checked = validate_costs(costs)
    return float(sum(
        checked[cost_key] * float(aggregate[component_key])
        for cost_key, component_key in COST_COMPONENT_KEYS.items()
    ))

# ln(0) is -inf, and unlike Garrido's always-positive APP series our variables do
# reach zero: tau is exactly 0 in 88 of 108 calibration episodes, because the thesis
# operating point carries enough stock that net requirements never go positive.
#
# The floor for the four one-sided variables is 1.0 -- not an arbitrary epsilon.
# ln(1) = 0, so a variable at its floor contributes exactly nothing, and every term
# is then bounded by SHARE_PER_TERM in magnitude across the calibration range. A
# smaller floor breaks that: at 1e-4, tau's term reached -9.9 against an intended
# 0.20, so the least informative of the five variables would have dominated the
# index outright. `assert_terms_bounded` enforces the invariant.
#
# kappa_dot is the exception and gets only a numerical guard. It is normalised to
# set-mean 1, so ln kappa_dot is meaningfully signed in both directions and flooring
# it at 1.0 would discard exactly the below-average-cost half of the signal.
FLOORS: dict[str, float] = {
    "zeta": 1.0, "epsilon": 1.0, "phi": 1.0, "tau": 1.0, "kappa_dot": 1e-9,
}

# ln(x_max) below this makes `0.20/ln(x_max)` ill-conditioned: the exponent exceeds
# 0.20 and its relative sensitivity to the maximum, `1/ln(x_max)`, exceeds 1, so
# noise in a single calibration episode swings it. Not an error -- a disclosure.
WELL_CONDITIONED_LOG_MAX: float = 1.0

VARIABLES: tuple[str, ...] = ("zeta", "epsilon", "phi", "tau", "kappa_dot")
# Eq. (4): zeta and phi raise R; epsilon, tau and kappa_dot lower it.
SIGNS: dict[str, float] = {
    "zeta": +1.0, "epsilon": -1.0, "phi": +1.0, "tau": -1.0, "kappa_dot": -1.0,
}


def derive_exponents(maxima: Mapping[str, float]) -> dict[str, float]:
    """Garrido's rule: `exponent_x = 0.20 / ln(x_max)`, applied to our own maxima.

    Each of the five terms then contributes at most 1/5 of the linear index at its
    own observed maximum, which is what makes five quantities in incompatible units
    commensurable. A maximum at or below 1.0 has a non-positive logarithm and cannot
    normalise anything, so it is rejected rather than clamped.
    """
    out: dict[str, float] = {}
    for name in VARIABLES:
        x_max = float(maxima[name])
        if x_max <= 1.0:
            raise ValueError(
                f"{name}_max = {x_max} <= 1; ln is non-positive and the exponent "
                "rule 0.20/ln(x_max) is undefined. Widen the calibration sweep."
            )
        out[name] = SHARE_PER_TERM / math.log(x_max)
    return out


def conditioning(maxima: Mapping[str, float]) -> dict[str, dict[str, float | bool]]:
    """How trustworthy each derived exponent is.

    Differentiating the rule gives `d(exponent)/exponent = -d(x_max)/(x_max * ln x_max)`,
    so `1/ln(x_max)` is the factor by which a relative error in the observed maximum is
    amplified into the exponent. For zeta (x_max ~ 1.3e6) that is 0.07 and the exponent
    is very stable; for tau (x_max ~ 1.2) it is 5.4 and a single unusual episode moves
    the exponent by more than the error in it. Reported, not silently accepted.
    """
    out: dict[str, dict[str, float | bool]] = {}
    for name in VARIABLES:
        log_max = math.log(float(maxima[name]))
        out[name] = {
            "log_max": log_max,
            "relative_sensitivity": (1.0 / log_max) if log_max > 0 else math.inf,
            "well_conditioned": log_max >= WELL_CONDITIONED_LOG_MAX,
        }
    return out


def assert_terms_bounded(exponents: Mapping[str, float],
                         maxima: Mapping[str, float]) -> None:
    """Each term must lie within +/- SHARE_PER_TERM across the calibration range.

    This is what the exponent rule is *for*: five quantities in incompatible units
    are made commensurable by capping each one's contribution at 1/5. The invariant
    holds only if the floor is 1.0 (so the term is 0 there) and the exponent came
    from the same maximum. Violating it means the index is being driven by whichever
    variable happens to be worst-scaled -- the failure mode that a 1e-4 floor on tau
    produced on first run.
    """
    for name in VARIABLES:
        if name == "kappa_dot":  # two-sided by construction; bounded by the set
            continue
        at_max = abs(float(exponents[name]) * math.log(float(maxima[name])))
        at_floor = abs(float(exponents[name]) * math.log(FLOORS[name]))
        if at_max > SHARE_PER_TERM + 1e-9 or at_floor > SHARE_PER_TERM + 1e-9:
            raise ValueError(
                f"{name}: term is unbounded ({at_max=:.4f}, {at_floor=:.4f}) against "
                f"a budget of {SHARE_PER_TERM}. Check the floor and the maximum."
            )


def resilience_index(components: Mapping[str, float],
                     exponents: Mapping[str, float]) -> dict[str, float]:
    """Eq. (4) then Eq. (6). Returns the index, the linear score, and each term."""
    terms: dict[str, float] = {}
    linear = 0.0
    for name in VARIABLES:
        x = max(float(components[name]), FLOORS[name])
        term = SIGNS[name] * float(exponents[name]) * math.log(x)
        terms[f"term_{name}"] = term
        linear += term
    return {
        "R_cobb_douglas": 1.0 / (1.0 + math.exp(-linear)),
        "linear_score": linear,
        **terms,
    }


def kappa_dot(kappa_by_policy: Mapping[str, float]) -> dict[str, float]:
    """Eq. (5)'s set-relative cost normaliser, generalised to any comparison set.

    `kappa_dot(s) = |S| * kappa(s) / sum_{s in S} kappa(s)`, so the set mean is 1 and
    `ln kappa_dot` is signed about it. Because the divisor is the whole set's cost,
    these values are meaningless outside the set they were computed in.
    """
    total = float(sum(kappa_by_policy.values()))
    if total <= 0.0:
        raise ValueError("total kappa over the comparison set is not positive")
    n = len(kappa_by_policy)
    return {k: n * float(v) / total for k, v in kappa_by_policy.items()}


@dataclass
class CobbDouglasRecorder:
    """Samples the five output variables off the physical ledger, period by period.

    Deliberately external to `MFSCSimulation`: it reads public attributes and writes
    nothing, so the frozen DES is untouched. The caller steps the simulator in
    `period_hours` increments and calls `sample(sim)` after each step.

    Mapping from Garrido's APP model to the MFSC DES, per period t:

    | his | ours |
    |---|---|
    | `I_t` inventory   | on-hand rations + raw material converted to ration-equivalents |
    | `B_t` backorders  | `pending_backorder_qty` |
    | `P_t` production  | period delta of `total_produced` |
    | `Theta_t` capacity| Table 6.20 `theoretical_capacity_rations` at the current shift count |
    | `U_t` spare       | `max(Theta_t - P_t, 0)` |
    | `GR_t` gross req. | period delta of `total_demanded` |
    | `NR_t` net req.   | `max(GR_t - I_{t-1} + B_{t-1}, 0)`, his Algorithm 2 line 23 |
    | `H_t`/`L_t`       | shift increases / decreases |
    | `O_t` overtime    | structurally 0 -- the MFSC DES has no overtime process |

    Raw material is divided by the 12-component BOM so a kit and the ration it
    becomes count once, not twelve times. Garrido has a single product and never
    faced this; leaving it in raw units would let raw material dominate both zeta
    and the holding term of kappa by an order of magnitude.
    """

    period_hours: float = 24.0
    costs: Mapping[str, float] = field(default_factory=lambda: dict(UNIT_COSTS))

    def __post_init__(self) -> None:
        self.costs = validate_costs(self.costs)
        self.reset()

    def reset(self) -> None:
        self.periods: list[dict[str, float]] = []
        self._prev_produced = 0.0
        self._prev_demanded = 0.0
        self._prev_shifts: int | None = None
        self._prev_inventory = 0.0
        self._prev_backorders = 0.0

    @staticmethod
    def _inventory_ration_equivalents(sim: Any) -> float:
        """On-hand across every stocking point, in ration-equivalents."""
        rations = sum(
            float(getattr(sim, n).level)
            for n in ("rations_al", "rations_sb", "rations_sb_dispatch",
                      "rations_cssu", "rations_theatre")
            if getattr(sim, n, None) is not None
        )
        raw = sum(
            float(getattr(sim, n).level)
            for n in ("raw_material_wdc", "raw_material_al")
            if getattr(sim, n, None) is not None
        )
        return rations + raw / float(NUM_RAW_MATERIALS)

    @staticmethod
    def _installed_capacity_per_period(sim: Any, period_hours: float) -> float:
        """Table 6.20 daily capacity at the current shift count, scaled to the period."""
        shifts = int(sim.params.get("assembly_shifts", getattr(sim, "shifts", 1)))
        row = CAPACITY_BY_SHIFTS[shifts]
        return float(row["theoretical_capacity_rations"]) * (period_hours / 24.0)

    def sample(self, sim: Any) -> dict[str, float]:
        produced = float(getattr(sim, "total_produced", 0.0))
        demanded = float(getattr(sim, "total_demanded", 0.0))
        shifts = int(sim.params.get("assembly_shifts", getattr(sim, "shifts", 1)))

        p_t = max(produced - self._prev_produced, 0.0)
        gr_t = max(demanded - self._prev_demanded, 0.0)
        theta_t = self._installed_capacity_per_period(sim, self.period_hours)
        u_t = max(theta_t - p_t, 0.0)
        i_t = self._inventory_ration_equivalents(sim)
        b_t = float(getattr(sim, "pending_backorder_qty", 0.0) or 0.0)

        # Algorithm 2 line 23: NR_t <- max{GR_{t+v} - I_{t-1} + B_{t-1}, 0}.
        nr_t = max(gr_t - self._prev_inventory + self._prev_backorders, 0.0)
        # Algorithm 2 line 36: tau_t = NR_t / min{GR_{t+v}, Theta_t}.
        denom = min(gr_t, theta_t)
        tau_t = nr_t / denom if denom > 0.0 else 0.0

        prev_shifts = self._prev_shifts if self._prev_shifts is not None else shifts
        h_t = float(max(shifts - prev_shifts, 0))
        l_t = float(max(prev_shifts - shifts, 0))
        o_t = 0.0  # no overtime process exists in the MFSC DES

        c = self.costs
        cost_t = (c["c_p"] * p_t + c["c_h"] * h_t + c["c_l"] * l_t
                  + c["c_u"] * u_t + c["c_i"] * i_t + c["c_b"] * b_t
                  + c["c_o"] * o_t)

        row = {
            "I_t": i_t, "B_t": b_t, "P_t": p_t, "U_t": u_t, "Theta_t": theta_t,
            "GR_t": gr_t, "NR_t": nr_t, "tau_t": tau_t,
            "H_t": h_t, "L_t": l_t, "O_t": o_t, "C_t": cost_t,
        }
        self.periods.append(row)
        self._prev_produced, self._prev_demanded = produced, demanded
        self._prev_shifts, self._prev_inventory = shifts, i_t
        self._prev_backorders = b_t
        return row

    def aggregate(self) -> dict[str, float]:
        """Algorithm 2 lines 33-37: the per-period means over the horizon T.

        `kappa` is returned raw. It only becomes `kappa_dot` once the comparison
        set is known, which is why this cannot produce an R on its own.
        """
        t = len(self.periods)
        if t == 0:
            raise ValueError("no periods sampled")

        def mean(key: str) -> float:
            return sum(r[key] for r in self.periods) / t

        return {
            "zeta": mean("I_t"),
            "epsilon": mean("B_t"),
            "phi": mean("U_t"),
            "tau": mean("tau_t"),
            "kappa": mean("C_t"),
            "mean_regular_production": mean("P_t"),
            "mean_shift_increases": mean("H_t"),
            "mean_shift_decreases": mean("L_t"),
            "mean_spare_capacity": mean("U_t"),
            "mean_inventory": mean("I_t"),
            "mean_backorders": mean("B_t"),
            "mean_overtime": mean("O_t"),
            "T_periods": float(t),
            "mean_production": mean("P_t"),
            "mean_capacity": mean("Theta_t"),
        }


def score_comparison_set(
    aggregates: Mapping[str, Mapping[str, float]],
    exponents: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    """Score a frozen comparison set: raw kappa -> kappa_dot -> Eq. (6), per policy.

    `aggregates` maps policy name to the output of `CobbDouglasRecorder.aggregate`.
    Every policy in the set influences every other policy's score through kappa_dot,
    so the set passed here must be the one declared before evaluation.
    """
    kd = kappa_dot({k: v["kappa"] for k, v in aggregates.items()})
    out: dict[str, dict[str, float]] = {}
    for name, agg in aggregates.items():
        components = {
            "zeta": agg["zeta"], "epsilon": agg["epsilon"], "phi": agg["phi"],
            "tau": agg["tau"], "kappa_dot": kd[name],
        }
        out[name] = {
            **resilience_index(components, exponents),
            **{f"component_{k}": v for k, v in components.items()},
            "kappa_raw": agg["kappa"],
        }
    return out


def maxima_over(aggregates: Iterable[Mapping[str, float]],
                kappa_dots: Sequence[float]) -> dict[str, float]:
    """Per-variable maxima for `derive_exponents`, over a calibration sweep."""
    aggregates = list(aggregates)
    if not aggregates:
        raise ValueError("empty calibration sweep")
    return {
        "zeta": max(a["zeta"] for a in aggregates),
        "epsilon": max(a["epsilon"] for a in aggregates),
        "phi": max(a["phi"] for a in aggregates),
        "tau": max(a["tau"] for a in aggregates),
        "kappa_dot": max(kappa_dots),
    }
