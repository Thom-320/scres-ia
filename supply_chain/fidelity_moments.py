"""Multi-moment fidelity comparison, defined as dominance rather than a score.

The consolidated contract (`contracts/paper_b_independent_calibration_v2.json`) forbids
selecting a winner from the calibration grid. So this cannot be a distance that ranks
cells. It is a **dominance relation over moments**, and the non-dominated set is the
output. If that set is the whole grid, the honest report is that the grid does not
discriminate.

Three things the earlier draft left undefined, and how each is fixed here.

**The scale of each moment.** Not chosen by us. Every moment is expressed in units of
its own *reference spread*: the between-configuration standard deviation of that moment
across the canonical sheets of one risk family. A moment whose reference spread is zero
is degenerate and is excluded, by name, in the artifact — never silently rescaled.

**Zeros and units.** Dividing by a reference spread handles both. A share, an hour count
and an order count all become dimensionless multiples of "how much this moment varies
between Garrido's own configurations", which is the only scale the data itself supplies.

**Reference uncertainty.** The reference mean is estimated from a finite number of
sheets and our own moment from a finite number of roots, so a gap smaller than the
combined standard error is not a gap. The discrepancy is therefore

    d_k = |M_k - Rbar_k| / sqrt( s_k^2 / n_ref + se_k^2 )

which is on a z-like scale: `d_k = 1` means one combined standard error away.

**Dominance, with a declared indifference band.** Cell A dominates cell B when it is no
worse on every moment and strictly better on at least one, where "no worse" allows a
declared epsilon so that ordering is not driven by noise:

    A dominates B  iff  for all k: d_k(A) <= d_k(B) + EPSILON
                        and exists k: d_k(A) <  d_k(B) - EPSILON

`EPSILON` is declared here, not fitted, and its sensitivity is swept in the artifact.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

# Half a combined standard error. Declared, not fitted; the runner sweeps it so a
# non-dominated set that moves with EPSILON is reported as unstable rather than shown.
EPSILON: float = 0.5

# The swept band, amended 2026-07-31 (contracts/epsilon_range_amendment_2026-07-31.json).
# Was 0.25/0.5/1.0/2.0 -- an 8x span. EPSILON is in COMBINED STANDARD ERRORS, so an
# epsilon of 2.0 declares indifference to a two-combined-SE difference, more than
# separates any pair of arms this project compares: `no_worse` becomes trivially true
# and `strictly` unsatisfiable, so the check degenerates regardless of the data. The
# band is now the declared EPSILON +-50%, inside the regime where both terms can bind.
EPSILON_BAND: tuple[float, ...] = (0.25, 0.375, 0.5, 0.625, 0.75)

# The canonical sheet-to-family map. Cf1-Cf10 are the R1r configurations and Cf11-Cf20
# the R2r ones (thesis Tables 6.13-6.15). Cf21-Cf30 are R3 and have NO reference
# workbook, so R3 is external validation and never part of the fit.
FAMILY_SHEETS: dict[str, tuple[str, ...]] = {
    "R1r": tuple(f"CF{i}" for i in range(1, 11)),
    "R2r": tuple(f"CF{i}" for i in range(11, 21)),
}

# Every moment must be computable identically from a canonical sheet and from one of our
# episodes -- same definition, same row filter. Anything that can only be read off a
# summary cell or a figure is not admissible.
MOMENT_NAMES: tuple[str, ...] = (
    "autotomy_share",      # count(APj > 0) / scored rows
    "ret_mean",            # mean of the Re column / our ret_excel per order
    "ret_above_one_share",  # count(ReT > 1) / scored rows
    "rpj_mean",            # mean RPj over rows with RPj > 0
    "rpj_p95",             # 95th percentile of RPj over rows with RPj > 0
    "scored_orders_per_year",  # population as a RATE, not a raw count
)

# CORRECTED 2026-07-30. An earlier version hard-coded 20 years for every canonical
# sheet, citing §6.8.1's "20 years or 161,280 hours". Measured from each sheet's own
# max(OPTj), only CF1 and CF2 run ~20 years; CF3-CF20 run ~10, exactly as Table 6.13
# prescribes. Dividing every sheet by 20 therefore halved most of the reference and
# manufactured a discrepancy: it made our order rate look 2x his when it is 1.09x
# generated and 1.27x scored.
#
# The horizon is now MEASURED per sheet rather than assumed. `REFERENCE_HORIZON_YEARS`
# survives only as the fallback when a sheet carries no OPTj column.
REFERENCE_HORIZON_YEARS: float = 20.0
# Thesis year basis, used to convert max(OPTj) in hours into years.
HOURS_PER_THESIS_YEAR: float = 8064.0


def horizon_years_from_optj(max_optj_hours: float) -> float:
    """A sheet's own simulated horizon, from its last order-placement time."""
    return max(float(max_optj_hours) / HOURS_PER_THESIS_YEAR, 1e-9)


def moments_from_rows(
    *,
    apj: Sequence[float],
    rpj: Sequence[float],
    ret: Sequence[float],
    horizon_years: float = REFERENCE_HORIZON_YEARS,
) -> dict[str, float]:
    """The six moments from per-order rows. One definition, used on both sides.

    `horizon_years` makes the population moment a rate. It defaults to the reference
    horizon so a canonical sheet needs no argument; our runs must pass their own.
    """
    n = len(ret)
    if n == 0:
        raise ValueError("no scored rows")
    pos_rpj = sorted(v for v in rpj if v > 0.0)
    return {
        "autotomy_share": sum(1 for v in apj if v > 0.0) / n,
        "ret_mean": sum(ret) / n,
        "ret_above_one_share": sum(1 for v in ret if v > 1.0) / n,
        "rpj_mean": (sum(pos_rpj) / len(pos_rpj)) if pos_rpj else 0.0,
        "rpj_p95": _quantile(pos_rpj, 0.95) if pos_rpj else 0.0,
        "scored_orders_per_year": float(n) / max(float(horizon_years), 1e-9),
    }


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = q * (len(sorted_values) - 1)
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return float(sorted_values[int(idx)])
    frac = idx - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


@dataclass(frozen=True)
class MomentReference:
    """One family's reference: the mean, the between-sheet spread, and the count."""

    mean: float
    spread: float          # between-configuration SD across the family's sheets
    n_sheets: int

    @property
    def degenerate(self) -> bool:
        """A moment that does not vary between his configurations cannot set a scale."""
        return not (self.spread > 0.0) or self.n_sheets < 2


def build_reference(per_sheet: Mapping[str, Mapping[str, float]],
                    sheets: Sequence[str]) -> dict[str, MomentReference]:
    """Reference mean and spread per moment, from the canonical sheets of one family."""
    present = [s for s in sheets if s in per_sheet]
    if len(present) < 2:
        raise ValueError(f"need >= 2 canonical sheets, got {present}")
    out: dict[str, MomentReference] = {}
    for name in MOMENT_NAMES:
        vals = [float(per_sheet[s][name]) for s in present]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        out[name] = MomentReference(mean=mean, spread=math.sqrt(var),
                                    n_sheets=len(vals))
    return out


def discrepancies(ours: Mapping[str, float],
                  our_se: Mapping[str, float],
                  reference: Mapping[str, MomentReference]) -> dict[str, float]:
    """`d_k` in combined standard errors. Degenerate moments come back as NaN."""
    out: dict[str, float] = {}
    for name in MOMENT_NAMES:
        ref = reference[name]
        if ref.degenerate:
            out[name] = math.nan
            continue
        combined = math.sqrt(ref.spread ** 2 / ref.n_sheets
                             + float(our_se.get(name, 0.0)) ** 2)
        if combined <= 0.0:
            out[name] = math.nan
            continue
        out[name] = abs(float(ours[name]) - ref.mean) / combined
    return out


def dominates(a: Mapping[str, float], b: Mapping[str, float],
              epsilon: float = EPSILON) -> bool:
    """A dominates B: no worse anywhere, strictly better somewhere, outside epsilon.

    Moments that are degenerate in the reference (NaN) are skipped in both directions,
    so a cell can neither win nor lose on a moment that carries no scale.
    """
    live = [k for k in MOMENT_NAMES
            if not math.isnan(a.get(k, math.nan))
            and not math.isnan(b.get(k, math.nan))]
    if not live:
        return False
    # `no_worse` must see EVERY live moment.
    no_worse = all(a[k] <= b[k] + epsilon for k in live)
    # Amended 2026-07-31: a moment IDENTICAL between the two arms can never satisfy
    # `a < b - epsilon` at any positive epsilon, so counting it among the candidates can
    # only REMOVE strictness and never supply it -- measured, a flip driven by an
    # autotomy_share gap of exactly +0.00. Zero-gap moments are excluded here only.
    strictly = any(a[k] < b[k] - epsilon for k in live if a[k] != b[k])
    return no_worse and strictly


def non_dominated(cells: Mapping[str, Mapping[str, float]],
                  epsilon: float = EPSILON) -> list[str]:
    """The output of the comparison. Not a winner -- a set."""
    return sorted(
        name for name, d in cells.items()
        if not any(dominates(other, d, epsilon)
                   for k, other in cells.items() if k != name))


def _flip_terms(a: Mapping[str, float], b: Mapping[str, float],
                epsilon: float) -> tuple[bool, bool, list[str]]:
    live = [k for k in MOMENT_NAMES
            if not math.isnan(a.get(k, math.nan)) and not math.isnan(b.get(k, math.nan))]
    no_worse = all(a[k] <= b[k] + epsilon for k in live)
    strictly = any(a[k] < b[k] - epsilon for k in live if a[k] != b[k])
    return no_worse, strictly, live


def _attribute_flip(a: Mapping[str, float], b: Mapping[str, float],
                    e_prev: float, e: float, now: bool,
                    name_a: str, name_b: str) -> dict[str, Any]:
    """Name the moment that actually drove a dominance change, and which term flipped.

    CORRECTED 2026-07-31. The first version reported `argmax(a - b)` unconditionally,
    which is right only when `no_worse` is what changed. When `strictly` is what changed
    -- an arm weakly better everywhere losing its strict edge -- the binding moment is the
    one with the most NEGATIVE gap, and the argmax came out at +0.000, naming a moment
    that had nothing to do with the flip.
    """
    nw0, st0, live = _flip_terms(a, b, e_prev)
    nw1, st1, _ = _flip_terms(a, b, e)
    if nw0 != nw1:
        term = "no_worse"
        crit = max(live, key=lambda k: a[k] - b[k])
    elif st0 != st1:
        term = "strictly"
        cand = [k for k in live if a[k] != b[k]] or live
        crit = min(cand, key=lambda k: a[k] - b[k])
    else:  # pragma: no cover - a flip implies one of the two changed
        term = "unattributed"
        crit = max(live, key=lambda k: abs(a[k] - b[k]))
    return {"pair": [name_a, name_b], "flips_at_epsilon": float(e),
            "now_dominates": bool(now), "term_that_flipped": term,
            "critical_moment": crit,
            "gap_dk": float(a[crit] - b[crit]),
            "binding_magnitude_dk": float(abs(a[crit] - b[crit]))}


def dominance_flips(cells: Mapping[str, Mapping[str, float]],
                    epsilons: Sequence[float] = EPSILON_BAND) -> list[dict[str, Any]]:
    """Every pair whose dominance changes across the band, with its critical moment.

    The boolean says "do not look"; this says WHICH comparison is fragile and BY HOW
    MUCH, which is the information the stability rule exists to protect. Amended in
    `contracts/epsilon_range_amendment_2026-07-31.json`.
    """
    out: list[dict[str, Any]] = []
    names = sorted(cells)
    prev_eps = float(epsilons[0]) if epsilons else 0.0
    for a in names:
        for b in names:
            if a == b:
                continue
            prev = None
            for e in epsilons:
                cur = dominates(cells[a], cells[b], e)
                if prev is not None and cur != prev:
                    out.append(_attribute_flip(cells[a], cells[b], prev_eps, e, cur, a, b))
                prev, prev_eps = cur, e
    return out


def epsilon_stability(cells: Mapping[str, Mapping[str, float]],
                      epsilons: Sequence[float] = EPSILON_BAND) -> dict[str, Any]:
    """Whether the non-dominated set is a property of the data or of epsilon."""
    sets = {e: non_dominated(cells, e) for e in epsilons}
    distinct = {tuple(v) for v in sets.values()}
    return {
        "by_epsilon": {str(e): v for e, v in sets.items()},
        "n_distinct_sets": len(distinct),
        "stable": len(distinct) == 1,
        "band": list(epsilons),
        "flips": dominance_flips(cells, epsilons),
    }
