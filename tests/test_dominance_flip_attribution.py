"""A reported flip must name the moment that actually drove it.

`dominance_flips` first reported `argmax(a - b)` unconditionally. That is right only when
`no_worse` is the term that changed; when `strictly` changes -- an arm weakly better
everywhere losing its strict edge -- the binding moment is the one with the most NEGATIVE
gap, and the argmax came out at exactly +0.000, naming a moment with nothing to do with the
flip. Two rows of the 2026-07-31 delta run showed that.

The invariant pinned here is self-checking: if the attribution is right, the binding gap
must be the one that CROSSED epsilon, so its magnitude has to fall between the previous
swept epsilon and the one where the flip is reported.
"""
from __future__ import annotations

import pytest

from supply_chain.fidelity_moments import EPSILON_BAND, dominance_flips, dominates


def _cells():
    """Two arms differing on one moment, plus a third that is weakly better everywhere."""
    return {
        "A": {"autotomy_share": 5.0, "ret_mean": 1.00, "ret_above_one_share": 3.0,
              "rpj_mean": 8.0, "rpj_p95": 9.0, "scored_orders_per_year": 1.0},
        "B": {"autotomy_share": 5.0, "ret_mean": 1.55, "ret_above_one_share": 3.0,
              "rpj_mean": 8.0, "rpj_p95": 9.0, "scored_orders_per_year": 1.0},
        "C": {"autotomy_share": 5.0, "ret_mean": 1.00, "ret_above_one_share": 3.0,
              "rpj_mean": 8.0, "rpj_p95": 9.0, "scored_orders_per_year": 1.0},
    }


def test_binding_gap_crossed_the_epsilon_where_the_flip_is_reported():
    """The self-consistency check the old implementation could not pass."""
    flips = dominance_flips(_cells(), EPSILON_BAND)
    assert flips, "the fixture must produce at least one flip"
    band = list(EPSILON_BAND)
    for f in flips:
        e = f["flips_at_epsilon"]
        prev = band[band.index(e) - 1]
        mag = f["binding_magnitude_dk"]
        assert prev < mag <= e + 1e-9, (
            f"{f['pair']} flips at {e} but its binding gap is {mag}, outside ({prev}, {e}]")


def test_flip_names_the_term_that_changed():
    for f in dominance_flips(_cells(), EPSILON_BAND):
        assert f["term_that_flipped"] in ("no_worse", "strictly")


def test_zero_gap_moments_cannot_supply_strictness():
    """Identical arms can never dominate, at any epsilon in the band."""
    c = _cells()
    for e in EPSILON_BAND:
        assert not dominates(c["A"], c["C"], e)
        assert not dominates(c["C"], c["A"], e)


@pytest.mark.parametrize("epsilon", EPSILON_BAND)
def test_band_stays_inside_the_regime_where_both_terms_can_bind(epsilon):
    """Amendment rationale: an epsilon above ~1 combined SE degenerates the check."""
    assert 0.25 <= epsilon <= 0.75
