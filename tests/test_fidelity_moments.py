"""The multi-moment comparison must be a dominance relation, not a score."""
from __future__ import annotations

import math

import pytest

from supply_chain.fidelity_moments import (
    MOMENT_NAMES,
    MomentReference,
    build_reference,
    discrepancies,
    dominates,
    epsilon_stability,
    moments_from_rows,
    non_dominated,
)


def test_moments_use_one_definition_for_both_sides():
    m = moments_from_rows(apj=[0, 0, 5, 0], rpj=[0, 10, 0, 30], ret=[0.1, 2.0, 0.3, 0.4])
    assert m["autotomy_share"] == pytest.approx(0.25)
    assert m["ret_above_one_share"] == pytest.approx(0.25)
    assert m["rpj_mean"] == pytest.approx(20.0)     # positives only
    assert m["scored_rows"] == 4.0


def test_a_moment_that_does_not_vary_cannot_set_a_scale():
    ref = MomentReference(mean=1.0, spread=0.0, n_sheets=9)
    assert ref.degenerate
    d = discrepancies({n: 1.0 for n in MOMENT_NAMES},
                      {n: 0.0 for n in MOMENT_NAMES},
                      {n: ref for n in MOMENT_NAMES})
    assert all(math.isnan(v) for v in d.values())


def test_discrepancy_is_in_combined_standard_errors():
    ref = {n: MomentReference(mean=0.0, spread=3.0, n_sheets=9) for n in MOMENT_NAMES}
    # spread^2/n = 1, our se = 0 -> combined = 1, so a gap of 2 reads as 2.0
    d = discrepancies({n: 2.0 for n in MOMENT_NAMES},
                      {n: 0.0 for n in MOMENT_NAMES}, ref)
    assert all(v == pytest.approx(2.0) for v in d.values())


def test_our_own_uncertainty_shrinks_the_discrepancy():
    ref = {n: MomentReference(mean=0.0, spread=3.0, n_sheets=9) for n in MOMENT_NAMES}
    tight = discrepancies({n: 2.0 for n in MOMENT_NAMES},
                          {n: 0.0 for n in MOMENT_NAMES}, ref)
    loose = discrepancies({n: 2.0 for n in MOMENT_NAMES},
                          {n: 4.0 for n in MOMENT_NAMES}, ref)
    assert all(loose[n] < tight[n] for n in MOMENT_NAMES)


def test_dominance_needs_strictly_better_outside_epsilon():
    a = {n: 1.0 for n in MOMENT_NAMES}
    b = {n: 1.2 for n in MOMENT_NAMES}          # better, but inside epsilon 0.5
    assert not dominates(a, b, epsilon=0.5)
    c = {n: 3.0 for n in MOMENT_NAMES}          # better outside epsilon
    assert dominates(a, c, epsilon=0.5)
    assert not dominates(c, a, epsilon=0.5)


def test_a_cell_cannot_win_or_lose_on_a_degenerate_moment():
    a = {n: math.nan for n in MOMENT_NAMES}
    b = {n: 10.0 for n in MOMENT_NAMES}
    assert not dominates(a, b) and not dominates(b, a)


def test_output_is_a_set_and_keeps_incomparable_cells():
    cells = {
        "trades_one_for_another": {**{n: 1.0 for n in MOMENT_NAMES},
                                   "autotomy_share": 0.0, "ret_mean": 5.0},
        "the_other_way": {**{n: 1.0 for n in MOMENT_NAMES},
                          "autotomy_share": 5.0, "ret_mean": 0.0},
        "worse_at_everything": {n: 9.0 for n in MOMENT_NAMES},
    }
    nd = non_dominated(cells)
    assert "worse_at_everything" not in nd
    assert len(nd) == 2, "incomparable cells must both survive, not be collapsed"


def test_epsilon_sensitivity_is_reported_not_hidden():
    cells = {"a": {n: 1.0 for n in MOMENT_NAMES},
             "b": {n: 1.6 for n in MOMENT_NAMES}}
    out = epsilon_stability(cells, [0.1, 0.5, 2.0])
    assert not out["stable"], "a set that moves with epsilon must be flagged"
    assert out["n_distinct_sets"] > 1


def test_build_reference_refuses_a_single_sheet():
    with pytest.raises(ValueError, match=">= 2"):
        build_reference({"CF1": {n: 1.0 for n in MOMENT_NAMES}}, ["CF1"])
