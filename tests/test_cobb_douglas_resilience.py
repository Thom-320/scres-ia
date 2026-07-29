"""Tests for the port of Garrido's 2024 Cobb-Douglas factory resilience index.

The first test is the one that matters: it proves our reading of the exponent rule
by reproducing the published exponents from the published maxima.
"""
from __future__ import annotations

import math

import pytest

from supply_chain.cobb_douglas_resilience import (
    GARRIDO_2024_EXPONENTS,
    SHARE_PER_TERM,
    CobbDouglasRecorder,
    derive_exponents,
    kappa_dot,
    resilience_index,
    score_comparison_set,
)


def test_exponent_rule_reproduces_the_published_value_for_zeta():
    """§3.4: "zeta^max ~ 3,612, from which a*Ln3,612 = 0.20, resulting in a = 0.024"."""
    a = SHARE_PER_TERM / math.log(3612.0)
    assert round(a, 3) == GARRIDO_2024_EXPONENTS["zeta"]


def test_published_exponents_invert_to_plausible_maxima():
    """The rule is invertible: x_max = exp(0.20 / exponent).

    Every implied maximum must be positive and finite, and kappa_dot's must not
    exceed its set cardinality of 7 -- kappa_dot is a share renormalised to mean 1
    over seven substrategies, so 7 is its ceiling.
    """
    implied = {k: math.exp(SHARE_PER_TERM / v) for k, v in GARRIDO_2024_EXPONENTS.items()}
    assert all(v > 1.0 and math.isfinite(v) for v in implied.values())
    assert implied["kappa_dot"] < 7.0


def test_published_exponents_are_too_rounded_to_invert():
    """An independent reason never to copy his five numbers.

    The exponent enters through `exp(0.20/a)`, so rounding is amplified. His printed
    a = 0.024 differs from the unrounded 0.2/ln(3,612) = 0.02441 by 1.7%, but inverts
    to 4,160 against his stated 3,612 -- a 15% error in the maximum. The published
    exponents are not precise enough to reconstruct the scale they encode, quite
    apart from that scale being his model's rather than ours.
    """
    unrounded = SHARE_PER_TERM / math.log(3612.0)
    assert abs(unrounded - 0.024) / 0.024 < 0.02
    implied = math.exp(SHARE_PER_TERM / GARRIDO_2024_EXPONENTS["zeta"])
    assert implied / 3612.0 > 1.10


def test_derive_exponents_makes_each_term_contribute_one_fifth_at_its_maximum():
    maxima = {"zeta": 4.2e6, "epsilon": 9.1e4, "phi": 3.3e5, "tau": 12.0,
              "kappa_dot": 2.5}
    exps = derive_exponents(maxima)
    for name, x_max in maxima.items():
        assert exps[name] * math.log(x_max) == pytest.approx(SHARE_PER_TERM)


def test_derive_exponents_rejects_a_maximum_that_cannot_normalise():
    with pytest.raises(ValueError, match="exponent rule"):
        derive_exponents({"zeta": 1.0, "epsilon": 10.0, "phi": 10.0,
                          "tau": 10.0, "kappa_dot": 2.0})


def test_index_signs_follow_equation_4():
    """zeta and phi raise R; epsilon, tau and kappa_dot lower it."""
    exps = derive_exponents({"zeta": 1e6, "epsilon": 1e4, "phi": 1e5,
                             "tau": 10.0, "kappa_dot": 2.0})
    base = {"zeta": 1e5, "epsilon": 1e3, "phi": 1e4, "tau": 1.0, "kappa_dot": 1.0}
    r0 = resilience_index(base, exps)["R_cobb_douglas"]
    for name, expect_up in [("zeta", True), ("phi", True),
                            ("epsilon", False), ("tau", False), ("kappa_dot", False)]:
        bumped = dict(base)
        bumped[name] = base[name] * 2.0
        r1 = resilience_index(bumped, exps)["R_cobb_douglas"]
        assert (r1 > r0) is expect_up, name


def test_index_is_bounded_and_floors_protect_against_log_of_zero():
    exps = derive_exponents({"zeta": 1e6, "epsilon": 1e4, "phi": 1e5,
                             "tau": 10.0, "kappa_dot": 2.0})
    out = resilience_index(
        {"zeta": 0.0, "epsilon": 0.0, "phi": 0.0, "tau": 0.0, "kappa_dot": 1.0}, exps)
    assert math.isfinite(out["R_cobb_douglas"])
    assert 0.0 < out["R_cobb_douglas"] < 1.0


def test_kappa_dot_has_set_mean_one_and_matches_the_papers_factor():
    """Eq. (5) writes `7*kappa/sum kappa` for a set of seven; the 7 is |S|."""
    kd = kappa_dot({f"s{i}": 100.0 for i in range(7)})
    assert all(v == pytest.approx(1.0) for v in kd.values())
    kd2 = kappa_dot({"a": 100.0, "b": 300.0})
    assert kd2["a"] == pytest.approx(0.5) and kd2["b"] == pytest.approx(1.5)
    assert sum(kd2.values()) == pytest.approx(len(kd2))


def test_kappa_dot_is_set_relative_so_R_moves_when_the_set_changes():
    """The hazard that forces the comparison set to be frozen before evaluation."""
    exps = derive_exponents({"zeta": 1e6, "epsilon": 1e4, "phi": 1e5,
                             "tau": 10.0, "kappa_dot": 3.0})
    def agg(k):
        return {"zeta": 1e5, "epsilon": 1e3, "phi": 1e4, "tau": 1.0, "kappa": k}

    small = score_comparison_set({"a": agg(100.0), "b": agg(100.0)}, exps)
    large = score_comparison_set(
        {"a": agg(100.0), "b": agg(100.0), "c": agg(1000.0)}, exps)
    assert small["a"]["R_cobb_douglas"] != large["a"]["R_cobb_douglas"]


class _FakeContainer:
    def __init__(self, level: float) -> None:
        self.level = level


class _FakeSim:
    """Minimal stand-in exposing only the public attributes the recorder reads."""

    def __init__(self, *, produced=0.0, demanded=0.0, backorder=0.0,
                 rations=0.0, raw=0.0, shifts=1):
        self.total_produced = produced
        self.total_demanded = demanded
        self.pending_backorder_qty = backorder
        self.rations_sb = _FakeContainer(rations)
        self.rations_al = _FakeContainer(0.0)
        self.rations_sb_dispatch = _FakeContainer(0.0)
        self.rations_cssu = _FakeContainer(0.0)
        self.rations_theatre = _FakeContainer(0.0)
        self.raw_material_wdc = _FakeContainer(raw)
        self.raw_material_al = _FakeContainer(0.0)
        self.params = {"assembly_shifts": shifts}
        self.shifts = shifts


def test_recorder_converts_raw_material_to_ration_equivalents():
    """A 12-part kit counts once, not twelve times."""
    rec = CobbDouglasRecorder()
    row = rec.sample(_FakeSim(rations=1000.0, raw=12_000.0))
    assert row["I_t"] == pytest.approx(2000.0)


def test_recorder_spare_capacity_tracks_the_shift_ladder():
    """Table 6.20: 2,564 / 5,128 / 7,692 rations per day at 1 / 2 / 3 shifts."""
    spare = {}
    for s in (1, 2, 3):
        rec = CobbDouglasRecorder()
        row = rec.sample(_FakeSim(produced=2000.0, shifts=s))
        spare[s] = row["U_t"]
    assert spare[1] == pytest.approx(564.0)
    assert spare[2] == pytest.approx(3128.0)
    assert spare[3] == pytest.approx(5692.0)
    assert spare[3] > spare[2] > spare[1]


def test_shift_three_is_not_free_under_the_cost_term():
    """The concrete defect this index is meant to repair.

    ReT prices no capacity, so nothing restrains always choosing shift 3. Garrido's
    kappa charges spare capacity through `c_u * U_t`, so the same production on a
    three-shift installation costs strictly more than on one shift.
    """
    costs = {}
    for s in (1, 3):
        rec = CobbDouglasRecorder()
        rec.sample(_FakeSim(produced=2000.0, shifts=s))
        costs[s] = rec.aggregate()["kappa"]
    assert costs[3] > costs[1]


def test_recorder_deltas_are_per_period_not_cumulative():
    rec = CobbDouglasRecorder()
    rec.sample(_FakeSim(produced=1000.0, demanded=900.0))
    row = rec.sample(_FakeSim(produced=2500.0, demanded=1800.0))
    assert row["P_t"] == pytest.approx(1500.0)
    assert row["GR_t"] == pytest.approx(900.0)


def test_net_requirements_follow_algorithm_2_line_23():
    """NR_t = max(GR_t - I_{t-1} + B_{t-1}, 0); stock on hand offsets the requirement."""
    rec = CobbDouglasRecorder()
    rec.sample(_FakeSim(rations=200.0, backorder=50.0))
    row = rec.sample(_FakeSim(demanded=1000.0, rations=200.0, backorder=50.0))
    assert row["NR_t"] == pytest.approx(1000.0 - 200.0 + 50.0)


def test_tau_is_zero_when_nothing_is_demanded():
    rec = CobbDouglasRecorder()
    row = rec.sample(_FakeSim())
    assert row["tau_t"] == 0.0


def test_shift_changes_are_recorded_as_hiring_and_firing():
    rec = CobbDouglasRecorder()
    rec.sample(_FakeSim(shifts=1))
    up = rec.sample(_FakeSim(shifts=3))
    down = rec.sample(_FakeSim(shifts=1))
    assert (up["H_t"], up["L_t"]) == (2.0, 0.0)
    assert (down["H_t"], down["L_t"]) == (0.0, 2.0)


def test_aggregate_refuses_an_empty_horizon():
    with pytest.raises(ValueError, match="no periods"):
        CobbDouglasRecorder().aggregate()
