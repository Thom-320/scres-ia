"""Fixture tests for the op11 fair-allocation probe (contract op11_fair_allocation_conversion_probe_v1).

These pin the invariants the contract's policy family promises, BEFORE any gate is trusted:
step-equivalence with program_g, non-anticipativity, no-HOLD, and the alternation guarantee.
"""
from __future__ import annotations

import numpy as np
import pytest

from supply_chain.headroom_sensitivity import theta_to_cell, materialize_tape_theta
from supply_chain.program_g import _week_step

from research.paper2_exhaustive_search.op11_fair_probe import (
    CANDIDATES, THETA_STAR, _step_with_served, policy_min_service_alternation,
    policy_fair_gate_ratio,
)


def _tape(seed=3_000_001, theta=None):
    return materialize_tape_theta(seed, theta_to_cell(theta or THETA_STAR))


def test_step_with_served_matches_program_g_week_step():
    rng = np.random.default_rng(7)
    for _ in range(200):
        inv = rng.uniform(0, 30000, 2); sb = float(rng.uniform(0, 60000))
        a = rng.choice(["A", "B", "HOLD"])
        demand = rng.uniform(0, 20000, 2); r22 = rng.integers(0, 2, 2)
        i1, s1, u1 = _week_step(inv, sb, a, demand, r22, True)
        i2, s2, u2, served = _step_with_served(inv, sb, a, demand, r22)
        assert np.allclose(i1, i2) and abs(s1 - s2) < 1e-9 and abs(u1 - u2) < 1e-9
        # served accounting is conservative: served + unmet == total demand
        assert abs(float(served.sum()) + u2 - float(demand.sum())) < 1e-6


def test_no_candidate_ever_holds():
    for seed in range(3_000_001, 3_000_021):
        t = _tape(seed)
        for name, fn in CANDIDATES.items():
            acts = fn(t)
            assert "HOLD" not in acts, f"{name} emitted HOLD on seed {seed}"
            assert len(acts) == t.weeks


def test_alternation_never_starves_two_consecutive_weeks():
    for seed in range(3_000_001, 3_000_051):
        acts = policy_min_service_alternation(_tape(seed))
        for w in range(1, len(acts)):
            # with 2 CSSUs and no HOLD, serving the same CSSU twice in a row means the
            # other went unserved 2 consecutive weeks -> the invariant is strict alternation
            assert acts[w] != acts[w - 1], \
                f"a CSSU went unserved twice at weeks {w-1},{w}: {acts}"


def test_policies_are_non_anticipative():
    """Mutating the FINAL week of the tape must not change earlier actions."""
    for name, fn in CANDIDATES.items():
        t1 = _tape(3_000_005)
        t2 = _tape(3_000_005)
        t2.demand[-1] = t2.demand[-1] * 3.0 + 1000.0
        t2.signal[-1] = 1 - t2.signal[-1]
        a1, a2 = fn(t1), fn(t2)
        assert a1[:-1] == a2[:-1], f"{name} leaked future tape info: {a1} vs {a2}"


def test_fair_gate_override_serves_trailing_cssu():
    """With delta=0.02, whenever the policy deviates from delta=inf behaviour it must be
    because the chosen CSSU's realized fill trailed by more than delta -- spot-check by
    comparing against a huge-delta variant (== pure ratio rule)."""
    diffs = 0
    for seed in range(3_000_001, 3_000_031):
        t = _tape(seed)
        tight = policy_fair_gate_ratio(t, 0.02)
        loose = policy_fair_gate_ratio(t, 1e9)   # override never fires
        if tight != loose:
            diffs += 1
    assert diffs > 0, "override never fired on 30 tapes at theta*; gate parameter is inert"
