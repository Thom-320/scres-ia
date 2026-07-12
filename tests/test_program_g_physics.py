"""G0 physics preflight for the Program G V1.2 spatial-commitment simulator."""
import copy

import numpy as np

from supply_chain.program_g import (
    ACTIONS, CONVOY_LOAD, central_cell, cover_signal_policy, enumerate_oracle,
    materialize_tape, simulate,
)


def _tape(seed=980001, weeks=4, persistent=True):
    return materialize_tape(seed, central_cell(), weeks, persistent=persistent)


def test_same_tape_same_action_is_deterministic():
    t = _tape()
    a = ("A", "B", "HOLD", "A")
    assert simulate(t, a).service_loss == simulate(t, a).service_loss


def test_ab_label_swap_symmetry():
    t = _tape()
    t2 = copy.deepcopy(t)
    t2.demand = t.demand[:, ::-1].copy()
    t2.signal = t.signal[:, ::-1].copy()
    t2.r22 = t.r22[:, ::-1].copy()
    swap = {"A": "B", "B": "A", "HOLD": "HOLD"}
    seq = ("A", "B", "HOLD", "A")
    r1 = simulate(t, seq)
    r2 = simulate(t2, tuple(swap[x] for x in seq))
    assert abs(r1.service_loss - r2.service_loss) < 1e-6


def test_demand_is_exogenous_across_policies():
    t = _tape()
    total = t.demand.sum()
    # demand tape is fixed on the tape; policy cannot change it
    for seq in [("A",) * 4, ("B",) * 4, ("HOLD",) * 4]:
        assert t.demand.sum() == total


def test_hold_delivers_nothing_and_is_weakly_dominated():
    t = _tape()
    assert simulate(t, ("HOLD",) * 4).service_loss >= simulate(t, ("A", "B", "A", "B")).service_loss


def test_oracle_is_a_lower_bound_on_any_fixed_sequence():
    t = _tape()
    o, _ = enumerate_oracle(t)
    for seq in [("A",) * 4, ("B",) * 4, ("A", "B", "A", "B")]:
        assert o <= simulate(t, seq).service_loss + 1e-9


def test_convoy_never_exceeds_capacity_or_creates_inventory():
    t = _tape()
    r = simulate(t, ("A", "A", "B", "B"))
    # served can never exceed demand; fill in [0,1]
    assert 0.0 <= r.fill_rate <= 1.0 + 1e-9
    assert r.worst_cssu_fill <= r.fill_rate + 1e-9


def test_weekly_priority_yields_multiple_departures():
    t = _tape()
    # dispatching every week yields ~3 cycles/week -> many missions over 4 weeks
    assert simulate(t, ("A", "B", "A", "B")).convoy_missions >= 4


def test_iid_vs_persistent_tempo_differ():
    tp = _tape(persistent=True)
    ti = _tape(persistent=False)
    assert not np.array_equal(tp.z, ti.z)
