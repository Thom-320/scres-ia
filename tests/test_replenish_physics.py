"""Program K2 replenishment physics gates (correct physics: holding cost + lead time + real shelf)."""
import sys; sys.path.insert(0, ".")
import numpy as np
from supply_chain.replenish import (materialize_tape, central_cell, simulate, week_step,
    enumerate_oracle, sS_policy, new_state, ORDER_LEVELS, D0)

CELL = central_cell()


def test_determinism_and_tape_immutable():
    t = materialize_tape(1, CELL, 8); sha = t.sha
    assert simulate(t, (2,) * 8).J == simulate(t, (2,) * 8).J and t.sha == sha


def test_lead_time_order_does_not_serve_same_week():
    # with lead>=1, an order placed at w=0 into an empty system cannot serve w=0 demand
    t = materialize_tape(2, CELL, 4)
    r0 = simulate(t, (3, 0, 0, 0))
    # first-week service loss must equal first-week demand (nothing on hand, order still in transit)
    st = new_state(CELL); st, sl0, _ = week_step(t, 0, ORDER_LEVELS[3], st)
    assert abs(sl0 - t.demand[0]) < 1e-6


def test_capacity_caps_inventory_position():
    t = materialize_tape(3, CELL, 8); cap = CELL["cap_mult"] * D0
    st = new_state(CELL)
    for w in range(8):
        st, _, _ = week_step(t, w, ORDER_LEVELS[-1], st)
        assert st.onhand + sum(st.pipeline) <= cap + 1e-6


def test_holding_paid_on_onhand_and_nonneg():
    t = materialize_tape(4, CELL, 8)
    r = simulate(t, (2,) * 8)
    assert r.holding >= 0 and r.service_loss >= 0


def test_objective_is_p_sl_plus_h_holding():
    t = materialize_tape(5, CELL, 8); r = simulate(t, (1, 2, 0, 3, 1, 2, 1, 0))
    assert abs(r.J - (CELL["p"] * r.service_loss + CELL["h"] * r.holding)) < 1e-6


def test_oracle_lower_bounds_all_policies():
    t = materialize_tape(6, CELL, 6)
    best, _ = enumerate_oracle(t)
    for seq in [(0,)*6, (3,)*6, sS_policy(t, 1.0, 2.0)]:
        assert best <= simulate(t, seq).J + 1e-6


def test_zero_order_loses_all_demand():
    t = materialize_tape(7, CELL, 5)
    assert abs(simulate(t, (0,) * 5).service_loss - float(t.demand.sum())) < 1e-6
