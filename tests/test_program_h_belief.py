import copy

import numpy as np

from supply_chain.program_g import materialize_tape
from supply_chain.program_h import belief_rollout_actions, o0_features, tempo_belief


CELL = {"cell_id": "H-test", "signal_q": 0.75, "lead_weeks": 1,
        "surge_mult": 1.50, "persistence": "short", "r22_weekly_prob": 0.05}


def tape(seed=1060001):
    return materialize_tape(seed, CELL, 4, persistent=True)


def test_belief_normalizes_and_is_finite():
    t = tape()
    for w in range(4):
        b = tempo_belief(t, w)
        assert np.all(np.isfinite(b))
        assert np.allclose(b.sum(axis=1), 1.0)


def test_label_swap_exchanges_beliefs_and_actions():
    t = tape(); s = copy.deepcopy(t)
    s.signal = t.signal[:, ::-1].copy(); s.demand = t.demand[:, ::-1].copy()
    s.r22 = t.r22[:, ::-1].copy(); s.z = t.z[:, ::-1].copy()
    assert np.allclose(tempo_belief(t, 2)[:, :], tempo_belief(s, 2)[::-1, :])
    swap = {"A": "B", "B": "A", "HOLD": "HOLD"}
    assert tuple(swap[a] for a in belief_rollout_actions(t)) == belief_rollout_actions(s)


def test_o0_feature_schema_does_not_use_latent_or_realized_demand():
    t = tape(); x = o0_features(np.zeros(2), 10000.0, t, 1)
    s = copy.deepcopy(t); s.z[:] = 2; s.demand[:] *= 9
    y = o0_features(np.zeros(2), 10000.0, s, 1)
    assert np.array_equal(x, y)
