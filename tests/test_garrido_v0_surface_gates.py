from __future__ import annotations

import numpy as np

from scripts.run_garrido_v0_surface_gates_v1 import _design_matrices, _loso_gain
from supply_chain.expanded_contract_controllers_v2 import ALL_POSTURES


def test_pairwise_design_strictly_extends_additive_design():
    additive, pairwise = _design_matrices()
    assert additive.shape == (len(ALL_POSTURES), 16)
    assert pairwise.shape == (len(ALL_POSTURES), 91)


def test_loso_interaction_gate_detects_a_nonadditive_surface():
    levels = np.asarray([posture[0] == posture[1] for posture in ALL_POSTURES], dtype=float)
    surfaces = np.stack([levels + seed * 1e-4 for seed in range(6)])
    gains = _loso_gain(surfaces)
    assert np.all(gains > 0.5)


def test_loso_interaction_gate_does_not_reward_an_additive_surface():
    values = np.asarray(
        [posture[0] / 1344.0 + 2 * posture[1] / 1344.0 for posture in ALL_POSTURES],
        dtype=float,
    )
    surfaces = np.stack([values + seed * 1e-4 for seed in range(6)])
    gains = _loso_gain(surfaces)
    assert np.max(np.abs(gains)) < 1e-8

