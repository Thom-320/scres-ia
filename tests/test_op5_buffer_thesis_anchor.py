"""Test: a5 (op5_q multiplier) should scale the Op5 buffer target as a continuous
relaxation of Table 6.16 (Garrido-Rios 2017, p. 106).

The thesis replenishes the raw-material buffer at Op5,j every
168/336/504/672/1344 h to a target level (the I_{t,S} values for Op5,j:
15,360 / 30,720 / 46,080 / 61,440 / 122,880). The repo exposes this as
a thesis-anchored continuous multiplier m = 1.0 + 0.5 * a5 over the
baseline target, yielding m ∈ [0.5, 1.5] for a5 ∈ [-1, 1].

This test verifies the end-to-end mapping:
  a5 = +1.0 -> op5_rm target = 1.5 * baseline
  a5 = -1.0 -> op5_rm target = 0.5 * baseline
  a5 =  0.0 -> op5_rm target = 1.0 * baseline
"""

from __future__ import annotations

import numpy as np
import pytest

from supply_chain.env_experimental_shifts import MFSCGymEnvShifts
from supply_chain.config import (
    SIMULATION_HORIZON,
    VALIDATION_TABLE_6_10,
)

# Thesis Table 6.16 default: middle level (I_{504,1} = 46,080 per RM) at Op5,j.
THESIS_OP5_DEFAULT = 46_080.0


def _make_env(max_steps: int = 2) -> MFSCGymEnvShifts:
    """Create an env with current risk + thesis annualization (thesis-anchored)."""
    return MFSCGymEnvShifts(
        step_size_hours=168,
        max_steps=max_steps,
        risk_level="current",
        stochastic_pt=False,
        reward_mode="ReT_seq_v1",
        observation_version="v7",
        initial_buffers={"op3_rm": 46_080, "op5_rm": 46_080, "op9_rations": 47_250},
    )


def test_op5_multiplier_at_max_doubles_buffer_target() -> None:
    """a5 = +1.0 should set op5_rm target to 1.5x baseline (thesis-anchored centred rule)."""
    env = _make_env(max_steps=1)
    env.reset(seed=42)
    assert env.sim is not None

    # Read the original Op5 baseline target.
    base_op5 = env.sim._op5_rm_base
    assert base_op5 is not None and base_op5 > 0.0
    expected_target = 1.5 * base_op5  # m = 1 + 0.5*1 = 1.5

    # a5 = +1.0 → multiplier = 1 + 0.5 = 1.5
    action = np.array([0.0, 0.0, 0.0, 0.0, +1.0, 0.0], dtype=np.float32)
    env.step(action)

    assert env.sim.inventory_buffer_targets["op5_rm"] == pytest.approx(
        expected_target, rel=1e-6
    )


def test_op5_multiplier_at_min_halves_buffer_target() -> None:
    """a5 = -1.0 should set op5_rm target to 0.5x baseline."""
    env = _make_env(max_steps=1)
    env.reset(seed=42)
    assert env.sim is not None

    base_op5 = env.sim._op5_rm_base
    assert base_op5 is not None and base_op5 > 0.0
    expected_target = 0.5 * base_op5  # m = 1 + 0.5*(-1) = 0.5

    # a5 = -1.0 → multiplier = 1 - 0.5 = 0.5
    action = np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float32)
    env.step(action)

    assert env.sim.inventory_buffer_targets["op5_rm"] == pytest.approx(
        expected_target, rel=1e-6
    )


def test_op5_multiplier_at_zero_yields_1_0x() -> None:
    """a5 = 0.0 should set op5_rm target to 1.0x baseline (neutral)."""
    env = _make_env(max_steps=1)
    env.reset(seed=42)
    assert env.sim is not None

    base_op5 = env.sim._op5_rm_base
    assert base_op5 is not None and base_op5 > 0.0
    expected_target = 1.0 * base_op5  # m = 1 + 0.5*0 = 1.0

    # a5 = 0.0 → multiplier = 1.0 (neutral, leaves thesis buffer unchanged)
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    env.step(action)

    assert env.sim.inventory_buffer_targets["op5_rm"] == pytest.approx(
        expected_target, rel=1e-6
    )


def test_op5_baseline_matches_thesis_table_6_16() -> None:
    """The default Op5 buffer target should be the middle thesis level
    (I_{504,1} = 46,080 per RM) for Op3/Op5 rm1..rm12 buffers.

    Garrido-Rios 2017, Table 6.16 (p. 106): Op5,j holds rm1..rm12 at
    15,360 / 30,720 / 46,080 / 61,440 / 122,880 for the 5 inventory levels.
    The repo's default config uses the middle level (I_{504,1}).
    """
    env = _make_env(max_steps=1)
    env.reset(seed=42)
    assert env.sim is not None

    base_op5 = env.sim._op5_rm_base
    assert base_op5 is not None
    # Default should be the middle thesis level: 46,080 per RM (I_{504,1}).
    assert base_op5 == pytest.approx(THESIS_OP5_DEFAULT, rel=1e-6)


def test_op5_multiplier_persists_across_steps() -> None:
    """Once set, the op5_rm target should persist until the next action."""
    env = _make_env(max_steps=3)
    env.reset(seed=42)
    assert env.sim is not None

    base_op5 = env.sim._op5_rm_base
    assert base_op5 is not None and base_op5 > 0.0

    # Step 1: set a5 = +1.0 → target becomes 1.5x baseline
    action = np.array([0.0, 0.0, 0.0, 0.0, +1.0, 0.0], dtype=np.float32)
    env.step(action)
    target_after_step_1 = env.sim.inventory_buffer_targets["op5_rm"]
    assert target_after_step_1 == pytest.approx(1.5 * base_op5, rel=1e-6)

    # Step 2: send a5 = 0.0 → target becomes 1.0x baseline (neutral).
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    env.step(action)
    target_after_step_2 = env.sim.inventory_buffer_targets["op5_rm"]
    assert target_after_step_2 == pytest.approx(1.0 * base_op5, rel=1e-6)
