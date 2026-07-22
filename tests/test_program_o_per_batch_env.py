from __future__ import annotations

import numpy as np

from scripts.evaluate_program_o_ret_learner import scheduler
from supply_chain.program_o_per_batch_env import (
    OBSERVATION_DIM,
    PATTERN_SCHEDULER,
    ProgramOPerBatchEnv,
    bits_to_weekly_patterns,
    structured_action,
    structured_configurations,
)
from supply_chain.program_o_full_des_transducer import simulate_full_des_frontier


def make_env() -> ProgramOPerBatchEnv:
    return ProgramOPerBatchEnv(
        scheduler=scheduler(), tape_seed_start=972_000_001, tape_seed_end=972_000_100
    )


def rollout(actions: list[int]):
    env = make_env()
    observation, info = env.reset(
        options={"tape_seed": 982_000_001, "cell_index": 2}
    )
    observations = [observation]
    for action in actions:
        observation, reward, terminated, truncated, step_info = env.step(action)
        observations.append(observation)
    return env, observations, reward, terminated, truncated, step_info, info


def test_24_real_epochs_and_terminal_exact_transducer_parity() -> None:
    actions = [0, 1, 0, 1, 1, 0] * 4
    env, observations, reward, terminated, truncated, info, reset_info = rollout(actions)
    assert len(observations) == 25
    assert all(row.shape == (OBSERVATION_DIM,) for row in observations)
    assert terminated and not truncated
    assert info["actions"] == actions
    assert info["weekly_patterns"] == list(bits_to_weekly_patterns(actions))
    direct = simulate_full_des_frontier(
        skeleton=env._skeleton,
        scheduler=PATTERN_SCHEDULER,
        calendars=np.asarray([info["weekly_patterns"]], dtype=np.uint8),
    )
    assert reward == direct["ret_visible"][0]
    for key in ("ret_visible", "worst_product_fill", "service_loss_auc", "unresolved_orders"):
        assert info["metrics"][key] == direct[key][0]
    assert reset_info["cell_id"] == "rho90_share90"


def test_action_changes_physical_observation_before_next_decision() -> None:
    env_h = make_env()
    env_c = make_env()
    options = {"tape_seed": 982_000_002, "cell_index": 2}
    first_h, _ = env_h.reset(options=options)
    first_c, _ = env_c.reset(options=options)
    np.testing.assert_array_equal(first_h, first_c)
    next_h, *_ = env_h.step(0)
    next_c, *_ = env_c.step(1)
    assert not np.array_equal(next_h, next_c)
    raw_h = env_h.raw_observation()
    raw_c = env_c.raw_observation()
    assert not np.array_equal(raw_h["on_hand"], raw_c["on_hand"])


def test_every_epoch_advances_to_a_distinct_physical_batch_time() -> None:
    env = make_env()
    _, info = env.reset(options={"tape_seed": 982_000_003, "cell_index": 0})
    times = [info["raw_observation"]["decision_time"]]
    for _ in range(23):
        _, _, terminated, _, info = env.step(0)
        assert not terminated
        times.append(info["raw_observation"]["decision_time"])
    assert len(times) == 24
    assert all(right > left for left, right in zip(times, times[1:]))


def test_structured_family_uses_same_binary_action_and_observation() -> None:
    env = make_env()
    env.reset(options={"tape_seed": 982_000_004, "cell_index": 1})
    raw = env.raw_observation()
    configs = structured_configurations()
    assert len(configs) >= 3
    assert all(structured_action(raw, config) in (0, 1) for config in configs)


def test_bits_to_weekly_patterns_preserves_position() -> None:
    bits = [1, 0, 0, 0, 1, 0, 0, 0, 1] + [0] * 15
    assert bits_to_weekly_patterns(bits)[:3] == (1, 2, 4)
