import numpy as np

from supply_chain.track_bp_env import (
    BUFFER_KEYS,
    TRACK_B_FIXED_BUFFER_ACTION_CONTRACT,
    make_track_b_fixed_buffer_env,
)


def test_fixed_buffer_contract_keeps_eight_actions_and_emits_posture():
    env = make_track_b_fixed_buffer_env(
        fixed_fracs=(0.1, 0.2, 0.3),
        max_steps=2,
        enabled_risks=(),
    )
    try:
        env.reset(seed=7)
        assert env.action_contract == TRACK_B_FIXED_BUFFER_ACTION_CONTRACT
        assert env.action_space.shape == (8,)
        _, _, _, _, info = env.step(np.zeros(8, dtype=np.float32))
        got = info["track_bp_fixed_buffer_fracs"]
        assert list(got) == list(BUFFER_KEYS)
        assert np.allclose(list(got.values()), [0.1, 0.2, 0.3])
    finally:
        env.close()
