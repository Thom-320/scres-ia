with open('tests/test_control_reward_benchmark.py', 'r') as f:
    code = f.read()

# I will replace the two tests
test_1_old = """def test_pbrs_phi_zero_above_target() -> None:
    \"\"\"When fill_rate >= tau, Φ should be 0 (no shaping above target).\"\"\"
    from supply_chain.env_experimental_shifts import MFSCGymEnvShifts

    env = MFSCGymEnvShifts(
        reward_mode="control_v1_pbrs",
        pbrs_alpha=1.0,
        pbrs_beta=0.95,
        pbrs_variant="cumulative",
    )
    obs = np.zeros(15, dtype=np.float32)
    obs[6] = 0.98  # above tau
    phi = env._compute_phi_cumulative(obs)
    assert phi == 0.0"""

test_1_new = """def test_pbrs_phi_value() -> None:
    \"\"\"Φ should be α * fill_rate - β * backorder_rate.\"\"\"
    from supply_chain.env_experimental_shifts import MFSCGymEnvShifts

    env = MFSCGymEnvShifts(
        reward_mode="control_v1_pbrs",
        pbrs_alpha=1.0,
        pbrs_beta=0.5,
        pbrs_variant="cumulative",
    )
    # create obs with length 17 so obs[16] is available
    obs = np.zeros(17, dtype=np.float32)
    obs[6] = 0.8  # fill_rate
    obs[16] = 0.2  # backorder_rate
    phi = env._compute_phi_cumulative(obs)
    expected = 1.0 * 0.8 - 0.5 * 0.2
    assert abs(phi - expected) < 1e-6"""

test_2_old = """def test_pbrs_phi_deficit_below_target() -> None:
    \"\"\"When fill_rate < tau, Φ should be -α * (τ - FR) / τ.\"\"\"
    from supply_chain.env_experimental_shifts import MFSCGymEnvShifts

    env = MFSCGymEnvShifts(
        reward_mode="control_v1_pbrs",
        pbrs_alpha=2.0,
        pbrs_beta=0.95,
        pbrs_variant="cumulative",
    )
    obs = np.zeros(15, dtype=np.float32)
    obs[6] = 0.80
    phi = env._compute_phi_cumulative(obs)
    expected = -2.0 * (0.95 - 0.80) / 0.95
    assert abs(phi - expected) < 1e-6"""

test_2_new = """def test_pbrs_phi_value_fallback() -> None:
    \"\"\"Φ should handle short obs safely.\"\"\"
    from supply_chain.env_experimental_shifts import MFSCGymEnvShifts

    env = MFSCGymEnvShifts(
        reward_mode="control_v1_pbrs",
        pbrs_alpha=2.0,
        pbrs_beta=0.5,
        pbrs_variant="cumulative",
    )
    obs = np.zeros(15, dtype=np.float32)
    obs[6] = 0.80
    phi = env._compute_phi_cumulative(obs)
    expected = 2.0 * 0.80 - 0.5 * 0.0 # fallback 0 for BO
    assert abs(phi - expected) < 1e-6"""

code = code.replace(test_1_old, test_1_new)
code = code.replace(test_2_old, test_2_new)

with open('tests/test_control_reward_benchmark.py', 'w') as f:
    f.write(code)
