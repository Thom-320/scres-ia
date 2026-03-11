from __future__ import annotations

import numpy as np

from scripts.control_reward_seed_inference import (
    exact_sign_flip_pvalue,
    paired_bootstrap_ci,
)


def test_exact_sign_flip_pvalue_detects_consistent_positive_diffs() -> None:
    diffs = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    p_value = exact_sign_flip_pvalue(diffs)
    assert 0.0 <= p_value <= 1.0
    assert p_value < 0.1


def test_paired_bootstrap_ci_contains_mean_for_constant_series() -> None:
    rng = np.random.default_rng(123)
    diffs = np.asarray([2.5, 2.5, 2.5, 2.5], dtype=np.float64)
    low, high = paired_bootstrap_ci(diffs, n_samples=1000, rng=rng)
    assert low == 2.5
    assert high == 2.5
