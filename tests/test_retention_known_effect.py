"""Pipeline-validation test: the transfer harness must recover a KNOWN retention
effect in a minimal hidden-regime Markov bandit (audit recommendation #4).

If this fails, no MFSC retained-vs-frozen null can be trusted.
"""

from __future__ import annotations

import pytest

from scripts.sanity_markov_retention import transfer_delta


@pytest.mark.slow
def test_harness_recovers_known_retention_effect():
    # Small but sufficient config: high rho should yield a clear positive head-start,
    # rho=0.5 should be ~0 (no predictable structure to remember).
    high = transfer_delta(0.9, seeds=[1, 2], n_blocks=14, train_per_block=200, horizon=30)
    flat = transfer_delta(0.5, seeds=[1, 2], n_blocks=14, train_per_block=200, horizon=30)
    assert high["delta_late_mean"] > 0.15, high
    assert high["delta_late_mean"] > flat["delta_late_mean"] + 0.10, (high, flat)
