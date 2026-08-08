"""Cheap structural tests for the corrective comparator; no scientific caches are opened."""

from __future__ import annotations

import numpy as np

import scripts.run_comparator_repair_v1 as repair


def test_frozen_factor_prior_has_positive_normalised_mass():
    prior = repair.level_prior(list(range(min(24, len(repair.G.BASE_CONFIGS)))))
    assert prior.shape == (len(repair.G.EXT_CONFIGS),)
    assert np.all(prior > 0.0)
    probabilities = prior / prior.sum()
    assert np.isclose(probabilities.sum(), 1.0)


def test_frozen_prior_digest_is_stable_and_replay_does_not_mutate_it():
    prior = repair.level_prior(list(range(min(24, len(repair.G.BASE_CONFIGS)))))
    before = repair.digest(prior)
    surface = repair.G.Surface(np.linspace(0.0, 1.0, len(repair.G.EXT_CONFIGS)))
    repair.G.marginal_replay(prior, surface, np.random.default_rng(7), 24)
    assert repair.digest(prior) == before


def test_current_case_tv_bound_is_about_distributions_not_auc():
    n = len(repair.G.EXT_CONFIGS)
    before = np.ones(n)
    after = before.copy()
    after[17] += 24.0
    tv = 0.5 * np.abs(before / before.sum() - after / after.sum()).sum()
    bound = 24.0 / (n + 24.0)
    assert tv <= bound + 1e-12

