import numpy as np

from scripts.run_track_b_contract_factorial import ARM_FROZEN_VALUES, FreezeDimsWrapper
from scripts.run_track_b_static_contract_search import Candidate, refinement_candidates
from supply_chain.external_env_interface import make_track_b_env
from scripts.run_track_b_crossed_eval import CANONICAL_ENV_KWARGS


def test_best_dispatch_anchor_decodes_to_requested_multipliers():
    values = ARM_FROZEN_VALUES["upstream_shift_best_dispatch"]
    assert values[6] == 1.0
    assert values[7] == 1.0 / 3.0
    assert 1.25 + 0.75 * values[6] == 2.0
    assert 1.25 + 0.75 * values[7] == 1.5


def test_freeze_wrapper_overrides_only_anchored_dimensions():
    env = FreezeDimsWrapper(
        make_track_b_env(**CANONICAL_ENV_KWARGS),
        ARM_FROZEN_VALUES["upstream_shift_best_dispatch"],
    )
    action = env.action(np.zeros(8, dtype=np.float32))
    np.testing.assert_allclose(action[:6], 0.0)
    assert action[6] == 1.0
    np.testing.assert_allclose(action[7], 1.0 / 3.0)
    env.close()


def test_refinement_is_single_bounded_neighborhood():
    leader = Candidate((0.0,) * 7, 2)
    candidates = refinement_candidates([leader], 0.15)
    assert len(candidates) == 17
    assert all(-1.0 <= value <= 1.0 for c in candidates for value in c.signals)
    assert {c.shift for c in candidates} == {1, 2, 3}
