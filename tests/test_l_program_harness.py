from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.analyze_l_program import two_way_bootstrap
from scripts.run_l_program_cycles import (
    build_model,
    reset_optimizer_state,
    torch_state_digest,
    train_on_tape,
)
from supply_chain.l_program_env import (
    CampaignTape,
    FixedNormalizerStats,
    RewardScales,
)


def _tape(seed: int) -> CampaignTape:
    return CampaignTape(
        campaign_id=f"harness-{seed}",
        family="R1",
        risk_level="current",
        base_seed=seed,
        horizon_weeks=2,
        split="smoke",
    )


def test_optimizer_reset_does_not_change_actor_critic_weights() -> None:
    tape = _tape(773001)
    normalizer = FixedNormalizerStats.identity()
    scales = RewardScales()
    model = build_model(
        tapes=[tape],
        buffer_level=0,
        normalizer=normalizer,
        reward_scales=scales,
        lambda_shift=0.25,
        learner_seed=1,
    )
    try:
        train_on_tape(
            model,
            tape=tape,
            buffer_level=0,
            normalizer=normalizer,
            reward_scales=scales,
            lambda_shift=0.25,
        )
        policy_before = torch_state_digest(model.policy.state_dict())
        assert model.policy.optimizer.state
        reset_optimizer_state(model)
        assert torch_state_digest(model.policy.state_dict()) == policy_before
        assert not model.policy.optimizer.state
    finally:
        model.get_env().close()


def test_two_way_bootstrap_detects_known_positive_and_null_effects() -> None:
    positive = np.full((5, 12), 0.5)
    null = np.zeros((5, 12))
    positive_result = two_way_bootstrap(positive, n_boot=500)
    null_result = two_way_bootstrap(null, n_boot=500)
    assert positive_result["ci95"][0] > 0.0
    assert null_result["mean"] == 0.0
    assert null_result["ci95"] == [0.0, 0.0]


def test_toy_retention_protocol_detects_memory_and_returns_null_without_it() -> None:
    """Falsify the inference gate on known-memory and known-null toy panels."""
    seed_effect = np.arange(5, dtype=float)[:, None] * 0.001
    tape_effect = np.arange(12, dtype=float)[None, :] * 0.0001
    reset = seed_effect + tape_effect
    persistent_with_memory = reset + 0.05
    persistent_without_memory = reset.copy()

    known_memory = two_way_bootstrap(persistent_with_memory - reset, n_boot=500)
    known_null = two_way_bootstrap(persistent_without_memory - reset, n_boot=500)
    assert known_memory["ci95"][0] > 0.0
    assert known_null["mean"] == 0.0
    assert known_null["ci95"] == [0.0, 0.0]


def test_l_runner_never_mentions_forbidden_metric_substitution() -> None:
    source = Path("scripts/run_l_program_cycles.py").read_text(encoding="utf-8")
    # It may name the forbidden field only in the audit manifest, never index it.
    assert '["order_level_ret_mean"]' not in source
