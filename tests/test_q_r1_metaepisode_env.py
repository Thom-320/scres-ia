from __future__ import annotations

import numpy as np

from scripts.evaluate_program_q_replication import scheduler
from supply_chain.q_r1_metaepisode_env import (
    CAMPAIGNS_PER_METAEPISODE,
    DECISIONS_PER_CAMPAIGN,
    DECISIONS_PER_METAEPISODE,
    META_OBSERVATION_DIM,
    QRetainedMetaEpisodeEnv,
)
from supply_chain.retained_context_discovery import build_campaign_history


def _history(root: int = 7_570_801):
    return build_campaign_history(
        history_root=root,
        campaigns=CAMPAIGNS_PER_METAEPISODE,
        kappa=0.90,
        scheduler=scheduler(),
        regime_persistence=0.90,
        dominant_share=0.90,
    )


def test_metaepisode_has_internal_physical_resets_and_one_terminal_boundary() -> None:
    env = QRetainedMetaEpisodeEnv(histories=[_history()], scheduler=scheduler())
    observation, info = env.reset(options={"history_index": 0})
    assert observation.shape == (META_OBSERVATION_DIM,)
    assert observation[-1] == 1.0
    assert info["physical_reset"] is True

    rewards = []
    boundaries = 0
    terminal_steps = []
    previous_skeleton = env.current_campaign.skeleton.skeleton_sha256
    for step in range(DECISIONS_PER_METAEPISODE):
        observation, reward, terminated, truncated, info = env.step(0)
        assert truncated is False
        if info.get("campaign_complete"):
            rewards.append(reward)
        if info.get("campaign_boundary"):
            boundaries += 1
            assert info["physical_reset"] is True
            assert info["previous_skeleton_sha256"] == previous_skeleton
            assert info["next_skeleton_sha256"] != previous_skeleton
            previous_skeleton = info["next_skeleton_sha256"]
            assert observation[-1] == 1.0
        if terminated:
            terminal_steps.append(step + 1)

    assert len(rewards) == CAMPAIGNS_PER_METAEPISODE
    assert boundaries == CAMPAIGNS_PER_METAEPISODE - 1
    assert terminal_steps == [DECISIONS_PER_METAEPISODE]
    assert len(info["campaign_rows"]) == CAMPAIGNS_PER_METAEPISODE


def test_boundary_marker_is_zero_inside_campaign() -> None:
    env = QRetainedMetaEpisodeEnv(histories=[_history()], scheduler=scheduler())
    observation, _ = env.reset(options={"history_index": 0})
    assert observation[-1] == 1.0
    for _ in range(DECISIONS_PER_CAMPAIGN - 1):
        observation, reward, terminated, _truncated, info = env.step(1)
        assert reward == 0.0
        assert terminated is False
        assert info["campaign_boundary"] is False
        assert observation[-1] == 0.0


def test_history_selection_is_reproducible() -> None:
    histories = [_history(7_570_801), _history(7_570_802)]
    env_a = QRetainedMetaEpisodeEnv(
        histories=histories, scheduler=scheduler(), sampling_seed=91
    )
    env_b = QRetainedMetaEpisodeEnv(
        histories=histories, scheduler=scheduler(), sampling_seed=91
    )
    sequence_a = []
    sequence_b = []
    for _ in range(8):
        env_a.reset()
        env_b.reset()
        sequence_a.append(env_a.current_campaign.history_root)
        sequence_b.append(env_b.current_campaign.history_root)
    assert sequence_a == sequence_b
    assert len(set(sequence_a)) == 2


def test_meta_observation_exposes_no_privileged_scalar() -> None:
    env = QRetainedMetaEpisodeEnv(histories=[_history()], scheduler=scheduler())
    observation, _ = env.reset(options={"history_index": 0})
    assert np.all((0.0 <= observation) & (observation <= 1.0))
    assert observation.shape[0] == META_OBSERVATION_DIM
