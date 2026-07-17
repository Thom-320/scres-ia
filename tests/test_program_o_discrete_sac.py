from pathlib import Path

import numpy as np
import torch

from research.program_o_david_workbench import DEV_SEED_MIN, make_development_env
from research.program_o_discrete_sac import (
    CategoricalSAC,
    DavidDMPLAEncoder,
    HistoryPolicyAdapter,
    HistoryStackWrapper,
    MLPEncoder,
)


ROOT = Path(__file__).resolve().parent.parent


def test_history_wrapper_and_categorical_sac_smoke() -> None:
    base = make_development_env(
        root=ROOT,
        tape_seed_start=DEV_SEED_MIN,
        tape_seed_end=DEV_SEED_MIN + 4,
    )
    env = HistoryStackWrapper(base, history_len=2)
    observation, _ = env.reset()
    assert observation.shape == (42,)
    model = CategoricalSAC(
        observation_dim=42,
        action_dim=4,
        encoder_factory=lambda: MLPEncoder(42, features_dim=32),
        device="cpu",
        seed=7,
    )
    summary = model.learn(
        env,
        total_steps=16,
        learning_starts=8,
        batch_size=8,
        replay_capacity=64,
    )
    assert summary.total_steps == 16
    assert summary.episodes == 2
    assert model.predict(observation) in range(4)
    adapter = HistoryPolicyAdapter(model, label="sac", history_len=2)
    adapter.reset_policy_state()
    assert adapter.predict_action(np.zeros(21, dtype=np.float32)) in range(4)


def test_david_encoder_supports_all_positional_modes() -> None:
    observations = torch.zeros(3, 21 * 8)
    for mode in ("learned", "sinusoidal", "none"):
        encoder = DavidDMPLAEncoder(
            21 * 8,
            obs_dim=21,
            history_len=8,
            features_dim=32,
            heads=4,
            layers=1,
            positional_mode=mode,
        )
        assert encoder(observations).shape == (3, 32)
