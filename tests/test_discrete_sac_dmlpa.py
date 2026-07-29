from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from scripts.discrete_sac_dmlpa import DiscreteSACAgent, DiscreteSACConfig


class TinyDiscreteEnv(gym.Env[np.ndarray, int]):
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Discrete(4)

    def __init__(self) -> None:
        self.step_index = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_index = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self.step_index += 1
        terminated = self.step_index == 2
        observation = np.full(4, self.step_index / 2, dtype=np.float32)
        reward = float(action == 2) if terminated else 0.0
        return observation, reward, terminated, False, {}


def extractor_factory() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU())


def tiny_config() -> DiscreteSACConfig:
    return DiscreteSACConfig(
        learning_rate=1e-3,
        buffer_size=128,
        batch_size=4,
        learning_starts=4,
        hidden_dims=(8,),
    )


def test_discrete_sac_updates_actor_twins_targets_and_temperature(tmp_path: Path) -> None:
    agent = DiscreteSACAgent(TinyDiscreteEnv(), extractor_factory, 8, 17, tiny_config())
    critic_before = next(agent.critic1.parameters()).detach().clone()
    target_before = next(agent.target_critic1.parameters()).detach().clone()
    alpha_before = float(agent.alpha.detach())
    agent.learn(20)
    assert agent.gradient_updates > 0
    assert not torch.equal(critic_before, next(agent.critic1.parameters()).detach())
    assert not torch.equal(target_before, next(agent.target_critic1.parameters()).detach())
    assert float(agent.alpha.detach()) != pytest.approx(alpha_before)
    action, state = agent.predict(np.zeros(4, dtype=np.float32))
    assert state is None
    assert int(action[0]) in range(4)

    path = tmp_path / "agent.pt"
    agent.save(path)
    restored = DiscreteSACAgent.load(path, TinyDiscreteEnv(), extractor_factory)
    assert restored.num_timesteps == 20
    assert restored.gradient_updates == agent.gradient_updates
    assert restored.predict(np.zeros(4, dtype=np.float32))[0].shape == (1,)


def test_discrete_sac_rejects_continuous_action_space() -> None:
    env = TinyDiscreteEnv()
    env.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))
    with pytest.raises(TypeError, match="Discrete"):
        DiscreteSACAgent(env, extractor_factory, 8, 1, tiny_config())
