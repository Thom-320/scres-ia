"""Discrete Soft Actor-Critic for David's Program O architecture sandbox.

This is deliberately small and auditable.  It implements the categorical SAC
objective exactly over the four Program O actions and exposes the SB3-like
``learn``/``predict``/``save`` interface used by the notebook.  It is a
development agent; scientific promotion is controlled by a separate contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ExtractorFactory = Callable[[], nn.Module]


@dataclass(frozen=True)
class DiscreteSACConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    buffer_size: int = 100_000
    batch_size: int = 256
    learning_starts: int = 2_000
    train_freq: int = 1
    gradient_steps: int = 1
    hidden_dims: tuple[int, ...] = (128, 64)
    target_entropy_fraction: float = 0.98


class ReplayBuffer:
    def __init__(self, capacity: int, observation_shape: tuple[int, ...], seed: int):
        self.capacity = int(capacity)
        self.observations = np.empty((capacity, *observation_shape), dtype=np.float32)
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        i = self.position
        self.observations[i] = observation
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.next_observations[i] = next_observation
        self.dones[i] = float(done)
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        if self.size < batch_size:
            raise RuntimeError("Replay buffer does not contain a full batch")
        indices = self.rng.integers(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.observations[indices], device=device),
            torch.as_tensor(self.actions[indices], device=device),
            torch.as_tensor(self.rewards[indices], device=device),
            torch.as_tensor(self.next_observations[indices], device=device),
            torch.as_tensor(self.dones[indices], device=device),
        )


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(current, width), nn.ReLU()))
        current = width
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class CategoricalActor(nn.Module):
    def __init__(self, extractor: nn.Module, features_dim: int, action_count: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        self.features_extractor = extractor
        self.head = _mlp(features_dim, hidden_dims, action_count)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.head(self.features_extractor(observations))

    def probabilities(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self(observations)
        log_probabilities = F.log_softmax(logits, dim=-1)
        return log_probabilities.exp(), log_probabilities


class DiscreteCritic(nn.Module):
    def __init__(self, extractor: nn.Module, features_dim: int, action_count: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        self.features_extractor = extractor
        self.head = _mlp(features_dim, hidden_dims, action_count)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.head(self.features_extractor(observations))


class DiscreteSACAgent(nn.Module):
    """Categorical SAC with twin critics and an automatically tuned entropy coefficient."""

    def __init__(
        self,
        env: gym.Env,
        extractor_factory: ExtractorFactory,
        features_dim: int,
        seed: int,
        config: DiscreteSACConfig | None = None,
        device: str | torch.device = "auto",
    ) -> None:
        super().__init__()
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("DiscreteSACAgent requires gym.spaces.Discrete")
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("DiscreteSACAgent requires a Box observation space")
        self.env = env
        self.config = config or DiscreteSACConfig()
        self.seed = int(seed)
        self.action_count = int(env.action_space.n)
        self.features_dim = int(features_dim)
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "cpu" if device == "auto" else device
        )
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.actor = CategoricalActor(
            extractor_factory(), features_dim, self.action_count, self.config.hidden_dims
        )
        self.critic1 = DiscreteCritic(
            extractor_factory(), features_dim, self.action_count, self.config.hidden_dims
        )
        self.critic2 = DiscreteCritic(
            extractor_factory(), features_dim, self.action_count, self.config.hidden_dims
        )
        self.target_critic1 = DiscreteCritic(
            extractor_factory(), features_dim, self.action_count, self.config.hidden_dims
        )
        self.target_critic2 = DiscreteCritic(
            extractor_factory(), features_dim, self.action_count, self.config.hidden_dims
        )
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        for target in (self.target_critic1, self.target_critic2):
            target.requires_grad_(False)

        self.log_alpha = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)
        critic_parameters = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self.critic_optimizer = torch.optim.Adam(critic_parameters, lr=self.config.learning_rate)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.learning_rate)
        self.target_entropy = self.config.target_entropy_fraction * float(np.log(self.action_count))
        self.replay_buffer = ReplayBuffer(
            self.config.buffer_size, tuple(env.observation_space.shape), self.seed + 1
        )
        self.rng = np.random.default_rng(self.seed + 2)
        self.num_timesteps = 0
        self.gradient_updates = 0

    @property
    def features_extractor(self) -> nn.Module:
        return self.actor.features_extractor

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def get_env(self) -> gym.Env:
        return self.env

    def _tensor(self, observation: np.ndarray) -> torch.Tensor:
        array = np.asarray(observation, dtype=np.float32)
        if array.ndim == len(self.env.observation_space.shape):
            array = array[None, ...]
        return torch.as_tensor(array, device=self.device)

    def predict(
        self,
        observation: np.ndarray,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        del state, episode_start
        with torch.no_grad():
            probabilities, _ = self.actor.probabilities(self._tensor(observation))
        if deterministic:
            actions = probabilities.argmax(dim=-1)
        else:
            actions = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
        return actions.cpu().numpy(), None

    def _update(self) -> dict[str, float]:
        observations, actions, rewards, next_observations, dones = self.replay_buffer.sample(
            self.config.batch_size, self.device
        )
        with torch.no_grad():
            next_probabilities, next_log_probabilities = self.actor.probabilities(next_observations)
            target_q = torch.minimum(
                self.target_critic1(next_observations), self.target_critic2(next_observations)
            )
            next_value = (
                next_probabilities * (target_q - self.alpha.detach() * next_log_probabilities)
            ).sum(dim=-1)
            target = rewards + self.config.gamma * (1.0 - dones) * next_value

        q1 = self.critic1(observations).gather(1, actions[:, None]).squeeze(1)
        q2 = self.critic2(observations).gather(1, actions[:, None]).squeeze(1)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        probabilities, log_probabilities = self.actor.probabilities(observations)
        with torch.no_grad():
            minimum_q = torch.minimum(self.critic1(observations), self.critic2(observations))
        actor_loss = (
            probabilities * (self.alpha.detach() * log_probabilities - minimum_q)
        ).sum(dim=-1).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        entropy = -(probabilities.detach() * log_probabilities.detach()).sum(dim=-1).mean()
        alpha_loss = self.log_alpha * (entropy - self.target_entropy)
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for source, target_network in (
                (self.critic1, self.target_critic1), (self.critic2, self.target_critic2)
            ):
                for source_parameter, target_parameter in zip(
                    source.parameters(), target_network.parameters(), strict=True
                ):
                    target_parameter.mul_(1.0 - self.config.tau)
                    target_parameter.add_(source_parameter, alpha=self.config.tau)
        self.gradient_updates += 1
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha_loss": float(alpha_loss.detach()),
            "alpha": float(self.alpha.detach()),
            "entropy": float(entropy.detach()),
        }

    def learn(self, total_timesteps: int, progress_bar: bool = False) -> "DiscreteSACAgent":
        del progress_bar
        observation, _ = self.env.reset(seed=self.seed)
        while self.num_timesteps < int(total_timesteps):
            if self.num_timesteps < self.config.learning_starts:
                action = int(self.env.action_space.sample())
            else:
                action = int(self.predict(observation, deterministic=False)[0][0])
            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            done = bool(terminated or truncated)
            self.replay_buffer.add(observation, action, reward, next_observation, done)
            observation = next_observation
            self.num_timesteps += 1
            if (
                self.num_timesteps >= self.config.learning_starts
                and self.num_timesteps % self.config.train_freq == 0
                and self.replay_buffer.size >= self.config.batch_size
            ):
                for _ in range(self.config.gradient_steps):
                    self._update()
            if done:
                observation, _ = self.env.reset()
        return self

    def save(self, path: str | Path) -> None:
        payload = {
            "state_dict": self.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "config": asdict(self.config),
            "seed": self.seed,
            "features_dim": self.features_dim,
            "action_count": self.action_count,
            "num_timesteps": self.num_timesteps,
            "gradient_updates": self.gradient_updates,
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        env: gym.Env,
        extractor_factory: ExtractorFactory,
        device: str | torch.device = "auto",
    ) -> "DiscreteSACAgent":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        config = DiscreteSACConfig(**payload["config"])
        agent = cls(
            env=env,
            extractor_factory=extractor_factory,
            features_dim=int(payload["features_dim"]),
            seed=int(payload["seed"]),
            config=config,
            device=device,
        )
        if int(payload["action_count"]) != agent.action_count:
            raise ValueError("Saved action count does not match the environment")
        agent.load_state_dict(payload["state_dict"])
        agent.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        agent.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        agent.num_timesteps = int(payload["num_timesteps"])
        agent.gradient_updates = int(payload["gradient_updates"])
        return agent
