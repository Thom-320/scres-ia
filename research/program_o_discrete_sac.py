"""Small categorical SAC implementation for Program O-R's Discrete(4) action.

This is a development workbench implementation, not a frozen paper learner.
Unlike Stable-Baselines3 SAC, it optimizes a categorical policy directly and
does not quantize a continuous action into the four production decisions.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except ImportError:  # pragma: no cover - keeps the categorical SAC standalone
    BaseFeaturesExtractor = nn.Module  # type: ignore[misc,assignment]


class HistoryStackWrapper(gym.Wrapper):
    """Flatten a causal history of observations without changing the action."""

    def __init__(self, env: gym.Env, history_len: int = 8):
        super().__init__(env)
        if history_len <= 0:
            raise ValueError("history_len must be positive")
        if len(env.observation_space.shape) != 1:
            raise ValueError("HistoryStackWrapper requires a vector observation")
        self.history_len = int(history_len)
        self.obs_dim = int(env.observation_space.shape[0])
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.history_len * self.obs_dim,),
            dtype=np.float32,
        )
        self._history: deque[np.ndarray] = deque(maxlen=self.history_len)

    def _stack(self) -> np.ndarray:
        return np.concatenate(tuple(self._history)).astype(np.float32, copy=False)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._history.clear()
        for _ in range(self.history_len - 1):
            self._history.append(np.zeros(self.obs_dim, dtype=np.float32))
        self._history.append(np.asarray(observation, dtype=np.float32))
        return self._stack(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._history.append(np.asarray(observation, dtype=np.float32))
        return self._stack(), reward, terminated, truncated, info


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, features_dim: int = 128):
        super().__init__()
        self.output_dim = int(features_dim)
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), 128),
            nn.GELU(),
            nn.Linear(128, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


class DavidDMPLAEncoder(nn.Module):
    """Editable temporal encoder adapted from David's earlier DMLPA notebook."""

    def __init__(
        self,
        input_dim: int,
        *,
        obs_dim: int = 21,
        history_len: int = 8,
        features_dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        positional_mode: str = "learned",
    ):
        super().__init__()
        if int(input_dim) != int(obs_dim) * int(history_len):
            raise ValueError("input_dim must equal obs_dim * history_len")
        if features_dim % heads:
            raise ValueError("features_dim must be divisible by heads")
        self.obs_dim = int(obs_dim)
        self.history_len = int(history_len)
        self.output_dim = int(features_dim)
        self.positional_mode = str(positional_mode)
        if self.positional_mode not in {"learned", "sinusoidal", "none"}:
            raise ValueError("positional_mode must be learned, sinusoidal, or none")
        self.embed = nn.Sequential(
            nn.Linear(self.obs_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.output_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.output_dim,
            nhead=int(heads),
            dim_feedforward=4 * self.output_dim,
            batch_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=int(layers))
        self.norm = nn.LayerNorm(self.output_dim)
        if self.positional_mode == "learned":
            self.position = nn.Parameter(
                torch.zeros(1, self.history_len, self.output_dim)
            )
            nn.init.normal_(self.position, std=0.02)
        elif self.positional_mode == "sinusoidal":
            position = torch.arange(self.history_len, dtype=torch.float32)[:, None]
            divisor = torch.exp(
                torch.arange(0, self.output_dim, 2, dtype=torch.float32)
                * (-np.log(10_000.0) / self.output_dim)
            )
            encoding = torch.zeros(self.history_len, self.output_dim)
            encoding[:, 0::2] = torch.sin(position * divisor)
            encoding[:, 1::2] = torch.cos(position * divisor[: encoding[:, 1::2].shape[1]])
            self.register_buffer("position", encoding[None, :, :])
        else:
            self.register_buffer(
                "position", torch.zeros(1, self.history_len, self.output_dim)
            )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        sequence = observation.reshape(-1, self.history_len, self.obs_dim)
        encoded = self.embed(sequence) + self.position
        encoded = self.temporal(encoded)
        return self.norm(encoded[:, -1])


class SB3DMPLAFeaturesExtractor(BaseFeaturesExtractor):
    """Stable-Baselines3 adapter for DavidDMPLAEncoder.

    The keyword arguments are deliberately exposed so David can change history,
    positional encoding, attention width, or replace this class in the notebook.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        *,
        obs_dim: int = 21,
        history_len: int = 8,
        features_dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        positional_mode: str = "learned",
    ):
        super().__init__(observation_space, features_dim=int(features_dim))
        input_dim = int(np.prod(observation_space.shape))
        self.encoder = DavidDMPLAEncoder(
            input_dim,
            obs_dim=obs_dim,
            history_len=history_len,
            features_dim=features_dim,
            heads=heads,
            layers=layers,
            positional_mode=positional_mode,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encoder(observations)


class _Actor(nn.Module):
    def __init__(self, encoder: nn.Module, action_dim: int):
        super().__init__()
        self.encoder = encoder
        self.logits = nn.Linear(int(encoder.output_dim), int(action_dim))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.logits(self.encoder(observation))


class _Critic(nn.Module):
    def __init__(self, encoder: nn.Module, action_dim: int):
        super().__init__()
        self.encoder = encoder
        self.values = nn.Linear(int(encoder.output_dim), int(action_dim))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.values(self.encoder(observation))


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = int(capacity)
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.cursor = 0

    def add(self, observation, action, reward, next_observation, done) -> None:
        index = self.cursor
        self.observations[index] = observation
        self.next_observations[index] = next_observation
        self.actions[index] = int(action)
        self.rewards[index] = float(reward)
        self.dones[index] = float(done)
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        indices = np.random.randint(0, self.size, size=int(batch_size))
        tensor = lambda value, dtype=None: torch.as_tensor(
            value[indices], dtype=dtype, device=device
        )
        return (
            tensor(self.observations),
            tensor(self.actions, torch.long),
            tensor(self.rewards),
            tensor(self.next_observations),
            tensor(self.dones),
        )


@dataclass(frozen=True)
class SACTrainingSummary:
    total_steps: int
    episodes: int
    mean_last_100_return: float
    updates: int
    alpha: float


class CategoricalSAC:
    """Twin-Q categorical Soft Actor-Critic for small discrete actions."""

    def __init__(
        self,
        *,
        observation_dim: int,
        action_dim: int,
        encoder_factory: Callable[[], nn.Module],
        device: str = "auto",
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_entropy_ratio: float = 0.98,
        seed: int = 0,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.actor = _Actor(encoder_factory(), action_dim).to(self.device)
        self.q1 = _Critic(encoder_factory(), action_dim).to(self.device)
        self.q2 = _Critic(encoder_factory(), action_dim).to(self.device)
        self.target_q1 = deepcopy(self.q1).eval()
        self.target_q2 = deepcopy(self.q2).eval()
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=learning_rate
        )
        self.q_optimizer = torch.optim.Adam(
            (*self.q1.parameters(), *self.q2.parameters()), lr=learning_rate
        )
        self.log_alpha = torch.tensor(
            0.0, requires_grad=True, device=self.device
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
        self.target_entropy = float(target_entropy_ratio) * float(np.log(action_dim))

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        with torch.no_grad():
            tensor = torch.as_tensor(
                observation, dtype=torch.float32, device=self.device
            ).reshape(1, -1)
            logits = self.actor(tensor)
            if deterministic:
                return int(logits.argmax(dim=1).item())
            return int(torch.distributions.Categorical(logits=logits).sample().item())

    def _update(self, replay: ReplayBuffer, batch_size: int) -> None:
        observation, action, reward, next_observation, done = replay.sample(
            batch_size, self.device
        )
        with torch.no_grad():
            next_logits = self.actor(next_observation)
            next_log_probability = F.log_softmax(next_logits, dim=1)
            next_probability = next_log_probability.exp()
            next_q = torch.minimum(
                self.target_q1(next_observation), self.target_q2(next_observation)
            )
            next_value = (
                next_probability * (next_q - self.alpha.detach() * next_log_probability)
            ).sum(dim=1)
            target = reward + self.gamma * (1.0 - done) * next_value

        q1 = self.q1(observation).gather(1, action[:, None]).squeeze(1)
        q2 = self.q2(observation).gather(1, action[:, None]).squeeze(1)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        logits = self.actor(observation)
        log_probability = F.log_softmax(logits, dim=1)
        probability = log_probability.exp()
        with torch.no_grad():
            minimum_q = torch.minimum(self.q1(observation), self.q2(observation))
        actor_loss = (
            probability * (self.alpha.detach() * log_probability - minimum_q)
        ).sum(dim=1).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        entropy = -(probability * log_probability).sum(dim=1)
        alpha_loss = (
            self.log_alpha * (entropy.detach() - self.target_entropy)
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_q1.parameters(), self.q1.parameters(), strict=True
            ):
                target_parameter.mul_(1.0 - self.tau).add_(parameter, alpha=self.tau)
            for target_parameter, parameter in zip(
                self.target_q2.parameters(), self.q2.parameters(), strict=True
            ):
                target_parameter.mul_(1.0 - self.tau).add_(parameter, alpha=self.tau)

    def learn(
        self,
        env: gym.Env,
        *,
        total_steps: int,
        learning_starts: int = 1_000,
        batch_size: int = 128,
        replay_capacity: int = 100_000,
        gradient_steps: int = 1,
    ) -> SACTrainingSummary:
        replay = ReplayBuffer(replay_capacity, self.observation_dim)
        observation, _ = env.reset()
        episode_return = 0.0
        returns: list[float] = []
        updates = 0
        for step in range(int(total_steps)):
            if step < int(learning_starts):
                action = int(env.action_space.sample())
            else:
                action = self.predict(observation, deterministic=False)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            replay.add(observation, action, reward, next_observation, done)
            episode_return += float(reward)
            observation = next_observation
            if done:
                returns.append(episode_return)
                episode_return = 0.0
                observation, _ = env.reset()
            if replay.size >= max(int(learning_starts), int(batch_size)):
                for _ in range(int(gradient_steps)):
                    self._update(replay, batch_size)
                    updates += 1
        tail = returns[-100:] if returns else [float("nan")]
        return SACTrainingSummary(
            total_steps=int(total_steps),
            episodes=len(returns),
            mean_last_100_return=float(np.mean(tail)),
            updates=updates,
            alpha=float(self.alpha.detach().cpu().item()),
        )

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "target_q1": self.target_q1.state_dict(),
                "target_q2": self.target_q2.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "observation_dim": self.observation_dim,
                "action_dim": self.action_dim,
            },
            path,
        )


class HistoryPolicyAdapter:
    """Apply a history-trained model to the raw 21D evaluation environment."""

    def __init__(self, model, *, label: str, history_len: int = 8):
        self.model = model
        self.label = str(label)
        self.history_len = int(history_len)
        self._history: deque[np.ndarray] = deque(maxlen=self.history_len)

    def reset_policy_state(self) -> None:
        self._history.clear()

    def predict_action(self, observation: np.ndarray) -> int:
        value = np.asarray(observation, dtype=np.float32)
        if not self._history:
            for _ in range(self.history_len - 1):
                self._history.append(np.zeros_like(value))
        self._history.append(value)
        stacked = np.concatenate(tuple(self._history)).astype(np.float32)
        predicted = self.model.predict(stacked, deterministic=True)
        action = predicted[0] if isinstance(predicted, tuple) else predicted
        return int(np.asarray(action).item())
