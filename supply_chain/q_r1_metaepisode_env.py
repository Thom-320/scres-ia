"""Q-R1 metaepisode environment with physical resets and causal history retention.

One Gymnasium episode contains twelve eight-decision physical campaigns.  Each
campaign is reconstructed from its own immutable FullDES skeleton, so inventory,
backlog, work in process, and transport state never cross campaign boundaries.
The observation stream does cross those boundaries and includes an explicit
boundary marker.  A recurrent policy may therefore retain only deployable
knowledge inferred from past observations.

The environment does not expose the true regime, retained Bayesian prior,
history root, tape seed, or future demand.  The retained/reset treatment is
applied during recurrent-policy evaluation by either carrying or zeroing the
same model's hidden state at the boundary marker.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from supply_chain.program_o_full_des_transducer import simulate_full_des_frontier
from supply_chain.program_o_ret_env import (
    OBSERVATION_DIM,
    normalized_state_rich_observation,
)
from supply_chain.program_o_state_rich import (
    StateRichConfiguration,
    state_rich_calendar,
)
from supply_chain.retained_context_discovery import CampaignSpec


CAMPAIGNS_PER_METAEPISODE = 12
DECISIONS_PER_CAMPAIGN = 8
DECISIONS_PER_METAEPISODE = CAMPAIGNS_PER_METAEPISODE * DECISIONS_PER_CAMPAIGN
META_OBSERVATION_DIM = OBSERVATION_DIM + 1
FACTORIAL_OBSERVATION_DIM = OBSERVATION_DIM + 2
OBJECTIVE = "early_ret_complete_cohort"
_REPLAY_CONFIG = StateRichConfiguration("belief_mpc", 3)


class QRetainedMetaEpisodeEnv(gym.Env[np.ndarray, int]):
    """Twelve-campaign Q-R1 environment for a retained recurrent learner."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        histories: Sequence[Sequence[CampaignSpec]],
        scheduler: Mapping[str, Sequence[str]],
        regime_persistence: float = 0.90,
        dominant_share: float = 0.90,
        sampling_seed: int = 0,
        prior_paths: Sequence[Sequence[float]] | None = None,
        expose_prior_feature: bool = False,
    ) -> None:
        super().__init__()
        self.histories = tuple(tuple(history) for history in histories)
        if not self.histories:
            raise ValueError("at least one campaign history is required")
        for history in self.histories:
            if len(history) != CAMPAIGNS_PER_METAEPISODE:
                raise ValueError(
                    f"each history must contain {CAMPAIGNS_PER_METAEPISODE} campaigns"
                )
            roots = {int(campaign.history_root) for campaign in history}
            if len(roots) != 1:
                raise ValueError("all campaigns in a metaepisode must share one history root")
            if tuple(int(c.campaign_index) for c in history) != tuple(
                range(CAMPAIGNS_PER_METAEPISODE)
            ):
                raise ValueError("campaign indices must be the ordered range 0..11")
        self.scheduler = {str(key): tuple(value) for key, value in scheduler.items()}
        self.regime_persistence = float(regime_persistence)
        self.dominant_share = float(dominant_share)
        self.expose_prior_feature = bool(expose_prior_feature)
        if prior_paths is None:
            self.prior_paths = tuple(
                tuple(0.5 for _ in history) for history in self.histories
            )
        else:
            self.prior_paths = tuple(
                tuple(float(value) for value in path) for path in prior_paths
            )
            if len(self.prior_paths) != len(self.histories):
                raise ValueError("prior_paths must match the number of histories")
            for history, path in zip(self.histories, self.prior_paths, strict=True):
                if len(path) != len(history):
                    raise ValueError("each prior path must match its campaign history")
                if any(not 0.0 <= value <= 1.0 for value in path):
                    raise ValueError("every explicit prior must be in [0, 1]")
        self._rng = np.random.default_rng(int(sampling_seed))
        self.action_space = spaces.Discrete(4)
        observation_dim = (
            FACTORIAL_OBSERVATION_DIM
            if self.expose_prior_feature
            else META_OBSERVATION_DIM
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(observation_dim,),
            dtype=np.float32,
        )
        self._history: tuple[CampaignSpec, ...] | None = None
        self._history_index: int | None = None
        self._campaign_position = 0
        self._actions: list[int] = []
        self._campaign_rows: list[dict[str, Any]] = []

    @property
    def current_campaign(self) -> CampaignSpec:
        if self._history is None:
            raise RuntimeError("reset() must be called before accessing the campaign")
        return self._history[self._campaign_position]

    @property
    def current_prior(self) -> float:
        if self._history_index is None:
            raise RuntimeError("reset() must be called before accessing the prior")
        return float(self.prior_paths[self._history_index][self._campaign_position])

    def _observation(self, *, boundary: bool) -> np.ndarray:
        campaign = self.current_campaign
        weeks = int(campaign.skeleton.decision_weeks)
        padded = tuple(self._actions) + (0,) * (weeks - len(self._actions))
        _calendar, decisions = state_rich_calendar(
            skeleton=campaign.skeleton.as_dict(),
            scheduler=self.scheduler,
            config=_REPLAY_CONFIG,
            regime_persistence=self.regime_persistence,
            dominant_share=self.dominant_share,
            action_overrides=padded,
            initial_belief_c=0.5,
        )
        decision_index = len(self._actions)
        if decision_index >= len(decisions):
            raise RuntimeError("no observation exists after a terminal campaign action")
        base = normalized_state_rich_observation(decisions[decision_index].observation)
        suffix = [float(boundary)]
        if self.expose_prior_feature:
            suffix.append(self.current_prior)
        return np.concatenate([base, np.asarray(suffix, dtype=np.float32)]).astype(
            np.float32, copy=False
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = dict(options or {})
        if "history_index" in options:
            history_index = int(options["history_index"])
        else:
            history_index = int(self._rng.integers(len(self.histories)))
        if history_index not in range(len(self.histories)):
            raise ValueError("history_index is outside the supplied history set")
        self._history_index = history_index
        self._history = self.histories[history_index]
        self._campaign_position = 0
        self._actions = []
        self._campaign_rows = []
        return self._observation(boundary=True), {
            "campaign_boundary": True,
            "campaign_position": 0,
            "physical_reset": True,
            "explicit_prior": self.current_prior,
        }

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_value = int(action)
        if not self.action_space.contains(action_value):
            raise ValueError("Q-R1 action must be in {0,1,2,3}")
        campaign = self.current_campaign
        self._actions.append(action_value)
        campaign_done = len(self._actions) == int(campaign.skeleton.decision_weeks)
        if not campaign_done:
            return self._observation(boundary=False), 0.0, False, False, {
                "campaign_boundary": False,
                "campaign_position": self._campaign_position,
                "physical_reset": False,
            }

        panel = simulate_full_des_frontier(
            skeleton=campaign.skeleton,
            scheduler=self.scheduler,
            calendars=np.asarray([self._actions], dtype=np.uint8),
            include_q_r1_metrics=True,
        )
        metrics = {key: float(np.asarray(value)[0]) for key, value in panel.items()}
        row = {
            "campaign_position": self._campaign_position,
            "campaign_index": int(campaign.campaign_index),
            "calendar": tuple(self._actions),
            "objective": metrics[OBJECTIVE],
            "early_ret_complete_cohort": metrics[OBJECTIVE],
            "early_ret_visible": metrics["early_ret_visible"],
            "ret_visible": metrics["ret_visible"],
            "ret_full": metrics["ret_full"],
            "whole_campaign_ret": metrics["ret_visible"],
            "worst_product_fill": metrics["worst_product_fill"],
            "unresolved_orders": metrics["unresolved_orders"],
            "unresolved_quantity": metrics["unresolved_quantity"],
            "lost_orders": metrics["lost_orders"],
            "lost_quantity": metrics["lost_quantity"],
            "service_loss": metrics["service_loss_auc"],
            "gross_policy_batch_slots": metrics["gross_policy_batch_slots"],
            "gross_production_quantity": metrics["gross_production_quantity"],
            "charged_daily_dispatch_slots": metrics["charged_daily_dispatch_slots"],
            "charged_downstream_vehicle_hours": metrics[
                "charged_downstream_vehicle_hours"
            ],
            "skeleton_sha256": campaign.skeleton.skeleton_sha256,
            "prefix_state_hash": campaign.skeleton.prefix_state_hash,
            "explicit_prior": self.current_prior,
        }
        self._campaign_rows.append(row)
        reward = float(metrics[OBJECTIVE])

        if self._campaign_position + 1 == len(self._history or ()):
            info = {
                **row,
                "campaign_complete": True,
                "campaign_boundary": False,
                "physical_reset": False,
                "metaepisode_complete": True,
                "campaign_rows": tuple(self._campaign_rows),
            }
            return (
                np.zeros(self.observation_space.shape, dtype=np.float32),
                reward,
                True,
                False,
                info,
            )

        previous_skeleton = campaign.skeleton.skeleton_sha256
        self._campaign_position += 1
        self._actions = []
        next_campaign = self.current_campaign
        info = {
            **row,
            "campaign_complete": True,
            "campaign_boundary": True,
            "physical_reset": True,
            "metaepisode_complete": False,
            "previous_skeleton_sha256": previous_skeleton,
            "next_skeleton_sha256": next_campaign.skeleton.skeleton_sha256,
        }
        return self._observation(boundary=True), reward, False, False, info
