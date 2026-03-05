from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from supply_chain.config import (
    DEFAULT_YEAR_BASIS,
    OPERATIONS,
    SIMULATION_HORIZON,
    WARMUP,
    YEAR_BASIS_OPTIONS,
)
from supply_chain.supply_chain import MFSCSimulation


class MFSCGymEnv(gym.Env[np.ndarray, np.ndarray]):
    """
    Gymnasium wrapper for the MFSC simulation.

    Observation: 15-dimensional continuous state vector.
    Action: 4-dimensional policy multipliers in [-1, 1].
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        step_size_hours: float = 168,
        max_steps: Optional[int] = None,
        year_basis: str = DEFAULT_YEAR_BASIS,
        risk_level: str = "current",
        reward_mode: str = "proxy",
        rt_alpha: float = 1.0,
        rt_beta: float = 1.0,
        rt_gamma: float = 1.0,
        rt_recovery_scale: float = 1.0,
        rt_inventory_scale: float = 1_000_000.0,
    ) -> None:
        super().__init__()
        if step_size_hours <= 0:
            raise ValueError("step_size_hours must be > 0")
        if year_basis not in YEAR_BASIS_OPTIONS:
            raise ValueError(
                f"Invalid year_basis={year_basis!r}. Expected one of {YEAR_BASIS_OPTIONS}."
            )
        if reward_mode not in ("proxy", "rt_v0"):
            raise ValueError(
                f"Invalid reward_mode={reward_mode!r}. Expected 'proxy' or 'rt_v0'."
            )
        if risk_level not in ("current", "increased"):
            raise ValueError(
                f"Invalid risk_level={risk_level!r}. Expected 'current' or 'increased'."
            )

        self.step_size = float(step_size_hours)
        self.year_basis = year_basis
        self.risk_level = risk_level
        self.reward_mode = reward_mode
        self.rt_alpha = float(rt_alpha)
        self.rt_beta = float(rt_beta)
        self.rt_gamma = float(rt_gamma)
        self.rt_recovery_scale = float(rt_recovery_scale)
        self.rt_inventory_scale = float(rt_inventory_scale)
        self.warmup_hours = float(WARMUP["estimated_deterministic_hrs"])
        if max_steps is None:
            self.max_steps = int(
                (SIMULATION_HORIZON - self.warmup_hours) / self.step_size
            )
        else:
            self.max_steps = max_steps

        self.current_step = 0
        self.sim: Optional[MFSCSimulation] = None

        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(15,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def _compute_rt_v0_components(self, info: dict[str, Any]) -> dict[str, float]:
        """Expose normalized reward terms for debugging and evaluation."""
        recovery_time = float(info.get("step_disruption_hours", 0.0))
        holding_cost_raw = float(info.get("total_inventory", 0.0))
        new_demanded = float(info.get("new_demanded", 0.0))
        new_backorder_qty = float(info.get("new_backorder_qty", 0.0))
        service_loss = new_backorder_qty / new_demanded if new_demanded > 0 else 0.0

        norm_recovery = recovery_time / max(1.0, self.rt_recovery_scale)
        norm_inventory = holding_cost_raw / max(1.0, self.rt_inventory_scale)
        weighted_recovery = self.rt_alpha * norm_recovery
        weighted_inventory = self.rt_beta * norm_inventory
        weighted_service = self.rt_gamma * service_loss

        return {
            "recovery_time_raw": recovery_time,
            "holding_cost_raw": holding_cost_raw,
            "service_loss_raw": service_loss,
            "recovery_time_norm": norm_recovery,
            "holding_cost_norm": norm_inventory,
            "weighted_recovery": weighted_recovery,
            "weighted_holding": weighted_inventory,
            "weighted_service_loss": weighted_service,
            "reward_total": -(
                weighted_recovery + weighted_inventory + weighted_service
            ),
        }

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        self.current_step = 0
        self.sim = MFSCSimulation(
            shifts=1,
            risks_enabled=True,
            risk_level=self.risk_level,
            seed=seed,
            horizon=SIMULATION_HORIZON,
            year_basis=self.year_basis,
        )
        self.sim._start_processes()
        self.sim.env.run(until=self.warmup_hours)
        obs = np.array(self.sim.get_observation(), dtype=np.float32)
        info = {"time": self.sim.env.now, "year_basis": self.year_basis}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.sim is None:
            raise RuntimeError("Environment not reset. Call reset() before step().")

        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.shape != (4,):
            raise ValueError(f"Action must have shape (4,), got {action_arr.shape}.")

        clipped_action = np.clip(
            action_arr, self.action_space.low, self.action_space.high
        )
        self.current_step += 1

        multipliers = 1.25 + 0.75 * clipped_action
        base_op9_min = OPERATIONS[9]["q"][0]
        base_op9_max = OPERATIONS[9]["q"][1]

        action_dict = {
            "op3_q": OPERATIONS[3]["q"] * float(multipliers[0]),
            "op9_q_min": base_op9_min * float(multipliers[1]),
            "op9_q_max": base_op9_max * float(multipliers[1]),
            "op3_rop": OPERATIONS[3]["rop"] * float(multipliers[2]),
            "op9_rop": OPERATIONS[9]["rop"] * float(multipliers[3]),
        }
        obs, reward, terminated, info = self.sim.step(
            action=action_dict,
            step_hours=self.step_size,
        )

        rt_components: Optional[dict[str, float]] = None
        if self.reward_mode == "rt_v0":
            rt_components = self._compute_rt_v0_components(info)
            reward = rt_components["reward_total"]

        truncated = self.current_step >= self.max_steps
        out_obs = np.array(obs, dtype=np.float32)
        out_info = {
            **info,
            "raw_action": action_arr.tolist(),
            "clipped_action": clipped_action.tolist(),
            "reward_mode": self.reward_mode,
        }
        if rt_components is not None:
            out_info["rt_components"] = rt_components
        return out_obs, float(reward), bool(terminated), bool(truncated), out_info
