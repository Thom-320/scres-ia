"""
Gymnasium env with 5th action dimension (shift control) and multiple rewards.

Supports three reward modes:
  - "rt_v0": Legacy R_t v0 (aggregated recovery/holding/service loss).
  - "ReT_thesis": Approximation of Garrido (2017) Eq. 5.5 at step level,
    with linear shift cost δ×(S−1).
  - "control_v1": Operational control reward for RL benchmarking, while
    still exposing corrected ReT_thesis as a reporting-only metric.

Action space: 5-dimensional [-1, 1]
  [0-3]: Inventory policy multipliers (op3_q, op9_q, op3_rop, op9_rop)
  [4]:   Shift selector — mapped to {1, 2, 3} shifts
         action[4] < -0.33 → S=1 (single shift, 8h/day)
         -0.33 ≤ action[4] < 0.33 → S=2 (double shift, 16h/day)
         action[4] ≥ 0.33 → S=3 (triple shift, 24h/day)

ReT Approximation (Eq. 5.5 mapped to step-level metrics):
  Re_max = 1, Re = 1, Re_min = 0 (confirmed with Garrido).

  Case 1 (No disruption):       Re(FR_t) = fill_rate
  Case 2 (Autotomy):            Re(AP)   = 1 - disruption_frac
  Case 3 (Recovery):            Re(RP)   = 1 / (1 + disruption_frac)
  Case 4 (Non-recovery):        Re(DP)   = 0

  Reward = ReT_step - δ × (S - 1)

  Assumptions (publishable):
    A1: Step-level aggregation approximates order-level ReT when
        step size coincides with the reorder cycle (168h).
    A2: AP proxied by fraction of step without disruption.
    A3: RP proxied inversely by disruption fraction.
    A4: Disruption fraction normalized by op-hours (13 ops × step_hours).
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from supply_chain.config import (
    DEFAULT_YEAR_BASIS,
    OPERATIONS,
    RET_CASE_THRESHOLDS,
    RET_SHIFT_COST_DELTA_DEFAULT,
    SIMULATION_HORIZON,
    WARMUP,
    YEAR_BASIS_OPTIONS,
)
from supply_chain.supply_chain import MFSCSimulation

NUM_TRACKED_OPS = 13


class MFSCGymEnvShifts(gym.Env[np.ndarray, np.ndarray]):
    """
    Gymnasium wrapper with dynamic shift control and ReT-based reward.

    Observation: 15-dimensional continuous state vector (same as base env).
    Action: 5-dimensional [-1, 1].

    Parameters
    ----------
    reward_mode : {"ReT_thesis", "rt_v0", "control_v1"}
        Which reward formulation to use.
    rt_delta : float
        Linear shift cost weight. Reward -= δ × (S − 1).
        Must be calibrated via DOE for desired shift selection ratio.
    autotomy_threshold : float
        Fill-rate threshold above which a disrupted step counts as
        autotomy rather than recovery. Default 0.95.
    nonrecovery_disruption_threshold : float
        Disruption fraction above which (combined with low FR) the step
        is classified as non-recovery. Default 0.5.
    nonrecovery_fr_threshold : float
        Fill-rate threshold below which (combined with high disruption)
        the step is classified as non-recovery. Default 0.5.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        step_size_hours: float = 168,
        max_steps: Optional[int] = None,
        year_basis: str = DEFAULT_YEAR_BASIS,
        risk_level: str = "current",
        stochastic_pt: bool = False,
        reward_mode: str = "ReT_thesis",
        # --- ReT_thesis parameters ---
        rt_delta: float = RET_SHIFT_COST_DELTA_DEFAULT,
        autotomy_threshold: float = RET_CASE_THRESHOLDS["autotomy_fill_rate_threshold"],
        nonrecovery_disruption_threshold: float = RET_CASE_THRESHOLDS[
            "nonrecovery_disruption_fraction_threshold"
        ],
        nonrecovery_fr_threshold: float = RET_CASE_THRESHOLDS[
            "nonrecovery_fill_rate_threshold"
        ],
        # --- rt_v0 legacy parameters ---
        rt_alpha: float = 8.0,
        rt_beta: float = 1.0,
        rt_gamma: float = 7.0,
        rt_recovery_scale: float = 46.0,
        rt_inventory_scale: float = 17_200_000.0,
        # --- control_v1 weights ---
        w_bo: float = 1.0,
        w_cost: float = 0.06,
        w_disr: float = 0.0,
    ) -> None:
        super().__init__()
        if step_size_hours <= 0:
            raise ValueError("step_size_hours must be > 0")
        if year_basis not in YEAR_BASIS_OPTIONS:
            raise ValueError(f"Invalid year_basis={year_basis!r}.")
        if risk_level not in ("current", "increased", "severe"):
            raise ValueError(f"Invalid risk_level={risk_level!r}.")
        if reward_mode not in ("ReT_thesis", "rt_v0", "control_v1"):
            raise ValueError(
                f"Invalid reward_mode={reward_mode!r}. "
                "Expected 'ReT_thesis', 'rt_v0', or 'control_v1'."
            )

        self.step_size = float(step_size_hours)
        self.year_basis = year_basis
        self.risk_level = risk_level
        self.stochastic_pt = stochastic_pt
        self.reward_mode = reward_mode

        self.rt_delta = float(rt_delta)
        self.autotomy_threshold = float(autotomy_threshold)
        self.nonrecovery_disruption_threshold = float(nonrecovery_disruption_threshold)
        self.nonrecovery_fr_threshold = float(nonrecovery_fr_threshold)

        self.rt_alpha = float(rt_alpha)
        self.rt_beta = float(rt_beta)
        self.rt_gamma = float(rt_gamma)
        self.rt_recovery_scale = float(rt_recovery_scale)
        self.rt_inventory_scale = float(rt_inventory_scale)
        self.w_bo = float(w_bo)
        self.w_cost = float(w_cost)
        self.w_disr = float(w_disr)

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
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

    # -----------------------------------------------------------------
    # ReT approximation (Garrido 2017, Eq. 5.5)
    # -----------------------------------------------------------------

    def _compute_ret_thesis_components(
        self, info: dict[str, Any], step_hours: float
    ) -> dict[str, float | str]:
        """
        Approximate thesis ReT components at step level.

        Returns the case classification plus the resulting step-level ReT value.

        Important: the thresholds used here are configurable repo assumptions for
        step-level classification. They are not thesis-derived cutoffs.
        """
        demanded = float(info.get("new_demanded", 0.0))
        backorder_qty = float(info.get("new_backorder_qty", 0.0))
        disruption_hrs = float(info.get("step_disruption_hours", 0.0))

        if demanded <= 0:
            return {
                "ret_case": "no_demand",
                "fill_rate": 1.0,
                "disruption_hours": disruption_hrs,
                "disruption_fraction": 0.0,
                "ret_value": 1.0,
            }

        fill_rate = max(0.0, 1.0 - backorder_qty / demanded)

        # Normalize by total possible op-hours to avoid saturation when
        # multiple operations are down simultaneously.
        max_op_hours = step_hours * NUM_TRACKED_OPS
        disruption_frac = min(1.0, disruption_hrs / max_op_hours)

        # Case 4: Non-recovery (Re_min = 0)
        if (
            disruption_frac > self.nonrecovery_disruption_threshold
            and fill_rate < self.nonrecovery_fr_threshold
        ):
            return {
                "ret_case": "non_recovery",
                "fill_rate": fill_rate,
                "disruption_hours": disruption_hrs,
                "disruption_fraction": disruption_frac,
                "ret_value": 0.0,
            }

        # Case 3: Recovery — Re(RP) ∈ (0.5, 1.0]
        if disruption_hrs > 0 and fill_rate < self.autotomy_threshold:
            return {
                "ret_case": "recovery",
                "fill_rate": fill_rate,
                "disruption_hours": disruption_hrs,
                "disruption_fraction": disruption_frac,
                "ret_value": 1.0 / (1.0 + disruption_frac),
            }

        # Case 2: Autotomy — Re(AP) ∈ [0, 1]
        if disruption_hrs > 0 and fill_rate >= self.autotomy_threshold:
            return {
                "ret_case": "autotomy",
                "fill_rate": fill_rate,
                "disruption_hours": disruption_hrs,
                "disruption_fraction": disruption_frac,
                "ret_value": 1.0 - disruption_frac,
            }

        # Case 1: No disruption — Re(FR_t) = fill_rate ∈ [0, 1]
        return {
            "ret_case": "fill_rate_only",
            "fill_rate": fill_rate,
            "disruption_hours": disruption_hrs,
            "disruption_fraction": disruption_frac,
            "ret_value": fill_rate,
        }

    def _compute_ret_thesis_corrected_components(
        self, info: dict[str, Any], step_hours: float
    ) -> dict[str, float | str]:
        """
        Reporting-only corrected ReT_thesis.

        Keeps the same case split as the benchmarked thesis approximation, but
        scores autotomy with the recovery formula to remove the local
        non-monotonicity identified in diagnostics.
        """
        components = self._compute_ret_thesis_components(info, step_hours)
        if components["ret_case"] == "autotomy":
            disruption_fraction = float(components["disruption_fraction"])
            components["ret_value"] = 1.0 / (1.0 + disruption_fraction)
        return components

    # -----------------------------------------------------------------
    # Legacy R_t v0
    # -----------------------------------------------------------------

    def _compute_rt_v0(self, info: dict, shifts: int) -> float:
        """Legacy reward: -(α·recovery + β·holding + γ·service + δ·shift)."""
        recovery_time = float(info.get("step_disruption_hours", 0.0))
        holding_cost_raw = float(info.get("total_inventory", 0.0))
        new_demanded = float(info.get("new_demanded", 0.0))
        new_backorder_qty = float(info.get("new_backorder_qty", 0.0))
        service_loss = new_backorder_qty / new_demanded if new_demanded > 0 else 0.0
        norm_recovery = recovery_time / max(1.0, self.rt_recovery_scale)
        norm_inventory = holding_cost_raw / max(1.0, self.rt_inventory_scale)
        shift_cost = float(shifts - 1)

        return -(
            self.rt_alpha * norm_recovery
            + self.rt_beta * norm_inventory
            + self.rt_gamma * service_loss
            + self.rt_delta * shift_cost
        )

    def _compute_control_v1_components(
        self, info: dict[str, Any], shifts: int
    ) -> dict[str, float]:
        """Operational control reward terms used for RL benchmarking."""
        new_demanded = float(info.get("new_demanded", 0.0))
        new_backorder_qty = float(info.get("new_backorder_qty", 0.0))
        disruption_hours = float(info.get("step_disruption_hours", 0.0))
        service_loss_step = new_backorder_qty / max(new_demanded, 1.0)
        shift_cost_step = float(shifts - 1)
        max_op_hours = self.step_size * NUM_TRACKED_OPS
        disruption_fraction_step = min(1.0, disruption_hours / max(1.0, max_op_hours))
        weighted_service_loss = self.w_bo * service_loss_step
        weighted_shift_cost = self.w_cost * shift_cost_step
        weighted_disruption = self.w_disr * disruption_fraction_step
        reward_total = -(
            weighted_service_loss + weighted_shift_cost + weighted_disruption
        )
        return {
            "service_loss_step": service_loss_step,
            "shift_cost_step": shift_cost_step,
            "disruption_fraction_step": disruption_fraction_step,
            "weighted_service_loss": weighted_service_loss,
            "weighted_shift_cost": weighted_shift_cost,
            "weighted_disruption": weighted_disruption,
            "reward_total": reward_total,
        }

    # -----------------------------------------------------------------
    # Gymnasium API
    # -----------------------------------------------------------------

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
            stochastic_pt=self.stochastic_pt,
        )
        self.sim._start_processes()
        self.sim.env.run(until=self.warmup_hours)
        obs = np.array(self.sim.get_observation(), dtype=np.float32)
        info: dict[str, Any] = {
            "time": self.sim.env.now,
            "year_basis": self.year_basis,
            "action_constraints": {
                "action_bounds": [(-1.0, 1.0)] * 5,
                "inventory_multiplier_range": {
                    "min": 0.5,
                    "max": 2.0,
                    "mapping": "multiplier = 1.25 + 0.75 * signal",
                },
                "shift_signal_bands": {
                    "signal_lt_-0.33": 1,
                    "signal_ge_-0.33_and_lt_0.33": 2,
                    "signal_ge_0.33": 3,
                },
                "base_control_parameters": {
                    "op3_q": float(OPERATIONS[3]["q"]),
                    "op3_rop": float(OPERATIONS[3]["rop"]),
                    "op9_q_min": float(OPERATIONS[9]["q"][0]),
                    "op9_q_max": float(OPERATIONS[9]["q"][1]),
                    "op9_rop": float(OPERATIONS[9]["rop"]),
                },
            },
            "ret_thresholds": {
                "autotomy_fill_rate_threshold": self.autotomy_threshold,
                "nonrecovery_disruption_fraction_threshold": (
                    self.nonrecovery_disruption_threshold
                ),
                "nonrecovery_fill_rate_threshold": self.nonrecovery_fr_threshold,
            },
            "ret_thresholds_source": "configurable_repo_approximation",
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.sim is None:
            raise RuntimeError("Call reset() before step().")

        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.shape != (5,):
            raise ValueError(f"Action must have shape (5,), got {action_arr.shape}.")

        clipped = np.clip(action_arr, -1.0, 1.0)
        self.current_step += 1

        # Inventory multipliers (dims 0-3)
        multipliers = 1.25 + 0.75 * clipped[:4]
        base_op9_min = OPERATIONS[9]["q"][0]
        base_op9_max = OPERATIONS[9]["q"][1]

        # Shift decision (dim 4): tri-level with hysteresis-friendly bands
        shift_signal = float(clipped[4])
        if shift_signal < -0.33:
            shifts = 1
        elif shift_signal < 0.33:
            shifts = 2
        else:
            shifts = 3

        action_dict = {
            "op3_q": OPERATIONS[3]["q"] * float(multipliers[0]),
            "op9_q_min": base_op9_min * float(multipliers[1]),
            "op9_q_max": base_op9_max * float(multipliers[1]),
            "op3_rop": OPERATIONS[3]["rop"] * float(multipliers[2]),
            "op9_rop": OPERATIONS[9]["rop"] * float(multipliers[3]),
            "assembly_shifts": shifts,
        }
        obs, _, terminated, info = self.sim.step(
            action=action_dict,
            step_hours=self.step_size,
        )

        # Compute reward
        ret_components: dict[str, float | str] | None = None
        corrected_ret_components: dict[str, float | str] | None = None
        control_components: dict[str, float] | None = None
        if self.reward_mode == "ReT_thesis":
            ret_components = self._compute_ret_thesis_components(info, self.step_size)
            ReT = float(ret_components["ret_value"])
            shift_cost = self.rt_delta * (shifts - 1)
            reward = ReT - shift_cost
        elif self.reward_mode == "control_v1":
            control_components = self._compute_control_v1_components(info, shifts)
            corrected_ret_components = self._compute_ret_thesis_corrected_components(
                info, self.step_size
            )
            reward = float(control_components["reward_total"])
        else:
            reward = self._compute_rt_v0(info, shifts)

        truncated = self.current_step >= self.max_steps
        out_obs = np.array(obs, dtype=np.float32)
        out_info: dict[str, Any] = {
            **info,
            "raw_action": action_arr.tolist(),
            "clipped_action": clipped.tolist(),
            "reward_mode": self.reward_mode,
            "shifts_active": shifts,
            "shift_cost_linear": self.rt_delta * (shifts - 1),
            "shift_cost_delta": self.rt_delta,
        }
        if self.reward_mode == "ReT_thesis":
            out_info["ReT_raw"] = ReT
            out_info["ret_components"] = {
                **ret_components,
                "shift_cost": shift_cost,
                "reward_total": reward,
                "thresholds": {
                    "autotomy_fill_rate_threshold": self.autotomy_threshold,
                    "nonrecovery_disruption_fraction_threshold": (
                        self.nonrecovery_disruption_threshold
                    ),
                    "nonrecovery_fill_rate_threshold": self.nonrecovery_fr_threshold,
                },
                "thresholds_source": "configurable_repo_approximation",
            }
        elif self.reward_mode == "control_v1":
            out_info["service_loss_step"] = float(
                control_components["service_loss_step"]
            )
            out_info["shift_cost_step"] = float(control_components["shift_cost_step"])
            out_info["disruption_fraction_step"] = float(
                control_components["disruption_fraction_step"]
            )
            out_info["control_components"] = {
                **control_components,
                "weights": {
                    "w_bo": self.w_bo,
                    "w_cost": self.w_cost,
                    "w_disr": self.w_disr,
                },
            }
            out_info["ret_thesis_corrected_step"] = float(
                corrected_ret_components["ret_value"]
            )
            out_info["ret_thesis_corrected"] = {
                **corrected_ret_components,
                "thresholds": {
                    "autotomy_fill_rate_threshold": self.autotomy_threshold,
                    "nonrecovery_disruption_fraction_threshold": (
                        self.nonrecovery_disruption_threshold
                    ),
                    "nonrecovery_fill_rate_threshold": self.nonrecovery_fr_threshold,
                },
                "thresholds_source": "configurable_repo_approximation",
                "correction_mode": "autotomy_equals_recovery",
            }

        return out_obs, float(reward), bool(terminated), bool(truncated), out_info
