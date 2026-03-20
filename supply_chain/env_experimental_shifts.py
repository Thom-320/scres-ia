"""
Gymnasium env with 5th action dimension (shift control) and multiple rewards.

Supports four reward modes:
  - "rt_v0": Legacy R_t v0 (aggregated recovery/holding/service loss).
  - "ReT_thesis": Approximation of Garrido (2017) Eq. 5.5 at step level,
    with linear shift cost δ×(S−1).
  - "control_v1": Operational control reward for RL benchmarking, while
    still exposing corrected ReT_thesis as a reporting-only metric.
  - "control_v1_pbrs": control_v1 + Potential-Based Reward Shaping (Ng et al.
    1999). Adds F(s,s') = γΦ(s') - Φ(s) to control_v1, preserving optimal
    policy. Two variants:
      * cumulative: Φ(s) = -α·max(0, τ - FR_cum(s)) / τ
      * step_level: Φ(s) = -α·prev_step_backorder_qty_norm(s)  [requires v2]

Action space: 5-dimensional [-1, 1]
  [0-3]: Inventory policy multipliers (op3_q, op9_q, op3_rop, op9_rop)
  [4]:   Shift selector — mapped to {1, 2, 3} shifts
         action[4] < -0.33 → S=1 (single shift, 8h/day)
         -0.33 ≤ action[4] < 0.33 → S=2 (double shift, 16h/day)
         action[4] ≥ 0.33 → S=3 (triple shift, 24h/day)

ReT Approximation (Eq. 5.5 mapped to step-level metrics):
  IMPORTANT: This is a STEP-LEVEL APPROXIMATION of the thesis Eq. 5.5.
  The thesis computes ReT per-order (indexed by j) using APj, RPj, DPj.
  This implementation aggregates disruption hours at the step level.

  Thesis values: Re^max=1 (APj), Re=0.5 (RPj per Figure 5.6), Re^min=0 (DPj-RPj).
  Confirmed with Prof. Garrido: operational weighting uses Re^max=1, Re=1, Re^min=0.

  Thesis Equations (Garrido-Rios 2017, Sec. 5.6.3):
    Eq. 5.1: Re(APj) = Re^max × (APj/LT)          [not directly used in step approx.]
    Eq. 5.2: Re(RPj) = Re × (1/RPj)               [not directly used in step approx.]
    Eq. 5.3: Re(DPj,RPj) = Re^min × (DPj-RPj)/CTj [always 0]
    Eq. 5.4: Re(FRt) = 1 - (Bt+Ut)/Dt             [order-count based fill rate]
    Eq. 5.5: ReT = {Re(APj), Re(RPj), Re(DPj,RPj), Re(FRt)}

  Step-level approximation (this implementation):
    Case 1 (No disruption):       Re(FR_t) = fill_rate   [step-level FR, ration-qty]
    Case 2 (Autotomy):            Re(AP)   = 1 - disruption_frac
    Case 3 (Recovery):            Re(RP)   = 1 / (1 + disruption_frac)
    Case 4 (Non-recovery):        Re(DP)   = 0

  Reward = ReT_step - δ × (S - 1)

  Assumptions (publishable):
    A1: Step-level aggregation approximates order-level ReT when
        step size coincides with the reorder cycle (168h).
    A2: AP proxied by fraction of step without disruption (1 - disruption_frac).
    A3: RP proxied inversely by disruption fraction: 1/(1+frac) ∈ (0.5, 1].
    A4: Disruption fraction normalized by op-hours (13 ops × step_hours).
    A5: Fill rate computed from ration quantities (not order counts);
        difference is negligible for narrow demand distribution U(2400,2600).

  These approximations are necessary because the DES tracks order-level APj/RPj/DPj
  implicitly through SimPy process timing, not through explicit per-order timestamps
  at the step boundary. A thesis-exact order-level ReT calculator is available via
  MFSCSimulation._order_level_fill_rate() and OrderRecord.(APj, RPj, DPj, LTj).
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
OBSERVATION_VERSION_OPTIONS = ("v1", "v2", "v3")
REWARD_MODE_OPTIONS = ("ReT_thesis", "rt_v0", "control_v1", "control_v1_pbrs")
PBRS_VARIANT_OPTIONS = ("cumulative", "step_level")
BASE_OBSERVATION_DIM = 15
V2_OBSERVATION_DIM = 18
V3_OBSERVATION_DIM = 20
PREV_STEP_DEMAND_SCALE = 18_200.0
PREV_STEP_BACKORDER_SCALE = 18_200.0
INVENTORY_NODE_FIELDS: tuple[str, ...] = (
    "raw_material_wdc",
    "raw_material_al",
    "rations_al",
    "rations_sb",
    "rations_sb_dispatch",
    "rations_cssu",
    "rations_theatre",
)


class MFSCGymEnvShifts(gym.Env[np.ndarray, np.ndarray]):
    """
    Gymnasium wrapper with dynamic shift control and ReT-based reward.

    Observation:
      - v1: 15-dimensional continuous state vector (historical contract).
      - v2: v1 + previous-step demand/backorder/disruption diagnostics.
      - v3: v2 + normalized cumulative backorder and disruption features.
    Action: 5-dimensional [-1, 1].

    Parameters
    ----------
    reward_mode : {"ReT_thesis", "rt_v0", "control_v1", "control_v1_pbrs"}
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
        observation_version: str = "v1",
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
        # --- PBRS parameters (control_v1_pbrs only) ---
        pbrs_alpha: float = 1.0,
        pbrs_tau: float = 0.95,
        pbrs_gamma: float = 0.99,
        pbrs_variant: str = "cumulative",
    ) -> None:
        super().__init__()
        if step_size_hours <= 0:
            raise ValueError("step_size_hours must be > 0")
        if year_basis not in YEAR_BASIS_OPTIONS:
            raise ValueError(f"Invalid year_basis={year_basis!r}.")
        if risk_level not in ("current", "increased", "severe"):
            raise ValueError(f"Invalid risk_level={risk_level!r}.")
        if reward_mode not in REWARD_MODE_OPTIONS:
            raise ValueError(
                f"Invalid reward_mode={reward_mode!r}. "
                f"Expected one of {REWARD_MODE_OPTIONS}."
            )
        if observation_version not in OBSERVATION_VERSION_OPTIONS:
            raise ValueError(
                f"Invalid observation_version={observation_version!r}. "
                "Expected one of ('v1', 'v2', 'v3')."
            )
        if pbrs_variant not in PBRS_VARIANT_OPTIONS:
            raise ValueError(
                f"Invalid pbrs_variant={pbrs_variant!r}. "
                f"Expected one of {PBRS_VARIANT_OPTIONS}."
            )
        if (
            reward_mode == "control_v1_pbrs"
            and pbrs_variant == "step_level"
            and observation_version not in ("v2", "v3")
        ):
            raise ValueError(
                "PBRS step_level variant requires observation_version='v2' "
                "or 'v3' "
                "because it uses prev_step_backorder_qty_norm (obs[16])."
            )

        self.step_size = float(step_size_hours)
        self.year_basis = year_basis
        self.risk_level = risk_level
        self.stochastic_pt = stochastic_pt
        self.reward_mode = reward_mode
        self.observation_version = observation_version

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

        self.pbrs_alpha = float(pbrs_alpha)
        self.pbrs_tau = float(pbrs_tau)
        self.pbrs_gamma = float(pbrs_gamma)
        self.pbrs_variant = pbrs_variant
        self._prev_phi: float = 0.0

        self.warmup_hours = float(WARMUP["estimated_deterministic_hrs"])
        if max_steps is None:
            self.max_steps = int(
                (SIMULATION_HORIZON - self.warmup_hours) / self.step_size
            )
        else:
            self.max_steps = max_steps

        self.current_step = 0
        self.sim: Optional[MFSCSimulation] = None
        self._prev_step_new_demanded = 0.0
        self._prev_step_new_backorder_qty = 0.0
        self._prev_step_disruption_hours = 0.0
        self._warmup_cumulative_backorder_qty = 0.0
        self._warmup_total_demanded = 0.0
        self._warmup_cumulative_down_hours = 0.0

        obs_dim = self._observation_dim()
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

    def _observation_dim(self) -> int:
        if self.observation_version == "v2":
            return V2_OBSERVATION_DIM
        if self.observation_version == "v3":
            return V3_OBSERVATION_DIM
        return BASE_OBSERVATION_DIM

    def _normalized_cumulative_features(self) -> np.ndarray:
        """Return normalized cumulative diagnostics measured since warmup end."""
        if self.sim is None:
            return np.zeros(2, dtype=np.float32)

        cumulative_backorder_qty = max(
            0.0,
            float(self.sim.cumulative_backorder_qty)
            - self._warmup_cumulative_backorder_qty,
        )
        cumulative_demanded = max(
            0.0,
            float(self.sim.total_demanded) - self._warmup_total_demanded,
        )
        cumulative_down_hours = max(
            0.0,
            float(self.sim._cumulative_down_hours) - self._warmup_cumulative_down_hours,
        )
        elapsed_post_warmup_hours = max(
            0.0, float(self.sim.env.now) - self.warmup_hours
        )

        cum_backorder_rate = min(
            1.0,
            cumulative_backorder_qty / max(cumulative_demanded, 1.0),
        )
        cum_downhours_fraction = min(
            1.0,
            cumulative_down_hours
            / max(elapsed_post_warmup_hours * NUM_TRACKED_OPS, 1.0),
        )
        return np.array(
            [cum_backorder_rate, cum_downhours_fraction],
            dtype=np.float32,
        )

    def _cumulative_backorder_rate_by_inventory_node(self) -> dict[str, float]:
        """
        Return a node-aligned cumulative backorder vector for external models.

        In the current DES, unmet demand materializes only at the final theatre
        sink, so upstream nodes remain zero by construction.
        """
        cumulative_features = self._normalized_cumulative_features()
        cumulative_backorder_rate = float(cumulative_features[0])
        return {
            field_name: (
                cumulative_backorder_rate if field_name == "rations_theatre" else 0.0
            )
            for field_name in INVENTORY_NODE_FIELDS
        }

    def _cumulative_disruption_fraction_by_operation(self) -> dict[str, float]:
        """Return per-operation cumulative disruption fractions since warmup end."""
        if self.sim is None:
            return {f"op{op_id}": 0.0 for op_id in range(1, NUM_TRACKED_OPS + 1)}

        current_time = float(self.sim.env.now)
        elapsed_post_warmup_hours = max(0.0, current_time - self.warmup_hours)
        disruption_hours_by_op = {f"op{op_id}": 0.0 for op_id in range(1, 14)}

        for event in self.sim.risk_events:
            overlap_start = max(float(event.start_time), self.warmup_hours)
            overlap_end = min(float(event.end_time), current_time)
            overlap_duration = max(0.0, overlap_end - overlap_start)
            if overlap_duration <= 0.0:
                continue
            for op_id in event.affected_ops:
                disruption_hours_by_op[f"op{int(op_id)}"] += overlap_duration

        for op_id in range(1, 14):
            down_since = self.sim._op_down_since[op_id]
            if self.sim.op_down_count[op_id] > 0 and down_since is not None:
                overlap_start = max(float(down_since), self.warmup_hours)
                disruption_hours_by_op[f"op{op_id}"] += max(
                    0.0, current_time - overlap_start
                )

        return {
            op_name: min(1.0, down_hours / max(elapsed_post_warmup_hours, 1.0))
            for op_name, down_hours in disruption_hours_by_op.items()
        }

    def _compose_observation(self, base_obs: np.ndarray) -> np.ndarray:
        if self.observation_version == "v1":
            return np.array(base_obs, dtype=np.float32)
        augmented_obs = np.concatenate(
            [
                np.asarray(base_obs, dtype=np.float32),
                np.array(
                    [
                        self._prev_step_new_demanded / PREV_STEP_DEMAND_SCALE,
                        self._prev_step_new_backorder_qty / PREV_STEP_BACKORDER_SCALE,
                        self._prev_step_disruption_hours
                        / max(1.0, self.step_size * NUM_TRACKED_OPS),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        if self.observation_version == "v3":
            augmented_obs = np.concatenate(
                [augmented_obs, self._normalized_cumulative_features()]
            )
        return augmented_obs

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
    # PBRS potential function (Ng et al. 1999)
    # -----------------------------------------------------------------

    def _compute_phi_cumulative(self, obs: np.ndarray) -> float:
        """Cumulative target-deficit potential: Φ = -α·max(0, τ - FR) / τ."""
        fill_rate = float(np.clip(obs[6], 0.0, 1.0))
        deficit = max(0.0, self.pbrs_tau - fill_rate) / self.pbrs_tau
        return -self.pbrs_alpha * deficit

    def _compute_phi_step_level(self, obs: np.ndarray) -> float:
        """Step-level potential using v2 prev_step_backorder_qty_norm (obs[16])."""
        backorder_norm = float(np.clip(obs[16], 0.0, 1.0))
        return -self.pbrs_alpha * backorder_norm

    def _compute_phi(self, obs: np.ndarray) -> float:
        """Dispatch to the active PBRS variant."""
        if self.pbrs_variant == "step_level":
            return self._compute_phi_step_level(obs)
        return self._compute_phi_cumulative(obs)

    # -----------------------------------------------------------------
    # Gymnasium API
    # -----------------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        self.current_step = 0
        self._prev_step_new_demanded = 0.0
        self._prev_step_new_backorder_qty = 0.0
        self._prev_step_disruption_hours = 0.0
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
        self._warmup_cumulative_backorder_qty = float(self.sim.cumulative_backorder_qty)
        self._warmup_total_demanded = float(self.sim.total_demanded)
        self._warmup_cumulative_down_hours = float(self.sim._cumulative_down_hours)
        obs = self._compose_observation(
            np.array(self.sim.get_observation(), dtype=np.float32)
        )
        if self.reward_mode == "control_v1_pbrs":
            self._prev_phi = self._compute_phi(obs)
        info: dict[str, Any] = {
            "time": self.sim.env.now,
            "year_basis": self.year_basis,
            "observation_version": self.observation_version,
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
        info["state_constraint_context"] = self.get_state_constraint_context()
        return obs, info

    def get_state_constraint_context(self) -> dict[str, Any]:
        """
        Return state-dependent operational constraint context.

        This supplements the fixed action constraints with live simulator state
        that affects what can be dispatched or processed at the current step.
        The PPO observation remains unchanged; this method exists for external
        models that need explicit state-conditioned feasibility signals.
        """
        if self.sim is None:
            raise RuntimeError("Call reset() before requesting state constraints.")

        obs = np.asarray(self.sim.get_observation(), dtype=np.float32)
        inventory_detail = self.sim._inventory_detail()
        total_inventory = float(sum(inventory_detail.values()))
        num_raw_materials = float(OPERATIONS[2]["num_units"])
        op3_total_dispatch_cap = float(inventory_detail["raw_material_wdc"])
        op3_per_material_dispatch_cap = op3_total_dispatch_cap / max(
            num_raw_materials, 1.0
        )
        op9_dispatch_cap = float(inventory_detail["rations_sb"])

        return {
            "time": float(self.sim.env.now),
            "inventory_detail": inventory_detail,
            "total_inventory": total_inventory,
            "op3_total_dispatch_cap": op3_total_dispatch_cap,
            "op3_per_material_dispatch_cap": op3_per_material_dispatch_cap,
            "op9_dispatch_cap": op9_dispatch_cap,
            "assembly_line_available": bool(obs[8] < 0.5),
            "any_location_available": bool(obs[9] < 0.5),
            "op9_available": bool(obs[10] < 0.5),
            "op11_available": bool(obs[11] < 0.5),
            "fill_rate": float(obs[6]),
            "backorder_rate": float(obs[7]),
            "time_fraction": float(obs[12]),
            "pending_batch_fraction": float(obs[13]),
            "contingent_demand_fraction": float(obs[14]),
            "cumulative_backorder_qty": float(self.sim.cumulative_backorder_qty),
            "cumulative_disruption_hours": float(self.sim._cumulative_down_hours),
            "pending_backorders_count": float(len(self.sim.pending_backorders)),
            "pending_backorder_qty": float(self.sim.pending_backorder_qty),
            "unattended_orders_total": float(self.sim.total_unattended_orders),
            "cumulative_backorder_rate_by_inventory_node": (
                self._cumulative_backorder_rate_by_inventory_node()
            ),
            "cumulative_disruption_fraction_by_operation": (
                self._cumulative_disruption_fraction_by_operation()
            ),
        }

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
        pbrs_shaping_bonus: float = 0.0
        pbrs_base_reward: float = 0.0
        pbrs_phi: float = 0.0
        if self.reward_mode == "ReT_thesis":
            ret_components = self._compute_ret_thesis_components(info, self.step_size)
            ReT = float(ret_components["ret_value"])
            shift_cost = self.rt_delta * (shifts - 1)
            reward = ReT - shift_cost
        elif self.reward_mode in ("control_v1", "control_v1_pbrs"):
            control_components = self._compute_control_v1_components(info, shifts)
            corrected_ret_components = self._compute_ret_thesis_corrected_components(
                info, self.step_size
            )
            pbrs_base_reward = float(control_components["reward_total"])
            reward = pbrs_base_reward
            if self.reward_mode == "control_v1_pbrs":
                # Compose obs for phi AFTER updating prev-step trackers
                # so v2 features reflect the current transition.
                self._prev_step_new_demanded = float(info.get("new_demanded", 0.0))
                self._prev_step_new_backorder_qty = float(
                    info.get("new_backorder_qty", 0.0)
                )
                self._prev_step_disruption_hours = float(
                    info.get("step_disruption_hours", 0.0)
                )
                phi_obs = self._compose_observation(np.array(obs, dtype=np.float32))
                pbrs_phi = self._compute_phi(phi_obs)
                pbrs_shaping_bonus = self.pbrs_gamma * pbrs_phi - self._prev_phi
                reward = pbrs_base_reward + pbrs_shaping_bonus
                self._prev_phi = pbrs_phi
        else:
            reward = self._compute_rt_v0(info, shifts)

        truncated = self.current_step >= self.max_steps
        if self.reward_mode != "control_v1_pbrs":
            # PBRS already updated these above (needed for phi_obs composition)
            self._prev_step_new_demanded = float(info.get("new_demanded", 0.0))
            self._prev_step_new_backorder_qty = float(
                info.get("new_backorder_qty", 0.0)
            )
            self._prev_step_disruption_hours = float(
                info.get("step_disruption_hours", 0.0)
            )
        out_obs = self._compose_observation(np.array(obs, dtype=np.float32))
        out_info: dict[str, Any] = {
            **info,
            "raw_action": action_arr.tolist(),
            "clipped_action": clipped.tolist(),
            "reward_mode": self.reward_mode,
            "observation_version": self.observation_version,
            "shifts_active": shifts,
            "shift_cost_linear": self.rt_delta * (shifts - 1),
            "shift_cost_delta": self.rt_delta,
        }
        out_info["state_constraint_context"] = self.get_state_constraint_context()
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
        elif self.reward_mode in ("control_v1", "control_v1_pbrs"):
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
            if self.reward_mode == "control_v1_pbrs":
                out_info["pbrs_phi"] = pbrs_phi
                out_info["pbrs_shaping_bonus"] = pbrs_shaping_bonus
                out_info["pbrs_base_reward"] = pbrs_base_reward
                out_info["pbrs_variant"] = self.pbrs_variant
                out_info["pbrs_params"] = {
                    "alpha": self.pbrs_alpha,
                    "tau": self.pbrs_tau,
                    "gamma": self.pbrs_gamma,
                    "variant": self.pbrs_variant,
                }

        return out_obs, float(reward), bool(terminated), bool(truncated), out_info
