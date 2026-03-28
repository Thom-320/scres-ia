"""
Gymnasium env with 5th action dimension (shift control) and multiple rewards.

Primary reward: ReT_seq_v1 (Sequential Operational Resilience)
================================================================
The repo's primary training reward is ``ReT_seq_v1`` with κ=0.20.  It extends
Garrido-Rios (2017) Eq. 5.5 into a smooth, RL-trainable objective using a
**Cobb-Douglas resilience function** (Garrido et al. 2024, IJPR):

    r_t = SC_t^0.60 × BC_t^0.25 × AE_t^0.15

The C-D form ensures non-compensability and smooth gradients for PPO.
See ``_compute_ret_seq_v1`` for the formal thesis mapping.

Other reward modes (historical / auxiliary):
  - "control_v1": Historical linear control reward.  Retained as comparator.
  - "control_v1_pbrs": control_v1 + PBRS shaping (phase-2 extension).
  - "ReT_thesis": Piecewise step-level approximation of Eq. 5.5, retained for
    audit and thesis comparison.  NOT suitable as training objective (collapses
    to S1 due to cost-avoidance incentive dominating the service signal).
  - "ReT_corrected" / "ReT_corrected_cost": Corrected thesis-aligned ReT.
  - "rt_v0": Legacy weighted sum.

Action space: 5-dimensional [-1, 1]
  RL EXTENSION: The thesis (Garrido-Rios 2017, Sec. 6.7.3-6.7.4) controls
  {It,S, S} via static simulation scenarios. Our RL extension generalises
  this to continuous, per-step control of dispatch quantities (Q), reorder
  points (ROP), and shift count (S).

  [0-3]: Inventory policy multipliers (op3_q, op9_q, op3_rop, op9_rop)
         — maps [-1,1] to [0.5, 2.0] via multiplier = 1.25 + 0.75*signal
  [4]:   Shift selector — mapped to {1, 2, 3} shifts
         action[4] < -0.33 → S=1 (single shift, 8h/day)
         -0.33 ≤ action[4] < 0.33 → S=2 (double shift, 16h/day)
         action[4] ≥ 0.33 → S=3 (triple shift, 24h/day)

ReT_thesis piecewise approximation (Eq. 5.5, audit-only):
  Thesis Equations (Garrido-Rios 2017, Sec. 5.6.3):
    Eq. 5.1: Re(APj) = Re^max × (APj/LT)
    Eq. 5.2: Re(RPj) = Re × (1/RPj)
    Eq. 5.3: Re(DPj,RPj) = Re^min × (DPj-RPj)/CTj  [always 0]
    Eq. 5.4: Re(FRt) = 1 - (Bt+Ut)/Dt
    Eq. 5.5: ReT = {Re(APj), Re(RPj), Re(DPj,RPj), Re(FRt)}

  This piecewise formulation is retained for audit but is NOT the training
  objective.  See ``_compute_ret_thesis_components`` for the step-level
  approximation and REWARD_DESIGN.md for why it fails as an RL reward.
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
OBSERVATION_VERSION_OPTIONS = ("v1", "v2", "v3", "v4")
REWARD_MODE_ALIAS_MAP = {"ReT_corrected_cost": "ReT_corrected"}
REWARD_MODE_OPTIONS = (
    "ReT_thesis",
    "ReT_corrected",
    "ReT_corrected_cost",
    "ReT_seq_v1",
    "ReT_cd",
    "ReT_garrido2024",
    "rt_v0",
    "control_v1",
    "control_v1_pbrs",
    "ReT_cd_v1",
    "ReT_cd_sigmoid",
)

# ReT_seq_v1 defaults (Sequential Operational Resilience)
RET_SEQ_W_SC = 0.60  # service continuity weight
RET_SEQ_W_BC = 0.25  # backlog containment weight
RET_SEQ_W_AE = 0.15  # adaptive efficiency weight
RET_SEQ_KAPPA = 0.20  # shift cost scaling

# ReT_cd defaults (Cobb-Douglas Resilience, Garrido et al. 2024 methodology)
RET_CD_A = 0.60  # fill rate exponent (directly proportional)
RET_CD_B = 0.15  # inverse backlog exponent (inversely proportional)
RET_CD_C = 0.10  # spare capacity exponent (directly proportional)
RET_CD_D = 0.15  # inverse cost exponent (inversely proportional)
RET_CD_KAPPA = 0.20  # cost scaling (same as ret_seq_kappa)
RET_CD_BO_NORM = 5000.0  # backorder normalization constant (typical demand scale)

# ReT_garrido2024: faithful C-D with raw DES variables + sigmoid (Eq. 3-6)
# Calibrated from 200 episodes (52,000 steps) under increased + stochastic_pt.
# Method: Garrido et al. (2024, IJPR) Section 3.3 — equate each ln-argument to 1/5.
G24_A_ZETA = 0.0118    # ζ (inventory) — directly proportional
G24_B_EPSILON = 0.0167  # ε (backorders) — inversely proportional
G24_C_PHI = 0.0184      # φ (production capacity) — directly proportional
G24_D_TAU = 0.0857      # τ (backlog-to-demand ratio) — inversely proportional
G24_N_KAPPA = 0.0390    # κ̇ (shift-hours cost) — inversely proportional
G24_RATE_PER_HOUR = 320.5  # assembly rate (rations/hour)

# ReT_cd_v1 defaults (Cobb-Douglas continuous bridge for ReT_thesis piecewise)
RET_CD_W_FR = 0.70   # fill-rate weight (primary service signal)
RET_CD_W_AT = 0.30   # availability (1 - disruption_frac) weight

PBRS_VARIANT_OPTIONS = ("cumulative", "step_level")
BASE_OBSERVATION_DIM = 15
V2_OBSERVATION_DIM = 18
V3_OBSERVATION_DIM = 20
V4_OBSERVATION_DIM = 24  # v3 (20) + rations_sb_dispatch + shifts + op1_down + op2_down
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
      - v4: v3 + current shift plus Op1/Op2 disruption state.
    Action: 5-dimensional [-1, 1].

    Parameters
    ----------
    reward_mode : {"ReT_thesis", "ReT_corrected", "ReT_corrected_cost", "ReT_seq_v1", "rt_v0", "control_v1", "control_v1_pbrs"}
        Which reward formulation to use. ``ReT_corrected_cost`` is the
        research-facing alias for the cost-extended corrected thesis lane and
        maps internally to ``ReT_corrected``.
    rt_delta : float
        Linear shift cost weight used by the thesis-aligned corrected lane.
        Reward -= δ × (S − 1). Must be calibrated via DOE for desired shift
        selection ratio.
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
        pbrs_beta: float = 0.5,
        pbrs_gamma: float = 0.99,
        pbrs_variant: str = "cumulative",
        # --- ReT_seq_v1 parameters ---
        ret_seq_w_sc: float = RET_SEQ_W_SC,
        ret_seq_w_bc: float = RET_SEQ_W_BC,
        ret_seq_w_ae: float = RET_SEQ_W_AE,
        ret_seq_kappa: float = RET_SEQ_KAPPA,
        # --- ReT_cd parameters (Cobb-Douglas resilience) ---
        ret_cd_use_sigmoid: bool = False,
        ret_cd_a: float = RET_CD_A,
        ret_cd_b: float = RET_CD_B,
        ret_cd_c: float = RET_CD_C,
        ret_cd_d: float = RET_CD_D,
        ret_cd_kappa: float = RET_CD_KAPPA,
        ret_cd_bo_norm: float = RET_CD_BO_NORM,
        # --- ReT_cd_v1 / ReT_cd_sigmoid parameters ---
        ret_cd_w_fr: float = RET_CD_W_FR,
        ret_cd_w_at: float = RET_CD_W_AT,
    ) -> None:
        super().__init__()
        if step_size_hours <= 0:
            raise ValueError("step_size_hours must be > 0")
        if year_basis not in YEAR_BASIS_OPTIONS:
            raise ValueError(f"Invalid year_basis={year_basis!r}.")
        if risk_level not in (
            "current",
            "increased",
            "severe",
            "severe_extended",
            "severe_training",
        ):
            raise ValueError(f"Invalid risk_level={risk_level!r}.")
        if reward_mode not in REWARD_MODE_OPTIONS:
            raise ValueError(
                f"Invalid reward_mode={reward_mode!r}. "
                f"Expected one of {REWARD_MODE_OPTIONS}."
            )
        if observation_version not in OBSERVATION_VERSION_OPTIONS:
            raise ValueError(
                f"Invalid observation_version={observation_version!r}. "
                f"Expected one of {OBSERVATION_VERSION_OPTIONS}."
            )
        if pbrs_variant not in PBRS_VARIANT_OPTIONS:
            raise ValueError(
                f"Invalid pbrs_variant={pbrs_variant!r}. "
                f"Expected one of {PBRS_VARIANT_OPTIONS}."
            )
        canonical_reward_mode = REWARD_MODE_ALIAS_MAP.get(reward_mode, reward_mode)
        if (
            canonical_reward_mode == "control_v1_pbrs"
            and pbrs_variant == "step_level"
            and observation_version not in ("v2", "v3", "v4")
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
        self._canonical_reward_mode = canonical_reward_mode
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
        self.pbrs_beta = float(pbrs_beta)
        self.pbrs_gamma = float(pbrs_gamma)
        self.pbrs_variant = pbrs_variant
        self._prev_phi: float = 0.0

        # ReT_seq_v1 params
        self.ret_seq_w_sc = float(ret_seq_w_sc)
        self.ret_seq_w_bc = float(ret_seq_w_bc)
        self.ret_seq_w_ae = float(ret_seq_w_ae)
        self.ret_seq_kappa = float(ret_seq_kappa)

        # ReT_cd params (Cobb-Douglas resilience, Garrido et al. 2024 methodology)
        self.ret_cd_use_sigmoid = bool(ret_cd_use_sigmoid)
        self.ret_cd_a = float(ret_cd_a)
        self.ret_cd_b = float(ret_cd_b)
        self.ret_cd_c = float(ret_cd_c)
        self.ret_cd_d = float(ret_cd_d)
        self.ret_cd_kappa = float(ret_cd_kappa)
        self.ret_cd_bo_norm = float(ret_cd_bo_norm)

        # ReT_cd_v1 / ReT_cd_sigmoid params
        self.ret_cd_w_fr = float(ret_cd_w_fr)
        self.ret_cd_w_at = float(ret_cd_w_at)

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
        # Observation bounds: inventory dims [0-5] are normalized by 1e6/1e5
        # and rarely exceed ~10 in practice; rates [6-7] are in [0,1];
        # flags [8-11] are binary; time/batch/demand [12-14] are in [0,~5].
        # VecNormalize further normalizes during training, but finite bounds
        # make the space definition correct for Gymnasium compliance.
        self.observation_space = spaces.Box(
            low=0.0, high=20.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

    def _observation_dim(self) -> int:
        if self.observation_version == "v4":
            return V4_OBSERVATION_DIM
        if self.observation_version == "v3":
            return V3_OBSERVATION_DIM
        if self.observation_version == "v2":
            return V2_OBSERVATION_DIM
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

    def _cumulative_demanded_post_warmup(self) -> float:
        """Return total demanded quantity accrued since warmup end."""
        if self.sim is None:
            return 0.0
        return max(0.0, float(self.sim.total_demanded) - self._warmup_total_demanded)

    def _cumulative_backorder_qty_post_warmup(self) -> float:
        """Return cumulative backorder quantity accrued since warmup end."""
        if self.sim is None:
            return 0.0
        return max(
            0.0,
            float(self.sim.cumulative_backorder_qty)
            - self._warmup_cumulative_backorder_qty,
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
        if self.observation_version in ("v3", "v4"):
            augmented_obs = np.concatenate(
                [augmented_obs, self._normalized_cumulative_features()]
            )
        if self.observation_version == "v4" and self.sim is not None:
            augmented_obs = np.concatenate(
                [augmented_obs, self.sim.get_observation_v4_extra()]
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
    # ReT_seq_v1: Sequential Operational Resilience
    # -----------------------------------------------------------------

    def _compute_ret_seq_v1(
        self, info: dict[str, Any], shifts: int
    ) -> dict[str, float]:
        """
        Sequential Operational Resilience (ReT-Seq).

        Primary training reward for the shift-control RL lane.  Extends
        Garrido-Rios (2017) Eq. 5.5 into a smooth, RL-trainable objective
        using a **Cobb-Douglas (C-D) resilience function** following the
        methodology of Garrido et al. (2024, IJPR):

            r_t = SC_t^w_sc × BC_t^w_bc × AE_t^w_ae

        This is the standard C-D multiplicative form where each factor
        captures one dimension of resilience.  In log-linear form:

            ln(r_t) = 0.60·ln(SC_t) + 0.25·ln(BC_t) + 0.15·ln(AE_t)

        The C-D form guarantees non-compensability (if any factor → 0,
        the product → 0) and yields smooth gradients suitable for
        policy-gradient optimization (PPO).

        Frozen defaults: w_sc=0.60, w_bc=0.25, w_ae=0.15, κ=0.20.

        Sub-indicator definitions and thesis mapping
        ---------------------------------------------

        SC_t  (Service Continuity) = 1 - B_step / D_step

            Maps to thesis Eq. 5.4: Re(FR_t) = 1 - (B_t + U_t) / D_t.
            U_t = 0 in the MFSC because all unmet demand is backordered
            (thesis assumption 6.5.4), so SC_t equals the step-level fill
            rate.  This captures *inherent resilience* — the fraction of
            demand satisfied regardless of disruption state.

        BC_t  (Backlog Containment) = 1 - min(1, pending_BO / cumul_D)

            Sequential proxy for the thesis recovery concept (Eq. 5.2:
            Re(RP_j) = Re_bar / RP_j).  The thesis measures recovery by
            *time* (shorter RP_j = higher resilience); at the step level,
            recovery time is not directly observable because orders have not
            yet completed.  Pending backorder stock relative to cumulative
            demand is the most direct observable consequence of delayed
            recovery: faster recovery drains the backlog, raising BC_t.
            The cumulative denominator intentionally measures *overall
            recovery health* across the episode rather than instantaneous
            state, complementing SC_t which handles immediate service impact.

        AE_t  (Adaptive Efficiency) = 1 - κ(S_t - 1) / 2

            Addresses two explicit thesis gaps:
            - Limitation 8.5.2: "the non-inclusion of the cost factor"
            - Future work 8.6.2: "in search of an optimum level of SCRes"
            Maps the discrete shift decision {1,2,3} to a cost penalty in
            [0, κ].  At κ=0.20: AE(S=1)=1.00, AE(S=2)=0.90, AE(S=3)=0.80.

        Thesis sub-indicators NOT explicitly represented
        -------------------------------------------------
        Re(AP_j) — autotomy (Eq. 5.1): manifests as SC_t ≈ 1 when the system
            absorbs disruptions without service loss.  No separate term is
            needed because RL rewards good outcomes regardless of whether
            they arise from autotomy or calm periods.
        Re(DP_j, RP_j) — non-recovery (Eq. 5.3): always zero in the thesis
            (Re^min = 0).  Manifests as SC_t → 0 AND BC_t → 0 when the
            system is fully disrupted and not recovering.

        Why Cobb-Douglas instead of piecewise (Eq. 5.5)
        --------------------------------------------------
        The thesis Eq. 5.5 selects one sub-indicator per order based on the
        disruption state.  This piecewise structure creates discontinuous
        reward landscapes unsuitable for policy-gradient optimization.

        The Cobb-Douglas (C-D) form is the standard continuous alternative
        for multi-factor resilience indices (Garrido et al. 2024, IJPR;
        Fan et al. 2022; Jandhana et al. 2018).  It preserves the key
        property: *non-compensability* — if any sub-indicator approaches
        zero, the entire product approaches zero, regardless of the others.
        Additional precedent: the Human Development Index uses C-D
        aggregation for the same reason (UNDP, 2010).

        Weight justification (0.60 / 0.25 / 0.15)
        -------------------------------------------
        Weights sum to 1.0 (proper weighted geometric mean) and reflect the
        thesis priority hierarchy:
        - Service dominates (0.60): thesis assigns Re^max = 1.0 to autotomy
          and no-disruption cases — both service-dominant states.
        - Recovery is secondary (0.25): thesis assigns Re_bar ≈ 0.5 to
          recovery, roughly half of Re^max (Figure 5.6).
        - Cost efficiency is tertiary (0.15): new extension not in the
          original thesis; given minimal weight as befits an auxiliary
          dimension in military logistics where feeding troops >>
          recovery speed >> operational cost.
        """
        EPS = 1e-6

        new_demanded = float(info.get("new_demanded", 0.0))
        new_backorder_qty = float(info.get("new_backorder_qty", 0.0))

        # SC_t: service continuity (maps to Re(FRt) Eq. 5.4)
        if new_demanded > 0:
            sc_t = max(EPS, 1.0 - new_backorder_qty / new_demanded)
        else:
            sc_t = 1.0

        # BC_t: backlog containment (captures recovery dynamics)
        pending_bo_qty = float(info.get("pending_backorder_qty", 0.0))
        cumulative_demanded = max(self._cumulative_demanded_post_warmup(), 1.0)
        bc_t = max(EPS, 1.0 - min(1.0, pending_bo_qty / cumulative_demanded))

        # AE_t: adaptive efficiency (cost dimension per Section 8.6.2)
        ae_t = max(EPS, 1.0 - self.ret_seq_kappa * (shifts - 1) / 2.0)

        # Geometric aggregation (reduces compensability)
        ret_seq_t = (
            sc_t**self.ret_seq_w_sc * bc_t**self.ret_seq_w_bc * ae_t**self.ret_seq_w_ae
        )

        return {
            "service_continuity": sc_t,
            "backlog_containment": bc_t,
            "adaptive_efficiency": ae_t,
            "ret_seq_step": ret_seq_t,
            "cumulative_demanded_post_warmup": cumulative_demanded,
            "pending_backorder_qty": pending_bo_qty,
            "weights": {
                "w_sc": self.ret_seq_w_sc,
                "w_bc": self.ret_seq_w_bc,
                "w_ae": self.ret_seq_w_ae,
            },
            "kappa": self.ret_seq_kappa,
        }

    # -----------------------------------------------------------------
    # ReT_cd: Cobb-Douglas Resilience (Garrido et al. 2024 methodology)
    # -----------------------------------------------------------------

    def _compute_ret_cd(
        self, info: dict[str, Any], shifts: int
    ) -> dict[str, float]:
        """
        Cobb-Douglas Resilience reward (ReT_cd).

        Applies the Garrido et al. (2024, IJPR) Eq. 3-6 methodology to the
        MFSC DES with four step-level variables:

            r_t = FR^a * IB^b * SC_cap^c * IC^d

        Or in log-linear form:

            ln(r_t) = a*ln(FR) + b*ln(IB) + c*ln(SC_cap) + d*ln(IC)

        **Variant A (bounded, default):** r_t = exp(z), already bounded
        because all inputs are in (0, 1].

        **Variant B (sigmoid):** r_t = 1 / (1 + exp(-z)), following
        Garrido 2024 Eq. 6.

        Variable definitions
        --------------------
        FR   (fill rate) -- directly proportional to R.
             Source: 1 - new_backorder_qty / new_demanded.
             Maps to thesis Eq. 5.4 Re(FRt).

        IB   (inverse backlog) -- inversely proportional to pending BO.
             Source: 1 / (1 + pending_bo_qty / bo_norm).
             Maps to thesis Eq. 5.2 Re(RPj): shorter recovery = fewer
             pending backorders.  Uses 1/(1+x) to maintain inverse
             proportionality without division-by-zero risk.

        SC_cap (spare capacity) -- directly proportional to R.
             Source: shifts_active / 3.0 (fraction of max capacity).
             Maps to thesis Sec. 6.7.4 short-term manufacturing capacity.

        IC   (inverse cost) -- inversely proportional to cost deviation.
             Source: 1 / (1 + kappa * (shifts - 1) / 2).
             Maps to thesis Sec. 8.6.2 cost integration.

        Key differences from ReT_seq_v1
        --------------------------------
        - 4 variables instead of 3 (separates capacity and cost).
        - Backlog uses hyperbolic 1/(1+x) form (thesis Eq. 5.2) instead
          of pre-normalized subtraction.
        - Capacity enters as a *positive* factor (more shifts = better
          service) while cost enters as a *negative* factor (more shifts =
          higher cost).  ReT_seq_v1 combines both into a single AE penalty.
        - Optional sigmoid bounding (Garrido 2024 Eq. 6).

        Frozen defaults: a=0.60, b=0.15, c=0.10, d=0.15, kappa=0.20.
        """
        EPS = 1e-6

        new_demanded = float(info.get("new_demanded", 0.0))
        new_backorder_qty = float(info.get("new_backorder_qty", 0.0))
        pending_bo_qty = float(info.get("pending_backorder_qty", 0.0))

        # FR: fill rate (directly proportional)
        if new_demanded > 0:
            fr = max(EPS, 1.0 - new_backorder_qty / new_demanded)
        else:
            fr = 1.0

        # IB: inverse backlog (inversely proportional, hyperbolic form)
        ib = 1.0 / (1.0 + pending_bo_qty / self.ret_cd_bo_norm)

        # SC_cap: spare capacity (directly proportional)
        sc_cap = max(EPS, float(shifts) / 3.0)

        # IC: inverse cost (inversely proportional)
        ic = 1.0 / (1.0 + self.ret_cd_kappa * (shifts - 1) / 2.0)

        # Log-linear Cobb-Douglas score
        log_score = float(
            self.ret_cd_a * np.log(max(EPS, fr))
            + self.ret_cd_b * np.log(max(EPS, ib))
            + self.ret_cd_c * np.log(max(EPS, sc_cap))
            + self.ret_cd_d * np.log(max(EPS, ic))
        )

        if self.ret_cd_use_sigmoid:
            # Variant B: sigmoid-bounded (Garrido 2024 Eq. 6)
            ret_cd_t = float(1.0 / (1.0 + np.exp(-log_score)))
            variant = "sigmoid"
        else:
            # Variant A: bounded C-D (inputs in (0,1] so product in (0,1])
            ret_cd_t = float(np.exp(log_score))
            variant = "bounded"

        return {
            "fill_rate": fr,
            "inverse_backlog": ib,
            "spare_capacity": sc_cap,
            "inverse_cost": ic,
            "log_score": log_score,
            "ret_cd_step": ret_cd_t,
            "variant": variant,
            "use_sigmoid": self.ret_cd_use_sigmoid,
            "pending_backorder_qty": pending_bo_qty,
            "exponents": {
                "a": self.ret_cd_a,
                "b": self.ret_cd_b,
                "c": self.ret_cd_c,
                "d": self.ret_cd_d,
            },
            "kappa": self.ret_cd_kappa,
            "bo_norm": self.ret_cd_bo_norm,
        }

    # -----------------------------------------------------------------
    # ReT_garrido2024: Faithful C-D with raw DES variables + sigmoid
    # -----------------------------------------------------------------

    def _compute_ret_garrido2024(
        self, info: dict[str, Any], shifts: int
    ) -> dict[str, float]:
        """
        Faithful Cobb-Douglas resilience per Garrido et al. (2024, IJPR) Eq. 3-6.

        Uses RAW DES output variables (not pre-normalized) and applies the
        log-linear + sigmoid methodology exactly as described in the paper.

        Variables (Garrido 2024 mapping to MFSC):
          ζ = total_inventory        (directly proportional to R)
          ε = pending_backorder_qty  (inversely proportional)
          φ = weekly_production_cap  (directly proportional)
          τ = backlog/demand ratio   (inversely proportional)
          κ̇ = shift-hours/week      (inversely proportional)

        Log-linear form (Eq. 4):
          z = a·ln(ζ) - b·ln(ε) + c·ln(φ) - d·ln(τ) - n·ln(κ̇)

        Sigmoid-bounded (Eq. 6):
          R = 1 / (1 + exp(-z))    ∈ (0, 1)

        Exponents calibrated from 200 episodes (52,000 steps) following
        Section 3.3: equate each ln-argument to 1/5 at its P99 maximum.
        """
        import math

        total_inv = max(1.0, float(info.get("total_inventory", 1.0)))
        pending_bo = max(1.0, float(info.get("pending_backorder_qty", 1.0)))
        new_demanded = max(1.0, float(info.get("new_demanded", 1.0)))

        # Raw C-D variables
        zeta = total_inv
        epsilon = pending_bo
        phi = G24_RATE_PER_HOUR * shifts * 8.0 * 7.0  # weekly capacity
        tau = max(1.0, pending_bo / new_demanded)
        kappa_dot = max(1.0, shifts * 8.0 * 7.0)  # shift-hours/week

        # Log-linear score (Garrido 2024 Eq. 4-5)
        z_score = (
            G24_A_ZETA * math.log(zeta)
            - G24_B_EPSILON * math.log(epsilon)
            + G24_C_PHI * math.log(phi)
            - G24_D_TAU * math.log(tau)
            - G24_N_KAPPA * math.log(kappa_dot)
        )

        # Sigmoid bounding (Garrido 2024 Eq. 6)
        r_sigmoid = 1.0 / (1.0 + math.exp(-z_score))

        return {
            "zeta": zeta,
            "epsilon": epsilon,
            "phi": phi,
            "tau": tau,
            "kappa_dot": kappa_dot,
            "z_score": z_score,
            "r_garrido2024": r_sigmoid,
            "coefficients": {
                "a_zeta": G24_A_ZETA,
                "b_epsilon": G24_B_EPSILON,
                "c_phi": G24_C_PHI,
                "d_tau": G24_D_TAU,
                "n_kappa": G24_N_KAPPA,
            },
        }

    # -----------------------------------------------------------------
    # ReT_cd_v1 / ReT_cd_sigmoid: Cobb-Douglas continuous bridge
    # -----------------------------------------------------------------

    def _compute_ret_cd_v1(
        self, info: dict[str, Any], step_hours: float
    ) -> dict[str, float | str]:
        """
        ReT_cd_v1: Cobb-Douglas continuous bridge for the piecewise ReT_thesis.

        Motivation
        ----------
        ReT_thesis (Garrido-Rios 2017, Eq. 5.5) selects one sub-indicator per
        step based on disruption state, producing a **piecewise-discontinuous**
        reward landscape that is theoretically ill-suited for policy-gradient
        optimization (PPO).  The case-boundary discontinuities create sharp
        gradients that destabilize training, and the autotomy branch
        (R = 1 − d_frac) is locally non-monotone relative to the recovery
        branch (R = 1/(1 + d_frac)).

        The Cobb-Douglas (C-D) form is the standard continuous alternative for
        multi-factor resilience indices (Garrido et al. 2024, IJPR; Fan et al.
        2022).  It preserves non-compensability while providing smooth,
        differentiable signals.

        Formulation
        -----------
            FR_t = max(EPS, 1 − backorder_qty / demand)      [thesis Eq. 5.4]
            AT_t = max(EPS, 1 − disruption_frac)             [availability]
            R_t  = FR_t^w_fr × AT_t^w_at                    [raw C-D, ∈ (0,1]]

        where w_fr + w_at = 1.0 (frozen: 0.70 / 0.30).

        Why NOT sigmoid here
        --------------------
        A sigmoid wrapper is appropriate when the log-linear sum is UNBOUNDED
        (e.g., Garrido 2024 with macroeconomic variables ζ, ε, φ, τ, κ̇ that
        can take large values).  Here, FR_t and AT_t are already in [0, 1],
        so their logs are NEGATIVE.  Sigmoid of a negative value is < 0.5,
        meaning the best possible reward when FR=1 and disruption=0 would be
        σ(0) = 0.5 — cutting the effective range in half and creating a
        systematic negative bias.  Raw C-D is the correct form.

        See `ReT_cd_sigmoid` for the sigmoid variant, included only to document
        this failure mode empirically.

        Weight justification (0.70 / 0.30)
        -----------------------------------
        - FR dominates (0.70): thesis assigns Re^max = 1.0 to no-disruption
          cases, which are fully service-signal dominated.
        - Availability secondary (0.30): thesis assigns Re_bar ≈ 0.5 to
          recovery — partial weight to the disruption dimension.
        - Weights sum to 1.0 → output is a proper weighted geometric mean
          in (0, 1].  No additional rescaling required for RL training.
        """
        EPS = 1e-6

        demanded = float(info.get("new_demanded", 0.0))
        backorder_qty = float(info.get("new_backorder_qty", 0.0))
        disruption_hrs = float(info.get("step_disruption_hours", 0.0))
        max_op_hours = step_hours * NUM_TRACKED_OPS
        disruption_frac = min(1.0, disruption_hrs / max(1.0, max_op_hours))

        if demanded > 0:
            fr_t = max(EPS, 1.0 - backorder_qty / demanded)
        else:
            fr_t = 1.0

        at_t = max(EPS, 1.0 - disruption_frac)

        # Raw Cobb-Douglas (log-linear form for numerical stability)
        log_r = self.ret_cd_w_fr * np.log(fr_t) + self.ret_cd_w_at * np.log(at_t)
        r_t = float(np.exp(log_r))

        return {
            "fill_rate_step": fr_t,
            "availability_step": at_t,
            "disruption_frac": disruption_frac,
            "log_r": float(log_r),
            "ret_cd_step": r_t,
            "reward_mode": "ReT_cd_v1",
            "weights": {"w_fr": self.ret_cd_w_fr, "w_at": self.ret_cd_w_at},
        }

    def _compute_ret_cd_sigmoid(
        self, info: dict[str, Any], step_hours: float
    ) -> dict[str, float | str]:
        """
        ReT_cd_sigmoid: experimental variant — sigmoid applied to C-D log score.

        NOT RECOMMENDED FOR TRAINING.  Included solely as a comparison to
        demonstrate the systematic downward bias of sigmoid when inputs are
        already in [0, 1]:

            FR=1.0, AT=1.0 → log_score = 0.0 → σ(0) = 0.50

        The best achievable reward is 0.50, not 1.0.  This artificially
        compresses the learning signal and makes convergence harder.

        Formulation
        -----------
            log_score = w_fr·ln(FR_t) + w_at·ln(AT_t)   ← always ≤ 0
            R_t = σ(log_score) = 1 / (1 + exp(−log_score))  ← always ≤ 0.5

        Equivalent to ReT_cd_v1 with an extra sigmoid wrapper.  Uses the same
        weights (0.70 / 0.30) to isolate the effect of sigmoid on scale.
        """
        components = self._compute_ret_cd_v1(info, step_hours)
        log_r = float(components["log_r"])
        r_sigmoid = float(1.0 / (1.0 + np.exp(-log_r)))
        return {
            **components,
            "ret_cd_sigmoid_step": r_sigmoid,
            "ret_cd_raw_step": float(components["ret_cd_step"]),
            "reward_mode": "ReT_cd_sigmoid",
            "sigmoid_bias_note": (
                "sigmoid(log_r) ≤ 0.5 always because log_r ≤ 0 when inputs ∈ (0,1]"
            ),
        }

    # -----------------------------------------------------------------
    # PBRS potential function (Ng et al. 1999)
    # -----------------------------------------------------------------

    def _compute_phi_cumulative(self, obs: np.ndarray) -> float:
        """New potential function: Φ(s) = α * fill_rate - β * backorder_rate"""
        fill_rate = float(np.clip(obs[6], 0.0, 1.0))
        # use prev_step_backorder_qty_norm if v2 or fallback to 0
        backorder_rate = float(np.clip(obs[16], 0.0, 1.0)) if len(obs) > 16 else 0.0
        alpha = getattr(self, "pbrs_alpha", 1.0)
        beta = getattr(self, "pbrs_beta", 0.5)
        return alpha * fill_rate - beta * backorder_rate

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
        if self._canonical_reward_mode == "control_v1_pbrs":
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
            "cumulative_backorder_qty_post_warmup": (
                self._cumulative_backorder_qty_post_warmup()
            ),
            "cumulative_demanded_post_warmup": self._cumulative_demanded_post_warmup(),
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
        ret_cd_components: dict[str, float | str] | None = None
        ret_cd_4v_components: dict[str, float] | None = None
        pbrs_shaping_bonus: float = 0.0
        pbrs_base_reward: float = 0.0
        pbrs_phi: float = 0.0
        if self._canonical_reward_mode == "ReT_thesis":
            ret_components = self._compute_ret_thesis_components(info, self.step_size)
            ReT = float(ret_components["ret_value"])
            shift_cost = self.rt_delta * (shifts - 1)
            reward = ReT - shift_cost
        elif self._canonical_reward_mode == "ReT_corrected":
            # Extended ReT: uses corrected autotomy + shift cost.
            # Fulfills Garrido-Rios (2017) Section 8.6.2 call for cost integration.
            corrected_ret_components = self._compute_ret_thesis_corrected_components(
                info, self.step_size
            )
            ReT_corr = float(corrected_ret_components["ret_value"])
            shift_cost = self.rt_delta * (shifts - 1)
            reward = ReT_corr - shift_cost
            ret_components = self._compute_ret_thesis_components(info, self.step_size)
        elif self._canonical_reward_mode == "ReT_seq_v1":
            ret_seq_components = self._compute_ret_seq_v1(info, shifts)
            reward = float(ret_seq_components["ret_seq_step"])
            corrected_ret_components = self._compute_ret_thesis_corrected_components(
                info, self.step_size
            )
            ret_components = self._compute_ret_thesis_components(info, self.step_size)
        elif self._canonical_reward_mode == "ReT_garrido2024":
            g24_components = self._compute_ret_garrido2024(info, shifts)
            reward = float(g24_components["r_garrido2024"])
            corrected_ret_components = self._compute_ret_thesis_corrected_components(
                info, self.step_size
            )
            ret_components = self._compute_ret_thesis_components(info, self.step_size)
        elif self._canonical_reward_mode in ("control_v1", "control_v1_pbrs"):
            control_components = self._compute_control_v1_components(info, shifts)
            corrected_ret_components = self._compute_ret_thesis_corrected_components(
                info, self.step_size
            )
            pbrs_base_reward = float(control_components["reward_total"])
            reward = pbrs_base_reward
            if self._canonical_reward_mode == "control_v1_pbrs":
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
        elif self._canonical_reward_mode == "ReT_cd":
            ret_cd_4v_components = self._compute_ret_cd(info, shifts)
            reward = float(ret_cd_4v_components["ret_cd_step"])
            corrected_ret_components = self._compute_ret_thesis_corrected_components(
                info, self.step_size
            )
            ret_components = self._compute_ret_thesis_components(info, self.step_size)
        elif self._canonical_reward_mode == "ReT_cd_v1":
            ret_cd_components = self._compute_ret_cd_v1(info, self.step_size)
            reward = float(ret_cd_components["ret_cd_step"])
            corrected_ret_components = self._compute_ret_thesis_corrected_components(
                info, self.step_size
            )
            ret_components = self._compute_ret_thesis_components(info, self.step_size)
        elif self._canonical_reward_mode == "ReT_cd_sigmoid":
            ret_cd_components = self._compute_ret_cd_sigmoid(info, self.step_size)
            reward = float(ret_cd_components["ret_cd_sigmoid_step"])
            corrected_ret_components = self._compute_ret_thesis_corrected_components(
                info, self.step_size
            )
            ret_components = self._compute_ret_thesis_components(info, self.step_size)
        else:
            reward = self._compute_rt_v0(info, shifts)

        truncated = self.current_step >= self.max_steps
        if self._canonical_reward_mode != "control_v1_pbrs":
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
            "cumulative_demanded_post_warmup": self._cumulative_demanded_post_warmup(),
            "cumulative_backorder_qty_post_warmup": (
                self._cumulative_backorder_qty_post_warmup()
            ),
        }
        out_info["state_constraint_context"] = self.get_state_constraint_context()
        if self._canonical_reward_mode == "ReT_thesis":
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
        elif self._canonical_reward_mode == "ReT_corrected":
            out_info["ReT_corrected_raw"] = ReT_corr
            out_info["ret_thesis_corrected_step"] = ReT_corr
            out_info["ret_thesis_corrected"] = {
                **corrected_ret_components,
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
                "correction_mode": "autotomy_equals_recovery",
            }
            # Also include uncorrected for comparison
            out_info["ret_components"] = {
                **ret_components,
                "shift_cost": shift_cost,
            }
        elif self._canonical_reward_mode == "ReT_seq_v1":
            out_info["ret_seq_step"] = float(ret_seq_components["ret_seq_step"])
            out_info["ret_seq_components"] = ret_seq_components
            out_info["service_continuity_step"] = float(
                ret_seq_components["service_continuity"]
            )
            out_info["backlog_containment_step"] = float(
                ret_seq_components["backlog_containment"]
            )
            out_info["adaptive_efficiency_step"] = float(
                ret_seq_components["adaptive_efficiency"]
            )
            out_info["ret_seq_kappa"] = float(self.ret_seq_kappa)
            out_info["ret_thesis_corrected_step"] = float(
                corrected_ret_components["ret_value"]
            )
            out_info["ret_thesis_corrected"] = corrected_ret_components
            out_info["ret_components"] = ret_components
        elif self._canonical_reward_mode == "ReT_garrido2024":
            out_info["ret_garrido2024_step"] = float(g24_components["r_garrido2024"])
            out_info["ret_garrido2024_z_score"] = float(g24_components["z_score"])
            out_info["ret_garrido2024_components"] = g24_components
            out_info["ret_thesis_corrected_step"] = float(
                corrected_ret_components["ret_value"]
            )
            out_info["ret_thesis_corrected"] = corrected_ret_components
            out_info["ret_components"] = ret_components
        elif self._canonical_reward_mode in ("control_v1", "control_v1_pbrs"):
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
            if self._canonical_reward_mode == "control_v1_pbrs":
                out_info["pbrs_phi"] = pbrs_phi
                out_info["pbrs_shaping_bonus"] = pbrs_shaping_bonus
                out_info["pbrs_base_reward"] = pbrs_base_reward
                out_info["pbrs_variant"] = self.pbrs_variant
                out_info["pbrs_params"] = {
                    "alpha": self.pbrs_alpha,
                    "beta": self.pbrs_beta,
                    "gamma": self.pbrs_gamma,
                    "variant": self.pbrs_variant,
                }
        elif self._canonical_reward_mode == "ReT_cd":
            out_info["ret_cd_step"] = float(ret_cd_4v_components["ret_cd_step"])
            out_info["ret_cd_components"] = ret_cd_4v_components
            out_info["ret_cd_fill_rate_step"] = float(
                ret_cd_4v_components["fill_rate"]
            )
            out_info["ret_cd_inverse_backlog_step"] = float(
                ret_cd_4v_components["inverse_backlog"]
            )
            out_info["ret_cd_spare_capacity_step"] = float(
                ret_cd_4v_components["spare_capacity"]
            )
            out_info["ret_cd_inverse_cost_step"] = float(
                ret_cd_4v_components["inverse_cost"]
            )
            out_info["ret_cd_kappa"] = float(self.ret_cd_kappa)
            out_info["ret_thesis_corrected_step"] = float(
                corrected_ret_components["ret_value"]
            )
            out_info["ret_thesis_corrected"] = corrected_ret_components
            out_info["ret_components"] = ret_components
        elif self._canonical_reward_mode in ("ReT_cd_v1", "ReT_cd_sigmoid"):
            key = (
                "ret_cd_step"
                if self._canonical_reward_mode == "ReT_cd_v1"
                else "ret_cd_sigmoid_step"
            )
            out_info["ret_cd_step"] = float(ret_cd_components[key])
            out_info["ret_cd_components"] = ret_cd_components
            out_info["ret_cd_fill_rate_step"] = float(
                ret_cd_components["fill_rate_step"]
            )
            out_info["ret_cd_availability_step"] = float(
                ret_cd_components["availability_step"]
            )
            out_info["ret_thesis_corrected_step"] = float(
                corrected_ret_components["ret_value"]
            )
            out_info["ret_thesis_corrected"] = corrected_ret_components
            out_info["ret_components"] = ret_components

        return out_obs, float(reward), bool(terminated), bool(truncated), out_info
