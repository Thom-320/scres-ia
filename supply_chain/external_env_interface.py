from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol

import numpy as np

from .config import (
    BENCHMARK_OBSERVATION_VERSION,
    BENCHMARK_REWARD_MODE,
    BENCHMARK_W_BO,
    BENCHMARK_W_COST,
    BENCHMARK_W_DISR,
    DEFAULT_YEAR_BASIS,
    OPERATIONS,
    THESIS_FAITHFUL_PROTOCOL,
    TRACK_A_TRAINING_RAW_MATERIAL_FLOW_MODE,
    TRACK_A_TRAINING_RAW_MATERIAL_ORDER_UP_TO_MULTIPLIER,
    TRACK_A_TRAINING_RISK_OCCURRENCE_MODE,
    TRACK_A_TRAINING_DOWNSTREAM_Q_SOURCE,
    WARMUP,
)
from .env_experimental_shifts import MFSCGymEnvShifts

OBSERVATION_FIELDS_V1: tuple[str, ...] = (
    "raw_material_wdc_norm",
    "raw_material_al_norm",
    "rations_al_norm",
    "rations_sb_norm",
    "rations_cssu_norm",
    "rations_theatre_norm",
    "fill_rate",
    "backorder_rate",
    "assembly_line_down",
    "any_location_down",
    "op9_down",
    "op11_down",
    "time_fraction",
    "pending_batch_fraction",
    "contingent_demand_fraction",
)
OBSERVATION_FIELDS_V2: tuple[str, ...] = OBSERVATION_FIELDS_V1 + (
    "prev_step_demand_norm",
    "prev_step_backorder_qty_norm",
    "prev_step_disruption_hours_norm",
)
OBSERVATION_FIELDS_V3: tuple[str, ...] = OBSERVATION_FIELDS_V2 + (
    "cum_backorder_rate",
    "cum_downhours_fraction",
)
OBSERVATION_FIELDS_V4: tuple[str, ...] = OBSERVATION_FIELDS_V3 + (
    "rations_sb_dispatch_norm",
    "assembly_shifts_active_norm",
    "op1_down",
    "op2_down",
)
OBSERVATION_FIELDS_V5: tuple[str, ...] = OBSERVATION_FIELDS_V4 + (
    "op1_cycle_phase_norm",
    "op2_cycle_phase_norm",
    "workweek_phase_sin_norm",
    "workweek_phase_cos_norm",
    "workday_phase_sin_norm",
    "workday_phase_cos_norm",
)
OBSERVATION_FIELDS_V6: tuple[str, ...] = OBSERVATION_FIELDS_V5 + (
    "regime_nominal",
    "regime_strained",
    "regime_pre_disruption",
    "regime_disrupted",
    "regime_recovery",
    "risk_forecast_48h_norm",
    "risk_forecast_168h_norm",
    "maintenance_debt_norm",
    "backlog_age_norm",
    "theatre_cover_days_norm",
)
OBSERVATION_FIELDS_V7: tuple[str, ...] = OBSERVATION_FIELDS_V6 + (
    "op10_down",
    "op12_down",
    "op10_queue_pressure_norm",
    "op12_queue_pressure_norm",
    "rolling_fill_rate_4w",
    "rolling_backorder_rate_4w",
)
REALIZED_RISK_OBSERVATION_IDS: tuple[str, ...] = (
    "R11",
    "R12",
    "R13",
    "R14",
    "R21",
    "R22",
    "R23",
    "R24",
    "R3",
)
OBSERVATION_FIELDS_V8: tuple[str, ...] = OBSERVATION_FIELDS_V7 + tuple(
    f"active_{risk_id.lower()}" for risk_id in REALIZED_RISK_OBSERVATION_IDS
) + tuple(
    f"recent_{risk_id.lower()}" for risk_id in REALIZED_RISK_OBSERVATION_IDS
) + tuple(
    f"recent_{risk_id.lower()}_duration_norm"
    for risk_id in REALIZED_RISK_OBSERVATION_IDS
)
OBSERVATION_FIELDS: tuple[str, ...] = OBSERVATION_FIELDS_V1

# Track A continuous-control contract. This is a trainable RL extension, not the
# strict Garrido decision contract. The thesis-faithful discrete contract is
# `thesis_factorized`: common I_{t,S} level plus S.
ACTION_FIELDS: tuple[str, ...] = (
    "op3_q_multiplier_signal",      # a1 — I_{t,S} on Op3,j (Table 6.16)
    "op9_q_multiplier_signal",      # a2 — I_{t,S} on Op9,j (Table 6.16)
    "op3_rop_multiplier_signal",    # a3 — repo extension (not in thesis)
    "op9_rop_multiplier_signal",    # a4 — repo extension (not in thesis)
    "op5_q_multiplier_signal",      # a5 — I_{t,S} on Op5,j (Table 6.16; added in v6D)
    "assembly_shift_signal",        # a6 — S ∈ {1,2,3} (Table 6.20)
)
ACTION_FIELDS_TRACK_B_V1: tuple[str, ...] = ACTION_FIELDS + (
    "op10_q_multiplier_signal",
    "op12_q_multiplier_signal",
)
THESIS_INVENTORY_PERIODS: tuple[int, ...] = (168, 336, 504, 672, 1344)
THESIS_INVENTORY_NODES: tuple[str, ...] = ("op3", "op5", "op9")
THESIS_DECISION_ACTION_FIELDS: tuple[str, ...] = tuple(
    f"{node}_I{period}_1"
    for node in THESIS_INVENTORY_NODES
    for period in THESIS_INVENTORY_PERIODS
) + ("S1", "S2", "S3")
THESIS_DECISION_OBSERVATION_FIELDS: tuple[str, ...] = THESIS_DECISION_ACTION_FIELDS + (
    "reward",
)
THESIS_FACTORIZED_ACTION_FIELDS: tuple[str, ...] = (
    "common_inventory_period_level",
    "assembly_shift_level",
)

ACTION_BOUNDS: tuple[tuple[float, float], ...] = (
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
)
ACTION_BOUNDS_TRACK_B_V1: tuple[tuple[float, float], ...] = ACTION_BOUNDS + (
    (-1.0, 1.0),
    (-1.0, 1.0),
)
INVENTORY_NODE_FIELDS: tuple[str, ...] = (
    "raw_material_wdc",
    "raw_material_al",
    "rations_al",
    "rations_sb",
    "rations_sb_dispatch",
    "rations_cssu",
    "rations_theatre",
)
DKANA_BACKORDER_VECTOR_FIELDS: tuple[str, ...] = tuple(
    f"cum_backorder_rate_{field_name}" for field_name in INVENTORY_NODE_FIELDS
)
DKANA_DISRUPTION_VECTOR_FIELDS: tuple[str, ...] = tuple(
    f"cum_disruption_fraction_op{op_id}" for op_id in range(1, 14)
)
CONTROL_CONTEXT_FIELDS: tuple[str, ...] = (
    "op3_q",
    "op3_rop",
    "op9_q_min",
    "op9_q_max",
    "op9_rop",
    "inventory_multiplier_min",
    "inventory_multiplier_max",
    "shift_signal_threshold_low",
    "shift_signal_threshold_high",
)
STATE_CONSTRAINT_FIELDS: tuple[str, ...] = (
    (
        "raw_material_wdc",
        "raw_material_al",
        "rations_al",
        "rations_sb",
        "rations_sb_dispatch",
        "rations_cssu",
        "rations_theatre",
        "total_inventory",
        "op3_total_dispatch_cap",
        "op3_per_material_dispatch_cap",
        "op9_dispatch_cap",
        "assembly_line_available",
        "any_location_available",
        "op9_available",
        "op11_available",
        "fill_rate",
        "backorder_rate",
        "time_fraction",
        "pending_batch_fraction",
        "contingent_demand_fraction",
        "cumulative_backorder_qty",
        "cumulative_disruption_hours",
        "pending_backorders_count",
        "pending_backorder_qty",
        "unattended_orders_total",
    )
    + DKANA_BACKORDER_VECTOR_FIELDS
    + DKANA_DISRUPTION_VECTOR_FIELDS
)
SDM_HISTORY_FIELDS: tuple[str, ...] = (
    "sdm_recent_order_count",
    "sdm_completed_order_count",
    "sdm_backorder_order_count",
    "sdm_lost_order_count",
    "sdm_recent_demand_qty",
    "sdm_recent_remaining_qty",
    "sdm_mean_ct_hours",
    "sdm_max_ct_hours",
    "sdm_sum_ap_hours",
    "sdm_sum_rp_hours",
    "sdm_sum_dp_hours",
    "sdm_ret_case_fill_rate_count",
    "sdm_ret_case_autotomy_count",
    "sdm_ret_case_recovery_count",
    "sdm_ret_case_non_recovery_count",
    "sdm_r1_event_count",
    "sdm_r2_event_count",
    "sdm_r3_event_count",
    "sdm_r1_duration_hours",
    "sdm_r2_duration_hours",
    "sdm_r3_duration_hours",
    "sdm_risk_affected_ops_count",
)
REWARD_TERM_FIELDS: tuple[str, ...] = (
    "reward_total",
    "service_loss_step",
    "shift_cost_step",
    "disruption_fraction_step",
    "ret_thesis_corrected_step",
    "ret_unified_step",
    "ret_unified_fr",
    "ret_unified_rc",
    "ret_unified_ce",
    "ret_unified_gate",
    "ret_garrido2024_sigmoid_step",
)


def get_episode_terminal_metrics(env: Any) -> dict[str, float]:
    """Return terminal service metrics from the underlying DES-backed env.

    The benchmark and external helpers previously reconstructed fill rate from
    step-level `new_backorder_qty/new_demanded` flows. That quantity-based
    aggregate is useful for auditing transition flow, but it is not the same as
    the thesis/order-level service metric already exposed by the simulator state.
    This helper centralizes the paper-facing terminal metrics so all evaluation
    paths use the same definitions.
    """
    base_env = getattr(env, "unwrapped", env)
    sim = getattr(base_env, "sim", None)
    if sim is None:
        return {
            "fill_rate_order_level": float("nan"),
            "backorder_rate_order_level": float("nan"),
            "fill_rate_state_terminal": float("nan"),
            "backorder_rate_state_terminal": float("nan"),
            "order_level_ret_mean": float("nan"),
            "order_level_ret_excel_formula_mean": float("nan"),
            "order_level_ret_text_formula_mean": float("nan"),
        }

    order_summary: dict[str, Any] = {}
    if hasattr(sim, "compute_order_level_ret"):
        order_summary = sim.compute_order_level_ret()
    fill_rate_order_level = float(
        order_summary.get("fill_rate_order_level", getattr(sim, "_fill_rate")())
    )
    backorder_rate_order_level = float(max(0.0, 1.0 - fill_rate_order_level))
    fill_rate_state_terminal = (
        float(sim._fill_rate()) if hasattr(sim, "_fill_rate") else fill_rate_order_level
    )
    backorder_rate_state_terminal = (
        float(sim._backorder_rate())
        if hasattr(sim, "_backorder_rate")
        else backorder_rate_order_level
    )
    return {
        "fill_rate_order_level": fill_rate_order_level,
        "backorder_rate_order_level": backorder_rate_order_level,
        "fill_rate_state_terminal": fill_rate_state_terminal,
        "backorder_rate_state_terminal": backorder_rate_state_terminal,
        "order_level_ret_mean": float(
            order_summary.get("mean_ret", fill_rate_order_level)
        ),
        "order_level_ret_excel_formula_mean": float(
            order_summary.get("mean_ret_excel_formula", fill_rate_order_level)
        ),
        "order_level_ret_text_formula_mean": float(
            order_summary.get("mean_ret_text_formula", fill_rate_order_level)
        ),
    }


@dataclass(frozen=True)
class ExternalEnvSpec:
    """Machine-readable contract for external models consuming the repo env."""

    env_variant: str
    reward_mode: str
    observation_version: str
    step_size_hours: float
    warmup_hours: float
    observation_fields: tuple[str, ...]
    action_fields: tuple[str, ...]
    action_bounds: tuple[tuple[float, float], ...]
    shift_mapping: dict[str, int]
    notes: tuple[str, ...]


def get_observation_fields(observation_version: str = "v1") -> tuple[str, ...]:
    """Return the observation schema for the requested environment contract version."""
    if observation_version == "v1":
        return OBSERVATION_FIELDS_V1
    if observation_version == "v2":
        return OBSERVATION_FIELDS_V2
    if observation_version == "v3":
        return OBSERVATION_FIELDS_V3
    if observation_version == "v4":
        return OBSERVATION_FIELDS_V4
    if observation_version == "v5":
        return OBSERVATION_FIELDS_V5
    if observation_version == "v6":
        return OBSERVATION_FIELDS_V6
    if observation_version == "v7":
        return OBSERVATION_FIELDS_V7
    if observation_version == "v8":
        return OBSERVATION_FIELDS_V8
    raise ValueError(
        f"Invalid observation_version={observation_version!r}. "
        "Expected 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', or 'v8'."
    )


def get_shift_control_env_spec(
    *,
    reward_mode: str = BENCHMARK_REWARD_MODE,
    observation_version: str = BENCHMARK_OBSERVATION_VERSION,
    step_size_hours: float = 168.0,
) -> ExternalEnvSpec:
    """Return the stable external contract for the benchmark shift env."""
    observation_fields = get_observation_fields(observation_version)
    return ExternalEnvSpec(
        env_variant="shift_control",
        reward_mode=reward_mode,
        observation_version=observation_version,
        step_size_hours=float(step_size_hours),
        warmup_hours=float(WARMUP["estimated_deterministic_hrs"]),
        observation_fields=observation_fields,
        action_fields=ACTION_FIELDS,
        action_bounds=ACTION_BOUNDS,
        shift_mapping={
            "signal_lt_-0.33": 1,
            "signal_ge_-0.33_and_lt_0.33": 2,
            "signal_ge_0.33": 3,
        },
        notes=(
            "Observation values are normalized continuous features emitted by the shift-control environment.",
            "The default external spec freezes reward_mode=control_v1 and observation_version=v4 for the current Track A paper benchmark contract.",
            "observation_version=v2 adds previous-step demand, backorder, and disruption diagnostics to the observed state.",
            "observation_version=v3 extends v2 with normalized cumulative backorder and disruption history since the end of warmup.",
            "observation_version=v4 extends v3 with current shift plus Op1/Op2 disruption flags and is the frozen online contract for Track A and DKANA handoff.",
            "observation_version=v5 extends v4 with thesis-faithful cycle/calendar precursor features and is intended only for research comparisons, not the frozen Track A contract.",
            "observation_version=v6 extends v5 with Track-B adaptive benchmark features: regime state, imperfect disruption forecasts, maintenance debt, backlog age, and theatre cover days.",
            "The fifth action dimension selects assembly capacity through discrete shifts.",
            "control_v1 is the frozen Track A training reward; ret_thesis_corrected remains the thesis-aligned audit metric.",
            "ReT_unified_v1 remains available as a service-first resilience reward for audit and exploratory research lanes.",
            "risk_level=adaptive_benchmark_v1 activates the Track-B research lane with persistent regimes and deferred S3 maintenance costs; it does not change the Track A defaults.",
            "ReT_garrido2024_raw is the paper-faithful five-variable Cobb-Douglas raw product (Eq. 3), intended only as a training-reward candidate.",
            "ReT_garrido2024 is the paper-faithful five-variable sigmoid index (Eq. 6), intended as the evaluation/audit index rather than the main PPO reward.",
            "ReT_cd_v1 is the continuous Cobb-Douglas bridge for the piecewise ReT_thesis (recommended over ReT_thesis for training).",
            "ReT_cd_sigmoid is an experimental variant documented to show sigmoid bias when log-inputs are already in (0,1] — NOT recommended.",
        ),
    )


def get_track_b_env_spec(
    *,
    reward_mode: str = "ReT_seq_v1",
    observation_version: str = "v7",
    step_size_hours: float = 168.0,
) -> ExternalEnvSpec:
    """Return the minimal Track B research contract with downstream control."""
    observation_fields = get_observation_fields(observation_version)
    return ExternalEnvSpec(
        env_variant="track_b_adaptive_control",
        reward_mode=reward_mode,
        observation_version=observation_version,
        step_size_hours=float(step_size_hours),
        warmup_hours=float(WARMUP["estimated_deterministic_hrs"]),
        observation_fields=observation_fields,
        action_fields=ACTION_FIELDS_TRACK_B_V1,
        action_bounds=ACTION_BOUNDS_TRACK_B_V1,
        shift_mapping={
            "signal_lt_-0.33": 1,
            "signal_ge_-0.33_and_lt_0.33": 2,
            "signal_ge_0.33": 3,
        },
        notes=(
            "Track B keeps the thesis-faithful DES structure but exposes downstream transport control at Op10 and Op12.",
            "observation_version=v7 extends v6 with downstream disruption state, queue pressure, and rolling 4-week service metrics.",
            "The seventh action contract uses 7 dimensions: the Track A 5D controls plus Op10 and Op12 dispatch quantity multipliers.",
            "risk_level=adaptive_benchmark_v2 is the intended Track B stress profile with stronger downstream transport and demand pressure.",
            "This contract is research-only and must not replace the frozen Track A paper-facing benchmark.",
        ),
    )


def get_thesis_aligned_training_env_spec(
    *,
    reward_mode: str = "ReT_seq_v1",
    observation_version: str = "v4",
    step_size_hours: float = 168.0,
) -> ExternalEnvSpec:
    """Return the trainable Gym contract after the thesis_1to1 validation gate."""
    observation_fields = get_observation_fields(observation_version)
    return ExternalEnvSpec(
        env_variant="thesis_aligned_training",
        reward_mode=reward_mode,
        observation_version=observation_version,
        step_size_hours=float(step_size_hours),
        warmup_hours=float(WARMUP["estimated_deterministic_hrs"]),
        observation_fields=observation_fields,
        action_fields=ACTION_FIELDS,
        action_bounds=ACTION_BOUNDS,
        shift_mapping={
            "signal_lt_-0.33": 1,
            "signal_ge_-0.33_and_lt_0.33": 2,
            "signal_ge_0.33": 3,
        },
        notes=(
            "This is a Gym training lane, not the strict thesis_1to1 reproduction lane.",
            f"It inherits the thesis validation knobs: year_basis=thesis, warmup_trigger=op9_arrival, downstream_q_source={TRACK_A_TRAINING_DOWNSTREAM_Q_SOURCE}, r14_defect_mode=thesis_strict_op6, raw_material_flow_mode={TRACK_A_TRAINING_RAW_MATERIAL_FLOW_MODE}, risk_occurrence_mode={TRACK_A_TRAINING_RISK_OCCURRENCE_MODE}.",
            "By default it disables post-warmup priming so episodes begin at the actual thesis warm-up trigger.",
            "Actions remain an RL extension: continuous inventory/ROP multipliers plus a discrete shift selector.",
            "Use direct DES action dictionaries for static Garrido baselines when comparing S1/S2/S3 without multiplier artifacts.",
        ),
    )


def get_dkana_thesis_faithful_env_spec(
    *,
    reward_mode: str = "ReT_seq_v1",
    step_size_hours: float = 168.0,
    observation_version: str = "v5",
    observation_mode: str = "decision_reward",
    action_space_mode: str = "onehot_18d",
) -> ExternalEnvSpec:
    """Return the DKANA thesis-decision contract for the requested action surface."""
    if observation_mode == "decision_reward":
        observation_fields = THESIS_DECISION_OBSERVATION_FIELDS
        observation_contract = "thesis_decision_reward_v1"
    elif observation_mode == "env_reward":
        observation_fields = get_observation_fields(observation_version) + ("reward",)
        observation_contract = f"env_reward_{observation_version}"
    elif observation_mode == "env_state_reward":
        observation_fields = (
            get_observation_fields(observation_version)
            + STATE_CONSTRAINT_FIELDS
            + ("reward",)
        )
        observation_contract = f"env_state_reward_{observation_version}"
    elif observation_mode == "env_sdm_history_reward":
        observation_fields = (
            get_observation_fields(observation_version)
            + SDM_HISTORY_FIELDS
            + ("reward",)
        )
        observation_contract = f"env_sdm_history_reward_{observation_version}"
    else:
        raise ValueError(
            "observation_mode must be 'decision_reward', 'env_reward', "
            "'env_state_reward', or 'env_sdm_history_reward'."
        )
    if action_space_mode not in ("onehot_18d", "factorized", "thesis_factorized"):
        raise ValueError(
            "action_space_mode must be 'onehot_18d', 'factorized', "
            "or 'thesis_factorized'."
        )
    if action_space_mode == "thesis_factorized":
        action_fields = THESIS_FACTORIZED_ACTION_FIELDS
        action_bounds = ((0.0, 5.0), (0.0, 2.0))
        action_notes = (
            "Thesis-decision DKANA adapter: common I_t,S level from Table 6.16 "
            "plus S from Table 6.20.",
            "The common inventory level maps to Op3, Op5, and Op9 buffer "
            "quantities using the thesis Table 6.16 rows; level 0 means no "
            "strategic inventory buffer.",
            "This is the action surface to use when the paper claims the "
            "agent controls the same decision variables as Garrido-Rios.",
        )
    else:
        action_fields = THESIS_DECISION_ACTION_FIELDS
        action_bounds = ((0.0, 1.0),) * len(THESIS_DECISION_ACTION_FIELDS)
        action_notes = (
            "DKANA adapter: 15 inventory-buffer dimensions from Table 6.16 "
            "plus 3 capacity dimensions from Table 6.20.",
            "Inventory dimensions are grouped by period I168,1, I336,1, "
            "I504,1, I672,1, I1344,1 across Op3, Op5, and Op9.",
            "The onehot_18d surface is a compatibility/export representation; "
            "factorized is a categorical per-node extension unless collapsed "
            "with inventory_period_mode=thesis_strict.",
        )
    return ExternalEnvSpec(
        env_variant="dkana_thesis_faithful_decision",
        reward_mode=reward_mode,
        observation_version=observation_contract,
        step_size_hours=float(step_size_hours),
        warmup_hours=float(WARMUP["estimated_deterministic_hrs"]),
        observation_fields=observation_fields,
        action_fields=action_fields,
        action_bounds=action_bounds,
        shift_mapping={
            "S1": 1,
            "S2": 2,
            "S3": 3,
        },
        notes=(
            *action_notes,
            "Observation is either the realized 18D decision handoff plus "
            "reward, or a richer environment/history surface plus reward.",
            f"action_space_mode={action_space_mode}: onehot_18d exports the raw "
            "18D thesis vector; thesis_factorized trains the two thesis "
            "decision variables (common I_t,S level and S) directly; "
            "factorized trains categorical decisions over Op3, Op5, Op9, "
            "and S, then records the equivalent 18D vector.",
            f"observation_mode={observation_mode}: decision_reward keeps the "
            "19D David handoff; env_reward/env_state_reward/env_sdm_history_reward "
            "are PPO research surfaces that keep actions fixed while enriching "
            "observability.",
            "This contract stays on the thesis-aligned Track A backbone and "
            "does not expose Track B Op10/Op12 downstream controls.",
        ),
    )


def spec_to_dict(spec: ExternalEnvSpec) -> dict[str, Any]:
    """Serialize the environment contract for JSON export or external tooling."""
    return asdict(spec)


def get_shift_control_constraint_context() -> dict[str, Any]:
    """
    Return non-observational control context for external models.

    These values are enforced by the environment/config rather than encoded
    inside the v1 15-dimensional observation vector. External models that need
    explicit constraints can consume this block alongside trajectories.
    """
    return {
        "action_bounds": ACTION_BOUNDS,
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
        "notes": (
            "Constraints live in config/environment dynamics, not in the 15-d observation vector.",
            "For PPO this is sufficient because the environment clips and maps actions.",
            "External models such as DKANA can consume this block as explicit context.",
        ),
    }


def build_shift_control_constraint_vector(
    constraint_context: dict[str, Any],
) -> np.ndarray:
    """Serialize fixed action constraints into a stable numeric vector."""
    base_parameters = constraint_context["base_control_parameters"]
    inventory_range = constraint_context["inventory_multiplier_range"]
    shift_bands = constraint_context["shift_signal_bands"]
    return np.array(
        [
            float(base_parameters["op3_q"]),
            float(base_parameters["op3_rop"]),
            float(base_parameters["op9_q_min"]),
            float(base_parameters["op9_q_max"]),
            float(base_parameters["op9_rop"]),
            float(inventory_range["min"]),
            float(inventory_range["max"]),
            -0.33 if "signal_lt_-0.33" in shift_bands else float("nan"),
            0.33 if "signal_ge_0.33" in shift_bands else float("nan"),
        ],
        dtype=np.float32,
    )


def build_shift_control_state_constraint_vector(
    state_context: dict[str, Any],
) -> np.ndarray:
    """Serialize live state constraints into a stable numeric vector."""
    inventory_detail = state_context["inventory_detail"]
    assert isinstance(inventory_detail, dict)
    values = [
        float(inventory_detail["raw_material_wdc"]),
        float(inventory_detail["raw_material_al"]),
        float(inventory_detail["rations_al"]),
        float(inventory_detail["rations_sb"]),
        float(inventory_detail["rations_sb_dispatch"]),
        float(inventory_detail["rations_cssu"]),
        float(inventory_detail["rations_theatre"]),
        float(state_context["total_inventory"]),
        float(state_context["op3_total_dispatch_cap"]),
        float(state_context["op3_per_material_dispatch_cap"]),
        float(state_context["op9_dispatch_cap"]),
        float(bool(state_context["assembly_line_available"])),
        float(bool(state_context["any_location_available"])),
        float(bool(state_context["op9_available"])),
        float(bool(state_context["op11_available"])),
        float(state_context["fill_rate"]),
        float(state_context["backorder_rate"]),
        float(state_context["time_fraction"]),
        float(state_context["pending_batch_fraction"]),
        float(state_context["contingent_demand_fraction"]),
        float(state_context["cumulative_backorder_qty"]),
        float(state_context["cumulative_disruption_hours"]),
        float(state_context["pending_backorders_count"]),
        float(state_context["pending_backorder_qty"]),
        float(state_context["unattended_orders_total"]),
    ]
    backorder_vector = state_context["cumulative_backorder_rate_by_inventory_node"]
    disruption_vector = state_context["cumulative_disruption_fraction_by_operation"]
    assert isinstance(backorder_vector, dict)
    assert isinstance(disruption_vector, dict)
    values.extend(
        float(backorder_vector[field_name]) for field_name in INVENTORY_NODE_FIELDS
    )
    values.extend(float(disruption_vector[f"op{op_id}"]) for op_id in range(1, 14))
    return np.array(values, dtype=np.float32)


def build_reward_term_vector(info: dict[str, Any], reward: float) -> np.ndarray:
    """Serialize step reward diagnostics into a stable numeric vector."""
    return np.array(
        [
            float(reward),
            float(info.get("service_loss_step", 0.0)),
            float(info.get("shift_cost_step", 0.0)),
            float(info.get("disruption_fraction_step", 0.0)),
            float(info.get("ret_thesis_corrected_step", 0.0)),
            float(info.get("ret_unified_step", 0.0)),
            float(info.get("ret_unified_fr", 0.0)),
            float(info.get("ret_unified_rc", 0.0)),
            float(info.get("ret_unified_ce", 0.0)),
            float(info.get("ret_unified_gate", 0.0)),
            float(info.get("ret_garrido2024_sigmoid_step", 0.0)),
        ],
        dtype=np.float32,
    )


def make_shift_control_env(**overrides: Any) -> MFSCGymEnvShifts:
    """Build the recommended benchmark environment for external models."""
    params: dict[str, Any] = {
        "reward_mode": BENCHMARK_REWARD_MODE,
        "observation_version": BENCHMARK_OBSERVATION_VERSION,
        "step_size_hours": 168.0,
        "year_basis": DEFAULT_YEAR_BASIS,
        "w_bo": BENCHMARK_W_BO,
        "w_cost": BENCHMARK_W_COST,
        "w_disr": BENCHMARK_W_DISR,
    }
    params.update(overrides)
    return MFSCGymEnvShifts(**params)


def make_track_b_env(**overrides: Any) -> MFSCGymEnvShifts:
    """Build the minimal Track B research environment with downstream control."""
    params: dict[str, Any] = {
        "reward_mode": "ReT_seq_v1",
        "observation_version": "v7",
        "action_contract": "track_b_v1",
        "risk_level": "adaptive_benchmark_v2",
        "step_size_hours": 168.0,
        "year_basis": "thesis",
        "stochastic_pt": True,
        "w_bo": BENCHMARK_W_BO,
        "w_cost": BENCHMARK_W_COST,
        "w_disr": BENCHMARK_W_DISR,
    }
    params.update(overrides)
    return MFSCGymEnvShifts(**params)


def make_thesis_aligned_training_env(**overrides: Any) -> MFSCGymEnvShifts:
    """Build the paper-serious trainable env gated by thesis-faithful settings."""
    params: dict[str, Any] = {
        "reward_mode": "ReT_seq_v1",
        "observation_version": "v4",
        "step_size_hours": 168.0,
        "year_basis": THESIS_FAITHFUL_PROTOCOL["year_basis"],
        "risk_level": "current",
        "stochastic_pt": False,
        "warmup_trigger": THESIS_FAITHFUL_PROTOCOL["warmup_trigger"],
        "downstream_q_source": TRACK_A_TRAINING_DOWNSTREAM_Q_SOURCE,
        "r14_defect_mode": THESIS_FAITHFUL_PROTOCOL["r14_defect_mode"],
        "risk_occurrence_mode": TRACK_A_TRAINING_RISK_OCCURRENCE_MODE,
        "raw_material_flow_mode": TRACK_A_TRAINING_RAW_MATERIAL_FLOW_MODE,
        "raw_material_order_up_to_multiplier": (
            TRACK_A_TRAINING_RAW_MATERIAL_ORDER_UP_TO_MULTIPLIER
        ),
        "demand_on_hand_fulfillment_delay": (
            THESIS_FAITHFUL_PROTOCOL["demand_on_hand_fulfillment_delay"]
        ),
        "priming_enabled": False,
        "clear_backlog_after_priming": False,
        "w_bo": BENCHMARK_W_BO,
        "w_cost": BENCHMARK_W_COST,
        "w_disr": BENCHMARK_W_DISR,
    }
    params.update(overrides)
    return MFSCGymEnvShifts(**params)


def make_dkana_track_b_env(**overrides: Any) -> Any:
    """
    Build Track B with DKANA context windows included in env ``info``.

    This is the simplest entry point for external DKANA users: the returned
    env behaves like the normal Track B Gymnasium env, but each reset/step info
    includes ``dkana_row_matrices``, ``dkana_config_context``, and
    ``dkana_time_mask``.
    """
    from .dkana_env import make_dkana_track_b_env as _make_env

    return _make_env(**overrides)


def make_dkana_thesis_faithful_env(**overrides: Any) -> Any:
    """
    Build a thesis-faithful DKANA decision-vector env.

    This adapter exposes Garrido-Rios decision variables directly:
    15 inventory-buffer dimensions from Table 6.16 plus 3 capacity-shift
    dimensions from Table 6.20. Observations mirror the realized 18 decision
    dimensions and append the latest reward, yielding 19 dimensions.
    """
    from .dkana_env import make_dkana_thesis_faithful_env as _make_env

    return _make_env(**overrides)


def make_thesis_factorized_track_a_env(**overrides: Any) -> Any:
    """
    Build the torch-free Track A thesis-decision env.

    This exposes the Garrido-Rios action surface as ``MultiDiscrete([6, 3])``:
    common inventory buffer level plus S1/S2/S3. It is intended for reward
    audits and non-DKANA PPO-style experiments.
    """
    from .thesis_decision_env import make_thesis_factorized_track_a_env as _make_env

    return _make_env(**overrides)


def make_discrete18_track_a_env(**overrides: Any) -> Any:
    """
    Build the torch-free Track A thesis-decision env as ``Discrete(18)``.

    This is the DQN-style view over the same 6 x 3 thesis action surface used by
    ``make_thesis_factorized_track_a_env``.
    """
    from .thesis_decision_env import make_discrete18_track_a_env as _make_env

    return _make_env(**overrides)


def make_continuous_its_track_a_env(**overrides: Any) -> Any:
    """
    Build the Track A continuous I_t,S env as ``Box([0,-1], [1,1])``.

    This is the nearest continuous relaxation of Garrido-Rios' two decision
    variables: a common strategic buffer fraction and S1/S2/S3 capacity signal.
    It does not expose Track B downstream controls or per-node buffers.
    """
    from .continuous_its_env import make_continuous_its_track_a_env as _make_env

    return _make_env(**overrides)


def make_per_op_buffer_track_a_env(**overrides: Any) -> Any:
    """
    Build the Track A per-operation continuous buffer env as
    ``Box([op3_frac, op5_frac, op9_frac, shift_signal])``.

    This keeps Garrido-Rios' inventory-buffer and shift decision families, but
    does not force Op3, Op5 and Op9 to share one common buffer fraction.
    """
    from .continuous_its_env import make_per_op_buffer_track_a_env as _make_env

    return _make_env(**overrides)


# ---------------------------------------------------------------------------
# Generic episode runner for any callable policy
# ---------------------------------------------------------------------------


class PolicyCallable(Protocol):
    """Any callable that maps (obs, info) -> action payload."""

    def __call__(
        self, obs: np.ndarray, info: dict[str, Any]
    ) -> np.ndarray | dict[str, float | int]: ...


def run_episodes(
    policy_fn: (
        PolicyCallable
        | Callable[[np.ndarray, dict[str, Any]], np.ndarray | dict[str, float | int]]
    ),
    *,
    n_episodes: int = 10,
    seed: int = 42,
    env_kwargs: dict[str, Any] | None = None,
    policy_name: str = "custom",
    collect_trajectories: bool = False,
) -> list[dict[str, Any]]:
    """
    Run *n_episodes* using any policy callable and return per-episode metrics.

    This is the entry point for external models (DKANA, custom heuristics, etc.)
    that want to evaluate against the same MFSC environment used in the
    benchmark, without depending on the benchmark script internals.

    Parameters
    ----------
    policy_fn :
        Callable ``(obs, info) -> action``. ``obs`` is an np.ndarray matching
        the env observation space; ``info`` is the dict returned by
        ``env.reset()`` or ``env.step()``. It may return either a 5-dim action
        array in [-1, 1] or a direct DES action dict accepted by the env.
    n_episodes :
        Number of evaluation episodes.
    seed :
        Base random seed.  Episode *i* uses ``seed + i``.
    env_kwargs :
        Keyword arguments forwarded to ``make_shift_control_env()``.
        Use this to set ``reward_mode``, ``risk_level``, ``w_bo``, etc.
    policy_name :
        Label stored in the ``"policy"`` field of each result row.
    collect_trajectories :
        If ``True``, each result row includes ``"trajectory"`` — a list of
        per-step dicts with ``obs``, ``action``, ``reward``, ``info``.

    Returns
    -------
    list[dict]
        One dict per episode with keys: ``policy``, ``seed``, ``episode``,
        ``steps``, ``reward_total``, ``fill_rate``, ``backorder_rate``,
        ``service_loss_total``, ``shift_cost_total``, ``mean_disruption_fraction``,
        ``ret_unified_total``, ``ret_garrido2024_sigmoid_total``,
        ``pct_steps_S1/S2/S3``, and optionally ``trajectory``.

    Example
    -------
    >>> from scresia.supply_chain.external_env_interface import run_episodes
    >>> results = run_episodes(
    ...     lambda obs, info: np.zeros(5, dtype=np.float32),  # neutral policy
    ...     n_episodes=3,
    ...     seed=1,
    ...     env_kwargs={"reward_mode": "control_v1", "risk_level": "increased",
    ...                 "w_bo": 4.0, "w_cost": 0.02, "w_disr": 0.0},
    ... )
    >>> print(results[0]["reward_total"], results[0]["fill_rate"])
    """
    env_kwargs = dict(env_kwargs or {})
    results: list[dict[str, Any]] = []

    for ep_idx in range(n_episodes):
        ep_seed = seed + ep_idx
        env = make_shift_control_env(**env_kwargs)
        obs, info = env.reset(seed=ep_seed)

        terminated = False
        truncated = False
        reward_total = 0.0
        service_loss_total = 0.0
        shift_cost_total = 0.0
        disruption_fraction_total = 0.0
        ret_thesis_corrected_total = 0.0
        ret_unified_total = 0.0
        ret_garrido2024_sigmoid_total = 0.0
        demanded_total = 0.0
        delivered_total = 0.0
        backorder_qty_total = 0.0
        steps = 0
        shift_counts = {1: 0, 2: 0, 3: 0}
        trajectory: list[dict[str, Any]] = []

        while not (terminated or truncated):
            action_payload = policy_fn(obs, info)
            prev_obs = obs
            if isinstance(action_payload, dict):
                env_action: np.ndarray | dict[str, float | int] = dict(action_payload)
            else:
                env_action = np.asarray(action_payload, dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(env_action)

            reward_total += float(reward)
            service_loss_total += float(info.get("service_loss_step", 0.0))
            shift_cost_total += float(info.get("shift_cost_step", 0.0))
            disruption_fraction_total += float(
                info.get("disruption_fraction_step", 0.0)
            )
            ret_thesis_corrected_total += float(
                info.get("ret_thesis_corrected_step", 0.0)
            )
            ret_unified_total += float(info.get("ret_unified_step", 0.0))
            ret_garrido2024_sigmoid_total += float(
                info.get("ret_garrido2024_sigmoid_step", 0.0)
            )
            demanded_total += float(info.get("new_demanded", 0.0))
            delivered_total += float(info.get("new_delivered", 0.0))
            backorder_qty_total += float(info.get("new_backorder_qty", 0.0))
            shift_counts[int(info.get("shifts_active", 1))] += 1
            steps += 1

            if collect_trajectories:
                trajectory.append(
                    {
                        "obs": prev_obs.copy(),
                        "action": (
                            env_action.copy()
                            if isinstance(env_action, np.ndarray)
                            else dict(env_action)
                        ),
                        "reward": float(reward),
                        "info": {
                            k: v
                            for k, v in info.items()
                            if isinstance(v, (int, float, str, bool))
                        },
                    }
                )

        total_steps = max(1, steps)
        terminal_metrics = get_episode_terminal_metrics(env)
        if demanded_total > 0:
            flow_backorder_rate = backorder_qty_total / demanded_total
            flow_fill_rate = max(0.0, min(1.0, 1.0 - flow_backorder_rate))
        else:
            flow_backorder_rate = 0.0
            flow_fill_rate = 1.0
        fill_rate = float(terminal_metrics["fill_rate_order_level"])
        backorder_rate = float(terminal_metrics["backorder_rate_order_level"])

        row: dict[str, Any] = {
            "policy": policy_name,
            "seed": ep_seed,
            "episode": ep_idx + 1,
            "steps": steps,
            "reward_total": reward_total,
            "fill_rate": fill_rate,
            "backorder_rate": backorder_rate,
            "service_loss_total": service_loss_total,
            "shift_cost_total": shift_cost_total,
            "mean_disruption_fraction": disruption_fraction_total / total_steps,
            "ret_unified_total": ret_unified_total,
            "ret_garrido2024_sigmoid_total": ret_garrido2024_sigmoid_total,
            "ret_thesis_corrected_total": ret_thesis_corrected_total,
            "order_level_ret_mean": float(terminal_metrics["order_level_ret_mean"]),
            "order_level_ret_text_formula_mean": float(
                terminal_metrics["order_level_ret_text_formula_mean"]
            ),
            "order_level_ret_excel_formula_mean": float(
                terminal_metrics["order_level_ret_excel_formula_mean"]
            ),
            "demanded_total": demanded_total,
            "delivered_total": delivered_total,
            "backorder_qty_total": backorder_qty_total,
            "flow_fill_rate": flow_fill_rate,
            "flow_backorder_rate": flow_backorder_rate,
            "fill_rate_state_terminal": float(
                terminal_metrics["fill_rate_state_terminal"]
            ),
            "backorder_rate_state_terminal": float(
                terminal_metrics["backorder_rate_state_terminal"]
            ),
            "pct_steps_S1": 100.0 * shift_counts.get(1, 0) / total_steps,
            "pct_steps_S2": 100.0 * shift_counts.get(2, 0) / total_steps,
            "pct_steps_S3": 100.0 * shift_counts.get(3, 0) / total_steps,
        }
        if collect_trajectories:
            row["trajectory"] = trajectory
        results.append(row)
        env.close()

    return results
