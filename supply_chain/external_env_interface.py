from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from supply_chain.config import DEFAULT_YEAR_BASIS, OPERATIONS, WARMUP
from supply_chain.env_experimental_shifts import MFSCGymEnvShifts

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
OBSERVATION_FIELDS: tuple[str, ...] = OBSERVATION_FIELDS_V1

ACTION_FIELDS: tuple[str, ...] = (
    "op3_q_multiplier_signal",
    "op9_q_multiplier_signal",
    "op3_rop_multiplier_signal",
    "op9_rop_multiplier_signal",
    "assembly_shift_signal",
)

ACTION_BOUNDS: tuple[tuple[float, float], ...] = (
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
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
)
REWARD_TERM_FIELDS: tuple[str, ...] = (
    "reward_total",
    "service_loss_step",
    "shift_cost_step",
    "disruption_fraction_step",
    "ret_thesis_corrected_step",
)


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
    raise ValueError(
        f"Invalid observation_version={observation_version!r}. Expected 'v1' or 'v2'."
    )


def get_shift_control_env_spec(
    *,
    reward_mode: str = "ReT_thesis",
    observation_version: str = "v1",
    step_size_hours: float = 168.0,
) -> ExternalEnvSpec:
    """Return the stable external contract for the thesis-aligned env."""
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
            "observation_version=v2 adds previous-step demand, backorder, and disruption diagnostics to the observed state.",
            "The fifth action dimension selects assembly capacity through discrete shifts.",
            "Reward mode ReT_thesis emits ret_components inside info for downstream auditing.",
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
    ]
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
        ],
        dtype=np.float32,
    )


def make_shift_control_env(**overrides: Any) -> MFSCGymEnvShifts:
    """Build the recommended thesis-aligned environment for external models."""
    params: dict[str, Any] = {
        "reward_mode": "ReT_thesis",
        "observation_version": "v1",
        "step_size_hours": 168.0,
        "year_basis": DEFAULT_YEAR_BASIS,
    }
    params.update(overrides)
    return MFSCGymEnvShifts(**params)
