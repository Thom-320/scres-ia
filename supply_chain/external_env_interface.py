from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from supply_chain.config import DEFAULT_YEAR_BASIS, OPERATIONS, WARMUP
from supply_chain.env_experimental_shifts import MFSCGymEnvShifts

OBSERVATION_FIELDS: tuple[str, ...] = (
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


@dataclass(frozen=True)
class ExternalEnvSpec:
    """Machine-readable contract for external models consuming the repo env."""

    env_variant: str
    reward_mode: str
    step_size_hours: float
    warmup_hours: float
    observation_fields: tuple[str, ...]
    action_fields: tuple[str, ...]
    action_bounds: tuple[tuple[float, float], ...]
    shift_mapping: dict[str, int]
    notes: tuple[str, ...]


def get_shift_control_env_spec(
    *,
    reward_mode: str = "ReT_thesis",
    step_size_hours: float = 168.0,
) -> ExternalEnvSpec:
    """Return the stable external contract for the thesis-aligned env."""
    return ExternalEnvSpec(
        env_variant="shift_control",
        reward_mode=reward_mode,
        step_size_hours=float(step_size_hours),
        warmup_hours=float(WARMUP["estimated_deterministic_hrs"]),
        observation_fields=OBSERVATION_FIELDS,
        action_fields=ACTION_FIELDS,
        action_bounds=ACTION_BOUNDS,
        shift_mapping={
            "signal_lt_-0.33": 1,
            "signal_ge_-0.33_and_lt_0.33": 2,
            "signal_ge_0.33": 3,
        },
        notes=(
            "Observation values are normalized continuous features emitted by MFSCSimulation.get_observation().",
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
    inside the 15-dimensional observation vector. External models that need
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


def make_shift_control_env(**overrides: Any) -> MFSCGymEnvShifts:
    """Build the recommended thesis-aligned environment for external models."""
    params: dict[str, Any] = {
        "reward_mode": "ReT_thesis",
        "step_size_hours": 168.0,
        "year_basis": DEFAULT_YEAR_BASIS,
    }
    params.update(overrides)
    return MFSCGymEnvShifts(**params)
