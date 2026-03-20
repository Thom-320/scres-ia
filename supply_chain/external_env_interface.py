from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol

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
    if observation_version == "v3":
        return OBSERVATION_FIELDS_V3
    if observation_version == "v4":
        return OBSERVATION_FIELDS_V4
    raise ValueError(
        f"Invalid observation_version={observation_version!r}. "
        "Expected 'v1', 'v2', 'v3', or 'v4'."
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
            "observation_version=v3 extends v2 with normalized cumulative backorder and disruption history since the end of warmup.",
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


# ---------------------------------------------------------------------------
# Generic episode runner for any callable policy
# ---------------------------------------------------------------------------


class PolicyCallable(Protocol):
    """Any callable that maps (obs, info) -> action array."""

    def __call__(self, obs: np.ndarray, info: dict[str, Any]) -> np.ndarray: ...


def run_episodes(
    policy_fn: PolicyCallable | Callable[[np.ndarray, dict[str, Any]], np.ndarray],
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
        Callable ``(obs, info) -> action``.  ``obs`` is an np.ndarray matching
        the env observation space; ``info`` is the dict returned by
        ``env.reset()`` or ``env.step()``.  Must return a 5-dim action array
        in [-1, 1].
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
        ``pct_steps_S1/S2/S3``, and optionally ``trajectory``.

    Example
    -------
    >>> from supply_chain.external_env_interface import run_episodes
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
        demanded_total = 0.0
        delivered_total = 0.0
        backorder_qty_total = 0.0
        steps = 0
        shift_counts = {1: 0, 2: 0, 3: 0}
        trajectory: list[dict[str, Any]] = []

        while not (terminated or truncated):
            action = np.asarray(policy_fn(obs, info), dtype=np.float32)
            prev_obs = obs

            obs, reward, terminated, truncated, info = env.step(action)

            reward_total += float(reward)
            service_loss_total += float(info.get("service_loss_step", 0.0))
            shift_cost_total += float(info.get("shift_cost_step", 0.0))
            disruption_fraction_total += float(
                info.get("disruption_fraction_step", 0.0)
            )
            ret_thesis_corrected_total += float(
                info.get("ret_thesis_corrected_step", 0.0)
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
                        "action": action.copy(),
                        "reward": float(reward),
                        "info": {
                            k: v
                            for k, v in info.items()
                            if isinstance(v, (int, float, str, bool))
                        },
                    }
                )

        env.close()

        total_steps = max(1, steps)
        if demanded_total > 0:
            backorder_rate = backorder_qty_total / demanded_total
            fill_rate = 1.0 - backorder_rate
        else:
            backorder_rate = 0.0
            fill_rate = 1.0

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
            "ret_thesis_corrected_total": ret_thesis_corrected_total,
            "demanded_total": demanded_total,
            "delivered_total": delivered_total,
            "backorder_qty_total": backorder_qty_total,
            "pct_steps_S1": 100.0 * shift_counts.get(1, 0) / total_steps,
            "pct_steps_S2": 100.0 * shift_counts.get(2, 0) / total_steps,
            "pct_steps_S3": 100.0 * shift_counts.get(3, 0) / total_steps,
        }
        if collect_trajectories:
            row["trajectory"] = trajectory
        results.append(row)

    return results
