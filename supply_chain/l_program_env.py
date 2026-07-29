"""Garrido-grounded retained-learning environment (``garrido_learning_v1``).

This module deliberately does not reuse the historical ``shift_only`` action
mode.  That mode expands a scalar signal into a six-dimensional Track-A action
and changes upstream quantities.  The contract below accepts a categorical
S1/S2/S3 request and changes only ``assembly_shifts`` (plus the simulator's
thesis-defined shift/batch coupling).

The environment is designed for cross-campaign identification.  Every reset
constructs a fresh DES; a campaign may retain neural state in an external
training harness, but it cannot retain inventory, backlog, WIP, RNG state, or
normalization statistics here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import (
    BACKORDER_QUEUE_CAP,
    HOURS_PER_WEEK,
    INVENTORY_BUFFERS,
    SIMULATION_HORIZON,
)
from .env_experimental_shifts import MFSCGymEnvShifts
from .episode_metrics import compute_episode_metrics, merge_resource_metrics
from .supply_chain import MFSCSimulation


CONTRACT_ID = "garrido_learning_v1"
PROXY_CONTRACT_PATH = (
    Path(__file__).resolve().parent / "data" / "garrido_proxy_v1_freeze_2026-07-10.json"
)
BUFFER_LEVELS: tuple[int, ...] = (0, 168, 336, 504, 672, 1344)
RISK_FAMILIES: dict[str, tuple[str, ...]] = {
    "R1": ("R11", "R12", "R13", "R14"),
    "R2": ("R21", "R22", "R23", "R24"),
    "mixed": ("R11", "R12", "R13", "R14", "R21", "R22", "R23", "R24"),
    "R3": ("R3",),
}
TRAINING_SPLITS = {"training"}

OBSERVATION_FIELDS: tuple[str, ...] = (
    "buffer_level_fraction",
    "effective_shift_fraction",
    "pending_shift_fraction",
    "raw_material_wdc",
    "raw_material_al",
    "rations_al",
    "rations_sb",
    "rations_sb_dispatch",
    "rations_cssu",
    "rations_theatre",
    "pending_batch",
    "in_transit",
    "pending_backorder_qty",
    "pending_backorder_count",
    "oldest_backorder_age_hours",
    "previous_week_demanded",
    "previous_week_produced",
    "previous_week_delivered",
    "rolling_fill_rate_4w",
    "previous_week_capacity_utilization",
    "fraction_operations_down",
    "previous_week_downtime_hours",
    "week_phase_sin",
    "week_phase_cos",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class CampaignTape:
    """Immutable, hash-addressed exogenous campaign description."""

    campaign_id: str
    family: str
    risk_level: str
    base_seed: int
    horizon_weeks: int
    split: str
    risk_events: tuple[dict[str, Any], ...] = ()
    calendar_materialized: bool = False
    contract_version: str = CONTRACT_ID

    def __post_init__(self) -> None:
        if self.family not in RISK_FAMILIES:
            raise ValueError(f"Unknown risk family {self.family!r}.")
        if self.risk_level not in {"current", "increased"}:
            raise ValueError("L-program tapes support current/increased risk levels only.")
        if int(self.horizon_weeks) <= 0:
            raise ValueError("horizon_weeks must be positive.")
        if self.family == "R3" and self.split in TRAINING_SPLITS:
            raise ValueError("R3 is OOD-only and cannot be assigned to training.")

    @property
    def enabled_risks(self) -> tuple[str, ...]:
        return RISK_FAMILIES[self.family]

    def payload(self, *, include_hash: bool = False) -> dict[str, Any]:
        payload = {
            "campaign_id": self.campaign_id,
            "family": self.family,
            "risk_level": self.risk_level,
            "base_seed": int(self.base_seed),
            "horizon_weeks": int(self.horizon_weeks),
            "split": self.split,
            "risk_events": [_jsonable(row) for row in self.risk_events],
            "calendar_materialized": bool(self.calendar_materialized),
            "contract_version": self.contract_version,
        }
        if include_hash:
            payload["sha256"] = self.digest()
        return payload

    def digest(self) -> str:
        raw = json.dumps(
            self.payload(include_hash=False), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return sha256(raw).hexdigest()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CampaignTape":
        clean = dict(payload)
        clean.pop("sha256", None)
        clean["risk_events"] = tuple(dict(row) for row in clean.get("risk_events", ()))
        tape = cls(**clean)
        expected = payload.get("sha256")
        if expected is not None and str(expected) != tape.digest():
            raise ValueError("CampaignTape hash mismatch.")
        return tape


@dataclass(frozen=True)
class FixedNormalizerStats:
    """Frozen observation transform fitted on calibration data only."""

    fields: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    clip: float = 10.0
    calibration_sha256: str = "unfitted_identity"

    def __post_init__(self) -> None:
        n = len(self.fields)
        if n != len(self.mean) or n != len(self.std):
            raise ValueError("Normalizer fields/mean/std lengths differ.")
        if any(float(value) <= 0.0 for value in self.std):
            raise ValueError("Normalizer standard deviations must be positive.")
        if float(self.clip) <= 0.0:
            raise ValueError("Normalizer clip must be positive.")

    def transform(self, observation: np.ndarray) -> np.ndarray:
        if observation.shape != (len(self.fields),):
            raise ValueError(
                f"Expected observation shape {(len(self.fields),)}, got {observation.shape}."
            )
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        normalized = (observation.astype(np.float64) - mean) / std
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def payload(self) -> dict[str, Any]:
        return {
            "fields": list(self.fields),
            "mean": list(self.mean),
            "std": list(self.std),
            "clip": float(self.clip),
            "calibration_sha256": self.calibration_sha256,
        }

    @classmethod
    def identity(cls) -> "FixedNormalizerStats":
        n = len(OBSERVATION_FIELDS)
        return cls(
            fields=OBSERVATION_FIELDS,
            mean=tuple(0.0 for _ in range(n)),
            std=tuple(1.0 for _ in range(n)),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FixedNormalizerStats":
        return cls(
            fields=tuple(str(value) for value in payload["fields"]),
            mean=tuple(float(value) for value in payload["mean"]),
            std=tuple(float(value) for value in payload["std"]),
            clip=float(payload.get("clip", 10.0)),
            calibration_sha256=str(
                payload.get("calibration_sha256", "unspecified")
            ),
        )


def fit_fixed_normalizer(
    observations: Iterable[Sequence[float]],
    *,
    calibration_sha256: str,
    clip: float = 10.0,
) -> FixedNormalizerStats:
    matrix = np.asarray(list(observations), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(OBSERVATION_FIELDS):
        raise ValueError(
            "Calibration observations must have shape "
            f"(n, {len(OBSERVATION_FIELDS)})."
        )
    if matrix.shape[0] < 2:
        raise ValueError("At least two calibration observations are required.")
    std = matrix.std(axis=0, ddof=1)
    std = np.where(std > 1e-12, std, 1.0)
    return FixedNormalizerStats(
        fields=OBSERVATION_FIELDS,
        mean=tuple(float(v) for v in matrix.mean(axis=0)),
        std=tuple(float(v) for v in std),
        clip=float(clip),
        calibration_sha256=str(calibration_sha256),
    )


def materialize_campaign_tape(
    tape: CampaignTape,
    *,
    proxy_contract_path: str | Path = PROXY_CONTRACT_PATH,
) -> CampaignTape:
    """Discover one exogenous risk calendar and express it post-warm-up.

    The discovery simulation uses a fixed B0/S1 reference and split RNG streams.
    Events overlapping the treatment start are clipped to their remaining duration.
    Replaying the returned tape after each policy's own warm-up guarantees the same
    shock times, durations, operations, and quantities relative to treatment.
    """
    if tape.calendar_materialized:
        return tape
    proxy = json.loads(Path(proxy_contract_path).read_text(encoding="utf-8"))
    sim_kwargs = dict(proxy["sim_kwargs"])
    sim_kwargs.pop("risk_level", None)
    sim_kwargs.pop("seed_stream_mode", None)
    horizon = max(
        float(SIMULATION_HORIZON),
        8_000.0 + float(tape.horizon_weeks) * HOURS_PER_WEEK,
    )
    sim = MFSCSimulation(
        shifts=1,
        initial_buffers={},
        seed=tape.base_seed,
        horizon=horizon,
        risks_enabled=False,
        risk_level=tape.risk_level,
        enabled_risks=set(tape.enabled_risks),
        strict_exogenous_crn=True,
        **sim_kwargs,
    )
    sim._start_processes()
    while not sim.warmup_complete and sim.env.now < sim.horizon:
        sim.env.run(until=min(sim.env.now + 1.0, sim.horizon))
    treatment_start = float(sim.env.now)
    # Risk processes begin at the treatment boundary, not at time zero.  This
    # prevents buffer/shift-dependent warm-up durations from changing the
    # realized risk calendar.  The generated calendar is subsequently replayed
    # under every policy with endogenous risk generators disabled.
    sim.risks_enabled = True
    sim._today_produced = 0
    risk_processes = {
        "R11": sim._risk_R11,
        "R12": sim._risk_R12,
        "R13": sim._risk_R13,
        "R14": sim._risk_R14,
        "R21": sim._risk_R21,
        "R22": sim._risk_R22,
        "R23": sim._risk_R23,
        "R24": sim._risk_R24,
        "R3": sim._risk_R3,
    }
    for risk_id in tape.enabled_risks:
        sim.env.process(risk_processes[risk_id]())
    treatment_end = min(
        treatment_start + float(tape.horizon_weeks) * HOURS_PER_WEEK,
        sim.horizon,
    )
    sim.env.run(until=treatment_end)
    rows: list[dict[str, Any]] = []
    for event in sim.risk_events:
        start = float(event.start_time)
        end = float(event.end_time)
        if end < treatment_start or start >= treatment_end:
            continue
        relative_start = max(0.0, start - treatment_start)
        relative_end = max(relative_start, min(end, treatment_end) - treatment_start)
        rows.append(
            {
                "risk_id": str(event.risk_id),
                "start_time": relative_start,
                "end_time": relative_end,
                "duration": max(0.0, relative_end - relative_start),
                "affected_ops": [int(value) for value in event.affected_ops],
                "description": str(event.description),
                "magnitude": float(event.magnitude),
                "unit": str(event.unit),
            }
        )
    rows.sort(key=lambda row: (float(row["start_time"]), str(row["risk_id"])))
    return replace(tape, risk_events=tuple(rows), calendar_materialized=True)


@dataclass(frozen=True)
class RewardScales:
    late_backlog_hours: float = 1.0
    total_backlog_hours: float = 1.0
    extra_shift_hours: float = 1.0

    def __post_init__(self) -> None:
        if min(asdict(self).values()) <= 0.0:
            raise ValueError("Reward scales must be positive.")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RewardScales":
        return cls(
            late_backlog_hours=float(payload["late_backlog_hours"]),
            total_backlog_hours=float(payload["total_backlog_hours"]),
            extra_shift_hours=float(payload["extra_shift_hours"]),
        )


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def compute_system_recovery_metrics(
    weekly_history: Sequence[Mapping[str, float]],
    risk_events: Sequence[Any],
    *,
    treatment_start: float,
    cluster_gap_hours: float = HOURS_PER_WEEK,
) -> dict[str, float]:
    """Compute system-level TTR from risk clusters and weekly service observations.

    Risk events separated by less than one week form one compound cluster.  Recovery
    requires two consecutive weekly observations with fill rate at least 95% of the
    four-week pre-onset median and backlog no more than 105% of its corresponding
    baseline.  Open clusters are right-censored.
    """

    def event_value(event: Any, field: str) -> float:
        if isinstance(event, Mapping):
            return float(event.get(field, 0.0))
        return float(getattr(event, field, 0.0))

    events = sorted(
        [
            (event_value(event, "start_time"), event_value(event, "end_time"))
            for event in risk_events
            if event_value(event, "end_time") >= treatment_start
        ],
        key=lambda item: item[0],
    )
    clusters: list[list[float]] = []
    for start, end in events:
        if not clusters or start - clusters[-1][1] >= cluster_gap_hours:
            clusters.append([start, max(start, end)])
        else:
            clusters[-1][1] = max(clusters[-1][1], end)

    history = sorted(weekly_history, key=lambda row: float(row["time"]))
    recovered: list[float] = []
    censored = 0
    for onset, end in clusters:
        baseline_rows = [
            row
            for row in history
            if onset - 4.0 * HOURS_PER_WEEK <= float(row["time"]) < onset
        ]
        if not baseline_rows:
            censored += 1
            continue
        fill_baseline = median(float(row["fill_rate"]) for row in baseline_rows)
        backlog_baseline = median(float(row["backlog_qty"]) for row in baseline_rows)
        candidates = [row for row in history if float(row["time"]) >= end]
        consecutive = 0
        recovery_time: float | None = None
        for row in candidates:
            healthy = (
                float(row["fill_rate"]) >= 0.95 * fill_baseline
                and float(row["backlog_qty"]) <= 1.05 * backlog_baseline
            )
            consecutive = consecutive + 1 if healthy else 0
            if consecutive >= 2:
                recovery_time = float(row["time"]) - onset
                break
        if recovery_time is None:
            censored += 1
        else:
            recovered.append(max(0.0, recovery_time))

    n_clusters = len(clusters)
    return {
        "system_ttr_mean": float(np.mean(recovered)) if recovered else 0.0,
        "system_ttr_p95": _percentile(recovered, 0.95),
        "system_ttr_n_recovered": float(len(recovered)),
        "system_ttr_n_censored": float(censored),
        "system_ttr_n_clusters": float(n_clusters),
        "system_ttr_censored_fraction": (
            float(censored / n_clusters) if n_clusters else 0.0
        ),
    }


class GarridoLearningEnv(gym.Env[np.ndarray, int]):
    """Categorical, fixed-buffer weekly control environment for Program L(e-1)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        max_steps: int = 104,
        buffer_level: int = 0,
        lambda_shift: float = 0.25,
        switching_weight: float = 0.10,
        reward_scales: RewardScales | None = None,
        normalizer: FixedNormalizerStats | None = None,
        proxy_contract_path: str | Path = PROXY_CONTRACT_PATH,
    ) -> None:
        super().__init__()
        if int(max_steps) <= 0:
            raise ValueError("max_steps must be positive.")
        self.max_steps = int(max_steps)
        self.default_buffer_level = self._validate_buffer_level(buffer_level)
        self.lambda_shift = max(0.0, float(lambda_shift))
        self.switching_weight = max(0.0, float(switching_weight))
        self.reward_scales = reward_scales or RewardScales()
        self.normalizer = normalizer or FixedNormalizerStats.identity()
        if tuple(self.normalizer.fields) != OBSERVATION_FIELDS:
            raise ValueError("Normalizer fields do not match garrido_learning_v1.")

        proxy_path = Path(proxy_contract_path)
        proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
        if proxy.get("contract_id") != "garrido_proxy_v1":
            raise ValueError("Program L must inherit the frozen garrido_proxy_v1 contract.")
        if not bool(proxy.get("rl_training_allowed", False)):
            raise ValueError("The selected proxy does not authorize RL training.")
        self.proxy_contract_path = proxy_path
        self.proxy_sha256 = sha256(proxy_path.read_bytes()).hexdigest()
        sim = dict(proxy["sim_kwargs"])
        self._base = MFSCGymEnvShifts(
            step_size_hours=HOURS_PER_WEEK,
            max_steps=self.max_steps,
            risk_level=str(sim["risk_level"]),
            reward_mode="control_v1",
            observation_version="v1",
            action_contract="track_a_v1",
            action_mode="full",
            year_basis=str(sim["year_basis"]),
            stochastic_pt=False,
            warmup_hours_override=0.0,
            warmup_trigger=str(sim["warmup_trigger"]),
            raw_material_flow_mode=str(sim["raw_material_flow_mode"]),
            raw_material_order_up_to_multiplier=float(
                sim["raw_material_order_up_to_multiplier"]
            ),
            demand_on_hand_fulfillment_delay=float(
                sim["demand_on_hand_fulfillment_delay"]
            ),
            risk_occurrence_mode=str(sim["risk_occurrence_mode"]),
            risk_attribution_source=str(sim["risk_attribution_source"]),
            ret_recovery_period_mode=str(sim["ret_recovery_period_mode"]),
            replenishment_route_aware=bool(sim["replenishment_route_aware"]),
            procurement_contract_mode=str(sim["procurement_contract_mode"]),
            order_fulfillment_mode=str(sim["order_fulfillment_mode"]),
            op9_dispatch_policy=str(sim["op9_dispatch_policy"]),
            downstream_transport_capacity_mode=str(
                sim["downstream_transport_capacity_mode"]
            ),
            op9_freight_offset_hours=float(sim["op9_freight_offset_hours"]),
            r24_attribution_window_hours=float(
                sim["r24_attribution_window_hours"]
            ),
            demand_start_after_warmup=bool(sim["demand_start_after_warmup"]),
            priming_enabled=False,
            clear_backlog_after_priming=False,
            risks_enabled=False,
        )
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-float(self.normalizer.clip),
            high=float(self.normalizer.clip),
            shape=(len(OBSERVATION_FIELDS),),
            dtype=np.float32,
        )
        self._campaign_tape: CampaignTape | None = None
        self._buffer_level = self.default_buffer_level
        self._buffer_targets: dict[str, float] = {}
        self._effective_shift = 1
        self._pending_shift = 1
        self._previous_effective_shift = 1
        self._previous_total_backlog_auc = 0.0
        self._previous_late_backlog_auc = 0.0
        self._recent_info: dict[str, Any] = {}
        self._treatment_start = 0.0
        self._weekly_history: list[dict[str, float]] = []
        self._resource_totals: dict[str, float] = {}
        self._reward_totals: dict[str, float] = {}
        self._noncontrol_params: dict[str, Any] = {}

    @staticmethod
    def _validate_buffer_level(level: int) -> int:
        value = int(level)
        if value not in BUFFER_LEVELS:
            raise ValueError(f"buffer_level must be one of {BUFFER_LEVELS}, got {value}.")
        return value

    @property
    def sim(self) -> Any:
        return self._base.sim

    @property
    def campaign_tape(self) -> CampaignTape:
        if self._campaign_tape is None:
            raise RuntimeError("Call reset() before requesting campaign_tape.")
        return self._campaign_tape

    def _cumulative_backlog_aucs(self) -> tuple[float, float]:
        """Return monotone total-wait and post-promise quantity-hour areas."""
        now = float(self.sim.env.now)
        total_auc = 0.0
        late_auc = 0.0
        for order in self.sim.orders:
            if float(order.OPTj) < self._treatment_start:
                continue
            end = float(order.OATj) if order.OATj is not None else now
            quantity = float(order.quantity or 0.0)
            total_auc += max(0.0, end - float(order.OPTj)) * quantity
            late_auc += max(
                0.0, end - (float(order.OPTj) + float(order.LTj or 0.0))
            ) * quantity
        return total_auc, late_auc

    def _oldest_backlog_age(self) -> float:
        if not self.sim.pending_backorders:
            return 0.0
        oldest = min(float(order.OPTj) for order in self.sim.pending_backorders)
        return max(0.0, float(self.sim.env.now) - oldest)

    def _rolling_fill_rate(self) -> float:
        return float(self.sim.get_observation_v7_extra()[4])

    def _raw_observation(self) -> np.ndarray:
        detail = self.sim._inventory_detail()
        info = self._recent_info
        available = float(info.get("new_available_assembly_capacity", 0.0))
        produced = float(info.get("new_produced", 0.0))
        utilization = produced / available if available > 0.0 else 0.0
        fraction_down = sum(
            1 for op in range(1, 14) if self.sim.op_down_count.get(op, 0) > 0
        ) / 13.0
        phase = 2.0 * math.pi * (float(self._base.current_step) % 52.0) / 52.0
        return np.asarray(
            [
                BUFFER_LEVELS.index(self._buffer_level) / 5.0,
                (self._effective_shift - 1) / 2.0,
                (self._pending_shift - 1) / 2.0,
                float(detail["raw_material_wdc"]),
                float(detail["raw_material_al"]),
                float(detail["rations_al"]),
                float(detail["rations_sb"]),
                float(detail["rations_sb_dispatch"]),
                float(detail["rations_cssu"]),
                float(detail["rations_theatre"]),
                float(self.sim._pending_batch),
                float(self.sim._in_transit),
                float(self.sim.pending_backorder_qty),
                float(len(self.sim.pending_backorders)),
                self._oldest_backlog_age(),
                float(info.get("new_demanded", 0.0)),
                produced,
                float(info.get("new_delivered", 0.0)),
                self._rolling_fill_rate(),
                float(np.clip(utilization, 0.0, 2.0)),
                fraction_down,
                float(info.get("step_disruption_hours", 0.0)),
                math.sin(phase),
                math.cos(phase),
            ],
            dtype=np.float32,
        )

    def raw_observation(self) -> np.ndarray:
        """Return the pre-normalization observation for calibration/audit."""
        return self._raw_observation().copy()

    def _observation(self) -> np.ndarray:
        return self.normalizer.transform(self._raw_observation())

    def _assert_control_invariants(self) -> None:
        current_targets = {
            key: float(value) for key, value in self.sim.inventory_buffer_targets.items()
        }
        if current_targets != self._buffer_targets:
            raise RuntimeError("Strategic buffer targets changed within a campaign.")
        allowed = {"assembly_shifts", "batch_size"}
        for key, expected in self._noncontrol_params.items():
            if key in allowed:
                continue
            if self.sim.params.get(key) != expected:
                raise RuntimeError(f"Shift action changed non-control parameter {key!r}.")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        options = dict(options or {})
        tape_value = options.pop("campaign_tape", None)
        if tape_value is None:
            base_seed = int(seed if seed is not None else 42)
            tape = CampaignTape(
                campaign_id=f"adhoc-{base_seed}",
                family="mixed",
                risk_level="current",
                base_seed=base_seed,
                horizon_weeks=self.max_steps,
                split="smoke",
            )
        elif isinstance(tape_value, CampaignTape):
            tape = tape_value
        else:
            tape = CampaignTape.from_mapping(tape_value)
        if int(tape.horizon_weeks) != self.max_steps:
            raise ValueError(
                f"Tape horizon {tape.horizon_weeks} != env max_steps {self.max_steps}."
            )
        self._campaign_tape = tape
        self._buffer_level = self._validate_buffer_level(
            int(options.pop("buffer_level", self.default_buffer_level))
        )
        initial_state_seed = int(
            options.pop(
                "initial_state_seed",
                seed if seed is not None else tape.base_seed,
            )
        )
        initial_shift = int(options.pop("initial_shift", 1))
        if initial_shift not in (1, 2, 3):
            raise ValueError("initial_shift must be 1, 2, or 3.")
        if options:
            raise ValueError(f"Unknown reset options: {sorted(options)}")

        targets = (
            {} if self._buffer_level == 0 else dict(INVENTORY_BUFFERS[self._buffer_level])
        )
        if tape.split != "smoke" and not tape.calendar_materialized:
            raise ValueError(
                "Scientific CampaignTapes must contain a materialized post-warm-up "
                "risk calendar. Use materialize_campaign_tape()."
            )
        self._base.risk_level = tape.risk_level
        self._base.enabled_risks = set()
        self._base.risk_event_tape = None
        _base_obs, base_info = self._base.reset(
            seed=initial_state_seed,
            options={
                "initial_buffers": targets,
                "initial_shifts": initial_shift,
                "inventory_replenishment_period": (
                    None if self._buffer_level == 0 else float(self._buffer_level)
                ),
            },
        )
        self._effective_shift = initial_shift
        self._pending_shift = initial_shift
        self._previous_effective_shift = initial_shift
        self._treatment_start = float(self.sim.env.now)
        if tape.risk_events:
            absolute_events = []
            for row in tape.risk_events:
                shifted = dict(row)
                shifted["start_time"] = self._treatment_start + float(
                    row["start_time"]
                )
                shifted["end_time"] = self._treatment_start + float(
                    row.get("end_time", row["start_time"])
                )
                absolute_events.append(shifted)
            self.sim.risk_event_tape = self.sim._normalize_risk_event_tape(
                absolute_events
            )
            self.sim.env.process(self.sim._risk_event_tape_replay())
        (
            self._previous_total_backlog_auc,
            self._previous_late_backlog_auc,
        ) = self._cumulative_backlog_aucs()
        self._recent_info = {}
        self._buffer_targets = {
            key: float(value) for key, value in self.sim.inventory_buffer_targets.items()
        }
        self._noncontrol_params = dict(self.sim.params)
        self._weekly_history = [
            {
                "time": float(self.sim.env.now),
                "fill_rate": float(self.sim._fill_rate()),
                "backlog_qty": float(self.sim.pending_backorder_qty),
            }
        ]
        self._resource_totals = {
            "shift_hours": 0.0,
            "extra_shift_hours": 0.0,
            "switches": 0.0,
        }
        self._reward_totals = {
            "late_backlog_hours": 0.0,
            "total_backlog_hours": 0.0,
            "return": 0.0,
        }
        info = {
            **base_info,
            "contract_id": CONTRACT_ID,
            "proxy_contract_id": "garrido_proxy_v1",
            "proxy_sha256": self.proxy_sha256,
            "campaign_tape": tape.payload(include_hash=True),
            "buffer_level": self._buffer_level,
            "buffer_targets": dict(self._buffer_targets),
            "effective_shift": self._effective_shift,
            "pending_shift": self._pending_shift,
            "initial_state_seed": initial_state_seed,
            "observation_fields": list(OBSERVATION_FIELDS),
            "normalizer": self.normalizer.payload(),
            "risk_calendar_mode": "post_warmup_realized_tape",
            "warmup_mode": "endogenous_physical_completion",
        }
        return self._observation(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Expected categorical action 0..2, got {action!r}.")
        requested_shift = int(action) + 1
        applied_shift = int(self._pending_shift)
        switched = float(applied_shift != self._previous_effective_shift)

        _base_obs, _base_reward, terminated, truncated, info = self._base.step(
            {"assembly_shifts": applied_shift}
        )
        self._effective_shift = applied_shift
        self._pending_shift = requested_shift
        self._recent_info = dict(info)

        current_pending = float(self.sim.pending_backorder_qty)
        total_auc, late_auc = self._cumulative_backlog_aucs()
        total_backlog_hours = max(
            0.0, total_auc - self._previous_total_backlog_auc
        )
        late_backlog_hours = max(
            0.0, late_auc - self._previous_late_backlog_auc
        )
        extra_shift_hours = max(0, applied_shift - 1) * HOURS_PER_WEEK
        shift_hours = applied_shift * HOURS_PER_WEEK
        reward = -(
            late_backlog_hours / self.reward_scales.late_backlog_hours
            + total_backlog_hours / self.reward_scales.total_backlog_hours
            + self.lambda_shift
            * extra_shift_hours
            / self.reward_scales.extra_shift_hours
            + self.switching_weight * switched
        )

        self._previous_total_backlog_auc = total_auc
        self._previous_late_backlog_auc = late_auc
        self._previous_effective_shift = applied_shift
        self._resource_totals["shift_hours"] += shift_hours
        self._resource_totals["extra_shift_hours"] += extra_shift_hours
        self._resource_totals["switches"] += switched
        self._reward_totals["late_backlog_hours"] += late_backlog_hours
        self._reward_totals["total_backlog_hours"] += total_backlog_hours
        self._reward_totals["return"] += reward
        self._weekly_history.append(
            {
                "time": float(self.sim.env.now),
                "fill_rate": float(self.sim._fill_rate()),
                "backlog_qty": current_pending,
            }
        )
        self._assert_control_invariants()

        enriched = dict(info)
        enriched.update(
            {
                "contract_id": CONTRACT_ID,
                "campaign_id": self.campaign_tape.campaign_id,
                "campaign_sha256": self.campaign_tape.digest(),
                "buffer_level": self._buffer_level,
                "buffer_targets": dict(self._buffer_targets),
                "requested_shift": requested_shift,
                "effective_shift": applied_shift,
                "pending_shift": self._pending_shift,
                "shift_switch": switched,
                "late_backlog_hours_step": late_backlog_hours,
                "total_backlog_hours_step": total_backlog_hours,
                "extra_shift_hours_step": extra_shift_hours,
                "l_program_reward": float(reward),
                "l_program_reward_components": {
                    "late_backlog_hours": late_backlog_hours,
                    "total_backlog_hours": total_backlog_hours,
                    "extra_shift_hours": extra_shift_hours,
                    "switch": switched,
                    "lambda_shift": self.lambda_shift,
                    "switching_weight": self.switching_weight,
                },
                "resource_totals": dict(self._resource_totals),
                "reward_totals": dict(self._reward_totals),
                "ret_excel": float(info["ret_excel_mean"]),
                "raw_observation": self._raw_observation().astype(float).tolist(),
            }
        )
        return self._observation(), float(reward), terminated, truncated, enriched

    def terminal_metrics(self) -> dict[str, float]:
        panel = compute_episode_metrics(self.sim, treatment_start=self._treatment_start)
        panel = merge_resource_metrics(
            panel,
            shift_hours=self._resource_totals["shift_hours"],
            extra_shift_hours=self._resource_totals["extra_shift_hours"],
            strategic_buffer_units=float(sum(self._buffer_targets.values())),
            end_state_inventory=float(sum(self.sim._inventory_detail().values())),
        )
        panel.update(
            compute_system_recovery_metrics(
                self._weekly_history,
                self.sim.risk_events,
                treatment_start=self._treatment_start,
            )
        )
        panel.update(
            {
                "switches": float(self._resource_totals["switches"]),
                "late_backlog_hours": float(
                    self._reward_totals["late_backlog_hours"]
                ),
                "total_backlog_hours": float(
                    self._reward_totals["total_backlog_hours"]
                ),
                "l_program_return": float(self._reward_totals["return"]),
            }
        )
        return panel

    def audit_state(self) -> dict[str, Any]:
        """Return a read-only checkpoint used by prefix-replay and V&V gates."""
        return {
            "time": float(self.sim.env.now),
            "raw_observation": self.raw_observation().astype(float).tolist(),
            "buffer_targets": dict(self._buffer_targets),
            "effective_shift": int(self._effective_shift),
            "pending_shift": int(self._pending_shift),
            "resource_totals": dict(self._resource_totals),
            "reward_totals": dict(self._reward_totals),
            "flow_ledger": dict(self.sim.flow_ledger()),
            "risk_events": [
                {
                    "risk_id": str(event.risk_id),
                    "start_time": float(event.start_time),
                    "end_time": float(event.end_time),
                    "duration": float(event.duration),
                    "affected_ops": list(event.affected_ops),
                }
                for event in self.sim.risk_events
            ],
        }

    def close(self) -> None:
        self._base.close()


def make_garrido_learning_env(**kwargs: Any) -> GarridoLearningEnv:
    return GarridoLearningEnv(**kwargs)
