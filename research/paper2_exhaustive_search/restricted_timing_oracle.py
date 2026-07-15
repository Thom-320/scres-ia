"""Mechanics for the preregistered restricted timing-ceiling experiment.

The module deliberately separates schedule construction, observable triggering,
resource accounting, and safe-oracle adjudication.  It never trains a policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from supply_chain.episode_metrics import compute_episode_metrics
from supply_chain.event_triggered_env import make_event_triggered_track_a_env


DAY = 24.0
WEEK = 168.0
TARGET_RISKS = frozenset({"R22", "R24"})

HIGHER_GUARDRAILS = (
    "ret_excel_full_ledger",
    "ration_ret_excel",
    "ret_excel_cvar10",
    "worst_node_or_product_fill",
)
LOWER_GUARDRAILS = (
    "lost_orders",
    "ret_excel_omitted_n",
    "backorder_qty_final",
    "backlog_age_max",
    "shift_hours",
    "surge_hours",
    "buffer_target_unit_hours",
    "op8_convoy_vehicle_hours",
)


@dataclass(frozen=True)
class Posture:
    label: str
    buffer_fraction: float
    shifts: int
    nominal_resource: float

    @property
    def shift_signal(self) -> float:
        return (-1.0, 0.0, 1.0)[self.shifts - 1]


@dataclass(frozen=True)
class ScheduleSpec:
    family: str
    schedule_id: str
    payload: Any


def posture_from_label(label: str, resource: float | None = None) -> Posture:
    """Parse the frozen ``I<frac>_S<shift>`` labels emitted by Track-A."""
    try:
        frac_text, shift_text = label.split("_S")
        fraction = float(frac_text.removeprefix("I").removeprefix("f"))
        shifts = int(shift_text)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"cannot parse Track-A posture label {label!r}") from exc
    nominal = (
        0.5 * fraction + 0.5 * ((shifts - 1) / 2.0)
        if resource is None
        else float(resource)
    )
    return Posture(label, fraction, shifts, nominal)


def periodic_binary_calendars() -> list[ScheduleSpec]:
    """The complete frozen 8-week periodic low/high open-loop frontier."""
    return [
        ScheduleSpec(
            family="open_loop_8week_periodic",
            schedule_id=f"periodic_{index:03d}",
            payload=tuple((index >> week) & 1 for week in range(8)),
        )
        for index in range(256)
    ]


def privileged_windows(
    risk_events: Iterable[Any], *, entry_offset_hours: float, exit_offset_hours: float = 72.0
) -> list[tuple[float, float]]:
    """Union high-posture windows around realized R22/R24 events."""
    raw = sorted(
        (
            float(getattr(event, "start_time")) + float(entry_offset_hours),
            float(getattr(event, "end_time")) + float(exit_offset_hours),
        )
        for event in risk_events
        if str(getattr(event, "risk_id", "")) in TARGET_RISKS
    )
    merged: list[tuple[float, float]] = []
    for start, end in raw:
        end = max(start, end)
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def schedule_is_high(
    spec: ScheduleSpec,
    *,
    now: float,
    treatment_start: float,
    risk_events: Sequence[Any],
) -> bool:
    if spec.family == "constant":
        return bool(spec.payload)
    if spec.family == "open_loop_8week_periodic":
        week = max(0, int((now - treatment_start) // WEEK))
        return bool(spec.payload[week % 8])
    if spec.family in {"restricted_privileged", "weekly_privileged"}:
        windows = privileged_windows(
            risk_events,
            entry_offset_hours=float(spec.payload),
        )
        if spec.family == "weekly_privileged":
            windows = [
                (
                    treatment_start
                    + math.floor((start - treatment_start) / WEEK) * WEEK,
                    treatment_start
                    + math.floor((end - treatment_start) / WEEK) * WEEK,
                )
                for start, end in windows
            ]
        return any(start <= now < end for start, end in windows)
    raise ValueError(f"schedule_is_high does not handle family {spec.family!r}")


def _risk_signal(events: Sequence[Any], now: float, lookback_hours: float = WEEK) -> float:
    return float(
        sum(
            str(getattr(event, "risk_id", "")) in TARGET_RISKS
            and now - lookback_hours <= float(getattr(event, "start_time")) <= now
            for event in events
        )
    )


class ObservableEwmaTrigger:
    """Frozen non-anticipative trigger; future events never enter ``decide``."""

    def __init__(
        self,
        *,
        decay: float = 0.85,
        enter: float = 0.15,
        exit: float = 0.05,
        minimum_high_dwell_hours: float = 72.0,
        backlog_age_scale_hours: float = 672.0,
        inventory_shortfall_weight: float = 0.25,
    ) -> None:
        self.decay = float(decay)
        self.enter = float(enter)
        self.exit = float(exit)
        self.minimum_high_dwell_hours = float(minimum_high_dwell_hours)
        self.backlog_age_scale_hours = float(backlog_age_scale_hours)
        self.inventory_shortfall_weight = float(inventory_shortfall_weight)
        self.ewma = 0.0
        self.high = False
        self.high_since = float("-inf")

    def decide(
        self,
        *,
        now: float,
        observed_signal: float,
        backlog_age_hours: float,
        inventory_shortfall_fraction: float,
    ) -> bool:
        risk_component = min(1.0, float(observed_signal))
        self.ewma = self.decay * self.ewma + (1.0 - self.decay) * risk_component
        score = (
            self.ewma
            + 0.25 * min(1.0, backlog_age_hours / self.backlog_age_scale_hours)
            + self.inventory_shortfall_weight * min(1.0, inventory_shortfall_fraction)
        )
        if not self.high and score >= self.enter:
            self.high = True
            self.high_since = now
        elif (
            self.high
            and now - self.high_since >= self.minimum_high_dwell_hours
            and score <= self.exit
        ):
            self.high = False
        return self.high


def placebo_signal_series(
    series: Sequence[float], *, family: str, seed: int, cross_tape: Sequence[float] | None = None
) -> list[float]:
    values = list(map(float, series))
    if family == "real":
        return values
    if family == "stale_168h":
        return [0.0] * 7 + values[:-7]
    if family == "shuffled_within_tape":
        rng = np.random.default_rng(int(seed) + 7460999)
        order = rng.permutation(len(values))
        return [values[int(index)] for index in order]
    if family == "cross_tape_shift17":
        if cross_tape is None or len(cross_tape) != len(values):
            raise ValueError("cross-tape placebo requires a same-length signal series")
        return list(map(float, cross_tape))
    raise ValueError(f"unknown placebo family {family!r}")


def select_frozen_postures(
    raw_rows_path: Path,
    *,
    budget_cap: float = 0.5,
    robust_label: str | None = None,
    bootstrap_seed: int = 7460999,
    n_bootstrap: int = 10_000,
    regime_order: Sequence[str] = (
        "R2_current",
        "R2_OAT_R24_increased",
        "R2_OAT_R22_increased",
        "Cf19",
    ),
) -> tuple[Posture, Posture, str] | None:
    """Apply the preregistered first-passing regime rule to burned risk rows."""
    with raw_rows_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by: dict[tuple[str, str], list[dict[str, str]]] = {}
    resources: dict[str, float] = {}
    for row in rows:
        by.setdefault((row["profile"], row["candidate"]), []).append(row)
        resources[row["candidate"]] = float(row["candidate_resource_nominal"])
    eligible = sorted(label for label, value in resources.items() if value <= budget_cap + 1e-12)
    if not eligible or not all((regime_order[0], label) in by for label in eligible):
        return None

    def metric_mean(profile: str, label: str, metric: str) -> float:
        return float(np.mean([float(row[metric]) for row in by[(profile, label)]]))

    current = regime_order[0]
    low_label = robust_label or max(
        eligible, key=lambda label: (metric_mean(current, label, "ret_excel"), label)
    )
    if low_label not in eligible:
        return None
    low = posture_from_label(low_label, resources[low_label])
    for profile in regime_order[1:]:
        if not all((profile, label) in by for label in eligible):
            continue
        admissible = []
        for label in eligible:
            if all(
                metric_mean(profile, label, metric)
                >= metric_mean(profile, low_label, metric) - 1e-12
                for metric in ("ration_ret_excel", "ret_excel_cvar10")
            ) and all(
                metric_mean(profile, label, metric)
                <= metric_mean(profile, low_label, metric) + 1e-12
                for metric in (
                    "lost_orders",
                    "backorder_qty_final",
                    "backlog_age_max",
                    "service_loss_auc_ration_hours",
                    "resource",
                )
            ):
                admissible.append(label)
        if not admissible:
            continue
        high_label = max(
            admissible, key=lambda label: (metric_mean(profile, label, "ret_excel"), label)
        )
        high_by_seed = {
            int(row["seed"]): float(row["ret_excel"])
            for row in by[(profile, high_label)]
        }
        low_by_seed = {
            int(row["seed"]): float(row["ret_excel"])
            for row in by[(profile, low_label)]
        }
        common_seeds = sorted(set(high_by_seed) & set(low_by_seed))
        if not common_seeds:
            continue
        deltas = np.asarray(
            [high_by_seed[seed] - low_by_seed[seed] for seed in common_seeds],
            dtype=float,
        )
        rng = np.random.default_rng(int(bootstrap_seed))
        boot = np.asarray([
            float(np.mean(deltas[rng.integers(0, len(deltas), len(deltas))]))
            for _ in range(int(n_bootstrap))
        ])
        lcb95 = float(np.quantile(boot, 0.05))
        if (
            float(np.mean(deltas)) >= 0.01
            and lcb95 > 0.0
            and int(np.sum(deltas > 0.0)) >= 4
        ):
            return low, posture_from_label(high_label, resources[high_label]), profile
    return None


def _aggregate_worst_fill(panel: Mapping[str, Any]) -> float:
    # The governing Track-A contract has aggregate CSSU topology and one ration
    # class, so shedding across nodes/products is physically unavailable.  The
    # aggregate flow fill is therefore the exact worst identifiable class.
    return float(panel["flow_fill_rate"])


def evaluate_schedule(
    *,
    seed: int,
    risk_overrides: Mapping[str, str],
    low: Posture,
    high: Posture,
    spec: ScheduleSpec,
    max_daily_steps: int,
    known_risk_events: Sequence[Any] = (),
    observed_signal_series: Sequence[float] | None = None,
) -> dict[str, Any]:
    env = make_event_triggered_track_a_env(
        init_frac=low.buffer_fraction,
        init_shifts=low.shifts,
        max_steps=int(max_daily_steps),
        enabled_risks=("R21", "R22", "R23", "R24"),
        risk_overrides=dict(risk_overrides),
        risk_rng_mode="per_risk",
        stochastic_pt=False,
        priming_enabled=False,
        surge_budget_hours=float(max_daily_steps) * DAY * 2.0,
    )
    obs, info = env.reset(seed=int(seed))
    treatment_start = float(env.unwrapped.sim.env.now)
    desired_high = False
    trigger = ObservableEwmaTrigger() if spec.family.startswith("observable_") else None
    shift_hours = surge_hours = buffer_target_unit_hours = 0.0
    signal_position = 0
    try:
        terminated = truncated = False
        while not (terminated or truncated):
            now = float(env.unwrapped.sim.env.now)
            if trigger is None:
                next_high = schedule_is_high(
                    spec,
                    now=now,
                    treatment_start=treatment_start,
                    risk_events=known_risk_events,
                )
            else:
                signal = (
                    float(observed_signal_series[signal_position])
                    if observed_signal_series is not None
                    else _risk_signal(env.unwrapped.sim.risk_events, now)
                )
                pending = list(env.unwrapped.sim.pending_backorders)
                backlog_age = max(
                    (now - float(order.OPTj) for order in pending), default=0.0
                )
                detail = env.unwrapped.sim._inventory_detail()
                target_total = float(sum(env.unwrapped.sim.inventory_buffer_targets.values()))
                actual_total = float(
                    detail.get("raw_material_wdc", 0.0)
                    + detail.get("raw_material_al", 0.0)
                    + detail.get("rations_sb", 0.0)
                )
                shortfall = max(0.0, target_total - actual_total) / max(target_total, 1.0)
                next_high = trigger.decide(
                    now=now,
                    observed_signal=signal,
                    backlog_age_hours=backlog_age,
                    inventory_shortfall_fraction=shortfall,
                )
            posture = high if next_high else low
            action = (
                [1.0, posture.buffer_fraction, posture.shift_signal]
                if next_high != desired_high
                else [-1.0, 0.0, 0.0]
            )
            obs, _reward, terminated, truncated, info = env.step(action)
            desired_high = next_high
            effective = int(info["effective_shift"])
            shift_hours += effective * DAY
            surge_hours += (effective - 1) * DAY
            buffer_target_unit_hours += (
                float(sum(env.unwrapped.sim.inventory_buffer_targets.values())) * DAY
            )
            signal_position += 1

        panel = compute_episode_metrics(
            env.unwrapped.sim, include_temporal_panel=True
        )
        full_trajectory = list(env.action_trajectory)
        trajectory_bytes = json.dumps(
            full_trajectory, sort_keys=True, separators=(",", ":")
        ).encode()
        panel.update(
            {
                "worst_node_or_product_fill": _aggregate_worst_fill(panel),
                "treatment_start": treatment_start,
                "shift_hours": shift_hours,
                "surge_hours": surge_hours,
                "buffer_target_unit_hours": buffer_target_unit_hours,
                "op8_convoy_vehicle_hours": float(
                    getattr(env.unwrapped.sim, "op8_convoy_vehicle_hours", 0.0)
                ),
                "intervention_count": int(env.intervention_count),
                "distinct_intervention_times": len(
                    {
                        row["decision_time"]
                        for row in env.action_trajectory
                        if row["decision"] == "INTERVENE"
                    }
                ),
                "action_trajectory": [
                    row for row in full_trajectory if row["decision"] == "INTERVENE"
                ],
                "action_trajectory_daily_length": len(full_trajectory),
                "action_trajectory_sha256": hashlib.sha256(trajectory_bytes).hexdigest(),
                "realized_risk_events": [
                    {
                        "risk_id": event.risk_id,
                        "start_time": event.start_time,
                        "end_time": event.end_time,
                    }
                    for event in env.unwrapped.sim.risk_events
                ],
            }
        )
        return panel
    finally:
        env.close()


def safe_against(candidate: Mapping[str, Any], comparator: Mapping[str, Any]) -> bool:
    return all(
        float(candidate[key]) >= float(comparator[key]) - 1e-12
        for key in HIGHER_GUARDRAILS
    ) and all(
        float(candidate[key]) <= float(comparator[key]) + 1e-12
        for key in LOWER_GUARDRAILS
    )


def paired_promotion_summary(
    candidate: Sequence[Mapping[str, Any]],
    comparator: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int = 7460999,
    n_bootstrap: int = 10_000,
) -> dict[str, Any]:
    if len(candidate) != len(comparator) or not candidate:
        raise ValueError("paired nonempty candidate/comparator rows are required")
    deltas = np.asarray(
        [float(a["ret_excel"]) - float(b["ret_excel"]) for a, b in zip(candidate, comparator)]
    )
    safe = np.asarray([safe_against(a, b) for a, b in zip(candidate, comparator)])
    rng = np.random.default_rng(int(bootstrap_seed))
    boot = np.asarray(
        [float(np.mean(deltas[rng.integers(0, len(deltas), len(deltas))])) for _ in range(n_bootstrap)]
    )
    lcb = float(np.quantile(boot, 0.05))
    distinct_times = len(
        {
            row["decision_time"]
            for result in candidate
            for row in result.get("action_trajectory", [])
            if row["decision"] == "INTERVENE"
        }
    )
    return {
        "mean_increment": float(np.mean(deltas)),
        "paired_lcb95": lcb,
        "favorable_tapes": int(np.sum(deltas > 0.0)),
        "safe_tapes": int(np.sum(safe)),
        "distinct_intervention_times": int(distinct_times),
        "promotion_pass": bool(
            float(np.mean(deltas)) >= 0.01
            and lcb > 0.0
            and int(np.sum(deltas > 0.0)) >= 34
            and bool(np.all(safe))
            and distinct_times >= 2
        ),
    }
