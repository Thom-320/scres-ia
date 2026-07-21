"""Observation saturation and cross-tape aliasing diagnostics for Program Q2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from supply_chain.program_o_full_des_transducer import FullDESSkeleton
from supply_chain.program_o_state_rich import StateRichObservation


Q21_NAMES = (
    "on_hand_c", "on_hand_h",
    "locked_pipeline_c", "locked_pipeline_h",
    "backlog_quantity_c", "backlog_quantity_h",
    "backlog_orders_c", "backlog_orders_h",
    "max_backlog_age_c", "max_backlog_age_h",
    "in_flight_c", "in_flight_h",
    "belief_c", "predicted_share_c",
    "prev_action_0", "prev_action_1", "prev_action_2", "prev_action_3", "prev_action_none",
    "week", "remaining_decisions",
)


@dataclass(frozen=True)
class ObservationAuditRow:
    seed: int
    week: int
    q21_preclip: tuple[float, ...]
    q21_clipped: tuple[float, ...]
    rich: tuple[float, ...]
    best_action: int
    action_ret: tuple[float, ...]


def q21_preclip(observation: StateRichObservation) -> np.ndarray:
    values: list[float] = []
    for field in (
        observation.on_hand,
        observation.locked_pipeline,
        observation.backlog_quantity,
    ):
        values.extend(float(value) / 120_000.0 for value in field)
    values.extend(float(value) / 48.0 for value in observation.backlog_orders)
    values.extend(float(value) / 1_344.0 for value in observation.max_backlog_age)
    values.extend(float(value) / 120_000.0 for value in observation.in_flight_quantity)
    values.extend((float(observation.belief_c), float(observation.predicted_share_c)))
    previous = np.zeros(5, dtype=float)
    previous[4 if observation.previous_action is None else int(observation.previous_action)] = 1.0
    values.extend(previous.tolist())
    values.extend((float(observation.week) / 7.0, float(observation.remaining_decisions) / 8.0))
    vector = np.asarray(values, dtype=float)
    if vector.shape != (21,):
        raise AssertionError(f"Q21 schema drift: {vector.shape}")
    return vector


def deployable_history_features(
    *, skeleton: FullDESSkeleton, observation: StateRichObservation
) -> np.ndarray:
    now = float(observation.decision_time)
    times = np.asarray(skeleton.order_times, dtype=float)
    quantities = np.asarray(skeleton.order_quantities, dtype=float)
    products = np.asarray([0 if value == "P_C" else 1 for value in skeleton.order_products])
    features: list[float] = []
    for weeks in (1, 2, 4):
        visible = (times < now - 1e-12) & (times >= now - 168.0 * weeks)
        for product in (0, 1):
            selected = visible & (products == product)
            features.append(float(selected.sum()) / max(1.0, 6.0 * weeks))
            features.append(float(quantities[selected].sum()) / max(1.0, 15_000.0 * weeks))
    arrivals = np.asarray([float(row[0]) for row in skeleton.batch_arrivals], dtype=float)
    for horizon in (168.0, 336.0):
        features.append(float(((arrivals >= now) & (arrivals < now + horizon)).sum()) / 6.0)
    return np.asarray(features, dtype=float)


def make_audit_row(
    *,
    seed: int,
    skeleton: FullDESSkeleton,
    observation: StateRichObservation,
    best_action: int,
    action_ret: Sequence[float],
) -> ObservationAuditRow:
    preclip = q21_preclip(observation)
    history = deployable_history_features(skeleton=skeleton, observation=observation)
    return ObservationAuditRow(
        seed=int(seed),
        week=int(observation.week),
        q21_preclip=tuple(map(float, preclip)),
        q21_clipped=tuple(map(float, np.clip(preclip, 0.0, 1.0))),
        rich=tuple(map(float, np.concatenate((preclip, history)))),
        best_action=int(best_action),
        action_ret=tuple(map(float, action_ret)),
    )


def _robust_scale(matrix: np.ndarray) -> np.ndarray:
    median = np.median(matrix, axis=0)
    q25 = np.quantile(matrix, 0.25, axis=0)
    q75 = np.quantile(matrix, 0.75, axis=0)
    scale = np.where(q75 - q25 > 1e-9, q75 - q25, 1.0)
    return (matrix - median) / scale


def cross_tape_nearest_regret(rows: Sequence[ObservationAuditRow], field: str) -> float:
    matrix = np.asarray([getattr(row, field) for row in rows], dtype=float)
    matrix = _robust_scale(matrix) if field == "rich" else matrix
    regrets = []
    for index, row in enumerate(rows):
        eligible = np.asarray([candidate.seed != row.seed for candidate in rows], dtype=bool)
        distances = np.square(matrix - matrix[index]).sum(axis=1)
        distances[~eligible] = np.inf
        donor_index = int(np.argmin(distances))
        donor_action = int(rows[donor_index].best_action)
        regrets.append(max(row.action_ret) - row.action_ret[donor_action])
    return float(np.mean(regrets))


def exact_collision_summary(rows: Sequence[ObservationAuditRow]) -> dict[str, int]:
    groups: dict[tuple[float, ...], set[int]] = {}
    for row in rows:
        key = tuple(round(value, 8) for value in row.q21_clipped)
        groups.setdefault(key, set()).add(int(row.best_action))
    ambiguous = [actions for actions in groups.values() if len(actions) > 1]
    return {
        "unique_vectors": len(groups),
        "ambiguous_collision_groups": len(ambiguous),
        "actions_in_ambiguous_groups": sum(len(actions) for actions in ambiguous),
    }
