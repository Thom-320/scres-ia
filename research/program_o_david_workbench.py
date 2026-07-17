"""Development-only helpers for David's Program O-R model workbench.

The helpers deliberately reject all non-949 seed namespaces.  They do not use
the reserved Program O-R calibration or confirmation tapes and cannot emit a
scientific verdict.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd

from supply_chain.program_o_full_des_transducer import (
    extract_full_des_skeleton,
    full_action_calendars,
    simulate_full_des_frontier,
)
from supply_chain.program_o_ret_env import (
    CONFIRMED_RET_CELLS,
    ProgramORetCell,
    ProgramORetOnlyEnv,
)
from supply_chain.program_o_state_rich import (
    finite_state_rich_configurations,
    state_rich_calendar,
)
from scripts.evaluate_program_o_ret_learner import (
    GUARDRAIL_KEYS,
    RESOURCE_EQUALITY_KEYS,
    derive_placebo_calendars,
    encode_calendar,
    trajectory_audit,
)


DEV_SEED_MIN = 949_100_001
DEV_SEED_MAX = 949_999_999
IMPORTANT_METRICS = (
    "ret_visible",
    "ret_full",
    "quantity_ret_full",
    "worst_product_fill",
    "gross_production_quantity",
    "charged_downstream_vehicle_hours",
    "generated_orders",
    "lost_orders",
    "unresolved_orders",
    "mass_residual",
    "partition_residual",
    "ret_visible_cvar10",
)


class WorkbenchPolicy(Protocol):
    """Small common API implemented by notebook model adapters."""

    label: str

    def reset_policy_state(self) -> None: ...

    def predict_action(self, observation: np.ndarray) -> int: ...


@dataclass(frozen=True)
class FullFrontierEvaluation:
    """Development scoreboard against the exact O-R comparator families."""

    policy_rows: pd.DataFrame
    scoreboard: pd.DataFrame
    diagnostics: dict[str, Any]


def assert_development_seed(seed: int) -> int:
    """Reject scientific and historical namespaces by construction."""
    value = int(seed)
    if not DEV_SEED_MIN <= value <= DEV_SEED_MAX:
        raise ValueError(
            f"seed {value} is outside the development-only 949 namespace "
            f"[{DEV_SEED_MIN}, {DEV_SEED_MAX}]"
        )
    return value


def load_scheduler(root: Path) -> dict[str, list[str]]:
    contract = json.loads(
        (root / "contracts/program_o_full_des_hpi_translation_v1.json").read_text()
    )
    key = contract["action"]["primary_scheduler"]
    return contract["action"]["within_week_schedulers"][key]


def make_development_env(
    *,
    root: Path,
    tape_seed_start: int,
    tape_seed_end: int,
    cells: Sequence[ProgramORetCell] = CONFIRMED_RET_CELLS,
) -> ProgramORetOnlyEnv:
    start = assert_development_seed(tape_seed_start)
    end = assert_development_seed(tape_seed_end)
    if end < start:
        raise ValueError("tape_seed_end must not precede tape_seed_start")
    return ProgramORetOnlyEnv(
        scheduler=load_scheduler(root),
        tape_seed_start=start,
        tape_seed_end=end,
        cells=cells,
    )


def rollout_calendar(policy: WorkbenchPolicy, env: ProgramORetOnlyEnv, *, seed: int, cell_index: int) -> tuple[int, ...]:
    """Roll out one deterministic eight-action calendar."""
    assert_development_seed(seed)
    observation, _ = env.reset(
        options={"tape_seed": int(seed), "cell_index": int(cell_index)}
    )
    policy.reset_policy_state()
    calendar: list[int] = []
    terminated = False
    while not terminated:
        action = int(policy.predict_action(observation))
        if action not in range(4):
            raise ValueError(f"policy returned invalid action {action}")
        calendar.append(action)
        observation, _reward, terminated, truncated, _info = env.step(action)
        if truncated:
            raise AssertionError("Program O-R workbench episode truncated")
    return tuple(calendar)


def evaluate_policy(
    *,
    root: Path,
    policy: WorkbenchPolicy,
    seeds: Sequence[int],
    cells: Sequence[ProgramORetCell] = CONFIRMED_RET_CELLS,
) -> pd.DataFrame:
    """Evaluate one model on the same development tapes in all three cells."""
    checked = [assert_development_seed(seed) for seed in seeds]
    scheduler = load_scheduler(root)
    rows: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(cells):
        env = make_development_env(
            root=root,
            tape_seed_start=min(checked),
            tape_seed_end=max(checked),
            cells=cells,
        )
        for seed in checked:
            skeleton, _sim = extract_full_des_skeleton(
                seed=seed,
                scheduler=scheduler,
                regime_persistence=cell.regime_persistence,
                dominant_share=cell.dominant_share,
                downstream_freight_physics_mode="fixed_clock_physical_v1",
            )
            observation, _ = env.reset(
                options={
                    "skeleton": skeleton,
                    "tape_seed": seed,
                    "cell_index": cell_index,
                }
            )
            policy.reset_policy_state()
            actions: list[int] = []
            terminated = False
            while not terminated:
                action = int(policy.predict_action(observation))
                if action not in range(4):
                    raise ValueError(f"policy returned invalid action {action}")
                actions.append(action)
                observation, _reward, terminated, truncated, _ = env.step(action)
                if truncated:
                    raise AssertionError("unexpected truncation")
            metrics = simulate_full_des_frontier(
                skeleton=skeleton,
                scheduler=scheduler,
                calendars=np.asarray([actions], dtype=np.uint8),
            )
            row = {
                "model": policy.label,
                "cell": cell.cell_id,
                "tape_seed": seed,
                "calendar": "".join(map(str, actions)),
                **{key: float(metrics[key][0]) for key in IMPORTANT_METRICS},
            }
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_policy_against_full_frontiers(
    *,
    root: Path,
    policy: WorkbenchPolicy,
    seeds: Sequence[int],
    cells: Sequence[ProgramORetCell] = CONFIRMED_RET_CELLS,
    placebo_seed: int = 949_299_999,
) -> FullFrontierEvaluation:
    """Evaluate a policy against all 65,536 calendars and ten classical rules.

    Comparator selection follows the scientific estimand: select the calendar
    and classical configuration with the highest *mean across tapes*.  It never
    takes a post-hoc maximum separately on each tape.
    """
    checked = [assert_development_seed(seed) for seed in seeds]
    if not checked:
        raise ValueError("at least one development evaluation tape is required")
    scheduler = load_scheduler(root)
    all_calendars = full_action_calendars()
    configurations = finite_state_rich_configurations()
    policy_rows: list[dict[str, Any]] = []
    scoreboard_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}

    for cell_index, cell in enumerate(cells):
        env = make_development_env(
            root=root,
            tape_seed_start=min(checked),
            tape_seed_end=max(checked),
            cells=cells,
        )
        frontier = {
            key: np.empty((len(checked), len(all_calendars)), dtype=np.float64)
            for key in ("ret_visible", *GUARDRAIL_KEYS)
        }
        classical = {
            key: np.empty((len(configurations), len(checked)), dtype=np.float64)
            for key in ("ret_visible", *GUARDRAIL_KEYS)
        }
        policy_metric = {
            key: np.empty(len(checked), dtype=np.float64)
            for key in IMPORTANT_METRICS
        }
        calendars: list[tuple[int, ...]] = []
        resource_max_spread = 0.0

        for tape_index, seed in enumerate(checked):
            skeleton, _sim = extract_full_des_skeleton(
                seed=seed,
                scheduler=scheduler,
                regime_persistence=cell.regime_persistence,
                dominant_share=cell.dominant_share,
                downstream_freight_physics_mode="fixed_clock_physical_v1",
            )
            panel = simulate_full_des_frontier(
                skeleton=skeleton,
                scheduler=scheduler,
                calendars=all_calendars,
            )
            for key in frontier:
                frontier[key][tape_index] = panel[key]
            for key in RESOURCE_EQUALITY_KEYS:
                resource_max_spread = max(
                    resource_max_spread,
                    float(np.max(panel[key]) - np.min(panel[key])),
                )

            observation, _ = env.reset(
                options={
                    "skeleton": skeleton,
                    "tape_seed": seed,
                    "cell_index": cell_index,
                }
            )
            policy.reset_policy_state()
            actions: list[int] = []
            terminated = False
            while not terminated:
                action = int(policy.predict_action(observation))
                if action not in range(4):
                    raise ValueError(f"policy returned invalid action {action}")
                actions.append(action)
                observation, _reward, terminated, truncated, _ = env.step(action)
                if truncated:
                    raise AssertionError("unexpected truncation")
            calendar = tuple(actions)
            calendars.append(calendar)
            policy_index = encode_calendar(calendar)
            row = {
                "model": policy.label,
                "cell": cell.cell_id,
                "tape_seed": seed,
                "calendar": "".join(map(str, calendar)),
            }
            for key in IMPORTANT_METRICS:
                value = float(panel[key][policy_index])
                policy_metric[key][tape_index] = value
                row[key] = value
            policy_rows.append(row)

            for config_index, config in enumerate(configurations):
                classical_calendar, _ = state_rich_calendar(
                    skeleton=skeleton.as_dict(),
                    scheduler=scheduler,
                    config=config,
                    regime_persistence=0.75,
                    dominant_share=0.90,
                )
                classical_index = encode_calendar(tuple(classical_calendar))
                for key in classical:
                    classical[key][config_index, tape_index] = panel[key][classical_index]

        open_index = int(np.argmax(frontier["ret_visible"].mean(axis=0)))
        classical_index = int(np.argmax(classical["ret_visible"].mean(axis=1)))
        best_open = frontier["ret_visible"][:, open_index]
        best_classical = classical["ret_visible"][classical_index]
        policy_ret = policy_metric["ret_visible"]
        audit = trajectory_audit(calendars)
        placebos = derive_placebo_calendars(
            calendars, rng_seed=int(placebo_seed) + cell_index
        )
        placebo_means = {
            name: float(frontier["ret_visible"][:, encode_calendar(calendar)].mean())
            for name, calendar in placebos.items()
        }
        selected_config = configurations[classical_index]
        score = {
            "model": policy.label,
            "cell": cell.cell_id,
            "mean_ret": float(policy_ret.mean()),
            "best_open_loop_ret": float(best_open.mean()),
            "best_classical_ret": float(best_classical.mean()),
            "H_learned": float((policy_ret - best_open).mean()),
            "H_neural": float((policy_ret - best_classical).mean()),
            "favorable_vs_open_loop": int(np.sum(policy_ret > best_open)),
            "favorable_vs_classical": int(np.sum(policy_ret > best_classical)),
            "best_open_loop_calendar": "".join(map(str, all_calendars[open_index])),
            "best_classical": selected_config.config_id,
            "unique_calendars": int(audit["unique_calendars"]),
            "modal_fraction": float(audit["modal_fraction"]),
            "varying_weeks": int(audit["varying_weeks"]),
            "feedback_audit_passed": bool(audit["passed"]),
            "scheduled_resource_max_spread": float(resource_max_spread),
        }
        for name, mean in placebo_means.items():
            score[f"delta_vs_{name}"] = float(policy_ret.mean() - mean)
        for key in GUARDRAIL_KEYS:
            score[f"{key}_vs_open_loop"] = float(
                policy_metric[key].mean() - frontier[key][:, open_index].mean()
            )
            score[f"{key}_vs_classical"] = float(
                policy_metric[key].mean() - classical[key][classical_index].mean()
            )
        scoreboard_rows.append(score)
        diagnostics[cell.cell_id] = {
            "trajectory_audit": audit,
            "placebo_calendars": {key: list(value) for key, value in placebos.items()},
            "placebo_mean_ret": placebo_means,
            "best_open_loop_index": open_index,
            "best_open_loop_calendar": list(map(int, all_calendars[open_index])),
            "best_classical_index": classical_index,
            "best_classical": selected_config.config_id,
            "comparator_rule": "maximum family mean across tapes; never per-tape maximum",
        }

    return FullFrontierEvaluation(
        policy_rows=pd.DataFrame(policy_rows),
        scoreboard=pd.DataFrame(scoreboard_rows),
        diagnostics=diagnostics,
    )


def swap_product_channels(observation: np.ndarray, *, obs_dim: int = 21) -> np.ndarray:
    """Swap P_C/P_H channels in one observation or a flattened causal history."""
    value = np.asarray(observation, dtype=np.float32)
    if value.ndim != 1 or value.size % int(obs_dim):
        raise ValueError("observation must be a flat sequence of 21D frames")
    output = value.reshape(-1, int(obs_dim)).copy()
    for frame in output:
        for left, right in ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)):
            frame[left], frame[right] = frame[right], frame[left]
        frame[12] = 1.0 - frame[12]
        frame[13] = 1.0 - frame[13]
        previous = frame[14:19].copy()
        frame[14:19] = previous[[3, 2, 1, 0, 4]]
    return output.reshape(value.shape)


def compact_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Return only the endpoint, guardrails, and integrity fields David needs."""
    metrics = [key for key in IMPORTANT_METRICS if key in rows.columns]
    summary = rows.groupby(["model", "cell"], as_index=False)[metrics].mean()
    unique = (
        rows.groupby(["model", "cell"])["calendar"].nunique().rename("unique_calendars")
    )
    modal = (
        rows.groupby(["model", "cell"])["calendar"]
        .apply(lambda values: float(values.value_counts(normalize=True).iloc[0]))
        .rename("modal_fraction")
    )
    return summary.merge(unique, on=["model", "cell"]).merge(
        modal, on=["model", "cell"]
    )


def integrity_report(rows: pd.DataFrame) -> dict[str, Any]:
    return {
        "max_abs_mass_residual": float(rows["mass_residual"].abs().max()),
        "max_abs_partition_residual": float(rows["partition_residual"].abs().max()),
        "gross_production_values": sorted(
            map(float, rows["gross_production_quantity"].unique())
        ),
        "charged_vehicle_hour_values": sorted(
            map(float, rows["charged_downstream_vehicle_hours"].unique())
        ),
        "passed": bool(
            rows["mass_residual"].abs().max() <= 1e-6
            and rows["partition_residual"].abs().max() <= 1e-6
            and rows["gross_production_quantity"].nunique() == 1
            and rows["charged_downstream_vehicle_hours"].nunique() == 1
        ),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_experiment_bundle(
    *,
    output_dir: Path,
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    model_path: Path | None,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "evaluation_rows.csv"
    summary_path = output_dir / "summary.csv"
    manifest_path = output_dir / "manifest.json"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "schema_version": "program_o_david_workbench_run_v1",
        "status": "DEVELOPMENT_ONLY_NO_SCIENTIFIC_PROMOTION",
        "configuration": configuration,
        "cells": [asdict(cell) for cell in CONFIRMED_RET_CELLS],
        "seed_namespace": [DEV_SEED_MIN, DEV_SEED_MAX],
        "integrity": integrity_report(rows),
        "files": {},
        "claim_boundary": "Exploratory model selection only; cannot promote Paper 2 or reopen any completed Program O verdict.",
    }
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path != manifest_path:
            manifest["files"][str(path.name)] = sha256(path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
