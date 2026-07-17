"""Development-only helpers for David's Program O-R model workbench.

The helpers deliberately reject all non-949 seed namespaces.  They do not use
the reserved Program O-R calibration or confirmation tapes and cannot emit a
scientific verdict.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd

from supply_chain.program_o_full_des_transducer import (
    extract_full_des_skeleton,
    simulate_full_des_frontier,
)
from supply_chain.program_o_ret_env import (
    CONFIRMED_RET_CELLS,
    ProgramORetCell,
    ProgramORetOnlyEnv,
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
        "files": {
            str(rows_path.name): sha256(rows_path),
            str(summary_path.name): sha256(summary_path),
        },
        "claim_boundary": "Exploratory model selection only; cannot promote Paper 2 or reopen any completed Program O verdict.",
    }
    if model_path is not None and model_path.exists():
        manifest["files"][str(model_path.name)] = sha256(model_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
