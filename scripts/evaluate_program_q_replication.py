#!/usr/bin/env python3
"""Resumable, fail-closed confirmatory evaluator for Program Q.

This module deliberately does not call the Program O-R confirmation path.  Program Q
has a different prospective question (open-loop superiority plus neural premium or
practical equivalence), a different tape namespace, and its own authorization chain.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sb3_contrib import RecurrentPPO  # noqa: E402

from supply_chain.program_o_eval_custody import sha256, write_sha256_manifest  # noqa: E402
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    MATRIX_KEYS,
    extract_full_des_skeleton,
    full_action_calendars,
    simulate_full_des_frontier,
)
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS, ProgramORetOnlyEnv  # noqa: E402
from supply_chain.program_o_state_rich import (  # noqa: E402
    finite_state_rich_configurations,
    state_rich_calendar,
)


CONTRACT = ROOT / "contracts/program_q_frozen_policy_replication_v1.json"
FREEZE = (
    ROOT
    / "research/paper2_exhaustive_search/"
    "program_q_historical_recurrentppo_fallback_freeze_20260717.json"
)
POWER_VERDICT = (
    ROOT
    / "research/paper2_exhaustive_search/"
    "program_q_power_preopen_v5_verdict_20260718.json"
)
SEED_AUDIT = (
    ROOT
    / "research/paper2_exhaustive_search/"
    "program_q_seed_custody_preopen_20260717.json"
)
DIRECT_AUDIT = ROOT / "scripts/audit_program_q_full_des.py"
ADJUDICATOR = ROOT / "scripts/adjudicate_program_q.py"
RUNNER = ROOT / "scripts/run_program_q_confirmation.py"
LAUNCHER = ROOT / "scripts/launch_program_q_confirmation.py"
PRIMARY_KEYS = ("ret_visible",)
GUARDRAIL_KEYS = ("ret_full", "quantity_ret_full", "worst_product_fill")
RESOURCE_KEYS = (
    "gross_policy_batch_slots",
    "gross_production_quantity",
    "charged_daily_dispatch_slots",
    "charged_downstream_vehicle_hours",
)
SECONDARY_KEYS = (
    "ret_visible_cvar10",
    "service_loss_auc",
    "max_backlog_age",
    "lost_orders",
    "unresolved_orders",
    "actual_downstream_vehicle_hours",
)
PLACEBO_FAMILIES = ("modal", "phase_only", "frequency_matched")


def scheduler() -> dict[str, list[str]]:
    parent = json.loads(
        (ROOT / "contracts/program_o_full_des_hpi_translation_v1.json").read_text()
    )
    key = parent["action"]["primary_scheduler"]
    return parent["action"]["within_week_schedulers"][key]


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text())


def _frozen_seeds(contract: Mapping[str, Any]) -> list[int]:
    start, end = map(int, contract["confirmation"]["reserved_block"])
    n = int(contract["confirmation"]["N"])
    seeds = list(range(start, start + n))
    if len(seeds) != n or seeds[-1] > end:
        raise RuntimeError("Program Q confirmation seed block does not contain frozen N")
    return seeds


def verify_model_hashes(models_dir: Path) -> dict[str, str]:
    freeze = json.loads(FREEZE.read_text())
    observed: dict[str, str] = {}
    for seed, expected in freeze["checkpoints_sha256"].items():
        path = models_dir / f"recurrent_ppo_seed_{seed}.zip"
        if not path.is_file():
            raise RuntimeError(f"missing frozen checkpoint: {path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"checkpoint hash mismatch for learner seed {seed}")
        observed[seed] = actual
    return observed


def verify_authorization(path: Path) -> dict[str, Any]:
    """Bind authorization to every mutable pre-open scientific input."""
    payload = json.loads(path.read_text())
    required = {
        "contract_sha256": sha256(CONTRACT),
        "evaluator_sha256": sha256(Path(__file__)),
        "candidate_freeze_sha256": sha256(FREEZE),
        "power_verdict_sha256": sha256(POWER_VERDICT),
        "seed_audit_sha256": sha256(SEED_AUDIT),
        "direct_audit_sha256": sha256(DIRECT_AUDIT),
        "adjudicator_sha256": sha256(ADJUDICATOR),
        "runner_sha256": sha256(RUNNER),
        "launcher_sha256": sha256(LAUNCHER),
    }
    if payload.get("status") != "AUTHORIZED_PROGRAM_Q_CONFIRMATION":
        raise RuntimeError("Program Q confirmation is not independently authorized")
    if payload.get("authorized_by") != "independent_auditor":
        raise RuntimeError("Program Q authorization must be independent")
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"Program Q authorization {key} mismatch")
    return payload


def model_calendar(model: RecurrentPPO, skeleton: Any, cell_index: int) -> tuple[int, ...]:
    env = ProgramORetOnlyEnv(
        scheduler=scheduler(),
        tape_seed_start=int(skeleton.seed),
        tape_seed_end=int(skeleton.seed),
    )
    observation, _ = env.reset(
        options={"skeleton": skeleton, "tape_seed": int(skeleton.seed), "cell_index": cell_index}
    )
    state = None
    episode_start = np.ones((1,), dtype=bool)
    actions: list[int] = []
    terminated = False
    while not terminated:
        action, state = model.predict(
            observation, state=state, episode_start=episode_start, deterministic=True
        )
        actions.append(int(np.asarray(action).item()))
        observation, _, terminated, _, _ = env.step(actions[-1])
        episode_start[:] = terminated
    return tuple(actions)


def encode_calendar(calendar: tuple[int, ...]) -> int:
    value = 0
    for action in calendar:
        value = 4 * value + int(action)
    return value


def trajectory_audit(calendars: list[tuple[int, ...]]) -> dict[str, Any]:
    counts = Counter(calendars)
    varying_weeks = sum(len({row[week] for row in calendars}) > 1 for week in range(8))
    modal_fraction = max(counts.values()) / len(calendars)
    return {
        "unique_calendars": len(counts),
        "modal_fraction": modal_fraction,
        "varying_weeks": varying_weeks,
        "passed": len(counts) >= 8 and modal_fraction <= 0.50 and varying_weeks >= 2,
    }


def derive_replacements(
    calendars: list[tuple[int, ...]], *, rng_seed: int
) -> dict[str, tuple[int, ...]]:
    counts = Counter(calendars)
    top = max(counts.values())
    modal = min(calendar for calendar, count in counts.items() if count == top)
    phase: list[int] = []
    for week in range(8):
        week_counts = Counter(row[week] for row in calendars)
        week_top = max(week_counts.values())
        phase.append(min(action for action, count in week_counts.items() if count == week_top))
    pooled = Counter(action for row in calendars for action in row)
    actions = sorted(pooled)
    probabilities = np.asarray([pooled[action] for action in actions], dtype=float)
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(rng_seed)
    frequency = tuple(
        int(actions[index])
        for index in rng.choice(len(actions), size=8, p=probabilities)
    )
    return {"modal": modal, "phase_only": tuple(phase), "frequency_matched": frequency}


def simultaneous_primary_inference(
    panels: Mapping[str, Mapping[str, np.ndarray]],
    *,
    resamples: int,
    rng_seed: int = 7490257,
) -> dict[str, Any]:
    """Two-way seed x tape max-t with comparator reselection in every draw.

    H_OL uses a simultaneous lower bound. Delta_N uses simultaneous two-sided
    bounds, so equivalence can never be inferred from a non-significant test.
    """
    cells = tuple(panels)
    points: list[float] = []
    names: list[str] = []
    for cell in cells:
        learner = panels[cell]["learner"]
        open_loop = panels[cell]["open_loop"]
        classical = panels[cell]["classical"]
        open_index = int(np.argmax(open_loop.mean(axis=0)))
        classical_index = int(np.argmax(classical.mean(axis=1)))
        points.extend(
            (
                float(learner.mean() - open_loop[:, open_index].mean()),
                float(learner.mean() - classical[classical_index].mean()),
            )
        )
        names.extend((f"{cell}::H_OL", f"{cell}::Delta_N"))
    point = np.asarray(points)
    boot = np.empty((resamples, len(point)), dtype=float)
    rng = np.random.default_rng(rng_seed)
    batch_size = 16
    first = panels[cells[0]]["learner"]
    learner_count, tape_count = first.shape
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        width = stop - start
        tape_indices = rng.integers(0, tape_count, size=(width, tape_count))
        learner_indices = rng.integers(
            0, learner_count, size=(width, learner_count)
        )
        tape_weights = np.zeros((width, tape_count), dtype=float)
        learner_weights = np.zeros((width, learner_count), dtype=float)
        for row in range(width):
            tape_weights[row] = (
                np.bincount(tape_indices[row], minlength=tape_count) / tape_count
            )
            learner_weights[row] = (
                np.bincount(learner_indices[row], minlength=learner_count)
                / learner_count
            )
        for cell_index, cell in enumerate(cells):
            learner = panels[cell]["learner"]
            open_loop = panels[cell]["open_loop"]
            classical = panels[cell]["classical"]
            learner_mean = np.einsum(
                "bs,st,bt->b",
                learner_weights,
                learner,
                tape_weights,
                optimize=True,
            )
            open_mean = tape_weights @ open_loop
            classical_mean = tape_weights @ classical.T
            offset = 2 * cell_index
            boot[start:stop, offset] = learner_mean - open_mean.max(axis=1)
            boot[start:stop, offset + 1] = learner_mean - classical_mean.max(axis=1)
    se = boot.std(axis=0, ddof=1)
    active = se > 1e-15
    statistics = np.zeros((resamples, len(point)), dtype=float)
    statistics[:, active] = (boot[:, active] - point[active]) / se[active]
    # One-sided protection for H_OL; two-sided protection for equivalence Delta_N.
    family = np.column_stack(
        [
            -statistics[:, 0::2],
            np.abs(statistics[:, 1::2]),
        ]
    )
    critical = float(np.quantile(np.max(family, axis=1), 0.95))
    estimates: dict[str, dict[str, float]] = {}
    for index, name in enumerate(names):
        estimates[name] = {
            "point": float(point[index]),
            "se": float(se[index]),
            "lcb95": float(point[index] - critical * se[index]),
            "ucb95": float(point[index] + critical * se[index]),
        }
    return {
        "method": "two-way learner-seed/tape studentized max-t",
        "comparator_reselection_inside_every_resample": True,
        "H_OL_one_sided_and_Delta_N_two_sided": True,
        "resamples": resamples,
        "simultaneous_critical": critical,
        "estimates": estimates,
    }


def simultaneous_guardrail_inference(
    panels: Mapping[str, Mapping[str, np.ndarray]],
    *,
    resamples: int,
    rng_seed: int = 7490258,
) -> dict[str, Any]:
    """Separate one-sided max-t family for the three preregistered guardrails."""
    cells = tuple(panels)
    names: list[str] = []
    points: list[float] = []
    for cell in cells:
        row = panels[cell]
        open_index = int(np.argmax(row["open_ret"].mean(axis=0)))
        classical_index = int(np.argmax(row["classical_ret"].mean(axis=1)))
        for metric in GUARDRAIL_KEYS:
            learner_mean = float(row[f"learner__{metric}"].mean())
            points.extend(
                (
                    learner_mean - float(row[f"open__{metric}"][:, open_index].mean()),
                    learner_mean
                    - float(row[f"classical__{metric}"][classical_index].mean()),
                )
            )
            names.extend(
                (f"{cell}::{metric}::vs_open_loop", f"{cell}::{metric}::vs_classical")
            )
    point = np.asarray(points)
    boot = np.empty((resamples, len(point)), dtype=float)
    first = panels[cells[0]]["learner_ret"]
    learner_count, tape_count = first.shape
    rng = np.random.default_rng(rng_seed)
    batch_size = 16
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        width = stop - start
        tape_indices = rng.integers(0, tape_count, size=(width, tape_count))
        learner_indices = rng.integers(0, learner_count, size=(width, learner_count))
        tape_weights = np.zeros((width, tape_count), dtype=float)
        learner_weights = np.zeros((width, learner_count), dtype=float)
        for row_index in range(width):
            tape_weights[row_index] = (
                np.bincount(tape_indices[row_index], minlength=tape_count) / tape_count
            )
            learner_weights[row_index] = (
                np.bincount(learner_indices[row_index], minlength=learner_count)
                / learner_count
            )
        for cell_index, cell in enumerate(cells):
            row = panels[cell]
            open_ret_means = tape_weights @ row["open_ret"]
            classical_ret_means = tape_weights @ row["classical_ret"].T
            open_indices = np.argmax(open_ret_means, axis=1)
            classical_indices = np.argmax(classical_ret_means, axis=1)
            for metric_index, metric in enumerate(GUARDRAIL_KEYS):
                learner_mean = np.einsum(
                    "bs,st,bt->b",
                    learner_weights,
                    row[f"learner__{metric}"],
                    tape_weights,
                    optimize=True,
                )
                open_means = tape_weights @ row[f"open__{metric}"]
                classical_means = tape_weights @ row[f"classical__{metric}"].T
                offset = cell_index * (2 * len(GUARDRAIL_KEYS)) + 2 * metric_index
                boot[start:stop, offset] = (
                    learner_mean - open_means[np.arange(width), open_indices]
                )
                boot[start:stop, offset + 1] = (
                    learner_mean - classical_means[np.arange(width), classical_indices]
                )
    se = boot.std(axis=0, ddof=1)
    active = se > 1e-15
    studentized = np.zeros_like(boot)
    studentized[:, active] = (point[active] - boot[:, active]) / se[active]
    critical = float(np.quantile(np.max(studentized, axis=1), 0.95))
    return {
        "method": "separate two-way one-sided studentized max-t guardrail family",
        "resamples": resamples,
        "simultaneous_critical": critical,
        "estimates": {
            name: {
                "point": float(point[index]),
                "se": float(se[index]),
                "lcb95": float(point[index] - critical * se[index]),
            }
            for index, name in enumerate(names)
        },
    }


def _atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
    temporary.replace(path)


def produce_shard(
    *,
    cell_index: int,
    tape_seed: int,
    models_dir: Path,
    output: Path,
    allow_development: bool = False,
) -> Path:
    contract = _contract()
    seeds = _frozen_seeds(contract)
    valid_seed = tape_seed in seeds
    if allow_development:
        valid_seed = 949_200_001 <= tape_seed <= 949_299_999
    if not valid_seed or cell_index not in range(len(CONFIRMED_RET_CELLS)):
        raise RuntimeError("requested shard lies outside the frozen Program Q design")
    verify_model_hashes(models_dir)
    cell = CONFIRMED_RET_CELLS[cell_index]
    path = output / cell.cell_id / f"tape_{tape_seed}.npz"
    if path.exists():
        return path
    skeleton, _ = extract_full_des_skeleton(
        seed=tape_seed,
        scheduler=scheduler(),
        regime_persistence=cell.regime_persistence,
        dominant_share=cell.dominant_share,
        downstream_freight_physics_mode="fixed_clock_physical_v1",
    )
    all_calendars = full_action_calendars()
    open_loop = simulate_full_des_frontier(
        skeleton=skeleton, scheduler=scheduler(), calendars=all_calendars
    )
    configs = finite_state_rich_configurations()
    classical = {key: np.empty(len(configs), dtype=float) for key in MATRIX_KEYS}
    classical_calendars = np.empty((len(configs), 8), dtype=np.uint8)
    for index, config in enumerate(configs):
        calendar, _ = state_rich_calendar(
            skeleton=skeleton.as_dict(),
            scheduler=scheduler(),
            config=config,
            regime_persistence=0.75,
            dominant_share=0.90,
        )
        classical_calendars[index] = calendar
        metrics = simulate_full_des_frontier(
            skeleton=skeleton,
            scheduler=scheduler(),
            calendars=np.asarray([calendar], dtype=np.uint8),
        )
        for key in MATRIX_KEYS:
            classical[key][index] = metrics[key][0]
    freeze = json.loads(FREEZE.read_text())
    learner_seeds = list(map(int, freeze["training"]["optimizer_seeds"]))
    learner = {key: np.empty(len(learner_seeds), dtype=float) for key in MATRIX_KEYS}
    learner_calendars = np.empty((len(learner_seeds), 8), dtype=np.uint8)
    for index, learner_seed in enumerate(learner_seeds):
        model = RecurrentPPO.load(
            models_dir / f"recurrent_ppo_seed_{learner_seed}.zip", device="cpu"
        )
        calendar = model_calendar(model, skeleton, cell_index)
        learner_calendars[index] = calendar
        metrics = simulate_full_des_frontier(
            skeleton=skeleton,
            scheduler=scheduler(),
            calendars=np.asarray([calendar], dtype=np.uint8),
        )
        for key in MATRIX_KEYS:
            learner[key][index] = metrics[key][0]
    arrays: dict[str, Any] = {
        "cell_index": np.asarray(cell_index, dtype=np.int16),
        "cell_id": np.asarray(cell.cell_id),
        "tape_seed": np.asarray(tape_seed, dtype=np.int64),
        "skeleton_sha256": np.asarray(skeleton.skeleton_sha256),
        "learner_seeds": np.asarray(learner_seeds, dtype=np.int64),
        "classical_config_ids": np.asarray([config.config_id for config in configs]),
        "learner_calendars": learner_calendars,
        "classical_calendars": classical_calendars,
    }
    for key in MATRIX_KEYS:
        arrays[f"open_loop__{key}"] = open_loop[key]
        arrays[f"classical__{key}"] = classical[key]
        arrays[f"learner__{key}"] = learner[key]
    _atomic_savez(path, **arrays)
    return path


def _demand_residuals(prefix: str, payload: Mapping[str, np.ndarray]) -> list[np.ndarray]:
    def value(key: str) -> np.ndarray:
        return np.asarray(payload[f"{prefix}__{key}"], dtype=float)

    generated = value("generated_orders")
    return [
        generated - np.ravel(generated)[0],
        generated - value("visible_rows") - value("omitted_rows"),
        value("omitted_rows") - value("unresolved_orders"),
        value("omitted_quantity") - value("unresolved_quantity"),
        value("remaining_quantity_P_C")
        + value("remaining_quantity_P_H")
        - value("unresolved_quantity"),
        value("lost_orders"),
        value("lost_quantity"),
    ]


def _expected_shards(shards: Path, seeds: list[int]) -> list[Path]:
    expected = [
        shards / cell.cell_id / f"tape_{seed}.npz"
        for cell in CONFIRMED_RET_CELLS
        for seed in seeds
    ]
    missing = [path for path in expected if not path.is_file()]
    extras = sorted(set(shards.rglob("*.npz")) - set(expected))
    if missing or extras:
        raise RuntimeError(
            f"Program Q shard custody incomplete: missing={len(missing)}, extras={len(extras)}"
        )
    return expected


def reduce_shards(*, shards: Path, output: Path, resamples: int) -> dict[str, Any]:
    """Reduce exactly 768 immutable shards into the frozen Program Q result schema."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    contract = _contract()
    seeds = _frozen_seeds(contract)
    paths = _expected_shards(shards, seeds)
    learner_seeds = list(
        map(int, json.loads(FREEZE.read_text())["training"]["optimizer_seeds"])
    )
    configs = finite_state_rich_configurations()
    primary_panels: dict[str, dict[str, np.ndarray]] = {}
    guardrail_panels: dict[str, dict[str, np.ndarray]] = {}
    calendars_by_cell: dict[str, np.ndarray] = {}
    resource_max_abs_diff = 0.0
    demand_max_abs_residual = 0.0
    mass_max_abs_residual = 0.0
    partition_max_abs_residual = 0.0
    summaries: dict[str, Any] = {}
    trajectory_audits: dict[str, Any] = {}
    replacements: dict[str, Any] = {}
    for cell_index, cell in enumerate(CONFIRMED_RET_CELLS):
        shape_open = (len(seeds), 65_536)
        open_values = {
            key: np.empty(shape_open, dtype=float)
            for key in ("ret_visible", *GUARDRAIL_KEYS)
        }
        classical_values = {
            key: np.empty((len(configs), len(seeds)), dtype=float)
            for key in ("ret_visible", *GUARDRAIL_KEYS)
        }
        learner_values = {
            key: np.empty((len(learner_seeds), len(seeds)), dtype=float)
            for key in ("ret_visible", *GUARDRAIL_KEYS)
        }
        learner_calendars = np.empty(
            (len(learner_seeds), len(seeds), 8), dtype=np.uint8
        )
        classical_calendars = np.empty((len(configs), len(seeds), 8), dtype=np.uint8)
        for tape_index, seed in enumerate(seeds):
            path = shards / cell.cell_id / f"tape_{seed}.npz"
            with np.load(path, allow_pickle=False) as payload:
                if int(payload["cell_index"]) != cell_index or int(payload["tape_seed"]) != seed:
                    raise RuntimeError(f"Program Q shard identity mismatch: {path}")
                if payload["learner_calendars"].shape != (len(learner_seeds), 8):
                    raise RuntimeError(f"Program Q learner calendar shape mismatch: {path}")
                for key in open_values:
                    open_values[key][tape_index] = payload[f"open_loop__{key}"]
                    classical_values[key][:, tape_index] = payload[f"classical__{key}"]
                    learner_values[key][:, tape_index] = payload[f"learner__{key}"]
                learner_calendars[:, tape_index] = payload["learner_calendars"]
                classical_calendars[:, tape_index] = payload["classical_calendars"]
                for key in RESOURCE_KEYS:
                    open_resource = np.asarray(payload[f"open_loop__{key}"], dtype=float)
                    anchor = float(open_resource[0])
                    resource_max_abs_diff = max(
                        resource_max_abs_diff,
                        float(open_resource.max() - open_resource.min()),
                        float(np.max(np.abs(payload[f"classical__{key}"] - anchor))),
                        float(np.max(np.abs(payload[f"learner__{key}"] - anchor))),
                    )
                for prefix in ("open_loop", "classical", "learner"):
                    demand_max_abs_residual = max(
                        demand_max_abs_residual,
                        *(float(np.max(np.abs(row))) for row in _demand_residuals(prefix, payload)),
                    )
                mass_max_abs_residual = max(
                    mass_max_abs_residual,
                    *(float(np.max(np.abs(payload[f"{prefix}__mass_residual"]))) for prefix in ("open_loop", "classical", "learner")),
                )
                partition_max_abs_residual = max(
                    partition_max_abs_residual,
                    *(float(np.max(np.abs(payload[f"{prefix}__partition_residual"]))) for prefix in ("open_loop", "classical", "learner")),
                )
        primary_panels[cell.cell_id] = {
            "learner": learner_values["ret_visible"],
            "open_loop": open_values["ret_visible"],
            "classical": classical_values["ret_visible"],
        }
        guardrail_panels[cell.cell_id] = {
            "learner_ret": learner_values["ret_visible"],
            "open_ret": open_values["ret_visible"],
            "classical_ret": classical_values["ret_visible"],
            **{f"learner__{key}": learner_values[key] for key in GUARDRAIL_KEYS},
            **{f"open__{key}": open_values[key] for key in GUARDRAIL_KEYS},
            **{f"classical__{key}": classical_values[key] for key in GUARDRAIL_KEYS},
        }
        calendars_by_cell[cell.cell_id] = learner_calendars
        open_index = int(np.argmax(open_values["ret_visible"].mean(axis=0)))
        classical_index = int(np.argmax(classical_values["ret_visible"].mean(axis=1)))
        learner_by_tape = learner_values["ret_visible"].mean(axis=0)
        open_delta = learner_by_tape - open_values["ret_visible"][:, open_index]
        summaries[cell.cell_id] = {
            "best_open_loop_index": open_index,
            "best_open_loop_calendar": full_action_calendars()[open_index].astype(int).tolist(),
            "best_classical_config": configs[classical_index].config_id,
            "best_classical_calendars": classical_calendars[classical_index].astype(int).tolist(),
            "favorable_tapes_fraction_vs_open_loop": float(np.mean(open_delta > 0.0)),
            "positive_learner_seeds_H_OL": int(
                np.sum(
                    learner_values["ret_visible"].mean(axis=1)
                    > open_values["ret_visible"][:, open_index].mean()
                )
            ),
        }
        cell_audits: dict[str, Any] = {}
        cell_replacements: dict[str, Any] = {
            family: {"executed": True, "learner_seeds_beating": 0, "per_seed": {}}
            for family in PLACEBO_FAMILIES
        }
        for learner_index, learner_seed in enumerate(learner_seeds):
            calendars = [tuple(map(int, row)) for row in learner_calendars[learner_index]]
            cell_audits[str(learner_seed)] = {
                **trajectory_audit(calendars),
                "calendars": [list(row) for row in calendars],
            }
            derived = derive_replacements(
                calendars, rng_seed=learner_seed * 1_000 + cell_index
            )
            learner_mean = float(learner_values["ret_visible"][learner_index].mean())
            for family, calendar in derived.items():
                replacement_mean = float(
                    open_values["ret_visible"][:, encode_calendar(calendar)].mean()
                )
                beats = learner_mean > replacement_mean
                cell_replacements[family]["learner_seeds_beating"] += int(beats)
                cell_replacements[family]["per_seed"][str(learner_seed)] = {
                    "calendar": list(calendar),
                    "learner_mean": learner_mean,
                    "replacement_mean": replacement_mean,
                    "beats": beats,
                }
        trajectory_audits[cell.cell_id] = cell_audits
        replacements[cell.cell_id] = cell_replacements
    primary_inference = simultaneous_primary_inference(
        primary_panels, resamples=resamples
    )
    guardrail_inference = simultaneous_guardrail_inference(
        guardrail_panels, resamples=resamples
    )
    margin = float(contract["class_b_integrity_gates"]["ret_full_margin"])
    feedback_pass = all(
        row["passed"]
        for cell in trajectory_audits.values()
        for row in cell.values()
    )
    replacement_pass = all(
        family["learner_seeds_beating"] >= 8
        for cell in replacements.values()
        for family in cell.values()
    )
    guardrail_pass = {
        key: all(
            estimate["lcb95"] >= margin
            for name, estimate in guardrail_inference["estimates"].items()
            if f"::{key}::" in name
        )
        for key in GUARDRAIL_KEYS
    }
    integrity = {
        "feedback": feedback_pass,
        "replacement_controls": replacement_pass,
        "scheduled_resources_exact": resource_max_abs_diff == 0.0,
        "mass_partition_demand": (
            mass_max_abs_residual <= 1e-6
            and partition_max_abs_residual <= 1e-6
            and demand_max_abs_residual <= 1e-8
        ),
        "ret_full_noninferior": guardrail_pass["ret_full"],
        "quantity_ret_full_noninferior": guardrail_pass["quantity_ret_full"],
        "worst_product_fill_noninferior": guardrail_pass["worst_product_fill"],
    }
    result = {
        "schema_version": "program_q_frozen_policy_replication_evaluation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": sha256(CONTRACT),
        "seed_range": [seeds[0], seeds[-1]],
        "N": len(seeds),
        "inference": primary_inference,
        "guardrail_inference": guardrail_inference,
        "cell_summaries": summaries,
        "trajectory_audits": trajectory_audits,
        "replacement_controls": replacements,
        "integrity_diagnostics": {
            "resource_max_abs_diff": resource_max_abs_diff,
            "demand_max_abs_residual": demand_max_abs_residual,
            "mass_max_abs_residual": mass_max_abs_residual,
            "partition_max_abs_residual": partition_max_abs_residual,
        },
        "integrity_gates": integrity,
        "direct_full_des_replay_required": True,
        "terminal_verdict": "PENDING_DIRECT_FULL_DES_REPLAY_AND_ADJUDICATION",
        "shard_count": len(paths),
    }
    output.mkdir(parents=True)
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    shard_manifest_path = shards / "shard_files.sha256"
    if shard_manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {shard_manifest_path}")
    write_sha256_manifest(shards, paths, shard_manifest_path)
    shard_manifest_copy = output / "shard_files.sha256"
    shard_manifest_copy.write_text(shard_manifest_path.read_text())
    write_sha256_manifest(
        output,
        [result_path, shard_manifest_copy],
        output / "evaluation_files.sha256",
    )
    return result


def write_plan(output: Path) -> None:
    contract = _contract()
    seeds = _frozen_seeds(contract)
    plan = {
        "schema_version": "program_q_confirmation_execution_plan_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": sha256(CONTRACT),
        "evaluator_sha256": sha256(Path(__file__)),
        "candidate_freeze_sha256": sha256(FREEZE),
        "power_verdict_sha256": sha256(POWER_VERDICT),
        "seed_audit_sha256": sha256(SEED_AUDIT),
        "direct_audit_sha256": sha256(DIRECT_AUDIT),
        "adjudicator_sha256": sha256(ADJUDICATOR),
        "runner_sha256": sha256(RUNNER),
        "launcher_sha256": sha256(LAUNCHER),
        "N": len(seeds),
        "seed_range": [seeds[0], seeds[-1]],
        "cells": [cell.cell_id for cell in CONFIRMED_RET_CELLS],
        "expected_shards": len(seeds) * len(CONFIRMED_RET_CELLS),
        "learner": "ten frozen historical RecurrentPPO final checkpoints",
        "external_collaborator_dependency": False,
        "open_loop_family_size": 65_536,
        "classical_family_size": len(finite_state_rich_configurations()),
        "scientific_seeds_opened_by_plan": 0,
        "status": "READY_FOR_INDEPENDENT_PREOPEN_AUDIT_NOT_AUTHORIZED_TO_OPEN",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--output", type=Path, required=True)
    shard = subparsers.add_parser("produce-shard")
    shard.add_argument("--cell-index", type=int, required=True)
    shard.add_argument("--tape-seed", type=int, required=True)
    shard.add_argument("--models", type=Path, required=True)
    shard.add_argument("--output", type=Path, required=True)
    shard.add_argument("--authorization", type=Path, required=True)
    smoke = subparsers.add_parser("smoke-shard")
    smoke.add_argument("--cell-index", type=int, required=True)
    smoke.add_argument("--tape-seed", type=int, required=True)
    smoke.add_argument("--models", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    reduce_parser = subparsers.add_parser("reduce")
    reduce_parser.add_argument("--shards", type=Path, required=True)
    reduce_parser.add_argument("--output", type=Path, required=True)
    reduce_parser.add_argument("--bootstrap", type=int, default=10_000)
    reduce_parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        write_plan(args.output)
        return 0
    if args.command == "smoke-shard":
        path = produce_shard(
            cell_index=args.cell_index,
            tape_seed=args.tape_seed,
            models_dir=args.models,
            output=args.output,
            allow_development=True,
        )
        print(path)
        return 0
    verify_authorization(args.authorization)
    if args.command == "reduce":
        reduce_shards(
            shards=args.shards, output=args.output, resamples=args.bootstrap
        )
        return 0
    path = produce_shard(
        cell_index=args.cell_index,
        tape_seed=args.tape_seed,
        models_dir=args.models,
        output=args.output,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
