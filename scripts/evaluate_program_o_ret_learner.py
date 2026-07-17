#!/usr/bin/env python3
"""Evaluate frozen Program O-R learners against complete comparator families."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sb3_contrib import RecurrentPPO  # noqa: E402

from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    MATRIX_KEYS,
    extract_full_des_skeleton,
    full_action_calendars,
    simulate_full_des_frontier,
)
from supply_chain.program_o_ret_env import (  # noqa: E402
    CONFIRMED_RET_CELLS,
    ProgramORetOnlyEnv,
)
from supply_chain.program_o_ret_freeze import verify_execution_freeze  # noqa: E402
from supply_chain.program_o_state_rich import (  # noqa: E402
    finite_state_rich_configurations,
    state_rich_calendar,
)


CONTRACT = ROOT / "contracts/program_o_ret_only_learner_v1.json"


def scheduler() -> dict[str, list[str]]:
    parent = json.loads(
        (ROOT / "contracts/program_o_full_des_hpi_translation_v1.json").read_text()
    )
    key = parent["action"]["primary_scheduler"]
    return parent["action"]["within_week_schedulers"][key]


def model_calendar(model: RecurrentPPO, skeleton, cell_index: int) -> tuple[int, ...]:
    env = ProgramORetOnlyEnv(
        scheduler=scheduler(), tape_seed_start=int(skeleton.seed), tape_seed_end=int(skeleton.seed)
    )
    observation, _info = env.reset(
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
        value = int(np.asarray(action).item())
        actions.append(value)
        observation, _reward, terminated, _truncated, _info = env.step(value)
        episode_start[:] = terminated
    return tuple(actions)


def trajectory_audit(calendars: list[tuple[int, ...]]) -> dict[str, object]:
    counts = Counter(calendars)
    action_counts = Counter(action for row in calendars for action in row)
    varying = sum(len({row[week] for row in calendars}) > 1 for week in range(8))
    return {
        "unique_calendars": len(counts),
        "modal_fraction": max(counts.values()) / len(calendars),
        "varying_weeks": varying,
        "action_counts": dict(sorted(action_counts.items())),
        "passed": len(counts) >= 8 and max(counts.values()) / len(calendars) <= 0.50 and varying >= 2,
    }


GUARDRAIL_KEYS = ("ret_full", "quantity_ret_full", "worst_product_fill")


def simultaneous_bootstrap(rows: dict[str, dict[str, np.ndarray]], resamples: int) -> dict:
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(b"program-o-ret-only-learner-v1").digest()[:8], "big")
    )
    names: list[str] = []
    points: list[float] = []
    estimands_per_cell = 2 + 2 * len(GUARDRAIL_KEYS)
    boot = np.empty((resamples, len(rows) * estimands_per_cell), dtype=float)
    for cell_index, (cell_id, row) in enumerate(rows.items()):
        learner = row["learner"]["ret_visible"]
        open_loop = row["open_loop"]["ret_visible"]
        classical = row["classical"]["ret_visible"]
        open_index = int(np.argmax(open_loop.mean(axis=0)))
        classical_index = int(np.argmax(classical.mean(axis=1)))
        points.extend(
            [
                float(learner.mean() - open_loop[:, open_index].mean()),
                float(learner.mean() - classical[classical_index].mean()),
            ]
        )
        names.extend((f"{cell_id}::H_learned", f"{cell_id}::H_neural"))
        for key in GUARDRAIL_KEYS:
            points.extend(
                [
                    float(row["learner"][key].mean() - row["open_loop"][key][:, open_index].mean()),
                    float(
                        row["learner"][key].mean()
                        - row["classical"][key][classical_index].mean()
                    ),
                ]
            )
            names.extend(
                (f"{cell_id}::{key}::vs_open_loop", f"{cell_id}::{key}::vs_classical")
            )
        for sample in range(resamples):
            tape = rng.integers(0, learner.shape[1], size=learner.shape[1])
            seeds = rng.integers(0, learner.shape[0], size=learner.shape[0])
            learner_mean = learner[np.ix_(seeds, tape)].mean()
            sampled_open_index = int(np.argmax(open_loop[tape].mean(axis=0)))
            sampled_classical_index = int(np.argmax(classical[:, tape].mean(axis=1)))
            offset = cell_index * estimands_per_cell
            boot[sample, offset] = (
                learner_mean - open_loop[tape, sampled_open_index].mean()
            )
            boot[sample, offset + 1] = (
                learner_mean - classical[sampled_classical_index, tape].mean()
            )
            for key_index, key in enumerate(GUARDRAIL_KEYS):
                learner_guardrail = row["learner"][key][np.ix_(seeds, tape)].mean()
                guardrail_offset = offset + 2 + 2 * key_index
                boot[sample, guardrail_offset] = (
                    learner_guardrail
                    - row["open_loop"][key][tape, sampled_open_index].mean()
                )
                boot[sample, guardrail_offset + 1] = (
                    learner_guardrail
                    - row["classical"][key][sampled_classical_index, tape].mean()
                )
    point = np.asarray(points)
    se = boot.std(axis=0, ddof=1)
    active = se > 1e-15
    max_t = np.zeros(resamples)
    if np.any(active):
        max_t[:] = np.max((point[active] - boot[:, active]) / se[active], axis=1)
    critical = float(np.quantile(max_t, 0.95))
    lcb = point - critical * se
    return {
        "method": "two-way learner-seed/tape studentized max-t",
        "resamples": resamples,
        "simultaneous_critical": critical,
        "estimates": {
            name: {"estimate": float(point[i]), "se": float(se[i]), "lcb95": float(lcb[i])}
            for i, name in enumerate(names)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("calibration", "confirmation"), required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    contract = json.loads(CONTRACT.read_text())
    if contract["status"] != "FROZEN_BEFORE_748_SCIENTIFIC_SEEDS":
        raise SystemExit("evaluation blocked until source/execution freeze")
    verify_execution_freeze(ROOT, CONTRACT)
    seed_range = contract["tape_custody"][
        "comparator_calibration" if args.phase == "calibration" else "virgin_confirmation"
    ]
    seeds = list(range(int(seed_range[0]), int(seed_range[1]) + 1))
    learner_seeds = list(map(int, contract["learner"]["learner_seeds"]))
    models = [
        RecurrentPPO.load(args.models / f"recurrent_ppo_seed_{seed}.zip", device="cpu")
        for seed in learner_seeds
    ]
    args.output.mkdir(parents=True)
    result_rows: dict[str, dict[str, np.ndarray]] = {}
    audit_rows: dict[str, object] = {}
    summary_rows: dict[str, object] = {}
    all_calendars = full_action_calendars()
    configs = finite_state_rich_configurations()
    for cell_index, cell in enumerate(CONFIRMED_RET_CELLS):
        open_rows = {key: [] for key in ("ret_visible", *GUARDRAIL_KEYS)}
        classical_metrics = {
            key: np.empty((len(configs), len(seeds)), dtype=float) for key in MATRIX_KEYS
        }
        classical_calendars: list[list[tuple[int, ...]]] = [[] for _ in configs]
        learner_metrics = {
            key: np.empty((len(models), len(seeds)), dtype=float) for key in MATRIX_KEYS
        }
        learner_calendars: list[list[tuple[int, ...]]] = [[] for _ in models]
        cell_dir = args.output / "raw_calendar_matrix" / cell.cell_id
        cell_dir.mkdir(parents=True)
        for tape_index, tape_seed in enumerate(seeds):
            skeleton, _sim = extract_full_des_skeleton(
                seed=tape_seed,
                scheduler=scheduler(),
                regime_persistence=cell.regime_persistence,
                dominant_share=cell.dominant_share,
                downstream_freight_physics_mode="fixed_clock_physical_v1",
            )
            panel = simulate_full_des_frontier(
                skeleton=skeleton, scheduler=scheduler(), calendars=all_calendars
            )
            np.savez_compressed(cell_dir / f"tape_{tape_seed}.npz", **panel)
            for key in open_rows:
                open_rows[key].append(panel[key])
            for config_index, config in enumerate(configs):
                calendar, _decisions = state_rich_calendar(
                    skeleton=skeleton.as_dict(), scheduler=scheduler(), config=config,
                    regime_persistence=0.75,
                    dominant_share=0.90,
                )
                classical_calendars[config_index].append(tuple(calendar))
                metrics = simulate_full_des_frontier(
                    skeleton=skeleton, scheduler=scheduler(),
                    calendars=np.asarray([calendar], dtype=np.uint8),
                )
                for key in MATRIX_KEYS:
                    classical_metrics[key][config_index, tape_index] = metrics[key][0]
            for model_index, model in enumerate(models):
                calendar = model_calendar(model, skeleton, cell_index)
                learner_calendars[model_index].append(calendar)
                metrics = simulate_full_des_frontier(
                    skeleton=skeleton, scheduler=scheduler(),
                    calendars=np.asarray([calendar], dtype=np.uint8),
                )
                for key in MATRIX_KEYS:
                    learner_metrics[key][model_index, tape_index] = metrics[key][0]
        result_rows[cell.cell_id] = {
            "learner": {key: learner_metrics[key] for key in ("ret_visible", *GUARDRAIL_KEYS)},
            "open_loop": {key: np.stack(values) for key, values in open_rows.items()},
            "classical": {
                key: classical_metrics[key] for key in ("ret_visible", *GUARDRAIL_KEYS)
            },
        }
        audit_rows[cell.cell_id] = {
            str(seed): {
                **trajectory_audit(learner_calendars[index]),
                "calendars": [list(row) for row in learner_calendars[index]],
            }
            for index, seed in enumerate(learner_seeds)
        }
        open_ret = result_rows[cell.cell_id]["open_loop"]["ret_visible"]
        classical_ret = classical_metrics["ret_visible"]
        learner_ret = learner_metrics["ret_visible"]
        open_index = int(np.argmax(open_ret.mean(axis=0)))
        classical_index = int(np.argmax(classical_ret.mean(axis=1)))
        learner_by_tape = learner_ret.mean(axis=0)
        open_delta = learner_by_tape - open_ret[:, open_index]
        classical_delta = learner_by_tape - classical_ret[classical_index]
        point_guardrails = {}
        for key in GUARDRAIL_KEYS:
            learner_value = learner_metrics[key].mean(axis=0)
            open_values = result_rows[cell.cell_id]["open_loop"][key][:, open_index]
            point_guardrails[key] = {
                "vs_open_loop": float((learner_value - open_values).mean()),
                "vs_classical": float(
                    (learner_value - classical_metrics[key][classical_index]).mean()
                ),
            }
        summary_rows[cell.cell_id] = {
            "best_open_loop_index": open_index,
            "best_open_loop_calendar": all_calendars[open_index].astype(int).tolist(),
            "best_classical_config": configs[classical_index].config_id,
            "best_classical_calendars": [
                list(calendar) for calendar in classical_calendars[classical_index]
            ],
            "favorable_tapes_vs_open_loop": int(np.sum(open_delta > 0.0)),
            "favorable_tapes_vs_classical": int(np.sum(classical_delta > 0.0)),
            "positive_learner_seeds_vs_both": int(
                sum(
                    learner_ret[index].mean() > open_ret[:, open_index].mean()
                    and learner_ret[index].mean() > classical_ret[classical_index].mean()
                    for index in range(len(models))
                )
            ),
            "point_guardrails": point_guardrails,
            "max_abs_mass_residual": float(np.max(np.abs(learner_metrics["mass_residual"]))),
            "max_abs_partition_residual": float(
                np.max(np.abs(learner_metrics["partition_residual"]))
            ),
            "secondary_means": {
                key: float(learner_metrics[key].mean())
                for key in (
                    "ret_visible_cvar10", "service_loss_auc", "max_backlog_age",
                    "lost_orders", "unresolved_orders", "actual_downstream_vehicle_hours",
                )
            },
        }
    inference = simultaneous_bootstrap(result_rows, args.bootstrap)
    passed = all(
        row["lcb95"] >= (0.01 if name.endswith(("H_learned", "H_neural")) else -0.02)
        for name, row in inference["estimates"].items()
    )
    passed = passed and all(
        row["favorable_tapes_vs_open_loop"] >= 34
        and row["favorable_tapes_vs_classical"] >= 34
        and row["positive_learner_seeds_vs_both"] >= 8
        and row["max_abs_mass_residual"] <= 1e-6
        and row["max_abs_partition_residual"] <= 1e-6
        and all(
            value >= -0.02
            for metric in row["point_guardrails"].values()
            for value in metric.values()
        )
        for row in summary_rows.values()
    )
    result = {
        "schema_version": "program_o_ret_only_learner_evaluation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "seed_range": seed_range,
        "inference": inference,
        "cell_summaries": summary_rows,
        "trajectory_audits": audit_rows,
        "direct_full_des_replay_required_before_terminal_verdict": True,
        "provisional_primary_pass": passed,
        "terminal_verdict": "PENDING_DIRECT_FULL_DES_REPLAY_AND_INTEGRITY_AUDIT",
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
