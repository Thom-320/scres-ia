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
from supply_chain.program_o_eval_custody import (  # noqa: E402
    sha256,
    verify_sha256_manifest,
    write_sha256_manifest,
)
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

# ---- evaluator amendment v1_1 (2026-07-17, frozen BEFORE calibration seed 7480001) -----------
# Closes four audited defects: trajectory audit not gated; modal/phase_only/frequency_matched
# comparators (contract comparators/named) not executed; scheduled_resource_equality_exact not
# verified; confirmation openable without calibration PASS + independent authorization.

RESOURCE_EQUALITY_KEYS = (
    "gross_policy_batch_slots", "gross_production_quantity",
    "charged_daily_dispatch_slots", "charged_downstream_vehicle_hours",
)
PLACEBO_FAMILIES = ("modal", "phase_only", "frequency_matched")
PLACEBO_SEEDS_MINIMUM = 8   # mirrors primary_gates/positive_learner_seeds_minimum
LEDGER_TOLERANCE = 1e-8
DEMAND_LEDGER_IDENTITIES = (
    "generated_orders_policy_invariant",
    "generated_equals_visible_plus_omitted",
    "omitted_rows_equal_unresolved_orders",
    "omitted_quantity_equal_unresolved_quantity",
    "product_remaining_equals_unresolved_quantity",
    "lost_orders_zero_risk_off",
    "lost_quantity_zero_risk_off",
)


def demand_ledger_residuals(metrics: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return the explicit risk-off Program O demand-ledger identities."""
    return {
        "generated_orders_policy_invariant": (
            np.asarray(metrics["generated_orders"])
            - np.asarray(metrics["generated_orders"]).reshape(-1)[0]
        ),
        "generated_equals_visible_plus_omitted": (
            np.asarray(metrics["generated_orders"])
            - np.asarray(metrics["visible_rows"])
            - np.asarray(metrics["omitted_rows"])
        ),
        "omitted_rows_equal_unresolved_orders": (
            np.asarray(metrics["omitted_rows"])
            - np.asarray(metrics["unresolved_orders"])
        ),
        "omitted_quantity_equal_unresolved_quantity": (
            np.asarray(metrics["omitted_quantity"])
            - np.asarray(metrics["unresolved_quantity"])
        ),
        "product_remaining_equals_unresolved_quantity": (
            np.asarray(metrics["remaining_quantity_P_C"])
            + np.asarray(metrics["remaining_quantity_P_H"])
            - np.asarray(metrics["unresolved_quantity"])
        ),
        "lost_orders_zero_risk_off": np.asarray(metrics["lost_orders"]),
        "lost_quantity_zero_risk_off": np.asarray(metrics["lost_quantity"]),
    }


def maximum_ledger_residual(metrics: dict[str, np.ndarray]) -> tuple[float, dict[str, float]]:
    rows = {
        name: float(np.max(np.abs(values)))
        for name, values in demand_ledger_residuals(metrics).items()
    }
    return max(rows.values()), rows


def encode_calendar(calendar: tuple[int, ...]) -> int:
    index = 0
    for action in calendar:
        index = index * 4 + int(action)
    return index


def derive_placebo_calendars(
    calendars: list[tuple[int, ...]], *, rng_seed: int
) -> dict[str, tuple[int, ...]]:
    """Contract comparators modal / phase_only / frequency_matched from realized calendars."""
    counts = Counter(calendars)
    top = max(counts.values())
    modal = min(cal for cal, n in counts.items() if n == top)
    weeks = len(calendars[0])
    phase_only = []
    for week in range(weeks):
        week_counts = Counter(row[week] for row in calendars)
        week_top = max(week_counts.values())
        phase_only.append(min(a for a, n in week_counts.items() if n == week_top))
    pooled = Counter(a for row in calendars for a in row)
    actions = sorted(pooled)
    probs = np.asarray([pooled[a] for a in actions], dtype=float)
    probs /= probs.sum()
    rng = np.random.default_rng(rng_seed)
    frequency_matched = tuple(
        int(actions[i]) for i in rng.choice(len(actions), size=weeks, p=probs)
    )
    return {
        "modal": tuple(modal),
        "phase_only": tuple(phase_only),
        "frequency_matched": frequency_matched,
    }


def compute_provisional_primary_pass(
    *,
    inference: dict,
    summary_rows: dict,
    audit_rows: dict,
    placebo_rows: dict,
    resource_equality: dict,
    demand_preservation: dict,
) -> tuple[bool, dict]:
    """FAIL-CLOSED amendment gate: every component must be populated AND passing.

    Anti-022abd0 property: an absent, empty, or unexecuted component is a FAIL, never a
    default pass. Pinned by tests/test_program_o_ret_learner_evaluator_amendment.py.
    """
    gates: dict[str, bool] = {}
    gates["base_lcb"] = bool(inference.get("estimates")) and all(
        row["lcb95"] >= (0.01 if name.endswith(("H_learned", "H_neural")) else -0.02)
        for name, row in inference["estimates"].items()
    )
    gates["base_cells"] = bool(summary_rows) and all(
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
    gates["trajectory_feedback"] = bool(audit_rows) and all(
        bool(seed_row.get("passed"))
        for cell in audit_rows.values()
        for seed_row in cell.values()
    )
    gates["information_placebos_executed_and_beaten"] = bool(placebo_rows) and all(
        set(cell) == set(PLACEBO_FAMILIES)
        and all(
            family_row.get("executed") is True
            and int(family_row.get("learner_seeds_beating", -1)) >= PLACEBO_SEEDS_MINIMUM
            for family_row in cell.values()
        )
        for cell in placebo_rows.values()
    )
    gates["scheduled_resource_equality_exact"] = (
        resource_equality.get("populated") is True
        and resource_equality.get("full_open_loop_frontier_included") is True
        and float(resource_equality.get("max_abs_diff", float("inf"))) == 0.0
    )
    gates["demand_preservation"] = (
        demand_preservation.get("populated") is True
        and set(demand_preservation.get("identities", {}))
        == set(DEMAND_LEDGER_IDENTITIES)
        and float(demand_preservation.get("max_abs_residual", float("inf")))
        <= LEDGER_TOLERANCE
    )
    return all(gates.values()), gates


def verify_confirmation_preconditions(
    *,
    calibration_result_path: Path | None,
    authorization_path: Path | None,
    contract_path: Path,
    full_des_audit_path: Path | None = None,
    adjudication_path: Path | None = None,
) -> None:
    """Virgin confirmation opens only on a complete, hashed authorization chain."""
    if any(
        path is None
        for path in (
            calibration_result_path,
            full_des_audit_path,
            adjudication_path,
            authorization_path,
        )
    ):
        raise SystemExit(
            "confirmation blocked: --calibration-result, --full-des-audit, "
            "--adjudication and --authorization are all mandatory"
        )
    calibration_result_path = Path(calibration_result_path)
    calibration_root = calibration_result_path.parent
    evaluation_manifest = verify_sha256_manifest(
        calibration_root, calibration_root / "evaluation_files.sha256"
    )
    if "result.json" not in evaluation_manifest:
        raise SystemExit("confirmation blocked: result.json absent from evaluation manifest")
    calibration = json.loads(calibration_result_path.read_text())
    if calibration.get("schema_version") != "program_o_ret_only_learner_evaluation_v1_2":
        raise SystemExit("confirmation blocked: calibration evaluator schema is not v1.2")
    if calibration.get("phase") != "calibration":
        raise SystemExit("confirmation blocked: supplied result is not a calibration result")
    if calibration.get("provisional_primary_pass") is not True:
        raise SystemExit("confirmation blocked: calibration provisional_primary_pass is not True")
    gate_map = calibration.get("amendment_gates") or {}
    if not gate_map or not all(gate_map.values()):
        raise SystemExit(
            f"confirmation blocked: calibration amendment gates not all True: {gate_map}"
        )

    full_des_audit_path = Path(full_des_audit_path)
    audit_manifest = verify_sha256_manifest(
        full_des_audit_path.parent, full_des_audit_path.parent / "audit_files.sha256"
    )
    if full_des_audit_path.name not in audit_manifest:
        raise SystemExit("confirmation blocked: full-DES audit absent from audit manifest")
    full_des_audit = json.loads(full_des_audit_path.read_text())
    calibration_sha = sha256(calibration_result_path)
    if full_des_audit.get("passed") is not True:
        raise SystemExit("confirmation blocked: direct full-DES audit did not pass")
    if full_des_audit.get("phase") != "calibration":
        raise SystemExit("confirmation blocked: full-DES audit must cover calibration")
    if full_des_audit.get("evaluation_result_sha256") != calibration_sha:
        raise SystemExit("confirmation blocked: full-DES audit does not bind calibration")

    adjudication_path = Path(adjudication_path)
    adjudication = json.loads(adjudication_path.read_text())
    direct_audit_sha = sha256(full_des_audit_path)
    if adjudication.get("status") != "ELIGIBLE_FOR_INDEPENDENT_AUTHORIZATION":
        raise SystemExit("confirmation blocked: calibration adjudication is not eligible")
    if adjudication.get("calibration_result_sha256") != calibration_sha:
        raise SystemExit("confirmation blocked: adjudication calibration hash mismatch")
    if adjudication.get("direct_audit_sha256") != direct_audit_sha:
        raise SystemExit("confirmation blocked: adjudication direct-audit hash mismatch")

    authorization = json.loads(Path(authorization_path).read_text())
    contract_sha = hashlib.sha256(Path(contract_path).read_bytes()).hexdigest()
    if authorization.get("authorized_by") != "independent_auditor":
        raise SystemExit(
            "confirmation blocked: authorization must come from the independent auditor"
        )
    if authorization.get("contract_sha256") != contract_sha:
        raise SystemExit("confirmation blocked: authorization contract_sha256 mismatch")
    if authorization.get("calibration_result_sha256") != calibration_sha:
        raise SystemExit("confirmation blocked: authorization calibration_result_sha256 mismatch")
    if authorization.get("direct_audit_sha256") != direct_audit_sha:
        raise SystemExit("confirmation blocked: authorization direct_audit_sha256 mismatch")
    if authorization.get("adjudication_sha256") != sha256(adjudication_path):
        raise SystemExit("confirmation blocked: authorization adjudication_sha256 mismatch")


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
    parser.add_argument("--calibration-result", type=Path, default=None)
    parser.add_argument("--authorization", type=Path, default=None)
    parser.add_argument("--full-des-audit", type=Path, default=None)
    parser.add_argument("--adjudication", type=Path, default=None)
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
    if args.phase == "confirmation":
        verify_confirmation_preconditions(
            calibration_result_path=args.calibration_result,
            authorization_path=args.authorization,
            contract_path=CONTRACT,
            full_des_audit_path=args.full_des_audit,
            adjudication_path=args.adjudication,
        )
    args.output.mkdir(parents=True)
    result_rows: dict[str, dict[str, np.ndarray]] = {}
    audit_rows: dict[str, object] = {}
    summary_rows: dict[str, object] = {}
    placebo_rows: dict[str, object] = {}
    resource_max_abs_diff = 0.0
    resource_equality_populated = False
    demand_identity_max = {name: 0.0 for name in DEMAND_LEDGER_IDENTITIES}
    demand_preservation_populated = False
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
            # ---- amendment v1_2: resource invariance must include the FULL 65,536 frontier,
            # and demand preservation is gated explicitly (rows identity + quantity residuals)
            for key in RESOURCE_EQUALITY_KEYS:
                frontier_column = panel[key]
                anchor = float(frontier_column[0])
                resource_max_abs_diff = max(
                    resource_max_abs_diff,
                    float(frontier_column.max() - frontier_column.min()),
                    float(np.max(np.abs(learner_metrics[key][:, tape_index] - anchor))),
                    float(np.max(np.abs(classical_metrics[key][:, tape_index] - anchor))),
                )
            for family_metrics in (
                panel,
                {key: learner_metrics[key][:, tape_index] for key in MATRIX_KEYS},
                {key: classical_metrics[key][:, tape_index] for key in MATRIX_KEYS},
            ):
                _maximum, identity_rows = maximum_ledger_residual(family_metrics)
                for identity, value in identity_rows.items():
                    demand_identity_max[identity] = max(
                        demand_identity_max[identity], value
                    )
            demand_preservation_populated = True
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
        # ---- amendment v1_1: execute the three contract placebo comparators per learner seed
        open_ret_full = result_rows[cell.cell_id]["open_loop"]["ret_visible"]  # (tapes, 65536)
        cell_placebos: dict[str, dict[str, object]] = {}
        for family in PLACEBO_FAMILIES:
            beating = 0
            per_seed = {}
            for model_index, learner_seed in enumerate(learner_seeds):
                placebos = derive_placebo_calendars(
                    learner_calendars[model_index],
                    rng_seed=int(learner_seed) * 1_000 + cell_index,
                )
                placebo_calendar = placebos[family]
                placebo_mean = float(
                    open_ret_full[:, encode_calendar(placebo_calendar)].mean()
                )
                learner_mean = float(learner_metrics["ret_visible"][model_index].mean())
                beats = bool(learner_mean > placebo_mean)
                beating += int(beats)
                per_seed[str(learner_seed)] = {
                    "calendar": list(placebo_calendar),
                    "placebo_mean_ret_visible": placebo_mean,
                    "learner_mean_ret_visible": learner_mean,
                    "beats": beats,
                }
            cell_placebos[family] = {
                "executed": True,
                "learner_seeds_beating": beating,
                "per_seed": per_seed,
            }
        placebo_rows[cell.cell_id] = cell_placebos
        # ---- amendment v1_1: exact scheduled-resource equality across learner AND classical
        for key in RESOURCE_EQUALITY_KEYS:
            pooled = np.concatenate(
                [learner_metrics[key], classical_metrics[key]], axis=0
            )  # (models+configs, tapes)
            spread = float(np.max(pooled.max(axis=0) - pooled.min(axis=0)))
            resource_max_abs_diff = max(resource_max_abs_diff, spread)
        resource_equality_populated = True

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
    resource_equality = {
        "populated": bool(resource_equality_populated),
        "full_open_loop_frontier_included": True,
        "max_abs_diff": float(resource_max_abs_diff),
        "keys": list(RESOURCE_EQUALITY_KEYS),
        "scope": "learners + classical + FULL 65,536-calendar frontier (amendment v1_2)",
    }
    demand_preservation = {
        "populated": bool(demand_preservation_populated),
        "contract_scope": "risk_off_program_o",
        "tolerance": LEDGER_TOLERANCE,
        "identities": demand_identity_max,
        "max_abs_residual": float(max(demand_identity_max.values())),
    }
    raw_paths = sorted((args.output / "raw_calendar_matrix").rglob("*.npz"))
    expected_raw_count = len(CONFIRMED_RET_CELLS) * len(seeds)
    if len(raw_paths) != expected_raw_count:
        raise RuntimeError(
            f"raw matrix custody incomplete: expected {expected_raw_count}, got {len(raw_paths)}"
        )
    raw_manifest_path = args.output / "raw_files.sha256"
    raw_manifest = write_sha256_manifest(args.output, raw_paths, raw_manifest_path)
    passed, amendment_gates = compute_provisional_primary_pass(
        inference=inference,
        summary_rows=summary_rows,
        audit_rows=audit_rows,
        placebo_rows=placebo_rows,
        resource_equality=resource_equality,
        demand_preservation=demand_preservation,
    )
    result = {
        "schema_version": "program_o_ret_only_learner_evaluation_v1_2",
        "amendment": "evaluator_amendment_v1_2_2026-07-17 (v1_1: trajectory gate, placebo comparators, resource equality, confirmation preconditions; v1_2: full-DES-audit precondition F5, demand-preservation gate, frontier-wide resource invariance, raw-matrix SHA-256 custody manifest)",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "seed_range": seed_range,
        "inference": inference,
        "cell_summaries": summary_rows,
        "trajectory_audits": audit_rows,
        "information_placebos": placebo_rows,
        "scheduled_resource_equality": resource_equality,
        "demand_preservation": demand_preservation,
        "raw_matrix_manifest": raw_manifest_path.name,
        "raw_matrix_manifest_sha256": sha256(raw_manifest_path),
        "raw_matrix_count": len(raw_manifest),
        "raw_matrix_expected_count": expected_raw_count,
        "amendment_gates": amendment_gates,
        "direct_full_des_replay_required_before_terminal_verdict": True,
        "provisional_primary_pass": passed,
        "terminal_verdict": "PENDING_DIRECT_FULL_DES_REPLAY_AND_INTEGRITY_AUDIT",
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_sha256_manifest(
        args.output,
        [result_path, raw_manifest_path, *raw_paths],
        args.output / "evaluation_files.sha256",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
