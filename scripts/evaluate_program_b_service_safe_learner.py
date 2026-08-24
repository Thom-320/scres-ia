#!/usr/bin/env python3
"""Evaluate Program B service-safe learners on the burned validation tapes."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supply_chain.program_o_full_des import run_program_o_full_des_episode  # noqa: E402
from supply_chain.program_o_full_des_transducer import (  # noqa: E402
    direct_full_des_vector,
    extract_full_des_skeleton,
    simulate_full_des_frontier,
)
from supply_chain.program_o_ret_env import (  # noqa: E402
    CONFIRMED_RET_CELLS,
    ProgramORetOnlyEnv,
    compute_service_safe_reward,
)

SCHEDULER_CONTRACT = ROOT / "contracts/program_o_full_des_hpi_translation_v1.json"
VALIDATION_ROOT = Path(
    "/home/ubuntu/program_o_runs/program-o-full-des-validation-v2-20260715/artifacts/validation"
)
DEVELOPMENT_RESULT = ROOT / "results/program_b_gate_v2/development_service_safe.json"


def scheduler() -> dict[str, list[str]]:
    parent = json.loads(SCHEDULER_CONTRACT.read_text())
    key = parent["action"]["primary_scheduler"]
    return parent["action"]["within_week_schedulers"][key]


def bootstrap(values: np.ndarray, seed: int) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(20000, len(values)))
    means = values[draws].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "lcb95_one_sided": float(np.quantile(means, 0.05)),
        "ucb95_one_sided": float(np.quantile(means, 0.95)),
        "positive_tapes": int((values > 0.0).sum()),
    }


def action_calendar(model, env: ProgramORetOnlyEnv, seed: int) -> tuple[int, ...]:
    observation, _ = env.reset(options={"tape_seed": int(seed), "cell_index": 0})
    state = None
    episode_start = np.ones((1,), dtype=bool)
    actions: list[int] = []
    terminated = False
    while not terminated:
        action, state = model.predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=True,
        )
        value = int(np.asarray(action).item())
        actions.append(value)
        observation, _reward, terminated, _truncated, _info = env.step(value)
        episode_start[:] = terminated
    return tuple(actions)


def transducer_metrics(seed: int, calendar: tuple[int, ...]) -> dict[str, float]:
    skeleton, _ = extract_full_des_skeleton(
        seed=int(seed),
        scheduler=scheduler(),
        regime_persistence=0.75,
        dominant_share=0.90,
        downstream_freight_physics_mode="fixed_clock_physical_v1",
    )
    matrix = simulate_full_des_frontier(
        skeleton=skeleton,
        scheduler=scheduler(),
        calendars=np.asarray([calendar], dtype=np.uint8),
    )
    values = {key: float(value[0]) for key, value in matrix.items()}
    values["ret_service_full_clipped"] = compute_service_safe_reward(
        values,
        demanded_quantity=float(sum(skeleton.order_quantities)),
    )
    values["demanded_rations"] = float(sum(skeleton.order_quantities))
    return values


def direct_secondary(seed: int, calendar: tuple[int, ...]) -> dict[str, float]:
    sim, panel = run_program_o_full_des_episode(
        seed=int(seed),
        calendar=calendar,
        scheduler=scheduler(),
        regime_persistence=0.75,
        dominant_share=0.90,
        downstream_freight_physics_mode="fixed_clock_physical_v1",
        risks_enabled=False,
    )
    metrics = panel["metrics"]
    resources = panel["resources"]
    vector = direct_full_des_vector(sim, panel)
    return {
        "ret_excel_clipped_0_1": float(metrics["ret_excel_visible_clipped_0_1"]),
        "ret_excel_full_ledger": float(metrics["ret_excel_full_ledger"]),
        "ret_thesis": float(metrics["ret_thesis"]),
        "flow_fill_rate": float(metrics["flow_fill_rate"]),
        "delivered_rations": float(metrics["delivered_rations"]),
        "lost_orders": float(metrics["lost_orders"]),
        "unresolved_orders": float(metrics["n_orders"] - metrics["n_served"]),
        "terminal_stock": float(vector["ending_inventory_total"]),
        "worst_product_fill": float(panel["worst_product_fill"]),
        "actual_payload": float(resources["actual_payload"]),
    }


def guardrail_report(
    candidate_rows: list[dict[str, float | int]],
    comparator_rows: list[dict[str, float | int]],
) -> dict[str, object]:
    """Check the declared operational guardrails tape by tape."""
    rules = {
        "lost_orders_leq": lambda c, b: float(c["lost_orders"]) <= float(b["lost_orders"]) + 1e-12,
        "unresolved_orders_leq": lambda c, b: float(c["unresolved_orders"]) <= float(b["unresolved_orders"]) + 1e-12,
        "delivered_rations_geq": lambda c, b: float(c["delivered_rations"]) + 1e-12 >= float(b["delivered_rations"]),
        "worst_product_fill_geq": lambda c, b: float(c["worst_product_fill"]) + 1e-12 >= float(b["worst_product_fill"]),
        "flow_fill_rate_geq_minus_0_01": lambda c, b: float(c["flow_fill_rate"]) + 1e-12 >= float(b["flow_fill_rate"]) - 0.01,
    }
    failures: dict[str, list[int]] = {}
    for name, rule in rules.items():
        failures[name] = [
            int(candidate["seed"])
            for candidate, comparator in zip(candidate_rows, comparator_rows)
            if not rule(candidate, comparator)
        ]
    return {
        "n": len(candidate_rows),
        "failures_by_rule": failures,
        "all_pass": not any(failures.values()),
        "terminal_stock_reported": True,
        "full_ledger_zero_in_all_tapes": all(
            abs(float(row["ret_excel_full_ledger"])) <= 1e-12 for row in candidate_rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("PPO_MLP", "RecurrentPPO_MLP"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.architecture == "PPO_MLP":
        from stable_baselines3 import PPO
        model_class = PPO
        policy = "MlpPolicy"
    else:
        from sb3_contrib import RecurrentPPO
        model_class = RecurrentPPO
        policy = "MlpLstmPolicy"

    model_paths = sorted(args.model_dir.glob("*.zip"))
    if not model_paths:
        raise SystemExit(f"no model checkpoints under {args.model_dir}")
    seeds = list(range(7400097, 7400121))
    development_result = json.loads(DEVELOPMENT_RESULT.read_text())
    frozen_calendar = tuple(development_result["frozen_incumbent"]["calendar"])
    in_sample_calendar = tuple(development_result["in_sample_static_incumbent"]["calendar"])
    env = ProgramORetOnlyEnv(
        scheduler=scheduler(),
        tape_seed_start=seeds[0],
        tape_seed_end=seeds[-1],
        cells=(CONFIRMED_RET_CELLS[0],),
        reward_mode="service_safe",
    )

    rows: dict[str, list[dict[str, float | int | list[int]]]] = defaultdict(list)
    all_calendars: dict[str, list[list[int]]] = {}
    for model_path in model_paths:
        model = model_class.load(model_path, env=env, device="cpu")
        model_name = model_path.stem
        model_rows = []
        calendars = []
        for seed in seeds:
            calendar = action_calendar(model, env, seed)
            calendars.append(list(calendar))
            values = transducer_metrics(seed, calendar)
            values.update(direct_secondary(seed, calendar))
            values["seed"] = seed
            model_rows.append(values)
        rows[model_name] = model_rows
        all_calendars[model_name] = calendars

    comparator_rows: dict[str, list[dict[str, float | int]]] = defaultdict(list)
    for label, calendar in {
        "frozen_incumbent": frozen_calendar,
        "in_sample_service_safe": in_sample_calendar,
    }.items():
        for seed in seeds:
            values = transducer_metrics(seed, calendar)
            values.update(direct_secondary(seed, calendar))
            values["seed"] = seed
            comparator_rows[label].append(values)

    summaries: dict[str, dict[str, object]] = {}
    all_series = {**comparator_rows, **rows}
    for name, values in all_series.items():
        primary = np.asarray([float(row["ret_service_full_clipped"]) for row in values])
        summaries[name] = {
            "primary": bootstrap(primary, seed=820000 + len(name)),
            "secondary_means": {
                key: float(np.mean([float(row[key]) for row in values]))
                for key in (
                    "ret_excel_clipped_0_1",
                    "ret_excel_full_ledger",
                    "ret_thesis",
                    "flow_fill_rate",
                    "delivered_rations",
                    "lost_orders",
                    "unresolved_orders",
                    "terminal_stock",
                    "worst_product_fill",
                    "actual_payload",
                )
            },
            "n_unique_calendars": len({tuple(row) for row in all_calendars.get(name, [list(frozen_calendar)])}),
        }
    for model_name in rows:
        summaries[model_name]["guardrails_vs_frozen"] = guardrail_report(
            rows[model_name], comparator_rows["frozen_incumbent"]
        )
        summaries[model_name]["guardrails_vs_in_sample"] = guardrail_report(
            rows[model_name], comparator_rows["in_sample_service_safe"]
        )

    contrasts: dict[str, object] = {}
    for model_name in rows:
        for baseline_name in ("frozen_incumbent", "in_sample_service_safe"):
            diff = np.asarray(
                [
                    float(model_row["ret_service_full_clipped"])
                    - float(base_row["ret_service_full_clipped"])
                    for model_row, base_row in zip(rows[model_name], comparator_rows[baseline_name])
                ]
            )
            contrasts[f"{model_name}_minus_{baseline_name}"] = {
                "primary": bootstrap(diff, seed=821000 + len(contrasts)),
                "secondary_mean_deltas": {
                    key: float(
                        np.mean(
                            [float(a[key]) - float(b[key]) for a, b in zip(rows[model_name], comparator_rows[baseline_name])]
                        )
                    )
                    for key in (
                        "ret_excel_clipped_0_1",
                        "ret_excel_full_ledger",
                        "ret_thesis",
                        "flow_fill_rate",
                        "delivered_rations",
                        "lost_orders",
                        "unresolved_orders",
                        "terminal_stock",
                        "worst_product_fill",
                        "actual_payload",
                    )
                },
            }

    adjudication: dict[str, object] = {}
    for model_name in rows:
        contrast = contrasts[model_name + "_minus_frozen_incumbent"]["primary"]
        guardrails = summaries[model_name]["guardrails_vs_frozen"]
        adjudication[model_name] = {
            "mean_exceeds_frozen": float(contrast["mean"]) > 0.0,
            "positive_lcb": float(contrast["lcb95_one_sided"]) > 0.0,
            "lcb_exceeds_sesoi_0_01": float(contrast["lcb95_one_sided"]) > 0.01,
            "guardrails_vs_frozen_pass": bool(guardrails["all_pass"]),
            "development_signal_rule": bool(
                float(contrast["mean"]) > 0.0
                and float(contrast["lcb95_one_sided"]) > 0.0
                and guardrails["all_pass"]
            ),
            "confirmatory_promotion": False,
        }

    result = {
        "schema_version": "program_b_service_safe_evaluation_v1",
        "architecture": args.architecture,
        "validation_seeds": [seeds[0], seeds[-1]],
        "model_paths": [str(path) for path in model_paths],
        "frozen_calendar": list(frozen_calendar),
        "in_sample_calendar": list(in_sample_calendar),
        "summaries": summaries,
        "contrasts": contrasts,
        "adjudication": adjudication,
        "calendars": all_calendars,
        "claim_boundary": "development evaluation only; no fresh confirmation and no neural superiority claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "summaries": summaries, "contrasts": contrasts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
