#!/usr/bin/env python3
"""Audit the frozen Program B contract against burned full-DES matrices.

This is a retrospective analysis of already executed Program O full-DES blocks.
It opens no seeds and does not train a learner.  The primary endpoint is the
full order ledger; the visible/clipped endpoint is deliberately secondary.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "program_b_full_des_gate_v2.json"
SCHEDULER = {
    0: ("P_H", "P_H", "P_H"),
    1: ("P_H", "P_C", "P_H"),
    2: ("P_C", "P_H", "P_C"),
    3: ("P_C", "P_C", "P_C"),
}
MATRIX_KEYS = (
    "ret_full",
    "ret_visible",
    "ret_visible_cvar10",
    "ration_ret_visible",
    "visible_rows",
    "omitted_rows",
    "omitted_quantity",
    "generated_orders",
    "lost_orders",
    "lost_quantity",
    "unresolved_orders",
    "unresolved_quantity",
    "remaining_quantity_P_C",
    "remaining_quantity_P_H",
    "max_backlog_age",
    "service_loss_auc",
    "fill_P_C",
    "fill_P_H",
    "worst_product_fill",
    "ending_inventory_total",
    "gross_policy_batch_slots",
    "gross_production_quantity",
    "charged_daily_dispatch_slots",
    "charged_downstream_vehicle_hours",
    "actual_payload",
    "mass_residual",
    "partition_residual",
    "aggregate_ration_residual",
    "raw_material_residual",
)
SECONDARY_KEYS = (
    "ret_excel_clipped_0_1",
    "ret_excel_full_ledger",
    "ret_thesis",
    "flow_fill_rate",
    "delivered_rations",
    "lost_orders",
    "unresolved_orders",
    "terminal_stock",
    "worst_product_fill",
)
FUNGIBLE_OUTCOME_KEYS = (
    "ret_full",
    "ret_visible",
    "ret_visible_cvar10",
    "ration_ret_visible",
    "visible_rows",
    "omitted_rows",
    "omitted_quantity",
    "generated_orders",
    "lost_orders",
    "lost_quantity",
    "unresolved_orders",
    "unresolved_quantity",
    "remaining_quantity_P_C",
    "remaining_quantity_P_H",
    "max_backlog_age",
    "service_loss_auc",
    "fill_P_C",
    "fill_P_H",
    "worst_product_fill",
    "ending_inventory_total",
    "actual_payload",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def calendar_index(calendar: list[int] | tuple[int, ...]) -> int:
    value = 0
    for action in calendar:
        value = value * 4 + int(action)
    return value


def index_calendar(index: int, weeks: int = 8) -> tuple[int, ...]:
    values = [0] * weeks
    for pos in range(weeks - 1, -1, -1):
        values[pos] = int(index) % 4
        index //= 4
    return tuple(values)


def bootstrap_summary(values: np.ndarray, *, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(int(seed))
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


def load_block_matrix(block_root: Path, profile: str) -> tuple[list[int], dict[str, np.ndarray]]:
    paths = sorted((block_root / "raw_calendar_matrix" / profile).glob("tape_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No tapes under {block_root / 'raw_calendar_matrix' / profile}")
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in MATRIX_KEYS}
    seeds: list[int] = []
    for path in paths:
        seed = int(path.stem.split("_")[-1])
        seeds.append(seed)
        with np.load(path, allow_pickle=False) as payload:
            missing = [key for key in MATRIX_KEYS if key not in payload]
            if missing:
                raise ValueError(f"{path} missing {missing}")
            for key in MATRIX_KEYS:
                arrays[key].append(np.asarray(payload[key], dtype=float))
    stacked = {key: np.stack(value, axis=0) for key, value in arrays.items()}
    n_tapes, n_calendars = stacked["ret_full"].shape
    if n_calendars != 4**8:
        raise ValueError(f"Expected 65536 calendars, got {n_calendars}")
    return seeds, stacked


def load_demand_totals(block_root: Path, seeds: list[int]) -> np.ndarray:
    totals = []
    for seed in seeds:
        skeleton = load_json(block_root / "skeletons" / f"tape_{seed}.json")
        totals.append(float(sum(skeleton["order_quantities"])))
    return np.asarray(totals, dtype=float)


def exact_fungible_null(block_root: Path, profile: str) -> dict[str, Any]:
    paths = sorted((block_root / "raw_calendar_matrix" / profile).glob("tape_*.npz"))
    max_ranges: dict[str, float] = {}
    partition_ranges: dict[str, float] = {}
    bit_identical = True
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            for key in payload.files:
                values = np.asarray(payload[key])
                span = float(np.max(values) - np.min(values)) if values.size else 0.0
                target = max_ranges if key in FUNGIBLE_OUTCOME_KEYS else partition_ranges
                target[key] = max(target.get(key, 0.0), span)
                if key in FUNGIBLE_OUTCOME_KEYS:
                    bit_identical = bit_identical and bool(np.all(values == values[0]))
    return {
        "bit_identical": bit_identical,
        "outcome_keys": list(FUNGIBLE_OUTCOME_KEYS),
        "max_ranges": max_ranges,
        "partition_ranges_reported_not_vetoed": partition_ranges,
    }


def safe_mask(
    arrays: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    primary_key: str,
) -> np.ndarray:
    n_tapes, n_calendars = arrays["ret_full"].shape
    mask = np.ones((n_tapes, n_calendars), dtype=bool)
    higher_or_equal = (
        primary_key,
        "ret_full",
        "quantity_ret_full" if "quantity_ret_full" in arrays else "ret_full",
        "ret_visible_cvar10",
        "fill_P_C",
        "fill_P_H",
        "worst_product_fill",
        "actual_payload",
    )
    lower_or_equal = (
        "omitted_rows",
        "omitted_quantity",
        "lost_orders",
        "lost_quantity",
        "unresolved_orders",
        "unresolved_quantity",
        "remaining_quantity_P_C",
        "remaining_quantity_P_H",
        "max_backlog_age",
        "service_loss_auc",
    )
    for key in higher_or_equal:
        if key not in arrays:
            continue
        mask &= arrays[key] >= baseline[key][:, None] - 1e-12
    for key in lower_or_equal:
        mask &= arrays[key] <= baseline[key][:, None] + 1e-12
    for key in (
        "generated_orders",
        "gross_policy_batch_slots",
        "gross_production_quantity",
        "charged_daily_dispatch_slots",
        "charged_downstream_vehicle_hours",
    ):
        mask &= np.isclose(arrays[key], baseline[key][:, None], atol=1e-8, rtol=0.0)
    for key in (
        "mass_residual",
        "partition_residual",
        "aggregate_ration_residual",
        "raw_material_residual",
    ):
        mask &= arrays[key] <= 1e-8
    return mask


def choose_safe_indices(
    arrays: dict[str, np.ndarray], baseline_index: int, primary_key: str
) -> tuple[np.ndarray, np.ndarray]:
    baseline = {key: arrays[key][:, baseline_index] for key in arrays}
    mask = safe_mask(arrays, baseline, primary_key)
    selected = np.empty(mask.shape[0], dtype=int)
    eligible_counts = mask.sum(axis=1)
    for tape in range(mask.shape[0]):
        candidates = np.flatnonzero(mask[tape])
        if len(candidates) == 0:
            selected[tape] = baseline_index
            continue
        # Primary max, then the preregistered deterministic tie rule.
        score = arrays[primary_key][tape, candidates]
        best = candidates[np.isclose(score, score.max(), atol=1e-12, rtol=0.0)]
        for key, sign in (
            ("unresolved_quantity", 1),
            ("worst_product_fill", -1),
            ("actual_payload", -1),
            ("ending_inventory_total", 1),
        ):
            vals = arrays[key][tape, best]
            target = vals.min() if sign == 1 else vals.max()
            best = best[np.isclose(vals, target, atol=1e-12, rtol=0.0)]
        selected[tape] = int(best.min())
    return selected, eligible_counts


def direct_import(source_root: Path):
    source_root = source_root.resolve()
    sys.path = [str(source_root)] + [
        item for item in sys.path if item not in {"", str(ROOT), str(ROOT / "scripts")}
    ]
    module = importlib.import_module("supply_chain.program_o_full_des")
    return module.run_program_o_full_des_episode


def direct_metric(panel: dict[str, Any]) -> dict[str, float]:
    metrics = panel["metrics"]
    products = panel["products"]
    conservation = panel["conservation"]
    return {
        "ret_excel_clipped_0_1": float(metrics["ret_excel_visible_clipped_0_1"]),
        "ret_excel_full_ledger": float(metrics["ret_excel_full_ledger"]),
        "ret_thesis": float(metrics["ret_thesis"]),
        "flow_fill_rate": float(metrics["flow_fill_rate"]),
        "delivered_rations": float(metrics["delivered_rations"]),
        "lost_orders": float(metrics["lost_orders"]),
        "unresolved_orders": float(metrics["n_orders"] - metrics["n_served"]),
        "terminal_stock": float(
            conservation["per_product"]["P_C"]["nodes"]["rations_sb"]
            + conservation["per_product"]["P_H"]["nodes"]["rations_sb"]
        ),
        "worst_product_fill": float(panel["worst_product_fill"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", choices=("development", "validation"), required=True)
    parser.add_argument("--block-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--primary-mode",
        choices=("ret_full", "service_safe"),
        default="ret_full",
    )
    args = parser.parse_args()

    contract = load_json(CONTRACT)
    profile = "rho75_share90__centered_minority_v1"
    seeds, arrays = load_block_matrix(args.block_root.resolve(), profile)
    demand_totals = load_demand_totals(args.block_root.resolve(), seeds)
    primary_key = "ret_full" if args.primary_mode == "ret_full" else "primary_metric"
    if args.primary_mode == "service_safe":
        arrays["primary_metric"] = np.clip(arrays["ration_ret_visible"], 0.0, 1.0) * (
            1.0 - arrays["omitted_quantity"] / demand_totals[:, None]
        )
    frozen_calendar = tuple(contract["comparators"]["frozen_incumbent"]["calendar"])
    frozen_index = calendar_index(frozen_calendar)
    in_sample_index = int(np.argmax(arrays[primary_key].mean(axis=0)))
    safe_index_by_tape, eligible_counts = choose_safe_indices(
        arrays, frozen_index, primary_key
    )

    direct_run = direct_import(args.source_root)
    direct_cache: dict[str, dict[str, float]] = {}
    calendars_needed: set[tuple[int, ...]] = {frozen_calendar, index_calendar(in_sample_index)}
    calendars_needed.update(index_calendar(int(index)) for index in safe_index_by_tape)
    # Run direct SimPy only for the predeclared comparator/oracle calendars.
    for seed, calendar in ((seed, tuple(index_calendar(int(index)))) for seed, index in zip(seeds, safe_index_by_tape)):
        calendars_needed.add(calendar)
    for seed in seeds:
        for calendar in (frozen_calendar, index_calendar(in_sample_index)):
            key = f"{seed}|{','.join(map(str, calendar))}"
            _sim, panel = direct_run(
                seed=int(seed),
                calendar=calendar,
                scheduler=SCHEDULER,
                regime_persistence=0.75,
                dominant_share=0.90,
                complete_substitution=False,
            )
            direct_cache[key] = direct_metric(panel)

    for seed, index in zip(seeds, safe_index_by_tape):
        calendar = index_calendar(int(index))
        key = f"{seed}|{','.join(map(str, calendar))}"
        if key not in direct_cache:
            _sim, panel = direct_run(
                seed=int(seed),
                calendar=calendar,
                scheduler=SCHEDULER,
                regime_persistence=0.75,
                dominant_share=0.90,
                complete_substitution=False,
            )
            direct_cache[key] = direct_metric(panel)

    results: dict[str, Any] = {
        "schema_version": "program_b_full_des_gate_result_v2",
        "contract": str(CONTRACT.relative_to(ROOT)),
        "block": args.block,
        "block_root": str(args.block_root.resolve()),
        "source_root": str(args.source_root.resolve()),
        "profile": profile,
        "primary_mode": args.primary_mode,
        "demand_totals": demand_totals.tolist(),
        "seeds": seeds,
        "calendar_count": int(arrays["ret_full"].shape[1]),
        "frozen_incumbent": {"calendar": list(frozen_calendar), "index": frozen_index},
        "in_sample_static_incumbent": {
            "calendar": list(index_calendar(in_sample_index)),
            "index": in_sample_index,
            "mean_primary": float(arrays[primary_key].mean(axis=0)[in_sample_index]),
        },
        "safe_oracle": {
            "eligible_counts": eligible_counts.tolist(),
            "unique_calendars": int(len(set(map(int, safe_index_by_tape)))),
            "indices": safe_index_by_tape.tolist(),
            "calendars": [list(index_calendar(int(index))) for index in safe_index_by_tape],
        },
        "fungible_null": exact_fungible_null(
            args.block_root.resolve(), "fungible_null__centered_minority_v1"
        ),
        "comparisons": {},
        "primary_gate": {},
    }

    for label, comparator_index in (
        ("frozen_incumbent", frozen_index),
        ("in_sample_static_incumbent", in_sample_index),
    ):
        base = {key: arrays[key][:, comparator_index] for key in arrays}
        safe = {key: arrays[key][np.arange(len(seeds)), safe_index_by_tape] for key in arrays}
        primary_delta = safe[primary_key] - base[primary_key]
        comparison: dict[str, Any] = {
            "primary_safe_h_pi": bootstrap_summary(primary_delta, seed=7400999),
            "raw_primary_range": [float(arrays[primary_key].min()), float(arrays[primary_key].max())],
            "safe_fallback_to_comparator": int((eligible_counts == 0).sum()),
            "secondary": {},
        }
        for key in (
            "ret_visible",
            "actual_payload",
            "lost_orders",
            "unresolved_orders",
            "ending_inventory_total",
            "worst_product_fill",
            "service_loss_auc",
            "max_backlog_age",
        ):
            comparison["secondary"][key] = bootstrap_summary(
                safe[key] - base[key], seed=7410000 + len(comparison["secondary"])
            )
        direct_deltas: dict[str, list[float]] = {key: [] for key in SECONDARY_KEYS}
        for tape, seed in enumerate(seeds):
            safe_key = f"{seed}|{','.join(map(str, index_calendar(int(safe_index_by_tape[tape]))))}"
            base_key = f"{seed}|{','.join(map(str, index_calendar(int(comparator_index))))}"
            safe_metrics = direct_cache[safe_key]
            base_metrics = direct_cache[base_key]
            for key in SECONDARY_KEYS:
                direct_deltas[key].append(safe_metrics[key] - base_metrics[key])
        comparison["direct_secondary"] = {
            key: bootstrap_summary(np.asarray(values), seed=7420000 + offset)
            for offset, (key, values) in enumerate(direct_deltas.items())
        }
        results["comparisons"][label] = comparison

    results["primary_gate"] = {
        "primary_mode": args.primary_mode,
        "sesoi": 0.01,
        "development_mean_threshold": 0.015,
        "primary_ret_full_identically_zero": bool(
            np.max(np.abs(arrays["ret_full"])) <= 1e-12
        ),
        "primary_metric_identically_zero": bool(
            np.max(np.abs(arrays[primary_key])) <= 1e-12
        ),
        "fungible_exact_null": bool(results["fungible_null"]["bit_identical"]),
        "status": (
            "STOP_PRIMARY_FULL_LEDGER_HAS_NO_HEADROOM"
            if args.primary_mode == "ret_full"
            and np.max(np.abs(arrays["ret_full"])) <= 1e-12
            else "REPORT_SERVICE_SAFE_METRIC_EXPLORATORY"
            if args.primary_mode == "service_safe"
            else "REPORT_PRIMARY_GATE"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "block": args.block,
        "status": results["primary_gate"]["status"],
        "frozen_primary": results["comparisons"]["frozen_incumbent"]["primary_safe_h_pi"],
        "in_sample_primary": results["comparisons"]["in_sample_static_incumbent"]["primary_safe_h_pi"],
        "fungible_null": results["fungible_null"]["bit_identical"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
