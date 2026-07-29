#!/usr/bin/env python3
"""Execute Program O Gate O-0 over the complete base-4 action frontier.

The producer is deliberately independent of the full DES.  It is the final
cheap falsification before product identity may be translated into Op5--Op13.
Every tape/calendar guardrail is persisted; promotion uses a per-tape safe
oracle while the formal resource-restricted raw H_PI remains separately
reported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.screen_program_o_exact_transducer import make_tape  # noqa: E402


DEFAULT_CONTRACT = ROOT / "contracts/program_o_full_action_transducer_v1.json"
DEFAULT_OUTPUT_ROOT = ROOT / "results/program_o/gate_o0_full_action_transducer_v1"
DEFAULT_VALIDATION_FREEZE = (
    ROOT
    / "research/paper2_exhaustive_search/"
    "program_o_gate_o0_validation_freeze_20260714.json"
)
PRODUCT_INDEX = {"P_C": 0, "P_H": 1}
PRODUCT_NAME = ("P_C", "P_H")

MATRIX_KEYS = (
    "ret_visible",
    "ret_full",
    "quantity_ret_full",
    "ret_full_cvar10",
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
    "fill_order_P_C",
    "fill_order_P_H",
    "fill_quantity_P_C",
    "fill_quantity_P_H",
    "worst_order_fill",
    "worst_quantity_fill",
    "ending_inventory_P_C",
    "ending_inventory_P_H",
    "ending_inventory_total",
    "allocated_quantity_P_C",
    "allocated_quantity_P_H",
    "completed_quantity_P_C",
    "completed_quantity_P_H",
    "production_batches",
    "production_quantity",
    "gross_batch_slots",
    "mass_residual_P_C",
    "mass_residual_P_H",
    "mass_residual_aggregate",
)

SAFE_HIGHER = (
    "visible_rows",
    "ret_full",
    "quantity_ret_full",
    "ret_full_cvar10",
    "fill_order_P_C",
    "fill_order_P_H",
    "fill_quantity_P_C",
    "fill_quantity_P_H",
    "worst_order_fill",
    "worst_quantity_fill",
)
SAFE_LOWER = (
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
SAFE_EQUAL = (
    "generated_orders",
    "production_batches",
    "production_quantity",
    "gross_batch_slots",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def full_action_calendars() -> np.ndarray:
    """Return all base-4, big-endian, length-eight calendars."""
    return np.asarray(tuple(product(range(4), repeat=8)), dtype=np.uint8)


def calendar_index(calendar: Iterable[int]) -> int:
    values = tuple(int(value) for value in calendar)
    if len(values) != 8 or any(value not in range(4) for value in values):
        raise ValueError("calendar must contain eight base-4 action levels")
    return int(sum(value * (4 ** (7 - idx)) for idx, value in enumerate(values)))


def scheduler_array(contract: Mapping[str, Any], scheduler_id: str) -> np.ndarray:
    raw = contract["within_week_schedulers"][scheduler_id]
    return np.asarray(
        [[PRODUCT_INDEX[name] for name in raw[str(action)]] for action in range(4)],
        dtype=np.uint8,
    )


def frozen_profiles(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    primary_scheduler = str(contract["primary_scheduler"])
    profiles: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for cell in [
        *contract["primary_cells"],
        *contract["plausibility_sensitivity_cells"],
    ]:
        by_id[str(cell["cell_id"])] = dict(cell)
    for cell in contract["primary_cells"]:
        profiles.append(
            {
                **cell,
                "profile_id": f"{cell['cell_id']}__{primary_scheduler}",
                "scheduler_id": primary_scheduler,
                "role": "primary",
                "complete_substitution": False,
            }
        )
    sensitivity_cell = by_id[str(contract["ordering_sensitivity"]["cell_id"])]
    for scheduler_id in contract["ordering_sensitivity"]["schedulers"]:
        profiles.append(
            {
                **sensitivity_cell,
                "profile_id": f"{sensitivity_cell['cell_id']}__{scheduler_id}",
                "scheduler_id": scheduler_id,
                "role": "ordering_sensitivity",
                "complete_substitution": False,
            }
        )
    for cell in contract["plausibility_sensitivity_cells"]:
        profiles.append(
            {
                **cell,
                "profile_id": f"{cell['cell_id']}__{primary_scheduler}",
                "scheduler_id": primary_scheduler,
                "role": "plausibility_sensitivity",
                "complete_substitution": False,
            }
        )
    null = contract["null_cell"]
    for scheduler_id in null["schedulers"]:
        profiles.append(
            {
                "cell_id": str(null["cell_id"]),
                "profile_id": f"{null['cell_id']}__{scheduler_id}",
                "scheduler_id": scheduler_id,
                "role": "exact_null",
                "regime_persistence": float(null["regime_persistence"]),
                "dominant_share": float(null["dominant_share"]),
                "complete_substitution": True,
            }
        )
    ids = [profile["profile_id"] for profile in profiles]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate frozen profile id")
    return profiles


def _event_schedule(contract: Mapping[str, Any]) -> list[tuple[float, str, int]]:
    events: list[tuple[float, str, int]] = []
    for week in range(int(contract["weeks"])):
        base = 168.0 * week
        for slot, offset in enumerate(contract["batch_completion_offsets_hours"]):
            events.append((base + float(offset), "batch", slot))
        for day, offset in enumerate(contract["demand_offsets_hours"]):
            events.append((base + float(offset), "demand", day))
    return sorted(events, key=lambda row: (row[0], 0 if row[1] == "batch" else 1))


def _metric_panel(
    *,
    remaining: np.ndarray,
    oat: np.ndarray,
    bt: np.ndarray,
    order_products: np.ndarray,
    order_opt: np.ndarray,
    inventories: tuple[np.ndarray, np.ndarray] | list[np.ndarray] | None,
    pool: np.ndarray | None,
    produced_by_product: tuple[np.ndarray, np.ndarray] | None,
    contract: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    n_orders, n_calendars = remaining.shape
    quantity = float(contract["daily_order_quantity"])
    completed = np.isfinite(oat)
    j = np.arange(1, n_orders + 1, dtype=float)[:, None]
    late = completed & (
        (oat - order_opt[:, None])
        > float(contract["delivery_lead_hours_after_allocation"])
    )
    # The historical corrected transducer sets DPj=CTj on a completed late
    # order.  With no APj/RPj this enters the Excel risk/no-recovery branch and
    # scores zero; on-time orders stay in the Bt/Ut fill-rate branch.
    visible_order_ret = np.where(late, 0.0, 1.0 - (bt.astype(float) / j))
    visible_rows = completed.sum(axis=0).astype(np.int16)
    visible_sum = np.where(completed, visible_order_ret, 0.0).sum(axis=0)
    ret_visible = np.divide(
        visible_sum,
        visible_rows,
        out=np.ones(n_calendars, dtype=float),
        where=visible_rows > 0,
    )

    cumulative_late = np.cumsum(late, axis=0)
    full_order_ret = np.where(
        completed,
        np.where(late, 0.0, 1.0 - (cumulative_late / j)),
        0.0,
    )
    ret_full = full_order_ret.mean(axis=0)
    quantity_ret_full = (
        (full_order_ret * quantity).sum(axis=0) / (n_orders * quantity)
    )
    tail_n = max(1, math.ceil(n_orders * 0.10))
    ret_full_cvar10 = np.partition(full_order_ret, tail_n - 1, axis=0)[
        :tail_n
    ].mean(axis=0)

    unresolved = ~completed
    unresolved_orders = unresolved.sum(axis=0).astype(np.int16)
    unresolved_quantity = remaining.sum(axis=0)
    omitted_rows = (n_orders - visible_rows).astype(np.int16)
    omitted_quantity = unresolved.sum(axis=0).astype(float) * quantity
    horizon = float(contract["clearance_score_time_hours"])
    end_times = np.where(completed, oat, horizon)
    lateness = np.maximum(
        0.0,
        end_times
        - (
            order_opt[:, None]
            + float(contract["delivery_lead_hours_after_allocation"])
        ),
    )
    service_loss_auc = (lateness * quantity).sum(axis=0)
    unresolved_age = np.where(unresolved, horizon - order_opt[:, None], 0.0)
    max_backlog_age = unresolved_age.max(axis=0)

    allocated_by_product: list[np.ndarray] = []
    completed_by_product: list[np.ndarray] = []
    remaining_by_product: list[np.ndarray] = []
    order_fill: list[np.ndarray] = []
    quantity_fill: list[np.ndarray] = []
    for product_id in range(2):
        mask = order_products == product_id
        demand_orders = int(mask.sum())
        demand_quantity = demand_orders * quantity
        remaining_product = remaining[mask].sum(axis=0)
        allocated_product = demand_quantity - remaining_product
        completed_product = completed[mask].sum(axis=0).astype(float) * quantity
        remaining_by_product.append(remaining_product)
        allocated_by_product.append(allocated_product)
        completed_by_product.append(completed_product)
        order_fill.append(
            completed[mask].mean(axis=0)
            if demand_orders
            else np.ones(n_calendars, dtype=float)
        )
        quantity_fill.append(
            allocated_product / demand_quantity
            if demand_quantity
            else np.ones(n_calendars, dtype=float)
        )

    if inventories is None:
        if pool is None:
            raise AssertionError("fungible panel requires a physical pool")
        ending_c = np.zeros(n_calendars, dtype=float)
        ending_h = np.zeros(n_calendars, dtype=float)
        ending_total = pool
        mass_c = np.zeros(n_calendars, dtype=float)
        mass_h = np.zeros(n_calendars, dtype=float)
        initial_total = sum(float(value) for value in contract["initial_inventory"].values())
        produced_total = (
            int(contract["weeks"])
            * int(contract["weekly_production_batches"])
            * float(contract["batch_quantity"])
        )
        mass_total = initial_total + produced_total - ending_total - sum(allocated_by_product)
    else:
        if produced_by_product is None:
            raise AssertionError("nonfungible panel requires product production")
        ending_c, ending_h = inventories
        ending_total = ending_c + ending_h
        initial_c = float(contract["initial_inventory"]["P_C"])
        initial_h = float(contract["initial_inventory"]["P_H"])
        mass_c = initial_c + produced_by_product[0] - ending_c - allocated_by_product[0]
        mass_h = initial_h + produced_by_product[1] - ending_h - allocated_by_product[1]
        mass_total = mass_c + mass_h

    constant_orders = np.full(n_calendars, n_orders, dtype=np.int16)
    production_batches = np.full(
        n_calendars,
        int(contract["weeks"]) * int(contract["weekly_production_batches"]),
        dtype=np.int16,
    )
    production_quantity = production_batches.astype(float) * float(
        contract["batch_quantity"]
    )
    zeros_i = np.zeros(n_calendars, dtype=np.int16)
    zeros_f = np.zeros(n_calendars, dtype=float)
    return {
        "ret_visible": ret_visible,
        "ret_full": ret_full,
        "quantity_ret_full": quantity_ret_full,
        "ret_full_cvar10": ret_full_cvar10,
        "visible_rows": visible_rows,
        "omitted_rows": omitted_rows,
        "omitted_quantity": omitted_quantity,
        "generated_orders": constant_orders,
        "lost_orders": zeros_i,
        "lost_quantity": zeros_f,
        "unresolved_orders": unresolved_orders,
        "unresolved_quantity": unresolved_quantity,
        "remaining_quantity_P_C": remaining_by_product[0],
        "remaining_quantity_P_H": remaining_by_product[1],
        "max_backlog_age": max_backlog_age,
        "service_loss_auc": service_loss_auc,
        "fill_order_P_C": order_fill[0],
        "fill_order_P_H": order_fill[1],
        "fill_quantity_P_C": quantity_fill[0],
        "fill_quantity_P_H": quantity_fill[1],
        "worst_order_fill": np.minimum(order_fill[0], order_fill[1]),
        "worst_quantity_fill": np.minimum(quantity_fill[0], quantity_fill[1]),
        "ending_inventory_P_C": ending_c,
        "ending_inventory_P_H": ending_h,
        "ending_inventory_total": ending_total,
        "allocated_quantity_P_C": allocated_by_product[0],
        "allocated_quantity_P_H": allocated_by_product[1],
        "completed_quantity_P_C": completed_by_product[0],
        "completed_quantity_P_H": completed_by_product[1],
        "production_batches": production_batches,
        "production_quantity": production_quantity,
        "gross_batch_slots": production_batches.copy(),
        "mass_residual_P_C": mass_c,
        "mass_residual_P_H": mass_h,
        "mass_residual_aggregate": mass_total,
    }


def simulate_frontier(
    *,
    tape: Any,
    contract: Mapping[str, Any],
    scheduler_id: str,
    complete_substitution: bool,
) -> dict[str, np.ndarray]:
    calendars = full_action_calendars()
    scheduler = scheduler_array(contract, scheduler_id)
    n_output = len(calendars)
    n_live = 1 if complete_substitution else n_output
    quantity = float(contract["daily_order_quantity"])
    batch_quantity = float(contract["batch_quantity"])
    n_orders = int(contract["weeks"]) * int(contract["workdays_per_week"])
    remaining = np.zeros((n_orders, n_live), dtype=float)
    oat = np.full((n_orders, n_live), np.nan, dtype=float)
    bt = np.zeros((n_orders, n_live), dtype=np.int16)
    order_opt = np.zeros(n_orders, dtype=float)
    order_products = np.asarray(
        [PRODUCT_INDEX[name] for name in tape.order_products], dtype=np.uint8
    )
    inventories = None
    pool = None
    produced = None
    if complete_substitution:
        pool = np.full(
            n_live,
            sum(float(value) for value in contract["initial_inventory"].values()),
            dtype=float,
        )
    else:
        inventories = [
            np.full(n_live, float(contract["initial_inventory"]["P_C"])),
            np.full(n_live, float(contract["initial_inventory"]["P_H"])),
        ]
        produced = (np.zeros(n_live, dtype=float), np.zeros(n_live, dtype=float))

    demand_index = 0
    for now, kind, slot_or_day in _event_schedule(contract):
        week = int(now // 168.0)
        if kind == "batch":
            if complete_substitution:
                available = np.full(n_live, batch_quantity, dtype=float)
                for prior in range(demand_index):
                    take = np.minimum(remaining[prior], available)
                    was_open = remaining[prior] > 1e-12
                    remaining[prior] -= take
                    available -= take
                    completed_now = was_open & (remaining[prior] <= 1e-12)
                    if np.any(completed_now):
                        oat[prior, completed_now] = max(
                            order_opt[prior]
                            + float(contract["delivery_lead_hours_after_allocation"]),
                            now
                            + float(contract["delivery_lead_hours_after_allocation"]),
                        )
                pool += available
                continue

            actions = calendars[:, week]
            produced_product = scheduler[actions, slot_or_day]
            for product_id in range(2):
                mask = produced_product == product_id
                available = np.where(mask, batch_quantity, 0.0)
                produced[product_id][mask] += batch_quantity
                for prior in range(demand_index):
                    if int(order_products[prior]) != product_id:
                        continue
                    take = np.minimum(remaining[prior], available)
                    was_open = remaining[prior] > 1e-12
                    remaining[prior] -= take
                    available -= take
                    completed_now = was_open & (remaining[prior] <= 1e-12)
                    if np.any(completed_now):
                        oat[prior, completed_now] = max(
                            order_opt[prior]
                            + float(contract["delivery_lead_hours_after_allocation"]),
                            now
                            + float(contract["delivery_lead_hours_after_allocation"]),
                        )
                inventories[product_id] += available
            continue

        order_opt[demand_index] = now
        if demand_index:
            activated = (
                order_opt[:demand_index]
                + float(contract["delivery_lead_hours_after_allocation"])
                <= now
            )
            active = activated[:, None] & (
                ~np.isfinite(oat[:demand_index]) | (oat[:demand_index] > now)
            )
            bt[demand_index] = np.minimum(60, active.sum(axis=0)).astype(np.int16)
        remaining[demand_index] = quantity
        if complete_substitution:
            take = np.minimum(pool, quantity)
            pool -= take
        else:
            product_id = int(order_products[demand_index])
            take = np.minimum(inventories[product_id], quantity)
            inventories[product_id] -= take
        remaining[demand_index] -= take
        completed_now = remaining[demand_index] <= 1e-12
        if np.any(completed_now):
            oat[demand_index, completed_now] = (
                now + float(contract["delivery_lead_hours_after_allocation"])
            )
        demand_index += 1

    if demand_index != n_orders:
        raise AssertionError("incomplete frozen demand schedule")
    panel = _metric_panel(
        remaining=remaining,
        oat=oat,
        bt=bt,
        order_products=order_products,
        order_opt=order_opt,
        inventories=inventories,
        pool=pool,
        produced_by_product=produced,
        contract=contract,
    )
    if complete_substitution:
        panel = {
            key: np.repeat(np.asarray(value), n_output).astype(value.dtype, copy=False)
            for key, value in panel.items()
        }
    if tuple(panel) != MATRIX_KEYS:
        raise AssertionError("matrix schema drift")
    if any(np.asarray(value).shape != (n_output,) for value in panel.values()):
        raise AssertionError("incomplete frontier matrix")
    return panel


def _profile_by_id(contract: Mapping[str, Any], profile_id: str) -> dict[str, Any]:
    return next(
        profile
        for profile in frozen_profiles(contract)
        if profile["profile_id"] == profile_id
    )


def produce_shard(
    contract_path: str,
    stage: str,
    profile_id: str,
    seed: int,
    output_root: str,
) -> dict[str, Any]:
    contract_file = Path(contract_path)
    contract = json.loads(contract_file.read_text())
    profile = _profile_by_id(contract, profile_id)
    tape = make_tape(
        int(seed),
        persistence=float(profile["regime_persistence"]),
        dominant_share=float(profile["dominant_share"]),
    )
    panel = simulate_frontier(
        tape=tape,
        contract=contract,
        scheduler_id=str(profile["scheduler_id"]),
        complete_substitution=bool(profile["complete_substitution"]),
    )
    destination = (
        Path(output_root)
        / stage
        / "raw_calendar_matrix"
        / profile_id
        / f"tape_{int(seed)}.npz"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    temporary = destination.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **panel)
    os.replace(temporary, destination)
    try:
        recorded_path = str(destination.relative_to(ROOT))
    except ValueError:
        recorded_path = str(destination)
    return {
        "stage": stage,
        "profile_id": profile_id,
        "seed": int(seed),
        "tape_sha256": tape.sha256,
        "path": recorded_path,
        "sha256": sha256(destination),
        "bytes": destination.stat().st_size,
    }


def load_profile_panel(
    output_root: Path,
    stage: str,
    profile_id: str,
    seeds: Iterable[int],
) -> dict[str, np.ndarray]:
    rows: dict[str, list[np.ndarray]] = {key: [] for key in MATRIX_KEYS}
    for seed in seeds:
        path = (
            output_root
            / stage
            / "raw_calendar_matrix"
            / profile_id
            / f"tape_{int(seed)}.npz"
        )
        with np.load(path) as shard:
            if set(shard.files) != set(MATRIX_KEYS):
                raise AssertionError(f"matrix schema mismatch: {path}")
            for key in MATRIX_KEYS:
                rows[key].append(np.asarray(shard[key]))
    return {key: np.stack(value) for key, value in rows.items()}


def select_static(ret_visible: np.ndarray) -> int:
    means = ret_visible.mean(axis=0)
    return int(np.argmax(means))


def safe_oracle_indices(
    panel: Mapping[str, np.ndarray], static_index: int, *, tolerance: float = 1e-12
) -> np.ndarray:
    n_tapes, n_calendars = panel["ret_visible"].shape
    selected = np.empty(n_tapes, dtype=np.int32)
    mass_tolerance = 1e-8
    for tape_index in range(n_tapes):
        eligible = np.ones(n_calendars, dtype=bool)
        for key in SAFE_HIGHER:
            row = panel[key][tape_index]
            eligible &= row >= float(row[static_index]) - tolerance
        for key in SAFE_LOWER:
            row = panel[key][tape_index]
            eligible &= row <= float(row[static_index]) + tolerance
        for key in SAFE_EQUAL:
            row = panel[key][tape_index]
            eligible &= np.abs(row - float(row[static_index])) <= tolerance
        for key in (
            "mass_residual_P_C",
            "mass_residual_P_H",
            "mass_residual_aggregate",
        ):
            eligible &= np.abs(panel[key][tape_index]) <= mass_tolerance
        eligible[static_index] = True
        safe_scores = np.where(eligible, panel["ret_visible"][tape_index], -np.inf)
        chosen = int(np.argmax(safe_scores))
        if not np.isfinite(safe_scores[chosen]):
            raise AssertionError("safe set unexpectedly empty")
        selected[tape_index] = chosen
    return selected


def _action_support(indices: np.ndarray) -> dict[str, Any]:
    calendars = full_action_calendars()
    actions = calendars[np.asarray(indices, dtype=int)].ravel()
    counts = np.bincount(actions, minlength=4)
    fractions = counts / max(1, counts.sum())
    return {
        "counts": {str(idx): int(value) for idx, value in enumerate(counts)},
        "fractions": {
            str(idx): float(value) for idx, value in enumerate(fractions)
        },
        "material_action_levels_at_10pct": int(np.sum(fractions >= 0.10)),
    }


def _guardrail_deltas(
    panel: Mapping[str, np.ndarray], selected: np.ndarray, static_index: int
) -> dict[str, float]:
    tapes = np.arange(len(selected))
    keys = (*SAFE_HIGHER, *SAFE_LOWER, *SAFE_EQUAL)
    return {
        key: float(
            np.mean(
                panel[key][tapes, selected] - panel[key][:, static_index]
            )
        )
        for key in keys
    }


def observed_profile(panel: Mapping[str, np.ndarray]) -> dict[str, Any]:
    ret = panel["ret_visible"]
    static_index = select_static(ret)
    raw_indices = np.argmax(ret, axis=1).astype(np.int32)
    safe_indices = safe_oracle_indices(panel, static_index)
    tapes = np.arange(ret.shape[0])
    raw_deltas = ret[tapes, raw_indices] - ret[:, static_index]
    safe_deltas = ret[tapes, safe_indices] - ret[:, static_index]
    safe_counts = Counter(map(int, safe_indices))
    modal = max(safe_counts.values()) if safe_counts else 0
    return {
        "best_static_calendar_index": int(static_index),
        "best_static_calendar": full_action_calendars()[static_index].astype(int).tolist(),
        "best_static_mean_ret": float(ret[:, static_index].mean()),
        "raw_h_pi": float(raw_deltas.mean()),
        "safe_h_pi": float(safe_deltas.mean()),
        "raw_h_pi_per_tape": raw_deltas.tolist(),
        "safe_h_pi_per_tape": safe_deltas.tolist(),
        "raw_oracle_indices": raw_indices.astype(int).tolist(),
        "safe_oracle_indices": safe_indices.astype(int).tolist(),
        "raw_favorable_tapes": int(np.sum(raw_deltas > 1e-15)),
        "safe_favorable_tapes": int(np.sum(safe_deltas > 1e-15)),
        "unique_raw_oracle_calendars": int(len(set(map(int, raw_indices)))),
        "unique_safe_oracle_calendars": int(len(safe_counts)),
        "modal_safe_oracle_fraction": float(modal / ret.shape[0]),
        "safe_action_support": _action_support(safe_indices),
        "raw_guardrail_mean_deltas": _guardrail_deltas(
            panel, raw_indices, static_index
        ),
        "safe_guardrail_mean_deltas": _guardrail_deltas(
            panel, safe_indices, static_index
        ),
    }


def bootstrap_counts(
    *, n_tapes: int, resamples: int, rng_seed: int
) -> np.ndarray:
    rng = np.random.default_rng(int(rng_seed))
    draws = rng.integers(0, n_tapes, size=(int(resamples), n_tapes))
    counts = np.zeros((int(resamples), n_tapes), dtype=np.uint8)
    for row, draw in enumerate(draws):
        counts[row] = np.bincount(draw, minlength=n_tapes)
    return counts


def bootstrap_static_indices(
    ret: np.ndarray, counts: np.ndarray, *, chunk_size: int = 64
) -> np.ndarray:
    selected = np.empty(len(counts), dtype=np.int32)
    for start in range(0, len(counts), chunk_size):
        stop = min(len(counts), start + chunk_size)
        weighted = counts[start:stop].astype(float) @ ret
        selected[start:stop] = np.argmax(weighted, axis=1).astype(np.int32)
    return selected


def bootstrap_profile(
    panel: Mapping[str, np.ndarray], counts: np.ndarray
) -> dict[str, Any]:
    ret = panel["ret_visible"]
    n_tapes = ret.shape[0]
    static_indices = bootstrap_static_indices(ret, counts)
    raw_row_max = ret.max(axis=1)
    raw_values = np.empty(len(counts), dtype=float)
    safe_values = np.empty(len(counts), dtype=float)
    safe_delta_cache: dict[int, np.ndarray] = {}
    for static_index in sorted(set(map(int, static_indices))):
        safe_indices = safe_oracle_indices(panel, static_index)
        tapes = np.arange(n_tapes)
        safe_delta_cache[static_index] = (
            ret[tapes, safe_indices] - ret[:, static_index]
        )
    for draw, static_index in enumerate(static_indices):
        weights = counts[draw].astype(float)
        raw_values[draw] = float(
            weights @ (raw_row_max - ret[:, int(static_index)]) / n_tapes
        )
        safe_values[draw] = float(
            weights @ safe_delta_cache[int(static_index)] / n_tapes
        )
    return {
        "raw_values": raw_values,
        "safe_values": safe_values,
        "static_index_distribution": {
            str(index): int(count)
            for index, count in sorted(Counter(map(int, static_indices)).items())
        },
        "raw_distribution_sha256": hashlib.sha256(raw_values.tobytes()).hexdigest(),
        "safe_distribution_sha256": hashlib.sha256(safe_values.tobytes()).hexdigest(),
    }


def _connected_primary_component(
    passing_cell_ids: set[str], contract: Mapping[str, Any]
) -> list[list[str]]:
    cells = {str(cell["cell_id"]): cell for cell in contract["primary_cells"]}
    remaining = set(passing_cell_ids)
    components: list[list[str]] = []
    while remaining:
        pending = [min(remaining)]
        remaining.remove(pending[0])
        component: list[str] = []
        while pending:
            current = pending.pop()
            component.append(current)
            source = cells[current]
            neighbors = {
                other
                for other in remaining
                if sum(
                    (
                        float(source[axis])
                        != float(cells[other][axis])
                    )
                    for axis in ("regime_persistence", "dominant_share")
                )
                == 1
            }
            remaining -= neighbors
            pending.extend(sorted(neighbors))
        components.append(sorted(component))
    return sorted(components)


def _structural_pass(row: Mapping[str, Any], rule: Mapping[str, Any]) -> bool:
    return bool(
        int(row["safe_favorable_tapes"]) >= int(rule["minimum_favorable_tapes"])
        and int(row["unique_safe_oracle_calendars"])
        >= int(rule["minimum_unique_safe_oracle_calendars"])
        and float(row["modal_safe_oracle_fraction"])
        <= float(rule["maximum_modal_safe_oracle_fraction"])
        and int(row["safe_action_support"]["material_action_levels_at_10pct"])
        >= int(rule["minimum_material_action_levels"])
    )


def analyze(
    *,
    contract_path: Path,
    output_root: Path,
    stage: str,
    seeds: tuple[int, ...],
    shard_manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text())
    profiles = frozen_profiles(contract)
    observed: dict[str, dict[str, Any]] = {}
    panels: dict[str, dict[str, np.ndarray]] = {}
    null_rows: list[dict[str, Any]] = []
    for profile in profiles:
        panel = load_profile_panel(
            output_root, stage, str(profile["profile_id"]), seeds
        )
        if profile["role"] == "exact_null":
            equality = {
                key: bool(np.all(values == values[:, :1]))
                for key, values in panel.items()
            }
            null_rows.append(
                {
                    **profile,
                    "all_calendar_matrices_identical": bool(all(equality.values())),
                    "matrix_identity": equality,
                    "raw_h_pi": 0.0,
                    "safe_h_pi": 0.0,
                }
            )
            continue
        panels[str(profile["profile_id"])] = panel
        observed[str(profile["profile_id"])] = {
            **profile,
            **observed_profile(panel),
        }

    simultaneous: dict[str, Any] | None = None
    if stage == "validation":
        rule = contract["validation_pass_rule"]
        bootstrap = rule["bootstrap"]
        counts = bootstrap_counts(
            n_tapes=len(seeds),
            resamples=int(bootstrap["resamples"]),
            rng_seed=int(bootstrap["rng_seed"]),
        )
        promotion_ids = [
            profile["profile_id"]
            for profile in profiles
            if profile["role"] in {"primary", "ordering_sensitivity"}
        ]
        distributions: dict[str, dict[str, Any]] = {}
        raw_errors = []
        safe_errors = []
        for profile_id in promotion_ids:
            distribution = bootstrap_profile(panels[profile_id], counts)
            distributions[profile_id] = distribution
            raw_errors.append(
                float(observed[profile_id]["raw_h_pi"])
                - distribution["raw_values"]
            )
            safe_errors.append(
                float(observed[profile_id]["safe_h_pi"])
                - distribution["safe_values"]
            )
        raw_critical = float(
            np.quantile(np.max(np.stack(raw_errors), axis=0), 0.95, method="higher")
        )
        safe_critical = float(
            np.quantile(np.max(np.stack(safe_errors), axis=0), 0.95, method="higher")
        )
        for profile_id in promotion_ids:
            observed[profile_id]["simultaneous_raw_lcb95"] = float(
                observed[profile_id]["raw_h_pi"] - raw_critical
            )
            observed[profile_id]["simultaneous_safe_lcb95"] = float(
                observed[profile_id]["safe_h_pi"] - safe_critical
            )
            observed[profile_id]["bootstrap_static_index_distribution"] = distributions[
                profile_id
            ]["static_index_distribution"]
            observed[profile_id]["bootstrap_raw_distribution_sha256"] = distributions[
                profile_id
            ]["raw_distribution_sha256"]
            observed[profile_id]["bootstrap_safe_distribution_sha256"] = distributions[
                profile_id
            ]["safe_distribution_sha256"]
        simultaneous = {
            "method": bootstrap["method"],
            "resamples": int(bootstrap["resamples"]),
            "rng_seed": int(bootstrap["rng_seed"]),
            "paired_tape_counts_sha256": hashlib.sha256(counts.tobytes()).hexdigest(),
            "promotion_profile_ids": promotion_ids,
            "static_reselected_over_65536_in_every_resample": True,
            "safe_set_recomputed_for_every_selected_static": True,
            "raw_critical_max_error": raw_critical,
            "safe_critical_max_error": safe_critical,
        }

    primary_rows = [row for row in observed.values() if row["role"] == "primary"]
    ordering_rows = [
        row for row in observed.values() if row["role"] == "ordering_sensitivity"
    ]
    null_pass = bool(null_rows and all(row["all_calendar_matrices_identical"] for row in null_rows))
    if stage == "screen":
        rule = contract["screen_pass_rule"]
        passing_cells = {
            str(row["cell_id"])
            for row in primary_rows
            if float(row["safe_h_pi"]) >= 0.015
            and _structural_pass(row, rule)
        }
        ordering_pass = all(float(row["safe_h_pi"]) >= 0.015 for row in ordering_rows)
    else:
        rule = contract["validation_pass_rule"]
        passing_cells = {
            str(row["cell_id"])
            for row in primary_rows
            if float(row["simultaneous_safe_lcb95"])
            >= float(rule["simultaneous_familywise_one_sided_lcb95_safe_h_pi_minimum"])
            and _structural_pass(row, rule)
        }
        ordering_pass = all(
            float(row["simultaneous_safe_lcb95"])
            >= float(rule["all_ordering_sensitivity_safe_lcb95_minimum"])
            for row in ordering_rows
        )
    components = _connected_primary_component(passing_cells, contract)
    eligible_components = []
    by_id = {str(cell["cell_id"]): cell for cell in contract["primary_cells"]}
    minimum_cells = int(
        rule.get(
            "connected_primary_component_minimum_cells",
            rule.get("primary_connected_cells_with_mean_safe_h_pi_at_least_0_015", 3),
        )
    )
    for component in components:
        cells = [by_id[cell_id] for cell_id in component]
        if (
            len(component) >= minimum_cells
            and len({float(cell["regime_persistence"]) for cell in cells}) >= 2
            and len({float(cell["dominant_share"]) for cell in cells}) >= 2
        ):
            eligible_components.append(component)
    raw_region_pass = any(
        float(row.get("simultaneous_raw_lcb95", row["raw_h_pi"])) >= 0.01
        for row in primary_rows
    )
    passed = bool(eligible_components and ordering_pass and null_pass)
    labels = contract["terminal_labels"]
    if passed:
        status = labels["pass"]
    elif raw_region_pass and not eligible_components:
        status = labels["raw_pass_safe_fail"]
    elif not ordering_pass:
        status = labels["ordering_fail"]
    else:
        status = labels["other_scientific_fail"]
    result: dict[str, Any] = {
        "schema_version": f"program_o_gate_o0_{stage}_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "stage": stage,
        "scientific_commit": git_commit(),
        "contract": str(contract_path.relative_to(ROOT)),
        "contract_sha256": sha256(contract_path),
        "metric": contract["metric"],
        "seeds": list(seeds),
        "calendar_count": int(contract["complete_open_loop_calendars"]),
        "profile_count": len(profiles),
        "raw_calendar_evaluations": len(profiles) * len(seeds) * int(contract["complete_open_loop_calendars"]),
        "profiles": [observed[key] for key in sorted(observed)],
        "null_profiles": sorted(null_rows, key=lambda row: row["profile_id"]),
        "passing_primary_cell_ids": sorted(passing_cells),
        "primary_components": components,
        "eligible_primary_components": eligible_components,
        "ordering_sensitivity_pass": ordering_pass,
        "exact_null_pass": null_pass,
        "simultaneous_inference": simultaneous,
        "shard_count": len(shard_manifest),
        "shard_manifest_sha256": digest_json(shard_manifest),
        "claim_boundary": {
            "gate_o0_full_action_transducer_passed": passed,
            "full_des_implementation_freeze_allowed": passed,
            "full_des_h_pi_established": False,
            "h_obs_authorized": False,
            "learner_authorized": False,
            "paper3_authorized": False,
        },
    }
    result["content_sha256"] = digest_json(result)
    return result


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def execute_stage(
    *,
    contract_path: Path,
    output_root: Path,
    stage: str,
    workers: int,
    validation_freeze_path: Path | None = None,
) -> Path:
    contract = json.loads(contract_path.read_text())
    stage_contract = contract["tape_blocks"][stage]
    if stage == "screen":
        if stage_contract["status"] != "SEALED_NOT_OPENED":
            raise RuntimeError(f"stage is not authorized: {stage_contract['status']}")
    else:
        freeze_path = validation_freeze_path or DEFAULT_VALIDATION_FREEZE
        if not freeze_path.is_file():
            raise RuntimeError(
                "validation remains sealed: additive validation freeze is absent"
            )
        freeze = json.loads(freeze_path.read_text())
        screen_result_path = output_root / "screen/result.json"
        if not screen_result_path.is_file():
            raise RuntimeError("validation remains sealed: screen result is absent")
        screen_result = json.loads(screen_result_path.read_text())
        expected_range = list(map(int, stage_contract["range"]))
        failures = []
        if freeze.get("status") != "AUTHORIZED_PROGRAM_O_GATE_O0_VALIDATION":
            failures.append("freeze status does not authorize validation")
        if freeze.get("contract_sha256") != sha256(contract_path):
            failures.append("freeze contract hash mismatch")
        if freeze.get("screen_result_sha256") != sha256(screen_result_path):
            failures.append("freeze screen result hash mismatch")
        if freeze.get("validation_seed_range") != expected_range:
            failures.append("freeze validation seed range mismatch")
        if screen_result.get("status") != contract["terminal_labels"]["pass"]:
            failures.append("screen verdict did not pass")
        if screen_result.get("claim_boundary", {}).get(
            "full_des_implementation_freeze_allowed"
        ) is not True:
            failures.append("screen claim boundary does not permit continuation")
        if failures:
            raise RuntimeError("validation remains sealed: " + "; ".join(failures))
    start, end = map(int, stage_contract["range"])
    seeds = tuple(range(start, end + 1))
    stage_root = output_root / stage
    if stage_root.exists():
        raise FileExistsError(f"refusing to overwrite {stage_root}")
    stage_root.mkdir(parents=True)
    profiles = frozen_profiles(contract)
    source_paths = (
        contract_path,
        Path(__file__).resolve(),
        ROOT / "scripts/screen_program_o_exact_transducer.py",
        ROOT / "supply_chain/ret_thesis.py",
        ROOT / "supply_chain/supply_chain.py",
    )
    run_manifest = {
        "schema_version": "program_o_gate_o0_run_manifest_v1",
        "stage": stage,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_commit": git_commit(),
        "contract_sha256": sha256(contract_path),
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path) for path in source_paths
        },
        "python": sys.version,
        "numpy": np.__version__,
        "workers": int(workers),
        "seeds": list(seeds),
        "profile_ids": [profile["profile_id"] for profile in profiles],
        "calendar_count": int(contract["complete_open_loop_calendars"]),
    }
    _write_json_atomic(stage_root / "run_manifest.json", run_manifest)
    watcher = stage_root / "watcher_state.jsonl"
    tasks = [
        (str(profile["profile_id"]), seed)
        for profile in profiles
        for seed in seeds
    ]
    completed: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        futures = {
            executor.submit(
                produce_shard,
                str(contract_path),
                stage,
                profile_id,
                seed,
                str(output_root),
            ): (profile_id, seed)
            for profile_id, seed in tasks
        }
        for future in as_completed(futures):
            row = future.result()
            completed.append(row)
            heartbeat = {
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "stage": stage,
                "completed_shards": len(completed),
                "total_shards": len(tasks),
                "last_profile_id": row["profile_id"],
                "last_seed": row["seed"],
                "last_artifact_sha256": row["sha256"],
            }
            with watcher.open("a") as handle:
                handle.write(json.dumps(heartbeat, sort_keys=True) + "\n")
    completed.sort(key=lambda row: (row["profile_id"], row["seed"]))
    _write_json_atomic(stage_root / "raw_shard_manifest.json", completed)
    result = analyze(
        contract_path=contract_path,
        output_root=output_root,
        stage=stage,
        seeds=seeds,
        shard_manifest=completed,
    )
    _write_json_atomic(stage_root / "result.json", result)
    checksums = []
    for path in sorted(stage_root.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            checksums.append(f"{sha256(path)}  {path.relative_to(stage_root)}")
    (stage_root / "checksums.sha256").write_text("\n".join(checksums) + "\n")
    return stage_root / "result.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage", choices=("screen", "validation"), required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument(
        "--validation-freeze",
        type=Path,
        default=None,
        help="Committed additive authorization; mandatory for validation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = execute_stage(
        contract_path=args.contract.resolve(),
        output_root=args.output_root.resolve(),
        stage=str(args.stage),
        workers=int(args.workers),
        validation_freeze_path=(
            args.validation_freeze.resolve() if args.validation_freeze else None
        ),
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
