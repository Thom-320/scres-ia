#!/usr/bin/env python3
"""Corrected exact two-product transducer and complete Program O screen."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supply_chain.ret_thesis import (  # noqa: E402
    compute_order_level_ret_excel_formula,
    compute_order_level_ret_excel_request_snapshot_ledger,
)
from supply_chain.supply_chain import OrderRecord  # noqa: E402

DEFAULT_CONTRACT = ROOT / "contracts/program_o_exact_transducer_v1.json"
DEFAULT_OUTPUT = ROOT / "results/program_o/exact_transducer_screen_v1/result.json"
PRODUCTS = ("P_C", "P_H")


def json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Tape:
    seed: int
    regimes: tuple[str, ...]
    order_products: tuple[str, ...]
    sha256: str


def make_tape(seed: int, *, persistence: float, dominant_share: float) -> Tape:
    rng = np.random.default_rng(seed)
    regime = "P_C" if int(rng.integers(0, 2)) == 0 else "P_H"
    regimes: list[str] = []
    order_products: list[str] = []
    for _week in range(8):
        regimes.append(regime)
        for _day in range(6):
            dominant = bool(rng.random() < dominant_share)
            order_products.append(regime if dominant else PRODUCTS[1 - PRODUCTS.index(regime)])
        if rng.random() > persistence:
            regime = PRODUCTS[1 - PRODUCTS.index(regime)]
    raw = {"seed": seed, "regimes": regimes, "order_products": order_products}
    return Tape(seed, tuple(regimes), tuple(order_products), json_sha256(raw))


def complete_calendars() -> tuple[tuple[str, ...], ...]:
    return tuple(product(("C_MAJOR", "H_MAJOR"), repeat=8))


def _allocate(order: OrderRecord, amount: float, allocation_time: float) -> float:
    take = min(float(order.remaining_qty), float(amount))
    order.remaining_qty -= take
    if order.remaining_qty <= 1e-9:
        order.remaining_qty = 0.0
        order.OATj = max(float(order.OPTj) + float(order.LTj), allocation_time + 48.0)
        order.CTj = float(order.OATj) - float(order.OPTj)
        order.backorder = bool(float(order.CTj) > float(order.LTj))
        order.DPj = float(order.CTj) if order.backorder else 0.0
    return take


def simulate(
    tape: Tape,
    calendar: tuple[str, ...],
    contract: dict[str, Any],
    *,
    complete_substitution: bool,
) -> dict[str, Any]:
    inventories = {key: float(value) for key, value in contract["initial_inventory"].items()}
    pool = sum(inventories.values()) if complete_substitution else 0.0
    orders: list[OrderRecord] = []
    events: list[tuple[float, int, str, str | None]] = []
    for week, action in enumerate(calendar):
        base = week * 168.0
        for offset, product_id in zip((24.0, 72.0, 120.0), contract["actions"][action], strict=True):
            events.append((base + offset, 0, "batch", product_id))
        for day, offset in enumerate((30.0, 54.0, 78.0, 102.0, 126.0, 150.0)):
            events.append((base + offset, 1, "demand", tape.order_products[week * 6 + day]))
    for now, _priority, kind, product_id in sorted(events):
        if kind == "batch":
            available = float(contract["batch_quantity"])
            if complete_substitution:
                pool += available
                for order in orders:
                    if pool <= 1e-9:
                        break
                    if order.remaining_qty > 0:
                        used = _allocate(order, pool, now)
                        pool -= used
            else:
                for order in orders:
                    if available <= 1e-9:
                        break
                    if getattr(order, "requested_product_id") == product_id and order.remaining_qty > 0:
                        used = _allocate(order, available, now)
                        available -= used
                inventories[str(product_id)] += available
            continue

        bt = sum(
            1
            for prior in orders
            if float(prior.OPTj) + float(prior.LTj) <= now
            and (prior.OATj is None or float(prior.OATj) > now)
        )
        order = OrderRecord(
            j=len(orders) + 1,
            OPTj=now,
            quantity=float(contract["daily_order_quantity"]),
            remaining_qty=float(contract["daily_order_quantity"]),
            ret_bt_at_request=bt,
            ret_ut_at_request=0,
            ret_ledger_snapshot_time=now,
            ret_ledger_event_sequence=len(orders) + 1,
        )
        setattr(order, "requested_product_id", product_id)
        available = pool if complete_substitution else inventories[str(product_id)]
        used = _allocate(order, available, now - 48.0)
        if complete_substitution:
            pool -= used
        else:
            inventories[str(product_id)] -= used
        orders.append(order)

    visible = compute_order_level_ret_excel_request_snapshot_ledger(orders)
    full = compute_order_level_ret_excel_formula(orders)
    demand_by_product = {
        product_id: sum(o.quantity for o in orders if getattr(o, "requested_product_id") == product_id)
        for product_id in PRODUCTS
    }
    served_by_product = {
        product_id: sum(o.quantity for o in orders if getattr(o, "requested_product_id") == product_id and o.OATj is not None)
        for product_id in PRODUCTS
    }
    fill_by_product = {
        product_id: (
            served_by_product[product_id] / demand_by_product[product_id]
            if demand_by_product[product_id] > 0 else 1.0
        )
        for product_id in PRODUCTS
    }
    return {
        "ret": float(visible["mean_ret_excel"]),
        "ret_full": float(full["mean_ret_excel"]),
        "visible_rows": int(visible["n_visible_rows"]),
        "unfulfilled_orders": sum(o.OATj is None for o in orders),
        "unfulfilled_quantity": sum(o.remaining_qty for o in orders),
        "worst_product_fill": min(fill_by_product.values()),
        "ending_inventory": (pool if complete_substitution else sum(inventories.values())),
        "production_batches": 24,
        "production_quantity": 120000.0,
    }


def evaluate_cell(contract: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    calendars = complete_calendars()
    complete_substitution = bool(cell.get("complete_substitution", False))
    matrices: dict[str, list[list[float]]] = {
        key: [] for key in ("ret", "ret_full", "unfulfilled_quantity", "worst_product_fill")
    }
    tape_hashes: list[str] = []
    seeds = range(contract["screen"]["seeds"][0], contract["screen"]["seeds"][1] + 1)
    for seed in seeds:
        tape = make_tape(
            seed,
            persistence=float(cell["regime_persistence"]),
            dominant_share=float(cell["dominant_share"]),
        )
        tape_hashes.append(tape.sha256)
        rows = [simulate(tape, calendar, contract, complete_substitution=complete_substitution) for calendar in calendars]
        for key in matrices:
            matrices[key].append([float(row[key]) for row in rows])
    arrays = {key: np.asarray(value, dtype=float) for key, value in matrices.items()}
    mean_ret = arrays["ret"].mean(axis=0)
    best_static_index = int(np.flatnonzero(np.isclose(mean_ret, mean_ret.max(), atol=1e-15, rtol=0))[0])
    oracle_indices = arrays["ret"].argmax(axis=1)
    tape_indices = np.arange(len(tape_hashes))
    oracle_ret = arrays["ret"][tape_indices, oracle_indices]
    static_ret = arrays["ret"][:, best_static_index]
    result = {
        "cell_id": cell["cell_id"],
        "complete_substitution": complete_substitution,
        "n_tapes": len(tape_hashes),
        "calendar_count": len(calendars),
        "tape_hashes": tape_hashes,
        "best_static_index": best_static_index,
        "best_static_calendar": list(calendars[best_static_index]),
        "best_static_mean_ret": float(static_ret.mean()),
        "oracle_mean_ret": float(oracle_ret.mean()),
        "mean_h_pi": float(np.mean(oracle_ret - static_ret)),
        "favorable_tapes": int(np.sum(oracle_ret > static_ret + 1e-15)),
        "unique_oracle_calendars": int(len(set(int(i) for i in oracle_indices))),
        "oracle_minus_static_unfulfilled_quantity": float(
            np.mean(arrays["unfulfilled_quantity"][tape_indices, oracle_indices] - arrays["unfulfilled_quantity"][:, best_static_index])
        ),
        "oracle_minus_static_worst_product_fill": float(
            np.mean(arrays["worst_product_fill"][tape_indices, oracle_indices] - arrays["worst_product_fill"][:, best_static_index])
        ),
        "oracle_minus_static_full_ret": float(
            np.mean(arrays["ret_full"][tape_indices, oracle_indices] - arrays["ret_full"][:, best_static_index])
        ),
        "all_calendar_ret_identical": bool(np.all(arrays["ret"] == arrays["ret"][:, :1])),
        "all_calendar_guardrails_identical": bool(
            np.all(arrays["unfulfilled_quantity"] == arrays["unfulfilled_quantity"][:, :1])
            and np.all(arrays["worst_product_fill"] == arrays["worst_product_fill"][:, :1])
        ),
    }
    return result


def produce(contract_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text())
    cells = [*contract["positive_cells"], contract["null_cell"]]
    results = [evaluate_cell(contract, cell) for cell in cells]
    positive = [row for row in results if not row["complete_substitution"]]
    passing_ids = sorted(
        row["cell_id"]
        for row in positive
        if row["mean_h_pi"] >= 0.015
        and row["oracle_minus_static_unfulfilled_quantity"] <= 1e-9
        and row["oracle_minus_static_worst_product_fill"] >= -1e-12
    )
    adjacency = {
        "rho75_share75": {"rho75_share90", "rho90_share75"},
        "rho75_share90": {"rho75_share75", "rho90_share90"},
        "rho90_share75": {"rho75_share75", "rho90_share90"},
        "rho90_share90": {"rho75_share90", "rho90_share75"},
    }
    adjacent_pair = any(b in adjacency[a] for a in passing_ids for b in passing_ids if a != b)
    null = next(row for row in results if row["complete_substitution"])
    null_pass = bool(null["all_calendar_ret_identical"] and null["all_calendar_guardrails_identical"] and null["mean_h_pi"] == 0.0)
    passed = bool(adjacent_pair and null_pass)
    payload: dict[str, Any] = {
        "schema_version": "program_o_exact_transducer_screen_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "PASS_SCREEN_ONLY__FREEZE_FRESH_VALIDATION_BEFORE_OPENING"
            if passed else "STOP_PROGRAM_O_EXACT_TRANSDUCER_SCREEN"
        ),
        "contract": str(contract_path.relative_to(ROOT)),
        "contract_sha256": sha256(contract_path),
        "metric": contract["metric"],
        "cells": results,
        "passing_cell_ids": passing_ids,
        "adjacent_passing_pair_exists": adjacent_pair,
        "null_pass": null_pass,
        "claim_boundary": {
            "screen_only": True,
            "h_pi_validated": False,
            "full_des_authorized": False,
            "h_obs_authorized": False,
            "learner_authorized": False,
            "paper3_authorized": False,
        },
    }
    payload["content_sha256"] = json_sha256(payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    payload = produce(args.contract.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
