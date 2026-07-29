#!/usr/bin/env python3
"""Audit the R2r tape whose raw ret_excel delta dominates v2's MPC mean.

The audit replays the frozen incumbent and recorded MPC action sequence on tape
1530011 under the cadence-corrected RPj implementation. It persists the visible
population and per-order contributions so a metric tail cannot be described as a
physical controller failure without evidence.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.ret_thesis import (  # noqa: E402
    compute_order_level_ret_excel_request_snapshot_ledger,
)
from scripts.run_expanded_contract_comparators_v2 import (  # noqa: E402
    apply_posture,
    make_replay_sim,
    materialize_tape,
)

SEED = 1_530_011
FAMILY = "R2r"
INCUMBENT = (336, 0, 168)
SOURCE_SHARD = Path(
    "/Users/thom/Projects/research/scres-ia-runs/"
    "expanded_contract_comparators_v2_1dc40c1_r2/full/shards/"
    "R2r_1530011.json")


def canonical_sha(payload: Any) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def replay(name: str, postures: list[tuple[int, int, int]]) -> dict[str, Any]:
    horizon = float(52 * HOURS_PER_WEEK)
    epoch = float(4 * HOURS_PER_WEEK)
    tape = materialize_tape(SEED, horizon, FAMILY)
    sim = make_replay_sim(
        seed=SEED, horizon=horizon, family=FAMILY, tape=tape)
    for posture in postures:
        apply_posture(sim, posture)
        sim.step(
            action=None,
            step_hours=min(epoch, horizon - float(sim.env.now)),
        )
    orders = [
        order for order in sim.orders
        if not bool(getattr(order, "metrics_excluded", False))
        and float(getattr(order, "OPTj", 0.0)) >= float(sim.warmup_time)
    ]
    orders_by_j = {int(order.j): order for order in orders}
    ledger = compute_order_level_ret_excel_request_snapshot_ledger(
        orders, current_time=horizon)
    rows = []
    for row in ledger["ret_rows"]:
        order = orders_by_j[int(row["j"])]
        rows.append({
            **row,
            "APj": float(order.APj or 0.0),
            "RPj": float(order.RPj or 0.0),
            "CTj": float(order.CTj or 0.0),
            "LTj": float(order.LTj or 0.0),
            "risk_indicators": dict(order.ret_risk_indicators),
        })
    values = np.asarray([row["ret"] for row in rows], dtype=float)
    max_index = int(np.argmax(values))
    max_row = rows[max_index]
    without_max = np.delete(values, max_index)
    metrics = compute_episode_metrics(sim)
    return {
        "arm": name,
        "ret_excel": float(ledger["mean_ret_excel"]),
        "ret_excel_clipped_0_1": float(np.mean(np.clip(values, 0.0, 1.0))),
        "ret_excel_leave_max_one_out": float(np.mean(without_max)),
        "max_order_contribution": max_row,
        "max_order_share_of_ret_sum": float(max_row["ret"] / values.sum()),
        "n_generated_orders": int(ledger["n_generated_orders"]),
        "n_visible_rows": int(ledger["n_visible_rows"]),
        "n_omitted_rows": int(ledger["n_omitted_rows"]),
        "case_counts": ledger["case_counts"],
        "physical_endpoints": {
            key: float(metrics[key])
            for key in (
                "flow_fill_rate", "delivered_rations", "lost_orders",
                "backorder_qty_final", "ret_excel_full_ledger", "ret_thesis")
        },
        "rows": rows,
        "rows_sha256": canonical_sha(rows),
    }


def main() -> int:
    shard = json.loads(SOURCE_SHARD.read_text())
    mpc_sequence = [
        tuple(int(x) for x in posture)
        for row in shard["rows"] if row["arm"] == "replay_mpc_v2"
        for posture in row["action_sequence"]
    ]
    static = replay("static_incumbent", [INCUMBENT] * len(mpc_sequence))
    mpc = replay("replay_mpc_v2", mpc_sequence)
    payload = {
        "schema_version": "r2r_1530011_ret_tail_audit_v1",
        "claim_status": "DEVELOPMENT_METRIC_TAIL_DIAGNOSTIC",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_shard": str(SOURCE_SHARD),
        "source_shard_sha256": sha256(SOURCE_SHARD.read_bytes()).hexdigest(),
        "tape_seed": SEED,
        "rpj_semantics": "immutable_onset_corrective",
        "static": {k: v for k, v in static.items() if k != "rows"},
        "mpc": {k: v for k, v in mpc.items() if k != "rows"},
        "contrasts": {
            "raw_ret_excel_mpc_minus_static": (
                mpc["ret_excel"] - static["ret_excel"]),
            "clipped_0_1_mpc_minus_static": (
                mpc["ret_excel_clipped_0_1"]
                - static["ret_excel_clipped_0_1"]),
            "leave_max_one_out_mpc_minus_static": (
                mpc["ret_excel_leave_max_one_out"]
                - static["ret_excel_leave_max_one_out"]),
            "flow_fill_mpc_minus_static": (
                mpc["physical_endpoints"]["flow_fill_rate"]
                - static["physical_endpoints"]["flow_fill_rate"]),
        },
        "diagnosis": (
            "The raw negative MPC delta is a metric-tail event only if it "
            "disappears when the single unbounded static ReT contribution is "
            "reported separately; this audit does not redefine the primary endpoint."),
        "rows_sha256": canonical_sha({
            "static": static["rows"], "mpc": mpc["rows"]}),
    }
    payload["self_sha256"] = canonical_sha(payload)
    out = Path("results/metric_audit/r2r_1530011_ret_tail_v1")
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n")
    (out / "per_order_rows.json").write_text(json.dumps(
        {"static": static["rows"], "mpc": mpc["rows"]},
        indent=1, sort_keys=True) + "\n")
    print(f"-> {out / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
