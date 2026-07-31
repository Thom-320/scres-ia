#!/usr/bin/env python3
"""Custody for the ReT out-of-range tail: family sweep and per-arm warm-up.

These numbers were first computed ad hoc and quoted in prose. Review correctly
objected that prose is not custody, so this regenerates every one from scratch with
per-order rows and a hash.

What it persists:

* the per-family sweep over all 24 v2 tapes at each family's true 216-posture
  incumbent -- scored orders, count above 1.0, maximum, and the inflation of the mean
  against the same values clipped into the range the metric declares;
* the monotonicity check that motivates the whole thing -- the most-delayed order
  against the highest-scoring one, which are not the same order;
* the per-arm warm-up on tape 1530011, because the warm-up is endogenous and differs
  by arm, so policy-dependent censoring sits on top of the tail.

Scope is stated rather than implied: this is the `ReT > 1` tail only, at the static
incumbent, on these tapes. It absolves no other arm, posture, or defect.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.provenance import calibration_stamp  # noqa: E402
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.episode_metrics import (  # noqa: E402
    compute_order_level_ret_excel_request_snapshot_ledger as ledger,
)
from scripts.run_expanded_contract_comparators_v2 import (  # noqa: E402
    apply_posture,
    make_replay_sim,
    materialize_tape,
)

INCUMBENTS = {"R1r": (0, 0, 336), "R2r": (336, 0, 168)}
RUN_DIRS = {"R1r": "expanded_contract_comparators_v2_1dc40c1_r1",
            "R2r": "expanded_contract_comparators_v2_1dc40c1_r2"}
TAIL_TAPE = 1_530_011


def replay(*, seed: int, family: str, horizon: float, epoch_hours: float,
           postures: list[tuple[int, int, int]]):
    tape = materialize_tape(seed, horizon, family)
    sim = make_replay_sim(seed=seed, horizon=horizon, family=family, tape=tape)
    elapsed, epoch, since = 0.0, 0, float("inf")
    while elapsed < horizon:
        if since >= epoch_hours:
            apply_posture(sim, postures[min(epoch, len(postures) - 1)])
            since, epoch = 0.0, epoch + 1
        step = min(epoch_hours, horizon - elapsed)
        sim.step(action=None, step_hours=step)
        elapsed += step
        since += step
    return sim


def scored(sim) -> list:
    return [o for o in sim.orders
            if not bool(getattr(o, "metrics_excluded", False))
            and float(getattr(o, "OPTj", 0.0)) >= float(sim.warmup_time)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path,
                    default=Path(__file__).resolve().parents[2] / "scres-ia-runs")
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--epoch-weeks", type=int, default=4)
    ap.add_argument("--output", type=Path,
                    default=Path("results/metric_audit/ret_tail_family_sweep_v1/"
                                 "result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    epoch_hours = float(args.epoch_weeks * HOURS_PER_WEEK)
    started = time.perf_counter()

    families: dict[str, dict] = {}
    all_rows: dict[str, list] = {}
    for family, incumbent in INCUMBENTS.items():
        shards = sorted((args.runs_root / RUN_DIRS[family] / "full" / "shards")
                        .glob("*.json"))
        values: list[float] = []
        rows: list[dict] = []
        for path in shards:
            seed = int(json.loads(path.read_text())["tape_seed"])
            sim = replay(seed=seed, family=family, horizon=horizon,
                         epoch_hours=epoch_hours, postures=[incumbent] * 13)
            orders = scored(sim)
            book = ledger(orders, current_time=float(sim.env.now))
            for order, value in zip(orders, book["ret_values"]):
                # Unserved orders have CTj None; they still carry a ReT value
                # through the risk-no-recovery branch, so they belong in the sweep.
                rows.append({"tape_seed": seed, "j": int(getattr(order, "j", -1)),
                             "CTj": float(order.CTj) if order.CTj is not None
                             else float("nan"),
                             "LTj": float(order.LTj or 0.0),
                             "RPj": float(order.RPj or 0.0),
                             "APj": float(order.APj or 0.0), "ret": float(value)})
                values.append(float(value))
            print(f"  {family} {seed} ({time.perf_counter() - started:.0f}s)",
                  flush=True)

        v = np.array(values)
        served = [r for r in rows if r["CTj"] == r["CTj"]]  # drop NaN
        by_ct = max(served, key=lambda r: r["CTj"])
        by_ret = max(rows, key=lambda r: r["ret"])
        families[family] = {
            "incumbent": list(incumbent),
            "n_tapes": len(shards),
            "n_scored_orders": int(len(v)),
            "n_above_one": int((v > 1.0).sum()),
            "pct_above_one": float(100.0 * (v > 1.0).sum() / len(v)),
            "max_ret": float(v.max()),
            "mean_ret": float(v.mean()),
            "mean_ret_clipped_0_1": float(np.clip(v, 0.0, 1.0).mean()),
            "inflation_vs_clipped": float(v.mean() / np.clip(v, 0.0, 1.0).mean()),
            # The monotonicity check. These are different orders, which is the point:
            # the metric is not monotone in lateness in either direction.
            "most_delayed_order": by_ct,
            "highest_scoring_order": by_ret,
            "most_delayed_is_highest_scoring": by_ct["j"] == by_ret["j"]
            and by_ct["tape_seed"] == by_ret["tape_seed"],
            "scope": ("ReT > 1 tail only, at this family's static incumbent, on these "
                      "tapes; absolves no other arm, posture, or ReT defect"),
        }
        all_rows[family] = rows

    # Per-arm warm-up on the tail tape: censoring compounds on top of the tail.
    shard = json.loads((args.runs_root / RUN_DIRS["R2r"] / "full" / "shards"
                        / f"R2r_{TAIL_TAPE}.json").read_text())
    mpc_postures = [tuple(int(x) for x in e["selected_posture"])
                    for e in shard["traces"]["mpc"]]
    warmups = {}
    for label, postures in (("static_incumbent", [INCUMBENTS["R2r"]] * 13),
                            ("replay_mpc_v2", mpc_postures)):
        sim = replay(seed=TAIL_TAPE, family="R2r", horizon=horizon,
                     epoch_hours=epoch_hours, postures=postures)
        warmups[label] = {"warmup_time_hours": float(sim.warmup_time),
                          "n_scored_orders": len(scored(sim))}

    rows_blob = json.dumps(all_rows, indent=1, sort_keys=True) + "\n"
    rows_path = args.output.parent / "per_order_rows.json"
    payload = {
        "schema_version": "ret_tail_family_sweep_v1",
        "calibration_provenance": calibration_stamp(),
        "claim_status": "DEVELOPMENT_METRIC_DIAGNOSTIC_NO_METRIC_CHANGED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rpj_semantics": "immutable_onset_corrective (commit 125b94f)",
        "families": families,
        "tail_tape_warmup_by_arm": {
            "tape_seed": TAIL_TAPE,
            "arms": warmups,
            "note": ("the warm-up is endogenous and differs by arm, so the scored "
                     "populations differ too -- policy-dependent censoring sits on "
                     "top of the unbounded tail"),
        },
        "rows_path": str(rows_path),
        "rows_sha256": sha256(rows_blob.encode()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    body = json.dumps(payload, indent=1, sort_keys=True)
    payload["self_sha256"] = sha256(body.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(rows_blob)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"\n-> {args.output}\n-> {rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
