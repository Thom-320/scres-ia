#!/usr/bin/env python3
"""Fold the v2 comparator arms into the five-column metric panel, by replay.

The v2 shards record `ret_excel`, `ret_excel_full_ledger`, fill, lost, unresolved
and injected -- but not the five Cobb-Douglas output variables, which have to be
sampled off the ledger period by period and cannot be reconstructed after the fact.

They do record, per epoch, the posture each arm selected. That is enough: replaying
a fixed posture sequence is one episode with no 216-candidate search, so the whole
fold costs a few seconds rather than re-running the instrument. The v2 run
establishes what the MPC *chose*; this measures that choice on the full panel.

**The replay is gated on reproducing v2's own numbers.** Every replayed episode must
return the shard's `ret_excel`, `ret_excel_full_ledger`, `flow_fill_rate` and
`lost_orders` to within `TOL`. A single mismatch means the replay conditions differ
from v2's and the fold is refused rather than reported -- the exogenous tape is
re-materialised from the seed here, so a mismatch would most likely mean the tape
did not reproduce, which silently invalidates everything downstream.

Note the sampling difference the gate is there to police: v2 steps by `epoch_hours`
(4 weeks), this must step at the same cadence -- because `ret_excel` is NOT cadence-invariant.
Measured on one identical trajectory (same fill, same delivered, same risk events),
ret_excel runs 0.004369 at a single step and 0.005981 at hourly steps, a 37% spread,
because `RPj` differs in 175 of 311 orders while `OPTj`, `OATj` and `APj` do not.
The first version of this script stepped daily and the gate caught it: 24 of 24 arms
failed by ~29%. That is what the gate is for.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.cobb_douglas_resilience import (  # noqa: E402
    CobbDouglasRecorder,
    score_comparison_set,
)
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402

from scripts.build_metric_panel import POSTURES, SHIFTS  # noqa: E402
from scripts.run_expanded_contract_comparators_v2 import (  # noqa: E402
    apply_posture,
    make_replay_sim,
    materialize_tape,
)

TOL = 1e-9
# Shard trace name -> the arm name in the shard's `rows`.
ARMS = {
    "mpc": "replay_mpc_v2",
    "pi": "greedy_pi_best_found_v2",
    "ddmrp": "ddmrp_projected_v2",
}
GATE_KEYS = ("ret_excel", "ret_excel_full_ledger", "flow_fill_rate", "lost_orders")


def replay_arm(*, seed: int, family: str, horizon: float, epoch_hours: float,
               period_hours: float, postures: list[tuple[int, int, int]],
               tape: dict) -> dict:
    """Re-run one arm's recorded posture sequence, sampling C-D every period."""
    sim = make_replay_sim(seed=seed, horizon=horizon, family=family, tape=tape)
    rec = CobbDouglasRecorder(period_hours=period_hours)
    elapsed, epoch, since = 0.0, 0, float("inf")
    while elapsed < horizon:
        if since >= epoch_hours:
            # Past the recorded horizon the arm holds its last posture, which is
            # what v2's own `finish_with_posture` does.
            apply_posture(sim, postures[min(epoch, len(postures) - 1)])
            since, epoch = 0.0, epoch + 1
        step = min(period_hours, horizon - elapsed)
        sim.step(action=None, step_hours=step)
        elapsed += step
        since += step
        rec.sample(sim)

    agg = rec.aggregate()
    m = compute_episode_metrics(sim)
    agg.update({k: float(m[k]) for k in (
        "ret_excel", "ret_excel_full_ledger", "ret_thesis", "ret_excel_cvar10",
        "ret_excel_cvar05", "flow_fill_rate", "fill_rate_on_time", "lost_orders",
        "backorder_qty_final", "delivered_rations")})
    agg["strategic_injected"] = float(sim.total_strategic_raw_injected
                                      + sim.total_strategic_rations_injected)
    agg["distinct_postures"] = len({tuple(p) for p in postures})
    agg["posture_changes"] = sum(1 for a, b in zip(postures, postures[1:])
                                 if a != b)
    return agg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dirs", nargs="+", type=Path, required=True,
                    help="v2 output dirs; shards are read from <dir>/full/shards")
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--epoch-weeks", type=int, default=4)
    # MUST match v2's epoch cadence. ret_excel is step-cadence dependent (RPj is),
    # so a daily replay of an identical trajectory fails the gate by 29%.
    ap.add_argument("--period-hours", type=float, default=672.0)
    ap.add_argument("--contract", type=Path,
                    default=Path("contracts/cobb_douglas_calibration_v1.json"))
    ap.add_argument("--panel", type=Path,
                    default=Path("results/metric_panel/panel_v1.json"))
    ap.add_argument("--output", type=Path,
                    default=Path("results/metric_panel/panel_with_v2_arms.json"))
    ap.add_argument("--require-complete", type=int, default=0,
                    help="refuse unless each run dir has at least this many shards")
    args = ap.parse_args()

    contract = json.loads(args.contract.read_text())
    exponents = contract["exponents"]
    panel = json.loads(args.panel.read_text())
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    epoch_hours = float(args.epoch_weeks * HOURS_PER_WEEK)
    started = time.perf_counter()

    shard_paths: list[Path] = []
    for d in args.run_dirs:
        found = sorted((d / "full" / "shards").glob("*.json"))
        if len(found) < args.require_complete:
            raise SystemExit(
                f"{d} has {len(found)} shards, --require-complete asks for "
                f"{args.require_complete}. Refusing a partial fold.")
        shard_paths.extend(found)
    if not shard_paths:
        raise SystemExit("no shards found")

    by_family: dict[str, dict[str, list[dict]]] = {}
    gate_failures: list[dict] = []
    tapes_replayed = 0

    for path in shard_paths:
        shard = json.loads(path.read_text())
        seed = int(shard["tape_seed"])
        family = shard["rows"][0]["family"]
        rows_by_arm = {r["arm"]: r for r in shard["rows"]}
        tape = materialize_tape(seed, horizon, family)

        # The panel's own cells were evaluated on seeds 1,620,001-4; the v2 arms on
        # 1,430,001+ / 1,530,001+. Merging those two directly would compare arms
        # across DIFFERENT exogenous streams, and any gap could be tape luck rather
        # than control. So the reference postures are re-evaluated here on THIS
        # shard's materialized tape, making every comparison paired within tape.
        # v2 pins shifts=1, so only the S1 postures are reproducible on its tapes.
        for posture in POSTURES:
            name = f"{'/'.join(str(h) for h in posture)}|S1"
            got = replay_arm(seed=seed, family=family, horizon=horizon,
                             epoch_hours=epoch_hours,
                             period_hours=args.period_hours,
                             postures=[posture] * len(shard["traces"]["mpc"]),
                             tape=tape)
            by_family.setdefault(family, {}).setdefault(name, []).append(got)

        for trace_key, arm in ARMS.items():
            postures = [tuple(int(x) for x in e["posture" if trace_key == "ddmrp"
                                                else "selected_posture"])
                        for e in shard["traces"][trace_key]]
            got = replay_arm(seed=seed, family=family, horizon=horizon,
                             epoch_hours=epoch_hours,
                             period_hours=args.period_hours,
                             postures=postures, tape=tape)
            want = rows_by_arm[arm]
            bad = {k: (want[k], got[k]) for k in GATE_KEYS
                   if abs(float(want[k]) - float(got[k])) > TOL}
            if bad:
                gate_failures.append({"shard": path.name, "arm": arm,
                                      "seed": seed, "mismatch": bad})
                continue
            by_family.setdefault(family, {}).setdefault(f"v2_{arm}", []).append(got)
        tapes_replayed += 1
        print(f"  {path.name} ({family}, seed {seed}) replayed "
              f"({time.perf_counter() - started:.0f}s)", flush=True)

    if gate_failures:
        print(f"\nREPLAY GATE FAILED on {len(gate_failures)} arm(s). "
              f"Not folding. First: {gate_failures[0]}", file=sys.stderr)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".gate_failure.json").write_text(
            json.dumps({"gate_failures": gate_failures, "tol": TOL},
                       indent=1, sort_keys=True) + "\n")
        return 1

    keys = ("zeta", "epsilon", "phi", "tau", "kappa", "ret_excel",
            "ret_excel_full_ledger", "ret_thesis", "ret_excel_cvar10",
            "ret_excel_cvar05", "flow_fill_rate", "fill_rate_on_time",
            "lost_orders", "backorder_qty_final", "delivered_rations",
            "strategic_injected", "distinct_postures", "posture_changes")

    out: dict[str, dict] = {}
    for family, arms in by_family.items():
        # Everything in `merged` was evaluated on the SAME v2 tapes. The base
        # panel is deliberately NOT merged in: its cells ran on different seeds.
        merged = {name: {k: sum(e[k] for e in eps) / len(eps) for k in keys}
                  for name, eps in arms.items()}
        per_tape_rows = {name: [{k: e[k] for k in keys} for e in eps]
                         for name, eps in arms.items()}
        # kappa_dot is set-relative, so adding arms REscores every incumbent cell.
        # The panel's own numbers are therefore not carried over; they are recomputed
        # here for the enlarged set, and the two sets are not comparable.
        scored = score_comparison_set(
            {n: v for n, v in merged.items()}, exponents)
        floors = panel["results"][family]["service_floors"]
        for n, v in merged.items():
            v.update(scored[n])
            v["service_pass"] = bool(
                v["flow_fill_rate"] >= floors["flow_fill_rate_min"]
                and v["lost_orders"] <= floors["lost_orders_max"]
                and v["backorder_qty_final"] <= floors["backorder_qty_final_max"])
        metrics = ("ret_excel", "ret_excel_full_ledger", "R_cobb_douglas",
                   "ret_excel_cvar10")
        rank = {m: sorted(merged, key=lambda n: -merged[n][m]) for m in metrics}
        elig = {n for n, v in merged.items() if v["service_pass"]}
        out[family] = {
            "per_cell": merged,
            "n_tapes_per_arm": {k: len(v) for k, v in arms.items()},
            "all_arms_share_the_same_tapes": len({len(v) for v in arms.values()}) == 1,
            "per_tape_rows": per_tape_rows,
            "shifts_note": "v2 pins shifts=1; S2/S3 cells cannot be paired here",
            "winner_by_metric": {m: r[0] for m, r in rank.items()},
            "winner_by_metric_among_service_pass": {
                m: next((n for n in r if n in elig), None) for m, r in rank.items()},
            "v2_arm_rank": {
                m: {n: r.index(n) + 1 for n in arms} for m, r in rank.items()},
            "n_cells": len(merged),
            "n_service_pass": len(elig),
        }
        print(f"  {family}: {len(merged)} cells after fold", flush=True)

    payload = {
        "schema_version": "metric_panel_with_v2_arms_v1",
        "claim_status": "DEVELOPMENT_SCREEN_NO_CLAIM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": ("replay of each v2 arm's recorded per-epoch posture sequence, "
                   "gated on reproducing the shard's own ret_excel, full_ledger, "
                   "fill and lost to 1e-9"),
        "replay_gate_tolerance": TOL,
        "replay_gate_keys": list(GATE_KEYS),
        "replay_gate_failures": 0,
        "tapes_replayed": tapes_replayed,
        "source_run_dirs": [str(d) for d in args.run_dirs],
        "base_panel_read_for_service_floors_only": str(args.panel),
        "paired_within_tape": True,
        "base_panel_cells_NOT_merged": (
            "the base panel ran seeds 1,620,001-4 while v2 ran 1,430,001+/1,530,001+; "
            "reference postures are re-evaluated on v2's own tapes instead"),
        "kappa_dot_rescored_for_enlarged_set": True,
        "not_comparable_to_base_panel_numbers": True,
        "contract_self_sha256": contract.get("self_sha256"),
        "exponents": exponents,
        "results": out,
        "elapsed_seconds": time.perf_counter() - started,
    }
    body = json.dumps(payload, indent=1, sort_keys=True)
    payload["self_sha256"] = sha256(body.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"\n-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
