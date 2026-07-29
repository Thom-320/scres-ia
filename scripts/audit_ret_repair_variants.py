#!/usr/bin/env python3
"""What each of the three candidate ReT repairs actually produces.

Defect 3 (`docs/RET_METRIC_DEFECTS_2026-07-29.md`): `RPj` is a *time* attribution,
but a delay can be caused by a *quantity* risk that contributes zero hours. When it
is, `RPj -> 0` and `ReT = 0.5/RPj` diverges on exactly the worst-served orders. On
R2r tape 1530011 one order delivered 192 h late scores 73.9082, the episode maximum,
on a metric defined on [0,1].

Three repairs were named. None is applied to the canonical metric here -- each
changes historical numbers and needs preregistration. This measures all three so the
preregistration can be written against evidence instead of intuition.

    canonical      untouched, and used as the validation gate
    clip_0_1       clamp each per-order ReT into the range the metric declares
    rpj_floor_*    floor RPj at a physical quantum before 0.5/RPj is taken
    quantity_time  when a quantity risk (R14/R24) touched the order, credit RPj with
                   the realised lateness CTj - LTj, in hours

The floor variant is swept rather than fixed. Choosing one floor and reporting it as
the answer is how the Cobb-Douglas port first went wrong -- a floor is a decision, and
a result that moves with it is a result about the decision.

**Method.** Variants are produced by mutating the inputs and re-calling the *official*
ledger, never by reimplementing the formula. `canonical` must reproduce
`compute_episode_metrics`' `ret_excel` exactly or the run aborts.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.episode_metrics import (  # noqa: E402
    compute_episode_metrics,
    compute_order_level_ret_excel_request_snapshot_ledger as ledger,
)
from scripts.run_expanded_contract_comparators_v2 import (  # noqa: E402
    apply_posture,
    make_replay_sim,
    materialize_tape,
)

TOL = 1e-9
# Risks whose effect on an order is a missing quantity rather than elapsed downtime.
QUANTITY_RISKS = ("R14", "R24")
RPJ_FLOORS = (0.5, 1.0, 6.0, 24.0)
# Each family's true 216-posture incumbent, from v2's own terminal result.json.
INCUMBENTS = {"R1r": (0, 0, 336), "R2r": (336, 0, 168)}
RUN_DIRS = {
    "R1r": "expanded_contract_comparators_v2_1dc40c1_r1",
    "R2r": "expanded_contract_comparators_v2_1dc40c1_r2",
}


def scored_orders(sim) -> list:
    return [o for o in sim.orders
            if not bool(getattr(o, "metrics_excluded", False))
            and float(getattr(o, "OPTj", 0.0)) >= float(sim.warmup_time)]


def mean_ret(orders: list, now: float) -> float:
    values = ledger(orders, current_time=now)["ret_values"]
    return float(np.mean(values)) if values else 0.0


def variant_values(orders: list, now: float, *, mode: str,
                   rpj_floor: float = 0.0) -> np.ndarray:
    """Per-order ReT under one repair. Inputs are mutated, the formula is not."""
    if mode == "canonical":
        return np.array(ledger(orders, current_time=now)["ret_values"], dtype=float)

    if mode == "clip_0_1":
        base = np.array(ledger(orders, current_time=now)["ret_values"], dtype=float)
        return np.clip(base, 0.0, 1.0)

    patched = []
    for order in orders:
        clone = copy.copy(order)
        rpj = float(getattr(clone, "RPj", 0.0) or 0.0)
        if mode == "rpj_floor":
            if rpj > 0.0:
                clone.RPj = max(rpj, rpj_floor)
        elif mode == "quantity_time":
            indicators = dict(getattr(clone, "ret_risk_indicators", {}) or {})
            touched_by_quantity = any(
                key == risk or key.startswith(f"{risk}_")
                for key in indicators for risk in QUANTITY_RISKS)
            if touched_by_quantity:
                lateness = float(getattr(clone, "CTj", 0.0) or 0.0) - float(
                    getattr(clone, "LTj", 0.0) or 0.0)
                if lateness > 0.0:
                    clone.RPj = max(rpj, lateness)
        else:  # pragma: no cover - guarded by argparse choices
            raise ValueError(mode)
        patched.append(clone)
    return np.array(ledger(patched, current_time=now)["ret_values"], dtype=float)


def run_arm(*, seed: int, family: str, horizon: float, epoch_hours: float,
            postures: list[tuple[int, int, int]]) -> dict:
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

    orders = scored_orders(sim)
    now = float(sim.env.now)
    official = float(compute_episode_metrics(sim)["ret_excel"])

    out = {"n_scored": len(orders), "official_ret_excel": official,
           "warmup_time": float(sim.warmup_time),
           "flow_fill_rate": float(compute_episode_metrics(sim)["flow_fill_rate"])}
    for mode in ("canonical", "clip_0_1", "quantity_time"):
        v = variant_values(orders, now, mode=mode)
        out[mode] = float(v.mean())
        out[f"{mode}_n_above_one"] = int((v > 1.0).sum())
        out[f"{mode}_max"] = float(v.max()) if len(v) else 0.0
    for floor in RPJ_FLOORS:
        v = variant_values(orders, now, mode="rpj_floor", rpj_floor=floor)
        key = f"rpj_floor_{floor:g}"
        out[key] = float(v.mean())
        out[f"{key}_n_above_one"] = int((v > 1.0).sum())
        out[f"{key}_max"] = float(v.max()) if len(v) else 0.0

    # The gate: the untouched variant must be the official metric.
    if abs(out["canonical"] - official) > TOL:
        raise SystemExit(
            f"canonical variant {out['canonical']} != official {official} "
            f"(seed {seed}, {family}); the reconstruction is wrong, refusing to report")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path,
                    default=Path("/Users/thom/Projects/research/scres-ia-runs"))
    ap.add_argument("--families", nargs="+", default=["R1r", "R2r"])
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--epoch-weeks", type=int, default=4)
    ap.add_argument("--output", type=Path,
                    default=Path("results/metric_audit/ret_repair_variants_v1/"
                                 "result.json"))
    args = ap.parse_args()

    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    epoch_hours = float(args.epoch_weeks * HOURS_PER_WEEK)
    variants = (["canonical", "clip_0_1", "quantity_time"]
                + [f"rpj_floor_{f:g}" for f in RPJ_FLOORS])
    started = time.perf_counter()
    rng = np.random.default_rng(20260729)
    out: dict = {}

    for family in args.families:
        shards = sorted((args.runs_root / RUN_DIRS[family] / "full" / "shards")
                        .glob("*.json"))
        rows: dict[str, list[dict]] = {"mpc": [], "static": []}
        for path in shards:
            shard = json.loads(path.read_text())
            seed = int(shard["tape_seed"])
            mpc_postures = [tuple(int(x) for x in e["selected_posture"])
                            for e in shard["traces"]["mpc"]]
            rows["mpc"].append(run_arm(seed=seed, family=family, horizon=horizon,
                                       epoch_hours=epoch_hours,
                                       postures=mpc_postures))
            rows["static"].append(run_arm(
                seed=seed, family=family, horizon=horizon, epoch_hours=epoch_hours,
                postures=[INCUMBENTS[family]] * len(mpc_postures)))
            rows["mpc"][-1]["tape_seed"] = seed
            rows["static"][-1]["tape_seed"] = seed
            print(f"  {family} {seed} ({time.perf_counter() - started:.0f}s)",
                  flush=True)

        per_variant = {}
        for v in variants:
            a = np.array([r[v] for r in rows["mpc"]])
            b = np.array([r[v] for r in rows["static"]])
            d = a - b
            boot = np.array([rng.choice(d, len(d), True).mean() for _ in range(10_000)])
            per_variant[v] = {
                "mpc_mean": float(a.mean()), "static_mean": float(b.mean()),
                "delta_mean": float(d.mean()),
                "ci95": [float(np.quantile(boot, 0.025)),
                         float(np.quantile(boot, 0.975))],
                "n_tapes_mpc_ahead": int((d > 0).sum()), "n_tapes": len(d),
                "verdict": ("MPC_AHEAD" if np.quantile(boot, 0.025) > 0
                            else "STATIC_AHEAD" if np.quantile(boot, 0.975) < 0
                            else "NOT_SEPARATED"),
                "n_orders_above_one_mpc": int(sum(r[f"{v}_n_above_one"]
                                                  for r in rows["mpc"])),
                "n_orders_above_one_static": int(sum(r[f"{v}_n_above_one"]
                                                     for r in rows["static"])),
                "max_ret_static": float(max(r[f"{v}_max"] for r in rows["static"])),
                "worst_tape_delta": float(d.min()),
                "worst_tape_seed": int(rows["mpc"][int(d.argmin())]["tape_seed"]),
            }
        out[family] = {"incumbent": list(INCUMBENTS[family]),
                       "per_variant": per_variant,
                       "per_tape_rows": rows}
        print(f"  {family} done", flush=True)

    payload = {
        "schema_version": "ret_repair_variants_v1",
        "claim_status": "DEVELOPMENT_PREREGISTRATION_INPUT_NO_METRIC_CHANGED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": ("measure what each named repair of ReT defect 3 produces, so a "
                    "preregistration can be written against evidence"),
        "canonical_metric_unchanged": True,
        "validation_gate": ("the canonical variant reproduces compute_episode_metrics' "
                            f"ret_excel to {TOL}; the run aborts otherwise"),
        "quantity_risks": list(QUANTITY_RISKS),
        "rpj_floors_swept_hours": list(RPJ_FLOORS),
        "variants": variants,
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
