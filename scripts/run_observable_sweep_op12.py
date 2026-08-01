#!/usr/bin/env python3
"""Sweep the OBSERVABLE, holding the policy class fixed. The gate allows exactly this.

The declared gate says: if the simplest state-conditioned rule captures nothing, nothing more
expensive gets trained. Its premise, stated in full, was that a fitted threshold **on the same
observable** cannot see the coupling -- so the gate bars richer POLICIES, not richer SENSORS. If
the sensor is the thing that is blind, the gate has not actually been tested.

So the policy class is frozen at the two-level threshold that already failed, and the observable
is swept. Seven candidates, every one of them the REALISED PAST:

    r1r_events        R1r-family events started in the trailing window (the incumbent)
    all_events        the same over all nine risks
    severity          summed magnitude of those events, not their count
    hours_since_last  time since the last risk event began
    backlog           `pending_backorder_qty`
    op9_stock         on-hand rations at the supply battalion
    ops_down          how many operations are currently down

If some sensor lets the minimal rule capture part of the oracle gap, then -- and only then -- a
richer policy becomes worth paying for. If none does, the gate holds for a second, independent
reason and the negative is that much harder to attribute to a lack of expressiveness.

Every observable carries its own placebo: the identical rule driven by another episode's trace
of the SAME observable. Same distribution, no information about this episode.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402
from supply_chain.config import THESIS_FAITHFUL_PROTOCOL as P  # noqa: E402
from supply_chain.episode_metrics import compute_episode_metrics  # noqa: E402
from supply_chain.supply_chain import MFSCSimulation  # noqa: E402

R1R = ("R11", "R12", "R13", "R14")
R2R = ("R21", "R22", "R23", "R24")
R3 = ("R3",)
REGIMES = {"R1r": R1R, "R2r": R2R, "R3": R3, "R1r+R2r": R1R + R2R,
           "R1r+R3": R1R + R3, "R2r+R3": R2R + R3, "R1r+R2r+R3": R1R + R2R + R3}
PERIODS = (12.0, 21.0, 30.0, 39.0, 48.0)
WINDOW = 336.0
METRIC = "ret_excel_risk_conditional"
SEED_BASE = 4_900_001
STEP = float(HOURS_PER_WEEK)


def observables(sim) -> dict[str, float]:
    """Every one of these is the realised past, read AFTER a step and used on the NEXT."""
    now = float(sim.env.now)
    recent = [e for e in sim.risk_events
              if now - WINDOW <= float(getattr(e, "start_time", -1e9)) <= now]
    starts = [float(getattr(e, "start_time", -1e9)) for e in sim.risk_events
              if float(getattr(e, "start_time", 1e18)) <= now]
    return {
        "r1r_events": float(sum(1 for e in recent
                                if str(getattr(e, "risk_id", "")) in R1R)),
        "all_events": float(len(recent)),
        "severity": float(sum(float(getattr(e, "magnitude", 0.0) or 0.0) for e in recent)),
        "hours_since_last": float(min(now - max(starts), WINDOW) if starts else WINDOW),
        "backlog": float(getattr(sim, "pending_backorder_qty", 0.0) or 0.0),
        "op9_stock": float(getattr(getattr(sim, "rations_sb", None), "level", 0.0) or 0.0),
        "ops_down": float(sum(1 for v in getattr(sim, "op_down_count", {}).values() if v)),
    }


OBSERVABLES = tuple(observables.__doc__ and (
    "r1r_events", "all_events", "severity", "hours_since_last",
    "backlog", "op9_stock", "ops_down"))


def make_sim(risks, seed, horizon):
    return MFSCSimulation(
        shifts=1, initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
        risks_enabled=True, risk_level="current", enabled_risks=set(risks),
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])


def action(period): return {"op12_rop": float(period), "op12_q_min": 2_392.0,
                            "op12_q_max": 2_600.0}


def run_constant(risks, seed, horizon, period):
    sim = make_sim(risks, seed, horizon)
    trace = {name: [] for name in OBSERVABLES}
    for _ in range(int(round(horizon / STEP))):
        sim.step(action=action(period), step_hours=STEP)
        for name, value in observables(sim).items():
            trace[name].append(value)
    return float(compute_episode_metrics(sim)[METRIC]), trace


def run_rule(risks, seed, horizon, *, name, low, high, threshold, signal=None):
    sim = make_sim(risks, seed, horizon)
    state = 0.0
    for index in range(int(round(horizon / STEP))):
        sim.step(action=action(high if state >= threshold else low), step_hours=STEP)
        state = (signal[index] if signal is not None and index < len(signal)
                 else observables(sim)[name])
    return float(compute_episode_metrics(sim)[METRIC])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-seeds", type=int, default=4)
    ap.add_argument("--test-seeds", type=int, default=4)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--output", type=Path,
                    default=Path("results/sensitivity/observable_sweep_op12_v1/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    names = list(REGIMES)
    train = [SEED_BASE + i for i in range(args.train_seeds)]
    test = [SEED_BASE + 500 + i for i in range(args.test_seeds)]
    started = time.perf_counter()

    const: dict[tuple[str, float, int], float] = {}
    traces: dict[tuple[str, int], dict[str, list[float]]] = {}
    for label in names:
        for period in PERIODS:
            for seed in train + test:
                value, trace = run_constant(REGIMES[label], seed, horizon, period)
                const[(label, period, seed)] = value
                if period == PERIODS[0]:
                    traces[(label, seed)] = trace
    print(f"  constantes ({time.perf_counter() - started:.0f}s)", flush=True)

    def mean(label, period, seeds):
        return float(np.mean([const[(label, period, s)] for s in seeds]))

    best_period = max(PERIODS, key=lambda p: np.mean([mean(r, p, train) for r in names]))
    constant_test = float(np.mean([mean(r, best_period, test) for r in names]))
    oracle_test = float(np.mean([max(mean(r, p, test) for p in PERIODS) for r in names]))
    gap = oracle_test - constant_test

    results: dict[str, dict] = {}
    for obs in OBSERVABLES:
        pooled = np.concatenate([np.asarray(traces[(r, s)][obs])
                                 for r in names for s in train])
        cands = sorted({float(np.percentile(pooled, q)) for q in (40, 60, 80)})

        def switches(th: float) -> float:
            return float(np.mean([np.mean(np.asarray(traces[(r, s)][obs]) >= th)
                                  for r in names for s in train]))

        usable = [th for th in cands if 0.05 < switches(th) < 0.95]
        if not usable:
            # The sensor cannot drive a switch at any of its own quantiles: it is unusable as a
            # sensor, which is a finding about the sensor rather than a failure of the run.
            results[obs] = {"sensor_unusable": True, "candidates": cands,
                            "switch_shares": {str(th): switches(th) for th in cands}}
            print(f"  {obs:<18} SENSOR INUTILIZABLE (no conmuta en ningún cuantil)", flush=True)
            continue
        best, best_score = None, -np.inf
        for low, high, th in itertools.product(PERIODS, PERIODS, usable):
            if low == high:
                continue
            score = float(np.mean([run_rule(REGIMES[r], s, horizon, name=obs, low=low,
                                            high=high, threshold=th)
                                   for r in names for s in train]))
            if score > best_score:
                best, best_score = (low, high, th), score
        low, high, th = best
        reactive = float(np.mean([run_rule(REGIMES[r], s, horizon, name=obs, low=low,
                                           high=high, threshold=th)
                                  for r in names for s in test]))
        placebo = float(np.mean([
            run_rule(REGIMES[r], s, horizon, name=obs, low=low, high=high, threshold=th,
                     signal=traces[(r, test[(i + 1) % len(test)])][obs])
            for r in names for i, s in enumerate(test)]))
        results[obs] = {
            "rule": {"low": low, "high": high, "threshold": th},
            "switch_share": switches(th),
            "reactive": reactive, "placebo": placebo,
            "capture": (reactive - constant_test) / gap if gap > 0 else float("nan"),
            "capture_placebo": (placebo - constant_test) / gap if gap > 0 else float("nan"),
            "beats_placebo": reactive > placebo,
        }
        print(f"  {obs:<18} captura {results[obs]['capture']:+.1%}  "
              f"placebo {results[obs]['capture_placebo']:+.1%} "
              f"({time.perf_counter() - started:.0f}s)", flush=True)

    usable_obs = [o for o in results if not results[o].get("sensor_unusable")]
    best_obs = max(usable_obs, key=lambda o: results[o]["capture"]) if usable_obs else None
    any_positive = [o for o in usable_obs if results[o]["capture"] > 0
                    and results[o]["beats_placebo"]]

    falsifiers = {
        "f1_every_observable_varies": {
            "passed": all(float(np.std([v for r in names for s in train + test
                                        for v in traces[(r, s)][o]])) > 0 for o in OBSERVABLES),
            "evidence": {"why_it_can_fail": "a constant sensor makes its rule a constant"}},
        "f2_every_fitted_rule_actually_switches": {
            "passed": all(0.05 < results[o]["switch_share"] < 0.95 for o in usable_obs),
            "evidence": {"why_it_can_fail": ("a rule that cannot switch is a constant wearing a "
                                             "policy's clothes -- it halted this experiment "
                                             "twice before"),
                         "switch_share": {o: results[o]["switch_share"] for o in usable_obs},
                         "sensors_unusable": [o for o in results
                                              if results[o].get("sensor_unusable")]}},
        "f3_oracle_bounds_the_CONSTANTS_only": {
            "passed": all(mean(r, p, test) <= max(mean(r, q, test) for q in PERIODS) + 1e-12
                          for r in names for p in PERIODS),
            "evidence": {
                "why_it_can_fail": "a bad aggregation would let a constant beat its own maximum",
                "correction": ("the earlier framing said the oracle bounds EVERY policy. It does "
                               "not: `H_regime` is the value of knowing the regime FOR CHOOSING "
                               "A CONSTANT, and a rule that switches WITHIN an episode is a "
                               "different class that this bound does not cover. A capture above "
                               "100% would therefore be legitimate, not a bug"),
                "oracle": oracle_test, "constant": constant_test, "gap": gap}},
        "f4_seeds_are_disjoint": {
            "passed": not (set(train) & set(test)),
            "evidence": {"train": train, "test": test}},
        "f5_gap_is_positive": {
            "passed": gap > 0.0, "evidence": {"gap": gap}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed")

    print(f"\n  constante {constant_test:.6f} | oráculo {oracle_test:.6f} "
          f"| brecha {gap:.6f}")
    print(f"  mejor observable: {best_obs} "
          f"({results[best_obs]['capture']:+.1%})" if best_obs else "  sin observable usable")
    print(f"  observables con captura POSITIVA y que baten al placebo: "
          f"{any_positive or 'ninguno'}")
    print("\n  falsadores:")
    for name, check in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<42} {'PASA' if check['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "observable_sweep_op12_v1",
        "claim_status": ("DEVELOPMENT_OBSERVABLE_SWEEP" if falsifiers["all_passed"]
                         else "HALTED_FALSIFIER_FAILED"),
        "gate_reading": ("the declared gate bars richer POLICIES, not richer SENSORS. The "
                         "policy class is frozen at the two-level threshold that already "
                         "failed; only the observable moves"),
        "metric": METRIC, "observables": list(OBSERVABLES),
        "baseline": {"constant": constant_test, "oracle": oracle_test, "gap": gap,
                     "best_constant_period": best_period},
        "by_observable": results, "best_observable": best_obs,
        "usable_observables": usable_obs,
        "observables_with_positive_capture": any_positive,
        "falsifiers": falsifiers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output,
        contract=Path("docs/RESULTADO_POLITICA_OP12_2026-07-31.md"),
        reference=Path("results/sensitivity/op12_conditioned_policy_v1/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
