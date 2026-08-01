#!/usr/bin/env python3
"""Put the decision variable where `S_ij` said, and see whether a policy captures anything.

`S_ij` named one pair: `op12_rop` x `impact_R1r` at +0.219 -- the dispatch PERIOD of the final
downstream leg, coupled to the impact of the R1r family. So: make that period conditionable on
the realised risk state, and measure how much of the oracle headroom a policy actually captures
against the best constant.

Four policies on identical CRN tapes:

* `constant`   -- the best single `op12_rop` over the whole regime mix. The incumbent, and the
                  baseline the headroom is defined against.
* `oracle`     -- the best `op12_rop` per regime, chosen knowing the regime. The CEILING; no
                  policy can beat it.
* `reactive`   -- a threshold rule on an OBSERVABLE: recent R1r-family event count in the last
                  window. Fitted on training seeds, scored on held-out seeds. This is the
                  cheapest possible controller, an MPC-style certainty-equivalent switch.
* `placebo`    -- the identical rule driven by a signal with the same distribution and no
                  information: the observable of a DIFFERENT seed's trajectory. If the placebo
                  captures as much as the reactive rule, the gain is the extra flexibility and
                  not the signal, and the whole thing is nothing.

The gate this establishes is deliberate: **if the simplest state-conditioned rule captures
nothing here, nothing more expensive should be trained.** Reinforcement learning cannot extract
a coupling that a fitted threshold on the same observable cannot see.

The observable is the REALISED PAST -- events that have already occurred -- never the future and
never the hyperparameters. That distinction is the project's standing threat (`privileged
observation`), so it is stated here and enforced by construction: the state is read after each
step and applied to the NEXT one.
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
PERIODS = (12.0, 21.0, 30.0, 39.0, 48.0)      # the op12_rop range from the sensitivity map
# Thresholds are DERIVED from the observable's own distribution, not guessed. The smoke run
# fixed them at (1,2,3,4), the observable almost never fell below 1, and the fitted rule
# degenerated into a constant -- reactive and placebo then scored identically to six decimals,
# which is what `f2` caught. A switch that cannot switch is not a policy.
THRESHOLD_QUANTILES = (25, 50, 75, 90)
WINDOW_HOURS = 336.0                          # two weeks of realised history
METRIC = "ret_excel_risk_conditional"
SEED_BASE = 4_600_001
STEP = float(HOURS_PER_WEEK)


def make_sim(risks: tuple[str, ...], seed: int, horizon: float) -> MFSCSimulation:
    return MFSCSimulation(
        shifts=1, initial_buffers={n: 0.0 for n in ("op3_rm", "op5_rm", "op9_rations")},
        inventory_replenishment_period=0.0, seed=seed, horizon=horizon,
        risks_enabled=True, risk_level="current", enabled_risks=set(risks),
        strict_exogenous_crn=True, year_basis=P["year_basis"],
        warmup_trigger=P["warmup_trigger"], r14_defect_mode=P["r14_defect_mode"])


def action(period: float) -> dict[str, float]:
    return {"op12_rop": float(period), "op12_q_min": 2_392.0, "op12_q_max": 2_600.0}


def recent_r1r(sim: MFSCSimulation) -> int:
    """Realised R1r events inside the trailing window. The past, never the future."""
    now = float(sim.env.now)
    return sum(1 for e in sim.risk_events
               if str(getattr(e, "risk_id", "")) in R1R
               and now - WINDOW_HOURS <= float(getattr(e, "start_time", -1e9)) <= now)


def run_constant(risks, seed, horizon, period) -> tuple[float, list[int]]:
    sim = make_sim(risks, seed, horizon)
    trace: list[int] = []
    for _ in range(int(round(horizon / STEP))):
        sim.step(action=action(period), step_hours=STEP)
        trace.append(recent_r1r(sim))
    return float(compute_engine(sim)), trace


def compute_engine(sim) -> float:
    return float(compute_episode_metrics(sim)[METRIC])


def run_reactive(risks, seed, horizon, *, low: float, high: float, threshold: int,
                 signal: list[int] | None = None) -> float:
    """Switch `op12_rop` on the observable. `signal` overrides it with a placebo trace."""
    sim = make_sim(risks, seed, horizon)
    steps = int(round(horizon / STEP))
    state = 0
    for index in range(steps):
        period = high if state >= threshold else low
        sim.step(action=action(period), step_hours=STEP)
        observed = recent_r1r(sim)
        state = (signal[index] if signal is not None and index < len(signal) else observed)
    return compute_engine(sim)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-seeds", type=int, default=6)
    ap.add_argument("--test-seeds", type=int, default=6)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--output", type=Path,
                    default=Path("results/sensitivity/op12_conditioned_policy_v1/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    names = list(REGIMES)
    train = [SEED_BASE + i for i in range(args.train_seeds)]
    test = [SEED_BASE + 1_000 + i for i in range(args.test_seeds)]
    started = time.perf_counter()

    # ---- constants, on both splits, plus the observable traces for the placebo -----------
    const: dict[str, dict[float, dict[int, float]]] = {r: {p: {} for p in PERIODS}
                                                       for r in names}
    traces: dict[tuple[str, int], list[int]] = {}
    for label in names:
        for period in PERIODS:
            for seed in train + test:
                score, trace = run_constant(REGIMES[label], seed, horizon, period)
                const[label][period][seed] = score
                if period == PERIODS[0]:
                    traces[(label, seed)] = trace
    print(f"  constantes ({time.perf_counter() - started:.0f}s)", flush=True)

    def mean_over(label: str, period: float, seeds: list[int]) -> float:
        return float(np.mean([const[label][period][s] for s in seeds]))

    best_constant_period = max(
        PERIODS, key=lambda p: float(np.mean([mean_over(r, p, train) for r in names])))
    constant_test = float(np.mean([mean_over(r, best_constant_period, test) for r in names]))
    oracle_test = float(np.mean([max(mean_over(r, p, test) for p in PERIODS) for r in names]))
    oracle_gap = oracle_test - constant_test

    # ---- fit the reactive rule on TRAIN only ---------------------------------------------
    # Thresholds come from the regimes that CONTAIN R1r. Pooling across all seven put the
    # 25th percentile at 0 -- R2r-only and R3-only trajectories have no R1r events at all -- and
    # the fit then chose threshold 0, a rule that is always "on" and therefore a constant. `f6`
    # caught it; this is the fix it demanded, not a relaxation of it.
    with_r1r = [np.asarray(v) for (lbl, sd), v in traces.items()
                if sd in train and "R1r" in lbl]
    pooled = np.concatenate(with_r1r)
    candidates = sorted({int(np.percentile(pooled, q)) for q in THRESHOLD_QUANTILES})
    # And a candidate only enters the grid if it actually switches in sample. A rule that
    # cannot switch is a constant wearing a policy's clothes, and comparing it to the constant
    # is comparing something to itself.
    def switches(th: int) -> float:
        return float(np.mean([np.mean(np.asarray(traces[(r, s)]) >= th)
                              for r in names for s in train]))

    thresholds = [th for th in candidates if 0.05 < switches(th) < 0.95]
    if not thresholds:
        thresholds = candidates
    grid = [(low, high, th) for low, high, th in
            itertools.product(PERIODS, PERIODS, thresholds) if low != high]
    fitted, best_train = None, -np.inf
    for low, high, th in grid:
        score = float(np.mean([run_reactive(REGIMES[r], s, horizon, low=low, high=high,
                                            threshold=th)
                               for r in names for s in train]))
        if score > best_train:
            fitted, best_train = (low, high, th), score
    low, high, th = fitted
    print(f"  regla ajustada: low={low} high={high} umbral={th} "
          f"({time.perf_counter() - started:.0f}s)", flush=True)

    reactive_test = float(np.mean([run_reactive(REGIMES[r], s, horizon, low=low, high=high,
                                                threshold=th)
                                   for r in names for s in test]))
    # placebo: same rule, driven by ANOTHER seed's realised trace -- same distribution, no
    # information about this episode.
    placebo_test = float(np.mean([
        run_reactive(REGIMES[r], s, horizon, low=low, high=high, threshold=th,
                     signal=traces[(r, test[(i + 1) % len(test)])])
        for r in names for i, s in enumerate(test)]))

    def capture(value: float) -> float:
        return (value - constant_test) / oracle_gap if oracle_gap > 0 else float("nan")

    observable_span = [int(min(min(t) for t in traces.values())),
                       int(max(max(t) for t in traces.values()))]
    # Does the fitted rule actually switch on the test traces, or is it a constant in disguise?
    switch_share = float(np.mean([np.mean(np.asarray(traces[(r, s)]) >= th)
                                  for r in names for s in test]))
    falsifiers = {
        "f1_observable_is_non_degenerate": {
            "passed": observable_span[1] > observable_span[0],
            "evidence": {"why_it_can_fail": ("a constant observable makes the rule a constant "
                                             "and the comparison empty"),
                         "span": observable_span, "window_hours": WINDOW_HOURS}},
        "f6_the_fitted_rule_actually_switches": {
            "passed": 0.05 < switch_share < 0.95,
            "evidence": {"why_it_can_fail": ("a threshold the observable almost never crosses "
                                             "makes the rule a constant wearing a policy's "
                                             "clothes -- exactly what the smoke run produced "
                                             "with hand-picked thresholds"),
                         "share_of_steps_above_threshold": switch_share,
                         "threshold": th, "candidates": thresholds,
                         "observable_span": observable_span}},
        "f2_placebo_captures_less_than_the_signal": {
            "passed": capture(placebo_test) < capture(reactive_test),
            "evidence": {"why_it_can_fail": ("THE check that matters. If a rule driven by "
                                             "another episode's trace captures as much, the "
                                             "gain is the extra flexibility, not the signal"),
                         "reactive_capture": capture(reactive_test),
                         "placebo_capture": capture(placebo_test)}},
        "f3_oracle_bounds_every_policy": {
            "passed": reactive_test <= oracle_test + 1e-12
            and placebo_test <= oracle_test + 1e-12,
            "evidence": {"why_it_can_fail": ("nothing can beat the regime-informed choice; "
                                             "exceeding it would mean the oracle is mis-built"),
                         "oracle": oracle_test, "reactive": reactive_test,
                         "placebo": placebo_test, "constant": constant_test}},
        "f4_train_and_test_seeds_are_disjoint": {
            "passed": not (set(train) & set(test)),
            "evidence": {"why_it_can_fail": "fitting and scoring on one seed is not a result",
                         "train": train, "test": test}},
        "f5_oracle_gap_is_positive": {
            "passed": oracle_gap > 0.0,
            "evidence": {"why_it_can_fail": ("with no gap there is no headroom to capture and "
                                             "the capture fraction is undefined"),
                         "oracle_gap": oracle_gap}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed")

    print(f"\n  === captura del headroom, fuera de muestra ({METRIC}) ===")
    print(f"    constante (mejor, rop={best_constant_period:.0f})  {constant_test:.6f}")
    print(f"    oráculo por régimen                    {oracle_test:.6f}   "
          f"(brecha {oracle_gap:.6f})")
    print(f"    reactiva sobre el observable           {reactive_test:.6f}   "
          f"captura {capture(reactive_test):+.1%}")
    print(f"    placebo (traza de otra semilla)        {placebo_test:.6f}   "
          f"captura {capture(placebo_test):+.1%}")
    print("\n  falsadores:")
    for name, check in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<46} {'PASA' if check['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "op12_conditioned_policy_v1",
        "claim_status": ("DEVELOPMENT_OP12_CONDITIONED_POLICY" if falsifiers["all_passed"]
                         else "HALTED_FALSIFIER_FAILED"),
        "target_pair": {"pair": "op12_rop x impact_R1r", "S_ij": 0.219,
                        "source": "results/sensitivity/second_order_risk_search_v1/result.json"},
        "metric": METRIC, "observable": f"realised R1r events in the last {WINDOW_HOURS} h",
        "fitted_rule": {"low": low, "high": high, "threshold": th,
                        "best_constant_period": best_constant_period,
                        "threshold_candidates": thresholds,
                        "switch_share_on_test": switch_share},
        "out_of_sample": {"constant": constant_test, "oracle": oracle_test,
                          "reactive": reactive_test, "placebo": placebo_test,
                          "oracle_gap": oracle_gap,
                          "capture_reactive": capture(reactive_test),
                          "capture_placebo": capture(placebo_test)},
        "regimes": {k: list(v) for k, v in REGIMES.items()},
        "train_seeds": train, "test_seeds": test,
        "falsifiers": falsifiers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output,
        contract=Path("docs/RESULTADO_SIJ_Y_NODO_2026-07-31.md"),
        reference=Path("results/sensitivity/second_order_risk_search_v1/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
