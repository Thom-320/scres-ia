#!/usr/bin/env python3
"""Confirm (or kill) the one positive sensor on virgin seeds, with a paired CI.

The observable sweep found exactly one sensor -- `backlog` -- whose minimal two-level rule
captured part of the oracle gap AND beat its own placebo. One positive out of six, picked after
looking, with no confidence interval, is not a result. This runner tests that single declared
hypothesis on seeds neither the fit nor the sweep's test set ever touched.

The rule is READ from the sealed artifact rather than retyped, so `f2` can prove the policy under
test is the frozen one. The difference is taken PAIRED per (regime, seed) and bootstrapped
GROUPED BY SEED, because the seven regimes inside one seed share an exogenous stream and are not
independent observations.

See `docs/PREREGISTRO_CONFIRMACION_BACKLOG_2026-07-31.md` for the reading rule, fixed in advance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.config import HOURS_PER_WEEK  # noqa: E402

from run_observable_sweep_op12 import (  # noqa: E402
    METRIC, REGIMES, run_constant, run_rule)

SWEEP = Path("results/sensitivity/observable_sweep_op12_v1/result.json")
OBS = "backlog"
SEED_BASE = 5_100_001
FIT_SEEDS = tuple(range(4_900_001, 4_900_007)) + tuple(range(4_900_501, 4_900_507))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--horizon-weeks", type=int, default=52)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--output", type=Path,
                    default=Path("results/sensitivity/backlog_confirmation_v1/result.json"))
    args = ap.parse_args()
    horizon = float(args.horizon_weeks * HOURS_PER_WEEK)
    started = time.perf_counter()

    sweep = json.loads(SWEEP.read_text())
    rule = sweep["by_observable"][OBS]["rule"]
    constant_period = float(sweep["baseline"]["best_constant_period"])
    low, high, threshold = float(rule["low"]), float(rule["high"]), float(rule["threshold"])

    seeds = [SEED_BASE + i for i in range(args.seeds)]
    names = list(REGIMES)

    # One constant episode per (regime, seed) gives BOTH the paired baseline and the placebo
    # trace donor: seed s takes its placebo signal from seed s+1's trace of the same observable.
    const, traces = {}, {}
    for label in names:
        for seed in seeds:
            value, trace = run_constant(REGIMES[label], seed, horizon, constant_period)
            const[(label, seed)] = value
            traces[(label, seed)] = trace[OBS]
        print(f"  constante {label} ({time.perf_counter() - started:.0f}s)", flush=True)

    react, plac, switches = {}, {}, []
    for label in names:
        for i, seed in enumerate(seeds):
            react[(label, seed)] = run_rule(REGIMES[label], seed, horizon, name=OBS,
                                            low=low, high=high, threshold=threshold)
            donor = traces[(label, seeds[(i + 1) % len(seeds)])]
            plac[(label, seed)] = run_rule(REGIMES[label], seed, horizon, name=OBS, low=low,
                                           high=high, threshold=threshold, signal=donor)
            switches.append(float(np.mean([v >= threshold for v in traces[(label, seed)]])))
        print(f"  reactiva  {label} ({time.perf_counter() - started:.0f}s)", flush=True)

    def paired(arm: dict) -> np.ndarray:
        """Per-seed mean of the (arm - constant) difference, averaged over regimes."""
        return np.array([np.mean([arm[(r, s)] - const[(r, s)] for r in names]) for s in seeds])

    d_react, d_plac = paired(react), paired(plac)
    rng = np.random.default_rng(20260731)

    def lcb(d: np.ndarray) -> tuple[float, float]:
        draws = d[rng.integers(0, d.size, size=(args.n_boot, d.size))].mean(axis=1)
        return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

    lo_r, hi_r = lcb(d_react)
    lo_p, hi_p = lcb(d_plac)
    beats_placebo = float(d_react.mean()) > float(d_plac.mean())
    confirmed = lo_r > 0.0 and beats_placebo

    falsifiers = {
        "f1_seeds_are_virgin": {
            "passed": not (set(seeds) & set(FIT_SEEDS)),
            "evidence": {"why_it_can_fail": ("reusing a fit or sweep seed would make this a "
                                             "re-read of the same data, not a confirmation"),
                         "seeds": seeds, "excluded": list(FIT_SEEDS)}},
        "f2_rule_is_the_frozen_one": {
            "passed": (rule == sweep["by_observable"][OBS]["rule"]
                       and sweep["best_observable"] == OBS
                       and sweep["falsifiers"]["all_passed"]),
            "evidence": {"why_it_can_fail": ("a retyped or drifted parameter would make this a "
                                             "different policy wearing the same name"),
                         "rule": rule, "sweep_sha256": sweep["self_sha256"],
                         "sweep_best_observable": sweep["best_observable"]}},
        "f3_rule_switches_on_new_seeds": {
            "passed": 0.02 < float(np.mean(switches)) < 0.98,
            "evidence": {"why_it_can_fail": ("a rule that never switches is the constant, and "
                                             "the comparison would be vacuous"),
                         "switch_share": float(np.mean(switches))}},
        "f4_paired_difference_has_variance": {
            "passed": float(np.std(d_react)) > 0.0,
            "evidence": {"why_it_can_fail": "zero variance makes the interval degenerate",
                         "sd": float(np.std(d_react)), "n_seeds": len(seeds)}},
        "f5_placebo_is_not_the_signal": {
            "passed": any(traces[(r, seeds[0])] != traces[(r, seeds[1])] for r in names),
            "evidence": {"why_it_can_fail": ("if the donor trace equalled the real one the "
                                             "control would control for nothing")}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed")

    print(f"\n  === confirmación de `{OBS}` sobre {len(seeds)} semillas vírgenes ===")
    print(f"  regla congelada: low={low} high={high} threshold={threshold}")
    print(f"  reactiva - constante   media {d_react.mean():+.3e}  "
          f"IC95 [{lo_r:+.3e}, {hi_r:+.3e}]")
    print(f"  placebo  - constante   media {d_plac.mean():+.3e}  "
          f"IC95 [{lo_p:+.3e}, {hi_p:+.3e}]")
    print(f"  semillas con diferencia > 0: {int((d_react > 0).sum())}/{len(seeds)}")
    print(f"\n  veredicto: {'CONFIRMADO' if confirmed else 'NO CONFIRMADO'} "
          f"(LCB95 {'>' if lo_r > 0 else '<='} 0, "
          f"{'bate' if beats_placebo else 'NO bate'} al placebo)")
    print("\n  falsadores:")
    for name, check in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<42} {'PASA' if check['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "backlog_confirmation_v1",
        "claim_status": (("CONFIRMED_BACKLOG_SENSOR" if confirmed
                          else "REFUTED_BACKLOG_SENSOR_WAS_SELECTION_NOISE")
                         if falsifiers["all_passed"] else "HALTED_FALSIFIER_FAILED"),
        "observable": OBS, "metric": METRIC, "rule": rule,
        "constant_period": constant_period, "seeds": seeds, "regimes": names,
        "reactive_minus_constant": {"mean": float(d_react.mean()), "lcb95": lo_r, "ucb95": hi_r,
                                    "per_seed": d_react.tolist()},
        "placebo_minus_constant": {"mean": float(d_plac.mean()), "lcb95": lo_p, "ucb95": hi_p,
                                   "per_seed": d_plac.tolist()},
        "beats_placebo": beats_placebo, "confirmed": confirmed,
        "oracle_gap_for_scale": sweep["baseline"]["gap"],
        "magnitude_note": ("even a full capture of the oracle gap is ~290x below the 0.01 bar; "
                           "a PASS authorises spending on a richer policy class, it does not "
                           "claim material headroom"),
        "falsifiers": falsifiers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output,
        contract=Path("docs/PREREGISTRO_CONFIRMACION_BACKLOG_2026-07-31.md"),
        reference=SWEEP)
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
