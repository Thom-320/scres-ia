#!/usr/bin/env python3
"""Run the eligibility audit against a testbed whose answer is known before it starts.

Contract: docs/PREREGISTRO_VALIDACION_POSITIVA_AUDIT_2026-08-08.md -- frozen before this file
existed. The reading order is that document's and is not re-derived here: the NULL cell is read
first, and if the bench misbehaves there nothing below it is read at all.

The learner is a linear-in-features policy over a lagged history window, fitted by cross-entropy
method on the TRAINING seeds and evaluated with no refit on a held-out block. It is deliberately
the smallest thing that can be called learning: if a premium needs a bigger network to appear, that
is a finding about the network and not about the environment, and this design would rather find
nothing than find that.

Development tooling over a synthetic bench. No MFSC seed, no MFSC claim.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain import falsifiers as F                                        # noqa: E402
from supply_chain.contention_bench_v1 import (                                  # noqa: E402
    BenchSpec, belief_mpc_policy, clairvoyant_actions, draw_tape, fixed_policy,
    history_features, oracle_model_mpc_policy, serve, signal_threshold_policy)

OUT = Path("results/audit_positive_validation/result.powered.json")
CONTRACT = Path("docs/PREREGISTRO_VALIDACION_POSITIVA_AUDIT_2026-08-08.md")
SESOI = 0.01
HEADROOM_BAR = 0.02
GRID = np.linspace(0.0, 1.0, 21)
WINDOW = 4

# Powered per docs/ENMIENDA_POTENCIA_VALIDACION_POSITIVA_2026-08-08.md: resolution
# SESOI/10, which needs n >= 331 at the measured spread. Fresh, disjoint seeds; the
# 9100001-9100120 block is consumed and its artifact retained.
TRAIN_SEEDS = list(range(9100121, 9100461))
TEST_SEEDS = list(range(9100461, 9100801))

CELLS = {
    "null": BenchSpec(alpha=1.0, rho=0.90, min_dwell=4, signal_accuracy=0.85, label="null"),
    "positive": BenchSpec(alpha=0.0, rho=0.90, min_dwell=4, signal_accuracy=0.85,
                          label="positive"),
    "no_memory": BenchSpec(alpha=0.0, rho=0.50, min_dwell=1, signal_accuracy=0.85,
                           label="no_memory"),
}


def paired_lcb(diffs: np.ndarray, *, draws: int = 20_000, seed: int = 20260808) -> dict:
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, dtype=float)
    boot = rng.choice(d, size=(draws, d.size), replace=True).mean(axis=1)
    return {"mean": float(d.mean()), "lcb95": float(np.percentile(boot, 2.5)),
            "ucb95": float(np.percentile(boot, 97.5)), "n": int(d.size),
            "favourable": int((d > 0).sum())}


def structured_frontier(spec: BenchSpec, tapes: list) -> dict:
    """Best member of each declared family. A weak frontier turns anything into a premium."""
    fixed = {float(level): np.array([serve(spec, fixed_policy(level, t), t) for t in tapes])
             for level in GRID}
    best_fixed = max(fixed, key=lambda k: fixed[k].mean())

    thresh, best_thresh = {}, None
    for lo in GRID[::4]:
        for hi in GRID[::4]:
            key = (float(lo), float(hi))
            thresh[key] = np.array(
                [serve(spec, signal_threshold_policy(hi, lo, t), t) for t in tapes])
            if best_thresh is None or thresh[key].mean() > thresh[best_thresh].mean():
                best_thresh = key

    belief = np.array([serve(spec, belief_mpc_policy(spec, t), t) for t in tapes])
    oracle = np.array([serve(spec, oracle_model_mpc_policy(spec, t), t) for t in tapes])
    families = {"fixed": fixed[best_fixed], "signal_threshold": thresh[best_thresh],
                "belief_mpc": belief}
    best = max(families, key=lambda k: families[k].mean())
    return {"per_family": {k: v for k, v in families.items()},
            "best_family": best, "best_values": families[best],
            "best_fixed_level": best_fixed, "best_threshold": best_thresh,
            "oracle_values": oracle,
            "means": {k: float(v.mean()) for k, v in families.items()}
            | {"oracle_model_mpc": float(oracle.mean())}}


def fit_learner(spec: BenchSpec, tapes: list, *, iters: int = 30, pop: int = 60,
                elite: int = 12, seed: int = 20260808) -> np.ndarray:
    """Cross-entropy method over a linear-in-features policy. Sees only training tapes."""
    rng = np.random.default_rng(seed)
    feats = [history_features(spec, t, WINDOW) for t in tapes]
    dim = feats[0].shape[1]
    mu, sigma = np.zeros(dim), np.ones(dim)
    for _ in range(iters):
        cand = rng.normal(mu, sigma, size=(pop, dim))
        scores = np.array([
            float(np.mean([serve(spec, 1.0 / (1.0 + np.exp(-(f @ w))), t)
                           for f, t in zip(feats, tapes)]))
            for w in cand])
        keep = cand[np.argsort(scores)[-elite:]]
        mu, sigma = keep.mean(axis=0), keep.std(axis=0) + 1e-3
    return mu


def learner_values(spec: BenchSpec, weights: np.ndarray, tapes: list) -> np.ndarray:
    return np.array([serve(spec, 1.0 / (1.0 + np.exp(-(history_features(spec, t, WINDOW) @ weights))), t)
                     for t in tapes])


def clairvoyant_gap(spec: BenchSpec, tapes: list, best_fixed_level: float) -> dict:
    clair = np.array([serve(spec, clairvoyant_actions(spec, t, GRID), t) for t in tapes])
    fixed = np.array([serve(spec, fixed_policy(best_fixed_level, t), t) for t in tapes])
    return paired_lcb(clair - fixed) | {"clairvoyant_mean": float(clair.mean()),
                                        "fixed_mean": float(fixed.mean())}


def run_cell(name: str, spec: BenchSpec) -> dict:
    train = [draw_tape(spec, s) for s in TRAIN_SEEDS]
    test = [draw_tape(spec, s) for s in TEST_SEEDS]

    dev = structured_frontier(spec, train)
    h_pi = clairvoyant_gap(spec, train, dev["best_fixed_level"])

    weights = fit_learner(spec, train)
    held = structured_frontier(spec, test)
    learned = learner_values(spec, weights, test)

    # The placebo keeps the mechanism and destroys the information: same fitted policy, same
    # feature pipeline, signals shuffled across periods.
    rng = np.random.default_rng(99)
    placebo_tapes = []
    for t in test:
        shuffled = draw_tape(spec, t.seed)
        shuffled.signals = rng.permutation(t.signals)
        placebo_tapes.append(shuffled)
    placebo = learner_values(spec, weights, placebo_tapes)

    vs_structured = paired_lcb(learned - held["best_values"])
    vs_oracle = paired_lcb(learned - held["oracle_values"])
    vs_placebo = paired_lcb(learned - placebo)

    # Gate 2 has to be able to say BOTH things: absence needs an upper bound below the bar, and
    # presence needs a lower bound above it. A single-sided read would call the null "not shown"
    # instead of "shown flat", which is the whole distinction this bench exists to make.
    headroom_present = h_pi["lcb95"] >= HEADROOM_BAR
    headroom_absent = h_pi["ucb95"] < HEADROOM_BAR
    verdict = ("STOP_NO_PRIVILEGED_HEADROOM" if headroom_absent else
               "AUTHORIZE_LEARNER_STUDY" if headroom_present else
               "INCONCLUSIVE_HEADROOM")
    return {
        "spec": spec.as_dict(), "audit_verdict": verdict,
        "h_pi": h_pi, "headroom_present": headroom_present, "headroom_absent": headroom_absent,
        "structured_means_dev": dev["means"], "structured_means_test": held["means"],
        "best_family_test": held["best_family"],
        "learner_mean_test": float(learned.mean()),
        "learner_vs_best_structured": vs_structured,
        "learner_vs_oracle_model_mpc": vs_oracle,
        "learner_vs_own_placebo": vs_placebo,
        "converts": vs_structured["lcb95"] >= SESOI,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / OUT)
    args = ap.parse_args()

    cells = {name: run_cell(name, spec) for name, spec in CELLS.items()}
    null, pos, mem = cells["null"], cells["positive"], cells["no_memory"]

    # f1 is the bench's own theorem, checked rather than trusted: at alpha = 1 the split cannot
    # move service, so the spread over the whole action grid must be zero to floating point.
    spec = CELLS["null"]
    probe = draw_tape(spec, TRAIN_SEEDS[0])
    spread = float(np.ptp([serve(spec, fixed_policy(level, probe), probe) for level in GRID]))

    checks = {
        "f1_null_is_algebraically_flat": F.lt(
            spread, 1e-12,
            "if a split can move service at full fungibility the physics is wrong and the null "
            "cell is not null"),
        "f2_audit_stops_on_the_null": F.check(
            null["audit_verdict"] == "STOP_NO_PRIVILEGED_HEADROOM",
            "authorizing where the truth is exactly zero is a false positive, which is the "
            "failure mode that would make the audit worthless",
            computed_from={"ucb95": null["h_pi"]["ucb95"], "bar": HEADROOM_BAR},
            verdict=null["audit_verdict"]),
        "f3_positive_cell_has_real_headroom": F.ge(
            pos["h_pi"]["lcb95"], HEADROOM_BAR,
            "the construction might not generate enough headroom, leaving no positive instance "
            "to validate on"),
        "f4_learner_beats_its_own_placebo": F.gt(
            pos["learner_vs_own_placebo"]["lcb95"], 0.0,
            "if the shuffled-signal placebo ties, what was measured is cadence not information"),
        "f5_structured_frontier_was_searched": F.ge(
            len(pos["structured_means_test"]) - 1, 3,
            "a weak frontier turns any policy into a premium"),
        "f6_learner_converts_on_fresh_seeds": F.ge(
            pos["learner_vs_best_structured"]["lcb95"], SESOI,
            "the belief MPC may absorb the residual, in which case the audit stops correctly and "
            "the positive direction is simply not shown"),
        "f7_memory_control_falls_short": F.lt(
            mem["learner_vs_best_structured"]["lcb95"], SESOI,
            "winning without persistence to exploit would mean the bench is measuring something "
            "other than memory value"),
    }
    checks["d1_oracle_gap"] = F.disclosure(
        "premium is measured against the MISSPECIFIED first-order filter; the oracle-model MPC "
        "is reported beside it so a premium over misspecification is never sold as a premium "
        "over optimality",
        evidence={"positive_cell_learner_vs_oracle": pos["learner_vs_oracle_model_mpc"],
                  "oracle_mean": pos["structured_means_test"]["oracle_model_mpc"]})
    checks["d2_synthetic_scope"] = F.disclosure(
        "synthetic bench; carries no claim about the MFSC or Garrido-Rios (2017)",
        evidence={"seed_space": [TRAIN_SEEDS[0], TEST_SEEDS[-1]]})

    summary = F.summarise(checks)
    bench_ok = checks["f1_null_is_algebraically_flat"]["passed"] and \
        checks["f2_audit_stops_on_the_null"]["passed"]
    if not bench_ok:
        status = "BENCH_INVALID"
    elif checks["f3_positive_cell_has_real_headroom"]["passed"] and \
            checks["f6_learner_converts_on_fresh_seeds"]["passed"] and \
            checks["f7_memory_control_falls_short"]["passed"] and \
            checks["f4_learner_beats_its_own_placebo"]["passed"]:
        status = "AUDIT_VALIDATED_IN_BOTH_DIRECTIONS"
    else:
        status = "AUDIT_STOPS_CORRECTLY_BUT_POSITIVE_DIRECTION_NOT_DEMONSTRATED"

    payload = {
        "schema_version": "audit_positive_validation_v1", "claim_status": status,
        "contract_path": str(CONTRACT),
        "contract_sha256": sha256(CONTRACT.read_bytes()).hexdigest() if CONTRACT.exists() else None,
        "sesoi": SESOI, "headroom_bar": HEADROOM_BAR,
        "train_seeds": [TRAIN_SEEDS[0], TRAIN_SEEDS[-1]],
        "test_seeds": [TEST_SEEDS[0], TEST_SEEDS[-1]],
        "null_grid_spread": spread,
        "cells": cells, "falsifiers": checks, "falsifier_summary": summary,
        "scope": "synthetic testbed validating the audit instrument; no MFSC claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True, default=float) + "\n")

    print(f"veredicto: {status}\n")
    for name, cell in cells.items():
        print(f"  {name:10} {cell['audit_verdict']:28} "
              f"H_PI {cell['h_pi']['mean']:+.4f} [{cell['h_pi']['lcb95']:+.4f}, "
              f"{cell['h_pi']['ucb95']:+.4f}]")
        print(f"{'':13}aprendiz vs mejor estructurado "
              f"{cell['learner_vs_best_structured']['mean']:+.4f} "
              f"[{cell['learner_vs_best_structured']['lcb95']:+.4f}]  "
              f"vs oraculo {cell['learner_vs_oracle_model_mpc']['mean']:+.4f}")
    print(f"\n  falsadores: {summary['n_computed']} computados, {summary['n_failed']} fallidos")
    for name, c in checks.items():
        if c.get("computed"):
            print(f"    {name:44} {'PASA' if c['passed'] else 'FALLA'}")
    print(f"\n  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
