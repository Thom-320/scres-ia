#!/usr/bin/env python3
"""How much H_regime can a monotone rescaling of the Cobb-Douglas index manufacture?

THE PI'S PROPOSAL, MADE MEASURABLE. He asked whether we could fine-tune the index so that it still
grows with the normalised one -- a strictly increasing transform, so the ordering never changes --
while giving more training signal. I answered with an argument. This measures it instead.

WHY THE ANSWER IS NOT OBVIOUS. Our estimator normalises each context by min-max AFTER the metric
is computed, so the maximum normalised value is 1 in every context and

    H_regime = 1 - max_a mean_r V_norm(r, a)

A monotone f does NOT commute with that outer max-of-means, so it can move H. What it cannot do is
move H when one configuration is the argmax in every regime: f preserves argmaxes, so that single
configuration stays optimal everywhere and both terms stay equal. The 288 grid has
scalar_h_regime = 0.0 exactly, which is that case -- and the prediction is a falsifier here.

WHAT THE CEILING COSTS. The supremum over increasing f is attained by a STEP, because H is one
minus a max (convex) of a mean that is linear in f, so the infimum of that max sits at an extreme
point of the bounded-increasing cone, and those extreme points are the indicators 1[V >= t]. A
step splits the configurations into two classes and cannot keep more than ~50% of the pairwise
ordering. So the transform that maximises headroom is the one that destroys the most signal, and
the decision-relevant number is not the ceiling but H_at_90pct_resolution: the best headroom
reachable while keeping 90% of the ordering the PI wants to keep.

Preregistration: docs/PREREGISTRO_TECHO_MONOTONO_COBB_DOUGLAS_2026-08-06.md
Development on burned tapes. Adjudicates nothing, changes no primary endpoint.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.cobb_douglas_resilience import (  # noqa: E402
    VARIABLES, derive_exponents, kappa_dot, resilience_index,
)
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402

GATE = 0.05
RESOLUTION_FLOOR = 0.90
N_RANDOM_MONOTONE = 2_000
MODULES = ("supply_chain/cobb_douglas_resilience.py", "supply_chain/arm_runner.py",
           "supply_chain/seed_custody.py")

GRIDS = {
    "wrap288_v1": Path("results/cobb_douglas_component_headroom"),
    "wrap288_compat_extended_v1": Path("results/cobb_douglas_component_headroom_extended"),
}


def h_regime(per_ctx: np.ndarray) -> float:
    """Byte-for-byte the estimator used everywhere else: per-context min-max, mean over seeds
    already taken, max over actions AFTER the seed average. A context that the transform has
    flattened normalises to zeros -- the same handling every other runner gives it, kept identical
    so these numbers stay comparable with the sealed ones."""
    norm = []
    for m in np.asarray(per_ctx, dtype=float):
        lo, hi = m.min(), m.max()
        norm.append((m - lo) / (hi - lo) if hi > lo else np.zeros_like(m))
    stacked = np.stack(norm)
    return float(stacked.max(axis=1).mean() - stacked.mean(axis=0).max())


def resolution(per_ctx: np.ndarray) -> float:
    """Fraction of configuration PAIRS the surface still orders strictly, averaged over contexts.
    Computed from tie counts rather than the O(n^2) pair list: 4,608 configurations would be 10.6M
    pairs per context."""
    out = []
    for m in np.asarray(per_ctx, dtype=float):
        n = m.size
        _, counts = np.unique(m, return_counts=True)
        tied = float(np.sum(counts * (counts - 1)) / 2.0)
        total = float(n * (n - 1) / 2.0)
        out.append(1.0 - tied / total if total else 0.0)
    return float(np.mean(out))


def load_surface(grid: str) -> tuple[np.ndarray, list[str], list[int]]:
    """The seed-averaged canonical index per (context, configuration), from the cached aggregates.

    Canonical means the project's index: his five variables, exponents from HIS rule applied to OUR
    maxima, kappa_dot relative to the set within one (context, seed). Anything else would not be
    comparable with the sealed scalar_h_regime this run anchors against.
    """
    home = GRIDS[grid]
    sealed = json.loads((home / "result.json").read_text())
    raw = json.loads((home / "aggregates.json").read_text())
    # rsplit, not split: context names contain "|" themselves (R1r|esc).
    agg = {(k.rsplit("|", 1)[0], int(k.rsplit("|", 1)[1])): v for k, v in raw.items()}
    contexts, seeds = list(sealed["contexts"]), list(sealed["seeds"])

    kd = {key: kappa_dot({str(i): a["kappa"] for i, a in enumerate(cell)})
          for key, cell in agg.items()}
    maxima = {v: max(float(a[v]) for cell in agg.values() for a in cell)
              for v in VARIABLES if v != "kappa_dot"}
    maxima["kappa_dot"] = max(float(k) for row in kd.values() for k in row.values())
    exponents = derive_exponents(maxima)

    scalar = {}
    for key, cell in agg.items():
        row = kd[key]
        scalar[key] = np.array([
            resilience_index({vv: (row[str(i)] if vv == "kappa_dot" else float(a[vv]))
                              for vv in VARIABLES}, exponents)["R_cobb_douglas"]
            for i, a in enumerate(cell)])

    surface = np.stack([np.mean([scalar[(c, s)] for s in seeds], axis=0) for c in contexts])
    return surface, contexts, seeds


def random_monotone(values: np.ndarray, rng, n_knots: int = 64) -> np.ndarray:
    """A random strictly increasing map, built by interpolating a random increasing knot sequence.
    These exist to try to BEAT the best step; if one does, the step argument is wrong."""
    lo, hi = values.min(), values.max()
    xs = np.linspace(lo, hi, n_knots)
    ys = np.cumsum(rng.exponential(size=n_knots))
    ys = (ys - ys[0]) / (ys[-1] - ys[0]) if ys[-1] > ys[0] else np.linspace(0, 1, n_knots)
    return np.interp(values, xs, ys)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/monotone_transform_ceiling/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(20260806)

    grids, all_seeds = {}, set()
    for grid in GRIDS:
        V, contexts, seeds = load_surface(grid)
        all_seeds.update(seeds)
        sealed = json.loads((GRIDS[grid] / "result.json").read_text())["scalar_h_regime"]
        h_id = h_regime(V)
        res_id = resolution(V)
        print(f"\n  {grid}: {V.shape[0]} contextos x {V.shape[1]:,} configuraciones")
        print(f"    H identidad {h_id:+.6f}  (sellado {sealed:+.6f})  resolución {res_id:.4f}")

        argmaxes = [int(np.argmax(V[r])) for r in range(V.shape[0])]
        universal = len(set(argmaxes)) == 1
        print(f"    argmax por régimen {argmaxes} -> "
              f"{'UNO SOLO en los seis' if universal else 'NO coinciden'}")

        # --- the step family: the predicted supremum -------------------------------------------
        pooled = np.unique(V)
        cuts = pooled if pooled.size <= 4_000 else np.quantile(pooled, np.linspace(0, 1, 4_000))
        best_step = {"H_regime": -1.0}
        for t in cuts:
            W = (V >= t).astype(float)
            h = h_regime(W)
            if h > best_step["H_regime"]:
                best_step = {"H_regime": h, "threshold": float(t), "resolution": resolution(W),
                             "above_fraction": float(np.mean(W))}

        # --- the logistic family: the PI's proposal, with sharpness as the dial -----------------
        # beta -> 0 is affine (the identity's ordering AND its spacing); beta -> inf is the step.
        # This is the frontier he asked about: how much signal survives at each level of headroom.
        scale = float(V.std()) or 1.0
        frontier = []
        for t in np.quantile(pooled, np.linspace(0.02, 0.98, 25)):
            for beta in np.geomspace(0.05, 500.0, 20):
                z = np.clip(beta * (V - t) / scale, -700.0, 700.0)   # sharp betas overflow exp
                W = 1.0 / (1.0 + np.exp(-z))
                frontier.append({"t": float(t), "beta": float(beta), "H_regime": h_regime(W),
                                 "resolution": resolution(W)})

        # --- random monotone maps: the falsifier for the step argument --------------------------
        best_random = {"H_regime": -1.0}
        for _ in range(N_RANDOM_MONOTONE):
            W = random_monotone(V, rng)
            h = h_regime(W)
            if h > best_random["H_regime"]:
                best_random = {"H_regime": h, "resolution": resolution(W)}

        sampled = frontier + [best_random]
        ceiling_sampled = max(s["H_regime"] for s in sampled)
        ceiling = max(best_step["H_regime"], ceiling_sampled)

        # THE decision-relevant number: what he asked for is more signal at the SAME ordering, so
        # the honest quote is the best headroom that keeps the ordering nearly intact.
        keepers = [s for s in frontier + [best_random] + [best_step]
                   if s["resolution"] >= RESOLUTION_FLOOR]
        h_at_floor = max((s["H_regime"] for s in keepers), default=0.0)
        best_kept = max(keepers, key=lambda s: s["H_regime"], default=None)

        print(f"    techo escalón      {best_step['H_regime']:+.6f} "
              f"(umbral {best_step['threshold']:.6f}, resolución {best_step['resolution']:.4f})")
        print(f"    techo muestreado   {ceiling_sampled:+.6f}   "
              f"(mejor aleatoria {best_random['H_regime']:+.6f})")
        print(f"    H con >= {RESOLUTION_FLOOR:.0%} de resolución  {h_at_floor:+.6f}"
              f"   contra umbral {GATE}")

        grids[grid] = {
            "contexts": contexts, "seeds": seeds,
            "n_configurations": int(V.shape[1]),
            "H_identity": h_id, "sealed_scalar_h_regime": float(sealed),
            "resolution_identity": res_id,
            "argmax_per_context": argmaxes, "argmax_is_universal": bool(universal),
            "ceiling_step": best_step, "ceiling_sampled": ceiling_sampled,
            "ceiling": ceiling, "best_random_monotone": best_random,
            "H_at_90pct_resolution": h_at_floor,
            "best_transform_keeping_resolution": best_kept,
            "frontier": frontier,
        }

    # --- verdict, on the rules fixed in the preregistration -------------------------------------
    ceiling = max(g["ceiling"] for g in grids.values())
    h_floor = max(g["H_at_90pct_resolution"] for g in grids.values())
    verdict = ("MONOTONE_RESCALING_CANNOT_REACH_THE_BAR" if ceiling < GATE
               else "THE_BAR_IS_ONLY_REACHED_BY_DESTROYING_THE_SIGNAL" if h_floor < GATE
               else "A_MONOTONE_RESCALING_REACHES_THE_BAR_WITH_SIGNAL_INTACT")

    falsifiers = {
        "f1_identity_reproduces_the_sealed_scalar": {
            "passed": all(abs(g["H_identity"] - g["sealed_scalar_h_regime"]) < 1e-9
                          for g in grids.values()),
            "evidence": {"why_it_can_fail": "the index is rebuilt here from the cached aggregates; "
                                            "any drift in the exponent derivation or the kappa_dot "
                                            "comparison set makes a new ceiling incomparable with "
                                            "everything already published",
                         "deviations": {k: abs(g["H_identity"] - g["sealed_scalar_h_regime"])
                                        for k, g in grids.items()}}},
        "f2_steps_attain_the_supremum": {
            "passed": all(g["ceiling_sampled"] <= g["ceiling_step"]["H_regime"] + 1e-9
                          for g in grids.values()),
            "evidence": {"why_it_can_fail": "if any sampled increasing map beats the best step, the "
                                            "extreme-point argument is wrong and the reported "
                                            "ceiling is a floor, not a ceiling",
                         "n_random": N_RANDOM_MONOTONE,
                         "margins": {k: g["ceiling_step"]["H_regime"] - g["ceiling_sampled"]
                                     for k, g in grids.items()}}},
        "f3_the_base_grid_ceiling_is_zero": {
            "passed": (grids["wrap288_v1"]["argmax_is_universal"]
                       and grids["wrap288_v1"]["ceiling"] < 1e-9),
            "evidence": {"why_it_can_fail": "I predicted that a scalar_h_regime of exactly 0 means "
                                            "one configuration is optimal in all six regimes, and "
                                            "that no increasing f can then move H off zero. If the "
                                            "six argmaxes differ, or the ceiling is positive, the "
                                            "prediction is false",
                         "argmax_per_context": grids["wrap288_v1"]["argmax_per_context"],
                         "ceiling": grids["wrap288_v1"]["ceiling"]}},
        "f4_headroom_and_resolution_trade_off": {
            "passed": all(g["ceiling_step"]["resolution"] < g["resolution_identity"]
                          for g in grids.values()),
            "evidence": {"why_it_can_fail": "the whole objection to the proposal is that the "
                                            "transform maximising headroom is the one destroying "
                                            "the training signal. If resolution survived at the "
                                            "optimum there would be no tension and the PI's idea "
                                            "would cost nothing",
                         "at_ceiling": {k: g["ceiling_step"]["resolution"]
                                        for k, g in grids.items()},
                         "identity": {k: g["resolution_identity"] for k, g in grids.items()}}},
        "f5_no_fresh_seeds": custody_falsifier(sorted(all_seeds), replay_of=args.replay_of,
                                               exclude=args.output),
    }
    falsifiers["all_passed"] = all(
        v["passed"] for k, v in falsifiers.items()
        if k != "all_passed" and isinstance(v, dict) and not v.get("not_applicable"))

    print(f"\n  veredicto: {verdict}")
    print(f"    techo máximo             {ceiling:+.6f}   contra umbral {GATE}")
    print(f"    H conservando >= 90 %    {h_floor:+.6f}   contra umbral {GATE}\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        label = "NO APLICA" if f.get("not_applicable") else ("PASA" if f["passed"] else "FALLA")
        print(f"    {name:<46} {label}")

    payload = {
        "schema_version": "monotone_transform_ceiling_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION",
        "run_role": "CACHE_ANALYSIS", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": "docs/PREREGISTRO_TECHO_MONOTONO_COBB_DOUGLAS_2026-08-06.md",
        "gate": GATE, "resolution_floor": RESOLUTION_FLOOR,
        "estimator": ("H_regime = 1 - max_a mean_r V_norm(r,a); per-context min-max normalisation "
                      "applied AFTER the transform, identical to every other runner"),
        "ceiling": ceiling, "H_at_90pct_resolution": h_floor,
        "what_a_positive_would_mean": (
            "A ceiling above the gate is a property of the metric's CURVATURE, not of the supply "
            "chain's physics: the ordering of configurations is unchanged by construction. Any "
            "headroom bought this way must be reported as manufactured, and the transform chosen "
            "on declared mechanism rather than on the H it yields."),
        "grids": grids,
        "falsifiers": falsifiers, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=GRIDS["wrap288_v1"] / "result.json")
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
