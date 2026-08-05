#!/usr/bin/env python3
"""The two surface gates, with estimators that can be cited.

g2 asks whether the design surface is SEPARABLE. It matters because one-factor-at-a-time -- the
thesis's own design -- is near-optimal by construction on a separable surface, so if the surface
separates there is no search problem to learn and the whole outer-loop lane has no mechanism.

g1 asks whether adapting to the risk context is WORTH anything, not merely whether the argmax
moves. A moving argmax is nearly free under noise; H_regime is the quantity that decides.

Both run off the sealed surface cache. No simulation, no seeds.

Contract: docs/ENMIENDA_GATES_SUPERFICIE_2026-08-05.md
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402

FACTORS = {
    "buffer_hours": (0.0, 168.0, 336.0, 504.0, 672.0, 1344.0),
    "shifts": (1, 2, 3),
    "op9_rop": (12.0, 24.0, 36.0, 48.0),
    "op12_rop": (12.0, 24.0, 36.0, 48.0),
}
FACTOR_NAMES = tuple(FACTORS)
CONFIGS = tuple(dict(zip(FACTOR_NAMES, combo)) for combo in itertools.product(*FACTORS.values()))
GATE_THRESHOLD = 0.05
N_BOOT = 5_000
MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")


def load_cache(root: Path) -> tuple[dict, list[str], list[int]]:
    """`surface[(context, seed)]` -> np.array of 288 values, in canonical CONFIGS order."""
    surface, contexts, seeds = {}, [], set()
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text())
        ctx, seed = payload["context"], int(payload["seed"])
        values = np.array([c["value"] for c in payload["cells"]], dtype=float)
        if values.size != len(CONFIGS):
            raise SystemExit(f"{path}: {values.size} cells, expected {len(CONFIGS)}")
        surface[(ctx, seed)] = values
        seeds.add(seed)
        if ctx not in contexts:
            contexts.append(ctx)
    return surface, contexts, sorted(seeds)


def design_matrix(*, interactions: bool) -> np.ndarray:
    """One-hot main effects, optionally plus the six pairwise factor products.

    Levels are coded by their INDEX scaled to [0,1] for the interaction terms, so a product is a
    genuine interaction rather than a re-encoding of the main effect.
    """
    cols = [np.ones(len(CONFIGS))]
    for name in FACTOR_NAMES:                       # main effects: one-hot, first level dropped
        for level in FACTORS[name][1:]:
            cols.append(np.array([1.0 if c[name] == level else 0.0 for c in CONFIGS]))
    if interactions:
        scaled = {n: np.array([FACTORS[n].index(c[n]) / (len(FACTORS[n]) - 1) for c in CONFIGS])
                  for n in FACTOR_NAMES}
        for a, b in itertools.combinations(FACTOR_NAMES, 2):
            cols.append(scaled[a] * scaled[b])
    return np.column_stack(cols)


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def leave_one_seed_out_delta(surface, ctx, seeds) -> np.ndarray:
    """Held-out R2 gain from interactions, one value per held-out seed.

    Grouping by SEED is the point: the 288 configurations repeat inside every seed, so a row-wise
    split would put the same configuration in train and test and both models would look perfect.
    """
    x_add, x_int = design_matrix(interactions=False), design_matrix(interactions=True)
    out = []
    for held in seeds:
        train = [s for s in seeds if s != held]
        y_tr = np.mean([surface[(ctx, s)] for s in train], axis=0)
        y_te = surface[(ctx, held)]
        preds = []
        for x in (x_add, x_int):
            beta, *_ = np.linalg.lstsq(x, y_tr, rcond=None)
            preds.append(x @ beta)
        out.append(r2(y_te, preds[1]) - r2(y_te, preds[0]))
    return np.asarray(out, dtype=float)


def h_regime(surface, contexts, seeds) -> float:
    """mean_ctx[max_cfg(mean_seed V)] - max_cfg(mean_ctx(mean_seed V)), V normalised per context.

    Averaging over seeds BEFORE the max is what makes this H_regime and not per-seed clairvoyance;
    taking the max per (context, seed) inflated an E*-C estimate roughly tenfold.
    """
    per_ctx = {}
    for ctx in contexts:
        mean = np.mean([surface[(ctx, s)] for s in seeds], axis=0)
        lo, hi = mean.min(), mean.max()
        per_ctx[ctx] = (mean - lo) / (hi - lo) if hi > lo else np.zeros_like(mean)
    stacked = np.stack([per_ctx[c] for c in contexts])          # (contexts, configs)
    return float(stacked.max(axis=1).mean() - stacked.mean(axis=0).max())


def boot_lcb(values: np.ndarray, rng, n_boot: int = N_BOOT) -> dict:
    draws = rng.integers(0, values.size, size=(n_boot, values.size))
    stats = values[draws].mean(axis=1)
    return {"mean": float(values.mean()), "lcb95": float(np.percentile(stats, 2.5)),
            "ucb95": float(np.percentile(stats, 97.5)), "n": int(values.size)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path("results/surface_cache/wrap288_v1"))
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/surface_gates/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(20260805)

    surface, contexts, seeds = load_cache(args.cache)
    print(f"  caché: {len(contexts)} contextos x {len(seeds)} semillas x {len(CONFIGS)} configs")

    # ---- g2: separability, leave-one-seed-out ------------------------------------------------
    g2 = {}
    for ctx in contexts:
        deltas = leave_one_seed_out_delta(surface, ctx, seeds)
        g2[ctx] = boot_lcb(deltas, rng)
    g2_pass = any(v["lcb95"] >= GATE_THRESHOLD for v in g2.values())

    # ---- g1: H_regime, bootstrapped over seeds ----------------------------------------------
    point = h_regime(surface, contexts, seeds)
    draws = np.array([h_regime(surface, contexts,
                               [seeds[i] for i in rng.integers(0, len(seeds), len(seeds))])
                      for _ in range(1_000)])
    g1 = {"H_regime": point, "lcb95": float(np.percentile(draws, 2.5)),
          "ucb95": float(np.percentile(draws, 97.5)), "n_boot": 1_000,
          "normalisation": "per-context min-max of the seed-averaged surface"}
    g1_pass = g1["lcb95"] >= GATE_THRESHOLD

    argmax = {}
    for ctx in contexts:
        mean = np.mean([surface[(ctx, s)] for s in seeds], axis=0)
        argmax[ctx] = dict(CONFIGS[int(mean.argmax())])
    common = np.stack([
        (lambda m: (m - m.min()) / (m.max() - m.min()) if m.max() > m.min() else m * 0.0)(
            np.mean([surface[(c, s)] for s in seeds], axis=0)) for c in contexts]).mean(axis=0)
    common_config = dict(CONFIGS[int(common.argmax())])

    verdict = ("SURFACE_SUPPORTS_A_SEARCH_LANE" if g2_pass and g1_pass
               else "NON_SEPARABLE_BUT_CONTEXT_INVARIANT" if g2_pass
               else "STOP_SEPARABLE_SURFACE_OFAT_IS_NEAR_OPTIMAL")

    falsifiers = {
        "g2_surface_is_non_separable": {
            "passed": bool(g2_pass),
            "evidence": {"why_it_can_fail": "on a separable surface the additive model predicts "
                                            "held-out seeds just as well and the interaction gain "
                                            "collapses to zero or below, paying for its extra "
                                            "parameters; then OFAT is near-optimal by construction",
                         "threshold": GATE_THRESHOLD, "by_context": g2,
                         "cv": "leave-one-seed-out, 12 folds; the seed is the resampling unit"}},
        "g1_context_adaptation_is_worth_something": {
            "passed": bool(g1_pass),
            "evidence": {"why_it_can_fail": "a moving argmax is nearly free under noise; H_regime "
                                            "asks whether knowing the regime BUYS anything, and it "
                                            "has measured exactly zero elsewhere in this project",
                         "threshold": GATE_THRESHOLD, "h_regime": g1,
                         "argmax_by_context": argmax, "best_common_config": common_config}},
        "f_no_fresh_seeds": custody_falsifier(seeds, replay_of=args.replay_of,
                                              exclude=args.output),
    }
    falsifiers["all_passed"] = all(
        v["passed"] for k, v in falsifiers.items()
        if k != "all_passed" and isinstance(v, dict) and not v.get("not_applicable"))

    print("\n  g2 separabilidad (ΔCV-R², leave-one-seed-out):")
    for ctx, v in g2.items():
        flag = "PASA" if v["lcb95"] >= GATE_THRESHOLD else "no"
        print(f"    {ctx:<16} {v['mean']:+.4f} [LCB95 {v['lcb95']:+.4f}]  {flag}")
    print(f"\n  g1 H_regime {g1['H_regime']:+.4f} [LCB95 {g1['lcb95']:+.4f}]  "
          f"(umbral {GATE_THRESHOLD})")
    print(f"  argmax por contexto: {len({json.dumps(v, sort_keys=True) for v in argmax.values()})}"
          f" distintos de {len(contexts)}")
    print(f"\n  veredicto: {verdict}\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        label = "NO APLICA" if f.get("not_applicable") else ("PASA" if f["passed"] else "FALLA")
        print(f"    {name:<46} {label}")

    payload = {
        "schema_version": "surface_gates_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION_NO_LEARNER",
        "run_role": "CACHE_ANALYSIS", "replay_of": args.replay_of,
        "module_manifest": module_manifest(MODULES, script=__file__),
        "cache": str(args.cache), "contexts": contexts, "seeds": seeds,
        "n_configurations": len(CONFIGS), "threshold": GATE_THRESHOLD,
        "g2_separability": g2, "g1_h_regime": g1,
        "argmax_by_context": argmax, "best_common_config": common_config,
        "falsifiers": falsifiers,
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/garrido_normaliser_audit/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
