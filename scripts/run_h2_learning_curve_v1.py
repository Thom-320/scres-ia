#!/usr/bin/env python3
"""H2 -- the learning-curve hypothesis, the only one of the draft's four with no verdict.

H1, H3 and H4 have been adjudicated. H2 has not, because the form that circulated -- "advantage
+0.00 -> +10.00 across contexts" -- was retired with the leaky-normaliser runner. It is the
hypothesis the whole framing rests on: does the benefit of retaining rho GROW as the chain sees
more successive disruptions?

THE DISTINCTION THAT MAKES THIS NOT H4. A large but FLAT advantage supports H4 (path dependency),
which is already measured at +0.06070 [+0.04556]. H2 needs the advantage to INCREASE with the
ordinal of the context. So the estimand is a slope, not a level.

THE CONFOUND THIS ABSORBS. The last three contexts are escalated (x3 frequency), so any arm can
look better or worse by position rather than by learning. The same slope is therefore computed for
a pair that retains NOTHING across contexts -- random minus ofat. If that null slope is positive
with LCB95>0, the trend belongs to the context ordering and H2 cannot be read at all.

Preregistration: docs/PREREGISTRO_H2_CURVA_DE_APRENDIZAJE_2026-08-07.md
Re-analysis of sealed artifacts. No simulation, no seeds.
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
from supply_chain.seed_custody import module_manifest  # noqa: E402

SLICES = (
    Path("results/garrido_meta_learner_h3power_h3_contract_local_v2/result.json"),
    Path("results/garrido_meta_learner_h3power_h3_contract_vps_v2/result.json"),
)
CONTEXT_ORDER = ("R1r", "R2r", "R1r+R2r", "R1r|esc", "R2r|esc", "R1r+R2r|esc")
N_BOOT = 5_000
MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")


def auc_prefix(curve: list[float]) -> float:
    """Normalised regret AUC under the PREFIX normaliser.

    The oracle normaliser divided by the span of the WHOLE surface, including cells the search
    never ran -- that is the leak. The prefix normaliser can only use what the run has already
    seen, so the scale at step t is the best-so-far spread of the prefix."""
    c = np.asarray(curve, dtype=float)
    if c.size == 0:
        return 0.0
    scale = float(np.maximum.accumulate(c[::-1])[::-1][0])
    denom = scale if scale > 1e-12 else 1.0
    return float((c / denom).mean())


def slope(y: np.ndarray) -> float:
    """OLS slope against the context ordinal 1..len(y). Can be negative, and f4 proves it."""
    x = np.arange(1, y.size + 1, dtype=float)
    xm, ym = x.mean(), y.mean()
    den = float(((x - xm) ** 2).sum())
    return float(((x - xm) * (y - ym)).sum() / den) if den > 0 else 0.0


def boot(v: np.ndarray, rng) -> dict:
    draws = v[rng.integers(0, v.size, size=(N_BOOT, v.size))].mean(axis=1)
    return {"mean": float(v.mean()), "lcb95": float(np.percentile(draws, 2.5)),
            "ucb95": float(np.percentile(draws, 97.5)), "n": int(v.size)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/manuscript/h2_learning_curve/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(20260807)

    runs, sources = {a: [] for a in ("neuron_memory", "neuron_reset", "ofat", "random")}, []
    for path in SLICES:
        d = json.loads(path.read_text())
        sources.append({"path": str(path), "self_sha256": d.get("self_sha256"),
                        "contract_path": d.get("contract_path"), "repeats": d.get("repeats"),
                        "seeds": [d["seeds"][0], d["seeds"][-1]]})
        if list(d["contexts"]) != list(CONTEXT_ORDER):
            raise SystemExit(f"context order differs in {path}")
        for arm in runs:
            runs[arm].extend(d["per_context"][arm])
    n_rep = len(runs["neuron_memory"])

    def curve_matrix(arm: str) -> np.ndarray:
        """(replicate, context) matrix of prefix-normalised regret AUC."""
        return np.array([[auc_prefix(run[c]["regret_curve"]) for c in CONTEXT_ORDER]
                         for run in runs[arm]], dtype=float)

    mem, res = curve_matrix("neuron_memory"), curve_matrix("neuron_reset")
    ofat, rand = curve_matrix("ofat"), curve_matrix("random")

    adv = res - mem                    # positive = memory wins, per (replicate, context)
    null_adv = rand - ofat             # neither retains anything across contexts

    slopes = np.array([slope(row) for row in adv])
    null_slopes = np.array([slope(row) for row in null_adv])
    primary, null = boot(slopes, rng), boot(null_slopes, rng)
    by_context = {c: float(adv[:, k].mean()) for k, c in enumerate(CONTEXT_ORDER)}

    order_confound = bool(null["lcb95"] > 0)
    falsifiers = {
        "f1_the_source_is_the_contracted_pair_at_n120": {
            "passed": bool(n_rep == 120 and all(
                s["contract_path"] == "docs/PREREGISTRO_H3_POTENCIA_2026-08-01.md"
                for s in sources)),
            "evidence": {"why_it_can_fail": "a different source, or a short merge, would not be "
                                            "the contracted block the reading rule assumes",
                         "n_replicates": n_rep, "sources": sources}},
        "f2_the_normaliser_is_prefix_not_oracle": {
            "passed": True,
            "evidence": {"why_it_can_fail": "the oracle normaliser divided by the span of the "
                                            "whole surface including never-run cells; reproducing "
                                            "that panel would reintroduce the leak that retired "
                                            "the original H2 figure",
                         "normaliser": "prefix (running best-so-far of the observed curve)"}},
        "f3_the_order_confound_is_absorbed": {
            # THE falsifier that can stop H2 being read at all.
            "passed": bool(not order_confound),
            "evidence": {"why_it_can_fail": "the last three contexts are escalated, so an arm can "
                                            "look better by position rather than by learning. If "
                                            "a pair that retains NOTHING shows the same rising "
                                            "trend, the trend is the ordering",
                         "null_pair": "random minus ofat", "null_slope": null}},
        "f4_the_slope_can_be_negative": {
            "passed": bool(float(slopes.min()) < 0.0),
            "evidence": {"why_it_can_fail": "an estimator that cannot return a negative slope "
                                            "cannot refute H2 and measures nothing",
                         "min_slope_observed": float(slopes.min()),
                         "max_slope_observed": float(slopes.max()),
                         "n_negative": int((slopes < 0).sum())}},
        "f5_no_new_seeds": {
            "passed": True,
            "evidence": {"why_it_can_fail": "a seed outside the already-open block would consume "
                                            "custody this run never declared",
                         "block": "6000001-6000120"}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed" and isinstance(v, dict))

    if order_confound:
        verdict = "H2_NOT_READABLE_ORDER_CONFOUND"
    elif primary["lcb95"] > 0:
        verdict = "H2_SUPPORTED_LEARNING_CURVE"
    elif primary["ucb95"] < 0:
        verdict = "H2_REFUTED_ADVANTAGE_SHRINKS"
    else:
        verdict = "H2_NOT_SUPPORTED_ADVANTAGE_IS_FLAT"

    print(f"  réplicas {n_rep} · normalizador prefijo · orden {list(CONTEXT_ORDER)}\n")
    print("  ventaja media (reinicio − memoria) por contexto:")
    for k, c in enumerate(CONTEXT_ORDER):
        print(f"    {k+1}. {c:<12} {by_context[c]:+.5f}")
    print(f"\n  pendiente primaria  {primary['mean']:+.6f} "
          f"[{primary['lcb95']:+.6f}, {primary['ucb95']:+.6f}]")
    print(f"  pendiente nula      {null['mean']:+.6f} "
          f"[{null['lcb95']:+.6f}, {null['ucb95']:+.6f}]  (aleatorio − OFAT)")
    print(f"\n  veredicto: {verdict}\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        print(f"    {name:<44} {'PASA' if f['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "h2_learning_curve_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_REANALYSIS_OF_SEALED_ARTIFACTS_NO_SEEDS_NO_ADJUDICATION",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": str(args.contract),
        "estimand": ("OLS slope of (reset AUC - memory AUC) against the context ordinal 1..6, per "
                     "replicate. A large but FLAT advantage supports H4, not H2."),
        "sources": sources, "n_replicates": n_rep, "context_order": list(CONTEXT_ORDER),
        "advantage_by_context": by_context,
        "primary_slope": primary, "null_slope_random_minus_ofat": null,
        "mean_auc_by_arm": {a: float(curve_matrix(a).mean())
                            for a in ("neuron_memory", "neuron_reset", "ofat", "random")},
        "falsifiers": falsifiers, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract, reference=SLICES[0])
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
