#!/usr/bin/env python3
"""Garrido's Pareto question: does the same resilience come cheaper in shifts?

HIS ASK, 22 July: "promedio de shifts usados como proxy de eficiencia de recursos -- un algoritmo
con resiliencia similar pero menor promedio de shifts dominaria en frontera Pareto."

WHAT THIS IS AND IS NOT. This is a DESCRIPTIVE diagnostic over the sealed 288 surface, not a
hypothesis test: it has no gate, no LCB and no verdict, because the question is not "is there an
effect" but "what does the frontier look like". Reporting it with a claim_status would dress a
description as an adjudication, and this project has been burned by exactly that.

The frontier is computed per context over the seed-averaged surface. A configuration is on it when
no other configuration has both at-least-as-good resilience and strictly fewer shifts. The number
that answers his question is `shifts_at_the_optimum` against `min_shifts_within_1pct`: if the best
configuration uses 3 shifts but something within 1% of it uses 1, the cheaper one dominates in his
sense and we should say so.

Development on the burned block. Adjudicates nothing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_search_comparator_ladder_v2 import CONFIGS, CONTEXT_ORDER, load_cache  # noqa: E402

MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")
WITHIN = 0.01                     # "resiliencia similar" = dentro del 1 % del optimo del contexto


def pareto_front(shifts: np.ndarray, value: np.ndarray) -> list[int]:
    """Indices not dominated: nothing else has value >= and shifts <, or value > and shifts <=."""
    out = []
    for i in range(len(value)):
        dominated = np.any(((value >= value[i]) & (shifts < shifts[i]))
                           | ((value > value[i]) & (shifts <= shifts[i])))
        if not dominated:
            out.append(int(i))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path("results/surface_cache/wrap288_v1"))
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/shift_pareto_diagnostic/result.json"))
    args = ap.parse_args()
    surface, contexts, seeds = load_cache(args.cache)
    shifts = np.array([float(c["shifts"]) for c in CONFIGS])

    report = {}
    for ctx in [c for c in CONTEXT_ORDER if c in contexts]:
        v = np.mean([surface[(ctx, s)] for s in seeds], axis=0)
        best = int(np.argmax(v))
        span = float(v.max() - v.min()) or 1.0
        near = np.where(v >= v.max() - WITHIN * span)[0]
        front = pareto_front(shifts, v)
        cheapest_near = int(near[np.argmin(shifts[near])])
        report[ctx] = {
            "optimum_index": best,
            "optimum_value": float(v[best]),
            "shifts_at_the_optimum": float(shifts[best]),
            "n_within_1pct": int(near.size),
            "min_shifts_within_1pct": float(shifts[near].min()),
            "cheapest_near_optimal": {
                "index": cheapest_near, "shifts": float(shifts[cheapest_near]),
                "value": float(v[cheapest_near]),
                "value_gap_vs_optimum": float(v[best] - v[cheapest_near]),
                "gap_as_pct_of_span": float((v[best] - v[cheapest_near]) / span * 100.0)},
            "shifts_can_be_reduced_at_near_equal_resilience":
                bool(shifts[near].min() < shifts[best]),
            "pareto_front_size": len(front),
            "mean_value_by_shift_level": {
                str(int(lv)): float(v[shifts == lv].mean()) for lv in np.unique(shifts)},
            "best_value_by_shift_level": {
                str(int(lv)): float(v[shifts == lv].max()) for lv in np.unique(shifts)},
        }

    savers = [c for c, r in report.items() if r["shifts_can_be_reduced_at_near_equal_resilience"]]
    print(f"  {len(contexts)} contextos · {len(seeds)} semillas · umbral 'similar' = {WITHIN:.0%} "
          f"del rango del contexto\n")
    for ctx, r in report.items():
        best_by = r["best_value_by_shift_level"]
        ladder = "  ".join(f"{k}t:{val:.5f}" for k, val in sorted(best_by.items()))
        flag = "  <-- SE PUEDE BAJAR" if r["shifts_can_be_reduced_at_near_equal_resilience"] else ""
        print(f"    {ctx:<14} óptimo con {r['shifts_at_the_optimum']:.0f} turnos · "
              f"mínimo dentro del 1 % = {r['min_shifts_within_1pct']:.0f} turnos "
              f"({r['n_within_1pct']} configs){flag}")
        print(f"                   mejor por nivel de turnos: {ladder}")

    falsifiers = {
        "f1_the_frontier_is_not_everything": {
            "passed": all(r["pareto_front_size"] < len(CONFIGS) for r in report.values()),
            "evidence": {"why_it_can_fail": "if every configuration were on the frontier the "
                                            "dominance test would be vacuous and the diagnostic "
                                            "would say nothing",
                         "front_sizes": {c: r["pareto_front_size"] for c, r in report.items()},
                         "n_configs": len(CONFIGS)}},
        "f2_shifts_actually_vary_in_the_grid": {
            "passed": len(set(shifts.tolist())) > 1,
            "evidence": {"why_it_can_fail": "a constant shift column would make the whole question "
                                            "unanswerable on this grid",
                         "levels": sorted(set(shifts.tolist()))}},
        "f3_no_fresh_seeds": custody_falsifier(seeds, replay_of=args.replay_of,
                                               exclude=args.output),
    }
    falsifiers["all_passed"] = all(
        v["passed"] for k, v in falsifiers.items()
        if k != "all_passed" and isinstance(v, dict) and not v.get("not_applicable"))

    payload = {
        "schema_version": "shift_pareto_diagnostic_v1",
        "claim_status": "DESCRIPTIVE_DIAGNOSTIC_NO_ADJUDICATION",
        "scope": "DEVELOPMENT_ON_BURNED_TAPES",
        "run_role": "CACHE_ANALYSIS", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "question": ("Garrido, 22 July: mean shifts as a resource-efficiency proxy -- an algorithm "
                     "with similar resilience but fewer shifts would dominate on the Pareto front."),
        "what_this_is_not": ("A hypothesis test. There is no gate, no LCB and no verdict, because "
                             "the question is what the frontier looks like, not whether an effect "
                             "exists."),
        "similarity_threshold": WITHIN, "contexts": contexts, "seeds": seeds,
        "by_context": report,
        "contexts_where_shifts_can_be_reduced": savers,
        "falsifiers": falsifiers,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/search_ladder_v5/result.json"))
    print(f"\n  contextos donde bajar turnos sale casi gratis: {len(savers)}/{len(report)}")
    print(f"  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
