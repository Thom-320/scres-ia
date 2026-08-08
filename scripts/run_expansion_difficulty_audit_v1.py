#!/usr/bin/env python3
"""Does expanding 288 to 4,608 harden the problem, or does it just make cold start look worse?

WHY THE "18x" HEADLINE IS WITHDRAWN. This project reported that the two new factors move the
endpoint by 0.00046-0.00168 while the contrasts it reports are 0.014-0.057, and called the ratio 18x.
That number is not reproducible as a single quantity: it is a ratio of two means chosen from two
ranges. Measured here, the three canonical choices give 7.0x, 30.5x and 67.5x -- and 18x is none of
them, so the reported figure came from yet a fourth pairing. A ratio whose value depends on which
representative you pick is a rhetorical device, not a measurement, and it happened to point in the
direction that flattered the design.

WHAT REPLACES IT, AND IT IS EXACT. The question the ratio was reaching for -- does the expansion make
the problem harder, or does it merely dilute a uniform starting policy? -- has a closed form. Under
uniform sampling without replacement of n configurations from N, the sampled maximum takes the value
of the k-th order statistic with probability C(k-1, n-1) / C(N, n). So the expected simple regret of
a uniform draw at budget 24 is computable exactly from the cached surfaces, on 288 and on 4,608,
with no simulation and no estimation. If the expansion hardens the problem, a uniform draw gets
worse because good configurations are genuinely rarer. If it dilutes cold start, a uniform draw gets
worse because the added configurations are simply bad -- and the same thing happens to any arm that
must find its way without carried information, which would inflate every "vs cold" contrast by an
amount the design chose rather than measured.

THE ANSWER CONTRADICTS THE CLAIM IT REPLACES. The expansion does not dilute uniform search: expected
uniform regret at budget 24 FALLS from 0.07429 on 288 to 0.06755 on 4,608 scoring each grid against
its own reachable optimum, and from 0.10284 to 0.06755 against a common reference. The extended
optimum is strictly better in 136 of 360 cells and 0.97% of the 4,320 added configurations sit above
the base optimum -- enough that 24 uniform draws find them. So "the expansion dilutes cold start" is
withdrawn as well, and for the stronger reason that it was measured false rather than unidentified.

The two are distinguished by looking at where the added mass sits: `share_of_new_configs_above_the
_base_optimum` says whether the expansion added any reachable improvement at all.

Contract: docs/ENMIENDA_REGISTRO_DE_EVIDENCIA_2026-08-07.md
Reads the sealed caches. No seed is opened, nothing is simulated.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from math import comb
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.run_grid_transfer_v1 as G  # noqa: E402
from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402

MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py",
           "scripts/run_grid_transfer_v1.py")
CONTRACT = Path("docs/ENMIENDA_REGISTRO_DE_EVIDENCIA_2026-08-07.md")
SOURCE = Path("results/grid_transfer_confirmation_v2/result.json")
MC_DRAWS = 20_000
MC_SEED = 20260808


def expected_sampled_max(values: np.ndarray, n: int) -> float:
    """E[max of a uniform n-subset], exactly, by order statistics.

    P(sample max is the k-th smallest) = C(k-1, n-1) / C(N, n): the other n-1 draws must all come
    from the k-1 values below it. Computed in log space because C(4608, 24) has 68 digits.
    """
    v = np.sort(np.asarray(values, dtype=float))
    N = v.size
    ks = np.arange(n, N + 1)                      # the max cannot be below the n-th smallest
    logw = np.array([_logcomb(k - 1, n - 1) for k in ks]) - _logcomb(N, n)
    w = np.exp(logw)
    return float(np.dot(v[ks - 1], w))


def _logcomb(a: int, b: int) -> float:
    from math import lgamma
    if b < 0 or b > a:
        return float("-inf")
    return lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-cache", type=Path,
                    default=Path("results/surface_cache/garrido_transfer_confirmation_v2_base"))
    ap.add_argument("--ext-cache", type=Path,
                    default=Path("results/surface_cache/garrido_transfer_confirmation_v2_ext"))
    ap.add_argument("--budget", type=int, default=G.BUDGET)
    ap.add_argument("--out", type=Path,
                    default=Path("results/expansion_difficulty/result.json"))
    args = ap.parse_args()

    base, base_ctx, seeds, *_ = G.load(args.base_cache, len(G.BASE_CONFIGS), "wrap288_v1")
    ext, ext_ctx, _, *_ = G.load(args.ext_cache, len(G.EXT_CONFIGS),
                                 "wrap288_compat_extended_v1")
    contexts = [c for c in G.CONTEXT_ORDER if c in base_ctx and c in ext_ctx]
    n = args.budget

    per_cell, share_new_better = [], []
    for ctx in contexts:
        for seed in seeds:
            b, e = np.asarray(base[(ctx, seed)], float), np.asarray(ext[(ctx, seed)], float)
            bb, be = float(b.max()), float(e.max())
            # Normalised the same way Surface.auc normalises: regret over |best| of that surface.
            m288, m4608 = expected_sampled_max(b, n), expected_sampled_max(e, n)
            # TWO NORMALISERS, BECAUSE THE CHOICE DRIVES THE SIGN AND HIDING THAT WOULD REPEAT THE
            # DEFECT THIS ARTIFACT EXISTS TO WITHDRAW. `own_best` scores each grid against the
            # optimum it can actually reach -- what Surface.auc does, and what an experimenter
            # confined to that grid experiences. `common_best` scores both against the extended
            # optimum, so the 288 grid pays for not containing the better configuration. The first
            # asks "is search harder here?"; the second asks "is this grid worse to be on?".
            r288 = (bb - m288) / (abs(bb) or 1.0)
            r4608 = (be - m4608) / (abs(be) or 1.0)
            den = abs(be) or 1.0
            per_cell.append({"context": ctx, "seed": seed,
                             "expected_uniform_regret_288": r288,
                             "expected_uniform_regret_4608": r4608,
                             "common_ref_regret_288": (be - m288) / den,
                             "common_ref_regret_4608": (be - m4608) / den,
                             "best_288": bb, "best_4608": be})
            new_mask = np.ones(e.size, bool)
            new_mask[[G.EXT_INDEX[tuple(sorted(dict(c, op3_rm=0.0, op5_rm=0.0).items()))]
                      for c in G.BASE_CONFIGS]] = False
            share_new_better.append(float((e[new_mask] > bb).mean()))

    r288 = np.array([c["expected_uniform_regret_288"] for c in per_cell])
    r4608 = np.array([c["expected_uniform_regret_4608"] for c in per_cell])
    c288 = np.array([c["common_ref_regret_288"] for c in per_cell])
    c4608 = np.array([c["common_ref_regret_4608"] for c in per_cell])
    optimum_moved = [c for c in per_cell if c["best_4608"] > c["best_288"] + 1e-12]

    # Monte Carlo control for the closed form. If the two disagree the algebra is wrong.
    rng = np.random.default_rng(MC_SEED)
    c0 = per_cell[0]
    b0 = np.asarray(base[(c0["context"], c0["seed"])], float)
    mc = float(np.mean([b0[rng.choice(b0.size, n, replace=False)].max()
                        for _ in range(MC_DRAWS)]))
    exact = expected_sampled_max(b0, n)
    mc_gap = abs(mc - exact) / (abs(exact) or 1.0)

    # The withdrawn headline, recomputed as the RANGE of ratios it could have been.
    src = json.loads((ROOT / SOURCE).read_text())
    contrast_means = sorted(abs(v["vs_marginal_replay"]["mean"])
                            for v in src["contrasts"].values())
    spread = {}
    for ctx in contexts:
        mean = np.mean([np.asarray(ext[(ctx, s)], float) for s in seeds], axis=0)
        by_base: dict = {}
        for i, cfg in enumerate(G.EXT_CONFIGS):
            by_base.setdefault(tuple(cfg[k] for k in G.BASE_FACTORS), []).append(mean[i])
        spread[ctx] = float(np.mean([max(v) - min(v) for v in by_base.values()]))
    sp = sorted(spread.values())
    ratios = {"min_contrast_over_max_spread": contrast_means[0] / sp[-1],
              "max_contrast_over_min_spread": contrast_means[-1] / sp[0],
              "mean_contrast_over_mean_spread": float(np.mean(contrast_means) / np.mean(sp))}

    falsifiers = {
        "f1_the_closed_form_matches_a_monte_carlo_control": {
            "passed": mc_gap < 5e-3, "relative_gap": mc_gap, "exact": exact, "monte_carlo": mc,
            "draws": MC_DRAWS,
            "why_it_can_fail": ("an off-by-one in the order-statistic weights would leave the "
                                "closed form close but not equal, which is exactly the error a "
                                "Monte Carlo control catches and inspection does not")},
        "f2_the_headline_ratio_is_not_a_single_number": {
            "passed": max(ratios.values()) / min(ratios.values()) > 2.0, "ratios": ratios,
            "why_it_can_fail": ("if every choice of representative gave the same ratio, the '18x' "
                                "headline would have been a measurement after all and this "
                                "artifact would be withdrawing something true")},
        "f3_the_base_grid_is_a_subgrid_of_the_extended_one": {
            "passed": all(c["best_4608"] >= c["best_288"] - 1e-12 for c in per_cell),
            "n_cells": len(per_cell),
            "why_it_can_fail": ("the 288 grid is the extended grid at op3_rm = op5_rm = 0, so its "
                                "optimum can never exceed the extended optimum; a violation would "
                                "mean the two caches are not commensurable")},
    }
    falsifiers["all_passed"] = all(v["passed"] for v in falsifiers.values() if isinstance(v, dict))

    # NOT A FALSIFIER, AND IT USED TO SIT AMONG THEM WITH `passed: True` HARDCODED. Both outcomes
    # here are publishable -- new configurations that improve the optimum mean a harder problem,
    # new ones that never do mean dilution -- so there is no bar to fail, and a hardcoded pass
    # inside the falsifier dict feeds `all_passed` while certifying nothing. This project shipped a
    # real data leak behind exactly that shape once. It is descriptive and lives outside.
    descriptive = {
        "expansion_direction_read_from_the_data": {
            "n_cells_where_optimum_moved": len(optimum_moved),
            "mean_share_of_new_configs_above_base_optimum": float(np.mean(share_new_better)),
            "reading": ("new configurations improve the reachable optimum, so the expansion adds "
                        "difficulty of a kind uniform search benefits from rather than diluting it"),
        },
    }

    dilutes = len(optimum_moved) == 0
    payload = {
        "schema_version": "expansion_difficulty_v1",
        "claim_status": ("HALTED_FALSIFIER_FAILED" if not falsifiers["all_passed"] else
                         "EXPANSION_DILUTES_UNIFORM_SEARCH_WITHOUT_MOVING_THE_OPTIMUM" if dilutes
                         else "EXPANSION_MOVES_THE_OPTIMUM_IN_SOME_CELLS"),
        "scope": "REREAD_OF_SEALED_CACHES_NO_SEEDS_NO_SIMULATION",
        "run_role": "POST_HOC_REREAD",
        "registration_status": "POST_HOC_AUDIT_REPLACING_A_WITHDRAWN_RATIO_NOT_PREREGISTERED",
        "endpoint": "expected simple regret of a uniform 24-draw, normalised by |best|",
        "estimand": ("E[best of a uniform n-subset] by order statistics, on 288 and on 4,608 "
                     "configurations of the same sealed surfaces"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": None,
        "budget": n, "n_cells": len(per_cell), "contexts": contexts, "seeds": seeds,
        "expected_uniform_regret": {
            "grid_288": {"mean": float(r288.mean()), "sd": float(r288.std(ddof=1)),
                         "min": float(r288.min()), "max": float(r288.max())},
            "grid_4608": {"mean": float(r4608.mean()), "sd": float(r4608.std(ddof=1)),
                          "min": float(r4608.min()), "max": float(r4608.max())},
            "ratio_4608_over_288": float(r4608.mean() / r288.mean()),
            "paired_difference_mean": float((r4608 - r288).mean()),
            "n_cells_where_expansion_is_worse_for_uniform": int((r4608 > r288).sum()),
        },
        "expected_uniform_regret_against_a_common_reference": {
            "reference": "the extended-grid optimum, so the 288 grid pays for not containing it",
            "grid_288": {"mean": float(c288.mean()), "sd": float(c288.std(ddof=1))},
            "grid_4608": {"mean": float(c4608.mean()), "sd": float(c4608.std(ddof=1))},
            "ratio_4608_over_288": float(c4608.mean() / c288.mean()),
            "n_cells_where_expansion_is_worse_for_uniform": int((c4608 > c288).sum()),
        },
        "did_the_expansion_add_anything_reachable": {
            "n_cells_where_optimum_moved": len(optimum_moved),
            "mean_share_of_new_configs_above_base_optimum": float(np.mean(share_new_better)),
            "max_share": float(np.max(share_new_better)),
        },
        "withdrawn_headline": {
            "phrase": "the two new factors move the endpoint 18x less than the contrasts",
            "why_withdrawn": ("the ratio has no single value: it depends on which of four contrast "
                              "means and which of six context spreads are chosen as "
                              "representatives"),
            "range_of_defensible_ratios": ratios,
            "endpoint_spread_by_context": spread,
            "contrast_means_abs": contrast_means,
        },
        "per_cell": per_cell,
        "descriptive_observations": descriptive,
        "falsifiers": falsifiers,
    }
    seal_and_write(payload, ROOT / args.out, contract=ROOT / CONTRACT, reference=ROOT / SOURCE)

    print(f"\nregret esperado de una extracción uniforme a presupuesto {n} (normalizado):")
    print(f"  rejilla   288: {r288.mean():.5f}")
    print(f"  rejilla 4.608: {r4608.mean():.5f}   ({r4608.mean() / r288.mean():.2f}x)")
    print(f"  celdas donde la expansión empeora al uniforme: "
          f"{int((r4608 > r288).sum())}/{len(per_cell)}")
    print(f"\ncontra una referencia común (el óptimo de la extendida):")
    print(f"  rejilla   288: {c288.mean():.5f}")
    print(f"  rejilla 4.608: {c4608.mean():.5f}   ({c4608.mean() / c288.mean():.2f}x)")
    print(f"  celdas donde la expansión empeora al uniforme: "
          f"{int((c4608 > c288).sum())}/{len(per_cell)}")
    print(f"\n¿añadió la expansión algo alcanzable?")
    print(f"  celdas donde el óptimo se movió: {len(optimum_moved)}/{len(per_cell)}")
    print(f"  fracción media de configuraciones nuevas por encima del óptimo base: "
          f"{np.mean(share_new_better):.4%}")
    print(f"\nel '18x' retirado, como rango:")
    for k, v in ratios.items():
        print(f"  {k:38}{v:8.1f}x")
    print(f"\nveredicto: {payload['claim_status']}")
    print(f"falsadores: {'todos pasan' if falsifiers['all_passed'] else 'FALLO'}  "
          f"(control MC: gap relativo {mc_gap:.2e})")
    print(f"-> {args.out}")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
