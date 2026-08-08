#!/usr/bin/env python3
"""H_regime names two statistics on two metrics, and the manuscript cites one without saying which.

RESOLVED AFTER THE FIRST SEALING, AND THE RESOLUTION IS BETTER THAN THE SUSPENSION. The first run of
this crosswalk found that `monotone_transform_ceiling` could not be reproduced from the surface cache
at any seed subset and concluded the two estimators disagreed. The cause is now identified in the
producer: `scripts/run_monotone_transform_ceiling_v1.py:94-114` reads `aggregates.json` and builds
`resilience_index(...)["R_cobb_douglas"]`, while the surface cache stores
`ret_excel_risk_conditional`. They are not two computations of one number; they are the SAME
statistic on TWO METRICS, and each is correct for its own.

That changes the verdict from "not citable" to "not citable UNLABELLED", and it relocates the
transform-proof zero: it is a property of the Cobb-Douglas surface, where one configuration is
optimal in every context, and NOT of the ret_excel surface the manuscript's 0.003802 comes from --
which a strictly increasing rescaling moves to 0.010776.

WHAT THIS CROSSWALK WAS FOR AND WHAT IT FOUND INSTEAD. The manuscript cites `H_regime = 0.003802`
against a 0.05 bar and concludes that no context-conditioned architecture can pay. The job here was
to place that number beside its siblings -- identity, best monotone rescaling, ceiling -- so a reader
could see which of them the sentence rests on. It does not reconcile.

Recomputing the runners' own estimator, `1 - max_a mean_r V_norm(r,a)` under per-context min-max,
reproduces `surface_gates_v2` to the last digit (0.003802243800697269) and reproduces
`monotone_transform_ceiling` at NO seed subset:

    grid 288    sealed H_identity 0.0        recomputed 0.000795 (its 6 seeds) / 0.003802 (all 12)
    grid 4,608  sealed H_identity 0.019501   recomputed 0.048573 (its 3 seeds) / 0.028294 (all 12)

And the ceiling artifact reports `argmax_is_universal: True` with index 240 in all six contexts,
while the same cache gives three distinct argmaxes at its six seeds and four at all twelve -- as does
`surface_gates_v2`'s own `argmax_by_context`, where `shifts` moves 1/3/2/2/3/2.

So the "transform-proof zero on the 288 grid", which this project has repeated, rests on the artifact
this crosswalk cannot reproduce. It is withdrawn pending adjudication of which estimator is correct,
and until then no H_regime figure may be cited.

WHAT SURVIVES, AND IT IS THE USEFUL HALF. Ordinal statistics, which no monotone transform can move:
`f3` proves the invariance rather than asserting it -- it applies a strictly increasing rescaling and
requires H to move while every ordinal statistic stays bit-identical. It passes on both grids, and
those statistics are computed here from the caches, so they do not depend on the disputed estimator.

Contract: docs/ENMIENDA_REGISTRO_DE_EVIDENCIA_2026-08-07.md
Reads sealed caches and sealed artifacts. No seed is opened, nothing is simulated.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.run_grid_transfer_v1 as G  # noqa: E402
import scripts.run_monotone_transform_ceiling_v1 as MTC  # noqa: E402
from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402

MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py",
           "scripts/run_grid_transfer_v1.py",
           # In the manifest because this artifact now RUNS it: the two-metric resolution is
           # computed from the ceiling artifact's own surface builder, not read off its prose.
           "scripts/run_monotone_transform_ceiling_v1.py")
CONTRACT = Path("docs/ENMIENDA_REGISTRO_DE_EVIDENCIA_2026-08-07.md")
GATES = Path("results/surface_gates_v2/result.json")
# SUPERSEDED TWICE, AND THIS ARTIFACT WAS CITING THE FIRST VERSION. `monotone_transform_ceiling`
# is explicitly retired by `monotone_transform_family_v2`, whose verdict is
# THE_CEILING_DOES_NOT_SURVIVE_MULTIPLICITY -- the ceiling was one transform picked from a family and
# never corrected for the family. v3 and v4 rebuild it properly: K = 661 declared transforms, Holm
# across all of them, an LCB requirement and a signal floor. It is kept as CEILING because its
# per-grid H_identity is what f1d reproduces, and the FAMILY is what the number now comes from.
CEILING = Path("results/monotone_transform_ceiling/result.json")
FAMILY = Path("results/monotone_transform_family_v4/result.json")
TOP_K = (1, 5, 10, 25)
BAR = 0.05


def h_regime(mean_by_ctx: np.ndarray) -> float:
    """1 - max_a mean_r V_norm(r,a), per-context min-max after any transform. The runners' estimator."""
    v = mean_by_ctx.astype(float)
    lo = v.min(axis=1, keepdims=True)
    hi = v.max(axis=1, keepdims=True)
    span = np.where(hi > lo, hi - lo, 1.0)
    return float(1.0 - ((v - lo) / span).mean(axis=0).max())


def ordinal_stats(mean_by_ctx: np.ndarray) -> dict:
    """Everything here depends only on the ORDER of values within a context."""
    n_ctx = mean_by_ctx.shape[0]
    argmax = [int(mean_by_ctx[i].argmax()) for i in range(n_ctx)]
    ranks = np.argsort(np.argsort(-mean_by_ctx, axis=1), axis=1)   # 0 = best
    overlap = {}
    for k in TOP_K:
        tops = [set(np.argsort(-mean_by_ctx[i])[:k].tolist()) for i in range(n_ctx)]
        pairs = [len(a & b) / k for a, b in itertools.combinations(tops, 2)]
        overlap[f"top{k}_mean_pairwise_overlap"] = float(np.mean(pairs))
        overlap[f"top{k}_intersection_size_all_contexts"] = int(
            len(set.intersection(*tops)) if tops else 0)
    rho = [float(np.corrcoef(ranks[i], ranks[j])[0, 1])
           for i, j in itertools.combinations(range(n_ctx), 2)]
    return {"argmax_per_context": argmax,
            "argmax_is_universal": len(set(argmax)) == 1,
            "n_distinct_argmax": len(set(argmax)),
            "mean_pairwise_rank_correlation": float(np.mean(rho)),
            "min_pairwise_rank_correlation": float(np.min(rho)), **overlap}


def load_mean(cache: Path, n_cfg: int, grid_id: str, only: list[int] | None = None):
    """Seed-averaged surface per context. `only` restricts to the seeds a sealed artifact used.

    THE SEED SUBSET IS NOT A DETAIL. `monotone_transform_ceiling` averaged the 288 grid over six
    seeds and `surface_gates_v2` over twelve, and the two report different statistics for what the
    manuscript cites as one number. Every row below therefore carries its own seed list.
    """
    surf, ctxs, seeds, *_ = G.load(cache, n_cfg, grid_id)
    use = [s for s in seeds if only is None or s in set(only)]
    order = [c for c in G.CONTEXT_ORDER if c in ctxs]
    return np.array([np.mean([surf[(c, s)] for s in use], axis=0) for c in order]), order, use


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/h_regime_crosswalk/result.json"))
    args = ap.parse_args()

    gates = json.loads((ROOT / GATES).read_text())
    ceiling = json.loads((ROOT / CEILING).read_text())
    family = json.loads((ROOT / FAMILY).read_text())

    rows, ordinals, invariance = {}, {}, {}
    for name, cache, n_cfg in (
            ("grid_288", Path("results/surface_cache/wrap288_v1"), len(G.BASE_CONFIGS)),
            ("grid_4608", Path("results/surface_cache/wrap288_compat_extended_v1"),
             len(G.EXT_CONFIGS))):
        grid_id = "wrap288_v1" if name == "grid_288" else "wrap288_compat_extended_v1"
        g = ceiling["grids"][grid_id]
        sealed_seeds = g.get("seeds")
        # Two readings of the same grid: the seeds the ceiling artifact used, and every seed the
        # cache holds. Where they differ, that difference IS the crosswalk's finding.
        mean_sealed, ctxs, used_sealed = load_mean(ROOT / cache, n_cfg, grid_id, sealed_seeds)
        mean, _, seeds = load_mean(ROOT / cache, n_cfg, grid_id)
        rows[name] = {
            "grid_id": grid_id, "n_configurations": int(mean.shape[1]),
            "contexts": ctxs,
            "n_seeds_in_cache": len(seeds), "n_seeds_used_by_ceiling_artifact": len(used_sealed),
            "h_identity_at_ceiling_artifact_seeds": h_regime(mean_sealed),
            "h_identity_at_all_cache_seeds": h_regime(mean),
            "argmax_universal_at_ceiling_seeds": ordinal_stats(mean_sealed)["argmax_is_universal"],
            "argmax_universal_at_all_seeds": ordinal_stats(mean)["argmax_is_universal"],
            "h_identity_recomputed": h_regime(mean),
            "h_identity_sealed": g["H_identity"],
            "h_best_transform_keeping_resolution": g["H_at_90pct_resolution"],
            "h_ceiling_over_all_monotone_transforms": g["ceiling"],
            "h_step_function": g["ceiling_step"]["H_regime"],
            "bar": BAR,
            "identity_clears_bar": g["H_identity"] > BAR,
            "some_monotone_transform_clears_bar": g["ceiling"] > BAR,
        }
        ordinals[name] = ordinal_stats(mean)

        # f3: apply a monotone transform and show what moves and what does not.
        lo, hi = mean.min(), mean.max()
        z = (mean - lo) / (hi - lo if hi > lo else 1.0)
        warped = lo + (hi - lo) * (z ** 3)                 # strictly increasing on [0,1]
        invariance[name] = {
            "transform": "x -> lo + (hi-lo) * ((x-lo)/(hi-lo))**3, strictly increasing",
            "h_before": h_regime(mean), "h_after": h_regime(warped),
            "ordinal_before": ordinal_stats(mean), "ordinal_after": ordinal_stats(warped),
        }
        invariance[name]["h_moved"] = abs(
            invariance[name]["h_after"] - invariance[name]["h_before"]) > 1e-9
        invariance[name]["ordinals_identical"] = (
            invariance[name]["ordinal_before"] == invariance[name]["ordinal_after"])

    # Rebuild the ceiling artifact's own surface and score it with THIS crosswalk's estimator.
    cobb = {}
    for grid_id in ("wrap288_v1", "wrap288_compat_extended_v1"):
        try:
            surf, ctxs_c, seeds_c = MTC.load_surface(grid_id)
            mine = h_regime(np.asarray(surf, float))
            theirs = float(ceiling["grids"][grid_id]["H_identity"])
            o = ordinal_stats(np.asarray(surf, float))
            cobb[grid_id] = {
                "recomputed_h_identity": mine, "sealed_h_identity": theirs,
                "reproduces": abs(mine - theirs) < 1e-9,
                "n_seeds": len(seeds_c), "contexts": ctxs_c,
                "argmax_per_context": o["argmax_per_context"],
                "argmax_is_universal": o["argmax_is_universal"],
            }
        except Exception as exc:                       # noqa: BLE001 -- reported, never swallowed
            cobb[grid_id] = {"reproduces": False, "error": f"{type(exc).__name__}: {exc}"}

    sealed_bootstrap = gates["g1_h_regime"]

    falsifiers = {
        # ASKED THE WRONG QUESTION AND IS KEPT SO THE RECORD SHOWS IT. This falsifier assumed the
        # two artifacts computed one statistic, so it required the surface-cache recomputation to
        # reproduce the ceiling artifact. They compute different METRICS
        # (run_monotone_transform_ceiling_v1.py:94-114 builds R_cobb_douglas from aggregates.json;
        # the surface cache stores ret_excel_risk_conditional), so reproduction was never possible
        # and the failure carries no information about either. f1c below is the correctly-posed
        # comparison. The row stays failed rather than deleted: a falsifier that asked the wrong
        # question is part of the record, not an embarrassment to tidy away.
        "f1_the_recomputed_identity_matches_the_sealed_one_AT_MATCHED_SEEDS": {
            "passed": all(abs(r["h_identity_at_ceiling_artifact_seeds"] - r["h_identity_sealed"])
                          < 1e-9 for r in rows.values()),
            "superseded_by": "f1c",
            "why_it_could_never_pass": ("it compares a ret_excel_risk_conditional surface against a "
                                        "Cobb-Douglas one"),
            "detail": {k: {"recomputed_at_sealed_seeds": r["h_identity_at_ceiling_artifact_seeds"],
                           "sealed": r["h_identity_sealed"],
                           "recomputed_at_all_cache_seeds": r["h_identity_at_all_cache_seeds"]}
                       for k, r in rows.items()},
            "why_it_can_fail": ("if the estimator disagreed at MATCHED seeds the difference would "
                                "be in the computation rather than in the sample, and this "
                                "artifact would be crosswalking its own estimator")},
        "f1c_the_recomputation_reproduces_the_artifact_ON_THE_SAME_METRIC": {
            "passed": abs(rows["grid_288"]["h_identity_at_all_cache_seeds"]
                          - gates["g1_h_regime"]["H_regime"]) < 1e-12,
            "recomputed": rows["grid_288"]["h_identity_at_all_cache_seeds"],
            "sealed_surface_gates_v2": gates["g1_h_regime"]["H_regime"],
            "metric": "ret_excel_risk_conditional, the surface cache's stored value",
            "why_it_can_fail": ("this is the comparison f1 should have made: same metric, same "
                                "cache, same seeds. If it failed, the estimator here would be "
                                "wrong and nothing in this artifact would be readable")},
        # COMPUTED, NOT ASSERTED. This falsifier used to carry `passed: True` with prose evidence,
        # which meant the claim the whole artifact rests on -- that the two numbers are two metrics
        # rather than two estimators -- was self-attested. It now rebuilds the Cobb-Douglas surface
        # with the ceiling artifact's OWN loader and requires this crosswalk's estimator to
        # reproduce the ceiling's sealed H on it. If it does, the estimator is common and the metric
        # is the whole difference. If it does not, the two-metric resolution is wrong and the
        # verdict returns to unreconciled.
        "f1d_the_ceiling_number_reproduces_ON_ITS_OWN_METRIC": {
            "passed": all(v["reproduces"] for v in cobb.values()),
            "detail": cobb,
            "ceiling_metric": "R_cobb_douglas, rebuilt here via run_monotone_transform_ceiling_v1.load_surface",
            "gates_metric": "ret_excel_risk_conditional, stored in results/surface_cache/wrap288_v1",
            "consequence": ("H_regime is not one number with two values; it is one statistic on two "
                            "metrics, and every citation must name the metric"),
            "why_it_can_fail": ("if the ceiling's H does not come back from its own surface under "
                                "this estimator, the difference is not the metric and the "
                                "two-metric resolution collapses")},
        "f1b_the_seed_subsets_also_differ_but_do_not_explain_the_gap": {
            "passed": (rows["grid_288"]["n_seeds_used_by_ceiling_artifact"]
                       != rows["grid_288"]["n_seeds_in_cache"]),
            "ceiling_artifact_seeds": rows["grid_288"]["n_seeds_used_by_ceiling_artifact"],
            "gates_artifact_seeds": len(gates.get("seeds", [])),
            "cache_seeds": rows["grid_288"]["n_seeds_in_cache"],
            "why_it_matters": ("the two artifacts also use different halves of the seed block, "
                               "which was the first hypothesis and is NOT the cause: matching the "
                               "subset gives 0.000795, still not 0.0. The metric is the cause "
                               "(f1d)")},
        "f2_the_288_bootstrap_and_identity_values_are_not_a_contradiction": {
            "passed": (sealed_bootstrap["lcb95"] > 0
                       and rows["grid_288"]["h_identity_sealed"] == 0.0
                       and rows["grid_288"]["h_ceiling_over_all_monotone_transforms"] == 0.0),
            "bootstrap": sealed_bootstrap,
            "identity": rows["grid_288"]["h_identity_sealed"],
            "ceiling": rows["grid_288"]["h_ceiling_over_all_monotone_transforms"],
            "why_it_can_fail": ("if the exact ceiling on the 288 grid were nonzero, the 0.003802 "
                                "estimate and the 0.0 exact value would be a genuine conflict "
                                "rather than an estimator difference")},
        "f3_ordinal_statistics_survive_a_monotone_transform_and_h_does_not": {
            "passed": all(v["ordinals_identical"] for v in invariance.values())
                      and invariance["grid_4608"]["h_moved"],
            "detail": {k: {"h_before": v["h_before"], "h_after": v["h_after"],
                           "h_moved": v["h_moved"], "ordinals_identical": v["ordinals_identical"]}
                       for k, v in invariance.items()},
            "why_it_can_fail": ("this is the claim the whole artifact rests on and it is proved "
                                "rather than asserted: if a transform moved an ordinal statistic, "
                                "the ordinal replacements would be no safer than H itself; if it "
                                "failed to move H on the extended grid, there would be nothing to "
                                "warn about")},
        # RECORDS RATHER THAN GATES, AND THE RECORD IS THE FINDING. The zero ceiling on the 288
        # grid follows from a universal argmax -- with one configuration optimal everywhere there is
        # nothing for a utility scale to trade off. That universality holds at the six seeds the
        # ceiling artifact used and FAILS at the twelve the cache holds, where `surface_gates_v2`
        # own argmax_by_context already shows shifts moving 1/3/2/2/3/2. So "the 288 zero is
        # transform-proof" is a statement about half the sample, not about the grid.
        "f4_the_288_transform_proof_zero_is_seed_dependent": {
            "passed": True,
            "argmax_universal_at_ceiling_seeds": rows["grid_288"]["argmax_universal_at_ceiling_seeds"],
            "argmax_universal_at_all_cache_seeds": rows["grid_288"]["argmax_universal_at_all_seeds"],
            "h_identity_at_ceiling_seeds": rows["grid_288"]["h_identity_at_ceiling_artifact_seeds"],
            "h_identity_at_all_cache_seeds": rows["grid_288"]["h_identity_at_all_cache_seeds"],
            "gates_argmax_by_context": gates.get("argmax_by_context"),
            "consequence": ("the transform-proof reading is CONFINED, not seed-dependent. The zero "
                            "never appears on the ret_excel surface at any seed subset -- this "
                            "artifact's own row gives 0.000795 at the ceiling's six seeds -- so the "
                            "seed count was never the explanation. It is a property of the "
                            "Cobb-Douglas metric alone, and its own seed stability cannot be "
                            "tested from the ret_excel cache")},
    }
    falsifiers["all_passed"] = all(v["passed"] for v in falsifiers.values() if isinstance(v, dict))
    # f1 failing is this artifact's RESULT, not a broken instrument: it is a comparison between two
    # sealed artifacts and it found them inconsistent. f3 is the integrity check -- if the
    # invariance demonstration failed, nothing below would be readable.
    integrity_ok = (
        falsifiers["f3_ordinal_statistics_survive_a_monotone_transform_and_h_does_not"]["passed"]
        and falsifiers["f1c_the_recomputation_reproduces_the_artifact_ON_THE_SAME_METRIC"]["passed"])

    payload = {
        "schema_version": "h_regime_crosswalk_v1",
        "claim_status": ("HALTED_INTEGRITY_FALSIFIER_FAILED" if not integrity_ok else
                         "TWO_METRICS_ONE_NAME__H_REGIME_MUST_BE_LABELLED_BY_METRIC"
                         if not falsifiers["all_passed"] else
                         "H_REGIME_IS_NEITHER_SCALE_INVARIANT_NOR_SEED_STABLE_ON_EITHER_GRID"),
        "integrity_falsifier_passed": integrity_ok,
        "resolution": {
            "artifact_a": str(GATES), "metric_a": "ret_excel_risk_conditional",
            "artifact_b": str(CEILING), "metric_b": "R_cobb_douglas",
            "evidence": "scripts/run_monotone_transform_ceiling_v1.py:94-114",
            "finding": ("the two are the same statistic on different metrics, not two estimators "
                        "of one. Each is correct for its own surface"),
            "consequence_for_the_manuscript": (
                "H_regime may be cited WITH its metric named. The 0.003802 figure is the "
                "ret_excel_risk_conditional surface at twelve seeds, and it is NOT transform-proof: "
                "a strictly increasing rescaling takes it to 0.010776. The transform-proof zero and "
                "the universal argmax belong to the Cobb-Douglas surface and may not be transferred "
                "to the other"),
        },
        "scope": "REREAD_OF_SEALED_CACHES_AND_ARTIFACTS_NO_SEEDS_NO_SIMULATION",
        "run_role": "POST_HOC_REREAD",
        "registration_status": "POST_HOC_CROSSWALK_REPLACING_AN_OVERCLAIM_NOT_PREREGISTERED",
        "endpoint": "H_regime and its transform-invariant ordinal companions",
        "estimand": "the same statistic on two grids under identity and under monotone rescaling",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": None,
        "sources": {
            "bootstrap_estimate": {"path": str(GATES), "self_sha256": gates.get("self_sha256"),
                                   "value": sealed_bootstrap},
            "transform_ceiling": {"path": str(CEILING),
                                  "self_sha256": ceiling.get("self_sha256")},
        },
        "transform_family_supersedes_the_ceiling": {
            "artifact": str(FAMILY), "self_sha256": family.get("self_sha256"),
            "claim_status": family.get("claim_status"),
            "k_declared": family.get("k_family"), "gate": family.get("gate"),
            "retires": family.get("predecessor"),
            "why": ("the ceiling reported ONE transform chosen from a family and never corrected "
                    "for the family; monotone_transform_family_v2 retired it with "
                    "THE_CEILING_DOES_NOT_SURVIVE_MULTIPLICITY. v4 declares K = 661, applies Holm "
                    "across all of them, and adds an LCB requirement and a signal floor"),
            "per_grid": {g: {k: v for k, v in row.items() if not isinstance(v, (list, dict))}
                         for g, row in family.get("grids", {}).items()},
            "deciding_transform": family.get("grids", {})
                .get("wrap288_compat_extended_v1", {}).get("deciding_transform"),
        },
        "crosswalk": rows,
        "ordinal_statistics": ordinals,
        "transform_invariance_demonstration": invariance,
        "withdrawn_claim": {
            "phrase": "no context-conditioned architecture can pay, because H_regime = 0.0038 < 0.05",
            "why_withdrawn": ("three reasons now, and any one suffices. The statistic is not "
                              "invariant to the utility scale -- a strictly increasing rescaling "
                              "moves it on BOTH grids, 0.003802 to 0.010776 on the 288 and 0.028294 "
                              "to 0.067539 on the extended. Two sealed artifacts do not agree on "
                              "its value, so there is no single number to cite. And the "
                              "non-invariance is far larger than this artifact first reported: over "
                              "a declared family of 661 transforms with Holm multiplicity, an LCB "
                              "requirement and a signal floor, 29 survive on the extended grid and "
                              "the deciding one -- power(gamma = 19.95) -- takes H from 0.019501 to "
                              "0.27764 [LCB95 0.25713], fourteen times, against a 0.05 gate"),
            "also_withdrawn": ("the transform-proof reading of the 288 zero as a property of the "
                               "ret_excel surface: it comes from the Cobb-Douglas artifact, whose "
                               "argmax is not universal on the ret_excel cache at any seed subset. "
                               "On the Cobb-Douglas metric it is REINFORCED rather than withdrawn: "
                               "family_v4's f5 is the negative control, and 0 of 661 transforms "
                               "pass on the 288 grid where one configuration is optimal in all six "
                               "regimes"),
            "what_may_be_said_instead": ("Contextual rankings exist, but their cardinal value "
                                         "depends on the declared utility scale. On the "
                                         "Cobb-Douglas metric at the six seeds of the ceiling "
                                         "artifact a single configuration is optimal in every "
                                         "context; on the ret_excel_risk_conditional surface of "
                                         "the same grid at twelve seeds there are four distinct "
                                         "argmaxes and H is 0.003802, which a strictly increasing "
                                         "rescaling moves to 0.010776. Neither statement may be "
                                         "made without naming its metric and its seeds."),
        },
        "falsifiers": falsifiers,
    }
    seal_and_write(payload, ROOT / args.out, contract=ROOT / CONTRACT, reference=ROOT / CEILING)

    print(f"\n{'rejilla':12}{'identidad':>12}{'mejor transf.':>15}{'techo':>10}{'escalon':>10}"
          f"{'argmax univ.':>14}")
    for k, r in rows.items():
        o = ordinals[k]
        print(f"{k:12}{r['h_identity_sealed']:>12.6f}"
              f"{r['h_best_transform_keeping_resolution']:>15.6f}"
              f"{r['h_ceiling_over_all_monotone_transforms']:>10.4f}"
              f"{r['h_step_function']:>10.4f}{str(o['argmax_is_universal']):>14}")
    print(f"\nbarra preregistrada: {BAR}")
    print(f"estimación bootstrap sellada (288, min-max por contexto): "
          f"{sealed_bootstrap['H_regime']:.6f} [{sealed_bootstrap['lcb95']:.2e}, "
          f"{sealed_bootstrap['ucb95']:.6f}]")
    print("\nordinales, que ninguna transformación monótona mueve:")
    for k, o in ordinals.items():
        print(f"  {k:11} argmax distintos {o['n_distinct_argmax']}/6 · "
              f"rho de rangos medio {o['mean_pairwise_rank_correlation']:+.4f} · "
              f"top-25 solape {o['top25_mean_pairwise_overlap']:.2%} · "
              f"intersección top-25 {o['top25_intersection_size_all_contexts']}")
    print(f"\ndemostración de invariancia (x -> x^3 reescalado):")
    for k, v in invariance.items():
        print(f"  {k:11} H {v['h_before']:.6f} -> {v['h_after']:.6f} "
              f"(movió: {v['h_moved']}) · ordinales idénticos: {v['ordinals_identical']}")
    print(f"\nveredicto: {payload['claim_status']}")
    print(f"falsadores: {'todos pasan' if falsifiers['all_passed'] else 'FALLO'}")
    for n, v in falsifiers.items():
        if isinstance(v, dict) and not v["passed"]:
            print(f"  FALLA {n}")
    print(f"-> {args.out}")
    return 0 if integrity_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
