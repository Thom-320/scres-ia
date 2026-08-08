#!/usr/bin/env python3
"""Six correlated looks at twelve tapes are not six adjudications. This closes RQ1's multiplicity.

WHAT `retention_contrasts` LEFT OPEN. It reports six marginal bootstrap intervals, one per family,
each computed from the SAME twelve seeds of `search_ladder_v5`. Four external audits independently
objected: the six contrasts are one inferential family, not six experiments, and reading "all six
exclude zero" off marginal intervals inflates the joint claim. The project already has the receipt
for why this matters -- `ofat_lcb_reconciliation` measured a bound whose SIGN depended on the
resampling seed, positive in only 65% of 40 seeds.

WHAT THIS ADDS, AND WHY EACH PIECE IS NOT OPTIONAL.

1. SHARED BOOTSTRAP INDICES. One (N_BOOT x n) index matrix is drawn once and used for all six
   families, so the six resampled means come from the same resampled seed-sets and their
   correlation is preserved. Six independently-drawn bootstraps would throw that correlation away
   and make any joint statement wrong in an unknown direction.

2. MAX-T SIMULTANEOUS INTERVALS. For each bootstrap replicate the studentized deviation is computed
   per family and the MAXIMUM over families is taken; its 95th percentile is the simultaneous
   critical value. This is the correct joint object, and it is strictly more conservative than 1.96.

3. HOLM over bootstrap two-sided p-values, reported alongside, because the audits asked for either
   and the two can disagree at n=12 -- which is itself worth knowing.

4. RESAMPLING-SEED SENSITIVITY over K independent seeds. At n=12 the percentile bootstrap has
   roughly 12 distinct order statistics in each tail; a bound near zero is not stable, and the only
   honest way to report it is to show how far it moves.

5. THE OTHER ENDPOINT. `search_ladder_v5` stores `per_arm[arm]["final"]` -- the simple regret of the
   recommendation actually deployed at budget 24 -- and no analysis in this repository has ever read
   it. The paper argues that only the final recommendation is deployed and then scores AUC; if the
   ordering under `final` disagrees with the ordering under `auc`, a reviewer will find it, and it
   should be us who finds it first.

THE GRADE DOES NOT IMPROVE. Same twelve burned tapes, same replay, no seed opened. Simultaneous
inference makes the statement honest; it does not make it prospective.

Contract: docs/ENMIENDA_REGISTRO_DE_EVIDENCIA_2026-08-07.md
Read-only over artifacts. No seed is opened.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402

SOURCE = Path("results/search_ladder_v5/result.json")
SEALED_MARGINALS = Path("results/retention_contrasts/result.json")
MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")

N_BOOT = 5_000
BOOT_SEED = 20260806          #: the seed `retention_contrasts` used; keeps f3 comparable
SENSITIVITY_SEEDS = tuple(range(20260806, 20260806 + 40))
ALPHA = 0.05

#: family -> (reset arm, retained arm). Orientation is reset minus retained: positive means
#: retention lowered regret. Identical to `run_retention_contrasts_v1.TWINS` by construction; f6
#: fails if the two scripts ever drift apart.
TWINS = {
    "neuron": ("neuron_reset", "neuron_memory"),
    "ucb1": ("ucb1", "ucb1_transfer"),
    "ofat": ("ofat", "ofat_transfer"),
    "lookahead_kg": ("lookahead_kg", "lookahead_kg_transfer"),
    "gp_ei": ("gp_ei", "gp_ei_transfer"),
    "thompson": ("thompson", "thompson_transfer"),
}


def paired(per_arm: dict, key: str) -> dict[str, np.ndarray]:
    """Delta_f per seed, for one endpoint column."""
    return {f: (np.asarray(per_arm[reset][key], float)
                - np.asarray(per_arm[ret][key], float))
            for f, (reset, ret) in TWINS.items()}


def simultaneous(diffs: dict[str, np.ndarray], rng: np.random.Generator) -> dict:
    """Marginal and max-T intervals from ONE shared index matrix."""
    fams = list(diffs)
    n = len(diffs[fams[0]])
    idx = rng.integers(0, n, size=(N_BOOT, n))          # <- drawn once, reused for every family

    boot = {f: diffs[f][idx].mean(axis=1) for f in fams}
    obs = {f: float(diffs[f].mean()) for f in fams}
    se = {f: float(boot[f].std(ddof=1)) for f in fams}

    # max-T: studentize each replicate, take the max over families, then its (1-alpha) quantile.
    t = np.stack([np.abs(boot[f] - obs[f]) / se[f] for f in fams])       # (n_fam, N_BOOT)
    c_simul = float(np.percentile(t.max(axis=0), 100 * (1 - ALPHA)))
    c_marg = float(np.percentile(np.abs(boot[fams[0]] - obs[fams[0]]) / se[fams[0]],
                                 100 * (1 - ALPHA)))

    out = {}
    for f in fams:
        # Two-sided bootstrap p-value, floored at the bootstrap's own resolution.
        p = 2.0 * min(float((boot[f] <= 0).mean()), float((boot[f] >= 0).mean()))
        out[f] = {
            "mean": obs[f], "se": se[f], "n": int(n),
            "marginal_lcb95": float(np.percentile(boot[f], 100 * ALPHA / 2)),
            "marginal_ucb95": float(np.percentile(boot[f], 100 * (1 - ALPHA / 2))),
            "simultaneous_lcb95": obs[f] - c_simul * se[f],
            "simultaneous_ucb95": obs[f] + c_simul * se[f],
            "p_boot": max(p, 1.0 / N_BOOT),
            "p_is_at_bootstrap_floor": p < 1.0 / N_BOOT,
        }
    return {"per_family": out, "c_simultaneous": c_simul, "c_marginal_reference": c_marg}


def holm(pvals: dict[str, float], alpha: float = ALPHA) -> dict:
    """Holm-Bonferroni. Returns per-family rejection and the adjusted threshold each faced."""
    order = sorted(pvals, key=lambda f: pvals[f])
    m, out, still = len(order), {}, True
    for i, f in enumerate(order):
        thr = alpha / (m - i)
        still = still and (pvals[f] <= thr)
        out[f] = {"p": pvals[f], "holm_threshold": thr, "rejected": bool(still), "rank": i + 1}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path,
                    default=Path("docs/ENMIENDA_REGISTRO_DE_EVIDENCIA_2026-08-07.md"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/retention_simultaneous/result.json"))
    args = ap.parse_args()

    src = json.loads((ROOT / SOURCE).read_text())
    per_arm, seeds = src["per_arm"], src["seeds"]

    lengths = {a: len(per_arm[a][k]) for pair in TWINS.values() for a in pair for k in ("auc", "final")}
    aligned = all(n == len(seeds) for n in lengths.values())

    results = {}
    if aligned:
        for endpoint in ("auc", "final"):
            d = paired(per_arm, endpoint)
            sim = simultaneous(d, np.random.default_rng(BOOT_SEED))
            sim["holm"] = holm({f: r["p_boot"] for f, r in sim["per_family"].items()})
            sim["ranking_best_first"] = sorted(d, key=lambda f: -float(d[f].mean()))
            results[endpoint] = sim

    # ---- resampling-seed sensitivity, on the primary endpoint ------------------------------
    sens = {}
    if aligned:
        d = paired(per_arm, "auc")
        per_seed = {f: [] for f in TWINS}
        for s in SENSITIVITY_SEEDS:
            r = simultaneous(d, np.random.default_rng(int(s)))
            for f, row in r["per_family"].items():
                per_seed[f].append(row["marginal_lcb95"])
        for f, vals in per_seed.items():
            v = np.asarray(vals)
            sens[f] = {"n_seeds": len(vals), "lcb_min": float(v.min()), "lcb_max": float(v.max()),
                       "lcb_median": float(np.median(v)),
                       "share_of_seeds_with_lcb_above_zero": float((v > 0).mean())}

    # ---- f5 control: destroy the pairing, Holm must stop rejecting -------------------------
    null_rejections = None
    if aligned:
        crng = np.random.default_rng(BOOT_SEED)
        shuffled = {}
        for f, (reset, ret) in TWINS.items():
            a, b = np.asarray(per_arm[reset]["auc"], float), np.asarray(per_arm[ret]["auc"], float)
            pool = np.concatenate([a, b])
            crng.shuffle(pool)
            shuffled[f] = pool[:len(a)] - pool[len(a):]
        cs = simultaneous(shuffled, np.random.default_rng(BOOT_SEED))
        ch = holm({f: r["p_boot"] for f, r in cs["per_family"].items()})
        null_rejections = int(sum(r["rejected"] for r in ch.values()))

    sealed = json.loads((ROOT / SEALED_MARGINALS).read_text()) if SEALED_MARGINALS.exists() else {}
    sealed_means = {f: r["mean"] for f, r in sealed.get("contrasts", {}).items()}
    mine_means = ({f: r["mean"] for f, r in results["auc"]["per_family"].items()}
                  if "auc" in results else {})

    falsifiers = {
        # Runs before any arithmetic, for the reason `retention_contrasts` learned the hard way: a
        # misaligned array raises inside the subtraction and the falsifier never gets to report.
        "f1_arrays_are_aligned_with_the_seed_list": {
            "passed": aligned, "lengths": lengths, "n_seeds": len(seeds)},
        "f2_simultaneous_is_stricter_than_marginal": {
            "passed": bool(results and results["auc"]["c_simultaneous"]
                           > results["auc"]["c_marginal_reference"]),
            "c_simultaneous": results.get("auc", {}).get("c_simultaneous"),
            "c_marginal_reference": results.get("auc", {}).get("c_marginal_reference"),
            "why_it_can_fail": ("if the max over families were not actually taken -- a per-family "
                                "loop that redraws indices -- the two criticals would coincide")},
        "f3_marginal_means_reproduce_the_sealed_artifact": {
            "passed": bool(sealed_means) and all(
                f in mine_means and float(sealed_means[f]) == float(mine_means[f])
                for f in sealed_means),
            "sealed": sealed_means, "recomputed": mine_means,
            "why_it_can_fail": "a different pairing, orientation or arm map gives different means"},
        "f4_the_resampling_seed_moves_the_bound": {
            "passed": bool(sens) and any(r["lcb_max"] > r["lcb_min"] for r in sens.values()),
            "spread_per_family": {f: r["lcb_max"] - r["lcb_min"] for f, r in sens.items()},
            "why_it_can_fail": ("if the seed argument were ignored every family would report an "
                                "identical bound across all 40 seeds")},
        "f5_holm_stops_rejecting_when_the_pairing_is_destroyed": {
            "passed": null_rejections is not None and null_rejections <= 1,
            "rejections_under_destroyed_pairing": null_rejections, "of": len(TWINS),
            "why_it_can_fail": ("if the procedure rejected under a null it would reject anything, "
                                "and every rejection above would be uninterpretable")},
        "f6_the_two_endpoints_are_different_numbers": {
            "passed": bool(results) and mine_means != {
                f: r["mean"] for f, r in results.get("final", {}).get("per_family", {}).items()},
            "why_it_can_fail": "reading the same column twice would silently fake an endpoint check"},
    }
    falsifiers["all_passed"] = all(v["passed"] for v in falsifiers.values() if isinstance(v, dict))

    n_simul = (sum(r["simultaneous_lcb95"] > 0 for r in results["auc"]["per_family"].values())
               if results else 0)
    n_holm = (sum(r["rejected"] for r in results["auc"]["holm"].values()) if results else 0)

    payload = {
        "schema_version": "retention_simultaneous_v1",
        "claim_status": (f"RETENTION_SURVIVES_SIMULTANEOUS_INFERENCE_IN_{n_simul}_OF_{len(TWINS)}"
                         if falsifiers["all_passed"] else "HALTED_FALSIFIER_FAILED"),
        "scope": "DEVELOPMENT_REANALYSIS_OF_A_SEALED_REPLAY_NO_SEEDS_NO_ADJUDICATION",
        "run_role": "REPLAY_REANALYSIS",
        "endpoint": "auc_regret_norm (primary) and final simple regret at budget 24 (secondary)",
        "estimand": ("paired per-seed AUC(reset) - AUC(retained) per family, with simultaneous "
                     "inference across the six families as one inferential family"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": None,
        "registration_status": "POST_HOC_MULTIPLICITY_REPAIR_REQUESTED_BY_REVIEW_NOT_PREREGISTERED",
        "supersedes_for_multiplicity": str(SEALED_MARGINALS),
        "supersession_note": ("the marginal intervals in that artifact are not retracted; they are "
                              "the per-family estimates. This file is what the manuscript cites for "
                              "any statement about the six families JOINTLY."),
        "source": str(SOURCE),
        "source_self_sha256": src.get("self_sha256"),
        "seeds": seeds,
        "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "alpha": ALPHA,
        "sensitivity_seeds": [int(s) for s in SENSITIVITY_SEEDS],
        "by_endpoint": results,
        "resampling_seed_sensitivity_auc": sens,
        "summary": {
            "n_families_simultaneous_lcb_above_zero": int(n_simul),
            "n_families_rejected_under_holm": int(n_holm),
            "endpoint_ranking_agrees": (results.get("auc", {}).get("ranking_best_first")
                                        == results.get("final", {}).get("ranking_best_first")),
        },
        "what_this_does_not_say": [
            "nothing here is prospective; the twelve tapes are burned and were re-read",
            "a family that survives simultaneous inference is not thereby causally attributed to "
            "retention -- the twins differ only in retention by construction, but the estimand is "
            "still an average over six contexts collapsed inside each seed",
            "the `final` endpoint is reported for robustness; the preregistered primary is AUC",
        ],
        "falsifiers": falsifiers,
    }
    seal_and_write(payload, ROOT / args.out, contract=ROOT / args.contract,
                   reference=ROOT / SOURCE)

    print(f"\ncritico simultaneo max-T: {results['auc']['c_simultaneous']:.4f} "
          f"(marginal de referencia {results['auc']['c_marginal_reference']:.4f})\n"
          if results else "sin resultados\n")
    print(f"{'familia':14}{'media':>12}{'IC marginal':>26}{'IC simultaneo':>26}"
          f"{'Holm':>7}{'estab.':>8}")
    for f, r in sorted(results.get("auc", {}).get("per_family", {}).items(),
                       key=lambda kv: -kv[1]["mean"]):
        h = results["auc"]["holm"][f]
        st = sens[f]["share_of_seeds_with_lcb_above_zero"]
        print(f"  {f:12}{r['mean']:+12.5f}"
              f"  [{r['marginal_lcb95']:+.5f}, {r['marginal_ucb95']:+.5f}]"
              f"  [{r['simultaneous_lcb95']:+.5f}, {r['simultaneous_ucb95']:+.5f}]"
              f"{'si' if h['rejected'] else 'NO':>7}{st:>8.0%}")
    print(f"\nfalsadores: {'todos pasan' if falsifiers['all_passed'] else 'FALLO'}"
          f"  ·  rechazos bajo emparejamiento destruido: {null_rejections}/{len(TWINS)}")
    print(f"ranking AUC   : {results.get('auc', {}).get('ranking_best_first')}")
    print(f"ranking final : {results.get('final', {}).get('ranking_best_first')}")
    print(f"-> {args.out}")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
