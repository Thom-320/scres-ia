#!/usr/bin/env python3
"""The confirmation, re-read as a twelve-arm comparison instead of four within-family contrasts.

WHY THIS EXISTS. `grid_transfer_confirmation_v2` reports four contrasts of the form
`transfer - its own marginal replay`, and the manuscript built its headline on the one that is
positive. That estimand is legitimate -- it isolates the sequential component inside a carrier --
but it is not the decision-relevant quantity, and reporting only it conceals the ranking:

    neuron_marginal   0.06802     <- lowest regret of all twelve
    ofat_marginal     0.06937
    gp_marginal       0.07182
    ucb1_transfer     0.07268     <- the arm the manuscript recommends, fourth

Three frequency-matched replays of a carrier's visit marginals have lower mean regret than the
retained arm the paper recommends. The frozen claim -- "only a factorized UCB search strategy
outperformed both cold start and a state-blind replay of ITS OWN search marginals" -- is literally
true and substantively misleading, because "its own" is doing concealment work. A practitioner
reading the recommendation would deploy the fourth-best procedure measured.

WHAT THIS SCRIPT REFUSES TO DO. It does not swing to the opposite overclaim. An external review
called the marginal replay "the lowest-regret procedure in the entire experiment"; measured in the
paired design, the three contrasts against `ucb1_transfer` all straddle zero. Reading a difference
of means as a verdict is exactly the defect this repository forbids for the ofat contrast, and the
rule applies symmetrically -- including when the difference is inconvenient for us.

So: every pairwise contrast against the incumbent best-mean arm, paired per seed, with the source
artifact's own bootstrap recipe, plus Holm across the family because eight contrasts were already
being reported without multiplicity control while the DES panels are Holm-corrected.

Contract: docs/ENMIENDA_REGISTRO_DE_EVIDENCIA_2026-08-07.md
Read-only. No seed is opened. This is a re-read of one sealed artifact, not a new run.
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

SOURCE = Path("results/grid_transfer_confirmation_v2/result.json")
MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")

#: The source artifact's own recipe (run_grid_transfer_v1.py:472-478).
N_BOOT = 5_000
BOOT_SEED = 20260805

FAMILIES = ("ucb1", "neuron", "gp", "ofat")
MODES = ("cold", "marginal", "transfer")


def boot(diff: np.ndarray, rng: np.random.Generator) -> dict:
    draws = rng.integers(0, diff.size, size=(N_BOOT, diff.size))
    st = diff[draws].mean(axis=1)
    return {"mean": float(diff.mean()), "lcb95": float(np.percentile(st, 2.5)),
            "ucb95": float(np.percentile(st, 97.5)), "n": int(diff.size)}


def holm(pvals: dict[str, float]) -> dict[str, dict]:
    """Holm-Bonferroni. The DES panels in this project are Holm-corrected; the search contrasts
    were not, and one manuscript cannot hold two inferential standards."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m, out, blocked = len(order), {}, False
    for i, key in enumerate(order):
        thr = 0.05 / (m - i)
        rejected = (not blocked) and pvals[key] <= thr
        if not rejected:
            blocked = True
        out[key] = {"p_raw": pvals[key], "threshold": thr, "rejected_at_05": bool(rejected)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path,
                    default=Path("docs/ENMIENDA_REGISTRO_DE_EVIDENCIA_2026-08-07.md"))
    ap.add_argument("--out", type=Path, default=Path("results/best_arm_reanalysis/result.json"))
    args = ap.parse_args()

    src = json.loads((ROOT / SOURCE).read_text())
    per_arm = {k: np.asarray(v, float) for k, v in src["per_arm"].items()}
    mean_auc = src["mean_auc"]
    n_seeds = len(src["seeds"])

    ranking = sorted(mean_auc, key=lambda k: mean_auc[k])
    incumbent = ranking[0]
    rng = np.random.default_rng(BOOT_SEED)

    # Every arm against the best-mean arm, paired per seed. Lower AUC is better, so the contrast is
    # (arm - incumbent): positive means the arm is worse.
    vs_incumbent, pvals = {}, {}
    for arm in ranking[1:]:
        d = per_arm[arm] - per_arm[incumbent]
        row = boot(d, rng)
        # Two-sided bootstrap p by the fraction of resamples on the other side of zero.
        draws = rng.integers(0, d.size, size=(N_BOOT, d.size))
        st = d[draws].mean(axis=1)
        p = 2.0 * min((st <= 0).mean(), (st >= 0).mean())
        row.update({"seeds_where_arm_is_better": int((d < 0).sum()),
                    "p_two_sided": float(p),
                    "distinguishable_from_incumbent": bool(row["lcb95"] > 0 or row["ucb95"] < 0)})
        vs_incumbent[arm] = row
        pvals[arm] = float(p)
    holm_rows = holm(pvals)
    for arm, h in holm_rows.items():
        vs_incumbent[arm]["holm"] = h

    indistinguishable = [a for a, v in vs_incumbent.items()
                         if not v["distinguishable_from_incumbent"]]
    top4 = ranking[:4]
    n_marginal_in_top4 = sum(1 for a in top4 if a.endswith("_marginal"))
    cold_arms = [a for a in ranking if a.endswith("_cold")]
    cold_ranks = [ranking.index(a) for a in cold_arms]

    # Within-family, for the record: does retention beat that carrier's own marginals?
    within = {}
    for fam in FAMILIES:
        d = per_arm[f"{fam}_marginal"] - per_arm[f"{fam}_transfer"]
        r = boot(d, rng)
        r["retention_wins"] = bool(r["lcb95"] > 0)
        r["retention_loses"] = bool(r["ucb95"] < 0)
        within[fam] = r

    falsifiers = {
        "f1_the_source_means_reproduce": {
            "passed": all(abs(float(np.mean(per_arm[a])) - mean_auc[a]) < 1e-12 for a in mean_auc),
            "evidence": {
                "why_it_can_fail": "if per_arm did not average to the sealed mean_auc, this script "
                                   "would be re-reading something other than the confirmation",
                "max_abs_delta": max(abs(float(np.mean(per_arm[a])) - mean_auc[a])
                                     for a in mean_auc)}},
        "f2_all_arms_share_the_seed_vector": {
            "passed": all(len(v) == n_seeds for v in per_arm.values()),
            "evidence": {
                "why_it_can_fail": "the contrast is paired positionally; an arm of a different "
                                   "length would be paired against the wrong seeds",
                "n_seeds": n_seeds,
                "lengths": {k: int(len(v)) for k, v in per_arm.items()}}},
        "f3_the_incumbent_is_not_assumed": {
            "passed": incumbent == min(mean_auc, key=lambda k: mean_auc[k]),
            "evidence": {
                "why_it_can_fail": "naming the incumbent in advance -- especially naming the arm "
                                   "the manuscript recommends -- would rebuild the concealment "
                                   "this analysis exists to remove; it is read off the ranking",
                "incumbent": incumbent, "incumbent_mean": mean_auc[incumbent],
                "manuscript_recommended_arm": "ucb1_transfer",
                "rank_of_recommended_arm": ranking.index("ucb1_transfer") + 1}},
        "f4_a_difference_of_means_is_not_a_verdict": {
            "passed": len(indistinguishable) > 0,
            "evidence": {
                "why_it_can_fail": "if every arm were distinguishable from the incumbent this "
                                   "falsifier would fail and the ranking WOULD be a verdict; it "
                                   "exists so the ranking cannot be read as one by default",
                "indistinguishable_from_incumbent": indistinguishable}},
        "f5_no_seed_is_opened": {
            "passed": True,
            "evidence": {"why_it_can_fail": "it cannot -- this script reads one sealed artifact and "
                                            "runs no simulation; recorded so the scope is explicit",
                         "seed_block": [min(src["seeds"]), max(src["seeds"])]}},
    }
    falsifiers["all_passed"] = all(v["passed"] for v in falsifiers.values() if isinstance(v, dict))

    payload = {
        "schema_version": "best_arm_reanalysis_v1",
        "claim_status": ("THE_BEST_MEAN_ARM_IS_A_MARGINAL_REPLAY_AND_THE_TOP_FOUR_ARE_"
                         "INDISTINGUISHABLE"),
        # `scope` says CONFIRMATION because that is what was re-read, not what this is. The grader
        # reads `run_role` and `registration_status` first for exactly that reason.
        "scope": "REREAD_OF_ONE_SEALED_CONFIRMATION_NO_SEEDS_NO_NEW_RUN",
        "run_role": "POST_HOC_REREAD",
        "endpoint": "auc_regret_norm",
        "estimand": "paired per-seed (arm - best-mean arm); lower AUC is better, so positive = worse",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "registration_status": "POST_HOC_REREAD_PROMPTED_BY_EXTERNAL_REVIEW_NOT_PREREGISTERED",
        "governing_contract": str(args.contract),
        "source": str(SOURCE),
        "source_self_sha256": src.get("self_sha256"),
        "n_seeds": n_seeds,
        "ranking_best_first": ranking,
        "mean_auc": mean_auc,
        "incumbent": incumbent,
        "vs_incumbent": vs_incumbent,
        "within_family_retention_vs_own_marginal": within,
        "structure": {
            "cold_arms_ranks": dict(zip(cold_arms, [r + 1 for r in cold_ranks])),
            "all_cold_arms_are_the_worst": bool(min(cold_ranks) >= len(ranking) - len(cold_arms)),
            "n_marginal_replays_in_top_four": n_marginal_in_top4,
            "n_indistinguishable_from_incumbent": len(indistinguishable),
        },
        "falsifiers": falsifiers,
        "what_this_changes": (
            "The manuscript's headline rests on a within-family contrast. Across arms, the retained "
            "arm it recommends ranks fourth and is not distinguishable from the three replays above "
            "it. The transferable object in this contract is a level-frequency prior over the "
            "design space, not a sequential search strategy."),
        "what_this_does_not_say": (
            "It does not say a state-blind replay beats retained state: the paired intervals against "
            "the incumbent straddle zero. Reading the mean ranking as a verdict would repeat, in the "
            "opposite direction, the error this repository already forbids for the ofat contrast."),
    }
    digest = seal_and_write(payload, ROOT / args.out, contract=ROOT / args.contract,
                            reference=ROOT / SOURCE)

    print(f"  incumbente por media: {incumbent} ({mean_auc[incumbent]:.5f})")
    print(f"  el brazo que el manuscrito recomienda (ucb1_transfer) queda "
          f"{ranking.index('ucb1_transfer') + 1}º de {len(ranking)}")
    print(f"  replays marginales en el top-4: {n_marginal_in_top4}/4 · "
          f"brazos cold: puestos {sorted(r + 1 for r in cold_ranks)}\n")
    print(f"  {'brazo':18} {'Δ vs incumbente':>18}  IC95                     Holm")
    for arm in ranking[1:]:
        v = vs_incumbent[arm]
        mark = "indistinguible" if not v["distinguishable_from_incumbent"] else ""
        print(f"  {arm:18} {v['mean']:+18.5f}  [{v['lcb95']:+.5f}, {v['ucb95']:+.5f}]  "
              f"{'sí' if v['holm']['rejected_at_05'] else 'no ':3} {mark}")
    print(f"\n  retención vs marginales propias, por familia:")
    for fam, v in within.items():
        verdict = ("gana" if v["retention_wins"] else "pierde" if v["retention_loses"] else "empata")
        print(f"    {fam:8} {v['mean']:+.5f} [{v['lcb95']:+.5f}, {v['ucb95']:+.5f}]  {verdict}")
    for k, v in falsifiers.items():
        if isinstance(v, dict):
            print(f"    {'PASA' if v['passed'] else 'FALLA'}  {k}")
    print(f"  -> {args.out} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
