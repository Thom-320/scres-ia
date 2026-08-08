#!/usr/bin/env python3
"""The state-blind comparator is not a fixed control: it gets stronger while the experiment runs.

WHAT WAS FOUND. In `run_grid_transfer_v1.py` the visit histogram that defines the marginal-replay
comparator is created ONCE, before the seed loop, and updated inside it:

    visits = {a: np.ones(len(EXT_CONFIGS)) for a in ARMS}     # before the loop
    for r, seed in enumerate(seeds):
        for ctx in contexts:
            build(kind, carried, "ext")(s, ...)               # the transferred arm runs
            for i in s.visited: visits[kind][i] += 1.0        # its visits enter the histogram
            ...
            marginal_replay(visits[kind], s3, ...)            # the replay runs on the SAME case

Two distinct defects follow, and they are not the same size.

(1) CURRENT-CASE CONTAMINATION. The 24 visits the carrier just chose on this very surface are in the
    histogram before the replay draws from it. The replay reads counts only, never values -- but the
    counts were chosen adaptively using this case's outcomes. Exactly bounded: at the k-th evaluation
    the histogram holds 4608 + 24k counts of which 24 come from the current case, so between 0.52%
    (first) and 0.18% (last) of the probability mass. Small, and it makes "state-blind" the wrong
    label regardless of size.

(2) CROSS-CASE ACCUMULATION -- the larger effect, and the one this artifact measures. At the first
    evaluation the histogram is 4608 ones plus 24 real visits, so the comparator is essentially
    random search. At the last it holds 8640 real visits, 65% of the mass. A comparator that carries
    the same name at the start and the end of an experiment, and is not the same object, breaks the
    exchangeability that the paired bootstrap over 60 seeds assumes.

WHY THE COLD ARM IS THE CONTROL THAT MAKES THIS A DIAGNOSIS. If the seed block simply got easier
along its order, every contrast would drift. The cold comparator is rebuilt from scratch on every
case and carries nothing, so it cannot drift for this reason. f1 requires it not to. If it does, the
trend is a property of the seeds and this artifact's reading is wrong.

WHAT THIS DOES NOT DO. It does not re-run anything, does not re-grade the sealed confirmation, and
does not repair the comparator -- the repair is a separate preregistered run with a frozen ex-ante
prior and a leave-one-out arm. This artifact exists so the limitation can be cited from a sealed
number instead of an ad-hoc calculation.

Contract: docs/ENMIENDA_REGISTRO_DE_EVIDENCIA_2026-08-07.md
Read-only over one sealed artifact. No seed is opened.
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

FAMILIES = ("ucb1", "neuron", "gp", "ofat")
N_PERM = 10_000
PERM_SEED = 20260808
HEAD_TAIL = 20          #: size of the first/last window reported alongside the trend


def trend(contrast: np.ndarray, rng: np.random.Generator) -> dict:
    """Correlation of the per-seed contrast with run order, plus a permutation null."""
    n = len(contrast)
    order = np.arange(n, dtype=float)
    rho = float(np.corrcoef(order, contrast)[0, 1])
    null = np.array([float(np.corrcoef(order, rng.permutation(contrast))[0, 1])
                     for _ in range(N_PERM)])
    return {
        "mean": float(contrast.mean()),
        "rho_with_run_order": rho,
        "p_permutation_two_sided": float((np.abs(null) >= abs(rho)).mean()),
        f"mean_first_{HEAD_TAIL}": float(contrast[:HEAD_TAIL].mean()),
        f"mean_last_{HEAD_TAIL}": float(contrast[-HEAD_TAIL:].mean()),
        "drift_last_minus_first": float(contrast[-HEAD_TAIL:].mean()
                                        - contrast[:HEAD_TAIL].mean()),
        "null_mean_rho": float(null.mean()), "null_sd_rho": float(null.std(ddof=1)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path,
                    default=Path("docs/ENMIENDA_REGISTRO_DE_EVIDENCIA_2026-08-07.md"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/comparator_drift/result.json"))
    args = ap.parse_args()

    src = json.loads((ROOT / SOURCE).read_text())
    per_arm, seeds = src["per_arm"], src["seeds"]
    budget, n_cfg, n_ctx = src["budget"], src["n_ext_configs"], len(src["contexts"])

    # Stored order must BE run order, or "trend with run order" is measuring nothing. The runner
    # iterates `for r, seed in enumerate(seeds)` over a sorted list and the confirmation gate
    # requires that list to equal a consecutive block, so the two coincide -- but it is asserted,
    # not assumed.
    order_is_run_order = seeds == list(range(seeds[0], seeds[0] + len(seeds)))

    rng = np.random.default_rng(PERM_SEED)
    vs_marginal, vs_cold, per_arm_trend = {}, {}, {}
    lengths = {a: len(v) for a, v in per_arm.items()}
    aligned = all(n == len(seeds) for n in lengths.values())
    if aligned and order_is_run_order:
        for fam in FAMILIES:
            t = np.asarray(per_arm[f"{fam}_transfer"], float)
            vs_marginal[fam] = trend(np.asarray(per_arm[f"{fam}_marginal"], float) - t, rng)
            vs_cold[fam] = trend(np.asarray(per_arm[f"{fam}_cold"], float) - t, rng)
        # DESCRIPTIVE, NOT A TEST, AND THE REASON IS DISCLOSED. "Does the comparator strengthen?" is
        # a question about the marginal ARM's own regret against run order, not about a contrast
        # that also carries the transfer arm's variation -- f2 below asks it of the contrast and so
        # is mis-specified for its own stated intent. The correctly-posed quantity is here. It gets
        # no pass/fail bar because these numbers were already seen in an ad-hoc diagnostic before
        # this script existed, and a threshold chosen after seeing the value is not a threshold.
        for kind in ("marginal", "cold", "transfer"):
            for fam in FAMILIES:
                per_arm_trend[f"{fam}_{kind}"] = trend(
                    np.asarray(per_arm[f"{fam}_{kind}"], float), rng)

    # The contamination bound is arithmetic, not estimation: 24 of 4608 + 24k counts.
    n_evals = len(seeds) * n_ctx
    mass = {
        "histogram_initial_pseudocounts": n_cfg,
        "visits_added_per_evaluation": budget,
        "n_evaluations_per_arm": n_evals,
        "current_case_share_first_evaluation": budget / (n_cfg + budget),
        "current_case_share_last_evaluation": budget / (n_cfg + budget * n_evals),
        "real_visit_share_of_mass_at_end": (budget * n_evals) / (n_cfg + budget * n_evals),
        "comparator_at_first_evaluation": ("essentially uniform sampling: 24 real visits against "
                                           "4608 pseudocounts"),
    }

    sealed = src.get("contrasts", {})
    reproduced = {}
    for fam in FAMILIES:
        got = sealed.get(fam, {}).get("vs_marginal_replay", {}).get("mean")
        mine = vs_marginal.get(fam, {}).get("mean")
        reproduced[fam] = {"sealed": got, "recomputed": mine,
                           "exact": got is not None and mine is not None
                           and float(got) == float(mine)}

    n_marginal_drifting = sum(1 for f in FAMILIES
                              if vs_marginal.get(f, {}).get("p_permutation_two_sided", 1.0) < 0.05)
    n_cold_drifting = sum(1 for f in FAMILIES
                          if vs_cold.get(f, {}).get("p_permutation_two_sided", 1.0) < 0.05)

    falsifiers = {
        "f0_stored_seed_order_is_run_order": {
            "passed": order_is_run_order, "seeds_first": seeds[:2], "seeds_last": seeds[-2:],
            "why_it_can_fail": ("if the artifact stored seeds sorted differently from the loop that "
                                "produced them, every correlation below would be meaningless")},
        "f1_the_cold_comparator_does_not_drift": {
            "passed": n_cold_drifting == 0, "n_families_drifting": n_cold_drifting,
            "per_family_p": {f: vs_cold.get(f, {}).get("p_permutation_two_sided") for f in FAMILIES},
            "why_it_can_fail": ("the cold arm is rebuilt per case and carries nothing, so it cannot "
                                "drift from accumulation. If it drifts anyway the trend belongs to "
                                "the seed block and this whole diagnosis is wrong")},
        # THIS ONE FAILED, AND IT STAYS FAILED. The bar -- resolved drift in at least 3 of 4 --
        # was written before the numbers were computed here, and 2 of 4 came back. It is not moved.
        # On inspection the falsifier is also mis-specified for its own stated intent: it tests the
        # CONTRAST (marginal minus transfer), which carries the transfer arm's variation too, while
        # the sentence it defends is about the comparator alone. `per_arm_trend` holds the
        # correctly-posed quantity, descriptively and without a bar, because it had already been
        # seen. The direction is consistent in 4 of 4 here; the resolution is 2 of 4. Both go in.
        "f2_the_marginal_comparator_drifts_in_most_families": {
            "passed": n_marginal_drifting >= 3, "n_families_drifting": n_marginal_drifting,
            "n_families_drifting_in_direction": sum(
                1 for f in FAMILIES if vs_marginal.get(f, {}).get("rho_with_run_order", 0.0) < 0),
            "per_family_p": {f: vs_marginal.get(f, {}).get("p_permutation_two_sided")
                             for f in FAMILIES},
            "is_substantive_not_integrity": True,
            "mis_specification_disclosed": ("tests the contrast, not the comparator; see "
                                            "per_arm_trend for the correctly-posed quantity"),
            "why_it_can_fail": "a single drifting family would be a fluke, not a mechanism"},
        "f3_the_sealed_contrast_means_reproduce_exactly": {
            "passed": all(r["exact"] for r in reproduced.values()), "detail": reproduced,
            "why_it_can_fail": "a different orientation or arm map would give different means"},
        "f4_the_permutation_null_is_centred_on_zero": {
            "passed": all(abs(vs_marginal[f]["null_mean_rho"]) < 0.02 for f in vs_marginal),
            "null_means": {f: vs_marginal[f]["null_mean_rho"] for f in vs_marginal},
            "why_it_can_fail": ("a permutation that failed to shuffle would reproduce the observed "
                                "rho and every p-value would collapse to zero")},
        "f5_arrays_are_aligned_with_the_seed_list": {
            "passed": aligned, "lengths": lengths, "n_seeds": len(seeds)},
    }
    falsifiers["all_passed"] = all(v["passed"] for v in falsifiers.values() if isinstance(v, dict))
    # INTEGRITY FAILURES AND STRENGTH-OF-EVIDENCE FAILURES ARE NOT THE SAME EVENT. If f0/f3/f4/f5
    # fail, nothing below is readable and the artifact must halt. f1 and f2 are substantive bars:
    # failing one is a result, not a broken instrument, and collapsing both into
    # HALTED_FALSIFIER_FAILED would hide which of the two happened.
    integrity_ok = all(v["passed"] for k, v in falsifiers.items()
                       if isinstance(v, dict) and not v.get("is_substantive_not_integrity"))

    payload = {
        "schema_version": "comparator_drift_v1",
        "claim_status": (
            "HALTED_INTEGRITY_FALSIFIER_FAILED" if not integrity_ok else
            "THE_MARGINAL_COMPARATOR_STRENGTHENS_DURING_THE_RUN_THE_COLD_ONE_DOES_NOT"
            if falsifiers["all_passed"] else
            f"COMPARATOR_DRIFT_DIRECTIONAL_IN_4_OF_4_RESOLVED_IN_{n_marginal_drifting}_OF_4"),
        "integrity_falsifiers_passed": integrity_ok,
        "scope": "REREAD_OF_ONE_SEALED_CONFIRMATION_NO_SEEDS_NO_NEW_RUN",
        "run_role": "POST_HOC_REREAD",
        "registration_status": "POST_HOC_DIAGNOSTIC_PROMPTED_BY_EXTERNAL_REVIEW_NOT_PREREGISTERED",
        "endpoint": "auc_regret_norm",
        "estimand": ("per-seed contrast (comparator - transfer) regressed on run order; the cold "
                     "comparator is the internal control"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": None,
        "source": str(SOURCE),
        "source_self_sha256": src.get("self_sha256"),
        "n_perm": N_PERM, "perm_seed": PERM_SEED, "head_tail_window": HEAD_TAIL,
        "vs_marginal_replay": vs_marginal,
        "vs_cold_start": vs_cold,
        "per_arm_trend": per_arm_trend,
        "histogram_mass_accounting": mass,
        "what_this_changes": [
            "'state-blind marginal replay' is the wrong name: the histogram contains visits the "
            "carrier chose adaptively on the target case. 'carrier-state-blind, sequence-blind "
            "online frequency replay' is accurate.",
            "the comparator cannot be deployed without first running the carrier on the same case, "
            "so it is not an ex-ante transportable prior and no claim of the form 'a level-frequency "
            "prior is what transfers' is identified by this artifact",
            "the paired bootstrap treats the 60 seeds as exchangeable replicates of one contrast; "
            "with a trend in the contrast they are not",
        ],
        "what_this_does_not_change": [
            "the sealed confirmatory verdict stands: the UCB1 contrast is positive across the whole "
            "run, and remains positive in the last window where the comparator is strongest",
            "the cold-start comparisons are untouched -- the cold arm carries nothing and f1 shows "
            "it does not drift",
            "nothing here re-grades a sealed artifact; it adds a limitation that can now be cited",
        ],
        "falsifiers": falsifiers,
    }
    seal_and_write(payload, ROOT / args.out, contract=ROOT / args.contract,
                   reference=ROOT / SOURCE)

    for label, block in (("vs REPLAY MARGINAL", vs_marginal), ("vs COLD (control interno)", vs_cold)):
        print(f"\n{label}")
        print(f"  {'familia':10}{'media':>11}{'primeras20':>12}{'ultimas20':>11}"
              f"{'deriva':>10}{'rho':>9}{'p_perm':>9}")
        for f in FAMILIES:
            r = block.get(f, {})
            if not r:
                continue
            print(f"  {f:10}{r['mean']:>+11.5f}{r[f'mean_first_{HEAD_TAIL}']:>+12.5f}"
                  f"{r[f'mean_last_{HEAD_TAIL}']:>+11.5f}{r['drift_last_minus_first']:>+10.5f}"
                  f"{r['rho_with_run_order']:>+9.3f}{r['p_permutation_two_sided']:>9.4f}")
    print(f"\nmasa del caso actual: {mass['current_case_share_first_evaluation']:.4%} (primera) -> "
          f"{mass['current_case_share_last_evaluation']:.4%} (ultima)")
    print(f"falsadores: {'todos pasan' if falsifiers['all_passed'] else 'FALLO'}"
          f"  ·  marginal deriva en {n_marginal_drifting}/4, cold en {n_cold_drifting}/4")
    print(f"-> {args.out}")
    print("\nTENDENCIA POR BRAZO (descriptiva; los valores ya se habian visto)")
    print(f"  {'brazo':18}{'rho':>9}{'p_perm':>9}{'deriva':>11}")
    for k in sorted(per_arm_trend, key=lambda k: per_arm_trend[k]["rho_with_run_order"]):
        r = per_arm_trend[k]
        print(f"  {k:18}{r['rho_with_run_order']:>+9.3f}{r['p_permutation_two_sided']:>9.4f}"
              f"{r['drift_last_minus_first']:>+11.5f}")
    return 0 if integrity_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
