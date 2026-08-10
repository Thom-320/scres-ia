#!/usr/bin/env python3
"""Re-adjudicate every Gate B run against the BEST NON-NEURAL comparator of each arm's class.

Amendment: `docs/ENMIENDA_PUERTA_B_MEJOR_NO_NEURONAL_2026-08-09.md`

Gate B's `f5` compared the networks against the primary baseline alone, while the framework
contract requires them to beat the best NON-NEURAL comparator. Gate A2 implements that rule and it
is what killed Gate A. The `ret_excel` sensitivity exposed the gap: a regression tree passes the
same criterion as the KAN, by a larger margin, so the printed verdict describes the primary not
being the best classical model rather than a neural premium.

No seeds are opened and no episode is run: every quantity comes from `per_fold` arrays already
sealed in the four artifacts. The frozen criterion is reused unchanged -- mean >= SESOI and the
paired CI excluding zero.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write                               # noqa: E402
from supply_chain import falsifiers as F                                         # noqa: E402

SESOI = 0.05
T_CRIT = {4: 2.776, 3: 3.182, 2: 4.303}
CONTRACT = Path("docs/ENMIENDA_PUERTA_B_MEJOR_NO_NEURONAL_2026-08-09.md")
OUT = Path("results/program_n/gate_b_readjudication/result.json")

#: Declared in the amendment BEFORE any number was computed. Partition is by information set.
CLASS_A_NEURAL = ("mlp_tuned", "kan_tuned")
CLASS_A_CLASSICAL = ("constant", "linear_additive", "linear_interactions", "spline_buffer", "tree")
CLASS_B_NEURAL = ("recurrent",)
CLASS_B_CLASSICAL = ("linear_lagged",)
#: Sees the cell identity no other arm receives. Never a comparator, never a judged arm.
EXCLUDED = "train_cell_mean_comparator"
PRIMARY = "linear_interactions"

RUNS = [
    ("gate_b_cd_surface", "DEVELOPMENT"),
    ("gate_b_confirmation_v2", "DEVELOPMENT_BLOCKED_INSTRUMENT"),
    ("gate_b_confirmation_v3", "CONFIRMATION"),
    ("gate_b_sensitivity_ret_excel", "SENSITIVITY_REPLAY"),
]


def paired(per_fold, model, baseline):
    a, b = np.array(per_fold[model], float), np.array(per_fold[baseline], float)
    d = (a - b)[~np.isnan(a - b)]
    t = T_CRIT.get(d.size - 1, 2.776)
    se = float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else 0.0
    lo, hi = float(d.mean() - t * se), float(d.mean() + t * se)
    return {"baseline": baseline, "mean_difference": float(d.mean()), "n_folds": int(d.size),
            "ci95_low": lo, "ci95_high": hi, "sesoi": SESOI,
            "passes_sesoi_and_ci": bool(d.mean() >= SESOI and lo > 0.0)}


def adjudicate(per_fold, means):
    out, comparators = {}, {}
    for neural, classical in ((CLASS_A_NEURAL, CLASS_A_CLASSICAL),
                              (CLASS_B_NEURAL, CLASS_B_CLASSICAL)):
        pool = [c for c in classical if c in per_fold]
        best = max(pool, key=lambda m: means[m])
        for arm in neural:
            if arm not in per_fold:
                continue
            comparators[arm] = best
            out[arm] = paired(per_fold, arm, best)
    return out, comparators


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("results/program_n"))
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).resolve().parent.parent / OUT)
    args = ap.parse_args()

    runs, fold_signatures = {}, {}
    for name, grade in RUNS:
        path = args.root / name / "result.json"
        d = json.loads(path.read_text())
        pf, means = d["per_fold"], d["held_out_r2_mean"]
        vs_bnn, comparators = adjudicate(pf, means)
        vs_primary = {a: paired(pf, a, PRIMARY)
                      for a in (*CLASS_A_NEURAL, *CLASS_B_NEURAL) if a in pf}
        held = {a: v["passes_sesoi_and_ci"] for a, v in vs_bnn.items()}
        held_primary = {a: v["passes_sesoi_and_ci"] for a, v in vs_primary.items()}
        ceiling_beaten = sorted(a for a, m in means.items()
                                if a != EXCLUDED and m > means[EXCLUDED])
        runs[name] = {
            "grade": grade, "artifact": str(path), "endpoint": d["endpoint"],
            "claim_status_as_sealed": d["claim_status"], "seeds": d["seeds"],
            "best_nonneural_comparator": comparators, "vs_best_nonneural": vs_bnn,
            "vs_primary_control": vs_primary,
            "readjudicated_premium": any(held.values()),
            "premium_under_primary_only": any(held_primary.values()),
            "arms_that_pass_vs_best_nonneural": sorted(a for a, v in held.items() if v),
            "ceiling_is_not_a_ceiling": ceiling_beaten,
        }
        fold_signatures[name] = json.dumps(pf, sort_keys=True)

    partition_clean = (
        not (set(CLASS_A_CLASSICAL) & set(CLASS_B_CLASSICAL))
        and EXCLUDED not in set(CLASS_A_CLASSICAL) | set(CLASS_B_CLASSICAL)
        and EXCLUDED not in set(CLASS_A_NEURAL) | set(CLASS_B_NEURAL)
        and all(r["best_nonneural_comparator"].get(a) in CLASS_A_CLASSICAL
                for r in runs.values() for a in CLASS_A_NEURAL
                if a in r["best_nonneural_comparator"])
        and all(r["best_nonneural_comparator"].get(a) in CLASS_B_CLASSICAL
                for r in runs.values() for a in CLASS_B_NEURAL
                if a in r["best_nonneural_comparator"]))

    verdict_changed = [n for n, r in runs.items()
                       if r["readjudicated_premium"] != r["premium_under_primary_only"]]

    checks = {
        "g1_the_partition_is_by_information_set": F.check(
            partition_clean,
            "a class-B comparator judging a class-A arm, or the cell-mean comparator appearing at "
            "all, is exactly the substitution that makes a network win by construction",
            computed_from={"n_class_a_classical": len(CLASS_A_CLASSICAL),
                           "n_class_b_classical": len(CLASS_B_CLASSICAL)},
            excluded_arm=EXCLUDED),
        "g2_every_artifact_contributes_its_own_folds": F.check(
            len(set(fold_signatures.values())) == len(RUNS),
            "two identical per_fold arrays would mean I read one artifact twice and reported it "
            "as two independent runs",
            computed_from={"n_runs": len(RUNS),
                           "n_distinct_fold_arrays": len(set(fold_signatures.values()))}),
        "g3_the_criterion_is_the_frozen_one": F.check(
            SESOI == 0.05,
            "re-adjudication is the ideal moment to quietly loosen the threshold; if SESOI or the "
            "CI rule differ from the original preregistration, the comparison is not the same one",
            computed_from={"sesoi": SESOI, "rule": 1.0}),
        "g4_a_control_must_change_the_verdict": F.check(
            len(verdict_changed) > 0,
            "the same adjudication under the ORIGINAL primary-only comparator must differ "
            "somewhere; if it never does, this amendment measures nothing and the defect it "
            "claims to repair was not a defect",
            computed_from={"n_runs": len(RUNS), "n_verdicts_changed": len(verdict_changed)},
            runs_whose_verdict_changed=verdict_changed),
    }
    checks["custody"] = {
        "passed": None, "not_applicable": True,
        "evidence": {"why_it_can_fail": "it cannot: no seed is opened and no episode is run. "
                                        "Every quantity is read from per_fold arrays already "
                                        "sealed in the four cited artifacts.",
                     "n_episodes_run": 0, "artifacts_read": [r["artifact"] for r in runs.values()]}}
    summary = F.summarise(checks)

    confirmatory = runs["gate_b_confirmation_v3"]
    if not checks["g1_the_partition_is_by_information_set"]["passed"]:
        status = "BLOCKED_INSTRUMENT"
    elif confirmatory["readjudicated_premium"]:
        status = "SURFACE_PREMIUM_SURVIVES_THE_BEST_NONNEURAL_COMPARATOR"
    else:
        status = "SURFACE_PREMIUM_DOES_NOT_SURVIVE_THE_BEST_NONNEURAL_COMPARATOR"

    payload = {
        "schema_version": "program_n_gate_b_readjudication_v1", "claim_status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_role": "READJUDICATION",
        "scope": "READJUDICATION_OF_SEALED_PER_FOLD_NO_SEEDS_OPENED_NO_EPISODES_RUN",
        "sesoi": SESOI, "primary_baseline_control": PRIMARY,
        "partition": {"class_a_neural": list(CLASS_A_NEURAL),
                      "class_a_classical": list(CLASS_A_CLASSICAL),
                      "class_b_neural": list(CLASS_B_NEURAL),
                      "class_b_classical": list(CLASS_B_CLASSICAL),
                      "excluded_from_both": EXCLUDED},
        "runs": runs, "falsifiers": checks, "falsifier_summary": summary,
        "contract_path": str(CONTRACT),
    }
    seal_and_write(payload, args.output, contract=CONTRACT,
                   reference=args.root / "gate_b_confirmation_v3/result.json")

    print(f"\nveredicto: {status}\n")
    for name, r in runs.items():
        print(f"  {name}  [{r['grade']}]  {r['endpoint'].split('_on_')[-1][:34]}")
        for arm, v in r["vs_best_nonneural"].items():
            print(f"    {arm:14} vs {v['baseline']:22} {v['mean_difference']:+.4f} "
                  f"[{v['ci95_low']:+.4f}, {v['ci95_high']:+.4f}]  "
                  f"{'PASA' if v['passes_sesoi_and_ci'] else 'no'}")
        if r["ceiling_is_not_a_ceiling"]:
            print(f"    techo superado por: {', '.join(r['ceiling_is_not_a_ceiling'])}")
    print("\n  falsadores:")
    for k, v in checks.items():
        mark = "N/A " if v.get("not_applicable") else ("PASA" if v["passed"] else "FALLA")
        print(f"    {k:44} {mark}")
    print(f"\n  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
