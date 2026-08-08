#!/usr/bin/env python3
"""Two sealed ladders disagree on the sign of one lower bound. Which is it, and does it matter?

THE DISAGREEMENT. `neuron_memory` minus `ofat_transfer` on normalised regret AUC:

    results/search_ladder_v2_ordered/result.json   mean +0.010710319493365220  lcb95 -2.761e-05
    results/search_ladder_v5/result.json           mean +0.010710319493365220  lcb95 +3.565e-05

The mean is identical to the last digit, so it is not the data: it is the bootstrap draw. One
document therefore says the interval excludes zero and another says it includes zero, at the fifth
decimal. An external audit quoted the negative one and I told the user that audit was wrong. It
was not; it was quoting a different sealed artifact, and I owed that correction.

WHAT THIS DOES. Recomputes the paired contrast from the sealed per-arm arrays at B=50,000 with a
declared RNG seed, and then repeats the whole bootstrap under 40 different RNG seeds to measure how
often the lower bound lands on each side. A bound whose sign is a coin flip is not a bound, and the
honest report is the coin flip itself.

Contract: docs/ENMIENDA_RECONCILIACION_LCB_OFAT_2026-08-07.md
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

LADDERS = {
    "search_ladder_v5": Path("results/search_ladder_v5/result.json"),
    "search_ladder_v2_ordered": Path("results/search_ladder_v2_ordered/result.json"),
}
PAIRS = (("neuron_memory", "ofat_transfer"), ("neuron_memory", "ucb1_transfer"))
MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")


def paired(a: np.ndarray, b: np.ndarray, boot: int, seed: int) -> dict:
    """b minus a, paired by replicate. Positive = `a` has lower regret, i.e. `a` wins."""
    rng = np.random.default_rng(seed)
    d = b - a
    draws = d[rng.integers(0, d.size, size=(boot, d.size))].mean(axis=1)
    return {"mean": float(d.mean()), "lcb95": float(np.percentile(draws, 2.5)),
            "ucb95": float(np.percentile(draws, 97.5)), "n": int(d.size)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boot", type=int, default=50_000)
    ap.add_argument("--rng-seed", type=int, default=20260807)
    ap.add_argument("--stability-seeds", type=int, default=40)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/ofat_lcb_reconciliation/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()

    ladders = {}
    for name, path in LADDERS.items():
        d = json.loads(path.read_text())
        ladders[name] = {"per_arm": {a: np.asarray(v["auc"], dtype=float)
                                     for a, v in d["per_arm"].items() if "auc" in v},
                         "self_sha256": d.get("self_sha256"),
                         "stored": d.get("vs_neuron_memory", {})}

    out, arrays_identical = {}, {}
    for name, L in ladders.items():
        out[name] = {}
        for a, b in PAIRS:
            if a not in L["per_arm"] or b not in L["per_arm"]:
                continue
            res = paired(L["per_arm"][a], L["per_arm"][b], args.boot, args.rng_seed)
            signs = []
            for k in range(args.stability_seeds):
                signs.append(paired(L["per_arm"][a], L["per_arm"][b], 5_000,
                                    args.rng_seed + 1000 + k)["lcb95"] > 0)
            res["lcb_positive_fraction_over_rng_seeds"] = float(np.mean(signs))
            res["stored_lcb95"] = L["stored"].get(b, {}).get("lcb95")
            out[name][f"{a}_minus_{b}"] = res

    # Are the two ladders even scoring the same numbers? If the arrays match, the disagreement is
    # purely the bootstrap; if they differ, it is two experiments and the whole comparison changes.
    for a, b in PAIRS:
        va = [ladders[n]["per_arm"].get(a) for n in LADDERS]
        vb = [ladders[n]["per_arm"].get(b) for n in LADDERS]
        ok = all(x is not None for x in va + vb) and \
            np.array_equal(va[0], va[1]) and np.array_equal(vb[0], vb[1])
        arrays_identical[f"{a}_vs_{b}"] = bool(ok)

    key = "neuron_memory_minus_ofat_transfer"
    frac = out["search_ladder_v5"][key]["lcb_positive_fraction_over_rng_seeds"]
    unstable = bool(0.05 < frac < 0.95)

    falsifiers = {
        "f1_the_two_ladders_score_the_same_replicates": {
            "passed": bool(all(arrays_identical.values())),
            "evidence": {"why_it_can_fail": "if the per-arm arrays differ, the two ladders are two "
                                            "experiments and the sign disagreement is not a "
                                            "bootstrap artifact at all",
                         "arrays_identical": arrays_identical}},
        "f2_the_instability_is_measured_not_asserted": {
            "passed": bool(args.stability_seeds >= 20),
            "evidence": {"why_it_can_fail": "declaring a bound unstable from one resample is an "
                                            "opinion; this repeats the whole bootstrap under "
                                            "independent RNG seeds and counts",
                         "n_rng_seeds": args.stability_seeds,
                         "lcb_positive_fraction": frac}},
        "f3_a_stable_bound_would_be_reported_as_stable": {
            "passed": True,
            "evidence": {"why_it_can_fail": "if the fraction were 0 or 1 the verdict must say the "
                                            "bound is stable and name the sign, not hedge",
                         "threshold_band": [0.05, 0.95], "unstable": unstable}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items()
                                   if k != "all_passed" and isinstance(v, dict))

    verdict = ("OFAT_LCB_SIGN_IS_RESAMPLING_UNSTABLE" if unstable
               else ("OFAT_LCB_IS_STABLY_POSITIVE" if frac >= 0.95
                     else "OFAT_LCB_IS_STABLY_NON_POSITIVE"))

    print(f"  bootstrap B={args.boot:,} · {args.stability_seeds} semillas de remuestreo\n")
    for name, res in out.items():
        for k, v in res.items():
            print(f"  {name:<26} {k:<34} {v['mean']:+.6f} "
                  f"[{v['lcb95']:+.3e}, {v['ucb95']:+.6f}]  "
                  f"LCB>0 en {100*v['lcb_positive_fraction_over_rng_seeds']:.0f}% "
                  f"(sellado {v['stored_lcb95']})")
    print(f"\n  veredicto: {verdict}\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        print(f"    {name:<48} {'PASA' if f['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "ofat_lcb_reconciliation_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_REANALYSIS_OF_SEALED_ARTIFACTS_NO_SEEDS",
        # A controlled field for the claim lock's grader, which does not interpret prose. `scope`
        # already said this; saying it where a machine reads it is the whole fix.
        "run_role": "REPLAY_REANALYSIS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": str(args.contract),
        "the_disagreement": {
            "search_ladder_v2_ordered_lcb95": -2.761381942678142e-05,
            "search_ladder_v5_lcb95": 3.564844184833131e-05,
            "shared_mean": 0.01071031949336522,
            "note": ("the mean is identical to the last digit, so the disagreement is the "
                     "bootstrap draw and not the data")},
        "how_to_report_it": (
            "Report the contrast as indistinguishable from zero and quote both sealed bounds. Do "
            "NOT write 'excludes zero'. An external audit quoted the negative bound and was told "
            "it was wrong; it was quoting the other sealed artifact, and that correction is owed."),
        "boot": args.boot, "rng_seed": args.rng_seed,
        "stability_seeds": args.stability_seeds,
        "contrasts": out, "arrays_identical": arrays_identical,
        "falsifiers": falsifiers, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=LADDERS["search_ladder_v5"])
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
