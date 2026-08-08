#!/usr/bin/env python3
"""RQ1 as a paired within-family contrast, because the ranking observation cannot carry it.

WHAT WAS WRONG WITH THE OLD FRAMING. Paper 2's RQ1 rested on "the top six of fifteen methods are
exactly the six that retain state". That is visually striking and inferentially empty: the six
methods differ from the other nine in exploration policy, inductive bias, update rule and
representation as well as in retention, so the ranking cannot attribute the gap to retention. Eight
external audits said so independently.

WHAT REPLACES IT. For each family f with a retained/reset twin pair,

    Delta_f = AUC(f_reset) - AUC(f_retained)          positive => retention helps

paired per seed under common random numbers. `search_ladder_v5` stores `per_arm[arm]["auc"]` as one
float per seed in `seeds` order, and the artifact already performs exactly this subtraction for its
`vs_neuron_memory` block -- so this is a re-read of sealed numbers, not a new simulation.

WHY IT GOES THROUGH THE PIPELINE ANYWAY. Only ONE of the six contrasts is sealed today
(`vs_neuron_memory.neuron_reset`). The other five are visible only against `neuron_memory`, which is
a CROSS-family baseline, not each arm's twin. Differencing two bootstrap intervals is not a valid
interval, so the five must be computed -- and a number that enters a manuscript is computed by
`arm_runner.py` and sealed, never lifted from an ad-hoc calculation.

TWO LIMITS THAT TRAVEL WITH THE RESULT. The six contexts are averaged inside each seed before
storage (`run_search_comparator_ladder_v5.py:218`), so a within-context contrast is NOT recoverable
from this artifact. And `f7` of the source records the twelve seeds as a declared re-execution of
the burned block `garrido_q2_des288`: this is REPLAY evidence, not a prospective confirmation.

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
MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")

#: The recipe is the source artifact's own: run_search_comparator_ladder_v5.py:226-231.
N_BOOT = 5_000
BOOT_SEED = 20260806

#: family -> (reset arm, retained arm). Orientation is reset minus retained, so a positive Delta
#: means retention lowered regret.
TWINS = {
    "neuron": ("neuron_reset", "neuron_memory"),
    "ucb1": ("ucb1", "ucb1_transfer"),
    "ofat": ("ofat", "ofat_transfer"),
    "lookahead_kg": ("lookahead_kg", "lookahead_kg_transfer"),
    "gp_ei": ("gp_ei", "gp_ei_transfer"),
    "thompson": ("thompson", "thompson_transfer"),
}

#: The one contrast the source already seals. Reproducing its point estimate bit-for-bit is what
#: proves this script implements THEIR estimand and not a lookalike of my own.
SEALED_CHECK = ("neuron", "vs_neuron_memory", "neuron_reset")


def boot(diff: np.ndarray, rng: np.random.Generator) -> dict:
    draws = [float(np.mean(diff[rng.integers(0, len(diff), len(diff))])) for _ in range(N_BOOT)]
    return {"mean": float(np.mean(diff)), "lcb95": float(np.percentile(draws, 2.5)),
            "ucb95": float(np.percentile(draws, 97.5)), "n": int(len(diff))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path,
                    default=Path("docs/ENMIENDA_REGISTRO_DE_EVIDENCIA_2026-08-07.md"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/retention_contrasts/result.json"))
    args = ap.parse_args()

    src = json.loads((ROOT / SOURCE).read_text())
    per_arm, seeds = src["per_arm"], src["seeds"]
    memory_arms = set(src["memory_arms"])
    rng = np.random.default_rng(BOOT_SEED)

    # The alignment check runs BEFORE any arithmetic. Caught by its own control: with a truncated
    # array the subtraction raises and the script dies, so f2 -- the falsifier that exists to catch
    # exactly that -- could never report its own failure. A falsifier that cannot be reached is not
    # a falsifier.
    lengths = {a: len(per_arm[a]["auc"]) for pair in TWINS.values() for a in pair}
    aligned = all(n == len(seeds) for n in lengths.values())

    contrasts, arrays = {}, {}
    for family, (reset_arm, retained_arm) in (TWINS.items() if aligned else ()):
        d = (np.asarray(per_arm[reset_arm]["auc"], float)
             - np.asarray(per_arm[retained_arm]["auc"], float))
        arrays[family] = d
        row = boot(d, rng)
        row.update({"reset_arm": reset_arm, "retained_arm": retained_arm,
                    "mean_auc_reset": float(np.mean(per_arm[reset_arm]["auc"])),
                    "mean_auc_retained": float(np.mean(per_arm[retained_arm]["auc"])),
                    "seeds_favouring_retention": int((d > 0).sum()),
                    "excludes_zero": bool(row["lcb95"] > 0.0)})
        contrasts[family] = row

    # ---- f3: does the sealed contrast come back bit-for-bit? --------------------------------
    family, block, key = SEALED_CHECK
    sealed = src[block][key]
    mine = contrasts.get(family, {"mean": None, "lcb95": None, "ucb95": None})
    point_exact = mine["mean"] is not None and float(sealed["mean"]) == float(mine["mean"])

    # ---- f4 control: destroy the pairing and show the estimator noticed ----------------------
    ctrl_rng = np.random.default_rng(BOOT_SEED)
    shuffled = {}
    for family, (reset_arm, retained_arm) in (TWINS.items() if aligned else ()):
        a = np.asarray(per_arm[reset_arm]["auc"], float)
        b = np.asarray(per_arm[retained_arm]["auc"], float)
        b_shuf = ctrl_rng.permutation(b)
        shuffled[family] = {"mean": float(np.mean(a - b_shuf)),
                            "interval_width_paired": mine and float(
                                contrasts[family]["ucb95"] - contrasts[family]["lcb95"]),
                            "interval_width_shuffled": float(
                                np.percentile([np.mean((a - b_shuf)[ctrl_rng.integers(0, len(a),
                                                                                      len(a))])
                                               for _ in range(500)], 97.5)
                                - np.percentile([np.mean((a - b_shuf)[ctrl_rng.integers(0, len(a),
                                                                                        len(a))])
                                                 for _ in range(500)], 2.5))}
    widened = sum(1 for f in TWINS if f in shuffled
                  and shuffled[f]["interval_width_shuffled"] > shuffled[f]["interval_width_paired"])

    falsifiers = {
        "f1_the_twins_are_labelled_the_way_the_source_labels_them": {
            "passed": all(ret in memory_arms and res not in memory_arms
                          for res, ret in TWINS.values()),
            "evidence": {
                "why_it_can_fail": "if a pair is inverted, or an arm the source does not count as "
                                   "stateful is used as the retained side, every Delta flips or "
                                   "compares the wrong things",
                "memory_arms_declared_by_source": sorted(memory_arms),
                "pairs": {f: {"reset": r, "retained": t,
                              "retained_is_declared_stateful": t in memory_arms,
                              "reset_is_declared_stateless": r not in memory_arms}
                          for f, (r, t) in TWINS.items()}}},
        "f2_every_arm_is_aligned_to_the_same_seed_vector": {
            "passed": aligned,
            "evidence": {
                "why_it_can_fail": "the pairing is POSITIONAL; an arm with a different array length "
                                   "would be silently paired against the wrong seeds and the "
                                   "contrast would be meaningless",
                "n_seeds": len(seeds), "lengths": lengths,
                "halted_before_arithmetic": not aligned}},
        "f3_the_already_sealed_contrast_reproduces_exactly": {
            "passed": point_exact,
            "evidence": {
                "why_it_can_fail": "if this script's estimand differed from the source's -- wrong "
                                   "orientation, unpaired means, a different array -- the point "
                                   "estimate would not be bit-identical",
                "sealed_mean": sealed["mean"], "recomputed_mean": mine["mean"],
                "bitwise_identical": point_exact,
                "interval_note": ("the intervals differ in the last decimals because the bootstrap "
                                  "stream position differs; the point estimate is deterministic "
                                  "arithmetic and must match exactly"),
                "sealed_interval": [sealed["lcb95"], sealed["ucb95"]],
                "recomputed_interval": [mine["lcb95"], mine["ucb95"]]}},
        "f4_the_pairing_carries_information": {
            "passed": widened >= 4,
            "evidence": {
                "why_it_can_fail": "if breaking the seed pairing left the intervals unchanged, the "
                                   "common random numbers would be doing nothing and a paired "
                                   "estimator would be an unpaired one wearing a better name",
                "families_whose_interval_widened_when_shuffled": widened,
                "of": len(TWINS), "detail": shuffled}},
        "f5_no_seed_outside_the_declared_replay_block": {
            "passed": all(5_300_001 <= s <= 5_300_012 for s in seeds),
            "evidence": {
                "why_it_can_fail": "a seed outside the block the source declares as a replay would "
                                   "mean this analysis consumed custody it never declared",
                "seeds": seeds, "replay_of": src.get("replay_of")}},
    }
    falsifiers["all_passed"] = all(v["passed"] for v in falsifiers.values() if isinstance(v, dict))

    n_positive = sum(1 for v in contrasts.values() if v["excludes_zero"])
    payload = {
        "schema_version": "retention_contrasts_v1",
        "claim_status": (f"RETENTION_LOWERS_REGRET_IN_{n_positive}_OF_{len(TWINS)}_FAMILIES"
                         if falsifiers["all_passed"] else "HALTED_FALSIFIER_FAILED"),
        "scope": "DEVELOPMENT_REANALYSIS_OF_A_SEALED_REPLAY_NO_SEEDS_NO_ADJUDICATION",
        # Without this the claim lock's grader has no controlled field to read and reports
        # GRADE_NOT_MACHINE_DISCOVERABLE -- a claim with no legible grade cannot govern a sentence.
        "run_role": "REPLAY_REANALYSIS",
        "endpoint": src["primary_metric"],
        "estimand": "paired per-seed AUC(reset) - AUC(retained), positive means retention helps",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        # NOT preregistered, and saying so is the point. `--contract` names the evidence-registry
        # amendment, which fixes custody and reporting rules but contains no estimand, no arms and
        # no falsifiers for THIS analysis. Grep of docs/ finds no document that preregisters these
        # six contrasts. What makes them citable is not a prior registration but that f3 reproduces
        # the one contrast the source already sealed, bit-for-bit. Calling this "preregistered"
        # would be the cheapest possible lie in a paper whose second contribution is falsifier
        # discipline.
        "preregistration": None,
        "registration_status": "POST_HOC_REANALYSIS_OF_A_SEALED_ARTIFACT_NOT_PREREGISTERED",
        "why_it_is_citable_anyway": (
            "the estimand is the source artifact's own -- f3 reproduces its sealed contrast to the "
            "last bit -- and the five new contrasts apply that identical estimand to twin pairs it "
            "never formed; no new degree of freedom is introduced by this script"),
        "governing_contract": str(args.contract),
        "source": str(SOURCE),
        "source_self_sha256": src.get("self_sha256"),
        "seeds": seeds,
        "bootstrap": {"n_boot": N_BOOT, "rng_seed": BOOT_SEED,
                      "recipe": "run_search_comparator_ladder_v5.py:226-231, percentile"},
        "contrasts": contrasts,
        "falsifiers": falsifiers,
        "limits_that_travel_with_this_result": [
            "The six contexts are averaged inside each seed before storage "
            "(run_search_comparator_ladder_v5.py:218), so a within-context contrast is NOT "
            "recoverable from this artifact and would require a re-run.",
            "The twelve seeds are a declared re-execution of the burned block garrido_q2_des288. "
            "This is REPLAY evidence and cannot be reported as a prospective confirmation.",
            "Six families is not a random sample of search methods; the claim is about the "
            "families tested under this budget and contract.",
        ],
    }
    digest = seal_and_write(payload, ROOT / args.out, contract=ROOT / args.contract,
                            reference=ROOT / SOURCE)

    print(f"  {'familia':14} {'Δ = reset − retenido':>22}   IC95                    semillas")
    for f, v in sorted(contrasts.items(), key=lambda kv: -kv[1]["mean"]):
        print(f"  {f:14} {v['mean']:+22.5f}   [{v['lcb95']:+.5f}, {v['ucb95']:+.5f}]   "
              f"{v['seeds_favouring_retention']}/{v['n']}")
    print(f"\n  {n_positive}/{len(TWINS)} familias con la cota inferior por encima de cero")
    for k, v in falsifiers.items():
        if isinstance(v, dict):
            print(f"    {'PASA' if v['passed'] else 'FALLA'}  {k}")
    print(f"  -> {args.out} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
