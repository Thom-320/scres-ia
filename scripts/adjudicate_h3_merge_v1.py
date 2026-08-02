#!/usr/bin/env python3
"""Adjudicate the H3' merge of 120 replicates from the two contracted slices.

The H3' power contract (docs/PREREGISTRO_H3_POTENCIA_2026-08-01.md) specifies the merge itself:
"Se fusionan por concatenación de réplicas, que es válido porque cada réplica es independiente y
lleva su propia semilla CRN." This runner performs exactly that concatenation, checks the three
merge falsifiers the contract declares, and applies its reading rule -- no new physics, no seeds.

THE ESTIMAND IS NOT THE ALZHEIMER MEAN. H3' as rewritten in
docs/PREREGISTRO_H1_H3_V2_2026-08-01.md is the VARIANCE OF SEARCH COST ACROSS CONTEXTS: memory
should make search cost more uniform between risk contexts than reset does. Adjudicating on
`reset - memory` in mean runs would settle a different hypothesis than the one preregistered, so
the variance function is lifted verbatim from run_h1_h3_v2.py rather than reinvented here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402

CONTRACT = Path("docs/PREREGISTRO_H3_POTENCIA_2026-08-01.md")
STRATEGIES = ("neuron_memory", "neuron_reset", "ofat", "random")
TARGET_N = 120


def search_cost_variance(payload: dict, strategy: str, ctx_order: list[str]) -> np.ndarray:
    """Per-replicate variance of search cost across contexts. Lifted from run_h1_h3_v2.py:191."""
    per = payload["per_context"][strategy]
    return np.array([
        float(np.var([per[r][c]["runs_to_within_1pct"] for c in ctx_order], ddof=1))
        for r in range(len(per))])


def paired(d: np.ndarray, n_boot: int, rng) -> dict:
    draws = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return {"mean": float(d.mean()), "lcb95": float(np.percentile(draws, 2.5)),
            "ucb95": float(np.percentile(draws, 97.5)), "n": int(d.size)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local", type=Path,
                    default=Path("results/garrido_meta_learner_h3power_h3_contract_local_v2/result.json"))
    ap.add_argument("--vps", type=Path,
                    default=Path("results/garrido_meta_learner_h3power_h3_contract_vps_v2/result.json"))
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--output", type=Path,
                    default=Path("results/garrido_h3_merge_adjudication/result.json"))
    args = ap.parse_args()

    A = json.loads(args.local.read_text())
    B = json.loads(args.vps.read_text())

    # ---- the three merge falsifiers the contract declares ------------------------------------
    seeds_a, seeds_b = list(A["seeds"]), list(B["seeds"])
    disjoint = not (set(seeds_a) & set(seeds_b))
    shared_keys = ("budget", "factors", "contexts", "metric", "n_configurations")
    design_match = {k: A.get(k) == B.get(k) for k in shared_keys}
    man_a, man_b = A.get("module_manifest", {}), B.get("module_manifest", {})
    source_identical = bool(
        man_a.get("modules") and man_a.get("modules") == man_b.get("modules")
        and man_a.get("entry_script_sha256") == man_b.get("entry_script_sha256")
        and A.get("contract_sha256") == B.get("contract_sha256"))

    falsifiers = {
        "f_merge_seeds_are_disjoint": {
            "passed": bool(disjoint),
            "evidence": {"why_it_can_fail": "an overlap would make two 'independent replicates' "
                                            "the same one",
                         "n_local": len(seeds_a), "n_vps": len(seeds_b),
                         "overlap": sorted(set(seeds_a) & set(seeds_b))}},
        "f_merge_contexts_and_budget_match": {
            "passed": all(design_match.values()),
            "evidence": {"why_it_can_fail": "merging runs with a different budget or context "
                                            "order would mix two experiments",
                         "per_key": design_match}},
        "f_merge_source_is_identical": {
            "passed": source_identical,
            "evidence": {"why_it_can_fail": "slices produced by different runner versions are not "
                                            "the same experiment. The contract says 'se compara el "
                                            "hash del script'; a script hash alone is insufficient "
                                            "because the runner imports supply_chain, so the whole "
                                            "declared module manifest is compared instead",
                         "module_hashes_equal": man_a.get("modules") == man_b.get("modules"),
                         "entry_script_equal":
                             man_a.get("entry_script_sha256") == man_b.get("entry_script_sha256"),
                         "contract_equal": A.get("contract_sha256") == B.get("contract_sha256"),
                         "manifest_schema": [man_a.get("schema"), man_b.get("schema")]}},
        "f_merge_reaches_the_contracted_n": {
            "passed": len(seeds_a) + len(seeds_b) == TARGET_N,
            "evidence": {"why_it_can_fail": "the power calculation fixed n=120; a short merge "
                                            "would not have the power the reading rule assumes",
                         "n_merged": len(seeds_a) + len(seeds_b), "n_contracted": TARGET_N}},
    }
    falsifiers["all_passed"] = all(v["passed"] for k, v in falsifiers.items() if k != "all_passed")

    if not falsifiers["all_passed"]:
        payload = {"schema_version": "garrido_h3_merge_adjudication_v1",
                   "claim_status": "HALTED_FALSIFIER_FAILED", "falsifiers": falsifiers}
        digest = seal_and_write(payload, args.output, contract=CONTRACT, reference=args.local)
        for name, f in falsifiers.items():
            if name != "all_passed":
                print(f"  {name:<40} {'PASA' if f['passed'] else 'FALLA'}")
        print(f"\n  DETENIDO -> {args.output} (sello {digest[:16]}…)")
        return 1

    # ---- the merge: concatenation of replicates, exactly as the contract specifies ------------
    ctx_order = list(A["contexts"])
    rng = np.random.default_rng(20260802)
    var = {s: np.concatenate([search_cost_variance(A, s, ctx_order),
                              search_cost_variance(B, s, ctx_order)]) for s in STRATEGIES}
    h3 = {"memory_vs_reset": paired(var["neuron_reset"] - var["neuron_memory"], args.n_boot, rng),
          "memory_vs_ofat": paired(var["ofat"] - var["neuron_memory"], args.n_boot, rng)}
    per_slice = {
        "local_90": paired(search_cost_variance(A, "neuron_reset", ctx_order)
                           - search_cost_variance(A, "neuron_memory", ctx_order), args.n_boot, rng),
        "vps_30": paired(search_cost_variance(B, "neuron_reset", ctx_order)
                         - search_cost_variance(B, "neuron_memory", ctx_order), args.n_boot, rng)}

    sustained = h3["memory_vs_reset"]["lcb95"] > 0
    verdict = "H3_PRIME_SUSTAINED_AT_N120" if sustained else "H3_PRIME_REFUTED_WITH_POWER_AT_N120"

    print("  === H3' — varianza del coste de búsqueda entre contextos, n = 120 ===")
    for s in STRATEGIES:
        print(f"    {s:<16} varianza media {var[s].mean():>10.4f}")
    for k, v in h3.items():
        print(f"\n  {k:<18} {v['mean']:+.4f} [LCB95 {v['lcb95']:+.4f}, UCB95 {v['ucb95']:+.4f}]"
              f"  n={v['n']}")
    print("\n  por rebanada (diagnóstico, no la adjudicación):")
    for k, v in per_slice.items():
        print(f"    {k:<10} {v['mean']:+.4f} [{v['lcb95']:+.4f}, {v['ucb95']:+.4f}] n={v['n']}")
    print(f"\n  veredicto: {verdict}\n")
    for name, f in falsifiers.items():
        if name != "all_passed":
            print(f"    {name:<40} {'PASA' if f['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "garrido_h3_merge_adjudication_v1",
        "claim_status": verdict,
        "estimand": ("variance of search cost ACROSS CONTEXTS, per replicate; memory minus reset. "
                     "NOT the Alzheimer mean (reset - memory in runs), which is a different "
                     "quantity that this contract does not adjudicate"),
        "reading_rule": "LCB95 > 0 -> sustained; LCB95 <= 0 at n=120 -> refuted with power",
        "h3_prime": h3,
        "per_slice_diagnostic": per_slice,
        "mean_variance_by_strategy": {s: float(var[s].mean()) for s in STRATEGIES},
        "sources": {"local": str(args.local), "local_sha256": A.get("self_sha256"),
                    "vps": str(args.vps), "vps_sha256": B.get("self_sha256")},
        "seeds": sorted(seeds_a + seeds_b),
        "custody_note": ("both slices carry f6 = DECLARED_REPLAY. The 6_000_001-120 block was "
                         "opened ONCE as this contract's own virgin block; the re-executions "
                         "reproduce the originals to the last decimal and only correct the sealed "
                         "contract path and add a module manifest"),
        "falsifiers": falsifiers,
    }
    digest = seal_and_write(payload, args.output, contract=CONTRACT, reference=args.local)
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
