#!/usr/bin/env python3
"""Pool the four step-3 shards into one paired analysis.

WHAT POOLING IS ALLOWED TO DO HERE. The shards differ only in which tapes they touch, so pooling is
concatenation of per-tape rows -- never a re-scoring. The paired contrast lives inside a tape: every
arm is compared against the best static posture ON THE SAME TAPE, and the bootstrap resamples
TAPES, which is the replication unit. f1 fails if any tape seed appears in two shards, because then
the "independent shards" claim is false and the bootstrap would double-count.

FAMILIES ARE NOT POOLED WITH EACH OTHER. R1r and R2r are different estimands, not replicates. They
are reported side by side and never averaged.

A DEVIATION THAT HAS TO BE DECLARED, NOT PAPERED OVER. The preregistration names
`worst_product_fill` as the blocking service guardrail. This runner does not persist it -- the rows
carry `flow_fill_rate`, which is an aggregate over products and therefore CANNOT see one product
being abandoned while the aggregate holds up. So the guardrail applied here is strictly weaker than
the one preregistered, it is reported under its own name, and `f4` records the gap rather than
hiding it. A controller that passes this weaker screen has NOT passed the preregistered one.

Preregistration: docs/PREREGISTRO_PASO3_GARRIDO_MPC_EXPANDIDO_2026-08-06.md
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402

METRIC = "ret_excel_full_ledger"
SERVICE = "flow_fill_rate"          # NOT worst_product_fill -- see the module docstring
N_BOOT = 5_000
MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")
STATIC_ARM = "static"
#: NOT a controller. The runner itself labels it GREEDY_PI_BEST_FOUND_NOT_EXACT_CEILING: it is a
#: perfect-information best-found diagnostic, so it belongs with the oracle and can never count as
#: a structured controller converting the rights. The first version of this script let it into the
#: winner set and produced A_STRUCTURED_CONTROLLER_CONVERTS off the back of it.
CEILING_ARM = "greedy_pi_best_found_v2"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", nargs="+", type=Path, required=True)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--reference", type=Path,
                    help="artifact this pooling is downstream of; defaults to the first shard")
    ap.add_argument("--output", type=Path, default=Path("results/step3_pooled/result.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(20260806)

    rows, per_shard_seeds, shard_names = [], {}, []
    for shard in args.shards:
        path = shard / "rows.json" if shard.is_dir() else shard
        data = json.loads(path.read_text())
        name = shard.name
        shard_names.append(name)
        per_shard_seeds[name] = sorted({int(r["tape_seed"]) for r in data})
        for r in data:
            rows.append(dict(r, shard=name))
    print(f"  {len(rows)} filas de {len(shard_names)} shards")

    # f1: shards must be disjoint in tapes, or pooling double-counts.
    overlaps = {}
    for i, a in enumerate(shard_names):
        for b in shard_names[i + 1:]:
            common = sorted(set(per_shard_seeds[a]) & set(per_shard_seeds[b]))
            if common:
                overlaps[f"{a}|{b}"] = common

    families = sorted({r["family"] for r in rows})
    report = {}
    for fam in families:
        fam_rows = [r for r in rows if r["family"] == fam]
        tapes = sorted({int(r["tape_seed"]) for r in fam_rows})

        # The incumbent: best static posture by pooled mean, then scored PER TAPE.
        static_rows = [r for r in fam_rows if str(r["arm"]).startswith(STATIC_ARM)]
        by_posture = defaultdict(dict)
        for r in static_rows:
            by_posture[str(r["posture"])][int(r["tape_seed"])] = float(r[METRIC])
        full = {p: v for p, v in by_posture.items() if len(v) == len(tapes)}
        if not full:
            print(f"  {fam}: ninguna postura cubre las {len(tapes)} tapes; se omite")
            continue
        best_posture = max(full, key=lambda p: float(np.mean(list(full[p].values()))))
        incumbent = full[best_posture]

        arms = sorted({str(r["arm"]) for r in fam_rows if not str(r["arm"]).startswith(STATIC_ARM)})
        comparisons = {}
        for arm in arms:
            paired, service = [], []
            for r in fam_rows:
                if str(r["arm"]) != arm:
                    continue
                t = int(r["tape_seed"])
                if t in incumbent:
                    paired.append(float(r[METRIC]) - incumbent[t])
                    service.append(float(r.get(SERVICE, float("nan"))))
            if not paired:
                continue
            d = np.asarray(paired)
            draws = [float(np.mean(d[rng.integers(0, len(d), len(d))])) for _ in range(N_BOOT)]
            comparisons[arm] = {
                "delta_mean": float(d.mean()),
                "lcb95": float(np.percentile(draws, 2.5)),
                "ucb95": float(np.percentile(draws, 97.5)),
                "n_tapes": int(len(d)),
                "positive_tapes": int(np.sum(d > 0)),
                f"mean_{SERVICE}": float(np.nanmean(service)) if service else None,
                "beats_incumbent": bool(float(np.percentile(draws, 2.5)) > 0),
            }
        report[fam] = {
            "n_tapes": len(tapes), "tapes": tapes,
            "best_static_posture": best_posture,
            "best_static_mean": float(np.mean(list(incumbent.values()))),
            "n_postures_complete": len(full),
            "comparisons": comparisons,
        }
        print(f"\n  == {fam}: {len(tapes)} tapes · mejor postura {best_posture} "
              f"({report[fam]['best_static_mean']:.6f}) de {len(full)} completas")
        for arm, c in sorted(comparisons.items(), key=lambda kv: -kv[1]["delta_mean"]):
            flag = "  <-- GANA" if c["beats_incumbent"] else ""
            print(f"     {arm:<26} Δ {c['delta_mean']:+.6f} "
                  f"[{c['lcb95']:+.6f} · {c['ucb95']:+.6f}]  "
                  f"{c['positive_tapes']}/{c['n_tapes']} tapes{flag}")

    winners = {f: [a for a, c in b["comparisons"].items()
                   if c["beats_incumbent"] and a != CEILING_ARM]
               for f, b in report.items()}
    ceiling = {f: b["comparisons"].get(CEILING_ARM) for f, b in report.items()}
    any_win = any(winners.values())
    verdict = ("A_STRUCTURED_CONTROLLER_CONVERTS_THE_EXPANDED_RIGHTS" if any_win
               else "NO_STRUCTURED_CONTROLLER_CONVERTS")

    falsifiers = {
        "f1_shards_are_disjoint_in_tapes": {
            "passed": not overlaps,
            "evidence": {"why_it_can_fail": "pooling is concatenation of per-tape rows; a tape in "
                                            "two shards would be counted twice and the bootstrap "
                                            "would understate its own uncertainty",
                         "seeds_per_shard": per_shard_seeds, "overlaps": overlaps}},
        "f2_the_incumbent_comes_from_the_full_posture_set": {
            "passed": all(b["n_postures_complete"] >= 200 for b in report.values()),
            "evidence": {"why_it_can_fail": "v1's central defect was picking the incumbent from 6 "
                                            "of 216 postures. If the pooled set is short, the "
                                            "incumbent is again the best HOMOGENEOUS posture "
                                            "rather than the real one",
                         "complete_postures": {f: b["n_postures_complete"]
                                               for f, b in report.items()}}},
        "f3_families_are_not_averaged_together": {
            "passed": len(report) == len(families),
            "evidence": {"why_it_can_fail": "R1r and R2r are different estimands; averaging them "
                                            "would invent a number that describes neither",
                         "families": families}},
        "f5_the_ceiling_arm_is_not_counted_as_a_winner": {
            "passed": all(CEILING_ARM not in w for w in winners.values()),
            "evidence": {"why_it_can_fail": "greedy_pi is perfect-information and beats the "
                                            "incumbent by construction. Counting it as a "
                                            "structured controller would manufacture the headline "
                                            "this whole run exists to test",
                         "winners": winners}},
        "f4_the_preregistered_guardrail_is_not_available": {
            "passed": False,
            "evidence": {"why_it_can_fail": "declared, not discovered: the preregistration names "
                                            "worst_product_fill as the blocking guardrail and this "
                                            "runner does not persist it. flow_fill_rate is an "
                                            "aggregate and cannot see one product abandoned while "
                                            "the aggregate holds. Any arm passing here has NOT "
                                            "passed the preregistered screen",
                         "preregistered": "worst_product_fill", "applied": SERVICE}},
    }
    falsifiers["all_passed"] = all(
        v["passed"] for k, v in falsifiers.items()
        if k != "all_passed" and isinstance(v, dict))

    print(f"\n  veredicto: {verdict}\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        print(f"    {name:<48} {'PASA' if f['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "step3_pooled_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_POOLED_ANALYSIS_SERVICE_GUARDRAIL_INCOMPLETE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": "docs/PREREGISTRO_PASO3_GARRIDO_MPC_EXPANDIDO_2026-08-06.md",
        "metric": METRIC, "service_metric_applied": SERVICE,
        "service_guardrail_deviation": (
            "The preregistration names worst_product_fill; the runner persists only "
            "flow_fill_rate, an aggregate that cannot detect a single product being abandoned. "
            "The screen applied here is strictly weaker than the one preregistered."),
        "shards": shard_names, "seeds_per_shard": per_shard_seeds,
        "families": report, "winners": winners,
        "ceiling_diagnostic": {
            "arm": CEILING_ARM, "by_family": ceiling,
            "why_excluded_from_the_verdict": (
                "perfect-information best-found, not a deployable policy; the runner labels it "
                "GREEDY_PI_BEST_FOUND_NOT_EXACT_CEILING and it belongs with the oracle")},
        "falsifiers": falsifiers,
    }
    # seal_and_write parses the reference as JSON, so it has to be an artifact and not the
    # preregistration markdown. The first shard's own result.json is the honest anchor: the pooled
    # analysis is downstream of it.
    reference = args.reference or (args.shards[0] / "result.json")
    digest = seal_and_write(payload, args.output, contract=args.contract, reference=reference)
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if not overlaps else 1


if __name__ == "__main__":
    raise SystemExit(main())
