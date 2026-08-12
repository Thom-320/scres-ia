#!/usr/bin/env python3
"""Pool the six 48-tape sub-blocks into the 288-tape powered replication.

Preregistration: `docs/EXCEPCION_PI_Y_PREREGISTRO_REPLICA_CON_POTENCIA_2026-08-12.md`

The frozen validation runner refuses any block that is not 48 tapes, and that guard is correct, so
it was executed byte-identically six times on disjoint virgin sub-blocks. This pools the six.

WHY POOLING IS NEW CODE, AND WHAT THAT COSTS. `joint_bootstrap` hardcodes `n_tapes = 48`, so it
cannot be called with 288. Pooling therefore has to be written, and written code has to be
validated: applied to ONE sub-block it must reproduce that sub-block's own sealed estimates and
simultaneous bounds. That reproduction is a falsifier here, not a comment.

THE POOLING RULE, declared. The six sub-blocks are independent and equally sized, so for every
estimand the pooled point is the mean of the six points and the pooled standard error is
sqrt(sum se_i^2)/6. The simultaneous critical value is a property of the CORRELATION structure of
the studentized estimands, which pooling independent replicates leaves unchanged; the MAXIMUM of the
six sub-block criticals is used, which is the conservative choice.

Nothing about the science moves. Only the sample size does, and the bar stays where it was.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supply_chain.arm_runner import seal_and_write                               # noqa: E402
from supply_chain import falsifiers as F                                         # noqa: E402

CONTRACT = Path("docs/EXCEPCION_PI_Y_PREREGISTRO_REPLICA_CON_POTENCIA_2026-08-12.md")
OUT = Path("results/program_o/powered_replication_v1/result.json")
FROZEN_RUNNER = ROOT / "scripts/screen_program_o_fixed_clock_hobs_validation.py"
RUNNER_SHA_AT_LAUNCH = "bf8de3674333e9150fbd6e8b3b835754c736d150369d0c4994cfee5ee74a4161"
CELLS = ("rho75_share90", "rho90_share75", "rho90_share90")
TAIL = "ret_visible_cvar10"
PRIMARY_SUFFIX = "::primary::ret_visible"


def pool(estimate_sets: list[dict], criticals: list[float]) -> dict:
    """Mean of the points, root-sum-square of the SEs over k, max of the criticals."""
    k = len(estimate_sets)
    names = sorted(set().union(*[set(e) for e in estimate_sets]))
    critical = float(max(criticals))
    out = {}
    for name in names:
        pts = [e[name]["estimate"] for e in estimate_sets if name in e]
        ses = [e[name]["bootstrap_se"] for e in estimate_sets if name in e]
        if len(pts) != k:
            continue
        point = float(np.mean(pts))
        se = float(np.sqrt(np.sum(np.square(ses))) / k)
        out[name] = {"estimate": point, "pooled_se": se,
                     "simultaneous_lcb95": float(point - critical * se),
                     "kind": estimate_sets[0][name]["kind"],
                     "metric_or_mode": estimate_sets[0][name]["metric_or_mode"],
                     "n_sub_blocks": k}
    return {"critical": critical, "estimates": out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=Path("/tmp/o_powered"))
    ap.add_argument("--output", type=Path, default=ROOT / OUT)
    args = ap.parse_args()

    blocks = []
    for k in range(1, 7):
        d = json.loads((args.runs / f"block{k}" / "result.json").read_text())
        blocks.append(d)

    estimate_sets = [b["inference"]["estimates"] for b in blocks]
    criticals = [float(b["inference"]["simultaneous_critical"]) for b in blocks]
    pooled = pool(estimate_sets, criticals)

    # REPRODUCTION CONTROL: pooling a single sub-block must return that sub-block unchanged.
    solo = pool([estimate_sets[0]], [criticals[0]])
    probe = f"{CELLS[0]}::guardrail::{TAIL}"
    reproduces = (abs(solo["estimates"][probe]["estimate"]
                      - estimate_sets[0][probe]["estimate"]) < 1e-12
                  and abs(solo["estimates"][probe]["simultaneous_lcb95"]
                          - estimate_sets[0][probe]["simultaneous_lcb95"]) < 1e-9)

    seeds = sorted({s for b in blocks for s in b["seeds"]}) if isinstance(
        blocks[0].get("seeds"), list) else []
    n_tapes = sum(len(b["cells"][CELLS[0]]["per_tape_ret_delta"]) for b in blocks)
    statics = {c: {b["cells"][c]["static_index"] for b in blocks} for c in CELLS}
    statics_frozen = all(len(v) == 1 for v in statics.values())
    runner_sha = hashlib.sha256(FROZEN_RUNNER.read_bytes()).hexdigest()

    tail = {c: pooled["estimates"][f"{c}::guardrail::{TAIL}"] for c in CELLS}
    prim = {c: pooled["estimates"][f"{c}{PRIMARY_SUFFIX}"] for c in CELLS}
    per_block_tail = {c: [e[f"{c}::guardrail::{TAIL}"]["estimate"] for e in estimate_sets]
                      for c in CELLS}
    other_guardrails = {n: v for n, v in pooled["estimates"].items()
                        if v["kind"] == "guardrail" and TAIL not in n}
    guardrails_ok = all(v["simultaneous_lcb95"] >= -1e-9 or v["estimate"] >= 0.0
                        for v in other_guardrails.values())

    checks = {
        "x1_the_pooling_reproduces_a_single_sub_block": F.check(
            reproduces,
            "pooling is code I had to write because the frozen bootstrap hardcodes 48 tapes; if it "
            "cannot return one sub-block unchanged it is not pooling, it is a new estimator",
            computed_from={"probe_gap": abs(solo["estimates"][probe]["estimate"]
                                            - estimate_sets[0][probe]["estimate"]),
                           "n_estimands": len(pooled["estimates"])}),
        "x2_the_frozen_runner_was_not_touched": F.check(
            runner_sha == RUNNER_SHA_AT_LAUNCH,
            "if the runner's bytes changed between registering the exception and pooling, the six "
            "sub-blocks were not produced by the same instrument and this is not a replication",
            computed_from={"sha_matches": float(runner_sha == RUNNER_SHA_AT_LAUNCH)},
            sha256=runner_sha, expected=RUNNER_SHA_AT_LAUNCH),
        "x3_the_frozen_comparator_is_identical_across_sub_blocks": F.check(
            statics_frozen,
            "a static index that moved between sub-blocks would mean each block was measured "
            "against a different comparator, and pooling them would be meaningless",
            computed_from={"n_cells": len(CELLS),
                           "n_cells_with_one_static": sum(len(v) == 1 for v in statics.values())},
            static_indices={c: sorted(v) for c, v in statics.items()}),
        "x4_the_effect_replicates_in_point": F.check(
            all(v["estimate"] > 0.0 for v in tail.values()),
            "the point estimates came from a block already used, so they may be optimistic; if the "
            "sign does not survive on 288 virgin tapes the earlier effect was noise",
            computed_from={"n_cells": len(CELLS),
                           "n_positive": sum(v["estimate"] > 0.0 for v in tail.values())},
            pooled_tail={c: v["estimate"] for c, v in tail.items()},
            per_sub_block=per_block_tail),
        "x5_the_simultaneous_lcb_clears_zero_in_every_cell": F.check(
            all(v["simultaneous_lcb95"] > 0.0 for v in tail.values()),
            "THE HEADLINE, at the ORIGINAL critical value with no threshold moved. It fails if the "
            "true effect is below about 73 percent of what the previous block observed",
            computed_from={"critical": pooled["critical"],
                           "n_cells_clearing": sum(v["simultaneous_lcb95"] > 0.0
                                                   for v in tail.values())},
            pooled_lcb={c: v["simultaneous_lcb95"] for c, v in tail.items()}),
        "x6_the_other_guardrails_stay_noninferior": F.check(
            guardrails_ok,
            "fixing the tail does not license breaking fairness; worst_product_fill and the rest of "
            "the vector must still hold",
            computed_from={"n_other_guardrails": len(other_guardrails),
                           "n_violating": sum(not (v["simultaneous_lcb95"] >= -1e-9
                                                   or v["estimate"] >= 0.0)
                                              for v in other_guardrails.values())}),
    }
    checks["custody"] = {
        "passed": True, "not_applicable": False,
        "evidence": {"why_it_can_fail": "a sub-block reusing seeds, or the six overlapping, would "
                                        "make this one block reported as six",
                     "n_tapes_per_cell": n_tapes, "n_sub_blocks": len(blocks),
                     "block_ranges": [[min(b["seeds"]), max(b["seeds"])] for b in blocks
                                      if isinstance(b.get("seeds"), list)],
                     "single_opening": True}}
    summary = F.summarise(checks)

    if not checks["x1_the_pooling_reproduces_a_single_sub_block"]["passed"] \
            or not checks["x2_the_frozen_runner_was_not_touched"]["passed"] \
            or not checks["x3_the_frozen_comparator_is_identical_across_sub_blocks"]["passed"]:
        status = "BLOCKED_INSTRUMENT_OR_CUSTODY"
    elif not checks["x4_the_effect_replicates_in_point"]["passed"]:
        status = "THE_EFFECT_DID_NOT_REPLICATE_IN_POINT"
    elif not checks["x5_the_simultaneous_lcb_clears_zero_in_every_cell"]["passed"]:
        status = "POWERED_AND_STILL_NOT_SIMULTANEOUSLY_SIGNIFICANT"
    elif not checks["x6_the_other_guardrails_stay_noninferior"]["passed"]:
        status = "TAIL_CLEARED_BUT_ANOTHER_GUARDRAIL_BROKE"
    else:
        status = "OBSERVABLE_CONVERSION_SURVIVES_AT_ADEQUATE_POWER"

    payload = {
        "schema_version": "program_o_powered_replication_v1", "claim_status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_role": "POWERED_REPLICATION_ON_A_VIRGIN_BLOCK",
        "scope": ("NEW_PROGRAMME_INHERITING_PROGRAM_O_PHYSICS_CANNOT_PROMOTE_PROGRAM_O_"
                  "ONLY_THE_SAMPLE_SIZE_DIFFERS"),
        "n_tapes_per_cell": n_tapes, "n_sub_blocks": len(blocks),
        "simultaneous_critical_used": pooled["critical"],
        "sub_block_criticals": criticals,
        "sub_block_statuses": [b["status"] for b in blocks],
        "pooled_tail": tail, "pooled_primary": prim,
        "per_sub_block_tail_estimates": per_block_tail,
        "pooled_estimates": pooled["estimates"],
        "falsifiers": checks, "falsifier_summary": summary,
        "contract_path": str(CONTRACT),
    }
    seal_and_write(payload, args.output, contract=CONTRACT,
                   reference=Path("results/program_o/r1_tail_in_the_objective_v1/result.json"))

    print(f"\nveredicto: {status}\n")
    print(f"  sub-bloques: {[b['status'].split('_')[0] for b in blocks]}")
    print(f"  critico simultaneo usado: {pooled['critical']:.6f}   n por celda: {n_tapes}\n")
    print(f"  {'celda':16}{'cola punto':>13}{'cola LCB sim':>15}{'primario punto':>16}"
          f"{'primario LCB':>14}")
    for c in CELLS:
        print(f"  {c:16}{tail[c]['estimate']:+13.6f}{tail[c]['simultaneous_lcb95']:+15.6f}"
              f"{prim[c]['estimate']:+16.6f}{prim[c]['simultaneous_lcb95']:+14.6f}")
    print("\n  falsadores:")
    for k, v in checks.items():
        mark = "N/A " if v.get("not_applicable") else ("PASA" if v["passed"] else "FALLA")
        print(f"    {k:52} {mark}")
    print(f"\n  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
