#!/usr/bin/env python3
"""Recompute the confirmation power under the CONTRACTUAL context order.

The confirmation contract sized its virgin block from `results/grid_transfer_v2/result.json`, which
was computed with the career ordered alphabetically by directory name rather than by the order the
contract declares. For an arm that carries state the context order IS the career, so the paired SD
that sized `n = 60` was measured on a career no contract ever declared.

WHAT IS ALLOWED TO MOVE, AND WHAT IS NOT. The design effect `delta* = 0.015` was fixed BEFORE the
block was reserved and it does NOT move here. Letting it drift toward whatever the corrected mean
turns out to be would make the power calculation circular -- the sample size would be chosen to
detect exactly the effect already observed, which guarantees adequacy by construction. Only the
paired SD is re-measured, because only the SD was mis-measured.

This is a recomputation of a declared calculation on corrected inputs, not a new estimand. It
adjudicates nothing and authorises nothing: the registry still says `new_seed_opening: false`.

Contract: docs/PREREGISTRO_CONFIRMACION_TRANSFERENCIA_REJILLA_2026-08-05.md, section 4
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402

#: Fixed in the contract before the block was reserved. It does not move.
DESIGN_EFFECT = 0.015
ALPHA = 0.05
TARGET_POWER = 0.86            # what the contract reported for n = 60 under the old SD
BLOCK_N = 60
MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")


def one_sided_power(delta: float, sd: float, n: int, alpha: float = ALPHA) -> float:
    """Normal approximation to the one-sided paired t-test power.

    The contract itself calls its number "aproximada", so this stays on the same footing rather
    than quietly upgrading the method while the inputs change.
    """
    from statistics import NormalDist
    if sd <= 0 or n < 2:
        return float("nan")
    z_alpha = NormalDist().inv_cdf(1.0 - alpha)
    return float(NormalDist().cdf(delta * math.sqrt(n) / sd - z_alpha))


def required_n(delta: float, sd: float, power: float, alpha: float = ALPHA) -> int:
    from statistics import NormalDist
    z_a, z_b = NormalDist().inv_cdf(1.0 - alpha), NormalDist().inv_cdf(power)
    return int(math.ceil(((z_a + z_b) * sd / delta) ** 2))


def paired_delta(payload: dict, arm: str, kind: str) -> np.ndarray:
    """delta_M = AUC_marginal - AUC_transfer, per seed. Lower AUC is better, so positive is good."""
    per_arm = payload["per_arm"]
    transfer = np.asarray(per_arm[f"{arm}_transfer"], dtype=float)
    other = np.asarray(per_arm[f"{arm}_{kind}"], dtype=float)
    if transfer.shape != other.shape:
        raise SystemExit(f"{arm}: {transfer.shape} transfer vs {other.shape} {kind}")
    return other - transfer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ordered", type=Path,
                    default=Path("results/grid_transfer_ordered_v1/result.json"))
    ap.add_argument("--superseded", type=Path,
                    default=Path("results/grid_transfer_v2/result.json"))
    ap.add_argument("--arm", default="ucb1")
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/custody/grid_transfer_confirmation_repower/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()

    ordered = json.loads(args.ordered.read_text())
    old = json.loads(args.superseded.read_text())

    rows = {}
    for label, payload in (("superseded_alphabetical", old), ("contractual_order", ordered)):
        d_m = paired_delta(payload, args.arm, "marginal")
        d_c = paired_delta(payload, args.arm, "cold")
        rows[label] = {
            "n_seeds": int(d_m.size),
            "delta_M_mean": float(d_m.mean()), "delta_M_sd": float(d_m.std(ddof=1)),
            "delta_C_mean": float(d_c.mean()), "delta_C_sd": float(d_c.std(ddof=1)),
            "power_at_block_n": one_sided_power(DESIGN_EFFECT, float(d_m.std(ddof=1)), BLOCK_N),
            "required_n_for_target": required_n(DESIGN_EFFECT, float(d_m.std(ddof=1)),
                                                TARGET_POWER),
        }

    new, prev = rows["contractual_order"], rows["superseded_alphabetical"]
    adequate = new["power_at_block_n"] >= TARGET_POWER
    verdict = ("BLOCK_SIZE_STILL_ADEQUATE_UNDER_CORRECTED_SD" if adequate
               else "BLOCK_SIZE_INSUFFICIENT_UNDER_CORRECTED_SD")

    checks = {
        "f1_design_effect_did_not_move": {
            "passed": DESIGN_EFFECT == 0.015,
            "evidence": {"why_it_can_fail": "if delta* were re-derived from the corrected mean the "
                                            "power calculation would be circular -- sized to detect "
                                            "exactly the effect already seen, adequate by "
                                            "construction. It was fixed before the block was "
                                            "reserved and is asserted against that value here",
                         "delta_star": DESIGN_EFFECT}},
        "f2_the_two_artifacts_really_differ": {
            "passed": abs(new["delta_M_sd"] - prev["delta_M_sd"]) > 1e-12,
            "evidence": {"why_it_can_fail": "if the ordered and alphabetical artifacts gave the "
                                            "identical paired SD, the ordering defect would not "
                                            "touch this estimand and re-powering would be theatre",
                         "sd_alphabetical": prev["delta_M_sd"],
                         "sd_contractual": new["delta_M_sd"]}},
        "f3_seed_counts_match": {
            "passed": new["n_seeds"] == prev["n_seeds"],
            "evidence": {"why_it_can_fail": "a different number of seeds would confound the SD "
                                            "change with a sample-size change",
                         "n": new["n_seeds"]}},
    }
    checks["all_passed"] = all(v["passed"] for k, v in checks.items() if k != "all_passed")

    print(f"  efecto de diseño fijado (NO se mueve): delta* = {DESIGN_EFFECT}")
    for label, r in rows.items():
        print(f"\n  {label}  (n = {r['n_seeds']})")
        print(f"    delta_M  media {r['delta_M_mean']:+.6f}   SD pareada {r['delta_M_sd']:.6f}")
        print(f"    potencia a n={BLOCK_N}: {r['power_at_block_n']:.3f}"
              f"   n para {TARGET_POWER}: {r['required_n_for_target']}")
    print(f"\n  veredicto: {verdict}\n")
    for name, c in checks.items():
        if name == "all_passed":
            continue
        print(f"    {name:<40} {'PASA' if c['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "grid_transfer_confirmation_repower_v1",
        "claim_status": verdict,
        "scope": "POWER_RECOMPUTATION_ONLY_NO_ADJUDICATION_NO_SEED_AUTHORISATION",
        "run_role": "CONTRACT_ARITHMETIC",
        "module_manifest": module_manifest(MODULES, script=__file__),
        "arm": args.arm, "design_effect": DESIGN_EFFECT, "alpha": ALPHA,
        "target_power": TARGET_POWER, "reserved_block_n": BLOCK_N,
        "sources": {"contractual_order": str(args.ordered),
                    "superseded_alphabetical": str(args.superseded)},
        "power": rows, "checks": checks,
        "authorisation_note": ("The registry still declares new_seed_opening: false. This artifact "
                               "sizes an experiment; it does not authorise one."),
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract, reference=args.ordered)
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if checks["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
