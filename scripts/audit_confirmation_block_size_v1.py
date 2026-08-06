#!/usr/bin/env python3
"""Is n = 26 actually the frugal choice, or is it the fragile one?

The re-powering used the paired SD as if it were known. It is not: it is an estimate from 12
development seeds, and a variance estimated from 12 observations is a noisy object. Sizing a
confirmation on the POINT estimate bets the whole block on that estimate being right.

The asymmetry decides it. Unused seeds are not lost -- they stay available to the next contract.
Seeds spent on an underpowered confirmation are gone AND produce nothing, because a virgin block
cannot be reopened to add power after the fact. So the sizing question is not "how few can we get
away with" but "what does this cost when the SD estimate is wrong in the bad direction".

This script prices that. It adjudicates nothing and authorises nothing.

Contract: docs/PREREGISTRO_CONFIRMACION_TRANSFERENCIA_REJILLA_2026-08-05.md, section 4
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402

DESIGN_EFFECT = 0.015          # fixed before the block was reserved; never re-derived
ALPHA = 0.05
TARGET_POWER = 0.86
CANDIDATES = (26, 40, 60, 72, 90)
MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")


def chi2_quantile(p: float, df: int) -> float:
    """Wilson-Hilferty approximation. Good to ~1e-3 in the relevant range, and stated as an
    approximation rather than dressed up as exact."""
    z = NormalDist().inv_cdf(p)
    return df * (1.0 - 2.0 / (9.0 * df) + z * math.sqrt(2.0 / (9.0 * df))) ** 3


def sd_confidence_band(sd_hat: float, n: int, conf: float = 0.95) -> tuple[float, float]:
    """A variance from `n` observations has (n-1)s^2/sigma^2 ~ chi2_{n-1}."""
    df = n - 1
    lo_q = chi2_quantile((1 + conf) / 2, df)      # large chi2 -> small sigma
    hi_q = chi2_quantile((1 - conf) / 2, df)
    return sd_hat * math.sqrt(df / lo_q), sd_hat * math.sqrt(df / hi_q)


def power(delta: float, sd: float, n: int, alpha: float = ALPHA) -> float:
    if sd <= 0 or n < 2:
        return float("nan")
    return float(NormalDist().cdf(delta * math.sqrt(n) / sd - NormalDist().inv_cdf(1 - alpha)))


def required_n(delta: float, sd: float, target: float, alpha: float = ALPHA) -> int:
    z_a, z_b = NormalDist().inv_cdf(1 - alpha), NormalDist().inv_cdf(target)
    return int(math.ceil(((z_a + z_b) * sd / delta) ** 2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repower", type=Path,
                    default=Path("results/custody/grid_transfer_confirmation_repower/result.json"))
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/custody/confirmation_block_size_audit/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()

    src = json.loads(args.repower.read_text())["power"]["contractual_order"]
    sd_hat, n_dev = src["delta_M_sd"], src["n_seeds"]
    sd_lo, sd_hi = sd_confidence_band(sd_hat, n_dev)

    grid = {}
    for n in CANDIDATES:
        grid[n] = {
            "power_at_sd_hat": power(DESIGN_EFFECT, sd_hat, n),
            "power_at_sd_upper": power(DESIGN_EFFECT, sd_hi, n),
            "power_at_sd_lower": power(DESIGN_EFFECT, sd_lo, n),
        }

    n_point = required_n(DESIGN_EFFECT, sd_hat, TARGET_POWER)
    n_robust = required_n(DESIGN_EFFECT, sd_hi, TARGET_POWER)

    #: The seeds a failed confirmation costs: all of them, with nothing to show. Unused seeds cost
    #: nothing -- they remain available. That asymmetry is the whole argument.
    verdict = ("KEEP_THE_RESERVED_BLOCK" if n_robust > 26 else "SHRINK_IS_SAFE")

    checks = {
        "f1_sd_is_treated_as_estimated_not_known": {
            "passed": sd_hi > sd_hat > sd_lo,
            "evidence": {"why_it_can_fail": "if the band collapsed onto the point estimate the "
                                            "audit would be asserting that 12 seeds pin a variance, "
                                            "which is the very assumption under test",
                         "sd_hat": sd_hat, "sd_95_band": [sd_lo, sd_hi], "n_development": n_dev}},
        "f2_design_effect_did_not_move": {
            "passed": DESIGN_EFFECT == 0.015,
            "evidence": {"why_it_can_fail": "re-deriving delta* from the observed mean makes every "
                                            "sample size adequate by construction",
                         "delta_star": DESIGN_EFFECT}},
        "f3_the_frugal_option_is_actually_evaluated": {
            "passed": 26 in grid,
            "evidence": {"why_it_can_fail": "an audit that only prices the option it prefers is "
                                            "advocacy; n = 26 is scored on the same footing",
                         "power_of_26_at_sd_upper": grid[26]["power_at_sd_upper"]}},
    }
    checks["all_passed"] = all(v["passed"] for k, v in checks.items() if k != "all_passed")

    print(f"  SD pareada estimada con {n_dev} semillas: {sd_hat:.6f}")
    print(f"  banda 95 % de la SD: [{sd_lo:.6f}, {sd_hi:.6f}]  (factor {sd_hi / sd_hat:.2f}x arriba)")
    print(f"\n  n   potencia@SD_est  potencia@SD_alta  potencia@SD_baja")
    for n, r in grid.items():
        print(f"  {n:<4}{r['power_at_sd_hat']:>13.3f}{r['power_at_sd_upper']:>18.3f}"
              f"{r['power_at_sd_lower']:>18.3f}")
    print(f"\n  n para {TARGET_POWER} con la SD estimada : {n_point}")
    print(f"  n para {TARGET_POWER} si la SD está en su límite alto: {n_robust}")
    print(f"\n  veredicto: {verdict}\n")
    for name, c in checks.items():
        if name != "all_passed":
            print(f"    {name:<44} {'PASA' if c['passed'] else 'FALLA'}")

    payload = {
        "schema_version": "confirmation_block_size_audit_v1",
        "claim_status": verdict,
        "scope": "DESIGN_ARITHMETIC_ONLY_NO_ADJUDICATION_NO_SEED_AUTHORISATION",
        "run_role": "CONTRACT_ARITHMETIC",
        "module_manifest": module_manifest(MODULES, script=__file__),
        "design_effect": DESIGN_EFFECT, "alpha": ALPHA, "target_power": TARGET_POWER,
        "sd_point_estimate": sd_hat, "sd_development_n": n_dev,
        "sd_95_band": [sd_lo, sd_hi],
        "power_by_n": grid,
        "required_n_point_estimate": n_point, "required_n_robust": n_robust,
        "asymmetry": ("Unused seeds remain available to the next contract. Seeds spent on an "
                      "underpowered confirmation are gone and yield nothing, because a virgin "
                      "block cannot be reopened to add power after the fact."),
        "checks": checks, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract, reference=args.repower)
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if checks["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
