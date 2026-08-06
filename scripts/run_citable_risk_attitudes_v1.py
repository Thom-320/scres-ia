#!/usr/bin/env python3
"""Does any citable risk attitude land in the qualifying set, or only risk-seeking curvature?

THE STRUCTURAL FACT THIS TURNS ON. power(gamma) with gamma > 1 is convex, and a convex utility
over outcomes is RISK-SEEKING. Every risk attitude in the supply-chain risk literature is averse --
concave -- and every concave utility lands at gamma <= 1. v4's qualifying set lives at gamma 3-32.
So if H is monotone in curvature, the entire concave side sits at or below the identity, which is
already below the gate, and the lane closes without anyone having to agree on which eta a
particular paper uses.

That argument can die in two places, and both are falsifiers here: f3 fails if any concave attitude
beats the identity, and f4 fails if H is not monotone in gamma. f2 is the could-have-detected
control: a convex arm that MUST qualify, or the run says nothing about the citable attitudes at
all.

CVaR IS NOT IN THE SAME FAMILY. It replaces mean_r with a lower-tail mean, which changes the
ESTIMATOR rather than the metric, so it is measured and reported in its own arm and never mixed
into the transform verdict.

THE COEFFICIENTS ARE STANDARD RANGES, NOT VERIFIED CITATIONS. Nothing here adopts one as "the
literature's value"; the weight of the result rests on f3 and f4, which are measurements on our own
surface. Every coefficient must be checked against its source before the manuscript.

Preregistration: docs/PREREGISTRO_ACTITUDES_DE_RIESGO_CITABLES_2026-08-06.md
Development on burned tapes. Adjudicates nothing, adopts nothing, changes no primary endpoint.
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
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_monotone_transform_ceiling_v1 import GRIDS, h_regime  # noqa: E402
from run_monotone_transform_family_v2 import (  # noqa: E402
    exact_bootstrap, holm, load_per_seed, weighted_quantile,
)
from run_monotone_transform_family_v3 import pair_index, resolvable  # noqa: E402

GATE = 0.05
SIGNAL_FLOOR = 0.90
GRID = "wrap288_compat_extended_v1"     # the 288 grid is curvature-proof; nothing to test there
MONOTONICITY_GRID = (0.001, 1000.0, 61)
MODULES = ("supply_chain/cobb_douglas_resilience.py", "supply_chain/arm_runner.py",
           "supply_chain/seed_custody.py")

CRRA_ETAS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
CARA_AS = (0.5, 1.0, 2.0, 5.0, 10.0)
SEEKING_GAMMAS = (2.0, 5.0, 10.0, 20.0, 30.0)
CVAR_ALPHAS = (0.90, 0.95, 0.99)


FLOOR = 1e-12


def unit(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Utilities are defined on outcomes in [0,1]; the index is put on that scale first so that
    eta and a mean what they mean in the literature rather than what our raw range makes them.

    lo/hi come from the PER-SEED values, not from the seed-averaged surface. The first version took
    them from the seed average and clipped, which put per-seed values outside [0,1] onto the
    boundary -- and clipping is not affine, so u(x)=x stopped being the identity and f1 caught it
    against the sealed scalar. No clipping is needed once the range is the true one."""
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def attitudes(lo: float, hi: float) -> list[dict]:
    """Every entry is a POINTWISE utility, so it maps exactly onto the monotone-transform family."""
    out = [{"arm": "risk_neutral", "label": "risk_neutral (u=x)  == Garrido's published curvature",
            "curvature": "linear", "f": lambda x: unit(x, lo, hi)}]
    for eta in CRRA_ETAS:
        if abs(eta - 1.0) < 1e-12:
            out.append({"arm": "averse", "label": f"CRRA(eta={eta:g})  u=ln x",
                        "curvature": "concave", "eta": eta,
                        # ln is monotone; the affine shift keeps it finite at 0 without changing
                        # any ordering, which is all H_regime and the pair test can see.
                        "f": lambda x, e=eta: np.log(np.maximum(unit(x, lo, hi), FLOOR))})
        else:
            # eta > 1 makes the exponent negative, so an outcome at exactly 0 returns inf and the
            # normalisation then returns NaN. The floor keeps it finite; it is monotone, so it
            # changes no ordering and neither H_regime nor the pair test can see it.
            out.append({"arm": "averse", "label": f"CRRA(eta={eta:g})",
                        "curvature": "concave", "eta": eta,
                        "f": lambda x, e=eta: (np.maximum(unit(x, lo, hi), FLOOR) ** (1.0 - e)
                                               / (1.0 - e))})
    for a in CARA_AS:
        out.append({"arm": "averse", "label": f"CARA(a={a:g})", "curvature": "concave", "a": a,
                    "f": lambda x, aa=a: (1.0 - np.exp(-aa * unit(x, lo, hi))) / aa})
    for g in SEEKING_GAMMAS:
        out.append({"arm": "seeking_control", "label": f"risk-seeking power(gamma={g:g})",
                    "curvature": "convex", "gamma": g,
                    "f": lambda x, gg=g: unit(x, lo, hi) ** gg})
    return out


def cvar_h_regime(per_ctx: np.ndarray, alpha: float) -> float:
    """CVaR over REGIMES replaces mean_r in the second term: a planner who fears the worst regimes
    picks the configuration maximising the lower-tail mean instead of the average. This is an
    estimator change, not a metric transform, which is exactly why it is reported apart."""
    norm = []
    for m in np.asarray(per_ctx, dtype=float):
        lo, hi = m.min(), m.max()
        norm.append((m - lo) / (hi - lo) if hi > lo else np.zeros_like(m))
    stacked = np.stack(norm)                                   # (contexts, configs)
    k = max(1, int(np.ceil((1.0 - alpha) * stacked.shape[0])))
    tail = np.sort(stacked, axis=0)[:k].mean(axis=0)           # worst-k mean per configuration
    return float(stacked.max(axis=1).mean() - tail.max())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/citable_risk_attitudes/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()

    raw, contexts, seeds = load_per_seed(GRID)
    sealed = json.loads((GRIDS[GRID] / "result.json").read_text())["scalar_h_regime"]
    idx = pair_index(raw.shape[2], raw.shape[0])
    lo, hi = float(raw.min()), float(raw.max())   # per-seed range: see unit()
    fam = attitudes(lo, hi)
    print(f"\n  {GRID}: {raw.shape[0]} ctx x {raw.shape[1]} semillas "
          f"x {raw.shape[2]:,} configuraciones · {len(fam)} actitudes")

    rows = []
    for spec in fam:
        tx = spec["f"](raw)
        values, weights = exact_bootstrap(tx)
        rows.append({k: v for k, v in spec.items() if k != "f"} | {
            "H_regime": h_regime(tx.mean(axis=1)),
            "lcb95": weighted_quantile(values, weights, 0.025),
            "p_not_above_gate": float(weights[values <= GATE].sum()),
            "resolvable": resolvable(tx, idx)})
    for row, p in zip(rows, holm([r["p_not_above_gate"] for r in rows])):
        row["holm_adjusted_p"] = p
    neutral = next(r for r in rows if r["arm"] == "risk_neutral")
    floor = SIGNAL_FLOOR * neutral["resolvable"]
    for row in rows:
        row["qualifies"] = bool(row["lcb95"] >= GATE and row["holm_adjusted_p"] < 0.05
                                and row["resolvable"] >= floor)

    # H against curvature over the full grid: the monotonicity the structural argument needs.
    curve = []
    for g in np.geomspace(*MONOTONICITY_GRID):
        curve.append({"gamma": float(g),
                      "H_regime": h_regime((unit(raw, lo, hi) ** g).mean(axis=1))})
    hs = [c["H_regime"] for c in curve]
    monotone = all(b >= a - 1e-12 for a, b in zip(hs, hs[1:]))

    cvar = [{"alpha": a, "H_regime_cvar": cvar_h_regime(raw.mean(axis=1), a)}
            for a in CVAR_ALPHAS]

    averse = [r for r in rows if r["arm"] == "averse"]
    control = [r for r in rows if r["arm"] == "seeking_control"]
    averse_qual = [r for r in averse if r["qualifies"]]
    control_qual = [r for r in control if r["qualifies"]]

    print(f"\n    neutral al riesgo (= Garrido)  H {neutral['H_regime']:+.6f} "
          f"(sellado {sealed:+.6f})  resolubles {neutral['resolvable']:.4f}")
    print(f"\n    AVERSAS (cóncavas) — piso de señal {floor:.4f}:")
    for r in averse:
        print(f"      H {r['H_regime']:+.6f}  LCB {r['lcb95']:+.6f}  resol {r['resolvable']:.4f}"
              f"  {'CALIFICA' if r['qualifies'] else '        '}  {r['label']}")
    print(f"\n    CONTROL amante del riesgo (convexas):")
    for r in control:
        print(f"      H {r['H_regime']:+.6f}  LCB {r['lcb95']:+.6f}  resol {r['resolvable']:.4f}"
              f"  {'CALIFICA' if r['qualifies'] else '        '}  {r['label']}")
    print(f"\n    CVaR sobre regímenes (cambia el ESTIMADOR, reportado aparte):")
    for c in cvar:
        print(f"      alpha {c['alpha']:.2f}  H {c['H_regime_cvar']:+.6f}")
    print(f"\n    H(gamma) monótona sobre 0,001–1000: {'SÍ' if monotone else 'NO'}"
          f"   (H(0,001)={hs[0]:+.6f}  H(1000)={hs[-1]:+.6f})")

    if not control_qual:
        verdict = "INSTRUMENT_DETECTS_NOTHING"
    elif averse_qual:
        verdict = "A_CITABLE_RISK_ATTITUDE_REACHES_THE_BAR"
    else:
        verdict = "ONLY_RISK_SEEKING_CURVATURE_REACHES_THE_BAR"

    falsifiers = {
        "f1_risk_neutral_reproduces_the_sealed_scalar": {
            "passed": abs(neutral["H_regime"] - float(sealed)) < 1e-9,
            "evidence": {"why_it_can_fail": "u(x)=x is an affine rescale, which H_regime is "
                                            "invariant to; if it does not reproduce the sealed "
                                            "scalar the rebuild has drifted",
                         "deviation": abs(neutral["H_regime"] - float(sealed))}},
        "f2_the_control_qualifies": {
            "passed": bool(control_qual),
            "evidence": {"why_it_can_fail": "if no convex control qualifies, this test cannot "
                                            "detect ANY attitude and its silence about the citable "
                                            "ones means nothing",
                         "n_control_qualifying": len(control_qual),
                         "qualifying": [r["label"] for r in control_qual]}},
        "f3_concave_attitudes_land_below_the_identity": {
            "passed": all(r["H_regime"] <= neutral["H_regime"] + 1e-9 for r in averse),
            "evidence": {"why_it_can_fail": "the whole structural argument is that risk aversion is "
                                            "concave and concavity cannot raise H. One averse "
                                            "attitude above the identity refutes it and the "
                                            "argument must be withdrawn",
                         "max_averse_H": max(r["H_regime"] for r in averse),
                         "identity_H": neutral["H_regime"]}},
        "f4_H_is_monotone_in_curvature": {
            "passed": bool(monotone),
            "evidence": {"why_it_can_fail": "reasoning by sides -- concave below, convex above -- "
                                            "requires H to be monotone in gamma. If it is not, a "
                                            "concave attitude could sit above the identity "
                                            "somewhere we did not sample",
                         "grid": list(MONOTONICITY_GRID),
                         "H_at_min_gamma": hs[0], "H_at_max_gamma": hs[-1]}},
        "f5_cvar_is_reported_separately": {
            "passed": all(r["arm"] in ("risk_neutral", "averse", "seeking_control") for r in rows),
            "evidence": {"why_it_can_fail": "CVaR replaces mean_r rather than transforming the "
                                            "metric; letting it into the transform verdict would "
                                            "be a category error",
                         "cvar_rows": cvar}},
        "f6_no_fresh_seeds": custody_falsifier(sorted(seeds), replay_of=args.replay_of,
                                               exclude=args.output),
    }
    falsifiers["all_passed"] = all(
        v["passed"] for k, v in falsifiers.items()
        if k != "all_passed" and isinstance(v, dict) and not v.get("not_applicable"))

    print(f"\n  veredicto: {verdict}\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        lab = "NO APLICA" if f.get("not_applicable") else ("PASA" if f["passed"] else "FALLA")
        print(f"    {name:<52} {lab}")

    payload = {
        "schema_version": "citable_risk_attitudes_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION_NO_ADOPTION",
        "run_role": "CACHE_ANALYSIS", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": "docs/PREREGISTRO_ACTITUDES_DE_RIESGO_CITABLES_2026-08-06.md",
        "predecessor": "results/monotone_transform_family_v4/result.json",
        "grid": GRID, "gate": GATE, "signal_floor": SIGNAL_FLOOR,
        "coefficients_are_ranges_not_citations": (
            "The etas, a's and gammas here are standard ranges, not verified citations, and none "
            "is adopted as 'the literature's value'. The result rests on f3 and f4, which are "
            "measurements on our own surface. Every coefficient cited in the manuscript must be "
            "checked against its source first."),
        "risk_neutral_is_garrido": (
            "u(x) = x is Garrido's own published curvature: his index is already "
            "sigma(sum sign*a_x*ln x), so the identity IS his declared risk attitude."),
        "attitudes": rows, "curvature_curve": curve, "H_is_monotone_in_gamma": bool(monotone),
        "cvar_over_regimes": cvar, "contexts": contexts, "seeds": seeds,
        "falsifiers": falsifiers, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/monotone_transform_family_v4/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
