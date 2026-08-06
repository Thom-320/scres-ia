#!/usr/bin/env python3
"""v4: the edge check declared against the transform that decides, and the floor reported as a curve.

TWO REPAIRS OVER v3.

The first is narrow. v3's f6 asked whether the best-by-LCB transform sat on a grid edge; it did
(power(gamma=100)), while the transform that actually decided the verdict -- the best QUALIFYING
one, power(gamma=21.5) -- was interior. I refused to redefine the falsifier after seeing the
result, so it is declared here, before the run, against the deciding transform, with the power grid
widened from 0.01-100 to 0.001-1000 so that staying interior means something.

The second is the one that matters. H_regime rises monotonically in gamma with no interior maximum
-- 0.0195 at identity, 0.293 at 21.5, 0.632 at 100 -- so the only thing bounding the answer is the
signal floor, and the 0.90 floor is a constant I picked. Reporting one number conditioned on my own
arbitrary constant is the same degree of freedom this project polices elsewhere. The repair is to
stop choosing: H*(floor) is reported as a CURVE over five floors, and f8 fails if that curve turns
out flat, in which case the objection was unfounded.

AND THE ANCHOR THAT WAS MISSING. The adoption rule says a transform may only be adopted on declared
mechanism. Exactly one curvature in this whole discussion has one, and it is Garrido's: his
published index is sigma(sum sign*a_x*ln x), which IS our identity, gamma = 1. So the sentence that
depends on no floor at all is that under HIS declared curvature H_regime is 0.0195, below the bar,
and the headroom exists only under curvature he did not declare.

Preregistration: docs/PREREGISTRO_TECHO_MONOTONO_V4_BORDE_Y_PISO_2026-08-06.md
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
    PLANTED_H, exact_bootstrap, holm, load_per_seed, planted_surface, weighted_quantile,
)
from run_monotone_transform_family_v3 import (  # noqa: E402
    PAIR_RNG_SEED, PAIR_SAMPLE, evaluate, pair_index,
)

GATE = 0.05
REFERENCE_FLOOR = 0.90
FLOOR_CURVE = (0.80, 0.85, 0.90, 0.95, 0.99)
POWER_GRID = (0.001, 1000.0, 61)      # v3 used (0.01, 100.0, 31)
LOGISTIC_BETA = (0.05, 500.0, 20)
K_DECLARED = 661
MODULES = ("supply_chain/cobb_douglas_resilience.py", "supply_chain/arm_runner.py",
           "supply_chain/seed_custody.py")


def build_family(pooled: np.ndarray) -> list[dict]:
    lo, hi = float(pooled.min()), float(pooled.max())
    scale = float(pooled.std()) or 1.0
    fam = [{"kind": "identity"}]                       # == Garrido's published parametrisation
    for t in np.quantile(pooled, np.linspace(0.02, 0.98, 25)):
        for beta in np.geomspace(*LOGISTIC_BETA):
            fam.append({"kind": "logistic", "t": float(t), "beta": float(beta), "scale": scale})
    for gamma in np.geomspace(*POWER_GRID):
        fam.append({"kind": "power", "gamma": float(gamma), "lo": lo, "hi": hi})
    for t in np.quantile(pooled, np.linspace(0.01, 0.99, 99)):
        fam.append({"kind": "step", "t": float(t)})
    return fam


def on_boundary(spec: dict) -> bool:
    if spec["kind"] == "power":
        return bool(abs(spec["gamma"] - POWER_GRID[0]) < 1e-12
                    or abs(spec["gamma"] - POWER_GRID[1]) < 1e-9)
    if spec["kind"] == "logistic":
        return bool(abs(spec["beta"] - LOGISTIC_BETA[0]) < 1e-12
                    or abs(spec["beta"] - LOGISTIC_BETA[1]) < 1e-9)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/monotone_transform_family_v4/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()

    grids, all_seeds = {}, set()
    for grid in GRIDS:
        raw, contexts, seeds = load_per_seed(grid)
        all_seeds.update(seeds)
        sealed = json.loads((GRIDS[grid] / "result.json").read_text())["scalar_h_regime"]
        family = build_family(raw.mean(axis=1).ravel())
        idx = pair_index(raw.shape[2], raw.shape[0])
        print(f"\n  {grid}: {raw.shape[0]} ctx x {raw.shape[1]} semillas "
              f"x {raw.shape[2]:,} configuraciones · K={len(family)}", flush=True)

        rows = [evaluate(spec, raw, idx) for spec in family]
        for row, p in zip(rows, holm([r["p_not_above_gate"] for r in rows])):
            row["holm_adjusted_p"] = p
        for row, spec in zip(rows, family):
            row["on_grid_boundary"] = on_boundary(spec)
        ident = next(r for r in rows if r["label"] == "identity")

        stat_pass = [r for r in rows if r["lcb95"] >= GATE and r["holm_adjusted_p"] < 0.05]

        # The floor is not chosen. It is swept, and the whole curve is the result.
        curve = {}
        for floor in FLOOR_CURVE:
            cut = floor * ident["resolvable"]
            keep = [r for r in stat_pass if r["resolvable"] >= cut]
            best = max(keep, key=lambda r: r["lcb95"], default=None)
            curve[f"{floor:.2f}"] = {
                "resolvable_cut": cut, "n_qualifying": len(keep),
                "best_H_regime": best["H_regime"] if best else None,
                "best_lcb95": best["lcb95"] if best else None,
                "best_label": best["label"] if best else None,
                "best_on_grid_boundary": bool(best["on_grid_boundary"]) if best else None,
            }

        ref = curve[f"{REFERENCE_FLOOR:.2f}"]
        deciding = max((r for r in stat_pass
                        if r["resolvable"] >= REFERENCE_FLOOR * ident["resolvable"]),
                       key=lambda r: r["lcb95"], default=None)
        rejected_on_signal_only = [r for r in stat_pass
                                   if r["resolvable"] < REFERENCE_FLOOR * ident["resolvable"]]

        steps = [r for r in rows if r["label"].startswith("step")]
        sharp = min(steps, key=lambda r: r["resolvable"])
        proxy_falls = sharp["resolvable"] < 0.5 * ident["resolvable"]

        planted, weight, planted_h = planted_surface(raw, PLANTED_H)
        pv, pw = exact_bootstrap(planted)
        planted_lcb = weighted_quantile(pv, pw, 0.025)

        print(f"    Garrido (identidad, γ=1)  H {ident['H_regime']:+.6f} "
              f"(sellado {sealed:+.6f})  resolubles {ident['resolvable']:.4f}")
        print(f"    proxy    identidad {ident['resolvable']:.4f} -> escalón "
              f"{sharp['resolvable']:.4f}  {'CAE' if proxy_falls else 'NO CAE'}")
        print(f"    potencia plantada H {planted_h:.4f} -> LCB {planted_lcb:+.6f}  "
              f"{'SUFICIENTE' if planted_lcb >= GATE else 'INSUFICIENTE'}")
        print("    curva H*(piso):")
        for floor, c in curve.items():
            tag = "" if c["best_label"] is None else (
                f"  {c['best_label']}{'  <-- BORDE' if c['best_on_grid_boundary'] else ''}")
            h = "     ---" if c["best_H_regime"] is None else f"{c['best_H_regime']:+.6f}"
            lcb = "   ---" if c["best_lcb95"] is None else f"{c['best_lcb95']:+.6f}"
            print(f"      piso {floor}  n={c['n_qualifying']:<3} H* {h}  LCB {lcb}{tag}")

        grids[grid] = {
            "contexts": contexts, "seeds": seeds, "n_configurations": int(raw.shape[2]),
            "k_family": len(family), "sealed_scalar_h_regime": float(sealed),
            "garrido_declared_curvature": {
                "what": "his published sigma(sum sign*a_x*ln x) IS the identity here, gamma = 1",
                "H_regime": ident["H_regime"], "resolvable": ident["resolvable"],
                "above_gate": bool(ident["H_regime"] >= GATE)},
            "identity": ident, "floor_curve": curve, "reference_floor": REFERENCE_FLOOR,
            "deciding_transform": deciding,
            "n_passing_lcb_and_holm": len(stat_pass),
            "n_rejected_on_signal_only": len(rejected_on_signal_only),
            "signal_proxy_falls_on_a_step": bool(proxy_falls), "sharpest_step": sharp,
            "power": {"target_H": PLANTED_H, "achieved_H": planted_h, "mix_weight": weight,
                      "lcb95": planted_lcb, "sufficient": bool(planted_lcb >= GATE)},
            "rows": rows,
        }

    ext = grids["wrap288_compat_extended_v1"]
    n_stat = sum(g["n_passing_lcb_and_holm"] for g in grids.values())
    n_qual = sum(g["floor_curve"][f"{REFERENCE_FLOOR:.2f}"]["n_qualifying"]
                 for g in grids.values())
    proxy_ok = all(g["signal_proxy_falls_on_a_step"] for g in grids.values())

    if not ext["power"]["sufficient"]:
        verdict = "UNDERPOWERED_NO_VERDICT"
    elif not proxy_ok:
        verdict = ("SURVIVES_LCB_AND_MULTIPLICITY__SIGNAL_CRITERION_VOID" if n_stat
                   else "NO_MONOTONE_RESCALING_SURVIVES__SIGNAL_CRITERION_VOID")
    elif n_qual:
        verdict = "A_MONOTONE_RESCALING_SURVIVES_ALL_THREE"
    elif n_stat:
        verdict = "SURVIVES_LCB_AND_MULTIPLICITY_BUT_COSTS_SIGNAL"
    else:
        verdict = "NO_MONOTONE_RESCALING_SURVIVES"

    c80 = ext["floor_curve"]["0.80"]["best_H_regime"]
    c99 = ext["floor_curve"]["0.99"]["best_H_regime"]
    falsifiers = {
        "f1_identity_reproduces_the_sealed_scalar": {
            "passed": all(abs(g["identity"]["H_regime"] - g["sealed_scalar_h_regime"]) < 1e-9
                          for g in grids.values()),
            "evidence": {"why_it_can_fail": "external anchor; drift in the rebuilt index makes this "
                                            "incomparable with the sealed numbers",
                         "deviations": {k: abs(g["identity"]["H_regime"]
                                               - g["sealed_scalar_h_regime"])
                                        for k, g in grids.items()}}},
        "f2_the_signal_proxy_can_actually_fall": {
            "passed": proxy_ok,
            "evidence": {"why_it_can_fail": "two earlier proxies passed without being able to fail; "
                                            "a step must collapse this one on BOTH grids",
                         "identity": {k: g["identity"]["resolvable"] for k, g in grids.items()},
                         "sharpest_step": {k: g["sharpest_step"]["resolvable"]
                                           for k, g in grids.items()}}},
        "f3_the_instrument_has_power": {
            "passed": bool(ext["power"]["sufficient"]),
            "evidence": {"why_it_can_fail": "a ten-atom bootstrap that cannot see a planted H = "
                                            "0.10 makes a null uninterpretable", **ext["power"]}},
        "f4_multiplicity_over_the_declared_family": {
            "passed": all(g["k_family"] == K_DECLARED for g in grids.values()),
            "evidence": {"why_it_can_fail": "the preregistration fixes K at 661",
                         "k_declared": K_DECLARED,
                         "k_per_grid": {k: g["k_family"] for k, g in grids.items()}}},
        "f5_the_base_grid_stays_at_zero": {
            "passed": grids["wrap288_v1"]["n_passing_lcb_and_holm"] == 0,
            "evidence": {"why_it_can_fail": "negative control: where one configuration is optimal "
                                            "in all six regimes, an instrument that finds headroom "
                                            "is manufacturing it",
                         "n_passing": grids["wrap288_v1"]["n_passing_lcb_and_holm"]}},
        "f6_the_deciding_transform_is_interior": {
            "passed": bool(ext["deciding_transform"] is not None
                           and not ext["deciding_transform"]["on_grid_boundary"]),
            "evidence": {"why_it_can_fail": "declared BEFORE the run against the transform that "
                                            "decides the verdict, not the global best. The power "
                                            "grid is three orders of magnitude wide; if the "
                                            "deciding transform still lands on an edge the family "
                                            "is truncated where it matters",
                         "deciding": (ext["deciding_transform"] or {}).get("label"),
                         "power_grid": list(POWER_GRID)}},
        "f7_the_signal_floor_actually_binds": {
            "passed": ext["n_rejected_on_signal_only"] > 0,
            "evidence": {"why_it_can_fail": "if no transform is ever rejected by the signal floor "
                                            "alone, the floor is decoration and the third hurdle "
                                            "is not a hurdle",
                         "n_rejected_on_signal_only": ext["n_rejected_on_signal_only"]}},
        "f8_the_answer_depends_on_the_floor": {
            "passed": bool(c80 is not None and c99 is not None and abs(c80 - c99) > 1e-6),
            "evidence": {"why_it_can_fail": "I objected that the reported number is hostage to a "
                                            "floor I chose. If H*(0.80) equals H*(0.99) the "
                                            "objection was unfounded and must be withdrawn",
                         "H_at_floor_080": c80, "H_at_floor_099": c99}},
        "f9_no_fresh_seeds": custody_falsifier(sorted(all_seeds), replay_of=args.replay_of,
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
        print(f"    {name:<48} {lab}")

    payload = {
        "schema_version": "monotone_transform_family_v4",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION_NO_ADOPTION",
        "run_role": "CACHE_ANALYSIS", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": "docs/PREREGISTRO_TECHO_MONOTONO_V4_BORDE_Y_PISO_2026-08-06.md",
        "predecessor": "results/monotone_transform_family_v3/result.json",
        "gate": GATE, "reference_floor": REFERENCE_FLOOR, "floor_curve_points": list(FLOOR_CURVE),
        "k_family": K_DECLARED, "power_grid": list(POWER_GRID),
        "signal_proxy": {"definition": "fraction of configuration pairs with "
                                       "|m_i - m_j| > sqrt(s_i^2 + s_j^2)",
                         "pair_sample": PAIR_SAMPLE, "rng_seed": PAIR_RNG_SEED},
        "the_sentence_that_needs_no_floor": (
            "Garrido's published index is sigma(sum sign*a_x*ln x), which is the identity here. "
            "Under the only curvature anyone has declared, H_regime on the extended grid is "
            "0.0195 and on the 288 grid it is exactly 0 -- both below the 0.05 gate. The headroom "
            "reported by this family exists only under curvature he did not declare, and the "
            "amount of it is a function of a signal floor rather than of the supply chain."),
        "grids": grids, "falsifiers": falsifiers,
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/monotone_transform_family_v3/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
