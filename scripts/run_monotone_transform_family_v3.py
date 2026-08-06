#!/usr/bin/env python3
"""v3: resolvable pairs as the signal proxy, an interior grid, same LCB and multiplicity.

TWO PROXIES HAVE FAILED, BOTH BECAUSE THEY COULD NOT FALL. v1 used pairwise ordering, which no
strictly increasing map can disturb. v2 used a signal-to-noise ratio, which collapses its own
denominator: when a step flattens most of 4,608 configurations their between-seed spread goes to
zero along with the between-configuration spread, so the ratio held at 0.65x identity exactly where
it should have collapsed.

RESOLVABLE PAIRS CANNOT BE RESCUED THAT WAY. A pair counts as resolvable when the gap between its
seed-means exceeds the pairwise seed noise. Under a step, saturated pairs have equal means AND zero
spread, so the test reads 0 > 0 and the pair is correctly counted as unresolvable. That is the
property the two previous proxies lacked, and f2 is where it gets checked rather than assumed.

The power grid also widens from 0.1-10 to 0.01-100: v2's optimum was power(gamma=10), sitting on
the boundary of its own declared grid, and a maximum at a boundary is not a maximum. f6 fails if it
happens again.

Preregistration: docs/PREREGISTRO_TECHO_MONOTONO_V3_PARES_RESOLUBLES_2026-08-06.md
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
    PLANTED_H, apply_transform, exact_bootstrap, holm, label, load_per_seed, planted_surface,
    weighted_quantile,
)

GATE = 0.05
SIGNAL_FLOOR_FRACTION = 0.90
PAIR_SAMPLE = 200_000
PAIR_RNG_SEED = 20260806          # fixed in the preregistration, not chosen at run time
MODULES = ("supply_chain/cobb_douglas_resilience.py", "supply_chain/arm_runner.py",
           "supply_chain/seed_custody.py")

POWER_GRID = (0.01, 100.0, 31)     # v2 used (0.1, 10.0, 21) and its optimum sat on the edge


def build_family(pooled: np.ndarray) -> list[dict]:
    lo, hi = float(pooled.min()), float(pooled.max())
    scale = float(pooled.std()) or 1.0
    fam = [{"kind": "identity"}]
    for t in np.quantile(pooled, np.linspace(0.02, 0.98, 25)):
        for beta in np.geomspace(0.05, 500.0, 20):
            fam.append({"kind": "logistic", "t": float(t), "beta": float(beta), "scale": scale})
    for gamma in np.geomspace(*POWER_GRID):
        fam.append({"kind": "power", "gamma": float(gamma), "lo": lo, "hi": hi})
    for t in np.quantile(pooled, np.linspace(0.01, 0.99, 99)):
        fam.append({"kind": "step", "t": float(t)})
    return fam


def pair_index(n_cfg: int, n_ctx: int) -> tuple[np.ndarray, np.ndarray]:
    """One fixed pair subsample, drawn once and reused for every transform so that differences
    between transforms are differences in the transform and not in the sample."""
    rng = np.random.default_rng(PAIR_RNG_SEED)
    i = rng.integers(0, n_cfg, size=(n_ctx, PAIR_SAMPLE))
    j = rng.integers(0, n_cfg, size=(n_ctx, PAIR_SAMPLE))
    bump = (i == j)
    j = np.where(bump, (j + 1) % n_cfg, j)
    return i, j


def resolvable(per_seed: np.ndarray, idx: tuple[np.ndarray, np.ndarray]) -> float:
    """Fraction of configuration pairs distinguishable above the replication noise.

    |m_i - m_j| > sqrt(s_i^2 + s_j^2). Saturated pairs give 0 > 0, which is false -- that is the
    whole point, and the reason this can fall where the v2 ratio could not."""
    i_all, j_all = idx
    out = []
    for r in range(per_seed.shape[0]):
        block = per_seed[r]
        m = block.mean(axis=0)
        s = block.std(axis=0, ddof=1) if block.shape[0] > 1 else np.zeros_like(m)
        i, j = i_all[r], j_all[r]
        gap = np.abs(m[i] - m[j])
        noise = np.sqrt(s[i] ** 2 + s[j] ** 2)
        out.append(float(np.mean(gap > noise)))
    return float(np.mean(out))


def evaluate(spec: dict, raw: np.ndarray, idx) -> dict:
    tx = apply_transform(spec, raw)
    values, weights = exact_bootstrap(tx)
    return {
        "label": label(spec), **{k: v for k, v in spec.items() if k != "scale"},
        "H_regime": h_regime(tx.mean(axis=1)),
        "lcb95": weighted_quantile(values, weights, 0.025),
        "p_not_above_gate": float(weights[values <= GATE].sum()),
        "resolvable": resolvable(tx, idx), "n_bootstrap_atoms": int(values.size),
    }


def on_boundary(spec: dict) -> bool:
    if spec["kind"] == "power":
        return bool(abs(spec["gamma"] - POWER_GRID[0]) < 1e-9
                    or abs(spec["gamma"] - POWER_GRID[1]) < 1e-9)
    if spec["kind"] == "logistic":
        return bool(abs(spec["beta"] - 0.05) < 1e-9 or abs(spec["beta"] - 500.0) < 1e-9)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/monotone_transform_family_v3/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()

    grids, all_seeds, k_declared = {}, set(), None
    for grid in GRIDS:
        raw, contexts, seeds = load_per_seed(grid)
        all_seeds.update(seeds)
        sealed = json.loads((GRIDS[grid] / "result.json").read_text())["scalar_h_regime"]
        family = build_family(raw.mean(axis=1).ravel())
        k_declared = k_declared or len(family)
        idx = pair_index(raw.shape[2], raw.shape[0])
        print(f"\n  {grid}: {raw.shape[0]} ctx x {raw.shape[1]} semillas "
              f"x {raw.shape[2]:,} configuraciones · K={len(family)}", flush=True)

        rows = [evaluate(spec, raw, idx) for spec in family]
        for row, p in zip(rows, holm([r["p_not_above_gate"] for r in rows])):
            row["holm_adjusted_p"] = p
        ident = next(r for r in rows if r["label"] == "identity")
        floor = SIGNAL_FLOOR_FRACTION * ident["resolvable"]
        for row, spec in zip(rows, family):
            row["keeps_signal"] = bool(row["resolvable"] >= floor)
            row["on_grid_boundary"] = on_boundary(spec)
            row["qualifies"] = bool(row["lcb95"] >= GATE and row["holm_adjusted_p"] < 0.05
                                    and row["keeps_signal"])

        # Reintroduce the defect: the proxy has to be SEEN to fall on a step.
        steps = [r for r in rows if r["label"].startswith("step")]
        sharp = min(steps, key=lambda r: r["resolvable"])
        proxy_falls = sharp["resolvable"] < 0.5 * ident["resolvable"]

        planted, weight, planted_h = planted_surface(raw, PLANTED_H)
        pv, pw = exact_bootstrap(planted)
        planted_lcb = weighted_quantile(pv, pw, 0.025)

        qualifying = [r for r in rows if r["qualifies"]]
        stat_pass = [r for r in rows if r["lcb95"] >= GATE and r["holm_adjusted_p"] < 0.05]
        best = max(rows, key=lambda r: r["lcb95"])

        print(f"    identidad  H {ident['H_regime']:+.6f} (sellado {sealed:+.6f})  "
              f"resolubles {ident['resolvable']:.4f}  átomos {ident['n_bootstrap_atoms']}")
        print(f"    mejor LCB  H {best['H_regime']:+.6f}  LCB {best['lcb95']:+.6f}  "
              f"Holm {best['holm_adjusted_p']:.4f}  resolubles {best['resolvable']:.4f}  "
              f"{best['label']}{'  <-- EN EL BORDE' if best['on_grid_boundary'] else ''}")
        print(f"    proxy      identidad {ident['resolvable']:.4f} -> escalón "
              f"{sharp['resolvable']:.4f}  {'CAE' if proxy_falls else 'NO CAE'}")
        print(f"    potencia   plantada H {planted_h:.4f} -> LCB {planted_lcb:+.6f}  "
              f"{'SUFICIENTE' if planted_lcb >= GATE else 'INSUFICIENTE'}")
        print(f"    califican {len(qualifying)} de {len(rows)}   "
              f"(pasan LCB+Holm: {len(stat_pass)})")

        grids[grid] = {
            "contexts": contexts, "seeds": seeds, "n_configurations": int(raw.shape[2]),
            "k_family": len(family), "sealed_scalar_h_regime": float(sealed),
            "identity": ident, "best_by_lcb": best, "signal_floor": floor,
            "n_qualifying": len(qualifying), "qualifying": qualifying[:20],
            "n_passing_lcb_and_holm": len(stat_pass),
            "signal_proxy_falls_on_a_step": bool(proxy_falls), "sharpest_step": sharp,
            "power": {"target_H": PLANTED_H, "achieved_H": planted_h, "mix_weight": weight,
                      "lcb95": planted_lcb, "sufficient": bool(planted_lcb >= GATE)},
            "rows": rows,
        }

    ext = grids["wrap288_compat_extended_v1"]
    n_stat = sum(g["n_passing_lcb_and_holm"] for g in grids.values())
    n_qual = sum(g["n_qualifying"] for g in grids.values())
    proxy_ok = all(g["signal_proxy_falls_on_a_step"] for g in grids.values())

    if not ext["power"]["sufficient"]:
        verdict = "UNDERPOWERED_NO_VERDICT"
    elif not proxy_ok:
        verdict = ("SURVIVES_LCB_AND_MULTIPLICITY__SIGNAL_CRITERION_VOID" if n_stat
                   else "NO_MONOTONE_RESCALING_SURVIVES__SIGNAL_CRITERION_VOID")
    elif n_qual:
        verdict = "A_MONOTONE_RESCALING_SURVIVES_LCB_MULTIPLICITY_AND_SIGNAL"
    elif n_stat:
        verdict = "SURVIVES_LCB_AND_MULTIPLICITY_BUT_COSTS_SIGNAL"
    else:
        verdict = "NO_MONOTONE_RESCALING_SURVIVES"

    falsifiers = {
        "f1_identity_reproduces_the_sealed_scalar": {
            "passed": all(abs(g["identity"]["H_regime"] - g["sealed_scalar_h_regime"]) < 1e-9
                          for g in grids.values()),
            "evidence": {"why_it_can_fail": "the index is rebuilt from the cached aggregates; "
                                            "any drift makes this incomparable with the sealed "
                                            "numbers",
                         "deviations": {k: abs(g["identity"]["H_regime"]
                                               - g["sealed_scalar_h_regime"])
                                        for k, g in grids.items()}}},
        "f2_the_signal_proxy_can_actually_fall": {
            "passed": proxy_ok,
            "evidence": {"why_it_can_fail": "this is the falsifier that killed v2's proxy. "
                                            "Resolvable pairs must collapse under a step on BOTH "
                                            "grids; if this fails a third time the proxy route is "
                                            "abandoned rather than patched again",
                         "identity": {k: g["identity"]["resolvable"] for k, g in grids.items()},
                         "sharpest_step": {k: g["sharpest_step"]["resolvable"]
                                           for k, g in grids.items()}}},
        "f3_the_instrument_has_power": {
            "passed": bool(ext["power"]["sufficient"]),
            "evidence": {"why_it_can_fail": "three seeds give a ten-atom bootstrap; if a planted "
                                            "per-regime optimum at H = 0.10 cannot clear the gate, "
                                            "a null cannot be told apart from no power",
                         **ext["power"]}},
        "f4_multiplicity_over_the_declared_family": {
            "passed": all(g["k_family"] == k_declared for g in grids.values())
                      and k_declared == 631,
            "evidence": {"why_it_can_fail": "the preregistration fixes K at 631; a family that grew "
                                            "during the run would make every Holm p optimistic",
                         "k_declared": k_declared,
                         "k_per_grid": {k: g["k_family"] for k, g in grids.items()}}},
        "f5_the_base_grid_stays_at_zero": {
            "passed": grids["wrap288_v1"]["best_by_lcb"]["lcb95"] < 1e-9,
            "evidence": {"why_it_can_fail": "negative control: where one configuration is optimal "
                                            "in all six regimes no increasing f can move H, so a "
                                            "positive here would mean the instrument manufactures "
                                            "headroom",
                         "best_lcb": grids["wrap288_v1"]["best_by_lcb"]["lcb95"]}},
        "f6_the_optimum_is_interior": {
            "passed": not ext["best_by_lcb"]["on_grid_boundary"],
            "evidence": {"why_it_can_fail": "v2's optimum was power(gamma=10) on the edge of its "
                                            "own grid, so the family was truncated and the reported "
                                            "maximum was not one. The grid widened to 0.01-100; if "
                                            "the optimum is still on an edge it is still truncated",
                         "best": ext["best_by_lcb"]["label"],
                         "power_grid": list(POWER_GRID)}},
        "f7_no_fresh_seeds": custody_falsifier(sorted(all_seeds), replay_of=args.replay_of,
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
        "schema_version": "monotone_transform_family_v3",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION_NO_ADOPTION",
        "run_role": "CACHE_ANALYSIS", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": "docs/PREREGISTRO_TECHO_MONOTONO_V3_PARES_RESOLUBLES_2026-08-06.md",
        "predecessors": ["results/monotone_transform_ceiling/result.json",
                         "results/monotone_transform_family_v2/result_after_verdict_fix.json"],
        "gate": GATE, "signal_floor_fraction": SIGNAL_FLOOR_FRACTION, "k_family": k_declared,
        "signal_proxy": {
            "definition": "fraction of configuration pairs with |m_i - m_j| > sqrt(s_i^2 + s_j^2)",
            "pair_sample": PAIR_SAMPLE, "rng_seed": PAIR_RNG_SEED,
            "why_this_one": ("the two failed proxies could not fall: ordering is invariant under "
                             "any increasing map, and the v2 ratio collapses its own denominator "
                             "under saturation. Here a saturated pair gives 0 > 0, which is false")},
        "power_grid": list(POWER_GRID),
        "what_a_positive_would_still_not_mean": (
            "The configuration ordering is unchanged by construction, so headroom gained this way "
            "is a property of the metric's curvature -- an undeclared risk attitude -- not of the "
            "supply chain. Adoption needs declared mechanism and a virgin confirmation block."),
        "grids": grids, "falsifiers": falsifiers,
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(
        payload, args.output, contract=args.contract,
        reference=Path("results/monotone_transform_family_v2/result_after_verdict_fix.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
