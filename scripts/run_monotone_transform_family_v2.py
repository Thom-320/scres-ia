#!/usr/bin/env python3
"""Successor to the monotone ceiling: LCB, multiplicity, a signal proxy that can fall, and power.

WHAT THE PREDECESSOR GOT WRONG. `results/monotone_transform_ceiling/result.json` reported that a
logistic reparametrisation lifts H_regime from 0.0195 to 0.0742 with the configuration ordering
intact. Three defects: the 0.0742 was a point estimate; it was a maximum over ~2,500 transforms
with no multiplicity correction; and the "signal intact" claim rested on pairwise ordering, which
no strictly increasing map can disturb except through numerical ties -- a falsifier that could not
fail in the case it existed to judge.

WHAT CHANGES HERE.

* The family is ENUMERATED and closed at K = 621, so Holm has something well defined to correct.
* Every transform carries a bootstrap LCB over SEEDS, the replication unit. With three seeds the
  bootstrap is EXACT rather than sampled: 3^3 = 27 ordered resamples collapse to 10 distinct
  multisets, enumerated with their exact weights. The artifact records how few atoms that is,
  because the honest reading of a 10-atom distribution is that its 2.5th percentile is close to
  the minimum.
* The signal proxy is a signal-to-noise ratio -- between-configuration spread over between-seed
  spread -- and it is VALIDATED by reintroducing the defect: a sharp step must destroy it.
* A synthetic surface with a planted per-regime optimum at H = 0.10 goes through the same
  machinery first. If its LCB cannot clear the gate, the verdict is UNDERPOWERED_NO_VERDICT and
  nothing else in the run is reported as a negative.

ORDER OF OPERATIONS, CORRECTED. The predecessor applied f to the seed-averaged index. A new metric
is a new metric per episode, so f belongs INSIDE the seed average: mean_s f(R), not f(mean_s R).
Jensen makes those different for every non-affine f. The identity is unaffected, so the sealed
anchor still binds.

Preregistration: docs/PREREGISTRO_TECHO_MONOTONO_SUCESOR_LCB_2026-08-06.md
Development on burned tapes. Adjudicates nothing, adopts nothing, changes no primary endpoint.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations_with_replacement
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402

from supply_chain.cobb_douglas_resilience import (  # noqa: E402
    VARIABLES, derive_exponents, kappa_dot, resilience_index,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_monotone_transform_ceiling_v1 import GRIDS, h_regime  # noqa: E402


def load_per_seed(grid: str) -> tuple[np.ndarray, list[str], list[int]]:
    """The canonical index per (context, seed, configuration) -- NOT seed-averaged, because f has
    to be applied before the average. Same construction as the predecessor otherwise: his five
    variables, his rule on our maxima, kappa_dot within one (context, seed). f1 anchors the result
    against the sealed scalar, so any drift in this rebuild is caught rather than assumed away."""
    home = GRIDS[grid]
    sealed = json.loads((home / "result.json").read_text())
    raw = json.loads((home / "aggregates.json").read_text())
    # rsplit, not split: context names contain "|" themselves (R1r|esc).
    agg = {(k.rsplit("|", 1)[0], int(k.rsplit("|", 1)[1])): v for k, v in raw.items()}
    contexts, seeds = list(sealed["contexts"]), list(sealed["seeds"])

    kd = {key: kappa_dot({str(i): a["kappa"] for i, a in enumerate(cell)})
          for key, cell in agg.items()}
    maxima = {v: max(float(a[v]) for cell in agg.values() for a in cell)
              for v in VARIABLES if v != "kappa_dot"}
    maxima["kappa_dot"] = max(float(k) for row in kd.values() for k in row.values())
    exponents = derive_exponents(maxima)

    def surface(key) -> np.ndarray:
        row = kd[key]
        return np.array([
            resilience_index({vv: (row[str(i)] if vv == "kappa_dot" else float(a[vv]))
                              for vv in VARIABLES}, exponents)["R_cobb_douglas"]
            for i, a in enumerate(agg[key])])

    stacked = np.stack([np.stack([surface((c, s)) for s in seeds]) for c in contexts])
    return stacked, contexts, seeds

GATE = 0.05
SNR_FLOOR_FRACTION = 0.90
PLANTED_H = 0.10
MODULES = ("supply_chain/cobb_douglas_resilience.py", "supply_chain/arm_runner.py",
           "supply_chain/seed_custody.py")


# --------------------------------------------------------------------------------------------
# The family, enumerated. A family that cannot be enumerated cannot be corrected.
# --------------------------------------------------------------------------------------------
def build_family(pooled: np.ndarray) -> list[dict]:
    lo, hi = float(pooled.min()), float(pooled.max())
    scale = float(pooled.std()) or 1.0
    fam = [{"kind": "identity"}]
    for t in np.quantile(pooled, np.linspace(0.02, 0.98, 25)):
        for beta in np.geomspace(0.05, 500.0, 20):
            fam.append({"kind": "logistic", "t": float(t), "beta": float(beta), "scale": scale})
    for gamma in np.geomspace(0.1, 10.0, 21):
        fam.append({"kind": "power", "gamma": float(gamma), "lo": lo, "hi": hi})
    for t in np.quantile(pooled, np.linspace(0.01, 0.99, 99)):
        fam.append({"kind": "step", "t": float(t)})
    return fam


def apply_transform(spec: dict, x: np.ndarray) -> np.ndarray:
    kind = spec["kind"]
    if kind == "identity":
        return x
    if kind == "logistic":
        z = np.clip(spec["beta"] * (x - spec["t"]) / spec["scale"], -700.0, 700.0)
        return 1.0 / (1.0 + np.exp(-z))
    if kind == "power":
        span = spec["hi"] - spec["lo"]
        u = np.clip((x - spec["lo"]) / span, 0.0, 1.0) if span else np.zeros_like(x)
        return u ** spec["gamma"]
    if kind == "step":
        return (x >= spec["t"]).astype(float)
    raise ValueError(kind)


def label(spec: dict) -> str:
    if spec["kind"] == "identity":
        return "identity"
    if spec["kind"] == "logistic":
        return f"logistic(t={spec['t']:.4f}, beta={spec['beta']:.4g})"
    if spec["kind"] == "power":
        return f"power(gamma={spec['gamma']:.4g})"
    return f"step(t={spec['t']:.4f})"


# --------------------------------------------------------------------------------------------
# Estimator, signal proxy, exact bootstrap
# --------------------------------------------------------------------------------------------
def surface_from(per_seed: np.ndarray, picks) -> np.ndarray:
    """Seed-average of the ALREADY transformed per-seed surfaces. f lives inside the average."""
    return per_seed[:, list(picks), :].mean(axis=1)


def snr(per_seed: np.ndarray) -> float:
    """Between-configuration spread over between-seed spread, median across contexts.

    This is the quantity a learner actually needs: how much of the variation between
    configurations survives above the noise between replications. A sharp logistic saturates the
    tails, the between-configuration spread collapses and the seed noise does not, so this falls.
    Pairwise ordering -- the proxy this replaces -- could not."""
    out = []
    for r in range(per_seed.shape[0]):
        block = per_seed[r]                                   # (seeds, configs)
        signal = float(block.mean(axis=0).std())
        noise = float(np.mean(block.std(axis=0, ddof=1))) if block.shape[0] > 1 else 0.0
        out.append(signal / noise if noise > 0 else math.inf)
    finite = [v for v in out if math.isfinite(v)]
    return float(np.median(finite)) if finite else math.inf


def exact_bootstrap(per_seed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every distinct resample-with-replacement of the seeds, with its exact multinomial weight.

    With three seeds there are 27 ordered resamples and only 10 distinct multisets. Enumerating
    them is both cheaper and more accurate than sampling -- and it makes the weakness impossible
    to hide: a distribution with ten atoms has a 2.5th percentile that is essentially its minimum.
    """
    n = per_seed.shape[1]
    values, weights = [], []
    for picks in combinations_with_replacement(range(n), n):
        counts = np.bincount(picks, minlength=n)
        w = math.factorial(n)
        for c in counts:
            w //= math.factorial(int(c))
        values.append(h_regime(surface_from(per_seed, picks)))
        weights.append(w)
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(v)
    return v[order], w[order] / w.sum()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    return float(values[np.searchsorted(np.cumsum(weights), q, side="left").clip(
        0, len(values) - 1)])


def evaluate(spec: dict, raw_per_seed: np.ndarray) -> dict:
    tx = apply_transform(spec, raw_per_seed)
    values, weights = exact_bootstrap(tx)
    return {
        "label": label(spec), **{k: v for k, v in spec.items() if k != "scale"},
        "H_regime": h_regime(tx.mean(axis=1)),
        "lcb95": weighted_quantile(values, weights, 0.025),
        "p_not_above_gate": float(weights[values <= GATE].sum()),
        "snr": snr(tx), "n_bootstrap_atoms": int(values.size),
    }


def holm(pvals: list[float]) -> list[float]:
    k = len(pvals)
    order = sorted(range(k), key=lambda i: pvals[i])
    adj, running = [0.0] * k, 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (k - rank) * pvals[idx]))
        adj[idx] = running
    return adj


# --------------------------------------------------------------------------------------------
def planted_surface(raw: np.ndarray, target: float) -> tuple[np.ndarray, float, float]:
    """A synthetic surface with a DISTINCT optimum per regime, calibrated to a known H, carrying
    the real per-seed residuals as noise. This is the power test: if the instrument cannot see a
    planted effect of this size, a null on the real surface means nothing."""
    n_ctx, _, n_cfg = raw.shape
    norm = np.empty_like(raw)
    for r in range(n_ctx):
        m = raw[r].mean(axis=0)
        lo, hi = m.min(), m.max()
        norm[r] = (raw[r] - lo) / (hi - lo) if hi > lo else 0.0
    resid = norm - norm.mean(axis=1, keepdims=True)
    signal = norm.mean(axis=1)                                  # (contexts, configs)
    picks = np.linspace(0, n_cfg - 1, n_ctx + 1)[:-1].astype(int)
    onehot = np.zeros_like(signal)
    for r in range(n_ctx):
        onehot[r, picks[r]] = 1.0

    def build(w: float) -> np.ndarray:
        return ((1.0 - w) * signal + w * onehot)[:, None, :] + resid

    lo_w, hi_w = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo_w + hi_w)
        if h_regime(build(mid).mean(axis=1)) < target:
            lo_w = mid
        else:
            hi_w = mid
    w = 0.5 * (lo_w + hi_w)
    surf = build(w)
    return surf, w, h_regime(surf.mean(axis=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/monotone_transform_family_v2/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()

    grids, all_seeds, k_declared = {}, set(), None
    for grid in GRIDS:
        raw, contexts, seeds = load_per_seed(grid)
        all_seeds.update(seeds)
        sealed = json.loads((GRIDS[grid] / "result.json").read_text())["scalar_h_regime"]
        family = build_family(raw.mean(axis=1).ravel())
        if k_declared is None:
            k_declared = len(family)
        print(f"\n  {grid}: {raw.shape[0]} contextos x {raw.shape[1]} semillas "
              f"x {raw.shape[2]:,} configuraciones · familia K={len(family)}")

        rows = [evaluate(spec, raw) for spec in family]
        for row, p in zip(rows, holm([r["p_not_above_gate"] for r in rows])):
            row["holm_adjusted_p"] = p
        ident = next(r for r in rows if r["label"] == "identity")
        snr_floor = SNR_FLOOR_FRACTION * ident["snr"]
        for row in rows:
            row["keeps_signal"] = bool(row["snr"] >= snr_floor)
            row["qualifies"] = bool(row["lcb95"] >= GATE and row["holm_adjusted_p"] < 0.05
                                    and row["keeps_signal"])

        # The defect, reintroduced. If a sharp step does NOT destroy the proxy, the proxy is as
        # inert as the pairwise ordering it replaces and this whole run is void.
        sharp = min(rows, key=lambda r: r["snr"])
        proxy_falls = sharp["snr"] < 0.5 * ident["snr"]

        planted, weight, planted_h = planted_surface(raw, PLANTED_H)
        pv, pw = exact_bootstrap(planted)
        planted_lcb = weighted_quantile(pv, pw, 0.025)
        has_power = planted_lcb >= GATE

        qualifying = [r for r in rows if r["qualifies"]]
        pre_holm = [r for r in rows if r["lcb95"] >= GATE]
        best = max(rows, key=lambda r: r["lcb95"])

        print(f"    identidad     H {ident['H_regime']:+.6f}  (sellado {sealed:+.6f})  "
              f"SNR {ident['snr']:.3f}  átomos bootstrap {ident['n_bootstrap_atoms']}")
        print(f"    mejor por LCB {best['H_regime']:+.6f}  LCB {best['lcb95']:+.6f}  "
              f"Holm {best['holm_adjusted_p']:.4f}  SNR {best['snr']:.3f}  {best['label']}")
        print(f"    potencia      superficie plantada H {planted_h:.4f} (w={weight:.4f}) -> "
              f"LCB {planted_lcb:+.6f}  {'SUFICIENTE' if has_power else 'INSUFICIENTE'}")
        print(f"    proxy         SNR identidad {ident['snr']:.3f} -> escalón "
              f"{sharp['snr']:.3f}  {'cae' if proxy_falls else 'NO CAE'}")
        print(f"    califican {len(qualifying)} de {len(rows)}   "
              f"(con LCB>={GATE} antes de Holm: {len(pre_holm)})")

        grids[grid] = {
            "contexts": contexts, "seeds": seeds, "n_configurations": int(raw.shape[2]),
            "k_family": len(family), "sealed_scalar_h_regime": float(sealed),
            "identity": ident, "best_by_lcb": best,
            "n_qualifying": len(qualifying), "qualifying": qualifying[:20],
            "n_lcb_above_gate_before_holm": len(pre_holm),
            "snr_floor": snr_floor, "signal_proxy_falls_on_a_step": bool(proxy_falls),
            "sharpest": sharp,
            "power": {"target_H": PLANTED_H, "achieved_H": planted_h, "mix_weight": weight,
                      "lcb95": planted_lcb, "sufficient": bool(has_power),
                      "n_bootstrap_atoms": int(pv.size)},
            "rows": rows,
        }

    ext = grids["wrap288_compat_extended_v1"]
    # A transform that clears BOTH statistical hurdles, independently of the signal criterion.
    # Keeping this separate matters: the first version of this branch read `n_qualifying == 0` as
    # "the ceiling did not survive multiplicity", when in fact every one of the nine survivors
    # cleared LCB and Holm and was dropped by the signal criterion alone. That mislabels a
    # positive as a negative whenever the signal proxy is the thing that failed.
    n_stat = sum(1 for g in grids.values() for r in g["rows"]
                 if r["lcb95"] >= GATE and r["holm_adjusted_p"] < 0.05)
    signal_criterion_valid = all(g["signal_proxy_falls_on_a_step"] for g in grids.values())

    if not ext["power"]["sufficient"]:
        verdict = "UNDERPOWERED_NO_VERDICT"
    elif not signal_criterion_valid:
        # Preregistration section 3: if the proxy does not fall on a step it is declared failed,
        # so no claim may rest on it -- in either direction.
        verdict = ("SURVIVES_LCB_AND_MULTIPLICITY__SIGNAL_CRITERION_VOID" if n_stat
                   else "NO_MONOTONE_RESCALING_SURVIVES_LCB__SIGNAL_CRITERION_VOID")
    elif any(g["n_qualifying"] > 0 for g in grids.values()):
        verdict = "A_MONOTONE_RESCALING_SURVIVES_LCB_AND_MULTIPLICITY"
    elif n_stat:
        verdict = "SURVIVES_LCB_AND_MULTIPLICITY_BUT_COSTS_SIGNAL"
    elif any(g["n_lcb_above_gate_before_holm"] > 0 for g in grids.values()):
        verdict = "THE_CEILING_DOES_NOT_SURVIVE_MULTIPLICITY"
    else:
        verdict = "NO_MONOTONE_RESCALING_SURVIVES_LCB"

    falsifiers = {
        "f1_identity_reproduces_the_sealed_scalar": {
            "passed": all(abs(g["identity"]["H_regime"] - g["sealed_scalar_h_regime"]) < 1e-9
                          for g in grids.values()),
            "evidence": {"why_it_can_fail": "the index is rebuilt from the cached aggregates and f "
                                            "now lives inside the seed average; if the identity no "
                                            "longer matches the sealed scalar, nothing downstream "
                                            "is comparable with what is already published",
                         "deviations": {k: abs(g["identity"]["H_regime"]
                                               - g["sealed_scalar_h_regime"])
                                        for k, g in grids.items()}}},
        "f2_the_signal_proxy_can_actually_fall": {
            "passed": all(g["signal_proxy_falls_on_a_step"] for g in grids.values()),
            "evidence": {"why_it_can_fail": "the proxy this replaces -- pairwise ordering -- could "
                                            "not fall under any strictly increasing map, so it "
                                            "passed without being able to fail. Reintroducing the "
                                            "defect is the only way to know this one is different",
                         "identity_snr": {k: g["identity"]["snr"] for k, g in grids.items()},
                         "sharpest_snr": {k: g["sharpest"]["snr"] for k, g in grids.items()}}},
        "f3_the_instrument_has_power": {
            "passed": bool(ext["power"]["sufficient"]),
            "evidence": {"why_it_can_fail": "three seeds give a bootstrap with ten atoms. If a "
                                            "planted per-regime optimum at H = 0.10 cannot clear "
                                            "the gate, a null on the real surface cannot be "
                                            "distinguished from no power and there is no verdict",
                         **ext["power"]}},
        "f4_multiplicity_is_applied_over_the_declared_family": {
            "passed": all(g["k_family"] == k_declared for g in grids.values()),
            "evidence": {"why_it_can_fail": "the preregistration fixes K at 621. A family that "
                                            "grew during the run would make every Holm p optimistic",
                         "k_declared": k_declared,
                         "k_per_grid": {k: g["k_family"] for k, g in grids.items()}}},
        "f5_the_base_grid_stays_at_zero": {
            "passed": grids["wrap288_v1"]["best_by_lcb"]["lcb95"] < 1e-9,
            "evidence": {"why_it_can_fail": "negative control: on the 288 grid one configuration is "
                                            "optimal in all six regimes, so no increasing f can "
                                            "move H off zero. If this instrument finds headroom "
                                            "there, the instrument is manufacturing it",
                         "best_lcb": grids["wrap288_v1"]["best_by_lcb"]["lcb95"]}},
        "f6_no_fresh_seeds": custody_falsifier(sorted(all_seeds), replay_of=args.replay_of,
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
        "schema_version": "monotone_transform_family_v2",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION_NO_ADOPTION",
        "run_role": "CACHE_ANALYSIS", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": "docs/PREREGISTRO_TECHO_MONOTONO_SUCESOR_LCB_2026-08-06.md",
        "predecessor": "results/monotone_transform_ceiling/result.json",
        "gate": GATE, "snr_floor_fraction": SNR_FLOOR_FRACTION, "k_family": k_declared,
        "order_of_operations": ("f is applied per (context, seed) and the seed average is taken "
                                "AFTER, because a new metric is a new metric per episode. The "
                                "predecessor transformed the seed average instead; Jensen makes "
                                "those differ for every non-affine f."),
        "bootstrap": ("exact enumeration of all distinct resamples-with-replacement of the seeds, "
                      "with exact multinomial weights"),
        "what_a_positive_would_still_not_mean": (
            "Surviving LCB and multiplicity would still not authorise adopting the transform. The "
            "configuration ordering is unchanged by construction, so any headroom gained is a "
            "property of the metric's curvature -- an undeclared risk attitude -- and not of the "
            "supply chain. Adoption requires declared mechanism and a virgin confirmation block."),
        "grids": grids, "falsifiers": falsifiers,
        "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/monotone_transform_ceiling/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
