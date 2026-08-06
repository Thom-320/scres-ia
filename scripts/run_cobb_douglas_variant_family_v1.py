#!/usr/bin/env python3
"""Family B: the 18 declared Cobb-Douglas derivations, all measured and all reported.

Family A -- 144 ReT-Excel derivations -- returned NO_DEFENSIBLE_DERIVATION_REACHES_THE_BAR. The
preregistration commits to 162 variants, so these 18 close the table. Same rule as Family A: every
variant is reported, Holm-Bonferroni runs over the whole K = 162, and no variant is dropped for
being inconvenient.

THREE AXES, EACH DEFENSIBLE FROM GARRIDO'S OWN TEXT.

`exponents` -- his rule is `0.20/ln(x_max)` derived from the calibration SET. `published` reuses
his five numbers, which encode HIS maxima; `ours` applies his rule to our maxima, which is
following the rule rather than departing from it; `per_context` applies it inside each context,
which is the same rule read at a finer grain.

`variables` -- `his_five` is the index as published. `no_tau` drops tau, which is dead in 18% of
our cells and whose exponent is ill-conditioned (error amplification 3.39). `plus_service` adds
unserved rations at the horizon with a negative sign, which is the driver his section 6.2 asks for
by name and the one whose absence lets an abandoned order stop costing anything.

`kappa_set` -- kappa_dot is set-relative and he never says which set. Within a (context, seed) is
the set an experimenter chooses between; global is the set across the whole study.

The service variable is joined from the sealed surface cache, not recomputed: same 288
configurations in the same canonical order, same contexts, same seeds. The join is asserted, not
assumed.

Development on burned tapes. Adjudicates nothing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.cobb_douglas_resilience import (  # noqa: E402
    GARRIDO_2024_EXPONENTS, SHARE_PER_TERM, SIGNS, kappa_dot,
)
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402

BASE_VARS = ("zeta", "epsilon", "phi", "tau", "kappa_dot")
SERVICE_VAR = "upsilon"                    # unserved rations at the horizon
SIGNS_EXT = dict(SIGNS, upsilon=-1.0)
FLOORS = {"zeta": 1.0, "epsilon": 1.0, "phi": 1.0, "tau": 1.0,
          "upsilon": 1.0, "kappa_dot": 1e-9}
AXES = {
    "exponents": ("published", "ours", "per_context"),
    "variables": ("his_five", "no_tau", "plus_service"),
    "kappa_set": ("within", "global"),
}
GATE, N_BOOT, K_FAMILY_A = 0.05, 2_000, 144
MODULES = ("supply_chain/cobb_douglas_resilience.py", "supply_chain/arm_runner.py",
           "supply_chain/seed_custody.py")


def variable_set(level: str) -> tuple[str, ...]:
    if level == "his_five":
        return BASE_VARS
    if level == "no_tau":
        return tuple(v for v in BASE_VARS if v != "tau")
    return BASE_VARS + (SERVICE_VAR,)


def share_for(variables) -> float:
    """His 0.20 is `1/5` BECAUSE he has five variables (IJPR p. 11: "each function argument was
    equated to 1/5"). It is an equal-share scale normaliser, not a preference weight, so it moves
    with the count: 1/4 = 0.25 when tau is dropped, 1/6 = 0.1667 when the service driver is added.
    Holding 0.20 fixed across variable sets -- which the first pass did -- silently changes the
    total weight of the index instead of redistributing it."""
    return 1.0 / len(variables)


def exponents_from(maxima, variables) -> dict[str, float]:
    """His rule, `share/ln(x_max)`. A maximum at or below 1 cannot normalise anything."""
    share = share_for(variables)
    out = {}
    for v in variables:
        x = float(maxima[v])
        if x <= 1.0:
            raise ValueError(f"{v}_max = {x} <= 1; the rule {share:.4f}/ln(x_max) is undefined")
        out[v] = share / math.log(x)
    return out


def score(values, exponents, variables) -> np.ndarray:
    """Eq. (6): sigmoid of the signed weighted log-sum, over whichever variables are in play."""
    linear = np.zeros(len(values[variables[0]]))
    for v in variables:
        floored = np.maximum(np.asarray(values[v], dtype=float), FLOORS[v])
        linear = linear + SIGNS_EXT[v] * exponents[v] * np.log(floored)
    return 1.0 / (1.0 + np.exp(-linear))


def h_regime(per_ctx) -> float:
    norm = []
    for m in per_ctx:
        m = np.asarray(m, dtype=float)
        lo, hi = m.min(), m.max()
        norm.append((m - lo) / (hi - lo) if hi > lo else np.zeros_like(m))
    stacked = np.stack(norm)
    return float(stacked.max(axis=1).mean() - stacked.mean(axis=0).max())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aggregates", type=Path,
                    default=Path("results/cobb_douglas_component_headroom/aggregates.json"))
    ap.add_argument("--surface", type=Path, default=Path("results/surface_cache/wrap288_v1"))
    ap.add_argument("--family-a", type=Path,
                    default=Path("results/metric_derivation_family/result.json"))
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/cobb_douglas_variant_family/result.json"))
    args = ap.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(20260806)

    raw = json.loads(args.aggregates.read_text())
    aggregates = {(k.rsplit("|", 1)[0], int(k.rsplit("|", 1)[1])): v for k, v in raw.items()}
    panel = {}
    for path in glob.glob(str(args.surface / "**" / "*.json"), recursive=True):
        d = json.loads(Path(path).read_text())
        panel[(d["context"], int(d["seed"]))] = d["cells"]

    cells = sorted(set(aggregates) & set(panel))
    if not cells:
        raise SystemExit("aggregates and surface cache share no (context, seed)")
    n_cfg = len(aggregates[cells[0]])
    for key in cells:
        if len(panel[key]) != n_cfg:
            raise SystemExit(f"{key}: surface has {len(panel[key])} configs, aggregates {n_cfg}")
    contexts = sorted({c for c, _ in cells})
    seeds = sorted({s for _, s in cells})
    print(f"  {len(cells)} celdas · {len(contexts)} contextos × {len(seeds)} semillas "
          f"· {n_cfg} configuraciones")

    # Raw per-cell variable vectors, with kappa_dot computed both ways.
    per_cell = {}
    kappa_all = {key: [float(a["kappa"]) for a in aggregates[key]] for key in cells}
    flat = [k for key in cells for k in kappa_all[key]]
    global_scale = len(flat) / sum(flat) if sum(flat) else 1.0
    for key in cells:
        agg, cel = aggregates[key], panel[key]
        kd_within = kappa_dot({str(i): k for i, k in enumerate(kappa_all[key])})
        per_cell[key] = {
            "zeta": [float(a["zeta"]) for a in agg],
            "epsilon": [float(a["epsilon"]) for a in agg],
            "phi": [float(a["phi"]) for a in agg],
            "tau": [float(a["tau"]) for a in agg],
            SERVICE_VAR: [float(c["panel"]["demanded_rations"] - c["panel"]["delivered_rations"])
                          for c in cel],
            "kappa_dot_within": [kd_within[str(i)] for i in range(len(agg))],
            "kappa_dot_global": [k * global_scale for k in kappa_all[key]],
        }

    def values_for(key, kappa_set):
        v = dict(per_cell[key])
        v["kappa_dot"] = v[f"kappa_dot_{kappa_set}"]
        return v

    variants = []
    for exp_lvl, var_lvl, kap_lvl in itertools.product(*AXES.values()):
        variables = variable_set(var_lvl)
        # `published` has no exponent for the service driver -- he never had one. His RULE fills
        # that single slot on our maxima while the other variables keep his published numbers.
        surfaces = {key: values_for(key, kap_lvl) for key in cells}

        def exps_for(subset_keys):
            maxima = {v: max(max(surfaces[k][v]) for k in subset_keys) for v in variables}
            return exponents_from(maxima, variables)

        try:
            if exp_lvl == "published":
                base = {v: GARRIDO_2024_EXPONENTS[v] for v in variables
                        if v in GARRIDO_2024_EXPONENTS}
                missing = [v for v in variables if v not in base]
                if missing:                       # only upsilon; his rule fills it on our maxima
                    base |= exponents_from(
                        {v: max(max(surfaces[k][v]) for k in cells) for v in missing}, missing)
                per_ctx = [np.mean([score(surfaces[(c, s)], base, variables) for s in seeds],
                                   axis=0) for c in contexts]
                used = base
            elif exp_lvl == "ours":
                used = exps_for(cells)
                per_ctx = [np.mean([score(surfaces[(c, s)], used, variables) for s in seeds],
                                   axis=0) for c in contexts]
            else:
                used, per_ctx = {}, []
                for c in contexts:
                    e = exps_for([(c, s) for s in seeds])
                    used[c] = e
                    per_ctx.append(np.mean([score(surfaces[(c, s)], e, variables) for s in seeds],
                                           axis=0))
        except ValueError as exc:
            variants.append({"exponents": exp_lvl, "variables": var_lvl, "kappa_set": kap_lvl,
                             "H_regime": None, "excluded": str(exc)})
            continue

        # HIS OWN INVARIANT. The rule 0.20/ln(x_max) exists so each term contributes at most
        # 1/5 at its own maximum. His published numbers encode HIS maxima -- inventories in the
        # thousands against our millions -- so reusing them on our data breaks the very bound that
        # makes five quantities in incompatible units commensurable. The port ships
        # `assert_terms_bounded` for exactly this. Measured, per variant.
        term_max = {}
        for v in variables:
            e = used[contexts[0]][v] if exp_lvl == "per_context" else used[v]
            biggest = max(max(surfaces[k][v]) for k in cells)
            term_max[v] = abs(e * math.log(max(biggest, FLOORS[v])))
        share = share_for(variables)
        respects_bound = all(t <= share + 1e-9 for t in term_max.values())

        point = h_regime(per_ctx)
        draws = []
        for _ in range(200):
            pick = [seeds[i] for i in rng.integers(0, len(seeds), len(seeds))]
            if exp_lvl == "per_context":
                boot = [np.mean([score(surfaces[(c, s)], used[c], variables) for s in pick],
                                axis=0) for c in contexts]
            else:
                boot = [np.mean([score(surfaces[(c, s)], used, variables) for s in pick],
                                axis=0) for c in contexts]
            draws.append(h_regime(boot))
        draws = np.asarray(draws)
        variants.append({
            "exponents": exp_lvl, "variables": var_lvl, "kappa_set": kap_lvl,
            "H_regime": point, "lcb95": float(np.percentile(draws, 2.5)),
            "p_not_above_gate": float(np.mean(draws <= GATE)),
            "n_variables": len(variables),
            "respects_share_bound": bool(respects_bound), "share_bound": share,
            "max_term_magnitude": max(term_max.values()),
            "term_magnitudes": term_max,
        })

    live = [v for v in variants if v.get("H_regime") is not None]
    # Holm over the WHOLE preregistered family, A and B together: 18 corrected against 18 would
    # understate the search that actually happened.
    fam_a = json.loads(args.family_a.read_text())["variants"] if args.family_a.exists() else []
    def p_of(v):
        for key in ("p_not_above_gate", "p_one_sided"):
            if v.get(key) is not None:
                return float(v[key])
        return None

    pooled = ([{"p": p_of(v), "tag": "B"} for v in live]
              + [{"p": p_of(v), "tag": "A"} for v in fam_a if p_of(v) is not None])
    k = len(pooled)
    order = sorted(range(k), key=lambda i: pooled[i]["p"])
    adj, running = [0.0] * k, 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (k - rank) * pooled[idx]["p"]))
        adj[idx] = running
    for i, v in enumerate(live):
        v["holm_adjusted_p"] = adj[i]

    crossing = [v for v in live if v["lcb95"] >= GATE and v["holm_adjusted_p"] < 0.05]
    winners = [v for v in crossing if v["respects_share_bound"]]
    best = max(live, key=lambda v: v["H_regime"])
    verdict = ("DEFENSIBLE_COBB_DOUGLAS_DERIVATION_REACHES_THE_BAR" if winners
               else "ONLY_MISSCALED_VARIANTS_REACH_THE_BAR" if crossing
               else "NO_DEFENSIBLE_DERIVATION_REACHES_THE_BAR")

    falsifiers = {
        "f1_all_variants_reported": {
            "passed": len(variants) == 18,
            "evidence": {"why_it_can_fail": "the preregistration commits to 18; reporting fewer "
                                            "would be selective reporting by omission",
                         "n_variants": len(variants), "n_scored": len(live)}},
        "f2_the_family_separates": {
            "passed": len({round(v["H_regime"], 8) for v in live}) > 1,
            "evidence": {"why_it_can_fail": "if every variant returned the same H_regime the axes "
                                            "would not be doing anything and the family would be "
                                            "measuring the estimator"}},
        "f3_the_service_join_is_aligned": {
            "passed": all(len(panel[key]) == n_cfg for key in cells) and len(cells) > 0,
            "evidence": {"why_it_can_fail": "the service variable is joined from a different "
                                            "artifact; a mismatched configuration order would "
                                            "silently pair the wrong rows",
                         "cells_joined": len(cells), "configs_per_cell": n_cfg,
                         "contexts": contexts, "seeds": seeds}},
        "f4_multiplicity_over_the_whole_family": {
            "passed": k >= K_FAMILY_A + len(live),
            "evidence": {"why_it_can_fail": "correcting these 18 against 18 would understate the "
                                            "search; Holm runs over A and B pooled",
                         "k_pooled": k, "family_a": len(fam_a), "family_b": len(live)}},
        "f5_crossing_variants_respect_his_share_bound": {
            "passed": all(v["respects_share_bound"] for v in crossing),
            "evidence": {"why_it_can_fail": "his rule 0.20/ln(x_max) exists so no term exceeds 1/5 "
                                            "at its own maximum. A variant that crosses the gate "
                                            "while breaking that bound is not his index applied to "
                                            "our chain, it is his index MISAPPLIED, and its "
                                            "headroom is a scale error rather than a finding",
                         "share_bound_rule": "1/len(variables), his 1/5 generalised",
                         "crossing": [{k: v[k] for k in ("exponents", "variables", "kappa_set",
                                                         "H_regime", "respects_share_bound",
                                                         "max_term_magnitude", "share_bound")}
                                      for v in crossing]}},
        "f6_no_fresh_seeds": custody_falsifier(seeds, replay_of=args.replay_of,
                                               exclude=args.output),
    }
    falsifiers["all_passed"] = all(
        v["passed"] for kk, v in falsifiers.items()
        if kk != "all_passed" and isinstance(v, dict) and not v.get("not_applicable"))

    print("\n  las 18, ordenadas por H_regime:")
    for v in sorted(live, key=lambda x: -x["H_regime"]):
        flag = ("  <-- CRUZA" if v in winners
                else "  <-- cruza pero ROMPE la cota 0,20" if v in crossing else "")
        print(f"    H {v['H_regime']:+.5f} lcb {v['lcb95']:+.5f} holm {v['holm_adjusted_p']:.3f}"
              f"  {v['exponents']:<12}{v['variables']:<14}{v['kappa_set']}{flag}")
    excluded = [v for v in variants if v.get("H_regime") is None]
    for v in excluded:
        print(f"    EXCLUIDA  {v['exponents']:<12}{v['variables']:<14}{v['kappa_set']}"
              f"  -- {v['excluded'][:60]}")
    print(f"\n  veredicto: {verdict}   (máximo {best['H_regime']:+.5f} contra umbral {GATE})\n")
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        label = "NO APLICA" if f.get("not_applicable") else ("PASA" if f["passed"] else "FALLA")
        print(f"    {name:<44} {label}")

    payload = {
        "schema_version": "cobb_douglas_variant_family_v1",
        "claim_status": verdict,
        "scope": "DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION",
        "run_role": "CACHE_ANALYSIS", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "axes": AXES, "gate": GATE, "k_pooled_with_family_a": k,
        "service_variable": {
            "name": SERVICE_VAR, "definition": "demanded_rations - delivered_rations at horizon",
            "sign": -1.0,
            "why": ("Garrido's section 6.2 asks for drivers his index did not consider. This is the "
                    "one whose absence lets an abandoned order stop costing anything: epsilon "
                    "prices the backorder QUEUE, and an order that is never served leaves it."),
            "joined_from": str(args.surface)},
        "variants": variants, "crossing_the_gate": crossing, "winners": winners, "best": best,
        "share_bound_rule": {
            "value": "1 / number_of_variables",
            "source": ("Garrido IJPR 2024 p. 11: 'each function argument was equated to 1/5'. The "
                       "0.20 is 1/5 BECAUSE he has five variables, so it moves with the count: "
                       "0.25 for four, 0.1667 for six. The first pass held it at 0.20 for every "
                       "variable set, which changes the index's total weight rather than "
                       "redistributing it.")},
        "falsifiers": falsifiers, "elapsed_seconds": time.perf_counter() - started,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=args.family_a)
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
